"""
Split conformal / CQR-style offsets for quantile prediction intervals (scaled label space).

Offsets are computed on a calibration set and applied as:
  q_lo' = q_lo - offsets_lower
  q_hi' = q_hi + offsets_upper
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def conformity_scores_interval(
    y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per (sample, label) one-sided gaps outside [q_lo, q_hi]."""
    s_lo = np.maximum(0.0, q_lo - y)
    s_hi = np.maximum(0.0, y - q_hi)
    return s_lo, s_hi


def calibrate_cqr_offsets(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    alpha: float = 0.1,
) -> Dict[str, Any]:
    """
    Asymmetric conformal offsets per label from val residuals.

    Parameters
    ----------
    y_val : (N, L) calibration labels (scaled space, same as preds).
    pred_val : (N, L, 3) with [:, :, 0]=lower, [:, :, 1]=median, [:, :, 2]=upper.
    alpha : miscoverage rate (target coverage 1 - alpha on calibration if exchangeable).

    Returns
    -------
    dict with keys offsets_lower, offsets_upper (length L), alpha, method.
    """
    if pred_val.ndim != 3 or pred_val.shape[2] != 3:
        raise ValueError("pred_val must have shape (N, L, 3)")
    y_val = np.asarray(y_val, dtype=np.float64)
    pred_val = np.asarray(pred_val, dtype=np.float64)
    n, ell = y_val.shape
    if pred_val.shape[:2] != (n, ell):
        raise ValueError("y_val and pred_val batch/label dims must match")

    q_lo = pred_val[:, :, 0]
    q_hi = pred_val[:, :, 2]
    s_lo, s_hi = conformity_scores_interval(y_val, q_lo, q_hi)

    m = np.isfinite(y_val) & np.isfinite(q_lo) & np.isfinite(q_hi)
    q_level = min(1.0, max(0.0, 1.0 - float(alpha)))

    offsets_lower: List[float] = []
    offsets_upper: List[float] = []
    for j in range(ell):
        ml = m[:, j]
        if int(np.count_nonzero(ml)) < 5:
            offsets_lower.append(0.0)
            offsets_upper.append(0.0)
            continue
        offsets_lower.append(float(np.quantile(s_lo[ml, j], q_level)))
        offsets_upper.append(float(np.quantile(s_hi[ml, j], q_level)))

    return {
        "version": 1,
        "method": "cqr_asymmetric_quantile_offsets",
        "alpha": float(alpha),
        "offsets_lower": offsets_lower,
        "offsets_upper": offsets_upper,
    }


def apply_cqr_offsets_inplace(pred: np.ndarray, calib: Dict[str, Any]) -> np.ndarray:
    """Apply offsets to (N, L, 3) predictions in place; returns pred."""
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError("pred must have shape (N, L, 3)")
    o_lo = np.asarray(calib["offsets_lower"], dtype=np.float64).reshape(-1)
    o_hi = np.asarray(calib["offsets_upper"], dtype=np.float64).reshape(-1)
    ell = pred.shape[1]
    if o_lo.size != ell or o_hi.size != ell:
        raise ValueError(
            f"Calibration offsets length ({o_lo.size}, {o_hi.size}) != pred labels ({ell})"
        )
    pred[:, :, 0] -= o_lo.reshape(1, -1)
    pred[:, :, 2] += o_hi.reshape(1, -1)
    return pred


def interval_coverage(
    y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
) -> np.ndarray:
    """Per-label coverage rate in (0,1), ignoring non-finite y or bounds."""
    y_true = np.asarray(y_true, dtype=np.float64)
    q_lo = np.asarray(q_lo, dtype=np.float64)
    q_hi = np.asarray(q_hi, dtype=np.float64)

    m = np.isfinite(y_true) & np.isfinite(q_lo) & np.isfinite(q_hi) & (q_lo <= q_hi)

    # Suppress RuntimeWarnings for invalid comparisons with NaNs, which we explicitly handle
    with np.errstate(invalid="ignore"):
        inside = (y_true >= q_lo) & (y_true <= q_hi) & m

    valid_counts = m.sum(axis=0)
    inside_counts = inside.sum(axis=0)

    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    np.divide(inside_counts, valid_counts, out=out, where=valid_counts > 0)

    return out
