"""
Principled handling of Gaia parallax / proper motions in the feature matrix.

The PARALLAX *slot* in feature_cols can carry either raw parallax (mas) or a
derived quantity (e.g. parallax signal-to-noise) so the encoder is not fed the
same real-valued quantity as the supervised parallax target without structure.
"""

from __future__ import annotations

import numpy as np


def apply_parallax_input_policy(
    train: np.ndarray,
    valid: np.ndarray,
    test: np.ndarray,
    etrain: np.ndarray,
    evalid: np.ndarray,
    etest: np.ndarray,
    parallax_col: int,
    policy: str,
    *,
    snr_cap: float = 10.0,
    sigma_floor_mas: float = 1e-6,
) -> None:
    """
    In-place: replace the PARALLAX feature column.

    Policies:
      - ``legacy_raw``: no change (raw parallax in mas as in Gaia table).
      - ``snr_clipped``: column becomes clip(pi / max(sigma_pi, floor), -cap, +cap).
        The associated error column is set to 1.0 so ``pert_features`` noise
        does not assume mas units for that slot.

    Proper motions (pmra, pmdec) are unchanged; they carry orthogonal kinematic
    information and are not duplicated as prediction targets.
    """
    if policy in (None, "", "legacy_raw"):
        return

    if policy == "snr_clipped":

        def _one(x: np.ndarray, ex: np.ndarray) -> None:
            pi = x[:, parallax_col].astype(np.float64, copy=False)
            sig = np.maximum(
                ex[:, parallax_col].astype(np.float64, copy=False), sigma_floor_mas
            )
            snr = pi / sig
            x[:, parallax_col] = np.clip(snr, -snr_cap, snr_cap).astype(np.float32)
            ex[:, parallax_col] = 1.0

        _one(train, etrain)
        _one(valid, evalid)
        _one(test, etest)
        return

    raise ValueError(f"Unknown astrometry_input_policy: {policy!r}")


def parallax_label_log10(
    pi_mas: np.ndarray,
    floor_mas: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Target values for log10(parallax/mas) with invalid (non-positive) training
    labels set to NaN (excluded by quantile loss mask).

    Returns (y_log10, valid_mask).
    """
    pi = np.asarray(pi_mas, dtype=np.float64)
    m = pi > 0
    y = np.full(pi.shape, np.nan, dtype=np.float64)
    y[m] = np.log10(np.maximum(pi[m], floor_mas))
    return y.astype(np.float32), m


def parallax_label_error_log10(
    pi_mas: np.ndarray,
    e_pi_mas: np.ndarray,
    floor_mas: float,
) -> np.ndarray:
    """
    Delta-method std dev of log10(pi) from Gaussian error on pi:
        sigma_log10 ~ e_pi / (ln(10) * max(pi, floor))
    Invalid rows get NaN.
    """
    pi = np.asarray(pi_mas, dtype=np.float64)
    e = np.asarray(e_pi_mas, dtype=np.float64)
    denom = np.log(10.0) * np.maximum(pi, floor_mas)
    out = np.full(pi.shape, np.nan, dtype=np.float64)
    m = pi > 0
    out[m] = np.maximum(e[m], 1e-12) / np.maximum(denom[m], 1e-12)
    return out.astype(np.float32)


def parallax_label_asinh(
    pi_mas: np.ndarray,
    scale_mas: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Target values for arcsinh(parallax/scale).
    Accepts all values (positive and negative).

    Returns (y_asinh, valid_mask).
    """
    pi = np.asarray(pi_mas, dtype=np.float64)
    # All finite values are valid
    m = np.isfinite(pi)
    y = np.full(pi.shape, np.nan, dtype=np.float64)
    y[m] = np.arcsinh(pi[m] / scale_mas)
    return y.astype(np.float32), m


def parallax_label_error_asinh(
    pi_mas: np.ndarray,
    e_pi_mas: np.ndarray,
    scale_mas: float = 1.0,
) -> np.ndarray:
    """
    Delta-method std dev of arcsinh(pi/scale) from Gaussian error on pi.
    d/dpi arcsinh(pi/scale) = 1 / (scale * sqrt((pi/scale)^2 + 1))
    """
    pi = np.asarray(pi_mas, dtype=np.float64)
    e = np.asarray(e_pi_mas, dtype=np.float64)
    out = np.full(pi.shape, np.nan, dtype=np.float64)
    m = np.isfinite(pi) & np.isfinite(e)
    denom = scale_mas * np.sqrt((pi[m] / scale_mas) ** 2 + 1.0)
    out[m] = e[m] / np.maximum(denom, 1e-12)
    return out.astype(np.float32)
