"""Per-feature scaling for error-scaled Gaussian input augmentation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def pert_channel_scale_vector(
    feature_cols: Sequence[str],
    *,
    pert_ebv_scale: float = 1.0,
) -> np.ndarray:
    """
    Length ``len(feature_cols)`` multipliers applied to ``pert_features`` noise.

    Schlegel-style E(B-V) is often much more certain than photometric errors;
    set ``pert_ebv_scale`` to ``0.0`` to disable jitter on that channel only.
    """
    out = np.ones(len(feature_cols), dtype=np.float32)
    if "EBV" in feature_cols:
        out[feature_cols.index("EBV")] = float(pert_ebv_scale)
    return out
