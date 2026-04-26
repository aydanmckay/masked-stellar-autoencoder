import numpy as np
import pytest

from masked_stellar_autoencoder.training.conformal import (  # noqa: E402
    apply_cqr_offsets_inplace,
    calibrate_cqr_offsets,
    interval_coverage,
)


def test_calibrate_and_apply_widens_interval():
    rng = np.random.default_rng(0)
    n, ell = 500, 2
    y = rng.normal(size=(n, ell))
    q_lo = y - 0.01
    q_med = y
    q_hi = y + 0.01
    pred = np.stack([q_lo, q_med, q_hi], axis=2)
    doc = calibrate_cqr_offsets(y, pred, alpha=0.1)
    assert len(doc["offsets_lower"]) == ell
    pred2 = pred.copy()
    apply_cqr_offsets_inplace(pred2, doc)
    assert np.all(pred2[:, :, 0] <= pred[:, :, 0] + 1e-9)
    assert np.all(pred2[:, :, 2] >= pred[:, :, 2] - 1e-9)


def test_interval_coverage_perfect_interval():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    lo = y - 1.0
    hi = y + 1.0
    cov = interval_coverage(y, lo, hi)
    assert np.allclose(cov, 1.0)


def test_apply_cqr_offsets_rejects_mismatched_length():
    pred = np.zeros((4, 2, 3))
    calib = {"offsets_lower": [0.1], "offsets_upper": [0.1, 0.2]}
    with pytest.raises(ValueError, match="Calibration offsets"):
        apply_cqr_offsets_inplace(pred, calib)
