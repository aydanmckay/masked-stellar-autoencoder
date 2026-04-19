import os
import sys

import numpy as np

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "training"))

from astrometry_features import (
    apply_parallax_input_policy,
    parallax_label_error_log10,
    parallax_label_log10,
)


def test_snr_clipped_policy():
    train = np.array([[1.0, 0.01], [2.0, 0.02]], dtype=np.float32)
    valid = train.copy()
    test = train.copy()
    et = np.array([[0.1, 0.001], [0.2, 0.002]], dtype=np.float32)
    ev = et.copy()
    es = et.copy()
    apply_parallax_input_policy(
        train, valid, test, et, ev, es, 0, "snr_clipped", snr_cap=50.0
    )
    assert np.allclose(train[:, 0], [10.0, 10.0])
    assert np.allclose(et[:, 0], 1.0)


def test_log10_labels_mask_nonpositive():
    pi = np.array([1.0, -1.0, 0.0], dtype=np.float64)
    y, m = parallax_label_log10(pi, floor_mas=1e-4)
    assert m[0] and not m[1] and not m[2]
    assert np.isfinite(y[0])
    assert np.isnan(y[1]) and np.isnan(y[2])


def test_log10_label_error_positive():
    pi = np.array([10.0])
    e = np.array([1.0])
    s = parallax_label_error_log10(pi, e, floor_mas=1e-4)
    assert s[0] > 0 and np.isfinite(s[0])
