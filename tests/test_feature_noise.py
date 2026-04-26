import os
import sys

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo, "training"))

from masked_stellar_autoencoder.training.feature_noise import pert_channel_scale_vector


def test_pert_channel_scale_ebv():
    cols = ["G", "EBV", "PARALLAX"]
    v = pert_channel_scale_vector(cols, pert_ebv_scale=0.0)
    assert v.shape == (3,)
    assert v[cols.index("EBV")] == 0.0
    assert v[0] == 1.0
