import numpy as np
import pytest

torch = pytest.importorskip("torch")

from masked_stellar_autoencoder.training.eval_ensemble import (  # noqa: E402
    _bins_true_parallax_quartiles,
    _ensemble_median_predictions,
    _feature_batch_tensor,
)


class _Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Identity()

    def forward(self, x):
        return self.encoder(x)


class _Head(torch.nn.Module):
    def forward(self, z):
        return z


def test_feature_batch_tensor_nan_sentinel():
    x = np.array([[1.0, np.nan]], dtype=np.float64)
    t = _feature_batch_tensor(x, torch.device("cpu"))
    assert t.shape == (1, 2)
    assert float(t[0, 1]) == -9999.0


def test_ensemble_median_predictions(monkeypatch):
    model = _Enc()
    head = _Head()
    states = [{"autoencoder_state_dict": {}, "prediction_head_state_dict": {}}]
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(
        "masked_stellar_autoencoder.training.eval_ensemble.autoencoder_state_dict",
        lambda _s: {},
    )
    monkeypatch.setattr(
        "masked_stellar_autoencoder.training.eval_ensemble.prediction_head_state_dict",
        lambda _s: {},
    )
    preds = _ensemble_median_predictions(
        model,
        head,
        states,
        X,
        torch.device("cpu"),
        batch_size=1,
        linear_probe=True,
    )

    np.testing.assert_allclose(preds, X)


def test_bins_true_parallax_quartiles():
    label_names = ["teff", "logg", "fe_h", "alpha", "age", "parallax"]
    n = 40
    y_t = np.tile(np.linspace(1, 4, n), (6, 1)).T
    y_p = y_t + 0.01
    block = _bins_true_parallax_quartiles(y_t, y_p, label_names, "xp_on")
    assert any(k.startswith("xp_on_pi_q") for k in block)
