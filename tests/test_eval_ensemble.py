import os
import sys

import numpy as np
import torch

_training = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "src", "masked_stellar_autoencoder", "training"
    )
)
sys.path.insert(0, _training)

from eval_ensemble import predict_batches


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Simply return x as the latent representation
        return x


class DummyHeadLinear(torch.nn.Module):
    def forward(self, z):
        # Linear probe output shape: (batch_size, num_labels)
        return z * 2.0


class DummyHeadQuantile(torch.nn.Module):
    def forward(self, z):
        # Quantile head output shape: (batch_size, num_labels, 3)
        # We will create a dummy tensor
        B, L = z.shape
        out = torch.zeros(B, L, 3, device=z.device)
        out[:, :, 0] = z - 1.0  # lower
        out[:, :, 1] = z  # median
        out[:, :, 2] = z + 1.0  # upper
        return out


def test_predict_batches_linear_probe():
    model = DummyModel()
    model.encoder = DummyModel()  # predict_batches uses model.encoder(xb)
    head = DummyHeadLinear()

    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    device = torch.device("cpu")

    preds = predict_batches(
        model,
        head,
        X,
        device=device,
        batch_size=2,
        linear_probe=True,
        return_full_quantiles=False,
    )

    assert preds.shape == (5, 2)
    expected = X * 2.0
    np.testing.assert_allclose(preds, expected)


def test_predict_batches_quantile_head_median_only():
    model = DummyModel()
    model.encoder = DummyModel()
    head = DummyHeadQuantile()

    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    device = torch.device("cpu")

    preds = predict_batches(
        model,
        head,
        X,
        device=device,
        batch_size=2,
        linear_probe=False,
        return_full_quantiles=False,
    )

    assert preds.shape == (5, 2)
    # Median is equal to z, which is x
    np.testing.assert_allclose(preds, X)


def test_predict_batches_quantile_head_full_quantiles():
    model = DummyModel()
    model.encoder = DummyModel()
    head = DummyHeadQuantile()

    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    device = torch.device("cpu")

    preds = predict_batches(
        model,
        head,
        X,
        device=device,
        batch_size=2,
        linear_probe=False,
        return_full_quantiles=True,
    )

    assert preds.shape == (5, 2, 3)
    np.testing.assert_allclose(preds[:, :, 0], X - 1.0)
    np.testing.assert_allclose(preds[:, :, 1], X)
    np.testing.assert_allclose(preds[:, :, 2], X + 1.0)
