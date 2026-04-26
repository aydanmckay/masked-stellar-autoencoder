"""Tests for quantile_loss and σ-weights (requires torch)."""

import os
import sys

import pytest


torch = pytest.importorskip("torch")

from masked_stellar_autoencoder.models.model import (
    _sigma_pinball_weights,
    quantile_loss,
)


def test_quantile_loss_sample_weight_changes_value():
    preds = torch.zeros(2, 3, 3)
    target = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quantiles = torch.tensor([0.16, 0.5, 0.84])
    loss_u = quantile_loss(preds, target, quantiles, None, None)
    sw = torch.ones_like(target)
    sw[0, :] = 2.0
    loss_w = quantile_loss(preds, target, quantiles, None, sw)
    assert loss_u.shape == ()
    assert loss_w.shape == ()
    assert not torch.allclose(loss_u, loss_w)


def test_sigma_pinball_weights_zero_for_nan_target():
    sig = torch.ones(2, 3)
    y = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]])
    w = _sigma_pinball_weights(sig, y, floor=1e-3, max_w=1e6, normalize_batch=False)
    assert w[0, 1].item() == 0.0
    assert w[0, 0].item() > 0.0


def test_sigma_pinball_weights_extreme_values():
    sig = torch.tensor([[0.0, 1e-12, float("inf")], [float("-inf"), float("nan"), 2.0]])
    y = torch.ones_like(sig)

    w = _sigma_pinball_weights(sig, y, floor=1e-3, max_w=100.0, normalize_batch=False)

    assert w[0, 0].item() == 100.0
    assert w[0, 1].item() == 100.0
    assert torch.isclose(w[0, 2], torch.tensor(1.0 / (1.0 + 1e-6)))
    assert torch.isclose(w[1, 0], torch.tensor(1.0 / (1.0 + 1e-6)))
    assert torch.isclose(w[1, 1], torch.tensor(1.0 / (1.0 + 1e-6)))
    assert torch.isclose(w[1, 2], torch.tensor(1.0 / (4.0 + 1e-6)))
