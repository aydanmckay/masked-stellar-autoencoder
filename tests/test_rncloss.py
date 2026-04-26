import os
import sys

import pytest

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _repo)

torch = pytest.importorskip("torch")

from masked_stellar_autoencoder.models.model import RnCLoss


def test_rncloss_forward_and_backward():
    """Test that RnCLoss processes the forward pass and allows backward gradients."""
    loss_fn = RnCLoss(temperature=2.0)

    bs = 4
    feat_dim = 8
    label_dim = 2

    # features: [bs, 2, feat_dim]
    features = torch.randn(bs, 2, feat_dim, requires_grad=True)
    # labels: [bs, label_dim]
    labels = torch.randn(bs, label_dim)

    loss = loss_fn(features, labels)

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

    # Check backward pass
    loss.backward()

    assert features.grad is not None
    assert features.grad.shape == (bs, 2, feat_dim)


def test_rncloss_zero_loss_same_labels():
    """Test RnCLoss with identical labels and features to ensure it handles it gracefully."""
    loss_fn = RnCLoss()

    bs = 3
    feat_dim = 5
    label_dim = 1

    features = torch.ones(bs, 2, feat_dim, requires_grad=True)
    labels = torch.ones(bs, label_dim)

    loss = loss_fn(features, labels)

    # Even if identical, should not crash, should just compute some scalar
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
