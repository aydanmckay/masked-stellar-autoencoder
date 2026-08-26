import pytest
import torch.nn as nn

from masked_stellar_autoencoder.models.blocks import ResBlock


def test_resblock_activation_elu():
    block = ResBlock(in_features=10, out_features=10, active="elu")
    assert isinstance(block.active, nn.ELU)
    assert block.active.inplace is True


def test_resblock_activation_gelu():
    block = ResBlock(in_features=10, out_features=10, active="gelu")
    assert isinstance(block.active, nn.GELU)


def test_resblock_activation_relu():
    block = ResBlock(in_features=10, out_features=10, active="relu")
    assert isinstance(block.active, nn.ReLU)
    assert block.active.inplace is True


def test_resblock_unsupported_activation():
    with pytest.raises(
        ValueError,
        match="Unsupported activation type: foo. Use 'elu', 'gelu', or 'relu'",
    ):
        ResBlock(in_features=10, out_features=10, active="foo")


def test_resblock_norm_batch():
    block = ResBlock(in_features=10, out_features=10, norm="batch")
    assert isinstance(block.normal, nn.BatchNorm1d)


def test_resblock_norm_layer():
    block = ResBlock(in_features=10, out_features=10, norm="layer")
    assert isinstance(block.normal, nn.LayerNorm)


def test_resblock_unsupported_norm():
    with pytest.raises(
        ValueError, match="Unsupported norm type: foo. Use 'batch' or 'layer'"
    ):
        ResBlock(in_features=10, out_features=10, norm="foo")
