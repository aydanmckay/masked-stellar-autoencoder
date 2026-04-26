"""Pure dict logic for eval checkpoint loading."""

import pytest

from masked_stellar_autoencoder.training.checkpoint_keys import (
    autoencoder_state_dict,
    prediction_head_state_dict,
)


def test_autoencoder_prefers_finetune_key():
    sd = {"w": 1}
    ckpt = {"autoencoder_state_dict": sd, "model_state_dict": {"x": 2}}
    assert autoencoder_state_dict(ckpt) is sd


def test_autoencoder_falls_back_to_pretrain_key():
    sd = {"m": 3}
    assert autoencoder_state_dict({"model_state_dict": sd}) is sd


def test_autoencoder_missing_raises():
    with pytest.raises(
        KeyError, match="Checkpoint must contain 'autoencoder_state_dict'"
    ):
        autoencoder_state_dict({})
    with pytest.raises(
        KeyError, match="Checkpoint must contain 'autoencoder_state_dict'"
    ):
        autoencoder_state_dict({"wrong_key": 1, "another_wrong_key": 2})


def test_prediction_head_required():
    with pytest.raises(KeyError, match="prediction_head_state_dict"):
        prediction_head_state_dict({"model_state_dict": {}})


def test_prediction_head_ok():
    h = {"h": 1}
    assert prediction_head_state_dict({"prediction_head_state_dict": h}) is h
