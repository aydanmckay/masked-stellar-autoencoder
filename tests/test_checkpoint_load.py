"""torch.load compatibility for training checkpoints (NumPy in dict, etc.)."""

import os
import sys

import numpy as np
import torch


from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted


def test_torch_load_trusted_accepts_numpy_in_dict(tmp_path):
    path = tmp_path / "ckpt_np.pth"
    torch.save({"meta": np.float64(1.5), "w": torch.ones(2)}, str(path))
    out = torch_load_trusted(str(path), map_location="cpu", weights_only=False)
    assert isinstance(out["meta"], np.floating)
    assert out["w"].shape == (2,)


def test_torch_load_trusted_state_dict_only(tmp_path):
    path = tmp_path / "ckpt_sd.pth"
    lin = torch.nn.Linear(2, 3)
    torch.save({"autoencoder_state_dict": lin.state_dict()}, str(path))
    out = torch_load_trusted(str(path), map_location="cpu")
    assert "autoencoder_state_dict" in out
