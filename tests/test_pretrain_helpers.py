from unittest.mock import MagicMock

import pytest
import torch

from masked_stellar_autoencoder.models.model import EncoderDecoderLoss, TabResnetWrapper


@pytest.fixture
def wrapper_stub():
    model = MagicMock()
    model.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
    scaler = MagicMock()
    scaler.scale_ = [1.0]
    scaler.center_ = [0.0]
    w = TabResnetWrapper.__new__(TabResnetWrapper)
    w.model = model
    w.featurescaler = scaler
    w.feature_cols = ["a", "b"]
    w.error_cols = ["e_a", "e_b"]
    w.recon_cols = ["a"]
    w.diff = 1
    w.device = torch.device("cpu")
    w.loss_fn = EncoderDecoderLoss(lf="mae")
    w.lasso = 0.0
    w.pert_features = False
    w.pt_save_str = "model.pth"
    w.pt_log_file = "loss.log"
    w.checkpoint_interval = None
    return w


def test_pretrain_checkpoint_payload_shape(wrapper_stub):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    wrapper_stub.model.state_dict = MagicMock(return_value={"w": torch.zeros(1)})
    payload = wrapper_stub._pretrain_checkpoint_payload(
        epoch=0, optimizer=optimizer, scheduler=scheduler, epoch_loss=1.0, loss_div=2.0
    )
    assert payload["epoch"] == 1
    assert payload["epoch_loss"] == 1.0
    assert "model_state_dict" in payload


def test_pretrain_reconstruction_loss_masked_mean(wrapper_stub):
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    eX = torch.ones_like(X)
    X_reconstructed = torch.tensor([[1.5, 0.0], [3.5, 0.0]])
    z = torch.zeros(2, 4)
    mask = torch.tensor([[False, True], [False, True]])
    nanmask = torch.ones_like(X, dtype=torch.bool)
    loss = wrapper_stub._pretrain_reconstruction_loss(
        X, eX, X_reconstructed, z, mask, nanmask
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
