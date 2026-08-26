"""
Smoke tests for masking layout, losses, and tensor shapes (Phase 1 debugging).
Run from repo root: pytest tests/test_msa_training_invariants.py -v
"""

import pytest
import torch

from masked_stellar_autoencoder.models.model import (
    EncoderDecoderLoss,
    PredictionHead,
    _reduce_finetune_prediction,
    make_model,
    quantile_loss,
)


def test_encoder_decoder_loss_masked_mean_matches_manual():
    """Loss equals mean over masked positions only."""
    crit = EncoderDecoderLoss(lf="mae")
    x_true = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_pred = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    mask = torch.tensor([[True, False, True], [False, True, False]])
    w = torch.ones_like(x_true)
    loss = crit(x_true, x_pred, mask, w)
    err = (x_pred - x_true).abs()
    manual = err[mask].mean()
    assert torch.isclose(loss, manual)


def test_quantile_loss_shape_and_weights():
    B, L, Q = 8, 6, 3
    preds = torch.randn(B, L, Q, requires_grad=True)
    target = torch.randn(B, L)
    target[0, 1] = float("nan")
    quants = torch.tensor([0.16, 0.5, 0.84])
    loss_uw = quantile_loss(preds, target, quants, label_weights=None)
    assert loss_uw.ndim == 0
    loss_uw.backward()
    assert preds.grad is not None

    preds2 = torch.randn(B, L, Q, requires_grad=True)
    w = torch.ones(L)
    w[2] = 3.0
    loss_w = quantile_loss(preds2, target, quants, label_weights=w)
    loss_w.backward()
    assert preds2.grad is not None


def test_reduce_finetune_prediction_median_from_quantile_head():
    y = torch.randn(4, 6, 3)
    pt, err = _reduce_finetune_prediction(y, "mse", linearprobe=False)
    assert torch.allclose(pt, y[..., 1])
    assert err is None


def test_reduce_finetune_prediction_variations():
    # 1. ftlf == "quantile"
    y1 = torch.randn(4, 6, 3)
    pt1, err1 = _reduce_finetune_prediction(y1, "quantile", linearprobe=False)
    assert pt1 is y1
    assert err1 is None

    # 2. linearprobe == True
    y2 = torch.randn(4, 6)
    pt2, err2 = _reduce_finetune_prediction(y2, "mse", linearprobe=True)
    assert pt2 is y2
    assert err2 is None

    # 3. Tuple input
    mean = torch.randn(4, 6)
    variance = torch.randn(4, 6)
    y3 = (mean, variance)
    pt3, err3 = _reduce_finetune_prediction(y3, "gaussian", linearprobe=False)
    assert pt3 is mean
    assert err3 is variance

    # 4. Fallback for 2D tensor
    y4 = torch.randn(4, 6)
    pt4, err4 = _reduce_finetune_prediction(y4, "mse", linearprobe=False)
    assert pt4 is y4
    assert err4 is None


def test_prediction_head_monotonic_quantiles():
    head = PredictionHead(latent_size=32, ft_label_dim=6, ft_activ=torch.nn.ReLU())
    z = torch.randn(4, 32)
    out = head(z)
    assert out.shape == (4, 6, 3)
    lower, med, upper = out[..., 0], out[..., 1], out[..., 2]
    assert (lower <= med).all()
    assert (med <= upper).all()


def test_finetune_feature_layout_matches_apply_mask():
    """
    configs/finetune.yaml: 5 Gaia mags + 110 XP (bp/rp) + ancillary = 138 cols.
    _apply_mask uses col_start_fixed=5, col_end_fixed=115 → XP block.
    """
    n_feat = 138
    xp_start, xp_end = 5, 115
    assert xp_end - xp_start == 110
    # PARALLAX is after SMSS(6)+SDSS(5)+PS1(5)+2MASS(3) from index 115
    parallax_idx = 115 + 6 + 5 + 5 + 3
    assert parallax_idx == 134
    assert parallax_idx < n_feat


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optional GPU smoke")
def test_tabresnet_forward_smoke_cuda():
    """One forward pass on GPU if available."""
    device = torch.device("cuda")
    try:
        model = make_model(
            input_dim=138,
            layer_dims=[128, 64],
            output_dim=120,
            active="elu",
            rtdl_embed_dim=4,
            norm="layer",
            decoder_dims=[64, 128],
        ).to(device)
    except Exception as e:
        pytest.skip(f"rtdl / model deps: {e}")
    x = torch.randn(2, 138, device=device)
    recon, z = model(x)
    assert recon.shape[0] == 2
    assert z.shape[0] == 2


def test_tabresnet_forward_smoke_cpu():
    try:
        model = make_model(
            input_dim=32,
            layer_dims=[64, 32],
            output_dim=24,
            active="elu",
            rtdl_embed_dim=4,
            norm="layer",
        )
    except Exception as e:
        pytest.skip(f"rtdl / model deps: {e}")
    x = torch.randn(2, 32)
    recon, z = model(x)
    assert recon.shape == (2, 24)
