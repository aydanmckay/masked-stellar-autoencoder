import pytest
import torch

from masked_stellar_autoencoder.models.model import (
    MaskedGaussianNLLLoss,
    MaskedMAELoss,
    MaskedMSELoss,
    WeightedMaskedMSELoss,
)


@pytest.fixture
def device():
    return "cpu"


def test_masked_mse_loss_backward_equivalence(device):
    # Setup inputs with NaNs
    target = torch.randn(10, 5, device=device)
    target[0, 0] = float("nan")
    target[2, 3] = float("nan")

    input1 = torch.randn(10, 5, device=device, requires_grad=True)
    input2 = input1.clone().detach().requires_grad_(True)

    # Simple reference implementation using boolean indexing
    def reference_masked_mse(target, input, reduction="mean"):
        mask = ~torch.isnan(target)
        masked_input = input[mask]
        masked_target = target[mask]
        masked_error = (masked_input - masked_target) ** 2

        if masked_error.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        if reduction == "mean":
            return masked_error.mean()
        elif reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error

    loss_fn = MaskedMSELoss(reduction="mean")

    # Forward
    l1 = reference_masked_mse(target, input1, reduction="mean")
    l2 = loss_fn(target, input2)

    # Backward
    l1.backward()
    l2.backward()

    assert torch.allclose(l1, l2)
    assert torch.allclose(input1.grad, input2.grad)


def test_masked_mae_loss_backward_equivalence(device):
    # Setup inputs with NaNs
    target = torch.randn(10, 5, device=device)
    target[0, 0] = float("nan")
    target[2, 3] = float("nan")

    input1 = torch.randn(10, 5, device=device, requires_grad=True)
    input2 = input1.clone().detach().requires_grad_(True)

    # Simple reference implementation using boolean indexing
    def reference_masked_mae(target, input, reduction="mean"):
        mask = ~torch.isnan(target)
        masked_input = input[mask]
        masked_target = target[mask]
        masked_error = torch.abs(masked_input - masked_target)

        if masked_error.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        if reduction == "mean":
            return masked_error.mean()
        elif reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error

    loss_fn = MaskedMAELoss(reduction="mean")

    # Forward
    l1 = reference_masked_mae(target, input1, reduction="mean")
    l2 = loss_fn(target, input2)

    # Backward
    l1.backward()
    l2.backward()

    assert torch.allclose(l1, l2)
    assert torch.allclose(input1.grad, input2.grad)


def test_weighted_masked_mse_loss_backward_equivalence(device):
    # Setup inputs with NaNs
    target = torch.randn(10, 5, device=device)
    target[0, 0] = float("nan")
    target[2, 3] = float("nan")

    weight = torch.rand(10, 5, device=device)
    weight[1, 1] = float("nan")

    input1 = torch.randn(10, 5, device=device, requires_grad=True)
    input2 = input1.clone().detach().requires_grad_(True)

    # Simple reference implementation using boolean indexing
    def reference_weighted_masked_mse(
        target, input, weight, reduction="mean", eps=1e-8
    ):
        mask = (~torch.isnan(target)) & (~torch.isnan(weight))
        masked_input = input[mask]
        masked_target = target[mask]
        masked_weights = weight[mask]
        masked_error = (masked_input - masked_target) ** 2
        masked_error = masked_error * masked_weights

        if reduction == "mean":
            return masked_error.sum() / (masked_weights.sum() + eps)
        elif reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error

    loss_fn = WeightedMaskedMSELoss(reduction="mean")

    # Forward
    l1 = reference_weighted_masked_mse(target, input1, weight, reduction="mean")
    l2 = loss_fn(target, input2, weight)

    # Backward
    l1.backward()
    l2.backward()

    assert torch.allclose(l1, l2)
    assert torch.allclose(input1.grad, input2.grad)


def test_masked_gaussian_nll_loss_backward_equivalence(device):
    import math

    # Setup inputs with NaNs
    target = torch.randn(10, 5, device=device)
    target[0, 0] = float("nan")
    target_var = torch.rand(10, 5, device=device) + 0.1
    target_var[1, 1] = float("nan")

    pred_mean1 = torch.randn(10, 5, device=device, requires_grad=True)
    pred_mean2 = pred_mean1.clone().detach().requires_grad_(True)

    pred_var_base = torch.rand(10, 5, device=device) + 0.1
    pred_var1 = pred_var_base.clone().requires_grad_(True)
    pred_var2 = pred_var_base.clone().requires_grad_(True)

    # Simple reference implementation using boolean indexing
    def reference_masked_gaussian_nll(
        pred_mean, target, pred_var, target_var, reduction="mean", eps=1e-6
    ):
        mask = (~torch.isnan(target)) & (~torch.isnan(target_var))

        pred_mean = pred_mean[mask]
        pred_var = pred_var[mask]
        target = target[mask]
        target_var = target_var[mask]

        var = pred_var.clamp(min=eps)
        obs_var = target_var.clamp(min=eps)

        err = var + obs_var
        diff_squared = (pred_mean - target) ** 2

        nll = 0.5 * (torch.log(err) + (diff_squared / err)) + 0.5 * math.log(
            2 * math.pi
        )

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll

    loss_fn = MaskedGaussianNLLLoss(reduction="mean")

    # Forward
    l1 = reference_masked_gaussian_nll(
        pred_mean1, target, pred_var1, target_var, reduction="mean"
    )
    l2 = loss_fn(pred_mean2, target, pred_var2, target_var)

    # Backward
    l1.backward()
    l2.backward()

    assert torch.allclose(l1, l2)
    assert torch.allclose(pred_mean1.grad, pred_mean2.grad)
    assert torch.allclose(pred_var1.grad, pred_var2.grad)
