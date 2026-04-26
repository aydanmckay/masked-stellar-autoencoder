# loading the packages
import logging
import math
import os
import random
from typing import Optional

# import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.base import BaseEstimator
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .blocks import TabResnet
from .checkpoint_load import torch_load_trusted


class MaskedGaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_mean, target, pred_var, target_var):
        # Entries of var must be non-negative
        if isinstance(target_var, float):
            if target_var < 0:
                raise ValueError("var has negative entry/entries")
            # target_var = target_var * torch.ones_like(input)
        elif torch.any(target_var < 0):
            raise ValueError("var has negative entry/entries")

        # mask = ~torch.isnan(target)
        mask = (~torch.isnan(target)) & (~torch.isnan(target_var))

        pred_mean = pred_mean[mask]
        pred_var = pred_var[mask]
        target = target[mask]
        target_var = target_var[mask]

        # Clamp variance to avoid instability
        var = pred_var.clamp(min=self.eps)
        obs_var = target_var.clamp(min=self.eps)

        err = var + obs_var
        diff_squared = (pred_mean - target) ** 2

        # Compute Gaussian NLL
        nll = 0.5 * (torch.log(err) + (diff_squared / err)) + 0.5 * math.log(
            2 * math.pi
        )

        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll


class WeightedMaskedMSELoss(nn.Module):
    def __init__(self, reduction="mean", eps=1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps  # To avoid divide-by-zero if all values are masked

    def forward(self, target, input, weight):
        # Create mask for non-NaN targets
        # mask = ~torch.isnan(weight)
        mask = (~torch.isnan(target)) & (~torch.isnan(weight))

        masked_input = input[mask]
        masked_target = target[mask]
        masked_weights = weight[mask]
        masked_error = (masked_input - masked_target) ** 2
        masked_error = masked_error * masked_weights

        # Apply mask
        # diff_squared = (input - target)**2
        # weighted_error = diff_squared * weight

        # masked_error = weighted_error[mask]
        # masked_weights = weight[mask]

        if self.reduction == "mean":
            return masked_error.sum() / (masked_weights.sum() + self.eps)
        elif self.reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, target, input):
        # Create a mask for non-NaN targets
        mask = ~torch.isnan(target)

        # Compute squared error only where target is not NaN
        masked_input = input[mask]
        masked_target = target[mask]
        masked_error = (masked_input - masked_target) ** 2

        if masked_error.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        if self.reduction == "mean":
            return masked_error.mean()
        elif self.reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error


class MaskedMAELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, target, input):
        # Create a mask for non-NaN targets
        mask = ~torch.isnan(target)

        # Compute absolute error only where target is not NaN
        masked_input = input[mask]
        masked_target = target[mask]
        masked_error = torch.abs(masked_input - masked_target)

        if masked_error.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        if self.reduction == "mean":
            return masked_error.mean()
        elif self.reduction == "sum":
            return masked_error.sum()
        else:
            return masked_error


class LabelDifference(nn.Module):
    """
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    """

    def __init__(self, distance_type="l1"):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == "l1":
            return torch.cdist(labels, labels, p=1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    """
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    """

    def __init__(self, similarity_type="l2"):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == "l2":
            return -torch.cdist(features, features, p=2)
        else:
            raise ValueError(self.similarity_type)


class RnCLoss(nn.Module):
    """
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    """

    def __init__(self, temperature=2, label_diff="l1", feature_sim="l2"):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(
            n, n - 1
        )
        exp_logits = exp_logits.masked_select(
            (1 - torch.eye(n).to(logits.device)).bool()
        ).view(n, n - 1)
        label_diffs = label_diffs.masked_select(
            (1 - torch.eye(n).to(logits.device)).bool()
        ).view(n, n - 1)

        loss = 0.0
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (
                label_diffs >= pos_label_diffs.view(-1, 1)
            ).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log(
                (neg_mask * exp_logits).sum(dim=-1)
            )  # 2bs
            loss += -(pos_log_probs / (n * (n - 1))).sum()

        return loss


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path  # Filepath to save the model
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss, model):
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.save_checkpoint(
                model
            )  # Save the model when the best validation loss is found
        elif validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(
                    f"Validation loss improved to {self.best_loss:.6f}, saving model."
                )
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class EncoderDecoderLoss(nn.Module):
    r"""
    From pytorch-widedeep with some of my own modifications:
    '_Standard_' Encoder Decoder Loss. Loss applied during the Endoder-Decoder
     Self-Supervised Pre-Training routine available in this library

    :information_source: **NOTE**: This loss is in principle not exposed to
     the user, as it is used internally in the library, but it is included
     here for completion.

    The implementation of this lost is based on that at the
    [tabnet repo](https://github.com/dreamquark-ai/tabnet), which is in itself an
    adaptation of that in the original paper [TabNet: Attentive
    Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442).

    Parameters
    ----------
    eps: float
        Simply a small number to avoid dividing by zero
    """

    def __init__(self, eps: float = 1e-9, lf="mse"):
        super(EncoderDecoderLoss, self).__init__()
        self.eps = eps
        self.cost = lf

    def forward(
        self, x_true: Tensor, x_pred: Tensor, mask: Tensor, w: Tensor
    ) -> Tensor:
        r"""
        Parameters
        ----------
        x_true: Tensor
            Embeddings of the input data
        x_pred: Tensor
            Reconstructed embeddings
        mask: Tensor
            Mask with 1s indicated that the reconstruction, and therefore the
            loss, is based on those features.

        Examples
        --------
        >>> import torch
        >>> from pytorch_widedeep.losses import EncoderDecoderLoss
        >>> x_true = torch.rand(3, 3)
        >>> x_pred = torch.rand(3, 3)
        >>> mask = torch.empty(3, 3).random_(2)
        >>> loss = EncoderDecoderLoss()
        >>> res = loss(x_true, x_pred, mask)
        """

        # Correctly apply mask to errors before squaring
        errors = torch.where(
            mask.bool(), x_pred - x_true, torch.tensor(0.0, device=x_true.device)
        )
        if self.cost == "mse":
            reconstruction_errors = errors**2
        elif self.cost == "mae":
            reconstruction_errors = abs(errors)
        elif self.cost == "wmse":
            if w is None:
                raise ValueError(
                    "Weight tensor w is required for wmse loss but got None"
                )
            reconstruction_errors = w * (errors**2)
        elif self.cost == "wmae":
            if w is None:
                raise ValueError(
                    "Weight tensor w is required for wmae loss but got None"
                )
            reconstruction_errors = w * abs(errors)

        # Mean squared (or absolute) error over masked elements only — avoids
        # per-column divisors that up-weight rarely masked features in a batch.
        denom = mask.to(dtype=reconstruction_errors.dtype).sum().clamp_min(self.eps)
        loss = reconstruction_errors.sum() / denom

        return loss


class PredictionHead(nn.Module):
    def __init__(self, latent_size, ft_label_dim, ft_activ):
        super(PredictionHead, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(latent_size, 2048),
            ft_activ,
            nn.Linear(2048, 2048),
            ft_activ,
            nn.Linear(2048, 1024),
            ft_activ,
            nn.Linear(1024, 512),
            ft_activ,
            nn.Linear(512, 256),
            ft_activ,
        )
        self.output_y = nn.Linear(256, ft_label_dim)
        self.output_upper = nn.Linear(256, ft_label_dim)
        self.output_lower = nn.Linear(256, ft_label_dim)

    def forward(self, x):
        h = self.shared(x)
        y_median = self.output_y(h)

        # Predict offsets from median to ensure monotonicity: lower ≤ median ≤ upper
        # Use softplus to ensure positive offsets
        lower_offset = torch.nn.functional.softplus(self.output_lower(h))
        upper_offset = torch.nn.functional.softplus(self.output_upper(h))

        y_lower = y_median - lower_offset
        y_upper = y_median + upper_offset

        return torch.stack([y_lower, y_median, y_upper], dim=2)


def quantile_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    label_weights: Optional[Tensor] = None,
    sample_weight: Optional[Tensor] = None,
) -> torch.Tensor:
    """
    Pinball / quantile loss. Optionally up-weight rare labels (e.g. [Fe/H]) so
    solar-metallicity stars do not dominate the gradient.

    ``sample_weight`` (B, L) scales each label per example, e.g. inverse variance
    from scaled label uncertainties; combined multiplicatively with ``label_weights``.
    """
    mask = ~torch.isnan(target)
    target_expanded = target.unsqueeze(2).expand_as(preds)
    quantiles = quantiles.view(1, 1, -1)
    error = target_expanded - preds
    loss = torch.max((quantiles - 1) * error, quantiles * error)
    mask_expanded = mask.unsqueeze(2).expand_as(loss)
    w_eff = mask_expanded.to(dtype=loss.dtype)
    if label_weights is not None:
        w_lab = (
            label_weights.to(device=loss.device, dtype=loss.dtype)
            .view(1, -1, 1)
            .expand_as(loss)
        )
        w_eff = w_eff * w_lab
    if sample_weight is not None:
        w_s = (
            sample_weight.to(device=loss.device, dtype=loss.dtype)
            .unsqueeze(2)
            .expand_as(loss)
        )
        w_eff = w_eff * w_s
    if label_weights is None and sample_weight is None:
        return loss[mask_expanded].mean()
    return (loss * w_eff).sum() / w_eff.sum().clamp_min(1e-8)


def _sigma_pinball_weights(
    sigma_scaled: Tensor,
    y: Tensor,
    floor: float,
    max_w: float,
    normalize_batch: bool,
) -> Tensor:
    """Inverse-variance style weights (B, L) in scaled label-error space."""
    sig = torch.nan_to_num(sigma_scaled, nan=1.0, posinf=1.0, neginf=1.0)
    w = 1.0 / (sig * sig + float(floor) ** 2)
    w = w.clamp(max=float(max_w))
    if normalize_batch:
        w = w / (w.mean(dim=0, keepdim=True).clamp_min(1e-8))
    w = torch.where(torch.isnan(y), torch.zeros_like(w), w)
    return w


def _reduce_finetune_prediction(y_raw: Tensor, ftlf: str, linearprobe: bool):
    """
    Non-quantile losses need a single (B, L) prediction. Quantile head returns (B, L, 3).
    Legacy code paths may return a (mean, err) tuple for Gaussian NLL.
    """
    if ftlf == "quantile":
        return y_raw, None
    if linearprobe:
        return y_raw, None
    if isinstance(y_raw, Tensor) and y_raw.dim() == 3:
        return y_raw[..., 1], None
    if isinstance(y_raw, (tuple, list)) and len(y_raw) >= 2:
        return y_raw[0], y_raw[1]
    return y_raw, None


# creating a training wrapper for the algorithm
class TabResnetWrapper(BaseEstimator):
    def __init__(
        self,
        model,
        datafile,
        scaler,
        feature_cols,
        error_cols,
        recon_cols,
        label_scalers=None,
        latent_size=256,
        xp_masking_ratio=0.9,
        m_masking_ratio=0.9,
        lr=1e-3,
        optimizer="adam",
        wd=0,
        lasso=0,
        lf="mse",
        pt_save_str="pt_model.pth",
        ft_save_str="ft_model.pth",
        pt_log_file="pt_loss.log",
        ft_log_file="ft_loss.log",
        checkpoint_interval=None,
        pert_features=False,
        pert_scale=1.0,
        mask_mixture_xp_full_frac: float = 0.0,
        pert_channel_scale: Optional[np.ndarray] = None,
        scheduler_cosine_t0: int = 10,
        scheduler_cosine_t_mult: int = 2,
        scheduler_eta_min_factor: float = 0.01,
    ):
        """
        Changes to the original that can predict ages are the following:
        periodic embeddings
        scaling the coefficients with the RobustScaler
        changing the mask value to -9999
        cosine LR schedule with warm restarts (see ``scheduler_cosine_*`` on the wrapper)
        different masking ratios

        """
        self.model = model
        # Validate and handle datafile
        if hasattr(datafile, "keys"):
            self.datafile = datafile
        elif isinstance(datafile, str):
            try:
                import h5py

                self.datafile = h5py.File(datafile, "r")
            except Exception as e:
                raise ValueError(f"Could not open datafile '{datafile}': {e}")
        else:
            raise ValueError("datafile must be an open HDF5 file or file path")

        self.featurescaler = scaler
        self.label_scalers = label_scalers
        if (
            hasattr(self.featurescaler, "scale_")
            and self.featurescaler.scale_ is not None
        ):
            self.scale_factors = (
                self.featurescaler.scale_
            )  # This is the IQR used by RobustScaler for each feature
        else:
            raise ValueError(
                "Scaler must be fitted and have scale_ attribute before initializing wrapper"
            )
        self.feature_cols = feature_cols
        self.error_cols = error_cols
        self.recon_cols = recon_cols
        self.diff = len(feature_cols) - len(recon_cols)
        self.xp_masking_ratio = xp_masking_ratio
        self.m_masking_ratio = m_masking_ratio
        self.mask_mixture_xp_full_frac = float(mask_mixture_xp_full_frac)
        self.lr = lr
        self.opt = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn = EncoderDecoderLoss(lf=lf)
        self.latent_size = latent_size
        self.lasso = lasso
        self.wd = wd

        self.pt_save_str = pt_save_str
        self.ft_save_str = ft_save_str
        self.pt_log_file = pt_log_file
        self.ft_log_file = ft_log_file
        self.checkpoint_interval = checkpoint_interval
        self.scheduler_cosine_t0 = int(scheduler_cosine_t0)
        self.scheduler_cosine_t_mult = int(scheduler_cosine_t_mult)
        self.scheduler_eta_min_factor = float(scheduler_eta_min_factor)
        self.pert_features = pert_features
        self.pert_scale = pert_scale
        nfeat = len(feature_cols)
        if pert_channel_scale is None:
            self._pert_channel_scale_np = np.ones(nfeat, dtype=np.float32)
        else:
            pc = np.asarray(pert_channel_scale, dtype=np.float32).reshape(-1)
            if pc.shape[0] != nfeat:
                raise ValueError(
                    f"pert_channel_scale length {pc.shape[0]} != len(feature_cols)={nfeat}"
                )
            self._pert_channel_scale_np = pc
        self.lp: Optional[nn.Linear] = None
        self.ft: Optional[PredictionHead] = None

        try:
            self.parallax_feature_idx = feature_cols.index("PARALLAX")
        except ValueError:
            self.parallax_feature_idx = None

    def _pert_noise(self, X_batch: Tensor, eX_batch: Tensor) -> Tensor:
        """Gaussian noise scaled by per-feature errors and ``pert_channel_scale``."""
        noise = torch.randn_like(X_batch) * eX_batch * self.pert_scale
        w = torch.as_tensor(
            self._pert_channel_scale_np,
            device=X_batch.device,
            dtype=noise.dtype,
        )
        if w.dim() != 1 or w.shape[0] != X_batch.shape[1]:
            raise RuntimeError("pert_channel_scale length must match feature dimension")
        return noise * w.unsqueeze(0)

    def _apply_mask(
        self, X, col_start_fixed=5, col_end_fixed=115, col_start_random=115
    ):
        """
        Apply masking strategies to the input tensor while tracking NaN locations:
        1. Mask columns [5:115] for a random subset of rows.
        2. Mask columns [0:5] and [115:] randomly per element.

        Args:
            X (Tensor): Input data tensor.
            col_start_fixed (int): Start index of the fixed subsection of columns to mask.
            col_end_fixed (int): End index (exclusive) of the fixed subsection to mask.
            col_start_random (int): Start index for columns to apply random masking.

        Returns:
            X_masked (Tensor): Tensor with masking applied.
            mask (Tensor): Boolean mask indicating where the mask was applied.
            nan_mask (Tensor): Boolean mask indicating original NaN locations.
        """
        X_masked = X.clone().detach().to(self.device)

        # get NaN locations
        nan_mask = ~torch.isnan(X_masked)
        X_masked[~nan_mask] = -9999

        # row-wise masking for cols [5:115] - XP coeffs
        num_rows_to_mask = int(self.xp_masking_ratio * X.shape[0])
        row_indices = torch.randperm(X.shape[0])[:num_rows_to_mask].to(self.device)

        mask_fixed = torch.zeros_like(X, dtype=torch.bool).to(self.device)
        mask_fixed[row_indices, col_start_fixed:col_end_fixed] = True

        # Extra rows with XP fully masked (mixture component toward XP-off at inference).
        mf = getattr(self, "mask_mixture_xp_full_frac", 0.0) or 0.0
        if mf > 0.0:
            n_add = int(mf * X.shape[0])
            if n_add > 0:
                add_idx = torch.randperm(X.shape[0], device=self.device)[:n_add]
                mask_fixed[add_idx, col_start_fixed:col_end_fixed] = True

        # random element-wise masking for cols [0:5] and [115:] - phot bands
        mask_random = torch.zeros_like(X, dtype=torch.bool).to(self.device)

        # mask [0:5] - W1, W2, G, BP, RP
        mask_random[:, :col_start_fixed] = (
            torch.rand(X[:, :col_start_fixed].shape, device=self.device)
            < self.m_masking_ratio
        )
        # mask [115:] - all other phot
        mask_random[:, col_start_random:] = (
            torch.rand(X[:, col_start_random:].shape, device=self.device)
            < self.m_masking_ratio
        )

        # apply masks
        X_masked[mask_fixed | mask_random] = -9999

        # combined mask
        combined_mask = mask_fixed | mask_random

        return X_masked, combined_mask, nan_mask

    def _load_data(self, key):
        """Load and validate data with proper error handling"""
        try:
            if key not in self.datafile:
                raise KeyError(f"Key '{key}' not found in datafile")

            data = self.datafile[key][:]
            if len(data) == 0:
                raise ValueError(f"Dataset '{key}' is empty")

            # Validate required columns exist
            missing_features = [
                col for col in self.feature_cols if col not in data.dtype.names
            ]
            missing_errors = [
                col for col in self.error_cols if col not in data.dtype.names
            ]

            if missing_features:
                raise ValueError(
                    f"Missing feature columns in '{key}': {missing_features}"
                )
            if missing_errors:
                raise ValueError(f"Missing error columns in '{key}': {missing_errors}")

            X = np.column_stack([data[col] for col in self.feature_cols])
            eX = np.column_stack([data[col] for col in self.error_cols])

            # Validate data shapes
            if X.shape[0] != eX.shape[0]:
                raise ValueError(
                    f"Feature and error arrays have mismatched lengths: {X.shape[0]} vs {eX.shape[0]}"
                )

            # Handle missing error values more robustly
            col_maxes = np.nanmax(eX, axis=0)
            # Replace inf values with column max
            eX = np.where(np.isinf(eX), col_maxes[None, :], eX)
            # Replace NaN with column max
            nan_mask = np.isnan(eX)
            eX[nan_mask] = np.take(col_maxes, np.where(nan_mask)[1])

            # Apply scaling with validation
            X = self.featurescaler.transform(X)
            eX = eX / self.scale_factors

            # Final validation
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print(f"Warning: Invalid values in features for key '{key}'")
            if np.any(np.isnan(eX)) or np.any(np.isinf(eX)):
                print(f"Warning: Invalid values in errors for key '{key}'")

            return torch.Tensor(X).to(self.device), torch.Tensor(eX).to(self.device)

        except Exception as e:
            raise RuntimeError(f"Error loading data for key '{key}': {e}")

    @staticmethod
    def _clean_column(col, col_data):
        """Convert byte strings to NaN and stack columns"""
        try:
            if col_data.dtype.kind in {
                "S",
                "U",
            }:  # If the column contains byte strings or unicode
                return np.array(
                    [np.nan if v in {b"", ""} else float(v) for v in col_data],
                    dtype=np.float32,
                )
            return col_data.astype(np.float32)  # Convert other numeric types to float32
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error processing column {col}: {e}")

    def init_weights_gelu(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def pretrain_hdf(
        self,
        train_keys,
        num_epochs=10,
        val_keys=None,
        ft_stuff=None,
        test_stuff=None,
        mini_batch=32,
        pretrained=None,
    ):
        """
        Pre-trains the model on the training dataset with optional validation.

        Args:
            train_keys: Training dataset files in the large h5 (features).
            num_epochs: Number of epochs for pretraining.
            val_keys: Optional validation dataset files in the large h5 (features).
            ft_stuff:
            test_stuff:
            mini_batch: Mini-batch size for pretraining.
        """

        # Separate decay/no_decay for L2 (weight decay)
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        if self.opt == "adam":
            optimizer = optim.Adam(
                [
                    {"params": decay, "weight_decay": self.wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr,
            )
        elif self.opt == "adamw":
            optimizer = optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr,
            )
        elif self.opt == "sgd":
            optimizer = optim.SGD(
                [
                    {"params": decay, "weight_decay": self.wd},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.lr,
                momentum=0.9,
            )
        # Use cosine annealing with warm restarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.scheduler_cosine_t0,
            T_mult=self.scheduler_cosine_t_mult,
            eta_min=self.lr * self.scheduler_eta_min_factor,
        )

        # Configure logging with proper file handling
        os.makedirs(
            os.path.dirname(self.pt_log_file)
            if os.path.dirname(self.pt_log_file)
            else ".",
            exist_ok=True,
        )
        _pt_sd = os.path.dirname(self.pt_save_str)
        if _pt_sd:
            os.makedirs(_pt_sd, exist_ok=True)
        logging.basicConfig(
            filename=self.pt_log_file,
            level=logging.INFO,
            format="%(asctime)s - Sub-Epoch: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode="a",
        )

        running_pt_loss = []
        running_pt_validation_loss = []

        epoch_loss = 0.0
        loss_div = 0.0
        pretrained_epoch = 0

        if pretrained is not None:
            checkpoint = torch_load_trusted(pretrained)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            epoch_loss = checkpoint["epoch_loss"]
            loss_div = checkpoint["loss_div"]
            pretrained_epoch = checkpoint["epoch"]
            print("Picking up pre-training from epoch", pretrained_epoch)

        for epoch in range(num_epochs):
            epoch += pretrained_epoch

            random.shuffle(train_keys)

            n_files = len(train_keys)
            pbar = tqdm.tqdm(
                enumerate(train_keys), total=n_files, desc="Iterating Training Files"
            )
            self.model.train()

            for subkeynum, key in pbar:
                try:
                    X_train, eX_train = self._load_data(key)

                    # Memory-efficient data loading
                    # if X_train.shape[0] > 100000:  # For large datasets, use pin_memory
                    #     train_loader = DataLoader(TensorDataset(X_train, eX_train),
                    #                              batch_size=mini_batch,
                    #                              shuffle=True,
                    #                              pin_memory=True,
                    #                              num_workers=2)
                    # else:
                    #     train_loader = DataLoader(TensorDataset(X_train, eX_train),
                    #                              batch_size=mini_batch,
                    #                              shuffle=True)
                    train_loader = DataLoader(
                        TensorDataset(X_train, eX_train),
                        batch_size=mini_batch,
                        shuffle=True,
                    )

                    for X_batch, eX_batch in train_loader:
                        # Apply data augmentation if enabled (add Gaussian noise scaled by errors)
                        if self.pert_features:
                            X_batch = X_batch + self._pert_noise(X_batch, eX_batch)

                        # Apply masking to training data batch
                        X_masked, mask, nanmask = self._apply_mask(X_batch)

                        # Forward pass (classification output is ignored for pretraining)
                        X_reconstructed, z = self.model(X_masked)

                        # Compute the reconstruction loss
                        # Combine masks: reconstruct only positions that were (1) originally valid AND (2) artificially masked
                        reconstruction_mask = (
                            mask[:, : -self.diff] & nanmask[:, : -self.diff]
                        )
                        l1_norm = z.abs().sum()
                        reconstruction_w = 1.0 / (eX_batch[:, : -self.diff] ** 2 + 1e-8)
                        loss = (
                            self.loss_fn(
                                X_batch[:, : -self.diff],
                                X_reconstructed,
                                reconstruction_mask,
                                reconstruction_w,
                            )
                            + self.lasso * l1_norm
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        # Clip gradients to prevent exploding gradients in deep networks
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        optimizer.step()

                        # print(loss)
                        epoch_loss += loss.item()

                    loss_div += len(train_loader)

                    # Clear GPU cache periodically
                    if torch.cuda.is_available() and subkeynum % 10 == 0:
                        torch.cuda.empty_cache()

                    logging.info(f"{subkeynum + 1}, Loss: {epoch_loss / loss_div}")
                except Exception as e:
                    print(f"Error in training loop for key {key}: {e}")
                    continue

            scheduler.step()

            print(
                f"Pre-training Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / loss_div}"
            )
            running_pt_loss.append(epoch_loss / loss_div)

            # Validation step (if provided)
            if val_keys is not None:
                validation_loss = self.validate(val_keys, self.loss_fn, mini_batch)
                logging.info(f"{epoch + 1}, Validation Loss: {validation_loss}")
                running_pt_validation_loss.append(validation_loss)

            # torch.save(self.model.state_dict(), self.pt_save_str) # old saving method
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch_loss": epoch_loss,
                    "loss_div": loss_div,
                },
                self.pt_save_str,
            )

            if self.checkpoint_interval is not None:
                if (epoch + 1) % self.checkpoint_interval == 0:
                    # torch.save(self.model.state_dict(), self.pt_save_str.split('.')[0]+'_checkpoint_'+str(epoch+1)+'.pth')
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "epoch_loss": epoch_loss,
                            "loss_div": loss_div,
                        },
                        self.pt_save_str.split(".")[0]
                        + "_checkpoint_"
                        + str(epoch + 1)
                        + ".pth",
                    )

        if ft_stuff is not None:
            self.fit(
                ft_stuff[0],
                ft_stuff[1],
                ft_stuff[2],
                e_y_train=ft_stuff[3],
                X_val=ft_stuff[4],
                eX_val=ft_stuff[5],
                y_val=ft_stuff[6],
                e_y_val=ft_stuff[7],
                num_epochs=ft_stuff[8],
                mini_batch=ft_stuff[9],
                linearprobe=ft_stuff[10],
                maskft=ft_stuff[11],
                multitask=ft_stuff[12],
                rncloss=ft_stuff[13],
                last=True,
                test_stuff=test_stuff,
            )

    def validate(self, val_keys, criterion, mini_batch=32):
        """
        Validates the model on a validation dataset during pretraining.

        Args:
            X_val: Validation dataset (features).
            criterion: Loss function used for validation (MSE).
            mini_batch: Mini-batch size for validation.

        """
        self.model.eval()
        with torch.no_grad():
            n_keys = len(val_keys)
            pbar = tqdm.tqdm(
                val_keys, total=n_keys, desc="Iterating Over Validation Keys"
            )
            loss_div = 0
            val_loss = 0
            for key in pbar:
                X_val, eX_val = self._load_data(key)

                # Create DataLoader for mini-batching validation data
                val_loader = DataLoader(
                    TensorDataset(X_val, eX_val), batch_size=mini_batch, shuffle=False
                )

                for X_batch, eX_batch in val_loader:
                    # Apply masking to validation data
                    X_masked, mask, nanmask = self._apply_mask(X_batch)

                    # Forward pass
                    X_reconstructed, _ = self.model(X_masked)

                    # Compute validation loss
                    # Combine masks: reconstruct only positions that were (1) originally valid AND (2) artificially masked
                    reconstruction_mask = (
                        mask[:, : -self.diff] & nanmask[:, : -self.diff]
                    )
                    batch_loss = self.loss_fn(
                        X_batch[:, : -self.diff],
                        X_reconstructed,
                        reconstruction_mask,
                        eX_batch[:, : -self.diff],
                    )

                    val_loss += batch_loss.item()
                loss_div += len(val_loader)

            print(f"Validation Loss: {val_loss / loss_div}")
            return val_loss / loss_div

    def fit(
        self,
        X_train,
        eX_train,
        y_train,
        e_y_train=None,
        X_val=None,
        eX_val=None,
        y_val=None,
        e_y_val=None,
        num_epochs=10,
        mini_batch=32,
        linearprobe=False,
        maskft=False,
        multitask=False,
        rncloss=False,
        last=False,
        ftlr=1e-3,
        ftopt="adam",
        ftact="relu",
        ftl2=0.0,
        ftlf="mse",
        ftdim="1layer512",
        ftlabeldim=5,
        test_stuff=None,
        pt_epoch=0,
        pert_features=False,
        pert_labels=False,
        feature_seed=42,
        ensemblepath=None,
        ft_lambda_pred=0.8,
        ft_lambda_rec=0.2,
        ft_quantile_label_weights: Optional[list] = None,
        ft_use_sigma_quantile_weights: bool = False,
        ft_sigma_weight_floor: float = 1e-6,
        ft_sigma_weight_max: float = 1e6,
        ft_sigma_weight_normalize_batch: bool = True,
        ft_encoder_lr: Optional[float] = None,
        ft_scheduler_encoder_decay: float = 0.95,
        ft_scheduler_head_decay: float = 0.5,
        ft_scheduler_head_step_epochs: int = 10,
        parallax_mle_weight: float = 0.0,
        parallax_use_masked_pred: bool = False,
        parallax_label_idx: Optional[int] = None,
        parallax_sigma_floor: float = 0.0,
        parallax_sigma_scale: float = 1.0,
        consistency_params: Optional[dict] = None,
    ):
        X_train = torch.Tensor(X_train).to(self.device)
        eX_train = torch.Tensor(eX_train).to(self.device)
        y_train = torch.Tensor(y_train).to(self.device)

        # Create DataLoader for mini-batching
        e_y_train = torch.Tensor(e_y_train).to(self.device)
        rdataset = TensorDataset(X_train, eX_train, y_train, e_y_train)
        train_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=True)

        if ftact == "relu":
            ftactivationfunc = nn.ReLU()
        elif ftact == "elu":
            ftactivationfunc = nn.ELU()
        elif ftact == "gelu":
            ftactivationfunc = nn.GELU()

        if linearprobe:
            if ftlf == "quantile":
                raise ValueError(
                    "linearprobe requires finetuning lf 'mse' or 'mae', not 'quantile'"
                )
            if ftlf in ("gnll", "wgnll", "wmse"):
                raise ValueError(f"linearprobe does not support loss type {ftlf!r}")
            if multitask:
                raise ValueError(
                    "linearprobe with multitask is unsupported (full autoencoder is frozen)"
                )
            if rncloss:
                raise ValueError("linearprobe with rncloss is unsupported")

        if (ftlf == "wmse") or (ftlf == "wgnll"):
            criterion = WeightedMaskedMSELoss()
        elif ftlf == "mse":
            criterion = MaskedMSELoss()
        elif ftlf == "mae":
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = RnCLoss(temperature=2, label_diff="l1", feature_sim="l2")

        if (ftlf == "gnll") or (ftlf == "wgnll"):
            criterion2 = MaskedGaussianNLLLoss()

        consistency_params = consistency_params or {}
        m_consistency = None
        c_consistency = None
        if parallax_mle_weight > 0:
            if "m" not in consistency_params or "c" not in consistency_params:
                raise ValueError(
                    "parallax_mle_weight > 0 requires consistency_params with 'm' and 'c'"
                )
            m_consistency = torch.tensor(consistency_params["m"], device=self.device)
            c_consistency = torch.tensor(consistency_params["c"], device=self.device)

        self.lp = None
        if linearprobe:
            self.lp = nn.Linear(self.latent_size, ftlabeldim).to(self.device)
            nn.init.xavier_uniform_(self.lp.weight)
            nn.init.zeros_(self.lp.bias)
            self.ft = None
        else:
            self.ft = PredictionHead(self.latent_size, ftlabeldim, ftactivationfunc).to(
                self.device
            )

        try:
            state_dict = torch_load_trusted(ensemblepath, map_location=self.device)
            self.model.load_state_dict(state_dict["autoencoder_state_dict"])
            if not linearprobe:
                self.ft.load_state_dict(state_dict["prediction_head_state_dict"])
            print("loaded checkpoint")
        except Exception:
            if not linearprobe:
                self.ft.apply(self.init_weights_gelu)
            print("restarting fine-tuning")

        if ft_quantile_label_weights is not None:
            if len(ft_quantile_label_weights) != ftlabeldim:
                raise ValueError(
                    f"ft_quantile_label_weights length {len(ft_quantile_label_weights)} "
                    f"does not match ftlabeldim={ftlabeldim}"
                )
            q_weight_t = torch.tensor(
                ft_quantile_label_weights, dtype=torch.float32, device=self.device
            )
        else:
            q_weight_t = None

        enc_lr = float(ft_encoder_lr) if ft_encoder_lr is not None else float(self.lr)
        head_step = max(1, int(ft_scheduler_head_step_epochs))
        head_lambda = lambda epoch, h=ft_scheduler_head_decay, s=head_step: (
            h ** (epoch // s)
        )
        encoder_lambda = lambda epoch, b=ft_scheduler_encoder_decay: b**epoch

        if linearprobe:
            for p in self.model.parameters():
                p.requires_grad = False
            if ftopt == "adam":
                optimizer = optim.Adam(self.lp.parameters(), lr=ftlr, weight_decay=ftl2)
            elif ftopt == "sgd":
                optimizer = optim.SGD(
                    self.lp.parameters(), lr=ftlr, momentum=0.9, weight_decay=ftl2
                )
            elif ftopt == "adamw":
                optimizer = optim.AdamW(
                    self.lp.parameters(), lr=ftlr, weight_decay=ftl2
                )
            else:
                raise ValueError(f"Unknown ftopt {ftopt!r}")
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=head_lambda)
        else:
            if ftopt == "adam":
                optimizer = optim.Adam(
                    [
                        {"params": self.model.parameters(), "lr": enc_lr},
                        {
                            "params": self.ft.parameters(),
                            "lr": ftlr,
                            "weight_decay": ftl2,
                        },
                    ]
                )
            elif ftopt == "sgd":
                optimizer = optim.SGD(
                    [
                        {"params": self.model.parameters(), "lr": enc_lr},
                        {
                            "params": self.ft.parameters(),
                            "lr": ftlr,
                            "momentum": 0.9,
                            "weight_decay": ftl2,
                        },
                    ]
                )
            elif ftopt == "adamw":
                optimizer = optim.AdamW(
                    [
                        {"params": self.model.parameters(), "lr": enc_lr},
                        {
                            "params": self.ft.parameters(),
                            "lr": ftlr,
                            "weight_decay": ftl2,
                        },
                    ]
                )
            else:
                raise ValueError(f"Unknown ftopt {ftopt!r}")
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[encoder_lambda, head_lambda]
            )

        running_ft_loss = []
        running_ft_validation_loss = []

        # Configure logging with proper file handling
        os.makedirs(
            os.path.dirname(self.ft_log_file)
            if os.path.dirname(self.ft_log_file)
            else ".",
            exist_ok=True,
        )
        _ft_sd = os.path.dirname(self.ft_save_str)
        if _ft_sd:
            os.makedirs(_ft_sd, exist_ok=True)
        logging.basicConfig(
            filename=self.ft_log_file,
            level=logging.INFO,
            format="%(asctime)s - Sub-Epoch: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode="a",
            force=True,
        )

        if pert_features or pert_labels:
            random.seed(feature_seed)
            torch.manual_seed(feature_seed)

        for epoch in range(num_epochs):
            if linearprobe:
                self.model.eval()
                self.lp.train()
            else:
                self.model.train()
                self.ft.train()
            epoch_loss = 0

            for batch in train_loader:
                X_batch = batch[0]
                eX_batch = batch[1]
                y_batch = batch[2]

                # Apply masking / feature noise (torch — not random.gauss, which only accepts scalars)
                if maskft and pert_features:
                    X_masked, mask, nanmask = self._apply_mask(
                        X_batch + self._pert_noise(X_batch, eX_batch)
                    )
                elif pert_features and not maskft:
                    X_masked = X_batch + self._pert_noise(X_batch, eX_batch)
                    mask = torch.zeros_like(
                        X_batch, dtype=torch.bool, device=X_batch.device
                    )
                    nanmask = ~torch.isnan(X_batch)
                elif maskft and not pert_features:
                    X_masked, mask, nanmask = self._apply_mask(X_batch)
                else:
                    X_masked = X_batch.clone()
                    mask = torch.zeros_like(
                        X_batch, dtype=torch.bool, device=X_batch.device
                    )
                    nanmask = ~torch.isnan(X_batch)

                if pert_labels:
                    y_noise = (
                        torch.randn_like(y_batch, device=y_batch.device) * batch[3]
                    )
                    y_batch = y_batch + y_noise

                if linearprobe:
                    encoded = self.model.encoder(X_masked)
                    y_raw = self.lp(encoded)
                else:
                    encoded = self.model.encoder(X_masked)
                    y_raw = self.ft(encoded)

                if parallax_use_masked_pred and self.parallax_feature_idx is not None:
                    if parallax_label_idx is None:
                        parallax_label_idx = y_batch.shape[1] - 1
                    # Mask parallax specifically for a second pass to get "photometric-only" prediction
                    parallax_masked = X_masked.clone()
                    parallax_masked[:, self.parallax_feature_idx] = -9999
                    # Also set mask indicator if available
                    indicator_idx = self.parallax_feature_idx + len(self.feature_cols)
                    if indicator_idx < parallax_masked.shape[1]:
                        parallax_masked[:, indicator_idx] = 1.0

                    if linearprobe:
                        encoded_masked = self.model.encoder(parallax_masked)
                        y_raw_masked = self.lp(encoded_masked)
                    else:
                        encoded_masked = self.model.encoder(parallax_masked)
                        y_raw_masked = self.ft(encoded_masked)

                    # Replace parallax prediction with the one from masked input
                    if y_raw.dim() == 3:
                        y_raw[:, parallax_label_idx, :] = y_raw_masked[
                            :, parallax_label_idx, :
                        ]
                    else:
                        y_raw[:, parallax_label_idx] = y_raw_masked[
                            :, parallax_label_idx
                        ]

                if ftlf == "quantile":
                    y_head = y_raw
                    y_pred_err = None
                else:
                    y_head, y_pred_err = _reduce_finetune_prediction(
                        y_raw, ftlf, linearprobe
                    )

                # Compute loss
                if (ftlf == "wmse") or (ftlf == "wgnll"):
                    loss = criterion(y_batch, y_head, 1 / (batch[3] + 1e-5) ** 2)
                elif (ftlf == "mse") or (ftlf == "mae"):
                    loss = criterion(y_batch, y_head)
                elif ftlf == "quantile":
                    quantiles = torch.tensor([0.16, 0.5, 0.84], device=self.device)
                    sw = None
                    if ft_use_sigma_quantile_weights:
                        sw = _sigma_pinball_weights(
                            batch[3],
                            y_batch,
                            ft_sigma_weight_floor,
                            ft_sigma_weight_max,
                            ft_sigma_weight_normalize_batch,
                        )
                    loss = quantile_loss(
                        y_head, y_batch, quantiles, q_weight_t, sample_weight=sw
                    )
                else:
                    loss = 0

                if (
                    parallax_mle_weight > 0
                    and self.parallax_feature_idx is not None
                    and m_consistency is not None
                ):
                    if parallax_label_idx is None:
                        parallax_label_idx = y_batch.shape[1] - 1

                    # Gaia parallax (scaled)
                    pi_gaia = (
                        m_consistency * X_batch[:, self.parallax_feature_idx]
                        + c_consistency
                    )
                    sigma_gaia = (
                        m_consistency
                        * eX_batch[:, self.parallax_feature_idx]
                        * parallax_sigma_scale
                    )

                    # Model prediction (mu and sigma)
                    if y_raw.dim() == 3:  # Quantile head
                        mu_phot = y_head[:, parallax_label_idx, 1]
                        sigma_phot = 0.5 * (
                            y_head[:, parallax_label_idx, 2]
                            - y_head[:, parallax_label_idx, 0]
                        )
                    else:
                        mu_phot = y_head[:, parallax_label_idx]
                        sigma_phot = None

                    var = sigma_gaia**2
                    if sigma_phot is not None:
                        var = var + sigma_phot**2
                    if parallax_sigma_floor > 0:
                        var = var + (parallax_sigma_floor**2)

                    mle_mask = (
                        (~torch.isnan(mu_phot))
                        & (~torch.isnan(pi_gaia))
                        & (~torch.isnan(var))
                    )
                    if mle_mask.any():
                        mle_loss = (((mu_phot - pi_gaia) ** 2) / (var + 1e-8))[
                            mle_mask
                        ].mean()
                        loss = loss + parallax_mle_weight * mle_loss

                if multitask:
                    X_reconstructed, _ = self.model(X_masked)
                    # Combine masks: reconstruct only positions that were (1) originally valid AND (2) artificially masked
                    reconstruction_mask = (
                        mask[:, : -self.diff] & nanmask[:, : -self.diff]
                    )
                    reconstruction_w = 1.0 / (eX_batch[:, : -self.diff] ** 2 + 1e-8)
                    rec = self.loss_fn(
                        X_batch[:, : -self.diff],
                        X_reconstructed,
                        reconstruction_mask,
                        reconstruction_w,
                    )
                    loss = ft_lambda_pred * loss + ft_lambda_rec * rec

                if rncloss:
                    if pert_features:
                        X_masked_2_in = X_batch + self._pert_noise(X_batch, eX_batch)
                    else:
                        X_masked_2_in = X_batch.clone()
                    if maskft:
                        X_masked_2, _, _ = self._apply_mask(X_masked_2_in)
                    else:
                        X_masked_2 = X_masked_2_in
                    encoded_2 = self.model.encoder(X_masked_2)
                    features = torch.stack(
                        (encoded, encoded_2), dim=1
                    )  # [bs, 2, latent]
                    try:
                        loss += rnc(features, y_batch)
                    except RuntimeError as e:
                        print(e)
                        print(torch.cuda.memory_summary())

                if (ftlf == "gnll") or (ftlf == "wgnll"):
                    loss += criterion2(
                        y_head,
                        y_batch,
                        torch.ones_like(y_pred_err),
                        torch.ones_like(batch[3]),
                    )

                optimizer.zero_grad()
                loss.backward()
                if linearprobe:
                    probe_params = list(self.lp.parameters())
                else:
                    probe_params = list(self.model.parameters()) + list(
                        self.ft.parameters()
                    )
                torch.nn.utils.clip_grad_norm_(probe_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()

            print(
                f"Training Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}"
            )
            running_ft_loss.append(epoch_loss / len(train_loader))
            logging.info(f"Training Loss: {epoch_loss / len(train_loader)}")

            if X_val is not None and y_val is not None:
                validation_loss = self.validate_fit(
                    X_val,
                    eX_val,
                    y_val,
                    e_y_val=e_y_val,
                    mini_batch=mini_batch,
                    linearprobe=linearprobe,
                    maskft=maskft,
                    multitask=multitask,
                    ftlf=ftlf,
                    rncloss=rncloss,
                    ftlabeldim=ftlabeldim,
                    ft_lambda_pred=ft_lambda_pred,
                    ft_lambda_rec=ft_lambda_rec,
                    ft_quantile_label_weights=ft_quantile_label_weights,
                    ft_use_sigma_quantile_weights=ft_use_sigma_quantile_weights,
                    ft_sigma_weight_floor=ft_sigma_weight_floor,
                    ft_sigma_weight_normalize_batch=ft_sigma_weight_normalize_batch,
                    parallax_mle_weight=parallax_mle_weight,
                    parallax_use_masked_pred=parallax_use_masked_pred,
                    parallax_label_idx=parallax_label_idx,
                    parallax_sigma_floor=parallax_sigma_floor,
                    parallax_sigma_scale=parallax_sigma_scale,
                    consistency_params=consistency_params,
                )
                running_ft_validation_loss.append(validation_loss)

                logging.info(f"Validation Loss: {validation_loss}")

            head_sd = self.lp.state_dict() if linearprobe else self.ft.state_dict()
            torch.save(
                {
                    "autoencoder_state_dict": self.model.state_dict(),
                    "prediction_head_state_dict": head_sd,
                    "linear_probe": bool(linearprobe),
                    "featurescaler": self.featurescaler,
                    "label_scalers": getattr(self, "label_scalers", None),
                },
                self.ft_save_str,
            )

            if self.checkpoint_interval is not None:
                if (epoch + 1) % self.checkpoint_interval == 0:
                    torch.save(
                        {
                            "autoencoder_state_dict": self.model.state_dict(),
                            "prediction_head_state_dict": head_sd,
                            "linear_probe": bool(linearprobe),
                            "featurescaler": self.featurescaler,
                            "label_scalers": getattr(self, "label_scalers", None),
                        },
                        self.ft_save_str.split(".")[0]
                        + "_checkpoint_"
                        + str(self.checkpoint_interval)
                        + ".pth",
                    )

    def validate_fit(
        self,
        X_val,
        eX_val,
        y_val,
        e_y_val=None,
        mini_batch=32,
        linearprobe=False,
        maskft=False,
        multitask=False,
        ftlf="mse",
        rncloss=False,
        ftlabeldim=5,
        ft_lambda_pred=0.8,
        ft_lambda_rec=0.2,
        ft_quantile_label_weights: Optional[list] = None,
        ft_use_sigma_quantile_weights: bool = False,
        ft_sigma_weight_floor: float = 1e-6,
        ft_sigma_weight_max: float = 1e6,
        ft_sigma_weight_normalize_batch: bool = True,
        parallax_mle_weight: float = 0.0,
        parallax_use_masked_pred: bool = False,
        parallax_label_idx: Optional[int] = None,
        parallax_sigma_floor: float = 0.0,
        parallax_sigma_scale: float = 1.0,
        consistency_params: Optional[dict] = None,
    ):
        self.model.eval()
        if linearprobe:
            self.lp.eval()
        else:
            self.ft.eval()

        val_loss = 0

        X_val = torch.Tensor(X_val).to(self.device)
        eX_val = torch.Tensor(eX_val).to(self.device)
        y_val = torch.Tensor(y_val).to(self.device)

        # Create DataLoader for mini-batching
        e_y_val = torch.Tensor(e_y_val).to(self.device)
        rdataset = TensorDataset(X_val, eX_val, y_val, e_y_val)
        val_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=True)

        if (ftlf == "wmse") or (ftlf == "wgnll"):
            criterion = WeightedMaskedMSELoss()
        elif ftlf == "mse":
            criterion = MaskedMSELoss()
        elif ftlf == "mae":
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = RnCLoss(temperature=2, label_diff="l1", feature_sim="l2")

        if (ftlf == "gnll") or (ftlf == "wgnll"):
            criterion2 = MaskedGaussianNLLLoss()

        consistency_params = consistency_params or {}
        m_consistency = None
        c_consistency = None
        if parallax_mle_weight > 0:
            if "m" not in consistency_params or "c" not in consistency_params:
                raise ValueError(
                    "parallax_mle_weight > 0 requires consistency_params with 'm' and 'c'"
                )
            m_consistency = torch.tensor(consistency_params["m"], device=self.device)
            c_consistency = torch.tensor(consistency_params["c"], device=self.device)

        if ft_quantile_label_weights is not None:
            if len(ft_quantile_label_weights) != ftlabeldim:
                raise ValueError(
                    f"ft_quantile_label_weights length {len(ft_quantile_label_weights)} "
                    f"does not match ftlabeldim={ftlabeldim}"
                )
            q_weight_t = torch.tensor(
                ft_quantile_label_weights, dtype=torch.float32, device=self.device
            )
        else:
            q_weight_t = None

        with torch.no_grad():
            for batch in val_loader:
                X_batch = batch[0]
                eX_batch = batch[1]
                y_batch = batch[2]

                # Apply masking to input features batch (define mask/nanmask whenever multitask may use them)
                if maskft:
                    X_masked, mask, nanmask = self._apply_mask(X_batch)
                else:
                    X_masked = X_batch.clone()
                    mask = torch.zeros_like(
                        X_batch, dtype=torch.bool, device=X_batch.device
                    )
                    nanmask = ~torch.isnan(X_batch)

                if linearprobe:
                    encoded = self.model.encoder(X_masked)
                    y_raw = self.lp(encoded)
                else:
                    encoded = self.model.encoder(X_masked)
                    y_raw = self.ft(encoded)

                if parallax_use_masked_pred and self.parallax_feature_idx is not None:
                    if parallax_label_idx is None:
                        parallax_label_idx = y_batch.shape[1] - 1
                    # Mask parallax specifically for a second pass
                    parallax_masked = X_masked.clone()
                    parallax_masked[:, self.parallax_feature_idx] = -9999
                    indicator_idx = self.parallax_feature_idx + len(self.feature_cols)
                    if indicator_idx < parallax_masked.shape[1]:
                        parallax_masked[:, indicator_idx] = 1.0

                    if linearprobe:
                        encoded_masked = self.model.encoder(parallax_masked)
                        y_raw_masked = self.lp(encoded_masked)
                    else:
                        encoded_masked = self.model.encoder(parallax_masked)
                        y_raw_masked = self.ft(encoded_masked)

                    if y_raw.dim() == 3:
                        y_raw[:, parallax_label_idx, :] = y_raw_masked[
                            :, parallax_label_idx, :
                        ]
                    else:
                        y_raw[:, parallax_label_idx] = y_raw_masked[
                            :, parallax_label_idx
                        ]

                if ftlf == "quantile":
                    y_head = y_raw
                    y_pred_err = None
                else:
                    y_head, y_pred_err = _reduce_finetune_prediction(
                        y_raw, ftlf, linearprobe
                    )

                if (ftlf == "wmse") or (ftlf == "wgnll"):
                    loss = criterion(y_batch, y_head, 1 / (batch[3] + 1e-5) ** 2)
                elif (ftlf == "mse") or (ftlf == "mae"):
                    loss = criterion(y_batch, y_head)
                elif ftlf == "quantile":
                    quantiles = torch.tensor([0.16, 0.5, 0.84], device=self.device)
                    sw = None
                    if ft_use_sigma_quantile_weights:
                        sw = _sigma_pinball_weights(
                            batch[3],
                            y_batch,
                            ft_sigma_weight_floor,
                            ft_sigma_weight_max,
                            ft_sigma_weight_normalize_batch,
                        )
                    loss = quantile_loss(
                        y_head, y_batch, quantiles, q_weight_t, sample_weight=sw
                    )
                else:
                    loss = 0

                if (
                    parallax_mle_weight > 0
                    and self.parallax_feature_idx is not None
                    and m_consistency is not None
                ):
                    if parallax_label_idx is None:
                        parallax_label_idx = y_batch.shape[1] - 1
                    pi_gaia = (
                        m_consistency * X_batch[:, self.parallax_feature_idx]
                        + c_consistency
                    )
                    sigma_gaia = (
                        m_consistency
                        * eX_batch[:, self.parallax_feature_idx]
                        * parallax_sigma_scale
                    )
                    if y_raw.dim() == 3:
                        mu_phot = y_head[:, parallax_label_idx, 1]
                        sigma_phot = 0.5 * (
                            y_head[:, parallax_label_idx, 2]
                            - y_head[:, parallax_label_idx, 0]
                        )
                    else:
                        mu_phot = y_head[:, parallax_label_idx]
                        sigma_phot = None
                    var = sigma_gaia**2
                    if sigma_phot is not None:
                        var = var + sigma_phot**2
                    if parallax_sigma_floor > 0:
                        var = var + (parallax_sigma_floor**2)
                    mle_mask = (
                        (~torch.isnan(mu_phot))
                        & (~torch.isnan(pi_gaia))
                        & (~torch.isnan(var))
                    )
                    if mle_mask.any():
                        mle_loss = (((mu_phot - pi_gaia) ** 2) / (var + 1e-8))[
                            mle_mask
                        ].mean()
                        loss = loss + parallax_mle_weight * mle_loss

                if multitask:
                    X_reconstructed, _ = self.model(X_masked)
                    # Combine masks: reconstruct only positions that were (1) originally valid AND (2) artificially masked
                    reconstruction_mask = (
                        mask[:, : -self.diff] & nanmask[:, : -self.diff]
                    )
                    reconstruction_w = 1.0 / (eX_batch[:, : -self.diff] ** 2 + 1e-8)
                    rec = self.loss_fn(
                        X_batch[:, : -self.diff],
                        X_reconstructed,
                        reconstruction_mask,
                        reconstruction_w,
                    )
                    loss = ft_lambda_pred * loss + ft_lambda_rec * rec

                if rncloss:
                    if maskft:
                        X_masked_2, _, _ = self._apply_mask(X_batch)
                    else:
                        X_masked_2 = X_batch.clone()
                    encoded_2 = self.model.encoder(X_masked_2)
                    features = torch.stack(
                        (encoded, encoded_2), dim=1
                    )  # [bs, 2, latent]
                    try:
                        val_loss += rnc(features, y_batch).item()
                    except RuntimeError as e:
                        print(e)
                    # We continue after val_loss evaluation

                if (ftlf == "gnll") or (ftlf == "wgnll"):
                    if y_pred_err is None:
                        raise RuntimeError(
                            "Gaussian NLL path requires a (mean, logvar) tuple head; not supported for quantile head"
                        )
                    loss += criterion2(
                        y_head,
                        y_batch,
                        torch.ones_like(y_pred_err),
                        torch.ones_like(batch[3]),
                    )

                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")
        return val_loss / len(val_loader)


def make_model(
    input_dim, layer_dims, output_dim, activ, rtdl_embed_dim, norm, decoder_dims=None
):
    """
    Helper function to make the MSA in the same file as the wrapper

    input_dim :: int
        length of the input features including positional information not reconstructed.
    layer_dims :: list
        Residual block dimensions. The list is discretized, being the specific widths for each individual layer.
    output_dim :: int
        Length of the output features, those features that are reconstructed.
    activ :: string
        String of the possible activation functions. Must be one of ('elu', 'relu', or 'gelu').
    rtdl_embed_dim :: int
        Embedding dimension the input data is blown up to.
    norm :: string
        String of the possible normalization options. Must be one of ('layer', or 'batch')
    decoder_dims :: list, optional
        Decoder dimensions. If None, uses symmetric (mirrored) encoder dimensions.
        For asymmetric decoder, specify custom dimensions (e.g., [256, 512, 1024])
    """

    model = TabResnet(
        continuous_cols=input_dim,
        blocks_dims=layer_dims,
        output_cols=output_dim,
        activ=activ,
        d_embedding=rtdl_embed_dim,
        norm=norm,
        decoder_dims=decoder_dims,
    )
    return model
