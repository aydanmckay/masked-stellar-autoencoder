# loading the packages
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
# import h5py

import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import tqdm
import logging
import wandb
import yaml

from sklearn.preprocessing import StandardScaler, RobustScaler

from .blocks import TabResnet

class MaskedGaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
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
        diff_squared = (pred_mean-target)**2

        # Compute Gaussian NLL
        nll = 0.5 * (torch.log(err) + (diff_squared/err)) + 0.5 * math.log(2 * math.pi)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll

class WeightedMaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
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

        if self.reduction == 'mean':
            return masked_error.sum() / (masked_weights.sum() + self.eps)
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:
            return masked_error

class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, target, input):
        # Create a mask for non-NaN targets
        mask = ~torch.isnan(target)
        
        # Compute squared error only where target is not NaN
        
        # old way
        # squared_error = (input - target)**2
        # masked_error = squared_error[mask]

        # new way
        masked_input = input[mask]
        masked_target = target[mask]
        masked_error = (masked_input - masked_target) ** 2

        if masked_error.numel() == 0:
            return torch.tensor(0.0, device=input.device, requires_grad=True)
        if self.reduction == 'mean':
            return masked_error.mean()
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:
            return masked_error

class LabelDifference(nn.Module):
    '''
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    '''
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)

class FeatureSimilarity(nn.Module):
    '''
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    '''
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)

class RnCLoss(nn.Module):
    '''
    @inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
    }
    '''
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
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
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False, path='checkpoint.pth'):
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
            self.save_checkpoint(model)  # Save the model when the best validation loss is found
        elif validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f"Validation loss improved to {self.best_loss:.6f}, saving model.")
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

    def __init__(self, eps: float = 1e-9, lf='mse'):
        super(EncoderDecoderLoss, self).__init__()
        self.eps = eps
        self.cost = lf

    def forward(self, x_true: Tensor, x_pred: Tensor, mask: Tensor, w: Tensor) -> Tensor:
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
        errors = torch.where(mask.bool(), x_pred - x_true, torch.tensor(0.0, device=x_true.device))
        if self.cost == 'mse':
            reconstruction_errors = errors ** 2
        elif self.cost == 'mae':
            reconstruction_errors = abs(errors)
        elif self.cost == 'wmse':
            reconstruction_errors = w * (errors ** 2)
        elif self.cost == 'wmae':
            reconstruction_errors = w * abs(errors)

        # features_loss = torch.matmul(reconstruction_errors, 1 / x_true_stds)
        features_loss = reconstruction_errors
        nb_reconstructed_variables = torch.sum(mask, dim=0)
        features_loss_norm = features_loss / (nb_reconstructed_variables + self.eps)
    
        loss = torch.mean(features_loss_norm)

        return loss

class PredictionHead(nn.Module):
    def __init__(self, latent_size, ft_label_dim, ft_activ):
        super(PredictionHead, self).__init__()

        self.shared = nn.Sequential(nn.Linear(latent_size, 2048),
                                    ft_activ,
                                    nn.Linear(2048, 2048),
                                    ft_activ,
                                    nn.Linear(2048, 1024),
                                    ft_activ,
                                    nn.Linear(1024, 512),
                                    ft_activ,
                                    nn.Linear(512, 256),
                                    ft_activ)
        self.output_y = nn.Linear(256, ft_label_dim)
        self.output_upper = nn.Linear(256, ft_label_dim)
        self.output_lower = nn.Linear(256, ft_label_dim)


    def forward(self, x):
        h = self.shared(x)
        y = self.output_y(h)
        yl = self.output_lower(h)
        yu = self.output_upper(h)
        return torch.stack([yl, y, yu], dim=2)

def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """
    Calculates the quantile loss for a batch of predictions and multiple labels.

    Args:
        preds (torch.Tensor): The model's predictions. Shape: (batch_size, num_labels, num_quantiles)
        target (torch.Tensor): The true values. Shape: (batch_size, num_labels)
        quantiles (torch.Tensor): The quantiles to be calculated. Shape: (num_quantiles,)

    Returns:
        torch.Tensor: The mean loss for the batch. A single scalar value.
    """
    # Expand target to match preds: (batch_size, num_labels, num_quantiles)
    mask = ~torch.isnan(target)
    target = target.unsqueeze(2).expand_as(preds)  # Add quantile dim

    # Expand quantiles to match target shape
    # Shape: (1, 1, num_quantiles)
    quantiles = quantiles.view(1, 1, -1)
    error = target - preds
    loss = torch.max((quantiles - 1) * error, quantiles * error)
    loss = loss[mask]

    return loss.mean()

# creating a training wrapper for the algorithm
class TabResnetWrapper(BaseEstimator):
    def __init__(self,
                 model,
                 datafile,
                 scaler,
                 feature_cols,
                 error_cols,
                 recon_cols,
                 latent_size=256,
                 xp_masking_ratio=0.9,
                 m_masking_ratio=0.9,
                 lr=1e-3,
                 optimizer='adam',
                 wd=0,
                 lasso=0,
                 lf='mse',
                 pt_save_str='pt_model.pth',
                 ft_save_str='ft_model.pth',
                 pt_log_file='pt_loss.log',
                 ft_log_file='ft_loss.log',
                 checkpoint_interval=None,
                 ):
        
        '''
        Changes to the original that can predict ages are the following:
        periodic embeddings
        scaling the coefficients with the RobustScaler
        changing the mask value to -9999
        exponential scheduler instead of stepLR
        different masking ratios

        '''
        self.model = model
        self.datafile = datafile
        self.featurescaler = scaler
        self.scale_factors = self.featurescaler.scale_  # This is the IQR used by RobustScaler for each feature
        self.feature_cols = feature_cols
        self.error_cols = error_cols
        self.recon_cols = recon_cols
        self.diff = len(feature_cols) - len(recon_cols)
        self.xp_masking_ratio = xp_masking_ratio
        self.m_masking_ratio = m_masking_ratio
        self.lr = lr
        self.opt = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def _apply_mask(self, X, col_start_fixed=5, col_end_fixed=115, col_start_random=115):
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
    
        # random element-wise masking for cols [0:5] and [115:] - phot bands
        mask_random = torch.zeros_like(X, dtype=torch.bool).to(self.device)
    
        # mask [0:5] - W1, W2, G, BP, RP
        mask_random[:, :col_start_fixed] = (
            torch.rand(X[:, :col_start_fixed].shape, device=self.device) < self.m_masking_ratio
        )
        # mask [115:] - all other phot
        mask_random[:, col_start_random:] = (
            torch.rand(X[:, col_start_random:].shape, device=self.device) < self.m_masking_ratio
        )
    
        # apply masks
        X_masked[mask_fixed | mask_random] = -9999
    
        # combined mask
        combined_mask = mask_fixed | mask_random
    
        return X_masked, combined_mask, nan_mask

    def _load_data(self, key):

        '''There is a temporary byte fix in here as for some reason smss got saved as 
        bytes and not as the proper format. If not needed then remove code and just
        column_stack.'''

        data = self.datafile[key][:]
        
        X = np.column_stack([data[col] for col in self.feature_cols])
        eX = np.column_stack([data[col] for col in self.error_cols])
        
        # if data needs decoding:
        # X = np.column_stack([self._clean_column(data[col]) for col in self.feature_cols])
        # eX = np.column_stack([self._clean_column(data[col]) for col in self.error_cols])

        # Find column-wise max ignoring NaNs
        col_maxes = np.nanmax(eX, axis=0)
        nan_mask = np.isnan(eX)
        eX[nan_mask] = np.take(col_maxes, np.where(nan_mask)[1])

        X = self.featurescaler.transform(X)
        eX = eX / self.scale_factors
        
        return torch.Tensor(X).to(self.device), torch.Tensor(eX).to(self.device)
    
    def _clean_column(self, col_data):
        '''Convert byte strings to NaN and stack columns'''
        if col_data.dtype.kind in {'S', 'U'}:  # If the column contains byte strings or unicode
            return np.array([np.nan if v in {b'', ''} else float(v) for v in col_data], dtype=np.float32)
        return col_data.astype(np.float32)  # Convert other numeric types to float3

    
    def pretrain_hdf(self, train_keys, num_epochs=10, val_keys=None, ft_stuff=None, test_stuff=None, mini_batch=32):
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
            if 'bias' in name or 'norm' in name:
                no_decay.append(param)
            else:
                decay.append(param)

        if self.opt == 'adam':
            optimizer = optim.Adam([
                {'params': decay, 'weight_decay': self.wd},
                {'params': no_decay, 'weight_decay': 0.0}
            ], lr=self.lr)
        elif self.opt == 'adamw':
            optimizer = optim.AdamW([
                {'params': decay, 'weight_decay': self.wd},
                {'params': no_decay, 'weight_decay': 0.0}
            ], lr=self.lr)
        elif self.opt == 'sgd':
            optimizer = optim.SGD([
                {'params': decay, 'weight_decay': self.wd},
                {'params': no_decay, 'weight_decay': 0.0}
            ], lr=self.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # Configure logging
        logging.basicConfig(filename=self.pt_log_file, 
                            level=logging.INFO, 
                            format="%(asctime)s - Sub-Epoch: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

        running_pt_loss = []
        running_pt_validation_loss = []

        epoch_loss = 0.
        loss_div = 0.

        for epoch in range(num_epochs):

            random.shuffle(train_keys)

            n_files = len(train_keys)
            pbar = tqdm.tqdm(enumerate(train_keys), total=n_files, desc='Iterating Training Files')
            self.model.train()

            for subkeynum,key in pbar:

                X_train, eX_train = self._load_data(key)
                
                train_loader = DataLoader(TensorDataset(X_train, eX_train), batch_size=mini_batch, shuffle=True)

                for X_batch,eX_batch in train_loader:

                    # Apply masking to training data batch
                    X_masked, mask, nanmask = self._apply_mask(X_batch)

                    # Forward pass (classification output is ignored for pretraining)
                    X_reconstructed, z = self.model(X_masked)

                    # Compute the reconstruction loss
                    # not reconstructing some rows as below
                    l1_norm = z.abs().sum()
                    loss = self.loss_fn(X_batch[:,:-self.diff], X_reconstructed, nanmask[:, :-self.diff], eX_batch[:, :-self.diff]) + self.lasso * l1_norm

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # print(loss)
                    epoch_loss += loss.item()
                loss_div += len(train_loader)
                
                logging.info(f"{subkeynum + 1}, Loss: {epoch_loss/loss_div}")

            scheduler.step()

            
            print(f"Pre-training Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / loss_div}")
            running_pt_loss.append(epoch_loss / loss_div)
    
            # Validation step (if provided)
            if val_keys is not None:
                validation_loss = self.validate(val_keys, self.loss_fn, mini_batch)
                logging.info(f"{epoch + 1}, Validation Loss: {validation_loss}")
                running_pt_validation_loss.append(validation_loss)

            torch.save(self.model.state_dict(), self.pt_save_str)

            if self.checkpoint_interval is not None:
                if (epoch+1) % self.checkpoint_interval == 0:
                    torch.save(self.model.state_dict(), self.pt_save_str.split('.')[0]+'_checkpoint_'+str(self.checkpoint_interval)+'.pth')

        if ft_stuff is not None:
            self.fit(ft_stuff[0],
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
            pbar = tqdm.tqdm(val_keys, total=n_keys, desc='Iterating Over Validation Keys')
            loss_div = 0
            val_loss = 0
            for key in pbar:
                X_val, eX_val = self._load_data(key)
    
                # Create DataLoader for mini-batching validation data
                val_loader = DataLoader(TensorDataset(X_val, eX_val), batch_size=mini_batch, shuffle=False)
    
                for X_batch,eX_batch in val_loader:
                    # Apply masking to validation data
                    X_masked, mask, nanmask = self._apply_mask(X_batch)
    
                    # Forward pass
                    X_reconstructed, _ = self.model(X_masked)
    
                    # Compute validation loss
                    # not counting the parallax and ebv
                    batch_loss = self.loss_fn(X_batch[:, :self.diff], X_reconstructed, nanmask[:, :self.diff], eX_batch)
                    
                    val_loss += batch_loss.item()
                loss_div += len(val_loader)
            
            print(f"Validation Loss: {val_loss / loss_div}")
            return val_loss / loss_div

    def fit(self,
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
            ftopt='adam',
            ftact='relu',
            ftl2=0.0,
            ftlf='mse',
            ftdim='1layer512',
            ftlabeldim=5,
            traintype='normal',
            test_stuff=None,
            pt_epoch=0,
            pert_features=False,
            pert_labels=False,
            feature_seed=42,
            ensemblepath=None,
           ):
        
        X_train = torch.Tensor(X_train).to(self.device)
        eX_train = torch.Tensor(eX_train).to(self.device)
        y_train = torch.Tensor(y_train).to(self.device)
        
        # Create DataLoader for mini-batching
        e_y_train = torch.Tensor(e_y_train).to(self.device)
        rdataset = TensorDataset(X_train, eX_train, y_train, e_y_train)
        train_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=True)
        
        if ftact == 'relu':
            ftactivationfunc = nn.ReLU()
        elif ftact == 'elu':
            ftactivationfunc = nn.ELU()
        elif ftact == 'gelu':
            ftactivationfunc = nn.GELU()

        if (ftlf == 'wmse') or (ftlf == 'wgnll'):
            criterion = WeightedMaskedMSELoss()
        elif ftlf == 'mse':
            criterion = MaskedMSELoss()
        elif ftlf == 'mae':
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')

        if (ftlf == 'gnll') or (ftlf == 'wgnll'):
            criterion2 = MaskedGaussianNLLLoss()
            
        self.ft = PredictionHead(self.latent_size,ftlabeldim,ftactivationfunc).to(self.device)

        try:
            state_dict = torch.load(ensemblepath, map_location=self.device)
            
            # assign to models
            self.model.load_state_dict(state_dict['autoencoder_state_dict'])
            self.ft.load_state_dict(state_dict['prediction_head_state_dict'])
            print('loaded checkpoint')
        except:
            self.ft.apply(self.init_weights_gelu)
            print('restarting fine-tuning')

        if ftopt == 'adam':
            optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': 1e-5},
                {'params': self.ft.parameters(), 'lr': ftlr, 'weight_decay': ftl2}
            ])
        elif ftopt == 'sgd':
            optimizer = optim.SGD([
                {'params': self.model.parameters(), 'lr': 1e-5},
                {'params': self.ft.parameters(), 'lr': ftlr, 'momentum': 0.9, 'weight_decay': ftl2}
            ])
        elif ftopt == 'adamw':
            optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': 1e-5},
                {'params': self.ft.parameters(), 'lr': ftlr, 'weight_decay': ftl2}
            ])

        # Define lambda functions for each group's schedule
        encoder_lambda = lambda epoch: 0.95 ** epoch         # Slow decay
        head_lambda = lambda epoch: 0.5 ** (epoch // 10)     # Step decay every 10 epochs

        # Scheduler applied to parameter groups
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[encoder_lambda, head_lambda])

        running_ft_loss = []
        running_ft_validation_loss = []
        
        logging.basicConfig(filename=self.ft_log_file, 
                            level=logging.INFO, 
                            format="%(asctime)s - Sub-Epoch: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            force=True)
                            
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

                # Apply masking to input features batch
                if maskft and pert_features:
                    X_masked, mask, nanmask = self._apply_mask(random.gauss(mu=X_batch, sigma=eX_batch))
                elif pert_features and not maskft:
                    X_masked = random.gauss(mu=X_batch, sigma=eX_batch)
                elif maskft and not pert_features:
                    X_masked, mask, nanmask = self._apply_mask(X_batch)
                else:
                    X_masked = X_batch.clone()

                if pert_labels:
                    y_batch = random.gauss(mu=y_batch, sigma=batch[3])

                if linearprobe:
                    # Forward pass (classification output is used for fitting)
                    encoded = self.model.encoder(X_masked)
                    y_pred = self.lp(encoded)
                else: 
                    encoded = self.model.encoder(X_masked)
                    y_pred = self.ft(encoded)

                if ftlf != 'quantile':
                    y_pred_err = y_pred[1]
                    y_pred = y_pred[0]
                    
                # Compute loss
                if (ftlf == 'wmse') or (ftlf == 'wgnll'):
                    loss = criterion(y_batch, y_pred, 1/(batch[3]+1e-5)**2)
                elif (ftlf == 'mse') or (ftlf == 'mae'):
                    loss = criterion(y_batch, y_pred)  # Assuming class labels are integers
                elif ftlf == 'quantile':
                    quantiles = torch.tensor([0.16, 0.5, 0.84], device=self.device)
                    loss = quantile_loss(y_pred, y_batch, quantiles)
                else:
                    loss = 0

                if multitask:
                    X_reconstructed, _ = self.model(X_masked)
                    loss += self.loss_fn(X_batch[:, :self.diff], X_reconstructed, nanmask[:, :self.diff], eX_batch[:, :self.diff])

                if rncloss:
                    features = torch.stack((y_pred, y_pred.clone()), dim=1)  # [bs, 2, feat_dim]
                    try:
                        loss += rnc(features, y_batch)
                    except RuntimeError as e:
                        print(e)
                        print(torch.cuda.memory_summary())

                if (ftlf == 'gnll') or (ftlf == 'wgnll'):
                    loss += criterion2(y_pred, y_batch, torch.ones_like(y_pred_err), torch.ones_like(batch[3]))
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()
                
                epoch_loss += loss.item()

            scheduler.step()

            print(f"Training Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader)}")
            running_ft_loss.append(epoch_loss / len(train_loader))
            logging.info(f"Training Loss: {epoch_loss / len(train_loader)}")

            if X_val is not None and y_val is not None:
                validation_loss = self.validate_fit(X_val,
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
                )
                running_ft_validation_loss.append(validation_loss)

                logging.info(f"Validation Loss: {validation_loss}")

            torch.save({'autoencoder_state_dict': self.model.state_dict(),
                        'prediction_head_state_dict': self.ft.state_dict()},
                        self.ft_save_str)

            if self.checkpoint_interval is not None:
                if (epoch+1) % self.checkpoint_interval == 0:
                    torch.save({'autoencoder_state_dict': self.model.state_dict(),
                                'prediction_head_state_dict': self.ft.state_dict()},
                                self.ft_save_str.split('.')[0]+'_checkpoint_'+str(self.checkpoint_interval)+'.pth')

    def validate_fit(self, X_val, eX_val, y_val, e_y_val=None, mini_batch=32, linearprobe=False, maskft=False, multitask=False, ftlf='mse', rncloss=False, ftlabeldim=5):
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

        if (ftlf == 'wmse') or (ftlf == 'wgnll'):
            criterion = WeightedMaskedMSELoss()
        elif ftlf == 'mse':
            criterion = MaskedMSELoss()
        elif ftlf == 'mae':
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')

        if (ftlf == 'gnll') or (ftlf == 'wgnll'):
            criterion2 = MaskedGaussianNLLLoss()

        with torch.no_grad():
            for batch in val_loader:

                X_batch = batch[0]
                eX_batch = batch[1]
                y_batch = batch[2]

                # Apply masking to input features batch
                if maskft:
                    X_masked, mask, nanmask = self._apply_mask(X_batch)
                else:
                    X_masked = X_batch.copy()

                if linearprobe:
                    # Forward pass (classification output is used for fitting)
                    encoded  = self.model.encoder(X_masked)
                    y_pred = self.lp(encoded)
                else: 
                    encoded = self.model.encoder(X_masked)
                    y_pred = self.ft(encoded)

                if ftlf != 'quantile':
                    y_pred_err = y_pred[1]
                    y_pred = y_pred[0]
                    
                # Compute loss
                if (ftlf == 'wmse') or (ftlf == 'wgnll'):
                    loss = criterion(y_batch, y_pred, 1/(batch[3]+1e-5)**2)
                elif (ftlf == 'mse') or (ftlf == 'mae'):
                    loss = criterion(y_batch, y_pred)  # Assuming class labels are integers
                elif ftlf == 'quantile':
                    quantiles = torch.tensor([0.16, 0.5, 0.84], device=self.device)
                    loss = quantile_loss(y_pred, y_batch, quantiles)
                else:
                    loss = 0

                if multitask:
                    X_reconstructed, _ = self.model(X_masked)
                    loss += self.loss_fn(X_batch[:, :self.diff], X_reconstructed, nanmask[:, :self.diff], eX_batch[:, :self.diff])

                if rncloss:
                    features = torch.stack((y_pred, y_pred.clone()), dim=1)  # [bs, 2, feat_dim]
                    try:
                        loss += rnc(features, y_batch)
                    except RuntimeError as e:
                        print(e)
                        print(torch.cuda.memory_summary())

                if (ftlf == 'gnll') or (ftlf == 'wgnll'):
                    loss += criterion2(y_pred, y_batch, torch.ones_like(y_pred_err), torch.ones_like(batch[3]))

                val_loss += loss.item()
            
        print(f"Validation Loss: {val_loss / len(val_loader)}")
        return val_loss / len(val_loader)

def make_model(input_dim, layer_dims, output_dim, activ, rtdl_embed_dim, norm):
    '''
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
    '''

    model = TabResnet(
        continuous_cols=input_dim,
        blocks_dims=layer_dims,
        output_cols=output_dim,
        activ=activ,
        d_embedding=rtdl_embed_dim,
        norm=norm,
    )
    return model