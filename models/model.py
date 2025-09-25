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

import blocks as modelfile

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

# creating a training wrapper for the algorithm
class TabResnetWrapper(BaseEstimator):
    def __init__(self, model, datafile, scaler, latent_size=256, num_classes=6, xp_masking_ratio=0.15, m_masking_ratio=0.15, lr=1e-3, optimizer='adam', wd=0, lasso=0, lf='mse'):
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
        self.xp_masking_ratio = xp_masking_ratio
        self.m_masking_ratio = m_masking_ratio
        self.lr = lr
        self.opt = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss_fn = EncoderDecoderLoss(lf=lf)
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.lasso = lasso
        self.wd = wd
        

        self.nonfeatures = ['source_id']
        self.nonfeatures.extend(['bpe_'+str(i) for i in range(1,56)])
        self.nonfeatures.extend(['rpe_'+str(i) for i in range(1,56)])
        self.nonfeatures.extend(['RA','DEC','E_U_SMSS','E_V_SMSS','E_G_SMSS','E_R_SMSS','E_I_SMSS','E_Z_SMSS',
                                'E_U_SDSS','E_G_SDSS','E_R_SDSS','E_I_SDSS','E_Z_SDSS',
                                'E_J','E_H','E_KS','E_G_PS1','E_R_PS1','E_I_PS1','E_Z_PS1','E_Y_PS1','AZERO',
                                'ruwe','e_pmra','e_pmdec','pmradec_corr','g_flux_error',
                                'bp_flux_error','rp_flux_error','e_parallax'])

        self.errorcols = ['W1', 'W2', 'g_flux_error', 'bp_flux_error', 'rp_flux_error'] # 10% taken as errors
        self.errorcols.extend(['bpe_'+str(i) for i in range(1,56)])
        self.errorcols.extend(['rpe_'+str(i) for i in range(1,56)])
        self.errorcols.extend(['E_U_SMSS','E_V_SMSS','E_G_SMSS','E_R_SMSS','E_I_SMSS','E_Z_SMSS',
                                'E_U_SDSS','E_G_SDSS','E_R_SDSS','E_I_SDSS','E_Z_SDSS',
                                'E_G_PS1','E_R_PS1','E_I_PS1','E_Z_PS1','E_Y_PS1','E_J','E_H','E_KS',
                                'PARALLAX','EBV','e_pmra','e_pmdec'])
        self.scales = [0,0,25.8010446445,25.3539555559,25.1039837393]

    def _apply_mask(self, X, col_start_fixed=5, col_end_fixed=115, col_start_random=115):
        """
        Apply two masking strategies to the input tensor while also tracking NaN locations:
        1. Mask a fixed subsection of columns (X[:, 5:115]) for a subset of rows.
        2. Randomly mask columns starting from X[:, 115:] for all rows.
        
        Args:
            X (Tensor): Input data tensor.
            col_start_fixed (int): Starting index of the fixed subsection of columns to mask.
            col_end_fixed (int): Ending index (exclusive) of the fixed subsection to mask.
            col_start_random (int): Starting index for columns to apply random masking.
        
        Returns:
            X_masked (Tensor): Tensor with both masking strategies applied.
            mask (Tensor): Boolean mask indicating where the mask was applied.
            nan_mask (Tensor): Boolean mask indicating where the input originally had NaNs.
        """
        X_masked = X.clone().detach().to(self.device)
    
        # Track NaN locations
        nan_mask = ~torch.isnan(X_masked)

        X_masked[~nan_mask] = -9999
    
        # 1. Apply fixed subsection column masking (X[:, 5:115]) for a portion of rows
        num_rows_to_mask = int(self.xp_masking_ratio * X.shape[0])
        row_indices = torch.randperm(X.shape[0])[:num_rows_to_mask].to(self.device)
    
        # Create a mask for the fixed subsection of columns
        mask_fixed = torch.zeros(X.shape, dtype=torch.bool).to(self.device)
        mask_fixed[row_indices, col_start_fixed:col_end_fixed] = True
    
        # Apply fixed column subsection mask to selected rows
        X_masked[mask_fixed] = -9999  
    
        # 2. Apply random masking for columns starting from col_start_random (X[:, 115:])
        mask_random = torch.rand(X[:, col_start_random:].shape).to(self.device) < self.m_masking_ratio
        X_masked[:, col_start_random:][mask_random] = -9999  
    
        # 3. Combine both masks
        combined_mask = torch.zeros_like(X, dtype=torch.bool).to(self.device)
        combined_mask[:, col_start_random:] = mask_random
        combined_mask[row_indices, col_start_fixed:col_end_fixed] = True
    
        return X_masked, combined_mask, nan_mask

    def _load_data(self, key):

        '''There is a temporary byte fix in here as for some reason smss got saved as 
        bytes and not as the proper format'''

        data = self.datafile[key][:]
        # print(X)
        cols = [col for col in list(data.dtype.names) if col not in self.nonfeatures]
        X = np.column_stack([self._clean_column(data[col]) for col in cols])
        eX = np.column_stack([self._clean_column(data[col]) for col in self.errorcols])

        for it in range(5):
            if self.scales[it] == 0:
                eX[it] = abs(0.1*eX[it])
            else:
                eX[it] = -2.5*np.log10(eX[it]) + self.scales[it]

        # Find column-wise max ignoring NaNs
        col_maxes = np.nanmax(eX, axis=0)
        nan_mask = np.isnan(eX)
        eX[nan_mask] = np.take(col_maxes, np.where(nan_mask)[1])

        X = self.featurescaler.transform(X)
        eX = eX / self.scale_factors
        eX = eX[:, :-4]
        
        return torch.Tensor(X).to(self.device), torch.Tensor(eX).to(self.device)

        # Convert byte strings to NaN and stack columns
    
    def _clean_column(self, col_data):
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
        logging.basicConfig(filename="/arc/home/aydanmckay/bprp_mae/final_model/logfiles/training_pm_sweep_results_loss_0605.log", 
                            level=logging.INFO, 
                            format="%(asctime)s - Sub-Epoch: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

        early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True,
                                       path='/arc/home/aydanmckay/bprp_mae/.model_instances/checkpoints/fullmodelrealmags0605.pth')

        running_pt_loss = []
        running_pt_validation_loss = []

        os.system('mkdir /arc/home/aydanmckay/bprp_mae/andraeplots/rtdlmodels/resmlp_realmag_full_0605')

        epoch_loss = 0.
        loss_div = 0.

        for epoch in range(num_epochs):

            random.shuffle(train_keys)

            n_files = len(train_keys)
            pbar = tqdm.tqdm(enumerate(train_keys), total=n_files, desc='Iterating Training Files')
            self.model.train()

            for subkeynum,key in pbar:

                X_train, eX_train = self._load_data(key)

                # # Convert X_train to tensor and create DataLoader for mini-batching
                # X_train = torch.Tensor(X_train).to(self.device)
                
                train_loader = DataLoader(TensorDataset(X_train, eX_train), batch_size=mini_batch, shuffle=True)

                for X_batch,eX_batch in train_loader:

                    # Apply masking to training data batch
                    X_masked, mask, nanmask = self._apply_mask(X_batch)

                    # Forward pass (classification output is ignored for pretraining)
                    X_reconstructed, z = self.model(X_masked)

                    # Compute the reconstruction loss
                    # not counting parallax and ebv and manual L1
                    l1_norm = z.abs().sum()
                    loss = self.loss_fn(X_batch[:,:-4], X_reconstructed, nanmask[:, :-4], eX_batch[:, :-4]) + self.lasso * l1_norm

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
                
                if early_stopping.early_stop:
                    print("Stop early...")
                else:
                    early_stopping(validation_loss, self.model)
                running_pt_validation_loss.append(validation_loss)

            # if early_stopping.early_stop:
            #     print("Stopping early...")
            #     break

            if (epoch+1) % 2 == 0:
                os.system('mkdir /arc/home/aydanmckay/bprp_mae/andraeplots/rtdlmodels/resmlp_realmag_full_0605/epoch'+str(epoch+1))

                plt.plot(range(1, epoch+2), running_pt_loss)
                plt.plot(range(1, epoch+2), running_pt_validation_loss)
                plt.xlabel('Epoch')
                plt.ylabel('Pretrain Loss (l1)')
                plt.title('Pretrain Loss Plot')
                plt.savefig('/arc/home/aydanmckay/bprp_mae/andraeplots/rtdlmodels/resmlp_realmag_full_0605/epoch'+str(epoch+1)+'/ptlossplot.png')
                plt.close()

                torch.save(self.model.state_dict(), '/arc/home/aydanmckay/bprp_mae/.model_instances/checkpoint_'+str(epoch+1)+'_fullmodelrealmags0605.pth')

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
                             last=False,
                             test_stuff=test_stuff,
                             pt_epoch=epoch
                             )
                    
                    self.model.load_state_dict(torch.load('/arc/home/aydanmckay/bprp_mae/.model_instances/checkpoint_'+str(epoch+1)+'_fullmodelrealmags0605.pth',
                                                          map_location=self.device))
        
        torch.save(self.model.state_dict(), '/arc/home/aydanmckay/bprp_mae/.model_instances/checkpoint_'+str(epoch+1)+'_fullmodelrealmags0605.pth')

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
                # X_val = torch.Tensor(X_val).to(self.device)
    
                # Create DataLoader for mini-batching validation data
                val_loader = DataLoader(TensorDataset(X_val, eX_val), batch_size=mini_batch, shuffle=False)
    
                for X_batch,eX_batch in val_loader:
                    # Apply masking to validation data
                    X_masked, mask, nanmask = self._apply_mask(X_batch)
    
                    # Forward pass
                    X_reconstructed, _ = self.model(X_masked)
    
                    # Compute validation loss
                    # not counting the parallax and ebv
                    batch_loss = self.loss_fn(X_batch[:, :-4], X_reconstructed, nanmask[:, :-4], eX_batch)
                    
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
            feature_seed=42,    # --------------
            ensemblepath=None,  # --------------
           ):
        
        X_train = torch.Tensor(X_train).to(self.device)
        eX_train = torch.Tensor(eX_train).to(self.device)
        y_train = torch.Tensor(y_train).to(self.device)
        
        # Create DataLoader for mini-batching
        # if (ftlf == 'wmse') or (ftlf == 'wgnll'):
        e_y_train = torch.Tensor(e_y_train).to(self.device)
        rdataset = TensorDataset(X_train, eX_train, y_train, e_y_train)
        train_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=True)
        # else:
        #     train_loader = DataLoader(TensorDataset(X_train, eX_train, y_train), batch_size=mini_batch, shuffle=True)
        
        if ftact == 'relu':
            ftactivationfunc = nn.ReLU()
        elif ftact == 'elu':
            ftactivationfunc = nn.ELU()
        elif ftact == 'gelu':
            ftactivationfunc = nn.GELU()

        if (ftlf == 'wmse') or (ftlf == 'wgnll'):
            criterion = ftfile.WeightedMaskedMSELoss()
        elif ftlf == 'mse':
            criterion = ftfile.MaskedMSELoss()
        elif ftlf == 'mae':
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = ftfile.RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')

        if (ftlf == 'gnll') or (ftlf == 'wgnll'):
            criterion2 = ftfile.MaskedGaussianNLLLoss()
            
        self.ft = PredictionHead(self.latent_size,ftlabeldim,ftdim,ftactivationfunc).to(self.device)

        try:
            # path = '/arc/home/aydanmckay/bprp_mae/.ensembles/mae_0604/overfitmodel_seed42_fseed42_epoch_211.pth'
            state_dict = torch.load(path, map_location=self.device)
            # assign to models
            self.model.load_state_dict(state_dict['autoencoder_state_dict'])
            # renamed_state_dict = {f"ft.{k}": v for k, v in state_dict['prediction_head_state_dict'].items()}
            # self.ft.load_state_dict(renamed_state_dict)
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
            # optimizer = optim.Adam(list(self.model.parameters()) + list(self.ft.parameters()), lr=ftlr, weight_decay=ftl2)
        elif ftopt == 'sgd':
            optimizer = optim.SGD([
                {'params': self.model.parameters(), 'lr': 1e-5},
                {'params': self.ft.parameters(), 'lr': ftlr, 'momentum': 0.9, 'weight_decay': ftl2}
            ])
            # optimizer = optim.SGD(list(self.model.parameters()) + list(self.ft.parameters()), lr=ftlr, momentum=0.9, weight_decay=ftl2)
        elif ftopt == 'adamw':
            optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': 1e-5},
                {'params': self.ft.parameters(), 'lr': ftlr, 'weight_decay': ftl2}
            ])
            # optimizer = optim.AdamW(list(self.model.parameters()) + list(self.ft.parameters()), lr=ftlr, weight_decay=ftl2)
        
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # Define lambda functions for each group's schedule
        encoder_lambda = lambda epoch: 0.95 ** epoch         # Slow decay
        head_lambda = lambda epoch: 0.5 ** (epoch // 10)     # Step decay every 10 epochs

        # Scheduler applied to parameter groups
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[encoder_lambda, head_lambda])

        running_ft_loss = []
        running_ft_validation_loss = []

        # -----------------------------------------------------------------------------------------------
        os.system('mkdir /arc/home/aydanmckay/bprp_mae/.ensembles/quantile_0617')
        
        logging.basicConfig(filename="/arc/home/aydanmckay/bprp_mae/.ensembles/quantile_0617/"+ensemblepath+".log", 
                            level=logging.INFO, 
                            format="%(asctime)s - Sub-Epoch: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            force=True)
                            
        if pert_features or pert_labels:
            random.seed(feature_seed)
            torch.manual_seed(feature_seed)

        early_stopping = EarlyStopping(
            patience=20,          # How many epochs val_loss > train_loss before stopping
            min_delta=0.0,       # Minimum improvement in val_loss to be considered "better"
            verbose=True,        # Whether to print updates
            path='/arc/home/aydanmckay/bprp_mae/.ensembles/quantile_0617/'+ensemblepath+'.pth'  # Where to save the best model
        )
        # -----------------------------------------------------------------------------------------------

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
                    loss += self.loss_fn(X_batch[:, :-4], X_reconstructed, nanmask[:, :-4], eX_batch[:, :-4])

                if rncloss:
                    features = torch.stack((y_pred, y_pred.clone()), dim=1)  # [bs, 2, feat_dim]
                    try:
                        loss += rnc(features, y_batch)
                    except RuntimeError as e:
                        print(e)
                        print(torch.cuda.memory_summary())

                if (ftlf == 'gnll') or (ftlf == 'wgnll'):
                    # loss += criterion2(X_batch[:, :-4], X_reconstructed, eX_batch[:, :-4]**2, nanmask[:, :-4])
                    # loss += criterion2(y_pred, y_batch, y_pred_err, batch[3]**2)
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

                # After computing training and validation loss:
                early_stopping(running_ft_loss[-1], running_ft_validation_loss[-1], self.model, self.ft)

                if early_stopping.early_stop:
                    print("Stopping training early")
                #     break

                if (epoch+1) % 100 == 0:
                    torch.save({
                        'autoencoder_state_dict': self.model.state_dict(),
                        'prediction_head_state_dict': self.ft.state_dict()
                    }, '/arc/home/aydanmckay/bprp_mae/.ensembles/quantile_0617/'+ensemblepath+'_epoch_'+str(epoch+1)+'.pth')

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
        # if ftlf == 'wmse':
        e_y_val = torch.Tensor(e_y_val).to(self.device)
        rdataset = TensorDataset(X_val, eX_val, y_val, e_y_val)
        val_loader = DataLoader(rdataset, batch_size=mini_batch, shuffle=True)
        # else:
        #     val_loader = DataLoader(TensorDataset(X_val, eX_val, y_val), batch_size=mini_batch, shuffle=True)

        if (ftlf == 'wmse') or (ftlf == 'wgnll'):
            criterion = ftfile.WeightedMaskedMSELoss()
        elif ftlf == 'mse':
            criterion = ftfile.MaskedMSELoss()
        elif ftlf == 'mae':
            criterion = MaskedMAELoss()

        if rncloss:
            rnc = ftfile.RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')

        if (ftlf == 'gnll') or (ftlf == 'wgnll'):
            criterion2 = ftfile.MaskedGaussianNLLLoss()

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
                    loss += self.loss_fn(X_batch[:, :-4], X_reconstructed, nanmask[:, :-4], eX_batch[:, :-4])

                if rncloss:
                    features = torch.stack((y_pred, y_pred.clone()), dim=1)  # [bs, 2, feat_dim]
                    try:
                        loss += rnc(features, y_batch)
                    except RuntimeError as e:
                        print(e)
                        print(torch.cuda.memory_summary())

                if (ftlf == 'gnll') or (ftlf == 'wgnll'):
                    # loss += criterion2(X_batch[:, :-4], X_reconstructed, eX_batch[:, :-4]**2, nanmask[:, :-4])
                    # loss += criterion2(y_pred, y_batch, y_pred_err, batch[3]**2)
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