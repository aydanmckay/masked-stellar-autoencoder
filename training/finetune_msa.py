import yaml
import h5py
import argparse
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from astropy.table import Table
import pandas as pd
import random
import os
import sys

# Add the repo root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from models.model import make_model, TabResnetWrapper

def main():

    parser = argparse.ArgumentParser(description="Train MSA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    # load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config['finetuning']['ensemble']:
        seeds = np.random.randint(0, 1000, size=100).tolist()
    else:
        seeds = [config['finetuning']['seed']]

    # loading the finetuning dataset
    data = Table.read(config['data']['ft_datafile']).to_pandas()
    errordata = data.copy()

    cols = config['data']['feature_cols']
    classes = config['data']['classes']
    error_cols = config['data']['error_cols']

    data = data[classes+cols]
    errordata = errordata[error_cols]
    errordata = errordata.fillna(errordata.max())

    trainset, validset, etrainset, evalidset = train_test_split(data.to_numpy(), errordata.to_numpy(), test_size=0.2, random_state=42)
    validset, testset, evalidset, etestset = train_test_split(validset, evalidset, test_size=0.33, random_state=42)

    # assuming that the layout of the file is label, label error, ..., features
    num_classes = len(classes)
    
    target_train = trainset[:, :num_classes]
    trainset = trainset[:, num_classes:]
    target_valid = validset[:, :num_classes]
    validset = validset[:, num_classes:]
    target_test = testset[:, :num_classes]
    testset = testset[:, num_classes:]

    # scaling the targets (individually in case of single task finetuning) and features
    scalers = [StandardScaler() for _ in range(int(num_classes/2))]
    labelled_set = []
    e_labelled_set = []
    vlabelled_set = []
    e_vlabelled_set = []
    
    for i in range(int(num_classes/2)):
        labelled_set.append(scalers[i].fit_transform(target_train[:, i*2].reshape(-1, 1)))
        elabel = target_train[:, i*2+1] / scalers[i].scale_
        e_labelled_set.append(elabel.reshape(-1, 1))
        
        vlabelled_set.append(scalers[i].transform(target_valid[:, i*2].reshape(-1, 1)))
        velabel = target_valid[:, i*2+1] / scalers[i].scale_
        e_vlabelled_set.append(velabel.reshape(-1, 1))
        

    target_set = target_test[:, [i for i in range(num_classes) if i % 2 == 0]]

    try:
        pos = cols.index("PARALLAX")
    except ValueError:
        raise ValueError("PARALLAX column not found in feature columns")
    
    scaler = StandardScaler()
    label = scaler.fit_transform(trainset[:, pos].reshape(-1, 1))
    elabel = etrainset[:, pos] / scaler.scale_
    vlabel = scaler.transform(validset[:, pos].reshape(-1, 1))
    velabel = evalidset[:, pos] / scaler.scale_

    labelled_set.append(label)
    labelled_set = np.concatenate(labelled_set, axis=1)

    e_labelled_set.append(elabel.reshape(-1, 1))
    e_labelled_set = np.concatenate(e_labelled_set, axis=1)

    scalers.append(scaler)

    target_set = np.concatenate([target_set, testset[:, pos].reshape(-1, 1)], axis=1)
    
    labels = [i for i in range(num_classes) if i % 2 == 0] + ['parallax']

    vlabelled_set.append(vlabel)
    vlabelled_set = np.concatenate(vlabelled_set, axis=1)

    e_vlabelled_set.append(velabel.reshape(-1, 1))
    e_vlabelled_set = np.concatenate(e_vlabelled_set, axis=1)

    # Validate data before scaling
    if np.any(np.isnan(trainset)) or np.any(np.isinf(trainset)):
        print("Warning: Invalid values in training features before scaling")
    if np.any(np.isnan(etrainset)) or np.any(np.isinf(etrainset)):
        print("Warning: Invalid values in training errors before scaling")
    
    featurescaler = RobustScaler()
    featurescaler.fit(trainset)
    
    # Validate scaler was fitted properly
    if not hasattr(featurescaler, 'scale_') or featurescaler.scale_ is None:
        raise ValueError("Feature scaler failed to fit properly")
    
    trainset = featurescaler.transform(trainset)
    validset = featurescaler.transform(validset)
    testset = featurescaler.transform(testset)
    scale_factors = featurescaler.scale_  # This is the IQR used by RobustScaler for each feature
    
    # Validate scale factors
    if np.any(scale_factors <= 0):
        print("Warning: Zero or negative scale factors detected")
        scale_factors = np.where(scale_factors <= 0, 1.0, scale_factors)
    
    etrainset = etrainset / scale_factors
    evalidset = evalidset / scale_factors
    etestset = etestset / scale_factors

    test_stuff = (testset, target_set, scalers, labels)

    for seed in seeds:

        random.seed(seed)
        torch.manual_seed(seed)

        blocks_dims = config['model']['layer_dims']
        pt_activ = config['model']['pt_activ_func']
        d_embed = config['model']['rtdl_embed']
        norm = config['model']['norm']

        recon_cols = config['data']['recon_cols']
    
        model = make_model(
            len(cols),
            blocks_dims,
            len(recon_cols),
            pt_activ,
            d_embed,
            norm,
        )

        model.load_state_dict(torch.load(config['model']['saved_weights'], map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    
        xp_ratio = config['training']['xp_masking_ratio']
        m_ratio = config['training']['m_masking_ratio']
        lr = config['training']['lr']
        wd = config['training']['weight_decay']
        lasso = config['training']['lasso']
        opt = config['training']['optimizer']
        lf = config['training']['loss_fn']
        
        ft_save_file = config['saving']['model_str']
        ft_log_file = config['saving']['log_file']
        ci = config['saving']['checkpoint_interval']

        pretrain_file = config['data']['datafile']

        # Initialize the pretraining wrapper
        wrapper = TabResnetWrapper(
            model=model,
            datafile=pretrain_file,
            scaler=featurescaler,
            feature_cols=cols,
            error_cols=error_cols,
            recon_cols=recon_cols,
            xp_masking_ratio=xp_ratio,
            m_masking_ratio=m_ratio,
            latent_size=blocks_dims[-1],
            lr=lr,
            optimizer=opt,
            wd=wd,
            lasso=lasso,
            lf=lf,
            ft_save_str=ft_save_file,
            ft_log_file=ft_log_file,
            checkpoint_interval=ci,
        )

        wrapper.fit(
            trainset,
            etrainset,
            labelled_set,
            e_y_train=e_labelled_set,
            X_val=validset, 
            eX_val=evalidset,
            y_val=vlabelled_set,
            e_y_val=e_vlabelled_set,
            num_epochs=config['finetuning']['num_epochs'], # needs to all be part of the config
            mini_batch=config['finetuning']['mini_batch'], 
            linearprobe=config['finetuning']['linearprobe'], 
            maskft=config['finetuning']['mask'],
            multitask=config['finetuning']['multitask'],
            rncloss=config['finetuning']['rncloss'],
            ftlr=config['finetuning']['lr'],
            ftopt=config['finetuning']['opt'],
            ftact=config['finetuning']['activ'],
            ftl2=config['finetuning']['l2'],
            ftlf=config['finetuning']['lf'],
            ftlabeldim=len(labels),
            pert_features=config['finetuning']['pert_features'],
            pert_labels=config['finetuning']['pert_labels'],
            feature_seed=config['finetuning']['pert_seed'],
            ensemblepath=config['finetuning']['ensemble_path']
        )

if __name__ == '__main__':
    main()