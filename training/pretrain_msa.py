import yaml
import h5py
import argparse
from sklearn.preprocessing import RobustScaler
import numpy as np
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
    
    # loading the pretraining file to pass to the wrapper
    pretrain_file = h5py.File(config['data']['datafile'])

    # splitting up the keys
    keys_valid = config['data']['valid_keys']
    keys_train = [item for item in list(pretrain_file.keys()) if item not in keys_valid]

    featurescaler = RobustScaler()
    # as a test since there currently isn't a finetuning set
    X = pretrain_file[keys_train[0]][:]

    cols = config['data']['feature_cols']
    
    X = np.column_stack([TabResnetWrapper._clean_column(col, X[col]) for col in cols])
    
    # Validate data before fitting scaler
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("Warning: Invalid values detected in training data before scaling")
        # Remove rows with all NaN values
        valid_rows = ~np.all(np.isnan(X), axis=1)
        X = X[valid_rows]
        if len(X) == 0:
            raise ValueError("No valid data remaining after removing NaN rows")
    
    featurescaler.fit(X)
    
    # Validate scaler was fitted properly
    if not hasattr(featurescaler, 'scale_') or featurescaler.scale_ is None:
        raise ValueError("Scaler failed to fit properly - scale_ attribute missing")
    
    del X

    blocks_dims = config['model']['layer_dims']
    pt_activ = config['model']['pt_activ_func']
    d_embed = config['model']['rtdl_embed']
    norm = config['model']['norm']
    decoder_dims = config['model'].get('decoder_dims', None)  # Optional asymmetric decoder

    recon_cols = config['data']['recon_cols']

    model = make_model(
        len(cols),
        blocks_dims,
        len(recon_cols),
        pt_activ,
        d_embed,
        norm,
        decoder_dims=decoder_dims,
    )

    xp_ratio = config['training']['xp_masking_ratio']
    m_ratio = config['training']['m_masking_ratio']
    lr = config['training']['lr']
    wd = config['training']['weight_decay']
    lasso = config['training']['lasso']
    opt = config['training']['optimizer']
    lf = config['training']['loss_fn']
    pert_features = config['training'].get('pert_features', False)  # Optional data augmentation
    pert_scale = config['training'].get('pert_scale', 1.0)  # Noise scale factor

    pt_save_file = config['saving']['model_str']
    pt_log_file = config['saving']['log_file']
    ci = config['saving']['checkpoint_interval']

    error_cols = config['data']['error_cols']

    # Initialize the pretraining wrapper
    pretrain_wrapper = TabResnetWrapper(
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
        pt_save_str=pt_save_file,
        pt_log_file=pt_log_file,
        checkpoint_interval=ci,
        pert_features=pert_features,
        pert_scale=pert_scale,
    )

    epochs = config['training']['epochs']
    batch = config['training']['mini_batch_size']

    # pretrain, train, and predict
    pretrain_wrapper.pretrain_hdf(
        keys_train,
        num_epochs=epochs,
        val_keys=keys_valid,
        mini_batch=batch,
    )

    pretrain_file.close()

if __name__ == "__main__":
    main()