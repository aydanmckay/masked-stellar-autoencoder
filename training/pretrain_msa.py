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
from data.data_validator import DataValidator

def main():

    parser = argparse.ArgumentParser(description="Train MSA")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML file")
    args = parser.parse_args()

    # load YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Construct absolute paths
    base_path = config['paths']['base']
    datafile = os.path.join(base_path, config['paths']['datafile'])
    model_str = os.path.join(base_path, config['paths']['model_str'])
    log_file = os.path.join(base_path, config['paths']['log_file'])

    # loading the pretraining file to pass to the wrapper
    DataValidator.validate_hdf5_file(datafile)
    pretrain_file = h5py.File(datafile, 'r')

    # splitting up the keys
    keys_valid = config['data']['valid_keys']
    keys_train = [item for item in list(pretrain_file.keys()) if item not in keys_valid]

    featurescaler = RobustScaler()
    # as a test since there currently isn't a finetuning set
    X_sample = pretrain_file[keys_train[0]][:]

    cols = config['data']['feature_cols']
    
    X_sample = np.column_stack([TabResnetWrapper._clean_column(col, X_sample[col]) for col in cols])
    
    # Validate data before fitting scaler
    validator_report = DataValidator.validate_stellar_data(X_sample, cols)
    if not validator_report['valid']:
        raise ValueError(f"Initial data validation failed: {validator_report['errors']}")

    # Remove rows with all NaN values before fitting scaler
    valid_rows = ~np.all(np.isnan(X_sample), axis=1)
    X_sample = X_sample[valid_rows]
    if len(X_sample) == 0:
        raise ValueError("No valid data remaining after removing NaN rows")
    
    featurescaler.fit(X_sample)
    
    # Validate scaler was fitted properly
    DataValidator.validate_scaling_consistency(featurescaler, X_sample)
    
    del X_sample

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

    initial_xp_ratio = config['training']['initial_xp_masking_ratio']
    initial_m_ratio = config['training']['initial_m_masking_ratio']
    xp_ratio = config['training']['xp_masking_ratio']
    m_ratio = config['training']['m_masking_ratio']
    lr = config['training']['lr']
    wd = config['training']['weight_decay']
    lasso = config['training']['lasso']
    opt = config['training']['optimizer']
    lf = config['training']['loss_fn']
    
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
        initial_xp_masking_ratio=initial_xp_ratio,
        initial_m_masking_ratio=initial_m_ratio,
        xp_masking_ratio=xp_ratio,
        m_masking_ratio=m_ratio,
        latent_size=blocks_dims[-1],
        lr=lr,
        optimizer=opt,
        wd=wd,
        lasso=lasso,
        lf=lf,
        pt_save_str=model_str,
        pt_log_file=log_file,
        checkpoint_interval=ci,
        use_curriculum=config['training'].get('use_curriculum', False),
        curriculum_start_ratio=config['training'].get('curriculum_start_ratio', 0.4),
        curriculum_end_ratio=config['training'].get('curriculum_end_ratio', 0.9),
        use_ema=config['training'].get('use_ema', False),
        ema_decay=config['training'].get('ema_decay', 0.999),
        warmup_epochs=config['training'].get('warmup_epochs', 5),
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