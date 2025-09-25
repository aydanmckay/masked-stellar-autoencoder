import yaml
import h5py
import argparse
from sklearn.preprocessing import RobustScaler
import numpy as np

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
    featurescaler.fit(X)
    
    del X
    del cols

    blocks_dims = config['model']['layer_dims']
    recon_cols = config['model']['recon_cols']
    pt_activ = config['model']['pt_activ_func']
    d_embed = config['model']['rtdl_embed']
    norm = config['model']['norm']
    
    
    model = make_model(
        len(cols),
        blocks_dims,
        len(recon_cols),
        pt_activ,
        d_embed,
        norm,
    )

    error_cols = config['model']['error_cols']
    xp_ratio = config['model']['xp_masking_ratio']
    m_ratio = config['model']['m_masking_ratio']
    lr = config['model']['lr']
    wd = config['model']['weight_decay']
    lasso = config['model']['lasso']
    opt = config['model']['optimizer']
    lf = config['model']['loss_fn']
    pt_save_file = config['model']['model_str']
    pt_log_file = config['model']['log_file']
    ci = config['model']['checkpoint_interval']

    classes = config['data']['num_classes']

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
        num_classes=classes,
        latent_size=blocks_dims[-1],
        lr=lr,
        optimizer=opt,
        wd=wd,
        lasso=lasso,
        lf=lf,
        pt_save_str=pt_save_file,
        pt_log_file=pt_log_file,
        checkpoint_interval=ci,
    )

    epochs = config['model']['epochs']
    batch = config['model']['mini_batch_size']

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