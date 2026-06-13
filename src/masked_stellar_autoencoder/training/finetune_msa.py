import argparse
import random

import numpy as np
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.models.model import TabResnetWrapper, make_model

from .config_paths import expand_config_paths, ft_checkpoint_paths
from .feature_noise import pert_channel_scale_vector
from .finetune_data import prepare_finetune_arrays


def _print_parallax_consistency(pack, cols, scalers) -> None:
    px_feat_idx = cols.index("PARALLAX")
    if (
        pack.get("astrometry_input_policy", "legacy_raw") == "legacy_raw"
        and pack.get("parallax_target_space", "linear_mas") == "linear_mas"
    ):
        feat_median = pack["featurescaler"].center_[px_feat_idx]
        feat_iqr = pack["featurescaler"].scale_[px_feat_idx]
        label_scaler = scalers[-1]
        label_mean = label_scaler.mean_[0]
        label_std = label_scaler.scale_[0]
        consistency_m = feat_iqr / label_std
        consistency_c = (feat_median - label_mean) / label_std
        print(f"Consistency Params for Parallax: m={consistency_m}, c={consistency_c}")
    else:
        print(
            "Skipping parallax feature/label consistency check "
            "(non-legacy astrometry input and/or log10 parallax target)."
        )


def _finetune_one_seed(
    seed: int,
    config: dict,
    pack: dict,
    trainset,
    etrainset,
    labelled_set,
    e_labelled_set,
    validset,
    evalidset,
    vlabelled_set,
    e_vlabelled_set,
    scalers,
    cols,
    error_cols,
    recon_cols,
) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))

    blocks_dims = config["model"]["layer_dims"]
    model = make_model(
        len(cols),
        blocks_dims,
        len(recon_cols),
        config["model"]["pt_activ_func"],
        config["model"]["rtdl_embed"],
        config["model"]["norm"],
        decoder_dims=config["model"].get("decoder_dims", None),
    )

    checkpoint = torch_load_trusted(
        config["model"]["saved_weights"],
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    xp_ratio = config["finetuning"].get("xp_masking_ratio", 0.3)
    m_ratio = config["finetuning"].get("m_masking_ratio", 0.3)
    ft_save_file, ft_log_file = ft_checkpoint_paths(config, seed)

    wrapper = TabResnetWrapper(
        model=model,
        datafile=config["data"]["datafile"],
        scaler=pack["featurescaler"],
        feature_cols=cols,
        error_cols=error_cols,
        recon_cols=recon_cols,
        label_scalers=pack["scalers"],
        xp_masking_ratio=xp_ratio,
        m_masking_ratio=m_ratio,
        latent_size=blocks_dims[-1],
        lr=config["training"]["lr"],
        optimizer=config["training"]["optimizer"],
        wd=config["training"]["weight_decay"],
        lasso=config["training"]["lasso"],
        lf=config["training"]["loss_fn"],
        pert_scale=float(config["finetuning"].get("pert_scale", 1.0)),
        ft_save_str=ft_save_file,
        ft_log_file=ft_log_file,
        checkpoint_interval=config["saving"]["checkpoint_interval"],
        mask_mixture_xp_full_frac=float(
            config["training"].get("mask_mixture_xp_full_frac", 0.0)
        ),
        pert_channel_scale=pert_channel_scale_vector(
            cols,
            pert_ebv_scale=float(config["finetuning"].get("pert_ebv_scale", 1.0)),
        ),
        scheduler_cosine_t0=int(config["training"].get("scheduler_cosine_t0", 10)),
        scheduler_cosine_t_mult=int(
            config["training"].get("scheduler_cosine_t_mult", 2)
        ),
        scheduler_eta_min_factor=float(
            config["training"].get("scheduler_eta_min_factor", 0.01)
        ),
    )

    _print_parallax_consistency(pack, cols, scalers)

    wrapper.fit(
        trainset,
        etrainset,
        labelled_set,
        e_y_train=e_labelled_set,
        X_val=validset,
        eX_val=evalidset,
        y_val=vlabelled_set,
        e_y_val=e_vlabelled_set,
        num_epochs=config["finetuning"]["num_epochs"],
        mini_batch=config["finetuning"]["mini_batch"],
        linearprobe=config["finetuning"]["linearprobe"],
        maskft=config["finetuning"]["mask"],
        multitask=config["finetuning"]["multitask"],
        rncloss=config["finetuning"]["rncloss"],
        ftlr=config["finetuning"]["lr"],
        ftopt=config["finetuning"]["opt"],
        ftact=config["finetuning"]["active"],
        ftl2=config["finetuning"]["l2"],
        ftlf=config["finetuning"]["lf"],
        ftlabeldim=len(pack["label_names"]),
        pert_features=config["finetuning"]["pert_features"],
        pert_labels=config["finetuning"]["pert_labels"],
        feature_seed=config["finetuning"]["pert_seed"],
        ensemblepath=config["finetuning"]["ensemble_path"],
        ft_lambda_pred=float(config["finetuning"].get("lambda_pred", 0.8)),
        ft_lambda_rec=float(config["finetuning"].get("lambda_rec", 0.2)),
        ft_quantile_label_weights=config["finetuning"].get("quantile_label_weights"),
        ft_use_sigma_quantile_weights=bool(
            config["finetuning"].get("quantile_use_label_errors", False)
        ),
        ft_sigma_weight_floor=float(
            config["finetuning"].get("quantile_sigma_weight_floor", 1e-6)
        ),
        ft_sigma_weight_max=float(
            config["finetuning"].get("quantile_sigma_weight_max", 1e6)
        ),
        ft_sigma_weight_normalize_batch=bool(
            config["finetuning"].get("quantile_sigma_weight_normalize_batch", True)
        ),
        ft_encoder_lr=(
            float(config["finetuning"]["encoder_lr"])
            if config["finetuning"].get("encoder_lr") is not None
            else None
        ),
        ft_scheduler_encoder_decay=float(
            config["finetuning"].get("lr_scheduler_encoder_decay", 0.95)
        ),
        ft_scheduler_head_decay=float(
            config["finetuning"].get("lr_scheduler_head_decay", 0.5)
        ),
        ft_scheduler_head_step_epochs=int(
            config["finetuning"].get("lr_scheduler_head_step_epochs", 10)
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Train MSA")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Subsample training rows for pilot runs only",
    )
    parser.add_argument(
        "--max-valid-rows",
        type=int,
        default=None,
        help="Subsample validation rows for pilot runs only",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    expand_config_paths(config)

    if config["finetuning"]["ensemble"]:
        rng = np.random.default_rng(config["finetuning"].get("ensemble_seed", 42))
        n_ens = int(config["finetuning"].get("ensemble_size", 20))
        seeds = rng.integers(0, 1000, size=n_ens).tolist()
    else:
        seeds = [config["finetuning"]["seed"]]

    pack = prepare_finetune_arrays(
        config,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
    )
    trainset = pack["trainset"]
    etrainset = pack["etrainset"]
    labelled_set = pack["labelled_set"]
    e_labelled_set = pack["e_labelled_set"]
    validset = pack["validset"]
    evalidset = pack["evalidset"]
    vlabelled_set = pack["vlabelled_set"]
    e_vlabelled_set = pack["e_vlabelled_set"]
    scalers = pack["scalers"]
    cols = pack["feature_cols"]
    error_cols = pack["error_cols"]
    recon_cols = pack["recon_cols"]

    for seed in seeds:
        _finetune_one_seed(
            seed,
            config,
            pack,
            trainset,
            etrainset,
            labelled_set,
            e_labelled_set,
            validset,
            evalidset,
            vlabelled_set,
            e_vlabelled_set,
            scalers,
            cols,
            error_cols,
            recon_cols,
        )


if __name__ == "__main__":
    main()
