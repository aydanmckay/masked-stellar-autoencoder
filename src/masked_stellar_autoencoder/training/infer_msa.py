#!/usr/bin/env python3
"""
Masked Stellar Autoencoder: Inference Pipeline

Evaluates the MAE backbone and predicting head over a catalogue.
Handles dynamic feature and label re-scaling mappings correctly using the original config datasets,
so that inferences on blind data correctly trace original distribution densities.

Usage:
  PYTHONPATH=. python training/infer_msa.py \
    --config configs/finetune.yaml \
    --checkpoint results/finetune_run/10M_finetuned.pth \
    --inference-data new_stars_to_infer.h5 \
    --out results/inference_catalogue.csv

Requirements:
If `inference-data` is not provided, it evaluates against the unmasked test-split defined in the config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from models.checkpoint_load import torch_load_trusted
from models.model import PredictionHead, make_model

from training.checkpoint_keys import autoencoder_state_dict, prediction_head_state_dict
from training.config_paths import expand_config_paths
from training.conformal import apply_cqr_offsets_inplace
from training.eval_ensemble import _inverse_quantile_block
from training.finetune_data import prepare_finetune_arrays


@torch.no_grad()
def infer_catalogue(
    model,
    head,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates encoder embeddings and quantile parameters."""
    model.eval()
    head.eval()
    embeddings = []
    quantiles = []

    for i in range(0, len(X_scaled), batch_size):
        xb = torch.tensor(
            X_scaled[i : i + batch_size], dtype=torch.float32, device=device
        )
        xb = torch.nan_to_num(xb, nan=-9999.0)

        # Extracted 256-D Latent Encoder Emdeddings
        z = model.encoder(xb)
        embeddings.append(z.cpu().numpy())

        # Parameter prediction Quantiles
        # For non-linear models, returns [batch, num_labels, 3 (lower, med, upper)]
        q = head(z)
        quantiles.append(q.cpu().numpy())

    return np.concatenate(embeddings, axis=0), np.concatenate(quantiles, axis=0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="Fine-tuned .pth file")
    ap.add_argument(
        "--inference-data",
        default=None,
        help="HDF5 file with columns matching feature_cols",
    )
    ap.add_argument("--out", required=True, help="Output csv catalog path")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--conformal-json",
        default=None,
        help="CQR bounds json to rigidly calibrate lower/upper bounds",
    )
    return ap.parse_args()


def load_scalers(state, config):
    if state.get("featurescaler") is None or state.get("label_scalers") is None:
        print(
            "Warning: Scalers not found in checkpoint. Falling back to dynamic pre-fitting (will read entire training catalogue into RAM)..."
        )
        pack = prepare_finetune_arrays(config)
        featurescaler = pack["featurescaler"]
        label_scalers = pack["scalers"]
        label_names = pack["label_names"]
        cols = pack["feature_cols"]
        recon_cols = pack["recon_cols"]
    else:
        print("Using rapidly-loaded serialized scalers from checkpoint!")
        featurescaler = state["featurescaler"]
        label_scalers = state["label_scalers"]
        label_names = ["teff", "logg", "fe_h", "alpha", "age", "parallax"]
        error_cols, cols = [], []
        for x in config["data"]["feature_cols"]:
            if x.startswith("e_"):
                error_cols.append(x[2:])
            else:
                cols.append(x)
        recon_cols = config["data"]["recon_cols"]
        # Mock pack object for downstream post_processing rules
        pack = {
            "parallax_target_space": config.get("preprocessing", {}).get(
                "parallax_target_space", "linear_mas"
            ),
            "teff_target_space": config.get("preprocessing", {}).get(
                "teff_target_space", "linear"
            ),
            "scalers": label_scalers,
            "featurescaler": featurescaler,
        }
    return pack, featurescaler, label_scalers, label_names, cols, recon_cols


def load_inference_data(inference_data, pack, featurescaler, cols):
    if inference_data:
        print(f"Loading external inference catalogue: {inference_data}")
        with h5py.File(inference_data, "r") as f:
            if "table" in f:
                dset = f["table"]
            else:
                dset = f[list(f.keys())[0]]  # Just pull first key
            # We strictly slice only the features requested in `config["data"]["feature_cols"]`
            df_inf = pd.DataFrame(
                {c: dset[c][:] for c in cols if c in dset.dtype.names}
            )
            source_ids = f.get("source_id", np.arange(len(df_inf)))[:]

        # Standardize missing as pandas nans before scaler
        X_infer = df_inf.values
        X_infer_scaled = featurescaler.transform(X_infer)
    else:
        print(
            "No --inference-data provided, inferring on base config testset fraction."
        )
        # Assuming prepare_finetune_arrays already robust scaled
        X_infer_scaled = pack["testset"]
        source_ids = np.arange(len(X_infer_scaled))
    return source_ids, X_infer_scaled


def build_and_load_models(config, num_cols, num_recon_cols, num_labels, state, device):
    blocks_dims = config["model"]["layer_dims"]
    model = make_model(
        num_cols,
        blocks_dims,
        num_recon_cols,
        config["model"]["pt_activ_func"],
        config["model"]["rtdl_embed"],
        config["model"]["norm"],
        decoder_dims=config["model"].get("decoder_dims"),
    ).to(device)

    act_name = config["finetuning"].get("active", "relu")
    ftact = (
        torch.nn.GELU()
        if act_name == "gelu"
        else (torch.nn.ELU() if act_name == "elu" else torch.nn.ReLU())
    )
    head = PredictionHead(blocks_dims[-1], num_labels, ftact).to(device)

    model.load_state_dict(autoencoder_state_dict(state))
    head.load_state_dict(prediction_head_state_dict(state))
    return model, head


def save_predictions(out_path, source_ids, label_names, phys_q, embeddings):
    out_dict = {"source_id": source_ids}

    # Store predictions via METHODOLOGY policy `PARAM_med` alongside uncertainty bounds
    for idx, name in enumerate(label_names):
        out_dict[f"{name}_lower"] = phys_q[:, idx, 0]
        out_dict[f"{name}_med"] = phys_q[:, idx, 1]
        out_dict[f"{name}_upper"] = phys_q[:, idx, 2]

    # Also publish extracted representations (usually the 256-D vectors)
    for embed_idx in range(embeddings.shape[1]):
        out_dict[f"embedding_{embed_idx}"] = embeddings[:, embed_idx]

    df_out = pd.DataFrame(out_dict)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nInference Complete. Saved {len(df_out)} targets to {out_path}")


def main():
    args = parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)
    expand_config_paths(config)

    print(f"Loading weights from {args.checkpoint}...")
    state = torch_load_trusted(args.checkpoint, map_location=device)

    # 1. Pipeline dynamic scaler generation (Requires generating split on finetune DB)
    pack, featurescaler, label_scalers, label_names, cols, recon_cols = load_scalers(
        state, config
    )

    # 2. Loading inference rows
    source_ids, X_infer_scaled = load_inference_data(
        args.inference_data, pack, featurescaler, cols
    )

    # 3. Model construction
    model, head = build_and_load_models(
        config, len(cols), len(recon_cols), len(label_names), state, device
    )

    # 4. Generate Embeddings & Predictions
    print("Executing GPU Inference...")
    embeddings, preds_q = infer_catalogue(
        model, head, X_infer_scaled, device, args.batch_size
    )

    # Apply empirical test-set Split Conformal Calibration modifiers if provided
    if args.conformal_json:
        print(f"Applying Conformal mapping from {args.conformal_json}...")
        with open(args.conformal_json) as f:
            calib_doc = json.load(f)
        # Apply the scale shifts directly to outputs inplace
        apply_cqr_offsets_inplace(preds_q, calib_doc)

    print("Inversing label scalers (Mapping physical units)...")
    # Mapping physical scaled quantiles to real dimension bounds
    # Shape of phys_q: (N, label_dim, 3) (lower, median, upper)
    phys_q = _inverse_quantile_block(preds_q, label_scalers, pack)

    # 5. Assemble and save mapping output
    save_predictions(args.out, source_ids, label_names, phys_q, embeddings)


if __name__ == "__main__":
    main()
