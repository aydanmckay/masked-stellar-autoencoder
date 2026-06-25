#!/usr/bin/env python3
"""
Unified evaluation: ensemble checkpoints, metrics (global + [Fe/H] + G + E(B-V) + ϖ/σ + parallax truth quartiles + XP-off), JSON + LaTeX.

Usage (from repo root):
  PYTHONPATH=. python training/eval_ensemble.py --config configs/finetune.yaml \\
    --checkpoints path/to/a.pth path/to/b.pth --out results/eval_run1

Requires torch, sklearn, pyyaml, astropy (same as finetune).

Optional ``--conformal-json`` applies CQR offsets (see ``training/conformal.py``) and
adds interval coverage entries to ``metrics.json``.

Checkpoints from linear-probe fine-tunes include ``linear_probe: true``; this script
builds ``nn.Linear`` instead of ``PredictionHead`` automatically.

Encoder weights may be stored as ``autoencoder_state_dict`` (fine-tune) or
``model_state_dict`` (pretrain naming); the prediction head must still be present
for evaluation.
"""

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.models.model import PredictionHead, make_model

from .checkpoint_keys import autoencoder_state_dict, prediction_head_state_dict
from .config_paths import expand_config_paths
from .conformal import apply_cqr_offsets_inplace, interval_coverage
from .finetune_data import prepare_finetune_arrays


def _inverse_labels(
    y_scaled: np.ndarray, scalers, pack: dict[str, Any] | None = None
) -> np.ndarray:
    """y_scaled: (N, 6) in training label space; inverse to physical units for metrics."""
    out = np.zeros_like(y_scaled, dtype=np.float64)
    parallax_space = "linear_mas"
    teff_space = "linear"
    if pack is not None:
        parallax_space = pack.get("parallax_target_space", "linear_mas")
        teff_space = pack.get("teff_target_space", "linear")
    for i in range(6):
        col = scalers[i].inverse_transform(y_scaled[:, i].reshape(-1, 1)).ravel()
        if i == 5 and parallax_space == "log10_mas" or i == 0 and teff_space == "log10":
            out[:, i] = np.power(10.0, col)
        else:
            out[:, i] = col
    return out


def _inverse_quantile_block(
    y_q: np.ndarray, scalers, pack: dict[str, Any] | None = None
) -> np.ndarray:
    """Map (N, L, 3) scaled quantiles to physical units (same rules as ``_inverse_labels``)."""
    y_q = np.asarray(y_q, dtype=np.float64)
    n, ell, kq = y_q.shape
    if kq != 3:
        raise ValueError("expected quantile dimension 3 (lower, median, upper)")
    out = np.zeros_like(y_q, dtype=np.float64)
    parallax_space = "linear_mas"
    teff_space = "linear"
    if pack is not None:
        parallax_space = pack.get("parallax_target_space", "linear_mas")
        teff_space = pack.get("teff_target_space", "linear")
    for j in range(ell):
        for k in range(3):
            col = scalers[j].inverse_transform(y_q[:, j, k].reshape(-1, 1)).ravel()
            if (
                j == 5
                and parallax_space == "log10_mas"
                or j == 0
                and teff_space == "log10"
            ):
                out[:, j, k] = np.power(10.0, col)
            else:
                out[:, j, k] = col
    return out


def _nmad(residuals: np.ndarray) -> float:
    x = residuals - np.nanmedian(residuals)
    return float(1.4826 * np.nanmedian(np.abs(x)))


def _quartile_bin_metrics(
    aux: np.ndarray | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: list,
    prefix: str,
    *,
    min_finite: int = 20,
    min_bin: int = 5,
) -> dict:
    """Metrics in quartiles of an auxiliary scalar (e.g. G mag, E(B-V), ϖ/σ)."""
    if aux is None:
        return {}
    aux = np.asarray(aux, dtype=np.float64).ravel()
    if aux.shape[0] != y_true.shape[0]:
        return {}
    fin = np.isfinite(aux)
    if int(np.count_nonzero(fin)) < min_finite:
        return {}
    qs = np.quantile(aux[fin], [0.25, 0.5, 0.75])
    edges = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
    tags = ["q0_25", "q25_50", "q50_75", "q75_100"]
    block: dict = {}
    for (lo, hi), tag in zip(edges, tags, strict=False):
        m = (aux > lo) & (aux <= hi) & fin
        key = f"{prefix}_{tag}"
        if int(np.count_nonzero(m)) < min_bin:
            block[key] = {"n": int(np.count_nonzero(m))}
        else:
            block[key] = _metrics_block(y_true[m], y_pred[m], names)
    return block


def _parallax_snr_test_vector(
    parallax_mas: np.ndarray, sigma_mas: np.ndarray, floor: float = 1e-6
) -> np.ndarray:
    """ϖ/σ using Gaia parallax and formal error on the test rows (physical units)."""
    pi = np.asarray(parallax_mas, dtype=np.float64).ravel()
    sig = np.asarray(sigma_mas, dtype=np.float64).ravel()
    out = np.full(pi.shape[0], np.nan, dtype=np.float64)
    m = np.isfinite(pi) & np.isfinite(sig) & (sig > floor) & (pi > 0)
    out[m] = pi[m] / np.maximum(sig[m], floor)
    return out


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray, names: list) -> dict:
    block = {}
    for i, name in enumerate(names):
        m = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        if m.sum() < 2:
            block[name] = {
                "n": int(m.sum()),
                "RMSE": None,
                "MAE": None,
                "R2": None,
                "NMAD": None,
            }
            continue
        yt, yp = y_true[m, i], y_pred[m, i]
        block[name] = {
            "n": int(m.sum()),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
            "NMAD": _nmad(yt - yp),
        }
    return block


def _mask_xp_columns(x: np.ndarray, xp_lo: int = 5, xp_hi: int = 115) -> np.ndarray:
    out = x.copy()
    out[:, xp_lo:xp_hi] = np.nan
    return out


def _feature_batch_tensor(chunk: np.ndarray, device: torch.device) -> torch.Tensor:
    """Contiguous float32 batch on device with NaN sentinel for the encoder."""
    arr = np.ascontiguousarray(chunk, dtype=np.float32)
    xb = torch.as_tensor(arr, device=device)
    return torch.nan_to_num(xb, nan=-9999.0)


@torch.inference_mode()
def predict_batches(
    model,
    head,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    linear_probe: bool,
    return_full_quantiles: bool = False,
) -> np.ndarray:
    model.eval()
    head.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        xb = _feature_batch_tensor(X[i : i + batch_size], device)
        z = model.encoder(xb)
        if linear_probe:
            med = head(z).cpu().numpy()
            outs.append(med)
        else:
            q = head(z)
            outs.append(
                q.cpu().numpy() if return_full_quantiles else q[:, :, 1].cpu().numpy()
            )
    return np.concatenate(outs, axis=0)


def _bins_true_parallax_quartiles(
    y_t: np.ndarray,
    y_p: np.ndarray,
    label_names: list,
    prefix: str,
) -> dict:
    """Binned metrics vs spectroscopic truth parallax (mas) quartiles on the test set."""
    i = label_names.index("parallax")
    pi = y_t[:, i]
    fin = np.isfinite(pi) & (pi > 0)
    if fin.sum() < 20:
        return {}
    qs = np.quantile(pi[fin], [0.25, 0.5, 0.75])
    edges = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]
    tag_names = ["pi_q0_25", "pi_q25_50", "pi_q50_75", "pi_q75_100"]
    block: dict = {}
    for (lo, hi), tag in zip(edges, tag_names, strict=False):
        m = (pi > lo) & (pi <= hi) & fin
        key = f"{prefix}_{tag}"
        if m.sum() < 5:
            block[key] = {"n": int(m.sum())}
        else:
            block[key] = _metrics_block(y_t[m], y_p[m], label_names)
    return block


def _ensemble_median_predictions(
    model,
    head,
    loaded_states: list,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    linear_probe: bool,
    return_full_quantiles: bool = False,
) -> np.ndarray:
    preds = []
    for state in loaded_states:
        model.load_state_dict(autoencoder_state_dict(state))
        head.load_state_dict(prediction_head_state_dict(state))
        preds.append(
            predict_batches(
                model,
                head,
                X,
                device,
                batch_size,
                linear_probe=linear_probe,
                return_full_quantiles=return_full_quantiles,
            )
        )
    return np.median(np.stack(preds, axis=0), axis=0)


def write_latex_metrics_table(rows_xp_on: dict, rows_xp_off: dict, path: str) -> None:
    """Minimal table: RMSE, MAE, R2, NMAD for XP on/off (two side-by-side blocks)."""
    names = ["teff", "logg", "fe_h", "alpha", "age", "parallax"]
    lines = [
        r"\begin{table*}[h!]",
        r"    \centering",
        r"    \caption{Metrics from \texttt{eval\_ensemble.py} (regenerate; do not hand-edit).}\label{tab:metrics_eval}",
        r"    \begin{tabular}{llllllllll}",
        r"        \toprule",
        r"        {Label} & {Unit} & {RMSE} & {MAE} & {R$^2$} & {NMAD} & {RMSE$_M$} & {MAE$_M$} & {R$^2$_M} & {NMAD$_M$} \\ \midrule",
    ]
    units = ["[K]", "", "", "", "[Gyr]", ""]
    for i, name in enumerate(names):
        a = rows_xp_on.get(name, {})
        b = rows_xp_off.get(name, {})
        u = units[i]

        def fmt(d, k):
            v = d.get(k)
            return "—" if v is None else f"{v:.4g}"

        name_latex = name.replace("_", "\\_")
        lines.append(
            f"        {{{name_latex}}} & {u} & {fmt(a, 'RMSE')} & {fmt(a, 'MAE')} & "
            f"{fmt(a, 'R2')} & {fmt(a, 'NMAD')} & {fmt(b, 'RMSE')} & {fmt(b, 'MAE')} & "
            f"{fmt(b, 'R2')} & {fmt(b, 'NMAD')} \\\\"
        )
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table*}"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Fine-tuned .pth files (autoencoder+head dict)",
    )
    ap.add_argument("--out", default="results/eval_default")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--conformal-json",
        default=None,
        help="Optional CQR offsets JSON (scaled space); widens q16/q84 and reports interval coverage",
    )
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)
    expand_config_paths(config)

    pack = prepare_finetune_arrays(config)
    X_test = pack["testset"].astype(np.float32)
    y_phys = pack["target_set"].astype(np.float64)
    scalers = pack["scalers"]
    label_names = pack["label_names"]
    cols = pack["feature_cols"]

    blocks_dims = config["model"]["layer_dims"]
    recon_cols = pack["recon_cols"]
    model = make_model(
        len(cols),
        blocks_dims,
        len(recon_cols),
        config["model"]["pt_activ_func"],
        config["model"]["rtdl_embed"],
        config["model"]["norm"],
        decoder_dims=config["model"].get("decoder_dims"),
    ).to(device)
    act = config["finetuning"].get("active", "relu")
    if act == "relu":
        ftact = torch.nn.ReLU()
    elif act == "elu":
        ftact = torch.nn.ELU()
    else:
        ftact = torch.nn.GELU()

    loaded_states = []
    for ckpt in args.checkpoints:
        state = torch_load_trusted(ckpt, map_location="cpu")
        loaded_states.append(state)

    ensemble_linear = bool(loaded_states[0].get("linear_probe", False))
    for i, state in enumerate(loaded_states):
        if bool(state.get("linear_probe", False)) != ensemble_linear:
            raise ValueError(
                f"All checkpoints must share the same linear_probe flag (mismatch at {args.checkpoints[i]})"
            )

    if ensemble_linear:
        head = torch.nn.Linear(blocks_dims[-1], len(label_names)).to(device)
    else:
        head = PredictionHead(blocks_dims[-1], len(label_names), ftact).to(device)

    ens_med = _ensemble_median_predictions(
        model,
        head,
        loaded_states,
        X_test,
        device,
        args.batch_size,
        linear_probe=ensemble_linear,
    )
    y_pred_phys = _inverse_labels(ens_med, scalers, pack)

    X_off = _mask_xp_columns(X_test)
    ens_med_off = _ensemble_median_predictions(
        model,
        head,
        loaded_states,
        X_off,
        device,
        args.batch_size,
        linear_probe=ensemble_linear,
    )
    y_pred_off_phys = _inverse_labels(ens_med_off, scalers, pack)

    out = {
        "global_xp_on": _metrics_block(y_phys, y_pred_phys, label_names),
        "global_xp_off": _metrics_block(y_phys, y_pred_off_phys, label_names),
        "bins_feh_xp_on": {},
        "bins_feh_xp_off": {},
    }

    feh_true = y_phys[:, 2]
    for lo, hi, tag in [
        (-np.inf, -2.0, "feh_lt_m2"),
        (-2.0, -1.0, "feh_m2_m1"),
        (-1.0, np.inf, "feh_gt_m1"),
    ]:
        m = (feh_true >= lo) & (feh_true < hi) & np.isfinite(feh_true)
        if m.sum() < 5:
            out["bins_feh_xp_on"][tag] = {"n": int(m.sum())}
            out["bins_feh_xp_off"][tag] = {"n": int(m.sum())}
            continue
        out["bins_feh_xp_on"][tag] = _metrics_block(
            y_phys[m], y_pred_phys[m], label_names
        )
        out["bins_feh_xp_off"][tag] = _metrics_block(
            y_phys[m], y_pred_off_phys[m], label_names
        )

    out["bins_parallax_truth_xp_on"] = _bins_true_parallax_quartiles(
        y_phys, y_pred_phys, label_names, "xp_on"
    )
    out["bins_parallax_truth_xp_off"] = _bins_true_parallax_quartiles(
        y_phys, y_pred_off_phys, label_names, "xp_off"
    )

    g_aux = pack.get("test_G_mag")
    out["bins_g_mag_xp_on"] = _quartile_bin_metrics(
        g_aux, y_phys, y_pred_phys, label_names, "g"
    )
    out["bins_g_mag_xp_off"] = _quartile_bin_metrics(
        g_aux, y_phys, y_pred_off_phys, label_names, "g"
    )

    ebv_aux = pack.get("test_ebv")
    out["bins_ebv_xp_on"] = _quartile_bin_metrics(
        ebv_aux, y_phys, y_pred_phys, label_names, "ebv"
    )
    out["bins_ebv_xp_off"] = _quartile_bin_metrics(
        ebv_aux, y_phys, y_pred_off_phys, label_names, "ebv"
    )

    sig_mas = pack.get("target_e_parallax_mas")
    pi_idx = label_names.index("parallax")
    snr_aux = None
    if sig_mas is not None:
        snr_aux = _parallax_snr_test_vector(y_phys[:, pi_idx], sig_mas)
    out["bins_parallax_snr_xp_on"] = _quartile_bin_metrics(
        snr_aux, y_phys, y_pred_phys, label_names, "pisigma"
    )
    out["bins_parallax_snr_xp_off"] = _quartile_bin_metrics(
        snr_aux, y_phys, y_pred_off_phys, label_names, "pisigma"
    )

    os.makedirs(args.out, exist_ok=True)
    out["preprocessing"] = {
        "parallax_target_space": pack.get("parallax_target_space", "linear_mas"),
        "parallax_floor_mas": pack.get("parallax_floor_mas"),
        "astrometry_input_policy": pack.get("astrometry_input_policy", "legacy_raw"),
        "label_scaler": pack.get("label_scaler", "standard"),
        "pert_ebv_scale": float(config["finetuning"].get("pert_ebv_scale", 1.0)),
        "pert_scale": float(config["finetuning"].get("pert_scale", 1.0)),
        "finetune_encoder_lr_effective": float(config["finetuning"]["encoder_lr"])
        if config["finetuning"].get("encoder_lr") is not None
        else float(config["training"]["lr"]),
        "lr_scheduler_encoder_decay": float(
            config["finetuning"].get("lr_scheduler_encoder_decay", 0.95)
        ),
        "lr_scheduler_head_decay": float(
            config["finetuning"].get("lr_scheduler_head_decay", 0.5)
        ),
        "lr_scheduler_head_step_epochs": int(
            config["finetuning"].get("lr_scheduler_head_step_epochs", 10)
        ),
        "scheduler_cosine_t0": int(config["training"].get("scheduler_cosine_t0", 10)),
        "scheduler_cosine_t_mult": int(
            config["training"].get("scheduler_cosine_t_mult", 2)
        ),
        "scheduler_eta_min_factor": float(
            config["training"].get("scheduler_eta_min_factor", 0.01)
        ),
    }

    if args.conformal_json:
        if ensemble_linear:
            print(
                "Warning: --conformal-json ignored for linear_probe checkpoints (no quantile head)."
            )
        else:
            with open(args.conformal_json) as f:
                calib_doc = json.load(f)
            ens_q_on = _ensemble_median_predictions(
                model,
                head,
                loaded_states,
                X_test,
                device,
                args.batch_size,
                linear_probe=False,
                return_full_quantiles=True,
            )
            ens_q_off = _ensemble_median_predictions(
                model,
                head,
                loaded_states,
                X_off,
                device,
                args.batch_size,
                linear_probe=False,
                return_full_quantiles=True,
            )
            apply_cqr_offsets_inplace(ens_q_on, calib_doc)
            apply_cqr_offsets_inplace(ens_q_off, calib_doc)
            phys_on = _inverse_quantile_block(ens_q_on, scalers, pack)
            phys_off = _inverse_quantile_block(ens_q_off, scalers, pack)
            cov_on = interval_coverage(y_phys, phys_on[:, :, 0], phys_on[:, :, 2])
            cov_off = interval_coverage(y_phys, phys_off[:, :, 0], phys_off[:, :, 2])
            out["interval_coverage_xp_on"] = {
                label_names[j]: float(cov_on[j])
                for j in range(len(label_names))
                if np.isfinite(cov_on[j])
            }
            out["interval_coverage_xp_off"] = {
                label_names[j]: float(cov_off[j])
                for j in range(len(label_names))
                if np.isfinite(cov_off[j])
            }
            out["conformal_calibration"] = {
                "path": os.path.abspath(args.conformal_json),
                "alpha": calib_doc.get("alpha"),
                "method": calib_doc.get("method"),
            }

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    write_latex_metrics_table(
        {k: v for k, v in out["global_xp_on"].items()},
        {k: v for k, v in out["global_xp_off"].items()},
        os.path.join(args.out, "metrics_table.tex"),
    )

    print("Wrote", os.path.join(args.out, "metrics.json"))
    print("Wrote", os.path.join(args.out, "metrics_table.tex"))


if __name__ == "__main__":
    main()
