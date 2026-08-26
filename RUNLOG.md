# MSA run log and paper–code gap audit

**Canonical codebase for new work:** `masked-stellar-autoencoder/` (this repo).
**Legacy / alternate:** `~/src/msa/` (`msa.py`, `msa_preprocess.py`, `msa_eval.py`) — different module layout; **not** assumed to match published paper numbers.

## Status: published Overleaf results vs this repo

| Claim (paper / figures) | Reproducible from this repo? | Notes |
|---------------------------|------------------------------|--------|
| Table metrics (RMSE/MAE/R², XP on/off) | **Unverified** | No commit tagged; configs point to `/arc/...` paths not portable. |
| Figure filenames (`performancetestset_0605.png`, etc.) | **Unverified** | Generated outside this tree or from older scripts. |
| Pre-train 80 epochs, fine-tune 100, ensemble 20 | **Partially** | `finetune.yaml` has `ensemble: false` by default; `ensemble_size: 20` used only if `ensemble: true`. |
| λ_pred=0.8, λ_rec=0.2 | **Yes (current code)** | Wired in `TabResnetWrapper.fit`; earlier paper run may pre-date this. |
| Masking: XP 90%, ancillary 60% | **Yes** | `xp_masking_ratio`, `m_masking_ratio` in YAML. |
| `feature_cols` layout vs `_apply_mask` (5:115) | **OK if ancillary order matches** | XP block is indices **5:114** inclusive (110 cols). Pretrain appends **RA, DEC** after finetune columns; leading **138** columns match finetune layout, so indices through **PARALLAX** are unchanged. |

## Hyperparameters (reference: current YAML)

- **Pretrain:** [configs/pretrain.yaml](configs/pretrain.yaml) — `layer_dims`, `rtdl_embed: 16`, MAE recon, batch size, AdamW, cosine warm restarts.
- **Finetune:** [configs/finetune.yaml](configs/finetune.yaml) — quantile head, `lambda_pred` / `lambda_rec`, optional `quantile_label_weights`, `metal_poor` curation block.

## Checkpoint keys

- **Loading:** use `models/checkpoint_load.torch_load_trusted` (or equivalent `weights_only=False` on PyTorch 2.6+) so older checkpoints that embed NumPy metadata still load.
- **Fine-tune encoder LR:** `finetuning.encoder_lr` (or `null` → `training.lr`); LambdaLR and cosine restart knobs are documented in `docs/METHODOLOGY.md`.
- **Fine-tune saves** (`TabResnetWrapper.fit`): `autoencoder_state_dict`, `prediction_head_state_dict`, optional `linear_probe` (bool).
- **Pretrain saves** (`pretrain_msa.py`): `model_state_dict` only.
- **`eval_ensemble.py`** loads the encoder from either `autoencoder_state_dict` **or** `model_state_dict`, but still **requires** `prediction_head_state_dict` (fine-tuned run).

## Data paths (placeholders in repo)

- Pretrain HDF5: `data.datafile` in `pretrain.yaml` (e.g. `sslset-realmags-full-052725.h5`).
- Fine-tune FITS: `data.ft_datafile` in `finetune.yaml` (e.g. `ftset_spec_ga_0602_realmags.fits`).

Replace with local paths and record the **exact file versions** (MD5 or archive row) in this table when you freeze the paper run.

## Methodology design record

Parallax / distance / astrometry policy and ablation checklist: **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**. Each production run should log frozen `preprocessing.*` YAML keys there and in this file.

**Eval bins** (from `training/eval_ensemble.py` → `metrics.json`): `bins_feh_*`, `bins_g_mag_*`, `bins_ebv_*`, `bins_parallax_snr_*` (ϖ/σ), `bins_parallax_truth_*`, plus global and XP-off blocks. Use these for dust and faint-end claims, not global MAE alone.

## Narval (Alliance Canada)

Slurm templates: `batch_scripts/narval_*.slurm`, `batch_scripts/env_narval.sh`, example YAML `configs/pretrain.narval.example.yaml` and `configs/finetune.narval.example.yaml`. Training code expands `$SCRATCH` in paths via `training/config_paths.py`.

## Next steps (see `docs/experiment_matrix.md`)

1. Phase 1 smoke tests: `pytest tests/` (on PEP 668–managed Pythons, create `.venv` and `pip install -r requirements-dev.txt` plus PyTorch).
2. Pilot ablations: `python training/pilot_objectives.py` (when FITS available)
3. Full train + `python training/eval_ensemble.py` → JSON + LaTeX fragments
4. Git tag matching paper submission after numbers are frozen
