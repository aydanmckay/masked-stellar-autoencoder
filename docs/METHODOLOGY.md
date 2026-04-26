# MSA methodology design record

This file is the **canonical place** to justify preprocessing, astrometry, and distance handling. Each production run should append a row to [RUNLOG.md](../RUNLOG.md) with **git tag**, **config hash**, and pointer here.

## Parallax, proper motions, and distances (frozen policy)

### Statistical framing

- **Goal:** predict a **spectro-photometrically refined** parallax (or its log) using XP, multi-survey photometry, line-of-sight reddening, and astrometry—not to ignore Gaia.
- **Conditioning on noisy Gaia parallax is not “double counting”** in the Bayesian sense: the model approximates a conditional distribution **p(refined labels | all inputs including π_obs, σ_π, …)**. The subtlety is **architectural**: if the **same scalar** π_obs (mas) is both the dominant **input** and the **regression target**, a flexible MLP can approximate the identity map and under-use spectra.
- **Chosen mitigation (default for new runs):**
  1. **Inputs:** keep **pmra, pmdec** and their uncertainties. For the **parallax feature slot**, use **ϖ / max(σ_π, ε)** clipped to ±`astrometry_snr_cap` instead of raw ϖ (mas). That preserves **signal-to-noise** information for **negative and low-S/N** parallaxes without forcing a nonnegative input scale. The associated error column for that slot is set to **1.0** so feature augmentation noise is not interpreted as mas.
  2. **Target:** train the head on **log₁₀(ϖ/mas)** for training stars with **ϖ > 0**; rows with **ϖ ≤ 0** receive **NaN** on that label so `quantile_loss` masks them out. Label uncertainties use a **delta-method** σ on log₁₀ ϖ from σ_π.
  3. **Catalogue output:** **inverse transform** log target to **̂ (mas)**; publish **distance** as **1 / max(̂, ϵ)** with ϵ from config. **No separate distance head** unless a future ablation shows a gain on validation in faint **ϖ/σ** bins.

### Alternatives logged for ablation

| Variant | When to prefer |
|---------|----------------|
| `legacy_raw` input + `linear_mas` target | Reproducing older runs / minimal code path. |
| SNR input + `log10_mas` target | **Recommended** for new science: better dynamic range and reduced identity mapping. |
| Drop parallax from inputs entirely | If ablations show SNR+PM suffice; risks losing bright-star leverage. |

### YAML keys

```yaml
preprocessing:
  astrometry_input_policy: snr_clipped   # or legacy_raw
  astrometry_snr_cap: 10.0
  parallax_target_space: log10_mas     # or linear_mas
  parallax_floor_mas: 1.0e-4
```

## Proper motions

Always **inputs** (not duplicated as prediction targets): they provide **tangential kinematics** orthogonal to a single scalar distance and help disentangle populations.

## Dust and extinction (explicit E(B-V) path)

### Policy

1. **Primary:** keep **E(B-V)** (Schlegel / `dustmaps`-style, line-of-sight, distance-independent) as an **explicit input** so the encoder can condition XP and photometry on known reddening. This matches the manuscript feature list (`EBV` in `feature_cols`).
2. **Implicit capacity:** the MLP may still fit **residual** extinction and model mismatch; **do not** treat that as a substitute for the explicit channel. Ablations that **drop E(B-V)** must report metrics in **E(B-V) quartile bins** (from `eval_ensemble.py` → `bins_ebv_*` in `metrics.json`) and, where relevant, Galactic latitude vs plane.
3. **Future:** 3D dust maps are optional extra channels only after the same hypothesis / baseline / binned-evidence / freeze template.

### Data flow

- Pretrain tables: E(B-V) is joined in the catalogue builder (e.g. `data/pretraining-partial-table-maker.py` using `dustmaps`).
- Fine-tune FITS: the same `EBV` column must align with `error_cols` (uncertainty used for scaling if `pert_features` is on; jitter on E(B-V) itself is scaled by `pert_ebv_scale`).

## Noisy features (training-time augmentation)

### Current behaviour

- **Fine-tune:** if `finetuning.pert_features` is true, Gaussian noise **∝** `pert_scale` × scaled error tensor `eX` (errors divided by `RobustScaler` IQR per feature after `prepare_finetune_arrays`). Default `finetuning.pert_scale: 1.0`. **Pretrain** uses `training.pert_scale`.
- **Per-feature scale:** `pert_ebv_scale` (finetune and pretrain YAML) multiplies **only** the E(B-V) channel’s augmentation noise (default **1.0**; set **0.0** to freeze reddening inputs under jitter).
- **Pretrain:** `training.pert_features` / `training.pert_scale` / `training.pert_ebv_scale` in `TabResnetWrapper` + `pretrain_hdf`.

### Missing feature uncertainties

- **Imputation:** in `finetune_data.py`, missing errors are filled with **per-column 90th percentile**, then **median**, then **1.0**. This is a **missing-data prior**; sensitivity tests should include worst-case uniform errors or dropping uncertain bands (log in RUNLOG when you change it).

### Reconstruction weighting (pretrain)

- Loss `wmse` / `wmae` uses propagated **feature uncertainties** in the reconstruction term (see `EncoderDecoderLoss` in `models/model.py`). Motivate stronger per-band weighting only if ablations show one family of coefficients dominating the MAE.

## Dynamic range (overview)

- Features: **RobustScaler** after optional per-column transforms (future work: asinh on XP).
- Labels: first five targets use **StandardScaler** or optional **RobustScaler** (`preprocessing.label_scaler`); parallax follows `parallax_target_space`.

## Catalogue release (minimum columns)

| Column | Description |
|--------|-------------|
| `source_id` | Gaia DR3 identifier (or survey key) |
| `embedding_*` | Latent vector (versioned scaler + model tag) |
| `teff_med`, … | Median predictions; optional `_q16`, `_q84` |
| `parallax_med_mas` | In mas after inverse label transform |
| `distance_pc` | `1 / max(parallax_med_mas, epsilon_mas)` in pc (document ϵ) |
| `neg_parallax_train` | Always false at inference; training-only flags not shipped |
| `preprocessing_tag` | Hash or tag matching `preprocessing` YAML block |

## Quantile fine-tuning: heteroscedastic pinball (implemented)

- Optional **per-sample, per-label** weights in scaled label-error space: $w_{b\ell} \propto 1 / \max(\sigma_{b\ell}^2 + \epsilon^2, \text{floor}^2)$, clamped, with optional **batch mean normalization** so scale stays comparable to unweighted training (`finetuning.quantile_use_label_errors` and related keys in `configs/finetune.yaml`).

## Conformal / CQR-style intervals (implemented)

- **Calibration:** on a held-out set in **scaled label space**, compute asymmetric offsets so nominal quantile intervals satisfy split conformal coverage at level $1-\alpha$ (`training/conformal.py`, `training/calibrate_conformal.py` on `.npy` val arrays).
- **Evaluation:** `training/eval_ensemble.py --conformal-json …` applies offsets to ensemble **q16/q84**, then reports **interval coverage** per label in `metrics.json` (median point metrics unchanged).

## Pretrain mask mixture (implemented)

- `training.mask_mixture_xp_full_frac`: fraction of rows per forward pass with XP coefficients **forced fully masked** on top of the usual row-wise XP mask—mixes toward XP-off behaviour without replacing the default regime.

## Learning rates and schedulers (implemented)

### Pretrain (`TabResnetWrapper.pretrain_hdf`)

- **Base LR:** `training.lr` on all non-bias / non-norm weights; **bias + norm** (`LayerNorm`/`BatchNorm` affine, matched via parameter name substring `norm`) use **zero weight decay**.
- **Optimiser:** `training.optimizer` → Adam, AdamW, or SGD (`momentum=0.9`).
- **Scheduler:** `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` with **`scheduler.step()` once per pretrain epoch** (after all HDF keys in the epoch). YAML keys (defaults match historical hard-codes):
  - `training.scheduler_cosine_t0` (default **10**) — first restart period in **epochs**.
  - `training.scheduler_cosine_t_mult` (default **2**) — multiplies restart spacing after each restart.
  - `training.scheduler_eta_min_factor` (default **0.01**) — `eta_min = lr ×` this factor.

### Fine-tune (`TabResnetWrapper.fit`, non–linear-probe)

- **Head base LR:** `finetuning.lr` (AdamW `weight_decay = finetuning.l2` on the head group only).
- **Encoder base LR:** `finetuning.encoder_lr` if set to a **number**; if **`null` / omitted**, uses **`training.lr`** (the same value passed into `TabResnetWrapper` as `lr=` from `finetune_msa.py`). This removes the old hard-coded `1e-5` mismatch with YAML.
- **LambdaLR** (per epoch, PyTorch passes `epoch` starting at 0):
  - **Encoder group:** multiplies base LR by **`lr_scheduler_encoder_decay` ^ epoch** (default **0.95**).
  - **Head group:** multiplies base LR by **`lr_scheduler_head_decay` ^ (epoch // lr_scheduler_head_step_epochs)`** (defaults **0.5** and **10**). `lr_scheduler_head_step_epochs` is clamped to **≥ 1**.

Linear probe: encoder frozen; only the probe uses `finetuning.lr` and the **head** schedule above (single param group).

### Activations (reminder)

- **Backbone:** `model.pt_activ_func` (`elu` / `relu` / `gelu`) for encoder and decoder `ResBlock`s.
- **Prediction head:** `finetuning.active` — may differ from the backbone (default trunk ELU, head ReLU in `finetune.yaml`).

## Architecture and optimisation (validation protocol)

**Scope:** latent width, depth (`model.layer_dims`, `rtdl_embed`, `norm`), decoder asymmetry (`decoder_dims`), pretrain and finetune learning rates / weight decay / optimiser, masking fractions (`xp_masking_ratio`, `m_masking_ratio`, `mask_mixture_xp_full_frac`), multitask weights (`lambda_pred`, `lambda_rec`), quantile heads and optional `quantile_label_weights` / σ pinball, and **LR schedule keys** above.

**Protocol (freeze only after this):**

1. Fix **data split** and **preprocessing** keys (`preprocessing.*`, `metal_poor.*` if used).
2. For each candidate setting, run **validation loss** (as logged by `finetune_msa`) and export **`eval_ensemble.py` metrics** with the same checkpoints.
3. **Pre-registered bins:** report at least **[Fe/H]** (`bins_feh_*`), **XP on/off**, **G mag quartiles** (`bins_g_mag_*`), **E(B-V) quartiles** (`bins_ebv_*`), **ϖ/σ quartiles** (`bins_parallax_snr_*`), and parallax-truth quartiles (`bins_parallax_truth_*`). Do not cherry-pick a single global MAE.
4. **Grid size:** start with **one knob at a time** or a **small factorial** (e.g. 2×2 on LR × `lambda_rec`); expand only when a clear interaction is hypothesised.
5. **Freeze:** record winning YAML values, git commit, and a row in `RUNLOG.md`.

| Knob (YAML) | Typical baseline | What to watch on val |
|-------------|------------------|----------------------|
| `model.layer_dims` / `rtdl_embed` | current default | Under/over-fit vs XP-off and faint G |
| `training.lr` / `weight_decay` | AdamW defaults in YAML | Stability with `pert_features` |
| `finetuning.lr` / `l2` / `encoder_lr` | current default | Head vs encoder drift; `encoder_lr: null` ties encoder to `training.lr` |
| `lr_scheduler_*` | 0.95 / 0.5 / 10 | Speed of encoder decay vs head step decay |
| `scheduler_cosine_*` (pretrain) | 10 / 2 / 0.01 | Restart cadence vs train stability |
| `xp_masking_ratio` / `m_masking_ratio` | 0.9 / 0.6 | XP-off metrics and recon loss |
| `lambda_pred` / `lambda_rec` | 0.8 / 0.2 | Multitask trade-off, spectroscopic bins |
| `mask_mixture_xp_full_frac` | 0 | XP-off generalisation vs train loss |

## Further work (training / optimisation — not implemented)

Use when you outgrow the current knobs:

- **Encoder weight decay** in fine-tune (today: encoder group has `weight_decay=0`; only the head uses `finetuning.l2`).
- **Independent schedulers** per param group (e.g. cosine for encoder, plateau for head) instead of shared `LambdaLR`.
- **Warmup** epochs before applying `LambdaLR` multipliers.
- **One-cycle** or **cosine annealing without restarts** for fine-tune, with pre-registered val + binned metrics.
- **Telemetry:** log effective LR per group each epoch (file or W&B) so frozen runs are auditable from artefacts alone.

## Checklist for each new knob

1. Hypothesis
2. Simpler baseline
3. Pre-registered val + **binned** metrics
4. Frozen YAML value + tag
