## Training and Fine-tuning Guide

This document explains the training workflow, the rationale for each stage, and how to configure parallax prediction in a principled way.

---

### 1) Pretraining: masked autoencoding

**Goal**
Learn robust representations by reconstructing masked inputs. This makes the encoder resilient to missing or noisy survey measurements.

**Key choices**
- **Masking**: XP coefficients are masked row-wise; photometric bands are masked element-wise.
- **Force-masking**: use `force_mask_cols` to always hide specific columns (e.g., `PARALLAX`) so the model learns to reconstruct them from photometry.
- **Mask indicators**: optional binary mask channels appended to the encoder input to make missingness explicit.
- **Perturbations**: optionally add Gaussian noise scaled by measurement errors (`pert_features`, `pert_scale`).
- **Scaler sampling**: fit the feature scaler using a random subset of rows to avoid loading huge datasets.

**Config knobs (pretrain)**
```
training.xp_masking_ratio
training.m_masking_ratio
training.force_mask_cols
model.use_mask_indicators
training.mask_ranges   # optional overrides for xp_start/xp_end/phot_tail_start
training.pert_features
training.pert_scale
training.scaler_keys
training.scaler_max_rows
training.scaler_seed
training.presaved
```

---

### 2) Fine-tuning: multi-label prediction

**Goal**
Predict stellar labels (Teff, logg, [Fe/H], alpha, age, parallax) with uncertainty estimates.

**Key choices**
- **Quantile regression**: predicts 16th/50th/84th percentiles.
- **Masking in fine-tuning**: keeps the encoder robust to missing data.
- **Leakage-free parallax**: predict parallax from photometry only while still allowing Gaia parallax to improve other labels.
- **Multitask weighting**: scale reconstruction loss relative to label loss when `multitask: true`.

**Config knobs (finetune)**
```
finetuning.lf
finetuning.mask
finetuning.mask_prediction
finetuning.force_mask_cols
finetuning.parallax_use_masked_pred
finetuning.parallax_mle_weight
finetuning.parallax_sigma_scale
finetuning.parallax_sigma_floor
finetuning.multitask_weight
training.mask_ranges   # optional overrides for xp_start/xp_end/phot_tail_start
```

**Mask indicator note**
If `model.use_mask_indicators: true`, the encoder input dimension doubles (`len(feature_cols) * 2`), so
old checkpoints are incompatible and you must pretrain from scratch.

**Mask range auto-detection**
By default, mask ranges are inferred from feature names (e.g., `bp_1` … `rp_55`).
If your feature order differs, set `training.mask_ranges` to override.

---

### 3) Parallax: principled, frequentist strategy

We want to:
1) use Gaia parallax to improve other labels, and
2) predict photometric parallax without leakage.

**Two-pass parallax prediction**
Enable `parallax_use_masked_pred`.
This runs a second forward pass where parallax is masked and replaces the parallax prediction from the main pass.

**MLE regularizer (optional)**
If `parallax_mle_weight > 0`, add a Gaussian-error penalty:
$$
\mathcal{L}_\mathrm{MLE} \propto \frac{(\mu_{\pi,\mathrm{phot}} - \pi_{\mathrm{Gaia}})^2}{\sigma_{\mathrm{Gaia}}^2 + \sigma_{\mathrm{phot}}^2 + \sigma_\mathrm{floor}^2}.
$$
This is frequentist (no priors), and it stabilizes training where Gaia is precise.

**Error inflation**
Because Gaia errors can be underestimated at large distances, we support:
```
parallax_sigma_scale  # multiplies Gaia sigma
parallax_sigma_floor  # adds a systematic floor
```

**Recommended starting values**
```
parallax_use_masked_pred: true
parallax_mle_weight: 0.1
parallax_sigma_scale: 1.0
parallax_sigma_floor: 0.0
```

**Quickstart config (Model B: best labels + leakage‑free parallax)**
```yaml
# pretrain.yaml
model:
  use_mask_indicators: true
training:
  force_mask_cols: ['PARALLAX']

# finetune.yaml
model:
  use_mask_indicators: true
finetuning:
  parallax_use_masked_pred: true
  # force_mask_cols: ['PARALLAX']  # leave off for Model B
  multitask_weight: 1.0
```

**Calibration check (recommended)**
Compute normalized residuals:
$$
z = \frac{\mu_{\pi,\mathrm{phot}} - \pi_{\mathrm{Gaia}}}{\sqrt{\sigma_{\mathrm{Gaia}}^2 + \sigma_{\mathrm{phot}}^2 + \sigma_\mathrm{floor}^2}}.
$$
If `std(z) > 1`, increase `sigma_scale` or `sigma_floor`.
If `std(z) < 1`, decrease them.

---

### 4) BLUE fusion (post-training)

For a final parallax estimate that smoothly transitions between Gaia and photometry, use inverse-variance weighting:
$$
\pi_\mathrm{fused} = \frac{\mu_{\pi,\mathrm{phot}}/\sigma_{\mathrm{phot}}^2 + \pi_{\mathrm{Gaia}}/\sigma_{\mathrm{Gaia}}^2}{1/\sigma_{\mathrm{phot}}^2 + 1/\sigma_{\mathrm{Gaia}}^2}.
$$

This is independent of training and can be applied at inference time.

---

### 5) Practical training recipe

1) Pretrain with `force_mask_cols: ['PARALLAX']` so the model learns to reconstruct parallax from photometry.
2) Fine-tune with `parallax_use_masked_pred: true`.
3) Turn on `parallax_mle_weight` only if you want Gaia-anchored regularization (start small).
4) Calibrate sigma inflation on a validation set.
