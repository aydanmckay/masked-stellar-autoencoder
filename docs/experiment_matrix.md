# MSA experiment matrix (Phase 2)

Pre-register decisions **before** full 220M pretrain / large ensembles. All rows use the **same** random train/val/test split (`random_state=42` as in `finetune_msa.py`) unless you explicitly change seeds for robustness.

## Primary readouts (test set)

| Readout | Definition |
|---------|------------|
| Global | MAE, RMSE, R² (and NMAD) per label: Teff, log g, [Fe/H], [α/Fe], age, parallax |
| [Fe/H] bins | e.g. [Fe/H] > −1, [−2, −1], < −2 (same stars, sliced) |
| XP mask | Full forward with XP zeroed/masked vs nominal (same stars) |
| G bins | Optional: faint vs bright using column `G` (or BP/RP proxy) |

Export via `python training/eval_ensemble.py` (see `--help`).

## Training axes

| ID | Encoder init | Fine-tune objective | Notes |
|----|--------------|---------------------|--------|
| A0 | Random | Prediction only (`multitask: false`) | Baseline: no recon term in FT |
| A1 | Pretrained | Prediction only | Is SSL encoder useful without recon loss? |
| B0 | Pretrained | Recon + pred (`multitask: true`, λ_p=0.8, λ_r=0.2) | Paper default hypothesis |
| B1 | Pretrained | Recon + pred, λ grid | e.g. (0.9,0.1), (0.7,0.3) on **val** only |
| C0 | Pretrained | Linear probe | `linearprobe: true`, `multitask: false`, `lf: mae` or `mse` |

**Decision rule:** Choose B vs A for production if B wins on **val** global + [Fe/H] < −2 bin + XP-off MAE for [Fe/H] (or pre-registered subset). If tied, prefer **simpler** (A) and report “no gain from joint recon”.

## Masking at fine-tune

| Variant | Config |
|---------|--------|
| Match pretrain | `mask: true`, same `xp_masking_ratio` / `m_masking_ratio` as pretrain YAML |
| Ablate | `mask: false` for one pilot row to measure sensitivity |

## Ensemble

After objective is fixed: `finetuning.ensemble: true`, `ensemble_size: 20`, fixed `ensemble_seed`. Report mean ± std of metrics across members or bag medians.

## Pilot scale (before full GPU)

Use `training/pilot_objectives.py`: few epochs, subset of batches, same config paths — to verify loss decreases and val is finite.

## Full scale (Phase 3)

1. Pretrain checkpoint (or document frozen `.pth` path).
2. Fine-tune per chosen row.
3. `eval_ensemble.py` → `results/metrics.json` + `results/metrics_table.tex`.
4. Git **tag** = paper submission.
