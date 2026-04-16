# Experiment log (multitask rationale, Phase 2)

Record pilot and full-scale results here after each run. **Do not** edit paper tables by hand; copy `results/.../metrics_table.tex` from `training/eval_ensemble.py`.

## Pred-only vs recon+pred (pilot)

**Config:** duplicate `configs/finetune.yaml` → `configs/pilot_local.yaml`; set `finetuning.num_epochs: 5`, `finetuning.ensemble: false`, local paths for `ft_datafile` and `saved_weights`.

| Run ID | multitask | lambda_pred / lambda_rec | max_train_rows | Final val loss (from log) | Notes |
|--------|-----------|--------------------------|----------------|---------------------------|--------|
| A | false | N/A | 8192 | | pred-only |
| B | true | 0.8 / 0.2 | 8192 | | recon+pred |

Commands:

```bash
python training/finetune_msa.py --config configs/pilot_local.yaml --max-train-rows 8192 --max-valid-rows 2048
```

Flip `finetuning.multitask` between runs; keep seed and data identical.

## Full-scale (Phase 3)

| Git tag | Checkpoint paths | eval_ensemble `--out` | Decision (A vs B) |
|---------|------------------|----------------------|-------------------|
| | | | |

## Linear probe baseline

**C0 (linear probe):** set `linearprobe: true`, `multitask: false`, `finetuning.lf: mae` (or `mse`). The wrapper trains a frozen encoder + `nn.Linear` head; checkpoints include `linear_probe: true` and work with `eval_ensemble.py`.
