#!/usr/bin/env python3
"""
Pilot helper: load fine-tuning data, print summary stats, optionally compare multitask flags.

Does not run full training unless you invoke finetune_msa separately with pilot YAML.

  PYTHONPATH=. python training/pilot_objectives.py --config configs/finetune.yaml
  PYTHONPATH=. python training/pilot_objectives.py --config configs/finetune.yaml --max-train-rows 4096

For multitask vs pred-only comparison, run twice with the same config (e.g. a copy of
``configs/finetune.yaml``) and only change ``finetuning.multitask`` (and set
``lambda_rec: 0`` / ``lambda_pred: 1`` for a clean pred-only pilot if desired):

  PYTHONPATH=. python training/finetune_msa.py --config configs/finetune.yaml --max-train-rows 8192 --max-valid-rows 2048

Compare validation loss in the fine-tune log; record numbers in ``docs/EXPERIMENT_LOG.md``.
"""

import argparse
import os
import sys

import numpy as np
import yaml

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_paths import expand_config_paths
from finetune_data import prepare_finetune_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-valid-rows", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    expand_config_paths(config)

    path = config["data"].get("ft_datafile", "")
    if not path or not os.path.isfile(path):
        print(f"pilot_objectives: FITS not found at {path!r}; skipping data load.")
        print(
            "Record multitask comparison in docs/EXPERIMENT_LOG.md when data are available."
        )
        return 0

    pack = prepare_finetune_arrays(
        config,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
    )
    feh = pack["train_feh_raw"]
    feh = feh[np.isfinite(feh)]
    print("Pilot data summary")
    print(
        "  train rows:",
        pack["trainset"].shape[0],
        " valid:",
        pack["validset"].shape[0],
        " test:",
        pack["testset"].shape[0],
    )
    print(
        "  [Fe/H] train percentiles (raw): p10=%.3f p50=%.3f p90=%.3f"
        % tuple(np.percentile(feh, [10, 50, 90]))
    )
    print("  [Fe/H] train fraction < -2:", float(np.mean(feh < -2)))
    print("  finetuning.multitask in config:", config["finetuning"].get("multitask"))
    print(
        "  lambda_pred, lambda_rec:",
        config["finetuning"].get("lambda_pred"),
        config["finetuning"].get("lambda_rec"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
