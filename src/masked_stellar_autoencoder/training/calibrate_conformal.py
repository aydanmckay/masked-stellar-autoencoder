#!/usr/bin/env python3
"""
Calibrate CQR-style interval offsets from numpy arrays (scaled label space).

Example:
  python training/calibrate_conformal.py \\
    --y-val path/to/y_val.npy --pred-val path/to/pred_val.npy \\
    --alpha 0.1 --out conformal.json

Arrays: y_val (N, L), pred_val (N, L, 3) lower/median/upper from the same scaler as training.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .conformal import calibrate_cqr_offsets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--y-val", required=True, help="Calibration labels (N, L) float32/64 .npy"
    )
    ap.add_argument("--pred-val", required=True, help="Val predictions (N, L, 3) .npy")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    y_val = np.load(args.y_val)
    pred_val = np.load(args.pred_val)
    doc = calibrate_cqr_offsets(y_val, pred_val, alpha=args.alpha)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
