#!/usr/bin/env bash
# Launch MSA pretrain + marimo monitor on CANFAR staging
# Run from your laptop after canfar_setup.sh has completed.
set -euo pipefail

SCRATCH_BASE="/scratch/msa-pretrain"

canfar login cadc --dev
canfar server use staging

# ── Training session (headless, GPU) ──
echo "Launching training session (16 cores, 64 GB, 1 GPU)..."
TRAIN_ID=$(canfar create --name msa-pretrain --json \
  --cpu 16 --memory 64 --gpu 1 \
  headless astroai/webterm:latest \
  -- bash -c '
    set -euo pipefail
    cd /srcdir/masked-stellar-autoencoder
    astroai-lab resume msa-gpu
    pixi install -e gpu
    mkdir -p '"$SCRATCH_BASE"'/checkpoints
    nvidia-smi || true
    python -u training/pretrain_msa.py --config configs/pretrain.canfar.yaml
  ')

# ── Marimo monitor (contributed, no GPU) ──
echo "Launching marimo monitor..."
MON_ID=$(canfar create --name msa-monitor --json \
  --cpu 2 --memory 8 \
  contributed astroai/marimo:latest)

echo ""
echo "=== MSA Pretrain on CANFAR Staging ==="
echo "Training: $TRAIN_ID"
echo "Monitor:  $MON_ID  ->  canfar open $MON_ID"
echo "Logs:     canfar logs $TRAIN_ID"
echo ""
echo "To stop both: canfar delete $TRAIN_ID $MON_ID"
