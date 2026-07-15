#!/usr/bin/env bash
# Launch MSA pretrain + marimo monitor on CANFAR staging
# Run from your laptop after canfar_setup.sh has completed.
set -euo pipefail

SCRATCH_BASE="/scratch/msa-pretrain"

canfar login cadc --dev
canfar server use staging

# ── Training session (headless, GPU) ──
echo "Launching training session (16 cores, 64 GB, 1 GPU)..."
canfar create -n msa-pretrain -c 16 -m 64 -g 1 \
  headless astroai/webterm:latest \
  -- bash -c '
    set -euo pipefail
    source /etc/astroai-lab/profile.sh
    cd /srcdir/masked-stellar-autoencoder
    astroai-lab resume msa-gpu
    pixi install -e gpu
    mkdir -p '"$SCRATCH_BASE"'/checkpoints
    nvidia-smi || true
    pixi run python -u training/pretrain_msa.py --config configs/pretrain.canfar.yaml
  '

TRAIN_ID=$(canfar ps -q -n msa-pretrain | head -1)

# ── Marimo monitor (contributed, no GPU) ──
echo "Launching marimo monitor..."
canfar create -n msa-monitor -c 2 -m 8 \
  contributed astroai/marimo:latest

MON_ID=$(canfar ps -q -n msa-monitor | head -1)

echo ""
echo "=== MSA Pretrain on CANFAR Staging ==="
echo "Training: $TRAIN_ID"
echo "Monitor:  $MON_ID  ->  canfar open $MON_ID"
echo "Logs:     canfar logs $TRAIN_ID"
echo ""
echo "To stop both: canfar delete $TRAIN_ID $MON_ID"
