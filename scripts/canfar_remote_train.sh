#!/usr/bin/env bash
# Runs INSIDE the CANFAR session. Launches MSA pretrain.
set -euo pipefail
source /etc/astroai-lab/profile.sh

SCRATCH_BASE="/scratch/msa-pretrain"

cd /srcdir/masked-stellar-autoencoder
astroai-lab resume msa-gpu
pixi install -e gpu
mkdir -p "${SCRATCH_BASE}/checkpoints"
nvidia-smi || true
pixi run python -u training/pretrain_msa.py --config configs/pretrain.canfar.yaml
