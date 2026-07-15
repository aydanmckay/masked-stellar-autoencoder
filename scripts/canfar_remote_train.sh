#!/usr/bin/env bash
# Runs INSIDE the CANFAR session. Launches MSA pretrain.
set -euo pipefail
source /etc/astroai-lab/profile.sh

cd /srcdir
git clone --depth 1 https://github.com/sfabbro/masked-stellar-autoencoder.git
cd masked-stellar-autoencoder

SCRATCH_BASE="/scratch/msa-pretrain"

astroai-lab resume msa-gpu
pixi install -e gpu
mkdir -p "${SCRATCH_BASE}/checkpoints"
nvidia-smi || true
pixi run python -u training/pretrain_msa.py --config configs/pretrain.canfar.yaml
