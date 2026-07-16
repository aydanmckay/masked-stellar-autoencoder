#!/bin/bash
# Runs INSIDE the CANFAR session. Launches MSA pretrain.
set -eu
. /etc/astroai-lab/profile.sh

SCRATCH_BASE="/scratch/msa-pretrain"

# pixi/uv default cache dir is /usr/local/share which is not user-writable
export PIXI_CACHE_DIR="/tmp/pixi-cache"
export PIXI_HOME="${SCRATCH_BASE}/.pixi"

cd /srcdir
git clone --depth 1 https://github.com/sfabbro/masked-stellar-autoencoder.git
cd masked-stellar-autoencoder

pixi install -e gpu
# CANFAR doesn't expose __cuda to conda; install CUDA pytorch via pip
pixi run -e gpu pip install torch --index-url https://download.pytorch.org/whl/cu121
mkdir -p "${SCRATCH_BASE}/checkpoints"
nvidia-smi || true
pixi run -e gpu python -u training/pretrain_msa.py --config configs/pretrain.canfar.yaml
