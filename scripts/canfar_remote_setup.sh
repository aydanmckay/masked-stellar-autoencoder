#!/bin/bash
# Runs INSIDE the CANFAR session. Installs MSA env + stages data.
set -eu
. /etc/astroai-lab/profile.sh

# pixi/uv default cache dir is /usr/local/share which is not user-writable
export PIXI_CACHE_DIR="/tmp/pixi-cache"
export PIXI_HOME="/scratch/msa-pretrain/.pixi"

REPO="sfabbro/masked-stellar-autoencoder"
ARC_BASE="/arc/projects/k-pop/msa_pretrain"
DATA_SRC="/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"
SCRATCH_BASE="/scratch/msa-pretrain"

cd /srcdir
git clone --depth 1 "https://github.com/${REPO}.git"
cd masked-stellar-autoencoder

if nvidia-smi >/dev/null 2>&1; then
  echo "CUDA detected, installing GPU environment..."
  pixi install -e gpu
else
  echo "WARNING: No CUDA detected — installing CPU-only environment"
  pixi install -e default
fi

mkdir -p /arc/projects/k-pop
astroai-lab save msa-gpu --full

mkdir -p "${ARC_BASE}/checkpoints" "${ARC_BASE}/plots"

mkdir -p "${SCRATCH_BASE}"
astroai-lab data stage "${DATA_SRC}" "${SCRATCH_BASE}/data.h5"

echo "=== SETUP COMPLETE ==="
