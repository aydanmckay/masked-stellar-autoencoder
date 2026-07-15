#!/usr/bin/env bash
# One-time CANFAR setup: install MSA environment + stage 443 GB training data
# Run from your laptop. Requires: canfar CLI authenticated to staging.
set -euo pipefail

REPO="sfabbro/masked-stellar-autoencoder"
ARC_BASE="/arc/projects/k-pop/msa_pretrain"
DATA_SRC="/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5"
SCRATCH_BASE="/scratch/msa-pretrain"

canfar login cadc --dev
canfar server use staging

echo "Creating setup session (CPU-only, for environment install + data staging)..."
SETUP_ID=$(canfar create --name msa-setup --json \
  --cpu 8 --memory 32 \
  headless astroai/webterm:latest \
  -- bash -c '
    set -euo pipefail

    # Clone repo
    cd /srcdir
    git clone --depth 1 https://github.com/'"$REPO"'.git
    cd masked-stellar-autoencoder

    # Install pixi environment (GPU resolves CUDA from conda-forge on CANFAR)
    if nvidia-smi >/dev/null 2>&1; then
      echo "CUDA detected, installing GPU environment..."
      pixi install -e gpu
    else
      echo "WARNING: No CUDA detected — installing CPU-only environment"
      pixi install -e default
    fi

    # Save full environment to /arc for fast restore in future sessions
    mkdir -p /arc/projects/k-pop
    astroai-lab save msa-gpu --full

    # Create output directories
    mkdir -p '"$ARC_BASE"'/checkpoints '"$ARC_BASE"'/plots

    # Stage 443 GB training data to /scratch (fast local NVMe)
    mkdir -p '"$SCRATCH_BASE"'
    astroai-lab data stage '"$DATA_SRC"' '"$SCRATCH_BASE"'/data.h5

    echo "=== SETUP COMPLETE ==="
  ')

echo "Setup session: $SETUP_ID"
echo "Monitor progress: canfar logs $SETUP_ID"
echo "Wait for '=== SETUP COMPLETE ===' before running canfar_launch.sh"
