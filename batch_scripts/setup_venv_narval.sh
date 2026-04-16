#!/bin/bash
# Create a venv on Narval (or any Alliance GPU node) with CUDA-enabled PyTorch.
#
# 1. module load a compiler, Python, CUDA, and cuDNN (versions from `module spider`).
# 2. Run from repo root:
#      bash batch_scripts/setup_venv_narval.sh /scratch/$USER/venvs/msa
#
# Then in env_narval.sh or your job script:
#      export MSA_VENV=/scratch/$USER/venvs/msa
#
# PyTorch wheels must match the CUDA version of the loaded module; see
# https://docs.alliancecan.ca/wiki/PyTorch

set -euo pipefail
VENV_TARGET="${1:?Usage: $0 /path/to/venv}"

python3 -m venv "$VENV_TARGET"
# shellcheck source=/dev/null
source "$VENV_TARGET/bin/activate"
pip install --upgrade pip wheel

# Install PyTorch first (CPU/CUDA build per your site instructions), then the rest.
pip install torch --index-url https://download.pytorch.org/whl/cu124 || pip install torch

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pip install -r "$REPO_ROOT/requirements.txt"

echo "Done. Activate with: source $VENV_TARGET/bin/activate"
echo "Set MSA_VENV=$VENV_TARGET in your job environment."
