#!/bin/bash
# shellcheck disable=SC2034
# Source from Slurm jobs after modules:  source batch_scripts/env_narval.sh
#
# Customize MSA_REPO and (optionally) SCRATCH before submitting.

export MSA_REPO="${MSA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
export SCRATCH="${SCRATCH:-$HOME/scratch}"

# HuggingFace / weights cache on scratch (optional)
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"

# Weights & Biases: offline on compute nodes without egress
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$MSA_REPO" || exit 1
export PYTHONPATH="$MSA_REPO:${PYTHONPATH:-}"

# Python venv (create with batch_scripts/setup_venv_narval.sh)
if [[ -n "${MSA_VENV:-}" && -f "$MSA_VENV/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$MSA_VENV/bin/activate"
fi

mkdir -p slurm_logs "$SCRATCH/msa/runs/pretrain" "$SCRATCH/msa/runs/ft" "$SCRATCH/msa/checkpoints" results
