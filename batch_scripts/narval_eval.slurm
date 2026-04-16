#!/bin/bash
#SBATCH --job-name=msa_eval
#SBATCH --account=def-USERNAME_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/msa_eval_%j.out
#SBATCH --error=slurm_logs/msa_eval_%j.err

set -euo pipefail

export MSA_REPO="${MSA_REPO:-$SLURM_SUBMIT_DIR}"
export SCRATCH="${SCRATCH:-$HOME/scratch}"
export MSA_VENV="${MSA_VENV:-$HOME/venvs/msa}"

module load python/3.11
module load cuda
module load cudnn

# shellcheck source=/dev/null
source "${MSA_REPO}/batch_scripts/env_narval.sh"

CONFIG="${CONFIG:-configs/finetune.narval.example.yaml}"
OUT="${OUT:-$SCRATCH/msa/runs/eval/run_${SLURM_JOB_ID}}"
GLOB="${EVAL_CKPT_GLOB:-$SCRATCH/msa/runs/ft/masked_stellar_autoencoder_ft_seed*.pth}"

shopt -s nullglob
CKPTS=( $GLOB )
shopt -u nullglob

if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No checkpoints matched: GLOB=$GLOB"
  echo "Set EVAL_CKPT_GLOB to a shell glob, or run eval interactively with explicit --checkpoints."
  exit 1
fi

python -u training/eval_ensemble.py --config "$CONFIG" --checkpoints "${CKPTS[@]}" --out "$OUT"
echo "Wrote metrics under $OUT"
