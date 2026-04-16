#!/bin/bash
#SBATCH --job-name=msa_finetune
#SBATCH --account=def-USERNAME_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=185G
#SBATCH --gpus=a100:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/msa_finetune_%j.out
#SBATCH --error=slurm_logs/msa_finetune_%j.err
##SBATCH --mail-user=you@institution.ca
##SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

export MSA_REPO="${MSA_REPO:-$SLURM_SUBMIT_DIR}"
export SCRATCH="${SCRATCH:-$HOME/scratch}"
export MSA_VENV="${MSA_VENV:-$HOME/venvs/msa}"

module load python/3.11
module load cuda
module load cudnn

# shellcheck source=/dev/null
source "${MSA_REPO}/batch_scripts/env_narval.sh"

echo "Job $SLURM_JOB_ID  node $(hostname)"
nvidia-smi || true

CONFIG="${CONFIG:-configs/finetune.narval.example.yaml}"
python -u training/finetune_msa.py --config "$CONFIG"
