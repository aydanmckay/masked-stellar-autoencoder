#!/bin/bash
#SBATCH --job-name=msa_pretrain
#SBATCH --account=def-USERNAME_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=185G
#SBATCH --gpus=a100:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/msa_pretrain_%j.out
#SBATCH --error=slurm_logs/msa_pretrain_%j.err
##SBATCH --mail-user=you@institution.ca
##SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# Repo checkout location (default: directory you submitted from)
export MSA_REPO="${MSA_REPO:-$SLURM_SUBMIT_DIR}"
# Alliance scratch (override if your default differs)
export SCRATCH="${SCRATCH:-$HOME/scratch}"
export MSA_VENV="${MSA_VENV:-$HOME/venvs/msa}"

# Inspect with: module spider cuda cudnn python
module load python/3.11
module load cuda
module load cudnn

# shellcheck source=/dev/null
source "${MSA_REPO}/batch_scripts/env_narval.sh"

echo "Job $SLURM_JOB_ID  node $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

CONFIG="${CONFIG:-configs/pretrain.narval.example.yaml}"
python -u training/pretrain_msa.py --config "$CONFIG"
