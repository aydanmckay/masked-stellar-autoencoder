#!/bin/bash
# Prefer batch_scripts/narval_pretrain.slurm for new Narval runs (array looping optional).
#SBATCH --job-name=pretrain_model_loop
#SBATCH --account=account-username
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Array Configuration ---
#SBATCH --array=21-100%1
#SBATCH --output=slurm_logs/train_job_%A_task_%a.out
#SBATCH --error=slurm_logs/train_job_%A_task_%a.err
#---------------------------

echo "--- Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID ---"

module load python/3.11
source ~/load/environment/with/libraries

cd ~/scratch/masked-stellar-autoencoder/

python training/pretrain_msa.py --config configs/pretrain.yaml
