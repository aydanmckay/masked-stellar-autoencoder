#!/bin/bash
#SBATCH --job-name=finetune_model
#SBATCH --account=account-username
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=slurm_logs/finetune_model_init_%j.out
#SBATCH --error=slurm_logs/finetune_model_init_%j.err
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=BEGIN,END,FAIL

echo "--- Starting solo job with ID: $SLURM_JOB_ID ---"

module load python/3.11
source ~/load/environment/with/libraries

cd ~/scratch/masked-stellar-autoencoder/

python training/finetune_msa.py --config configs/finetune.yaml
