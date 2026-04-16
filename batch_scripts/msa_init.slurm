#!/bin/bash
# Prefer batch_scripts/narval_pretrain.slurm + configs/pretrain.narval.example.yaml for Narval.
#SBATCH --job-name=pretrain_model_init
#SBATCH --account=account-username
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --output=slurm_logs/pretrain_model_init_%j.out
#SBATCH --error=slurm_logs/pretrain_model_init_%j.err
#SBATCH --mail-user=user@email.com
#SBATCH --mail-type=BEGIN,END,FAIL

echo "--- Starting solo job with ID: $SLURM_JOB_ID ---"

module load python/3.11
source ~/load/environment/with/libraries

cd ~/scratch/masked-stellar-autoencoder/

python training/pretrain_msa.py --config configs/pretrain.yaml
