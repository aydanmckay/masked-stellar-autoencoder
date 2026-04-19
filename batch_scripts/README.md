# Narval / Alliance Canada batch jobs

These scripts target the [Narval](https://docs.alliancecan.ca/wiki/Narval/en) GPU cluster (Slurm, NVIDIA A100). Adjust `#SBATCH --account` to your allocation (`def-*-gpu`, `rrg-*`, etc.) and verify module names with `module spider cuda cudnn python` on the login node.

## One-time setup

1. Clone the repo under your scratch or project space and record the path as `MSA_REPO`.
2. Create a Python venv with a **CUDA build of PyTorch** that matches the loaded `cuda` module (see [Alliance PyTorch notes](https://docs.alliancecan.ca/wiki/PyTorch)):

   ```bash
   module load python/3.11 cuda cudnn   # versions per site
   bash batch_scripts/setup_venv_narval.sh "$SCRATCH/venvs/msa"
   ```

3. Copy the example configs and edit HDF5/FITS locations:

   ```bash
   cp configs/pretrain.narval.example.yaml configs/pretrain.active.yaml
   cp configs/finetune.narval.example.yaml configs/finetune.active.yaml
   # Point datafile / ft_datafile / saved_weights at your staged data.
   ```

4. In job scripts (or your shell profile), set:

   ```bash
   export SCRATCH=/scratch/$USER          # or your allocation scratch path
   export MSA_VENV=$SCRATCH/venvs/msa
   ```

Paths in YAML may use `$SCRATCH/...`; `training/config_paths.py` expands them at runtime.

## Submit jobs

From the repo root (so `slurm_logs/` is writable):

```bash
mkdir -p slurm_logs
export SCRATCH=/scratch/$USER
export MSA_VENV=$SCRATCH/venvs/msa

# Pretrain (long)
CONFIG=configs/pretrain.active.yaml sbatch batch_scripts/narval_pretrain.slurm

# Fine-tune ensemble (writes ..._seed{seed}.pth per member when finetuning.ensemble: true)
CONFIG=configs/finetune.active.yaml sbatch batch_scripts/narval_finetune.slurm

# Eval: globs ensemble members (override glob if needed)
EVAL_CKPT_GLOB="$SCRATCH/msa/runs/ft/masked_stellar_autoencoder_ft_seed*.pth" \
  CONFIG=configs/finetune.active.yaml sbatch batch_scripts/narval_eval.slurm
```

`batch_scripts/env_narval.sh` sets `PYTHONPATH`, optional venv activation, `WANDB_MODE=offline`, and creates scratch subdirectories.

## Large HDF5 staging (optional)

For faster I/O, copy the pretrain `.h5` to node-local storage at job start (add to the Slurm script after `env_narval.sh`):

```bash
cp "$SCRATCH/msa/data/sslset-realmags-full.h5" "$SLURM_TMPDIR/"
# then point data.datafile in the active pretrain YAML at $SLURM_TMPDIR/...
```

## Legacy scripts

`msa_init.slurm` and `msa_looping.slurm` are older templates. Prefer `narval_*.slurm` and the `*.narval.example.yaml` configs.

## Ensemble outputs

With `finetuning.ensemble: true`, each member is saved as
`{model_str without ext}_seed{seed}{ext}`
so runs no longer overwrite a single checkpoint file.
