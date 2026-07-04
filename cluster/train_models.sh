#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=16:00:00
#SBATCH --job-name=crps_train
#SBATCH --output=slurm_logs/train_%j.out

# Eight-model SOC benchmark; measures sensor noise sigma_i and manipulation
# sensitivity alpha_i on mineral LUCAS spectra. Needs one GPU.
set -euo pipefail

SCRATCH=${SCRATCH:?set SCRATCH to a large scratch directory}
PROJ=${PROJ:-$PWD}
export TMPDIR=$SCRATCH/tmp XDG_CACHE_HOME=$SCRATCH/.cache TORCH_HOME=$SCRATCH/.cache/torch
export OMP_NUM_THREADS=6
export CRPS_EPOCHS=30 CRPS_PRE=8 CRPS_FOLDS=5

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crps-wls

echo "=== node $(hostname); $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$PROJ"
python -m src.spectral_models
echo "TRAIN_DONE"
