#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:25:00
#SBATCH --job-name=crps_smoke
#SBATCH --output=slurm_logs/smoke_%j.out

# Quick correctness and timing check: all eight models on a 1500-sample subset,
# 2 folds, few epochs. Catches architecture/CUDA bugs before the full run.
set -euo pipefail

SCRATCH=${SCRATCH:?set SCRATCH to a large scratch directory}
PROJ=${PROJ:-$PWD}
export TMPDIR=$SCRATCH/tmp XDG_CACHE_HOME=$SCRATCH/.cache TORCH_HOME=$SCRATCH/.cache/torch
export OMP_NUM_THREADS=6
export CRPS_NSUB=1500 CRPS_FOLDS=2 CRPS_EPOCHS=4 CRPS_PRE=2

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crps-wls

echo "=== node $(hostname) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
cd "$PROJ"
python -m src.spectral_models
echo "SMOKE_DONE"
