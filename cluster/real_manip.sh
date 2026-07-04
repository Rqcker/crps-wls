#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --job-name=crps_real
#SBATCH --output=slurm_logs/real_%j.out

# Fully-real manipulation experiment: real models on real perturbed spectra.
set -euo pipefail

SCRATCH=${SCRATCH:?set SCRATCH to a large scratch directory}
PROJ=${PROJ:-$PWD}
export TMPDIR=$SCRATCH/tmp XDG_CACHE_HOME=$SCRATCH/.cache TORCH_HOME=$SCRATCH/.cache/torch
export OMP_NUM_THREADS=6

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate crps-wls

echo "=== node $(hostname); $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
cd "$PROJ"
python -m src.real_manipulation
echo "REAL_DONE"
