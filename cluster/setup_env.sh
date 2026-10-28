#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=crps_setup
#SBATCH --output=slurm_logs/setup_%j.out

# Environment and data setup for the spectral benchmark (run once).
# Set SCRATCH to a filesystem with room for package caches and the OSSL data;
# add a --partition line for your cluster's GPU partition if needed.
set -euo pipefail

SCRATCH=${SCRATCH:?set SCRATCH to a large scratch directory}
PROJ=${PROJ:-$PWD}

# Keep caches off the (often quota-limited) home filesystem.
export PIP_CACHE_DIR=$SCRATCH/.pip_cache
export TMPDIR=$SCRATCH/tmp
export XDG_CACHE_HOME=$SCRATCH/.cache
export TORCH_HOME=$SCRATCH/.cache/torch
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "envs/crps-wls"; then
  conda create -n crps-wls python=3.12 -y
fi
conda activate crps-wls

pip install --cache-dir "$PIP_CACHE_DIR" -r "$PROJ/requirements.txt" torch

# OSSL vis-NIR data (public bucket)
cd "$PROJ/ossl_data"
for f in ossl_visnir_L0_v1.2.csv.gz ossl_soillab_L1_v1.2.csv.gz; do
  [ -f "$f" ] || curl -sS -o "$f" "https://storage.googleapis.com/soilspec4gg-public/$f"
done
ls -la "$PROJ/ossl_data"

python - <<'PY'
import numpy, pandas, scipy, sklearn, matplotlib, torch
print("versions:", numpy.__version__, pandas.__version__, sklearn.__version__, torch.__version__)
print("CUDA available:", torch.cuda.is_available(),
      "| device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"))
PY
echo "SETUP_DONE"
