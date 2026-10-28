# GPU stage

The sensor parameters in `src/agents.py` were measured, not assumed. Eight SOC
models were trained on 37,280 mineral-soil visible and near-infrared (vis-NIR)
spectra, the LUCAS subset of the Open Soil Spectral Library (OSSL). Each model's
sensitivity comes from a physically motivated wetting perturbation of the
spectra.

To re-measure them on a Slurm cluster with one GPU, set `SCRATCH` to a large
scratch directory and run from the repository root:

```bash
mkdir -p slurm_logs              # Slurm writes the job logs here
sbatch cluster/setup_env.sh      # once: conda environment and OSSL download (~185 MB)
sbatch cluster/smoke_test.sh     # 25-minute subset check, output to $SCRATCH
sbatch cluster/train_models.sh   # full benchmark -> ossl_data/measured_params_cluster.json
sbatch cluster/real_manip.sh     # real-spectra experiment -> ossl_data/real_manipulation.csv
```

See [`ossl_data/README.md`](../ossl_data/README.md) for the data files.
