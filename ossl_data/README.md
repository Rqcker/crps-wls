# OSSL spectra and measured sensor parameters

The spectral benchmark trains eight SOC models on LUCAS vis-NIR spectra from
the Open Soil Spectral Library (OSSL). Download the two raw files (public
bucket, no registration):

```
https://storage.googleapis.com/soilspec4gg-public/ossl_visnir_L0_v1.2.csv.gz    (~175 MB)
https://storage.googleapis.com/soilspec4gg-public/ossl_soillab_L1_v1.2.csv.gz   (~10 MB)
```

Place them in this directory. `cluster/setup_env.sh` fetches them
automatically. On first run the pipeline builds a local `.npz` cache from
these files.

Committed outputs of the pipeline, so that everything downstream runs without
the GPU stage:

- `measured_params_cluster.json`: measured sensor noise `sigma_i` and
  manipulation sensitivity `alpha_i`, plus the R^2 benchmark (paper Table 2).
  These values define the agents in `src/agents.py`.
- `real_manipulation.csv`: bias of each aggregator when the trained models
  predict from real perturbed spectra (paper Fig. 2(a), file fig13).
- `real_manip_clean_bias.json`: per-model bias on clean spectra for the same
  experiment (the 7% to 27% over-prediction in paper Section 5.2).
