# OSSL spectra and measured sensor parameters

The spectral benchmark trains eight SOC models on LUCAS vis-NIR spectra from
the Open Soil Spectral Library (OSSL). Download the three raw files (public
bucket, no registration):

```
https://storage.googleapis.com/soilspec4gg-public/ossl_visnir_L0_v1.2.csv.gz    (~175 MB)
https://storage.googleapis.com/soilspec4gg-public/ossl_soillab_L1_v1.2.csv.gz   (~10 MB)
https://storage.googleapis.com/soilspec4gg-public/ossl_soilsite_L0_v1.2.csv.gz  (~6 MB)
```

Place them in this directory. `cluster/setup_env.sh` fetches them
automatically. On first run the pipeline builds a local `.npz` cache from
these files.

Committed outputs of the pipeline, so that everything downstream runs without
the GPU stage:

- `measured_params.json` and `measured_params_cluster.json`: measured sensor
  noise `sigma_i` and manipulation sensitivity `alpha_i` (plus the R^2
  benchmark) that define the agents in `src/agents.py`.
- `real_manipulation.csv`: bias of each aggregator when the trained models
  predict from real perturbed spectra (Figure 13).
- `real_manip_clean_bias.json`: clean-spectra bias reference for the same
  experiment.
