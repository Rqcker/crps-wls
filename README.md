# Manipulation-Resistant Oracles via Continuous Peer Prediction

Code and experiments for *Manipulation-Resistant Oracles via Continuous Peer
Prediction*, accepted at MARBLE 2026, the 7th International Conference on
Mathematical Research for Blockchain Economy, Limassol, Cyprus, 16-18
September 2026.

**[Project page](https://junhaosong.com/crps-wls/)** · **[Extended preprint, with the full proofs](https://junhaosong.com/crps-wls/crps-wls-extended.pdf)**

![Mechanism overview](assets/teaser.png)

A carbon project operator can dry the soil before measurement, raise its
apparent soil organic carbon (SOC), and earn credits for carbon that was never
stored. This repository implements a mechanism for oracle networks whose
sensors report Gaussian beliefs about a physical quantity the source can
manipulate. Sensors are paid with the closed-form Gaussian CRPS against their
peers, which makes truthful reporting a strict equilibrium. A weighted least
squares (WLS) read-out jointly estimates the true value and the manipulation,
issues credits on the former, and slashes the source's stake on the latter.
The paper's main result is an if-and-only-if characterisation: report-based
rules identify and deter source manipulation precisely when the sensors
respond differently to the same manipulation. Every step is a deterministic
function of the reports, so the mechanism runs on-chain without a trusted
aggregator.

## Layout

```
src/                  Mechanism and experiments
  agents.py             Eight sensor agents with measured sigma_i, alpha_i
  mechanism.py          Closed-form Gaussian CRPS and peer scoring
  source.py             WLS estimator, detection power, source best response
  dataset.py            LUCAS 2018 topsoil loader
  experiments.py        Exp1-Exp11 plus stress tests (16 experiments)
  blockchain_sim.py     On-chain oracle round simulation
  spectral_models.py    Eight-model SOC benchmark on real vis-NIR spectra (GPU)
  real_manipulation.py  Trained models on real perturbed spectra (GPU)
  plots.py              All paper figures
  data/                 Committed experiment outputs (CSV)
ossl_data/            Measured sensor parameters; raw OSSL spectra go here
data/                 LUCAS 2018 CSV goes here (not redistributed)
cluster/              Slurm scripts for the GPU stages
figures/              Generated figures (not committed)
```

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

PyTorch is needed only for the spectral benchmark; see `requirements.txt`.

## Reproducing the paper

Every simulation figure can be regenerated from the committed experiment
outputs, without any dataset or GPU:

```
python -m src.plots            # figures/fig2 ... fig15 from src/data/*.csv
```

To re-run the experiments themselves, place the LUCAS 2018 CSV as described in
`data/README.md`, then:

```
python -m src.experiments      # all 16 experiments, rewrites src/data/*.csv
python -m src.plots
```

| Experiment | Figure | Question it answers |
|---|---|---|
| exp1 | fig2 | Is truthful reporting the best response for every sensor? |
| exp2 | fig3 | Does the estimate stay unbiased as manipulation grows? |
| exp3 | fig4 | How fast does estimation fail as sensor diversity vanishes? |
| exp4 | fig5, fig12 | How much do deployed aggregators over-credit (up to +57%)? |
| exp5 | fig6 | What stake deters manipulation at a given detection power? |
| exp6 | fig7 | How do error and detection scale with the number of sensors? |
| exp7 | fig8 | Does unbiasedness survive non-Gaussian noise? |
| exp8 | table | Are the WLS gains statistically significant (paired tests)? |
| exp9 | fig9 | Does the mechanism generalise across soil regimes? |
| exp10 | fig10 | How much sigma/alpha calibration error is tolerable? |
| exp11 | fig11 | What can a budgeted report-level attacker still inflate? |
| exp_corr | table | Does correlated sensor noise break anything? |
| exp_collusion | fig14 | Are partial sensor coalitions detected or penalised? |
| exp_misspec | table | Is the reported mean truthful under a wrong covariance? |
| exp_alpha_mismatch | fig15 | What bias does sensitivity drift cause? |
| exp_regression_baseline | table | Is invariance a property of the estimator or the loss? |

Standalone closed-form checks (no data needed):

```
python -m src.exp_e1_outlier        # drop the near-uninformative sensor
python -m src.exp_corr_detection    # detection power under correlated noise
python -m src.stake_worked_example  # per-round stake economics at the median parcel
```

On-chain simulation (four market scenarios, deterministic contract logic):

```
python -m src.blockchain_sim
```

## Real-spectra stage (GPU)

The sensor parameters in `src/agents.py` were measured, not assumed: eight SOC
models trained on 37,280 mineral-soil vis-NIR spectra (the LUCAS subset of the
OSSL), with sensitivities measured from a physically motivated wetting
perturbation of the spectra. To re-measure them on a Slurm cluster with one
GPU:

```
sbatch cluster/setup_env.sh      # once: conda env + OSSL download (~190 MB)
sbatch cluster/smoke_test.sh     # 25-minute subset check
sbatch cluster/train_models.sh   # full benchmark -> ossl_data/measured_params_cluster.json
sbatch cluster/real_manip.sh     # real-manipulation experiment -> data behind fig13
```

The scripts expect `SCRATCH` to point at a large scratch directory; see
`ossl_data/README.md` for the data files.

## Citation

```bibtex
@inproceedings{song2026manipulation,
  author    = {Song, Junhao and Rolandi, Tanis and Novocky, Adam and
               Biffis, Enrico and Knottenbelt, William},
  title     = {Manipulation-Resistant Oracles via Continuous Peer Prediction},
  booktitle = {Mathematical Research for Blockchain Economy (MARBLE 2026)},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

The DOI, page numbers and final series details will be added here once the
proceedings are published.

## License

MIT. See [LICENSE](LICENSE).
