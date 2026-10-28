# Manipulation-Resistant Oracles via Continuous Peer Prediction

Code and experiments for *Manipulation-Resistant Oracles via Continuous Peer
Prediction*, accepted at MARBLE 2026, the 7th International Conference on
Mathematical Research for Blockchain Economy, Limassol, Cyprus, 16-18
September 2026.

**[Project page](https://junhaosong.com/crps-wls/)** · **[Extended version on arXiv, with the full proofs](https://arxiv.org/abs/2610.35888)**

![Mechanism overview](assets/teaser.png)

A carbon project operator can dry the soil before measurement, raise its
apparent soil organic carbon (SOC), and earn credits for carbon that was never
stored. This repository implements a mechanism for oracle networks whose
sensors report Gaussian beliefs about a physical quantity the source can
manipulate. Sensors are paid with the closed-form Gaussian continuous ranked
probability score (CRPS) against their peers, which makes truthful reporting a
strict equilibrium. A weighted least squares (WLS) read-out jointly estimates
the true value and the manipulation, issues credits on the former, and slashes
the source's stake on the latter. The paper's main result is an
if-and-only-if characterisation: report-based rules identify and deter source
manipulation precisely when the sensors respond differently to the same
manipulation. Every step is a deterministic function of the reports, so the
mechanism suits smart-contract execution without a trusted aggregator.

## Layout

```
src/                          Mechanism and experiments
  agents.py                     Eight sensor agents with measured sigma_i, alpha_i
  mechanism.py                  Closed-form Gaussian CRPS and the peer score
  source.py                     WLS estimator, detection power, baseline aggregators
  dataset.py                    LUCAS 2018 topsoil loader
  experiments.py                Exp1-Exp11 and exp_corr; run_all() runs all 16
  exp_collusion.py              Stress test: partial sensor coalitions
  exp_misspec.py                Stress test: mean truthfulness under a wrong covariance
  exp_alpha_mismatch.py         Stress test: sensitivity drift
  exp_regression_baseline.py    Stress test: Huber and soft-L1 fits of the same model
  exp_corr_detection.py         Detection power under correlated noise (closed form)
  exp_e1_outlier.py             Ablation without the least informative sensor
  stake_worked_example.py       Per-round stake economics at the median parcel
  spectral_models.py            Eight-model SOC benchmark on real spectra (GPU)
  real_manipulation.py          Trained models on real perturbed spectra (GPU)
  plots.py, plot_style.py       Figures and their shared style
  data/                         Committed experiment outputs (CSV)
ossl_data/                    Measured sensor parameters; raw OSSL spectra go here
data/                         LUCAS 2018 CSV goes here (not redistributed)
cluster/                      Slurm scripts for the GPU stages
assets/                       Overview figure for this README
figures/                      Generated figures (not committed)
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

The figure files keep the code's own numbering; the table below maps them to
the paper. To re-run the experiments, place the Land Use/Cover Area frame
Survey (LUCAS) 2018 topsoil CSV as described in `data/README.md`, then:

```
python -m src.experiments      # all 16 experiments, rewrites src/data/*.csv
python -m src.plots
```

The full run takes about a quarter of an hour on a laptop CPU. The committed
outputs come from this version of the code.

| Experiment | Output | In the paper | Question it answers |
|---|---|---|---|
| exp1 | fig2 | Section 5.2 | Is truthful reporting the best response for every sensor? |
| exp2 | fig3 | Fig. 5(b) | Does the estimate stay unbiased as manipulation grows? |
| exp3 | fig4 | Fig. 2(b) | How fast does estimation fail as sensor diversity vanishes? |
| exp4 | fig5, fig12 | Fig. 4(a), Fig. 5(a), Table 3 | How much do shift-naive aggregators over-credit? |
| exp5 | fig6 | Fig. 3(b) | What stake deters manipulation at a given detection power? |
| exp6 | fig7 | Fig. 3(a) | How do error and detection scale with the number of sensors? |
| exp7 | fig8 | Section 5.3 | Does unbiasedness survive non-Gaussian noise? |
| exp8 | CSV | Table 3 | Are the WLS gains statistically significant (paired tests)? |
| exp9 | fig9 | not in the paper | Does the mechanism generalise across land-cover classes? |
| exp10 | fig10 | Section 5.3 | How much error in the assumed sigma_i and alpha_i is tolerable? |
| exp11 | fig11 | not in the paper | What can a budgeted report-level attacker still inflate? |
| exp_corr | CSV | not in the paper | Does correlated sensor noise bias the estimate? |
| exp_collusion | fig14 | Fig. 4(b) | Are partial sensor coalitions detected or penalised? |
| exp_misspec | CSV | Appendix C | Is the reported mean truthful under a wrong covariance? |
| exp_alpha_mismatch | fig15 | Appendix F | What bias does sensitivity drift cause? |
| exp_regression_baseline | CSV | Table 5 | Is invariance a property of the estimator or the loss? |

The shift-naive aggregators in exp4 and exp8 apply the core rules of deployed
systems (OFP, WeatherXM and Boomitra). They are not evaluations of those
systems' full pipelines. In exp10, each calibration error is held fixed across
rounds and enters the WLS weights: a 50% error in the assumed sigma_i raises
the error at fifty percent manipulation from 57% to 63%.

The other paper items come from the files below:

| In the paper | Source |
|---|---|
| Fig. 1 | `assets/teaser.png` |
| Fig. 2(a) | `src/real_manipulation.py` (GPU), data in `ossl_data/real_manipulation.csv`, figure fig13 |
| Table 2 | `src/spectral_models.py` (GPU), data in `ossl_data/measured_params_cluster.json` |
| Table 4 | `src/stake_worked_example.py` |
| Section 5.3, correlated residuals | `src/exp_corr_detection.py` |

Standalone closed-form checks (no data needed):

```
python -m src.exp_e1_outlier        # drop the least informative sensor
python -m src.exp_corr_detection    # detection power under correlated noise
python -m src.stake_worked_example  # per-round stake economics at the median parcel
```

## Real-spectra stage (GPU)

The sensor parameters in `src/agents.py` were measured, not assumed: eight SOC
models trained on 37,280 mineral-soil visible and near-infrared (vis-NIR)
spectra, the LUCAS subset of the Open Soil Spectral Library (OSSL), with
sensitivities measured from a physically motivated wetting perturbation of the
spectra. To re-measure them on a Slurm cluster with one GPU, run from the
repository root:

```
mkdir -p slurm_logs              # Slurm writes the job logs here
sbatch cluster/setup_env.sh      # once: conda env + OSSL download (~185 MB)
sbatch cluster/smoke_test.sh     # 25-minute subset check, output to $SCRATCH
sbatch cluster/train_models.sh   # full benchmark -> ossl_data/measured_params_cluster.json
sbatch cluster/real_manip.sh     # real-manipulation experiment -> data behind fig13
```

The scripts expect `SCRATCH` to point at a large scratch directory; see
`ossl_data/README.md` for the data files.

## Citation

```bibtex
@inproceedings{song2026manipulation,
  author        = {Song, Junhao and Rolandi, Tanis and Novocky, Adam and
                   Biffis, Enrico and Knottenbelt, William},
  title         = {Manipulation-Resistant Oracles via Continuous Peer
                   Prediction},
  booktitle     = {Mathematical Research for Blockchain Economy (MARBLE 2026)},
  publisher     = {Springer},
  year          = {2026},
  eprint        = {2610.35888},
  archivePrefix = {arXiv},
  primaryClass  = {stat.AP}
}
```

The DOI, page numbers and final series details of the proceedings will be added
here once they are published.

## Licence

MIT. See [LICENSE](LICENSE).
