"""
Experiments for the MARBLE 2026 paper: Exp1--Exp11 and the correlated-noise
check (exp_corr). run_all() also runs the four stress tests that live in their
own modules (exp_collusion, exp_misspec, exp_alpha_mismatch,
exp_regression_baseline), sixteen experiments in all.

All experiments draw ground-truth theta values from the LUCAS 2018 topsoil OC
distribution (17,231 mineral samples with OC <= 100 g/kg, 0-20 cm depth).

Model:
  Agent i observes y_i = theta + alpha_i * s + eps_i, eps_i ~ N(0, sigma_i^2).
  Agent reports honestly: N(y_i, sigma_i^2).

Exp1 (equilibrium verification) and Exp5 (stake economics) are analytical; the
others aggregate stochastic reports. Each experiment returns a DataFrame.

Random seed: RNG_SEED = 42 (fixed for reproducibility).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from scipy.stats import ttest_rel, wilcoxon

from src.dataset import load_oc, load_oc_by_landcover, sample_theta
from src.agents import (
    Agent, AGENTS, AGENTS_UNIFORM,
    get_noisy_reports, get_noisy_reports_corr, get_noisy_reports_dist, make_agents,
)
from src.mechanism import agent_mechanism_payment
from src.source import (
    estimate_theta_s,
    aggregate_naive,
    aggregate_weighted,
    aggregate_threshold,
    aggregate_output_agreement,
    aggregate_median,
    aggregate_peer_crps_weighted,
    aggregate_weatherxm_cqc,
    wls_covariance,
    manipulation_detection_power,
    worst_case_evasion_bias,
)
from src.exp_collusion import run_exp_collusion
from src.exp_misspec import run_exp_misspec
from src.exp_alpha_mismatch import run_exp_alpha_mismatch
from src.exp_regression_baseline import run_exp_regression_baseline

RNG_SEED      = 42
N_THETA       = 1_000   # ground-truth draws per condition
N_DELTA       = 61      # deviation grid for Exp1
N_S           = 41      # grid points for manipulation fraction in Exp2/Exp4
N_SPREAD      = 20      # diversity spread grid for Exp3
N_SEEDS       = 5       # independent seeds for stability
S_FRAC_MAX    = 1.0     # maximum source manipulation as fraction of theta (100%)
FIXED_S_FRAC  = 0.2     # fixed manipulation fraction for Exp3 (20%)


# ---------------------------------------------------------------------------
# Exp1 — Equilibrium and truthfulness (clean, analytical)
# ---------------------------------------------------------------------------

def run_exp1(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Verify the strict best response of Proposition 1 at the honest-source
    equilibrium: under the mechanism score (paper Eq. (5)), truthful reporting
    uniquely maximises an agent's payment.

    At the equilibrium the source plays s=0, so every peer reports mu_j=theta and
    agent i's posterior predictive for each peer is N(Y_i, sigma_i^2 + sigma_j^2).
    The mechanism score peer_crps inflates agent i's variance by the known sigma_j^2
    and scores against the peer mean, so the payment depends only on the reporting
    deviation delta = mu_reported - mu_true and on the sigma values, not on theta
    (CRPS is translation invariant when all agents share theta under s=0). We
    therefore fix theta=0 and vary delta, which is exact for any theta. Each agent's
    payment is uniquely maximised at delta=0, confirming the strict best response on
    the eight measured (heterogeneous) sensitivities.

    Returns DataFrame: agent, delta, payment (std=0 by construction).
    """
    deltas = np.linspace(-15.0, 15.0, N_DELTA)

    rows = []
    for i, agent_i in enumerate(AGENTS):
        # With theta=0 and s=0, every agent j reports mu_j=0.
        # Agent i deviates to delta; payment = f(delta, sigma_i, {sigma_j}).
        baseline_reports = [(0.0, a.sigma) for a in AGENTS]
        for delta in deltas:
            reports_dev = list(baseline_reports)
            reports_dev[i] = (delta, agent_i.sigma)
            rows.append({
                "agent":   agent_i.name,
                "delta":   delta,
                "payment": agent_mechanism_payment(reports_dev, i),
                "std":     0.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp2 — Manipulation resistance (stochastic)
# ---------------------------------------------------------------------------

def run_exp2(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Compare relative theta estimation error under increasing source manipulation.

    Source manipulation is expressed as a FRACTION of the true theta value:
        s_i = frac * theta  (so a 20% inflation is frac=0.2 for any theta)
    This makes the experiment scale-invariant across the LUCAS OC distribution,
    avoiding the problem of absolute s=20 g/kg being implausible for low-OC soils.

    Metric: mean absolute relative error = E[|theta_hat - theta| / theta].

    Setting "Diverse":  AGENTS with alpha_i in [0.004, 1.0] (diversity holds).
    Setting "Uniform":  AGENTS_UNIFORM with alpha_i = 1.0 (diversity fails).

    Returns DataFrame: setting, frac, mare, std.
    """
    if oc is None:
        oc = load_oc()

    fracs = np.linspace(0.0, S_FRAC_MAX, N_S)
    rows = []

    for frac in fracs:
        for setting, agents in [("Diverse", AGENTS), ("Uniform", AGENTS_UNIFORM)]:
            seed_mares = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(RNG_SEED + seed)
                thetas = sample_theta(oc, N_THETA, rng)
                rel_errors = []
                for theta in thetas:
                    s = frac * theta   # relative manipulation
                    reports = get_noisy_reports(agents, theta, s, rng)
                    try:
                        theta_hat, _ = estimate_theta_s(reports, agents)
                    except np.linalg.LinAlgError:
                        theta_hat = aggregate_weighted(reports)
                    rel_errors.append(abs(theta_hat - theta) / max(theta, 1e-6))
                seed_mares.append(float(np.mean(rel_errors)))
            rows.append({
                "setting": setting,
                "frac":    frac,
                "mare":    float(np.mean(seed_mares)),
                "std":     float(np.std(seed_mares)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp3 — Sensor diversity necessity (stochastic)
# ---------------------------------------------------------------------------

def run_exp3(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Vary the spread of alpha_i and measure theta estimation error.

    Alpha values interpolate linearly from uniform to the measured diverse set:
        alpha_i = alpha_base + spread * (alpha_full - alpha_base)
    where alpha_full are the measured AGENTS sensitivities and alpha_base sets
    every agent to their mean (spread=0, diversity fails). At spread=1 the
    ensemble is exactly AGENTS.

    Fixed source action s = FIXED_S_FRAC * theta (20% relative manipulation)
    to keep the experiment scale-invariant across the LUCAS OC distribution.

    Returns DataFrame: alpha_spread, mare, std.
    """
    if oc is None:
        oc = load_oc()

    spreads = np.linspace(0.01, 1.0, N_SPREAD)  # start at 0.01 to avoid exact singularity
    alpha_full = np.array([a.alpha for a in AGENTS])      # spread=1: measured AGENTS
    alpha_base = np.full(len(AGENTS), float(alpha_full.mean()))  # spread=0: uniform

    rows = []
    for spread in spreads:
        alphas = alpha_base + spread * (alpha_full - alpha_base)
        agents = [
            Agent(name=AGENTS[k].name, sigma=AGENTS[k].sigma, alpha=float(alphas[k]))
            for k in range(len(AGENTS))
        ]
        seed_mares = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(RNG_SEED + seed)
            thetas = sample_theta(oc, N_THETA, rng)
            rel_errors = []
            for theta in thetas:
                s = FIXED_S_FRAC * theta   # relative manipulation
                reports = get_noisy_reports(agents, theta, s, rng)
                try:
                    theta_hat, _ = estimate_theta_s(reports, agents)
                except np.linalg.LinAlgError:
                    theta_hat = aggregate_weighted(reports)
                rel_errors.append(abs(theta_hat - theta) / max(theta, 1e-6))
            seed_mares.append(float(np.mean(rel_errors)))
        rows.append({
            "alpha_spread": float(spread),
            "mare":         float(np.mean(seed_mares)),
            "std":          float(np.std(seed_mares)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp4 — Baseline comparison (stochastic)
# ---------------------------------------------------------------------------

def run_exp4(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Compare WLS mechanism against baselines under relative source manipulation.

    Source manipulation: s = frac * theta (scale-invariant).
    Metric: mean absolute relative error = E[|aggregate - theta| / theta].

    Baselines:
        B1  OFP-style output agreement (majority-cluster)
        B2  WeatherXM QoD v1.0 outlier-removal mean
        B3  Naive unweighted mean
        B4  Boomitra-style precision-weighted mean
        B5  WLS with 20% alpha estimation noise (robustness check)
        B6  Sample median (classical robust statistics)
        B7  CRPS-peer-score-reweighted mean (R&F 2024, source-naive)
        B8  WeatherXM CQC-style (planned v1.3 peer comparison)

    Returns DataFrame: method, frac, mare, std.
    """
    if oc is None:
        oc = load_oc()

    fracs = np.linspace(0.0, S_FRAC_MAX, N_S)

    def wls_noisy_alpha(rpts, agts, alpha_noise_frac=0.2, n_realizations=10):
        """WLS averaged over multiple alpha estimation noise realisations.

        Models the case where alpha_i are estimated with ±20% error.
        Averages theta_hat over 10 independent noise draws to give
        the expected performance under alpha estimation uncertainty.
        """
        estimates = []
        for k in range(n_realizations):
            rng_local = np.random.default_rng(200 + k)
            noisy_agents = [
                Agent(name=a.name, sigma=a.sigma,
                      alpha=max(0.01, a.alpha * (1.0 + rng_local.uniform(-alpha_noise_frac, alpha_noise_frac))))
                for a in agts
            ]
            try:
                estimates.append(estimate_theta_s(rpts, noisy_agents)[0])
            except np.linalg.LinAlgError:
                estimates.append(aggregate_weighted(rpts))
        return float(np.mean(estimates))

    methods = {
        "Ours (WLS)":           lambda rpts, agts: estimate_theta_s(rpts, agts)[0],
        "OFP (B1)":             lambda rpts, agts: aggregate_output_agreement(rpts),
        "WeatherXM (B2)":       lambda rpts, agts: aggregate_threshold(rpts),
        "Naive mean (B3)":      lambda rpts, agts: aggregate_naive(rpts),
        "Boomitra (B4)":        lambda rpts, agts: aggregate_weighted(rpts),
        "WLS+alpha noise (B5)": wls_noisy_alpha,
        "Median (B6)":          lambda rpts, agts: aggregate_median(rpts),
        "Peer-CRPS (B7)":       lambda rpts, agts: aggregate_peer_crps_weighted(rpts),
        "WeatherXM CQC (B8)":   lambda rpts, agts: aggregate_weatherxm_cqc(rpts),
    }

    rows = []
    for frac in fracs:
        for method_name, agg_fn in methods.items():
            seed_mares, seed_bias = [], []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(RNG_SEED + seed)
                thetas = sample_theta(oc, N_THETA, rng)
                rel_errors, signed = [], []
                for theta in thetas:
                    s = frac * theta
                    reports = get_noisy_reports(AGENTS, theta, s, rng)
                    try:
                        agg = agg_fn(reports, AGENTS)
                    except np.linalg.LinAlgError:
                        agg = aggregate_weighted(reports)
                    rel_errors.append(abs(agg - theta) / max(theta, 1e-6))
                    signed.append((agg - theta) / max(theta, 1e-6))
                seed_mares.append(float(np.mean(rel_errors)))
                seed_bias.append(float(np.mean(signed)))   # systematic over/under-estimation
            rows.append({
                "method": method_name,
                "frac":   frac,
                "mare":   float(np.mean(seed_mares)),
                "std":    float(np.std(seed_mares)),
                "bias":   float(np.mean(seed_bias)),        # signed relative bias
                "bias_std": float(np.std(seed_bias)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp5 — On-chain stake economics (blockchain)
# ---------------------------------------------------------------------------

def run_exp5(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Minimum stake economics for on-chain manipulation deterrence.

    Blockchain context: the source deposits a stake T before each measurement
    round, and the whole stake is slashed whenever s_hat exceeds the detection
    threshold delta, whether or not the source manipulated.  An honest source
    (frac = 0) is still slashed with the false-alarm probability a = FAR;
    manipulating by frac raises that probability to P_detect(frac, theta).
    Cheating is unprofitable when the *incremental* expected slashing covers
    the per-round gain G from manipulating:

        (P_detect(frac, theta) - a) * T >= G

    so the minimum stake-to-gain multiplier M*(frac) = T* / G is:

        M*(frac) = 1 / (P_detect(frac, theta) - a)

    where P_detect(frac, theta) = P(s_hat > delta | s = frac * theta) and
    delta is the (1 - a)-quantile of s_hat under honest behaviour (a = 5% FAR).
    The relevant deterrent is the detector's power above its size, not its raw
    power: as the sensitivity spread vanishes P_detect -> a and M* -> infinity.

    Two settings:
      "Diverse":  AGENTS (alpha_i in [0.004, 1.0]) -- detection possible,
                  M*(frac) is finite.
      "Uniform":  AGENTS_UNIFORM (all alpha_i = 1.0) -- rank-deficient,
                  P_detect = 0, M* = infinity for any frac > 0.

    The Uniform result is the blockchain-economics corollary of Theorem 1
    necessity: without sensor diversity, no finite stake can deter manipulation.

    Detection power is evaluated at theta = median(LUCAS mineral) = 19.9 g/kg,
    which represents the typical European agricultural soil.  Using the mean
    (26.5 g/kg) would over-weight the right-skewed high-OC tail of the mineral
    subset rather than the typical deployment soil.

    Records per row:
      p_detect    -- P(s_hat > delta | s = frac * theta_median), 5% FAR
      stake_mult  -- 1 / (p_detect - a) = minimum stake as multiple of the gain,
                     net of the false-alarm slashing the honest source also pays
      sd_shat     -- analytical SD(s_hat) from WLS covariance matrix
      delta       -- detection threshold (g/kg)

    Returns DataFrame: setting, frac, p_detect, stake_mult (NaN = infinity).
    """
    if oc is None:
        oc = load_oc()

    fracs = np.linspace(0.0, S_FRAC_MAX, N_S)

    # Representative theta: LUCAS median (farmland, not peatland)
    theta_median = float(np.median(oc))

    # Analytical WLS covariance -- exact, not Monte Carlo
    far             = 0.05                      # detector false-alarm rate (size)
    cov_diverse     = wls_covariance(AGENTS)
    sd_shat_diverse = float(np.sqrt(cov_diverse[1, 1]))
    delta_diverse   = float(_norm.ppf(1.0 - far) * sd_shat_diverse)

    rows = []
    for frac in fracs:
        s_median = frac * theta_median   # manipulation at median theta

        # --- Diverse setting ---
        # Honest source is slashed with prob `far`; the deterrent is the excess
        # detection above that floor, P_detect - far (power above size).
        p_det = float(1.0 - _norm.cdf((delta_diverse - s_median) / sd_shat_diverse))
        excess = p_det - far
        stake_mult = (1.0 / excess) if excess > 1e-9 else np.nan

        rows.append({
            "setting":    "Diverse",
            "frac":       frac,
            "p_detect":   p_det,
            "stake_mult": stake_mult,
            "sd_shat":    sd_shat_diverse,
            "delta":      delta_diverse,
            "theta_ref":  theta_median,
        })

        # --- Uniform setting (diversity fails) ---
        rows.append({
            "setting":    "Uniform",
            "frac":       frac,
            "p_detect":   0.0,
            "stake_mult": np.nan,   # infinity: no finite stake deters
            "sd_shat":    np.nan,
            "delta":      np.nan,
            "theta_ref":  theta_median,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp6 — Scaling with the number of agents N
# ---------------------------------------------------------------------------

N_AGENTS_GRID = [3, 5, 7, 10, 15, 20]


def run_exp6(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Measure how the mechanism scales with the number of agents N.

    For each N in N_AGENTS_GRID, agents are built by make_agents(N): sigma_i
    spaced over the measured span [13.22, 17.98] g/kg and alpha_i over
    [0.20, 1.0] so the diversity condition always holds.

    Three quantities are reported per N:
      sd_theta      -- analytical SD(theta_hat) from the WLS covariance (exact)
      mare_honest   -- E[|theta_hat - theta|/theta] at frac=0   (noise floor)
      mare_manip    -- E[|theta_hat - theta|/theta] at frac=0.5 (under attack)
      p_detect      -- detection power at frac=0.5, theta=median, 5% FAR

    Expected behaviour (the experiment verifies, does not assume):
      sd_theta decreases roughly as 1/sqrt(N); mare_honest decreases with N;
      mare_manip stays equal to mare_honest (WLS unbiased under manipulation);
      p_detect increases with N (more sensors -> sharper detection).

    Returns DataFrame: n_agents, sd_theta, mare_honest, mare_manip, p_detect.
    """
    if oc is None:
        oc = load_oc()

    theta_median = float(np.median(oc))
    frac_attack = 0.5

    rows = []
    for n in N_AGENTS_GRID:
        agents = make_agents(n)

        # Analytical WLS covariance (exact)
        cov = wls_covariance(agents)
        sd_theta = float(np.sqrt(cov[0, 0]))
        sd_shat  = float(np.sqrt(cov[1, 1]))
        delta    = float(_norm.ppf(0.95) * sd_shat)
        p_detect = float(1.0 - _norm.cdf((delta - frac_attack * theta_median) / sd_shat))

        # Monte Carlo MARE at frac=0 and frac=0.5, averaged over seeds
        mare_by_frac = {}
        for frac in (0.0, frac_attack):
            seed_mares = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(RNG_SEED + seed)
                thetas = sample_theta(oc, N_THETA, rng)
                rel_errors = []
                for theta in thetas:
                    s = frac * theta
                    reports = get_noisy_reports(agents, theta, s, rng)
                    try:
                        theta_hat, _ = estimate_theta_s(reports, agents)
                    except np.linalg.LinAlgError:
                        theta_hat = aggregate_weighted(reports)
                    rel_errors.append(abs(theta_hat - theta) / max(theta, 1e-6))
                seed_mares.append(float(np.mean(rel_errors)))
            mare_by_frac[frac] = float(np.mean(seed_mares))

        rows.append({
            "n_agents":    n,
            "sd_theta":    sd_theta,
            "mare_honest": mare_by_frac[0.0],
            "mare_manip":  mare_by_frac[frac_attack],
            "p_detect":    p_detect,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp7 — Robustness to non-Gaussian noise (model misspecification)
# ---------------------------------------------------------------------------

NOISE_MODELS = ["gaussian", "student_t", "lognormal", "uniform"]


def run_exp7(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Test WLS robustness when the measurement noise is not Gaussian.

    All noise models share the same per-agent standard deviation sigma_i but
    differ in shape: Gaussian (baseline), Student-t df=3 (heavy tails),
    lognormal (right-skewed), uniform (light, bounded tails).  Agents still
    report (y_i, sigma_i); only the residual distribution changes.

    The WLS estimate theta_hat = (A^T W A)^{-1} A^T W mu is unbiased whenever
    the noise is mean-zero and the linear model mu_i = theta + alpha_i s holds,
    regardless of the noise shape.  This experiment verifies that the
    manipulation invariance (flat MARE vs frac) survives misspecification,
    while the noise floor may shift with the tail weight.

    Returns DataFrame: noise_model, frac, mare, std.
    """
    if oc is None:
        oc = load_oc()

    fracs = np.linspace(0.0, S_FRAC_MAX, N_S)
    rows = []
    for dist in NOISE_MODELS:
        for frac in fracs:
            seed_mares = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(RNG_SEED + seed)
                thetas = sample_theta(oc, N_THETA, rng)
                rel_errors = []
                for theta in thetas:
                    s = frac * theta
                    reports = get_noisy_reports_dist(AGENTS, theta, s, rng, dist=dist)
                    try:
                        theta_hat, _ = estimate_theta_s(reports, AGENTS)
                    except np.linalg.LinAlgError:
                        theta_hat = aggregate_weighted(reports)
                    rel_errors.append(abs(theta_hat - theta) / max(theta, 1e-6))
                seed_mares.append(float(np.mean(rel_errors)))
            rows.append({
                "noise_model": dist,
                "frac":        frac,
                "mare":        float(np.mean(seed_mares)),
                "std":         float(np.std(seed_mares)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp8 — Statistical significance vs baselines
# ---------------------------------------------------------------------------

N_SEEDS_SIG = 10   # independent seeds for the significance test


def run_exp8(oc: np.ndarray | None = None, fracs=(0.5, 1.0)) -> pd.DataFrame:
    """Paired significance test: WLS vs each baseline under manipulation.

    For each manipulation level in `fracs` and each of N_SEEDS_SIG independent
    seeds, every method aggregates the same set of reports (paired design).  We
    record each method's per-seed mean absolute relative error (MARE), then test
    H0: WLS MARE == baseline MARE against H1: WLS MARE < baseline MARE using both
    a paired one-sided t-test and a one-sided Wilcoxon signed-rank test
    (alpha = 5%), mirroring standard reporting practice.

    Two levels are reported for an honest picture: at frac=0.5 the
    precision-weighted baselines are still more accurate than WLS (the
    efficiency cost of robustness), while frac=1.0 is the high-manipulation
    regime the mechanism targets, where WLS is significantly better than six of
    the seven baselines.

    Returns DataFrame: method, frac, mare_mean, mare_std, t_p, wilcoxon_p,
    significant.
    """
    if oc is None:
        oc = load_oc()

    methods = {
        "Ours (WLS)":         lambda r: estimate_theta_s(r, AGENTS)[0],
        "OFP (B1)":           lambda r: aggregate_output_agreement(r),
        "WeatherXM (B2)":     lambda r: aggregate_threshold(r),
        "Naive mean (B3)":    lambda r: aggregate_naive(r),
        "Boomitra (B4)":      lambda r: aggregate_weighted(r),
        "Median (B6)":        lambda r: aggregate_median(r),
        "Peer-CRPS (B7)":     lambda r: aggregate_peer_crps_weighted(r),
        "WeatherXM CQC (B8)": lambda r: aggregate_weatherxm_cqc(r),
    }

    rows = []
    for frac in fracs:
        # Per-seed MARE for each method (paired across seeds)
        seed_mare = {m: [] for m in methods}
        for seed in range(N_SEEDS_SIG):
            rng = np.random.default_rng(RNG_SEED + seed)
            thetas = sample_theta(oc, N_THETA, rng)
            per_method_err = {m: [] for m in methods}
            for theta in thetas:
                s = frac * theta
                reports = get_noisy_reports(AGENTS, theta, s, rng)
                for m, fn in methods.items():
                    try:
                        est = fn(reports)
                    except np.linalg.LinAlgError:
                        est = aggregate_weighted(reports)
                    per_method_err[m].append(abs(est - theta) / max(theta, 1e-6))
            for m in methods:
                seed_mare[m].append(float(np.mean(per_method_err[m])))

        wls = np.array(seed_mare["Ours (WLS)"])
        for m in methods:
            arr = np.array(seed_mare[m])
            if m == "Ours (WLS)":
                t_p = wx_p = np.nan
                sig = False
            else:
                t_p = float(ttest_rel(wls, arr, alternative="less").pvalue)
                diff = wls - arr
                wx_p = 1.0 if np.allclose(diff, 0.0) else \
                    float(wilcoxon(wls, arr, alternative="less").pvalue)
                sig = bool(t_p < 0.05 and wx_p < 0.05)
            rows.append({
                "method":      m,
                "frac":        frac,
                "mare_mean":   float(arr.mean()),
                "mare_std":    float(arr.std()),
                "t_p":         t_p,
                "wilcoxon_p":  wx_p,
                "significant": sig,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp9 — Generalisation across soil regimes (domain shift)
# ---------------------------------------------------------------------------

def run_exp9() -> pd.DataFrame:
    """Run the manipulation-resistance test on each LUCAS land-cover class.

    Cropland, Grassland, Woodland (and any class with >=500 samples) have
    increasing OC level and variance, providing a real domain-shift battery.
    For each class we report the WLS noise floor (frac=0) and the MARE under
    50% manipulation, alongside the precision-weighted Boomitra baseline, to
    show that the manipulation invariance generalises across soil regimes.

    Returns DataFrame: land_cover, n_samples, median_oc,
                       wls_mare_honest, wls_mare_manip, boomitra_mare_manip.
    """
    groups = load_oc_by_landcover()
    frac = 0.5

    # Order by median OC (increasing domain shift)
    order = sorted(groups.items(), key=lambda kv: np.median(kv[1]))

    rows = []
    for name, oc_g in order:
        wls_honest, wls_manip, boom_manip = [], [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(RNG_SEED + seed)
            thetas = sample_theta(oc_g, N_THETA, rng)
            e_h, e_m, e_b = [], [], []
            for theta in thetas:
                # honest
                r0 = get_noisy_reports(AGENTS, theta, 0.0, rng)
                try:
                    th0 = estimate_theta_s(r0, AGENTS)[0]
                except np.linalg.LinAlgError:
                    th0 = aggregate_weighted(r0)
                e_h.append(abs(th0 - theta) / max(theta, 1e-6))
                # manipulated
                rm = get_noisy_reports(AGENTS, theta, frac * theta, rng)
                try:
                    thm = estimate_theta_s(rm, AGENTS)[0]
                except np.linalg.LinAlgError:
                    thm = aggregate_weighted(rm)
                e_m.append(abs(thm - theta) / max(theta, 1e-6))
                e_b.append(abs(aggregate_weighted(rm) - theta) / max(theta, 1e-6))
            wls_honest.append(float(np.mean(e_h)))
            wls_manip.append(float(np.mean(e_m)))
            boom_manip.append(float(np.mean(e_b)))
        rows.append({
            "land_cover":          name,
            "n_samples":           int(len(oc_g)),
            "median_oc":           float(np.median(oc_g)),
            "wls_mare_honest":     float(np.mean(wls_honest)),
            "wls_mare_manip":      float(np.mean(wls_manip)),
            "boomitra_mare_manip": float(np.mean(boom_manip)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp10 — Robustness to imperfect sigma / alpha calibration
# ---------------------------------------------------------------------------

CALIB_ERRORS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
N_CAL = 20   # fixed calibration draws averaged within each seed


def run_exp10(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Degrade the mechanism's knowledge of sigma_i and alpha_i and measure MARE.

    The mechanism uses calibration estimates that differ from the true values
    by a multiplicative error drawn uniformly from [1-e, 1+e] for each agent.
    A draw stays fixed over all rounds, like a real calibration, and each seed
    averages N_CAL such draws.  Two sweeps:
      param="sigma": only sigma_i are mis-estimated (affects WLS weights)
      param="alpha": only alpha_i are mis-estimated (affects source separation)
    Manipulation is fixed at frac=0.5.  This tests the perfect-calibration
    idealisation: the true reports come from the true (sigma, alpha), but the
    WLS solver is given perturbed parameters.

    Returns DataFrame: param, error, mare_mean, mare_std.
    """
    if oc is None:
        oc = load_oc()
    frac = 0.5

    rows = []
    for param in ("sigma", "alpha"):
        for err in CALIB_ERRORS:
            seed_mares = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(RNG_SEED + seed)
                # The mechanism's mis-calibrated views of the agents, drawn from
                # their own stream so the reports match the err=0 run.
                cal = np.random.default_rng(RNG_SEED + 1000 + seed).uniform(
                    1 - err, 1 + err, (N_CAL, len(AGENTS)))
                views = [
                    [
                        Agent(
                            name=a.name,
                            sigma=a.sigma * (c if param == "sigma" else 1.0),
                            alpha=max(1e-3, a.alpha * (c if param == "alpha" else 1.0)),
                        )
                        for a, c in zip(AGENTS, row)
                    ]
                    for row in cal
                ]
                thetas = sample_theta(oc, N_THETA, rng)
                rel_errors = []
                for theta in thetas:
                    reports = get_noisy_reports(AGENTS, theta, frac * theta, rng)
                    for mis_agents in views:
                        # The WLS weights use the mechanism's calibrated sigma_i.
                        weighted = [(mu, a.sigma) for (mu, _), a in zip(reports, mis_agents)]
                        try:
                            th = estimate_theta_s(weighted, mis_agents)[0]
                        except np.linalg.LinAlgError:
                            th = aggregate_weighted(weighted)
                        rel_errors.append(abs(th - theta) / max(theta, 1e-6))
                seed_mares.append(float(np.mean(rel_errors)))
            rows.append({
                "param":     param,
                "error":     err,
                "mare_mean": float(np.mean(seed_mares)),
                "mare_std":  float(np.std(seed_mares)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Exp11 — Worst-case strategic evasion adversary
# ---------------------------------------------------------------------------

EVASION_BUDGETS = np.linspace(0.0, 2.0, 21)   # per-sensor budget in units of sigma_i
EVASION_FAR     = [0.01, 0.05, 0.10]          # detector false-alarm rates


def run_exp11(oc: np.ndarray | None = None) -> pd.DataFrame:
    """Security frontier against a worst-case mechanism-aware tampering attacker.

    This experiment stress-tests the idealisation that the source is limited to
    the physical model (scalar s applied through alpha_i, which WLS makes theta
    invariant to, Exp2).  Here a stronger attacker has arbitrary per-sensor
    write access and knows alpha_i, sigma_i and the detector.  It chooses a
    perturbation d with |d_i| <= budget * sigma_i to maximise theta inflation
    while keeping s_hat below the detection threshold (an LP; see
    src.source.worst_case_evasion_bias).

    We sweep the per-sensor budget and the detector false-alarm rate (FAR).
    A tighter FAR lowers the detection threshold and shrinks the attacker's
    undetected room, at the cost of more false alarms on honest sources.

    Honest reading: the physical source achieves zero theta-bias under Diverse
    agents (Exp2), but an arbitrary-tampering attacker can still inject a
    BOUNDED undetected bias.  The bound quantifies the residual security gap
    that complementary measures (trusted execution, hardware attestation) must
    close.  The bias is reported relative to median LUCAS theta.

    Returns DataFrame: far, budget, bias_abs, bias_rel.
    """
    if oc is None:
        oc = load_oc()
    theta_med = float(np.median(oc))

    rows = []
    for far in EVASION_FAR:
        for b in EVASION_BUDGETS:
            bias = worst_case_evasion_bias(AGENTS, budget=float(b),
                                           false_alarm_rate=far)
            rows.append({
                "far":      far,
                "budget":   float(b),
                "bias_abs": float(bias),
                "bias_rel": float(bias / theta_med),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# exp_corr — Robustness to correlated sensor noise
# ---------------------------------------------------------------------------

RHO_GRID = [0.0, 0.3, 0.6, 0.9]


def run_exp_corr(oc: np.ndarray | None = None) -> pd.DataFrame:
    """WLS robustness under equicorrelated sensor noise.

    Constructs equicorrelated noise Sigma_ij = rho * sigma_i * sigma_j (i != j)
    and runs the STANDARD diagonal WLS estimator (no knowledge of Sigma).
    The diagonal estimator is unbiased for any rho because E[epsilon] = 0.

    For each rho in RHO_GRID and each manipulation fraction, reports:
      mare       -- mean absolute relative error of diagonal WLS
      std        -- std across N_SEEDS seeds

    Expected: MARE is flat in frac for every rho (manipulation invariance holds),
    and the noise floor rises with rho. Detection power under correlation is
    computed in closed form by src/exp_corr_detection.py: it rises with rho,
    because s_hat is a contrast that cancels the common noise component.

    Returns DataFrame: rho, frac, mare, std.
    """
    if oc is None:
        oc = load_oc()

    fracs = np.linspace(0.0, S_FRAC_MAX, N_S)

    rows = []
    for rho in RHO_GRID:
        for frac in fracs:
            seed_mares = []
            for seed in range(N_SEEDS):
                rng = np.random.default_rng(RNG_SEED + seed)
                thetas = sample_theta(oc, N_THETA, rng)
                rel_errors = []
                for theta in thetas:
                    s = frac * theta
                    reports = get_noisy_reports_corr(AGENTS, theta, s, rng, rho=rho)
                    try:
                        theta_hat, _ = estimate_theta_s(reports, AGENTS)
                    except np.linalg.LinAlgError:
                        theta_hat = aggregate_weighted(reports)
                    rel_errors.append(abs(theta_hat - theta) / max(theta, 1e-6))
                seed_mares.append(float(np.mean(rel_errors)))
            rows.append({
                "rho":  float(rho),
                "frac": float(frac),
                "mare": float(np.mean(seed_mares)),
                "std":  float(np.std(seed_mares)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(save_csv: bool = True) -> dict[str, pd.DataFrame]:
    """Run all sixteen experiments. Optionally save CSVs to src/data/."""
    from pathlib import Path

    print("Loading LUCAS 2018 OC data...")
    oc = load_oc()
    print(f"  {len(oc):,} valid OC samples  "
          f"mean={oc.mean():.1f} g/kg  std={oc.std():.1f} g/kg  "
          f"median={np.median(oc):.1f} g/kg")

    results = {}
    labels = [
        ("exp1", lambda: run_exp1(oc)),
        ("exp2", lambda: run_exp2(oc)),
        ("exp3", lambda: run_exp3(oc)),
        ("exp4", lambda: run_exp4(oc)),
        ("exp5", lambda: run_exp5(oc)),
        ("exp6", lambda: run_exp6(oc)),
        ("exp7", lambda: run_exp7(oc)),
        ("exp8", lambda: run_exp8(oc)),
        ("exp9", lambda: run_exp9()),
        ("exp10", lambda: run_exp10(oc)),
        ("exp11", lambda: run_exp11(oc)),
        ("exp_corr", lambda: run_exp_corr(oc)),
        ("exp_collusion", lambda: run_exp_collusion(oc)),
        ("exp_misspec", lambda: run_exp_misspec()),
        ("exp_alpha_mismatch", lambda: run_exp_alpha_mismatch(oc)),
        ("exp_regression_baseline", lambda: run_exp_regression_baseline(oc)),
    ]
    for name, fn in labels:
        print(f"Running {name}...")
        df = fn()
        results[name] = df
        if save_csv:
            out = Path(__file__).parent / "data" / f"{name}_data.csv"
            df.to_csv(out, index=False)
            print(f"  saved -> {out}")

    print("All experiments complete.")
    return results


if __name__ == "__main__":
    run_all()
