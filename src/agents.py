"""
Eight heterogeneous Gaussian sensor agents for SOC measurement.

Model: under source action s, agent i observes
    y_i = theta + alpha_i * s + eps_i,  eps_i ~ N(0, sigma_i^2)
and reports honestly:
    f_i = N(mu_i, sigma_i^2)  where mu_i = theta + alpha_i * s.

Parameters (both measured, not assumed; see the AGENTS comment below):
    sigma_i  -- measurement noise (g/kg), each model's 5-fold CV RMSE
    alpha_i  -- sensitivity to source manipulation, measured from a spectral
                soil-moisture perturbation

The SOC Sensor Diversity Condition requires that alpha_i are not all equal.
Here alpha values span [0.004, 1.0], satisfying the condition.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Agent:
    name: str
    sigma: float   # measurement noise std dev (g/kg)
    alpha: float   # sensitivity of reported mean to source manipulation s


# Eight ML sensor agents. Both sigma_i and alpha_i are MEASURED, not assumed:
#   sigma_i = 5-fold cross-validation RMSE (g/kg) of each model trained to predict
#             organic carbon from real LUCAS vis-NIR spectra (37,280 mineral-soil
#             samples, OC <= 100 g/kg; OSSL compilation).
#   alpha_i = each model's SOC-prediction sensitivity to a soil-moisture
#             perturbation of the spectrum, normalized so the largest is 1.0.
# Source: src/spectral_models.py, run on a university GPU cluster.
# The measured alpha_i span 0.004 to 1.000: the SOC Sensor Diversity Condition
# holds empirically, it is not assumed. (alpha is stored as |sensitivity|; the
# measured sign is negative for all but one model, i.e. wetting lowers predicted
# OC, so credit inflation comes from drying. The sign does not affect the
# mechanism, which depends only on the spread of alpha_i.)
AGENTS: list[Agent] = [
    Agent(name="CNN",          sigma=13.22, alpha=1.000),
    Agent(name="RF",           sigma=13.23, alpha=0.418),
    Agent(name="ViT",          sigma=13.41, alpha=0.538),
    Agent(name="Swin",         sigma=13.66, alpha=0.705),
    Agent(name="SAE",          sigma=14.02, alpha=0.716),
    Agent(name="SSL-SoilNet",  sigma=14.60, alpha=0.202),
    Agent(name="Linear",       sigma=14.72, alpha=0.665),
    Agent(name="CNN-LSTM-ECA", sigma=17.98, alpha=0.004),
]

# Uniform-sensitivity counterpart: same measured sigma_i, all alpha_i = 1.0
# (diversity condition fails). Used to demonstrate Theorem 2 necessity.
AGENTS_UNIFORM: list[Agent] = [
    Agent(name=a.name + "-U", sigma=a.sigma, alpha=1.0) for a in AGENTS
]


def get_report(agent: Agent, theta: float, s: float) -> tuple[float, float]:
    """Return honest noiseless report (mu_i, sigma_i) for agent under source action s.

    Used in Exp1 (equilibrium verification, clean theoretical setting).
    """
    return (theta + agent.alpha * s, agent.sigma)


def get_all_reports(
    agents: list[Agent], theta: float, s: float
) -> list[tuple[float, float]]:
    """Return honest noiseless reports for all agents. Used in Exp1."""
    return [get_report(a, theta, s) for a in agents]


def get_noisy_reports(
    agents: list[Agent],
    theta: float,
    s: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Return stochastic honest reports under measurement noise.

    Agent i observes y_i = theta + alpha_i * s + eps_i, eps_i ~ N(0, sigma_i^2),
    and reports N(y_i, sigma_i^2). This is the realistic setting for Exp2--Exp4.
    """
    return [
        (theta + a.alpha * s + rng.normal(0.0, a.sigma), a.sigma)
        for a in agents
    ]


# Measured span of SOC model noise (g/kg) and sensitivity, used by Exp6 to build
# ensembles of varying size N. Range taken from the eight measured agents
# (sigma 13.2-18.0 g/kg; alpha 0.2-1.0, excluding the near-zero outlier).
SIGMA_MIN: float = 13.22
SIGMA_MAX: float = 17.98
ALPHA_MIN: float = 0.20
ALPHA_MAX: float = 1.0


def make_agents(n: int) -> list[Agent]:
    """Construct n heterogeneous agents with measurement-anchored parameters.

    sigma_i is spaced linearly across the measured span [13.22, 17.98] g/kg.
    alpha_i is spaced linearly across [0.20, 1.0] so the SOC Sensor Diversity
    Condition holds for every n >= 2.  The span matches the noise range of the
    canonical eight-agent AGENTS set used in Exp1--Exp5.

    Used by Exp6 to study how the mechanism scales with the number of agents.
    """
    if n < 2:
        raise ValueError("need at least 2 agents for the diversity condition")
    sigmas = np.linspace(SIGMA_MIN, SIGMA_MAX, n)
    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, n)
    return [
        Agent(name=f"S{i+1}", sigma=float(sigmas[i]), alpha=float(alphas[i]))
        for i in range(n)
    ]


def standardized_noise(
    dist: str, size, rng: np.random.Generator
) -> np.ndarray:
    """Sample mean-zero, unit-variance noise from a named distribution.

    Distributions (all normalised to mean 0 and variance 1):
      "gaussian"   -- standard normal (baseline)
      "student_t"  -- Student-t, df=3 (heavy tails, kurtosis undefined>finite)
      "lognormal"  -- shifted lognormal (right-skewed)
      "uniform"    -- uniform (light tails, bounded)

    Used by Exp7 to test robustness of the WLS aggregation to deviations from
    the Gaussian noise assumption.  The estimator's unbiasedness depends only
    on the noise being mean-zero, not on its shape, so this isolates the effect
    of the distribution on the noise floor versus the manipulation invariance.
    """
    if dist == "gaussian":
        return rng.standard_normal(size)
    if dist == "student_t":
        df = 3.0
        x = rng.standard_t(df, size=size)
        return x / np.sqrt(df / (df - 2.0))      # var(t_df) = df/(df-2)
    if dist == "lognormal":
        sln = 0.5
        x = rng.lognormal(mean=0.0, sigma=sln, size=size)
        m = np.exp(sln**2 / 2.0)
        v = (np.exp(sln**2) - 1.0) * np.exp(sln**2)
        return (x - m) / np.sqrt(v)
    if dist == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=size)
    raise ValueError(f"unknown noise distribution: {dist}")


def get_noisy_reports_corr(
    agents: list[Agent],
    theta: float,
    s: float,
    rng: np.random.Generator,
    rho: float = 0.0,
) -> list[tuple[float, float]]:
    """Honest reports with equicorrelated measurement noise.

    Noise covariance: Sigma_ij = rho * sigma_i * sigma_j (i != j),
                      Sigma_ii = sigma_i^2.
    Samples epsilon = L z, z ~ N(0, I_N), L = chol(Sigma).

    rho=0 reduces to independent noise (same as get_noisy_reports).
    The diagonal WLS estimator is unbiased for any rho because E[epsilon]=0;
    only the noise floor and detection power change.
    """
    n = len(agents)
    sigmas = np.array([a.sigma for a in agents])
    C = rho * np.ones((n, n)) + (1.0 - rho) * np.eye(n)
    Sigma = np.outer(sigmas, sigmas) * C
    L = np.linalg.cholesky(Sigma)
    eps = L @ rng.standard_normal(n)
    return [
        (theta + agents[i].alpha * s + float(eps[i]), agents[i].sigma)
        for i in range(n)
    ]


def get_noisy_reports_dist(
    agents: list[Agent],
    theta: float,
    s: float,
    rng: np.random.Generator,
    dist: str = "gaussian",
) -> list[tuple[float, float]]:
    """Honest reports where measurement noise follows a named distribution.

    Agent i observes y_i = theta + alpha_i * s + eps_i, where eps_i has
    mean 0 and standard deviation sigma_i but follows the shape given by
    `dist`.  The agent still reports the pair (y_i, sigma_i): its point
    estimate and its (correctly calibrated) noise scale.  Only the shape of
    the residual differs from Gaussian.  Used by Exp7.
    """
    return [
        (theta + a.alpha * s + a.sigma * float(standardized_noise(dist, 1, rng)[0]),
         a.sigma)
        for a in agents
    ]
