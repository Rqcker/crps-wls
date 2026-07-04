"""
CRPS peer neighborhood mechanism.

Implements:
  1. Single-observation CRPS for a Gaussian predictive distribution.
  2. peer_crps: the mechanism's incentive score (paper Eq. 4) -- CRPS of agent i's
     variance-inflated forecast N(mu_i, sigma_i^2 + sigma_j^2) against peer j's
     reported mean mu_j. Strictly proper; uniquely minimized by truthful reporting,
     simultaneously against every peer, because the mechanism supplies sigma_j^2.
  3. expected_peer_crps: E_{y~N(mu_j,sigma_j^2)}[CRPS(N(mu_i,sigma_i^2), y)], a helper
     used only by the peer-CRPS-reweighted baseline, not by the mechanism.
  4. A peer scoring matrix over all agent pairs.

CRPS formula (positive orientation; Gneiting et al. 2005 MWR, eq. for Gaussian;
confirmed against properscoring library crps_gaussian):
    CRPS(N(mu, sigma^2), y) = sigma * [2*phi(z) + z*(2*Phi(z) - 1) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.

Analytical expected peer CRPS (energy-score decomposition):
    E_{y~N(mu_j,sigma_j^2)}[CRPS(N(mu_i,sigma_i^2), y)]
      = rho * [2*phi(d) + d*(2*Phi(d) - 1)] - sigma_i / sqrt(pi)
    where d = (mu_i - mu_j) / rho, rho = sqrt(sigma_i^2 + sigma_j^2).

Derivation outline:
    CRPS(F, y) = E_F|X - y| - (1/2) * E_F|X - X'|
    For F = N(mu_i, sigma_i^2): (1/2)*E_F|X - X'| = sigma_i / sqrt(pi).
    X - y ~ N(mu_i - mu_j, sigma_i^2 + sigma_j^2), so
    E[|X - y|] = rho * E[|Z + d|]  (Z standard normal)
               = rho * [2*phi(d) + d*(2*Phi(d) - 1)].

Translation equivariance (key for Theorem 2 necessity):
    CRPS(N(mu + c, sigma^2), y + c) = CRPS(N(mu, sigma^2), y)  for any c.
    Consequence: if all alpha_i = alpha, adding source bias s shifts all mu_i
    by the same alpha*s, leaving all peer scores unchanged -- manipulation
    cannot be detected or penalized.
"""

import numpy as np
from scipy.stats import norm
from src.agents import Agent


_SQRT_PI = np.sqrt(np.pi)


def crps_gaussian(mu: float, sigma: float, y: float) -> float:
    """CRPS of predictive distribution N(mu, sigma^2) evaluated at observation y.

    Lower is better (CRPS is negatively oriented).
    Returns 0 when mu == y and sigma -> 0 (perfect deterministic forecast).
    """
    z = (y - mu) / sigma
    return float(sigma * (2.0 * norm.pdf(z) + z * (2.0 * norm.cdf(z) - 1.0) - 1.0 / _SQRT_PI))


def crps_gaussian_vec(
    mu: float, sigma: float, y_vec: np.ndarray
) -> np.ndarray:
    """Vectorized CRPS over an array of observations."""
    z = (y_vec - mu) / sigma
    return sigma * (2.0 * norm.pdf(z) + z * (2.0 * norm.cdf(z) - 1.0) - 1.0 / _SQRT_PI)


def expected_peer_crps(
    mu_i: float, sigma_i: float, mu_j: float, sigma_j: float
) -> float:
    """Analytical expected CRPS of agent i's report against peer j's distribution.

    Computes E_{y ~ N(mu_j, sigma_j^2)} [CRPS(N(mu_i, sigma_i^2), y)].

    NOTE: this is NOT the mechanism's incentive score. Its minimizer over agent i's
    report is the law of y, N(mu_j, sigma_j^2), i.e. the peer's parameters, not
    agent i's own observation. The mechanism's strictly proper score is peer_crps
    below. This helper feeds only the peer-CRPS-reweighted baseline.
    """
    delta = mu_i - mu_j
    rho = np.sqrt(sigma_i**2 + sigma_j**2)
    d = delta / rho
    e_abs = rho * (2.0 * norm.pdf(d) + d * (2.0 * norm.cdf(d) - 1.0))
    return float(e_abs - sigma_i / _SQRT_PI)


def peer_crps(mu_i: float, sigma_i: float, mu_j: float, sigma_j: float) -> float:
    """Mechanism incentive score (paper Eq. 4): CRPS of agent i's variance-inflated
    forecast against peer j's reported mean.

    The mechanism forms agent i's predictive for peer j's observation by inflating
    the reported variance with the known peer noise sigma_j^2, then scores it
    against the peer's reported mean m_j:
        S_ij = CRPS(N(mu_i, sigma_i^2 + sigma_j^2), mu_j).
    At the honest-source equilibrium agent i's posterior predictive for Y_j is
    N(Y_i, sigma_i^2 + sigma_j^2); by strict propriety of CRPS this is the unique
    minimizer, so mu_i = Y_i and the inflated variance matches. The peer-specific
    sigma_j^2 is supplied by the mechanism, so the single report (Y_i, sigma_i^2)
    is optimal against every peer at once.
    """
    rho = np.sqrt(sigma_i**2 + sigma_j**2)
    return crps_gaussian(mu_i, rho, mu_j)


def peer_payment(mu_i: float, sigma_i: float, mu_j: float, sigma_j: float) -> float:
    """Baseline payment helper: negative expected CRPS against the peer's
    distribution. Used by the peer-CRPS-reweighted baseline, not the mechanism."""
    return -expected_peer_crps(mu_i, sigma_i, mu_j, sigma_j)


def score_matrix(
    reports: list[tuple[float, float]], agents: list[Agent]
) -> np.ndarray:
    """Compute N x N matrix of expected peer CRPS scores.

    Entry (i, j) = E_{y~f_j}[CRPS(f_i, y)] for i != j, and 0 on diagonal.
    """
    n = len(agents)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mu_i, sigma_i = reports[i]
                mu_j, sigma_j = reports[j]
                mat[i, j] = expected_peer_crps(mu_i, sigma_i, mu_j, sigma_j)
    return mat


def agent_expected_payment(
    reports: list[tuple[float, float]], agent_idx: int
) -> float:
    """Average baseline payment to agent i across all peers j != i (helper)."""
    n = len(reports)
    mu_i, sigma_i = reports[agent_idx]
    total = 0.0
    count = 0
    for j, (mu_j, sigma_j) in enumerate(reports):
        if j != agent_idx:
            total += peer_payment(mu_i, sigma_i, mu_j, sigma_j)
            count += 1
    return total / count if count > 0 else 0.0


def agent_mechanism_payment(
    reports: list[tuple[float, float]], agent_idx: int
) -> float:
    """Average mechanism payment to agent i: negative mean peer_crps over peers.

    This is the payment under the mechanism's incentive score (paper Eq. 4). Used
    to verify the strict best response of Proposition 1 at the honest-source
    equilibrium (Exp1). Higher payment = better score; maximized when agent i
    reports truthfully.
    """
    mu_i, sigma_i = reports[agent_idx]
    total = 0.0
    count = 0
    for j, (mu_j, sigma_j) in enumerate(reports):
        if j != agent_idx:
            total += -peer_crps(mu_i, sigma_i, mu_j, sigma_j)
            count += 1
    return total / count if count > 0 else 0.0
