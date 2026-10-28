"""
Source manipulation model and theta/s estimation.

The source (carbon project operator) chooses action s >= 0, which additively
shifts agent i's observation by alpha_i * s. When s = 0, the source is honest.

Estimation: given diverse alpha_i (SOC Sensor Diversity Condition holds), we
can jointly recover theta and s from the N reports using weighted least squares.

System:  mu_i = theta + alpha_i * s  for i = 1, ..., N
         weights: w_i = 1 / sigma_i^2

Design matrix A = [[1, alpha_1], ..., [1, alpha_N]]  (shape N x 2)
WLS solution: [theta_hat, s_hat] = (A^T W A)^{-1} A^T W mu_vec

When all alpha_i are equal (diversity fails), A is rank-deficient and theta
and s cannot be separated -- this is the necessity direction of Theorem 1.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog
from src.agents import Agent
from src.mechanism import expected_peer_crps


def estimate_theta_s(
    reports: list[tuple[float, float]], agents: list[Agent]
) -> tuple[float, float]:
    """Estimate true theta and source manipulation s via weighted least squares.

    Args:
        reports: list of (mu_i, sigma_i) from each agent.
        agents:  corresponding Agent objects (needed for alpha_i).

    Returns:
        (theta_hat, s_hat): WLS estimates. Raises np.linalg.LinAlgError
        if the design matrix is rank-deficient (diversity condition fails).
    """
    mu_vec = np.array([mu for mu, _ in reports])
    sigma_vec = np.array([sigma for _, sigma in reports])
    alpha_vec = np.array([a.alpha for a in agents])

    A = np.column_stack([np.ones(len(agents)), alpha_vec])  # (N, 2)
    W = np.diag(1.0 / sigma_vec**2)

    AtWA = A.T @ W @ A
    AtWb = A.T @ W @ mu_vec

    result = np.linalg.solve(AtWA, AtWb)
    return float(result[0]), float(result[1])


def aggregate_naive(reports: list[tuple[float, float]]) -> float:
    """Baseline B3: unweighted mean of reported means. No manipulation resistance."""
    return float(np.mean([mu for mu, _ in reports]))


def aggregate_weighted(reports: list[tuple[float, float]]) -> float:
    """Baseline B4 (Boomitra-style): precision-weighted mean. No manipulation resistance."""
    mus = np.array([mu for mu, _ in reports])
    sigmas = np.array([sigma for _, sigma in reports])
    weights = 1.0 / sigmas**2
    return float(np.sum(weights * mus) / np.sum(weights))


def aggregate_threshold(
    reports: list[tuple[float, float]], k: float = 2.0
) -> float:
    """Baseline B2 (WeatherXM-style): remove outliers beyond k*median_sigma, then mean."""
    mus = np.array([mu for mu, _ in reports])
    sigmas = np.array([sigma for _, sigma in reports])
    median_mu = np.median(mus)
    median_sigma = np.median(sigmas)
    mask = np.abs(mus - median_mu) <= k * median_sigma
    if mask.sum() == 0:
        return float(np.mean(mus))
    return float(np.mean(mus[mask]))


def aggregate_median(reports: list[tuple[float, float]]) -> float:
    """Baseline B6: sample median of reported means.

    Classical robust statistics estimator — resistant to up to 50% outliers.
    Manipulation-naive: does not separate theta from source bias.
    """
    return float(np.median([mu for mu, _ in reports]))


def aggregate_peer_crps_weighted(reports: list[tuple[float, float]]) -> float:
    """Baseline B7: CRPS-peer-score-reweighted mean (R&F 2024, source-naive).

    Uses analytical expected peer CRPS to reweight agents: lower average peer
    CRPS => more consistent with peers => higher weight.  Represents applying
    Richardson & Faltings 2024 peer scoring without source-manipulation
    separation.  Partially penalises high-alpha agents at large manipulation
    fractions, but the estimate is still theta + weighted_alpha * s, so it
    grows with s and is NOT manipulation-resistant.
    """
    n = len(reports)
    mus    = np.array([mu    for mu, _     in reports])
    sigmas = np.array([sigma for _,  sigma in reports])

    avg_crps = np.array([
        np.mean([
            expected_peer_crps(mus[i], sigmas[i], mus[j], sigmas[j])
            for j in range(n) if j != i
        ])
        for i in range(n)
    ])
    weights = 1.0 / np.maximum(avg_crps, 1e-10)
    return float(np.dot(weights, mus) / weights.sum())


def aggregate_weatherxm_cqc(
    reports: list[tuple[float, float]], k: float = 2.0
) -> float:
    """Baseline B8: WeatherXM Comparative Quality Check (CQC) weighted mean.

    WeatherXM's current QoD v1.0 has no peer-comparison component (CQC is
    scheduled for v1.3 and not yet deployed as of June 2026).  This baseline
    implements the planned CQC spirit: score each agent by how consistently
    its reported mean aligns with the median of all agents, normalised by the
    median absolute deviation (MAD).

        score_i = max(0, 1 - |mu_i - median(mu)| / (k * MAD))

    where k=2.0 is a common 2-MAD window from robust statistics.
    The aggregated estimate is the score-weighted mean.  This is a heuristic
    that partially down-weights manipulated agents but does NOT separate theta
    from s, so the estimate grows with source manipulation.

    Reference: github.com/weatherxm-network/qod
    """
    mus = np.array([mu for mu, _ in reports])
    med = float(np.median(mus))
    mad = float(np.median(np.abs(mus - med)))
    if mad < 1e-10:
        return med
    scores = np.maximum(0.0, 1.0 - np.abs(mus - med) / (k * mad))
    total = float(scores.sum())
    if total < 1e-10:
        return float(np.mean(mus))
    return float(np.dot(scores, mus) / total)


def wls_covariance(agents: list[Agent]) -> np.ndarray:
    """Analytical WLS covariance matrix Cov([theta_hat, s_hat]).

    Cov = (A^T W A)^{-1}  where A_i = [1, alpha_i], W_ii = 1/sigma_i^2.

    Returns 2x2 matrix:
        [Var(theta_hat),  Cov(theta_hat, s_hat)]
        [Cov(theta_hat, s_hat),  Var(s_hat)   ]

    Raises np.linalg.LinAlgError if diversity condition fails (rank < 2).
    """
    sigmas = np.array([a.sigma for a in agents])
    alphas = np.array([a.alpha for a in agents])
    A = np.column_stack([np.ones(len(agents)), alphas])
    W = np.diag(1.0 / sigmas ** 2)
    return np.linalg.inv(A.T @ W @ A)


def manipulation_detection_power(
    agents: list[Agent],
    s: float,
    false_alarm_rate: float = 0.05,
) -> float:
    """P(s_hat > delta | true manipulation = s) at the given false-alarm rate.

    One-sided test: reject honest hypothesis when s_hat > delta, where delta
    is chosen so that P(s_hat > delta | s=0) = false_alarm_rate.

    Returns detection power in [0, 1].  Returns 0.0 if diversity fails.
    """
    try:
        cov = wls_covariance(agents)
    except np.linalg.LinAlgError:
        return 0.0
    sd_shat = float(np.sqrt(cov[1, 1]))
    delta = norm.ppf(1.0 - false_alarm_rate) * sd_shat
    return float(1.0 - norm.cdf((delta - s) / sd_shat))


def wls_coefficients(agents: list[Agent]) -> tuple[np.ndarray, np.ndarray]:
    """Return the linear WLS coefficient rows (c_theta, c_s).

    theta_hat = c_theta . mu_vec,  s_hat = c_s . mu_vec, where
    [c_theta; c_s] = (A^T W A)^{-1} A^T W.

    By construction c_theta . 1 = 1, c_theta . alpha = 0 (unbiased for theta),
    and c_s . 1 = 0, c_s . alpha = 1 (unbiased for s).

    Raises np.linalg.LinAlgError if the diversity condition fails.
    """
    sigmas = np.array([a.sigma for a in agents])
    alphas = np.array([a.alpha for a in agents])
    A = np.column_stack([np.ones(len(agents)), alphas])
    W = np.diag(1.0 / sigmas ** 2)
    M = np.linalg.inv(A.T @ W @ A) @ A.T @ W      # 2 x N
    return M[0, :], M[1, :]


def worst_case_evasion_bias(
    agents: list[Agent],
    budget: float,
    false_alarm_rate: float = 0.05,
) -> float:
    """Maximum theta_hat inflation an optimal attacker achieves while evading detection.

    Threat model (stronger than the physical source model): the attacker knows
    the mechanism (alpha_i, sigma_i, detection threshold delta) and adds an
    arbitrary per-sensor perturbation d_i to the reports, bounded by
    |d_i| <= budget * sigma_i (gross outliers beyond this are caught by
    per-sensor range/self-consistency checks).  The attacker solves

        maximize    c_theta . d            (inflate theta_hat)
        subject to  |c_s . d| <= delta     (evade manipulation detection)
                    |d_i| <= budget * sigma_i

    where delta is the 95th-percentile detection threshold on s_hat.

    Returns the maximum achievable positive bias on theta_hat (g/kg).

    For uniform agents the diversity condition fails: there is no s_hat to
    constrain, so the attacker maximises the precision-weighted mean directly
    and the bias is unbounded by detection (limited only by the budget).
    """
    sigmas = np.array([a.sigma for a in agents])

    try:
        c_theta, c_s = wls_coefficients(agents)
    except np.linalg.LinAlgError:
        # Uniform: no detection. Precision-weighted mean; push every d_i up.
        w = 1.0 / sigmas ** 2
        c_theta_unif = w / w.sum()
        return float(np.sum(c_theta_unif * budget * sigmas))

    cov = wls_covariance(agents)
    sd_shat = float(np.sqrt(cov[1, 1]))
    delta = float(norm.ppf(1.0 - false_alarm_rate) * sd_shat)

    n = len(agents)
    bounds = [(-budget * s, budget * s) for s in sigmas]
    # max c_theta.d  ==  min (-c_theta).d
    # |c_s . d| <= delta  ->  c_s.d <= delta  and  -c_s.d <= delta
    A_ub = np.vstack([c_s, -c_s])
    b_ub = np.array([delta, delta])
    res = linprog(c=-c_theta, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return float("nan")
    return float(-res.fun)


def aggregate_output_agreement(
    reports: list[tuple[float, float]], bin_width_sigmas: float = 1.5
) -> float:
    """Baseline B1 (OFP-style): majority-cluster output agreement.

    Bins the reported means using a bin width proportional to the median
    measurement noise, then returns the mean of agents in the largest cluster.
    Using a fixed number of bins (e.g. 20) is degenerate for N=8 agents since
    each bin holds at most one agent. Sensor-noise-adaptive binning is used
    instead, matching the spirit of real deployed output-agreement protocols.
    """
    mus = np.array([mu for mu, _ in reports])
    sigmas = np.array([sigma for _, sigma in reports])
    bin_width = bin_width_sigmas * float(np.median(sigmas))

    lo = mus.min() - bin_width
    hi = mus.max() + bin_width
    bins = np.arange(lo, hi + bin_width, bin_width)
    if len(bins) < 2:
        return float(np.mean(mus))

    counts, edges = np.histogram(mus, bins=bins)
    majority_bin = int(np.argmax(counts))
    in_bin = (mus >= edges[majority_bin]) & (mus < edges[majority_bin + 1])
    if in_bin.sum() == 0:
        return float(0.5 * (edges[majority_bin] + edges[majority_bin + 1]))
    return float(np.mean(mus[in_bin]))
