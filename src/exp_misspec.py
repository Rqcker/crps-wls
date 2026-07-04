"""
Mean-truthfulness under a misspecified noise covariance.

Claim (Section: Strict Truthfulness, "Correlated noise"): strict
propriety of the FULL report needs the Sigma-dependent inflation
sigma_i^2 + sigma_j^2 - 2 Sigma_ij, but the reported MEAN is pinned to the
predictive mean Y_i under ANY fixed positive inflation, because the Gaussian
CRPS fixes the location independently of the scored variance. Hence the WLS
estimator, which uses only the means, stays unbiased even when Sigma is unknown
or misspecified; only the auxiliary variance elicitation degrades.

This check confirms it numerically. Under equicorrelated noise with correlation
rho, we score each agent's reported mean with the WRONG independent inflation
sigma_i^2 + sigma_j^2 (ignoring Sigma_ij) and locate the agent's expected
peer-CRPS minimizer over a grid of mean deviations. The minimizer is the
truthful Y_i (deviation 0) for every agent and every rho.

Output: src/data/exp_misspec_data.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.agents import AGENTS

_Z = np.linspace(-8.0, 8.0, 4001)       # integration grid for the peer law
_W = norm.pdf(_Z)
_DG = np.linspace(-3.0, 3.0, 1201)      # reported-mean deviation grid (g/kg)


def _expected_crps(delta: float, omega: float, tau: float) -> float:
    """E_y CRPS(N(Y_i + delta, omega^2), y), y ~ N(Y_i, tau^2). Y_i cancels."""
    y = tau * _Z
    z = (y - delta) / omega
    crps = omega * (2.0 * norm.pdf(z) + z * (2.0 * norm.cdf(z) - 1.0) - 1.0 / np.sqrt(np.pi))
    return float(np.trapz(crps * _W, _Z))


def run_exp_misspec(rhos=(0.0, 0.3, 0.6, 0.9)) -> pd.DataFrame:
    sig = np.array([a.sigma for a in AGENTS])
    n = len(sig)
    rows = []
    for rho in rhos:
        worst = 0.0
        for i in range(n):
            total = np.zeros_like(_DG)
            for j in range(n):
                if j == i:
                    continue
                tau = np.sqrt(sig[i] ** 2 + sig[j] ** 2 - 2.0 * rho * sig[i] * sig[j])
                omega = np.sqrt(sig[i] ** 2 + sig[j] ** 2)     # misspecified (independent)
                total += np.array([_expected_crps(d, omega, tau) for d in _DG])
            worst = max(worst, abs(float(_DG[int(np.argmin(total))])))
        rows.append({"rho": rho, "max_abs_mean_deviation_gkg": worst})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_exp_misspec()
    out = Path(__file__).parent / "data" / "exp_misspec_data.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"  saved {out}")
