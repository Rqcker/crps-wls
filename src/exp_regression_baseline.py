"""
Regression baselines that model the source shift as a nuisance parameter.

The WLS estimator jointly fits the true value theta and the manipulation s on the
design [1 | alpha]. To isolate the estimator from the incentive layer, and to test
robustness to the loss function, we fit the same (theta, s) model by weighted least
squares (L2), a Huber loss, and an L1 loss, and measure the credit bias under
manipulation. All three model the common shift, so all three are manipulation
invariant; the invariance is a property of the estimator structure, not of the loss
or the payments. The naive aggregators of the main baselines table do not model the
shift and therefore accumulate bias.

Output: src/data/exp_regression_baseline_data.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.dataset import load_oc, sample_theta
from src.agents import AGENTS, get_noisy_reports

RNG_SEED = 42
N_THETA = 1000
N_SEEDS = 5
FRACS = [0.0, 0.5, 1.0]


def _wls(m, alpha, sigma):
    A = np.column_stack([np.ones(len(alpha)), alpha])
    W = np.diag(1.0 / sigma ** 2)
    return float((np.linalg.inv(A.T @ W @ A) @ A.T @ W @ m)[0])


def _robust(m, alpha, sigma, loss):
    def res(x):
        return (m - x[0] - alpha * x[1]) / sigma
    r = least_squares(res, [float(np.mean(m)), 0.0], loss=loss, f_scale=1.0)
    return float(r.x[0])


def run_exp_regression_baseline(oc: np.ndarray | None = None) -> pd.DataFrame:
    if oc is None:
        oc = load_oc()
    alpha = np.array([a.alpha for a in AGENTS])
    sigma = np.array([a.sigma for a in AGENTS])

    rows = []
    for frac in FRACS:
        b = {"WLS": [], "Huber": [], "L1": []}
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(RNG_SEED + seed)
            for theta in sample_theta(oc, N_THETA, rng):
                s = frac * theta
                m = np.array([mu for mu, _ in get_noisy_reports(AGENTS, theta, s, rng)])
                b["WLS"].append((_wls(m, alpha, sigma) - theta) / theta)
                b["Huber"].append((_robust(m, alpha, sigma, "huber") - theta) / theta)
                b["L1"].append((_robust(m, alpha, sigma, "soft_l1") - theta) / theta)
        rows.append({
            "frac": frac,
            "bias_wls_pct": 100.0 * float(np.mean(b["WLS"])),
            "bias_huber_pct": 100.0 * float(np.mean(b["Huber"])),
            "bias_l1_pct": 100.0 * float(np.mean(b["L1"])),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_exp_regression_baseline()
    out = Path(__file__).parent / "data" / "exp_regression_baseline_data.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"  saved {out}")
