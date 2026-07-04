"""
Alpha mis-specification ablation.

The aggregator runs WLS with sensitivities alpha_tilde = alpha + Delta in place
of the true alpha (drift, ageing, calibration error). Reports are generated with
the TRUE alpha and a manipulating source at full manipulation (s = theta). We
measure the resulting systematic credit bias E[(theta_hat - theta)/theta].

Two error models:
  random        per-sensor Delta_i ~ N(0, e * mean(alpha)), non-proportional
  proportional  alpha_tilde = (1+e) * alpha, a uniform rescaling

The theory (Section: Strict Truthfulness) predicts bias = s*(c_theta . alpha),
linear in the error and exactly zero for a proportional rescaling because the
design column space [1, alpha] = [1, c*alpha] is unchanged. The baseline
aggregators over-credit by +53% to +57% at full manipulation, so the WLS bias
under heavy alpha drift stays far below the manipulation bias it removes.

Output: src/data/exp_alpha_mismatch_data.csv and figures/fig15_alpha_mismatch.pdf.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import load_oc, sample_theta
from src.agents import AGENTS, Agent
from src.source import estimate_theta_s

RNG_SEED = 42
N_THETA = 400
N_SEEDS = 5
N_ALPHA_DRAWS = 200          # Monte Carlo over alpha_tilde realisations
ERRORS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
S_FRAC = 1.0                 # full manipulation


def _bias(agents_tilde, oc, rng) -> float:
    """Mean systematic bias E[(theta_hat - theta)/theta] using mis-specified alpha."""
    thetas = sample_theta(oc, N_THETA, rng)
    errs = []
    for theta in thetas:
        s = S_FRAC * theta
        # reports use the TRUE AGENTS (true alpha); noiseless mean isolates bias
        reports = [(theta + a.alpha * s, a.sigma) for a in AGENTS]
        try:
            theta_hat, _ = estimate_theta_s(reports, agents_tilde)
        except np.linalg.LinAlgError:
            continue
        errs.append((theta_hat - theta) / max(theta, 1e-6))
    return float(np.mean(errs))


def run_exp_alpha_mismatch(oc: np.ndarray | None = None) -> pd.DataFrame:
    if oc is None:
        oc = load_oc()
    alpha = np.array([a.alpha for a in AGENTS])
    amean = float(alpha.mean())
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for e in ERRORS:
        # proportional rescaling: exactly one design, bias is deterministic
        prop_agents = [Agent(a.name, a.sigma, a.alpha * (1.0 + e)) for a in AGENTS]
        prop_bias = _bias(prop_agents, oc, np.random.default_rng(RNG_SEED))

        # random per-sensor drift: average |bias| over alpha_tilde draws
        rand_biases = []
        for _ in range(N_ALPHA_DRAWS if e > 0 else 1):
            delta = rng.normal(0.0, e * amean, len(AGENTS))
            at = np.maximum(alpha + delta, 1e-3)      # keep positive sensitivities
            rand_agents = [Agent(a.name, a.sigma, float(at[i]))
                           for i, a in enumerate(AGENTS)]
            rand_biases.append(_bias(rand_agents, oc, np.random.default_rng(RNG_SEED)))
        rows.append({
            "alpha_error": e,
            "bias_random_mean_pct": 100.0 * float(np.mean(np.abs(rand_biases))),
            "bias_random_p90_pct": 100.0 * float(np.percentile(np.abs(rand_biases), 90)),
            "bias_proportional_pct": 100.0 * prop_bias,
        })
    return pd.DataFrame(rows)


def plot_alpha_mismatch(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import src.plot_style as ps
    ps.apply()

    figs = Path(__file__).parent.parent / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    x = 100.0 * df["alpha_error"].to_numpy()
    rand = df["bias_random_mean_pct"].to_numpy()
    ax.axhline(57.0, color=ps.GREY, ls="--", lw=1.1, label=r"Baseline bias ($+57\%$)")
    # Margin the WLS estimate keeps below the baseline over the whole drift range.
    ax.fill_between(x, rand, 57.0, color=ps.HERO, alpha=0.05, lw=0, zorder=0)
    ps.line(ax, x, rand, 0, label="Random drift", hero=True)
    ps.line(ax, x, df["bias_proportional_pct"].to_numpy(), 1,
            label="Proportional drift (exactly 0)")
    ax.set_xlabel("Sensitivity error (%)")
    ax.set_ylabel("WLS credit bias (%)")
    ax.set_ylim(-3, 62)
    ax.grid(axis="y")
    ps.legend(ax, loc="center left")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig15_alpha_mismatch.pdf", figs)


if __name__ == "__main__":
    oc = load_oc()
    df = run_exp_alpha_mismatch(oc)
    out = Path(__file__).parent / "data" / "exp_alpha_mismatch_data.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"  saved {out}")
    plot_alpha_mismatch(df)
