"""
Partial-collusion stress test.

A coalition of k of the N=8 sensors coordinates to inflate the credited SOC by
adding a common upward bias c to their reported means, with the source honest
(s=0). This isolates sensor collusion from source manipulation.

We measure three quantities as the coalition grows, with members added in
decreasing order of credit influence c_theta,i (the strongest attackers first):

  over-credit   (theta_hat - theta)/theta  from the analytical WLS theta-row
  detection     P(s_hat > delta) at a 5% false-alarm rate, from SD(s_hat)
  peer penalty  mean rise in the colluders' mechanism peer-CRPS vs honest

The point is the necessity boundary of Theorem 1. Only the full coalition (k=N)
is a pure common shift along the all-ones direction: it inflates the credit fully,
leaves s_hat = 0 (undetected, because c_s . 1 = 0), and incurs zero peer-CRPS
penalty (translation equivariance leaves every pairwise score unchanged). Any
partial coalition (k<N) is either detected through s_hat or penalised through the
peer score, or both. Collusion that escapes both is exactly the uniform-shift mode
the diversity condition rules out.

Output: src/data/exp_collusion_data.csv and figures/fig14_collusion.pdf.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.dataset import load_oc, sample_theta
from src.agents import AGENTS, get_noisy_reports
from src.mechanism import peer_crps
from src.source import wls_coefficients, wls_covariance

RNG_SEED = 42
N_THETA = 500          # ground-truth draws per seed for the peer-CRPS penalty
N_SEEDS = 5            # independent seeds
INFLATE_FRAC = 0.20    # per-colluder common bias as a fraction of theta


def _mean_peer_crps(reports, idx):
    """Mean mechanism peer-CRPS of agent idx against all peers (Eq. (5) score)."""
    mu_i, sig_i = reports[idx]
    n = len(reports)
    return float(np.mean([
        peer_crps(mu_i, sig_i, reports[j][0], reports[j][1])
        for j in range(n) if j != idx
    ]))


def run_exp_collusion(oc: np.ndarray | None = None) -> pd.DataFrame:
    if oc is None:
        oc = load_oc()
    theta_med = float(np.median(oc))

    c_theta, c_s = wls_coefficients(AGENTS)
    cov = wls_covariance(AGENTS)
    sd_shat = float(np.sqrt(cov[1, 1]))
    # The penalty is one-sided (P(z) = 0 for z <= 0), so detection is too: reject
    # when s_hat > delta, with delta set to a 5% false-alarm rate.
    delta = norm.ppf(0.95) * sd_shat

    # Strongest attackers first: largest credit influence c_theta,i.
    order = list(np.argsort(-c_theta))
    n = len(AGENTS)

    rows = []
    for k in range(0, n + 1):
        coalition = set(order[:k])

        # Analytical credit inflation and detection signal for a common bias
        # c = INFLATE_FRAC * theta added on the coalition coordinates.
        # (theta_hat - theta)/theta = INFLATE_FRAC * sum_{i in C} c_theta,i.
        overcredit = INFLATE_FRAC * float(sum(c_theta[i] for i in coalition))
        shat_mean = INFLATE_FRAC * theta_med * float(sum(c_s[i] for i in coalition))
        # One-sided detection power P(s_hat > delta) at this mean shift.
        detect_power = float(1.0 - norm.cdf((delta - shat_mean) / sd_shat))

        # Simulated peer-CRPS penalty paid by the colluders.
        penalties = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(RNG_SEED + seed)
            thetas = sample_theta(oc, N_THETA, rng)
            for theta in thetas:
                base = get_noisy_reports(AGENTS, theta, 0.0, rng)
                c = INFLATE_FRAC * theta
                coll = [
                    (mu + c, sig) if i in coalition else (mu, sig)
                    for i, (mu, sig) in enumerate(base)
                ]
                if not coalition:
                    continue
                d = np.mean([
                    _mean_peer_crps(coll, i) - _mean_peer_crps(base, i)
                    for i in coalition
                ])
                penalties.append(float(d))
        penalty = float(np.mean(penalties)) if penalties else 0.0

        rows.append({
            "k": k,
            "overcredit_pct": 100.0 * overcredit,
            "detection_power_pct": 100.0 * detect_power,
            "peer_crps_penalty": penalty,
        })
    return pd.DataFrame(rows)


def plot_collusion(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import src.plot_style as ps
    ps.apply()

    figs_dir = Path(__file__).parent.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    c_oc, c_pen = ps.color(0), ps.color(1)

    fig, ax1 = plt.subplots(figsize=(ps.W, ps.H))
    k = df["k"].to_numpy()
    ax1.plot(k, df["overcredit_pct"], "-", color=c_oc, marker="o", markersize=4.4,
             markeredgecolor="white", markeredgewidth=0.6, lw=2.0,
             label="Over-credit (left)")
    ax1.set_xlabel("Coalition size $k$ (of 8)")
    ax1.set_ylabel("Over-credit (%)", color=c_oc)
    ax1.tick_params(axis="y", labelcolor=c_oc)
    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(k, df["peer_crps_penalty"], "--", color=c_pen, marker="^", markersize=4.2,
             markeredgecolor="white", markeredgewidth=0.6, lw=1.8,
             label="Peer-CRPS penalty (right)")
    ax2.axhline(0.0, color=ps.GREY, ls=":", lw=0.8)
    ax2.set_ylabel("Peer-CRPS penalty (g/kg)", color=c_pen)
    ax2.tick_params(axis="y", labelcolor=c_pen)
    # Necessity boundary: only the full coalition is a pure uniform shift, so it
    # alone escapes the peer-CRPS penalty (and stays undetected).
    ax2.annotate("full coalition:\nzero penalty\n(uniform shift)", xy=(8, 0.005),
                 xytext=(2.9, 0.05), fontsize=6.4, color=c_pen, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=c_pen, lw=0.8, shrinkB=9))

    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig14_collusion.pdf", figs_dir)


if __name__ == "__main__":
    oc = load_oc()
    df = run_exp_collusion(oc)
    out = Path(__file__).parent / "data" / "exp_collusion_data.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"  saved {out}")
    plot_collusion(df)
