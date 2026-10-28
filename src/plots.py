"""Figure generators for the paper.

Output: figures/  (single-column LNCS, 300 dpi).
Style:  all figures share src/plot_style.py (one visual language across the
        paper): muted seaborn-deep palette, one marker per series, light y-grid,
        framed legends. A method keeps the same colour/marker in every figure.

Figure numbers in this module are file numbers (fig2 ... fig15), not the
paper's figure numbers; README.md maps one to the other.

Note on units:
  Exp1 (fig2): Y = mean peer CRPS under the mechanism score, Eq. (5), in g/kg.
  Exp2--Exp4 (fig3--fig5, fig12): Y = mean absolute relative error / signed bias (%).
  X-axis for Exp2/Exp4: source manipulation as fraction of true theta (%).

Fast restyle without re-running experiments:
    python -c "import sys; sys.path.insert(0,'.'); from src.plots import replot_from_csv; replot_from_csv()"
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

import src.plot_style as ps

_FIGS_DIR = Path(__file__).parent.parent / "figures"
_FIGS_DIR.mkdir(parents=True, exist_ok=True)
_N_SEEDS = 5
_PCT = ticker.PercentFormatter(xmax=1.0, decimals=0)

ps.apply()


# ---------------------------------------------------------------------------
# Fig 2 — Exp1: equilibrium and truthfulness
# ---------------------------------------------------------------------------

def plot_exp1(df: pd.DataFrame) -> None:
    """Fig 2: expected peer CRPS vs reporting deviation delta.

    The minimum (best score) is at delta=0 for every agent. Payment is
    theta-independent, so the curves are deterministic.
    """
    # The eight sensors give near-identical U-curves, so eight overlapping lines
    # are unreadable. Show their mean and the across-sensor range instead.
    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    piv = df.pivot_table(index="delta", columns="agent", values="payment")
    d = piv.index.values
    vals = -piv.values                         # expected peer CRPS, (n_delta, 8)
    mean, lo, hi = vals.mean(1), vals.min(1), vals.max(1)
    ax.fill_between(d, lo, hi, color=ps.HERO, alpha=0.16, lw=0,
                    label="All 8 sensors (range)")
    ax.plot(d, mean, "-", color=ps.HERO, lw=2.2, marker="o",
            markevery=ps._markevery(len(d)), markersize=4.4,
            markeredgecolor="white", markeredgewidth=0.6, label="Mean", zorder=5)
    ax.axvline(0, color=ps.GREY, linestyle="--", linewidth=0.9, zorder=1)
    imin = int(np.argmin(mean))
    ax.annotate("truthful $\\delta{=}0$\nminimises every sensor",
                xy=(0, mean[imin]), xytext=(0.50, 0.40), textcoords="axes fraction",
                fontsize=7.2, color=ps.HERO, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=ps.HERO, lw=0.8))
    ax.set_xlabel(r"Reporting deviation $\delta$ (g/kg)")
    ax.set_ylabel("Expected peer CRPS (lower = better)")
    ax.grid(axis="y")
    ps.legend(ax, loc="upper center")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig2_equilibrium.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 3 — Exp2: manipulation resistance (diverse vs uniform)
# ---------------------------------------------------------------------------

def plot_exp2(df: pd.DataFrame) -> None:
    """Fig 3: MARE vs manipulation fraction. Diverse stays at the noise floor;
    uniform grows linearly (diversity condition fails)."""
    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    spec = [("Diverse", 0, "-", True), ("Uniform", 3, "--", False)]
    for key, i, lstyle, hero in spec:
        s = df[df["setting"] == key].sort_values("frac")
        se = s["std"].values / np.sqrt(_N_SEEDS)
        ps.line(ax, s["frac"].values, s["mare"].values, i, label=key, ls=lstyle,
                hero=hero, band=(s["mare"].values - se, s["mare"].values + se))
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel("Manipulation (fraction of SOC)")
    ax.set_ylabel("Mean abs. relative error")
    ax.grid(axis="y")
    ps.legend(ax, loc="upper left")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig3_manipulation.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 4 — Exp3: sensor diversity necessity (log-log)
# ---------------------------------------------------------------------------

def plot_exp3(df: pd.DataFrame) -> None:
    """Fig 4: MARE vs alpha spread on log-log axes. As spread -> 0 the error
    diverges as O(1/spread); the dotted line is the theoretical slope."""
    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    s = df.sort_values("alpha_spread")
    se = s["std"].values / np.sqrt(_N_SEEDS)
    ps.line(ax, s["alpha_spread"].values, s["mare"].values, 0, label="Observed",
            hero=True, band=(np.maximum(s["mare"].values - se, 1e-4),
                             s["mare"].values + se))
    spreads = s["alpha_spread"].values
    ref = np.argmin(np.abs(spreads - 0.5))
    ref_s, ref_m = spreads[ref], float(s.iloc[ref]["mare"])
    ax.plot(spreads, ref_m * (ref_s / spreads), color=ps.GREY, linestyle=":",
            linewidth=1.1, label=r"Theory $\propto 1/$spread", zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sensitivity spread (diversity)")
    ax.set_ylabel("Mean abs. relative error")
    ax.yaxis.set_major_formatter(_PCT)
    ax.grid(axis="y", which="both")
    ps.legend(ax, loc="upper right")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig4_diversity.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 5 — Exp4: baseline comparison (MARE)
# ---------------------------------------------------------------------------

_LBL4 = {
    "Ours (WLS)": "WLS (ours)", "OFP (B1)": "OFP", "WeatherXM (B2)": "WeatherXM",
    "Naive mean (B3)": "Naive mean", "Boomitra (B4)": "Boomitra",
    "WLS+alpha noise (B5)": r"WLS, noisy $\alpha$", "Median (B6)": "Median",
    "Peer-CRPS (B7)": "Peer-CRPS", "WeatherXM CQC (B8)": "WeatherXM CQC",
}
_ORDER4 = ["Ours (WLS)", "OFP (B1)", "WeatherXM (B2)", "Naive mean (B3)",
           "Boomitra (B4)", "WLS+alpha noise (B5)", "Median (B6)",
           "Peer-CRPS (B7)", "WeatherXM CQC (B8)"]


def plot_exp4(df: pd.DataFrame) -> None:
    """Fig 5: MARE vs manipulation fraction for all methods. WLS stays flat
    (manipulation-resistant); every baseline grows."""
    fig, ax = plt.subplots(figsize=(ps.W, ps.H + 0.3))
    for m in _ORDER4:
        if m not in df["method"].values:
            continue
        s = df[df["method"] == m].sort_values("frac")
        se = s["std"].values / np.sqrt(_N_SEEDS)
        ps.line(ax, s["frac"].values, s["mare"].values, ps.MIDX[m], label=_LBL4[m],
                hero=(m == "Ours (WLS)"),
                band=(s["mare"].values - se, s["mare"].values + se))
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel("Manipulation (fraction of SOC)")
    ax.set_ylabel("Mean abs. relative error")
    ax.grid(axis="y")
    ps.legend(ax, fontsize=ps.FS - 2.7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.34), handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig5_baselines.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 12 — Exp4: signed manipulation bias (the headline over-credit figure)
# ---------------------------------------------------------------------------

def plot_exp4_bias(df: pd.DataFrame) -> None:
    """Fig 12: signed over-credit E[(theta_hat-theta)/theta] vs manipulation.
    WLS stays at zero (manipulation-invariant); baselines over-credit, the
    policy-relevant failure. Shaded band = the invariance margin."""
    if "bias" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(ps.W, ps.H + 0.15))
    methods = ["Ours (WLS)", "Boomitra (B4)", "Naive mean (B3)", "Median (B6)", "OFP (B1)"]
    w = df[df["method"] == "Ours (WLS)"].sort_values("frac")
    b = df[df["method"] == "Boomitra (B4)"].sort_values("frac")
    if not w.empty and not b.empty:
        ax.fill_between(w["frac"].values, w["bias"].values, b["bias"].values,
                        color=ps.HERO, alpha=0.06, lw=0, zorder=0)
    for m in methods:
        s = df[df["method"] == m].sort_values("frac")
        if s.empty:
            continue
        ps.line(ax, s["frac"].values, s["bias"].values, ps.MIDX[m],
                label=_LBL4[m], hero=(m == "Ours (WLS)"))
    ax.axhline(0, color=ps.GREY, linewidth=0.7, zorder=0)
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel("Manipulation (fraction of SOC)")
    ax.set_ylabel(r"Systematic over-credit $E[(\hat\theta-\theta)/\theta]$")
    ax.grid(axis="y")
    if not b.empty:
        bmax = float(b["bias"].max())
        ax.annotate(f"+{bmax * 100:.0f}%", xy=(1.0, bmax), xytext=(0.80, bmax + 0.015),
                    color=ps.color(1), fontsize=8.5, fontweight="bold")
    ps.legend(ax, loc="upper left")
    fig.tight_layout(pad=0.3)
    ps.save(fig, "fig12_bias.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 6 — Exp5: on-chain stake economics (twin axes)
# ---------------------------------------------------------------------------

def plot_exp5(df: pd.DataFrame) -> None:
    """Fig 6: detection power (left) and minimum deterring stake M*/G (right)
    vs manipulation, for diverse sensors."""
    diverse = df[df["setting"] == "Diverse"].sort_values("frac")
    valid = diverse.dropna(subset=["stake_mult"])
    valid = valid[valid["stake_mult"] < 200]
    c_det, c_stk = ps.color(0), ps.color(1)

    fig, ax1 = plt.subplots(figsize=(ps.W, ps.H))
    ax1.plot(diverse["frac"].values, diverse["p_detect"].values, "-", color=c_det,
             marker="o", markevery=ps._markevery(len(diverse)), markersize=4.4,
             markeredgecolor="white", markeredgewidth=0.6, lw=2.0, label="Detection power")
    ax1.xaxis.set_major_formatter(_PCT)
    ax1.yaxis.set_major_formatter(_PCT)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Manipulation (fraction of SOC)")
    ax1.set_ylabel("Detection power (5% FAR)", color=c_det)
    ax1.tick_params(axis="y", labelcolor=c_det)
    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(valid["frac"].values, valid["stake_mult"].values, "--", color=c_stk,
             marker="s", markevery=ps._markevery(len(valid)), markersize=4.0,
             markeredgecolor="white", markeredgewidth=0.6, lw=1.8, label="Min. stake")
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(r"Min. stake $M^\star/G$", color=c_stk)
    ax2.tick_params(axis="y", labelcolor=c_stk)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ps.legend(ax1, handles=h1 + h2, labels=l1 + l2, loc="upper center")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig6_stake.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 7 — Exp6: scaling with the number of agents N (twin axes)
# ---------------------------------------------------------------------------

def plot_exp6(df: pd.DataFrame) -> None:
    """Fig 7: estimation error (left) and detection power at 50% manipulation
    (right) vs number of sensors N."""
    s = df.sort_values("n_agents")
    n = s["n_agents"].values
    c_err, c_det = ps.color(0), ps.color(1)

    fig, ax1 = plt.subplots(figsize=(ps.W, ps.H))
    ax1.plot(n, s["mare_honest"].values, "-", color=c_err, marker="o",
             markersize=4.4, markeredgecolor="white", markeredgewidth=0.6, lw=2.0)
    ax1.yaxis.set_major_formatter(_PCT)
    ax1.set_xlabel(r"Number of sensors $N$")
    ax1.set_ylabel("Mean abs. relative error", color=c_err)
    ax1.tick_params(axis="y", labelcolor=c_err)
    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(n, s["p_detect"].values, "--", color=c_det, marker="s",
             markersize=4.0, markeredgecolor="white", markeredgewidth=0.6, lw=1.8)
    ax2.yaxis.set_major_formatter(_PCT)
    ax2.set_ylabel("Detection (50% manip)", color=c_det)
    ax2.tick_params(axis="y", labelcolor=c_det)

    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig7_scaling.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 8 — Exp7: robustness to non-Gaussian noise
# ---------------------------------------------------------------------------

def plot_exp7(df: pd.DataFrame) -> None:
    """Fig 8: MARE vs manipulation for four noise shapes. All flat (WLS
    unbiasedness holds for any mean-zero noise); only the floor shifts."""
    # Gaussian and uniform noise give the same floor (56.8%), so they coincide;
    # merge their label. Zoom the y-axis so the shape-dependent floors separate.
    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    spec = [("gaussian", 0, "-", "Gaussian / Uniform", True),
            ("lognormal", 1, "-.", "Lognormal (skew)", False),
            ("student_t", 3, "--", r"Student-$t$ ($\nu{=}3$)", False)]
    for dist, i, lstyle, label, hero in spec:
        s = df[df["noise_model"] == dist].sort_values("frac")
        se = s["std"].values / np.sqrt(_N_SEEDS)
        ps.line(ax, s["frac"].values, s["mare"].values, i, label=label, ls=lstyle,
                hero=hero, band=(s["mare"].values - se, s["mare"].values + se))
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_ylim(0.46, 0.595)
    ax.set_xlabel("Manipulation (fraction of SOC)")
    ax.set_ylabel("Mean abs. relative error")
    ax.grid(axis="y")
    # Direct labels read more cleanly than a legend box for three flat lines.
    ax.text(0.5, 0.5715, "Gaussian / Uniform", color=ps.color(0), fontsize=7,
            ha="center", va="bottom")
    ax.text(0.5, 0.5505, "Lognormal (skew)", color=ps.color(1), fontsize=7,
            ha="center", va="bottom")
    ax.text(0.5, 0.5075, r"Student-$t$ ($\nu{=}3$)", color=ps.color(3), fontsize=7,
            ha="center", va="bottom")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig8_noise.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 9 — Exp9: generalisation across soil regimes (grouped bars)
# ---------------------------------------------------------------------------

def plot_exp9(df: pd.DataFrame) -> None:
    """Fig 9: WLS vs Boomitra MARE under 50% manipulation across land covers.
    WLS honest and manipulated bars coincide (invariance); only the level
    changes with the soil regime."""
    s = df.sort_values("median_oc")
    labels = [f"{r.land_cover}\n({r.median_oc:.0f})" for r in s.itertuples()]
    x = np.arange(len(s))
    w = 0.27
    groups = [("wls_mare_honest", "WLS, honest", ps.PASTEL[0]),
              ("wls_mare_manip", "WLS, 50% manip", ps.PASTEL[4]),
              ("boomitra_mare_manip", "Boomitra, 50% manip", ps.PASTEL[1])]
    fig, ax = plt.subplots(figsize=(2.0 * ps.W, ps.H))
    for j, (col, lab, c) in enumerate(groups):
        ax.bar(x + (j - 1) * w, s[col].values, w, color=c, edgecolor=ps.EDGE,
               linewidth=0.7, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=ps.FS - 2)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel("Land cover (median SOC, g/kg)")
    ax.set_ylabel("Mean abs. relative error")
    ax.grid(axis="y")
    ps.legend(ax, fontsize=ps.FS - 2)
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig9_domainshift.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 10 — Exp10: robustness to calibration error
# ---------------------------------------------------------------------------

def plot_exp10(df: pd.DataFrame) -> None:
    """Fig 10: MARE vs calibration error for sigma and alpha mis-estimation.
    Nearly flat: unbiasedness does not depend on exact weights."""
    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    spec = [("sigma", 0, "-", r"$\sigma_i$ mis-calibrated", True),
            ("alpha", 3, "--", r"$\alpha_i$ mis-calibrated", False)]
    for param, i, lstyle, label, hero in spec:
        s = df[df["param"] == param].sort_values("error")
        se = s["mare_std"].values / np.sqrt(_N_SEEDS)
        ps.line(ax, s["error"].values, s["mare_mean"].values, i, label=label,
                ls=lstyle, hero=hero,
                band=(s["mare_mean"].values - se, s["mare_mean"].values + se))
    ax.xaxis.set_major_formatter(_PCT)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Calibration error (uniform $\pm$)")
    ax.set_ylabel("Mean abs. relative error")
    ax.grid(axis="y")
    ps.legend(ax, fontsize=ps.FS - 1)
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig10_calibration.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 11 — Exp11: worst-case evasion security frontier
# ---------------------------------------------------------------------------

def plot_exp11(df: pd.DataFrame) -> None:
    """Fig 11: maximum undetected theta inflation vs attacker budget. Tighter
    false-alarm rates shrink the attacker's room. The physical source achieves
    zero bias (reference line)."""
    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    spec = [(0.01, 2, "-", "FAR 1%"), (0.05, 3, "--", "FAR 5%"),
            (0.10, 7, "-.", "FAR 10%")]
    for far, i, lstyle, label in spec:
        s = df[df["far"] == far].sort_values("budget")
        ps.line(ax, s["budget"].values, s["bias_rel"].values, i, label=label, ls=lstyle)
    ax.axhline(0.0, color=ps.HERO, linewidth=2.0, zorder=4,
               label="Physical source (Exp2)")
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel(r"Attacker budget (per-sensor, units of $\sigma_i$)")
    ax.set_ylabel("Max undetected SOC inflation")
    ax.grid(axis="y")
    ps.legend(ax, fontsize=ps.FS - 2, loc="upper left")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig11_evasion.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Fig 13 — Real models on real perturbed spectra (needs trained models)
# ---------------------------------------------------------------------------

def plot_real_manip(df: pd.DataFrame) -> None:
    """Fig 13: systematic bias of real model predictions under real wetting.
    Baselines collapse as wetting grows; WLS stays in a narrow band (invariance
    on real, nonlinear responses). The nonzero level is the sensors' own bias."""
    fig, ax = plt.subplots(figsize=(ps.W, ps.H))
    methods = ["Ours (WLS)", "Boomitra", "Naive mean", "Median"]
    for m in methods:
        s = df[df["method"] == m].sort_values("w")
        if s.empty:
            continue
        ps.line(ax, s["w"].values, s["bias"].values, ps.MIDX[m], label=m,
                hero=(m == "Ours (WLS)"))
    ax.axhline(0, color=ps.GREY, linewidth=0.7, alpha=0.7, zorder=0)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_xlabel(r"Real wetting perturbation $w$")
    ax.set_ylabel(r"Systematic bias $E[(\hat\theta-\theta)/\theta]$")
    ax.grid(axis="y")
    ps.legend(ax, fontsize=ps.FS - 1.5, loc="center right")
    fig.tight_layout(pad=0.4)
    ps.save(fig, "fig13_real_manip.pdf", _FIGS_DIR)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def plot_all(results: dict) -> None:
    """Generate all figures from the dict returned by run_all()."""
    print("Generating figures...")
    plot_exp1(results["exp1"])
    plot_exp2(results["exp2"])
    plot_exp3(results["exp3"])
    plot_exp4(results["exp4"])
    plot_exp4_bias(results["exp4"])
    plot_exp5(results["exp5"])
    plot_exp6(results["exp6"])
    plot_exp7(results["exp7"])
    plot_exp9(results["exp9"])
    plot_exp10(results["exp10"])
    plot_exp11(results["exp11"])
    if "exp_collusion" in results:
        from src.exp_collusion import plot_collusion
        plot_collusion(results["exp_collusion"])
    if "exp_alpha_mismatch" in results:
        from src.exp_alpha_mismatch import plot_alpha_mismatch
        plot_alpha_mismatch(results["exp_alpha_mismatch"])
    print(f"All figures saved to {_FIGS_DIR}")


def replot_from_csv(data_dir=None) -> None:
    """Regenerate every figure from saved CSVs (no experiment re-run).

    Restyles all data-backed figures in seconds. Figures whose data needs the
    trained models (real_manip, spectral) must be regenerated on the cluster.
    """
    d = Path(data_dir) if data_dir else (Path(__file__).parent / "data")
    print(f"Restyling figures from {d} ...")

    def go(csv, fn):
        p = d / csv
        if not p.exists():
            print(f"  missing {csv} (skip)")
            return
        try:
            fn(pd.read_csv(p))
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {csv}: {exc}")

    go("exp1_data.csv", plot_exp1)
    go("exp2_data.csv", plot_exp2)
    go("exp3_data.csv", plot_exp3)
    p4 = d / "exp4_data.csv"
    if p4.exists():
        df4 = pd.read_csv(p4)
        plot_exp4(df4)
        plot_exp4_bias(df4)
    go("exp5_data.csv", plot_exp5)
    go("exp6_data.csv", plot_exp6)
    go("exp7_data.csv", plot_exp7)
    go("exp9_data.csv", plot_exp9)
    go("exp10_data.csv", plot_exp10)
    go("exp11_data.csv", plot_exp11)
    try:
        from src.exp_collusion import plot_collusion
        go("exp_collusion_data.csv", plot_collusion)
    except Exception as exc:  # noqa: BLE001
        print(f"  collusion skip: {exc}")
    try:
        from src.exp_alpha_mismatch import plot_alpha_mismatch
        go("exp_alpha_mismatch_data.csv", plot_alpha_mismatch)
    except Exception as exc:  # noqa: BLE001
        print(f"  alpha_mismatch skip: {exc}")
    rm = Path(__file__).parent.parent / "ossl_data" / "real_manipulation.csv"
    if rm.exists():
        try:
            plot_real_manip(pd.read_csv(rm))
        except Exception as exc:  # noqa: BLE001
            print(f"  real_manip skip: {exc}")
    else:
        print("  ossl_data/real_manipulation.csv missing (re-run on cluster)")
    print(f"Done. Figures in {_FIGS_DIR}")


if __name__ == "__main__":
    replot_from_csv()
