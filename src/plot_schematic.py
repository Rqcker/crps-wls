"""
System schematic for the mechanism (Figure 1, Model section).

A non-data pipeline diagram: the source manipulates the site, N diverse sensors
report calibrated Gaussian beliefs, CRPS peer scoring pays the agents, and a WLS
read-out issues credits on theta_hat and slashes the source stake on |s_hat|.

Output: figures/fig1_pipeline.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_C = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]


def _box(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=7.4, zorder=3, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.0, color="#444444", zorder=1,
        shrinkA=2, shrinkB=2))


def plot_schematic() -> None:
    plt.rcParams.update({"font.size": 7.4})
    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    # Stage 1: source
    _box(ax, 0.15, 1.25, 1.95, 1.0,
         "Source\nchooses $s\\geq 0$\n(conditions site)", "#EAF0FF", _C[0])
    # Stage 2: sensors
    _box(ax, 2.75, 1.15, 2.35, 1.2,
         "$N$ diverse sensors\n$Y_i=\\theta+\\alpha_i s+\\varepsilon_i$\nreport $\\mathcal{N}(m_i,v_i)$",
         "#F3EFFF", _C[1])
    # Stage 3a: CRPS peer scoring (top branch)
    _box(ax, 5.75, 2.05, 2.05, 0.95,
         "CRPS peer score\n$S_{ij}$  (pays agents)", "#FDEBF3", _C[2])
    # Stage 3b: WLS (bottom branch)
    _box(ax, 5.75, 0.5, 2.05, 0.95,
         "WLS read-out\n$(\\hat\\theta,\\hat s)$", "#FFEFE3", _C[3])
    # Stage 4: settlement
    _box(ax, 8.15, 1.15, 1.7, 1.2,
         "Credits $\\propto\\hat\\theta$\nPenalty $P(|\\hat s|)$\non source stake", "#FFF6E0", _C[4])

    # arrows
    _arrow(ax, 2.1, 1.75, 2.75, 1.75)      # source -> sensors
    _arrow(ax, 5.1, 1.9, 5.75, 2.45)        # sensors -> CRPS (agent-incentive layer)
    _arrow(ax, 5.1, 1.6, 5.75, 0.97)        # sensors -> WLS (source-settlement layer)
    _arrow(ax, 7.8, 0.97, 8.4, 1.55)        # WLS -> settle

    # diversity annotation banner
    ax.text(5.0, 0.12,
            "Strict truthfulness (sensors and source)  $\\Leftrightarrow$  "
            "sensor diversity: $\\alpha_i$ not all equal",
            ha="center", va="bottom", fontsize=7.2, style="italic",
            color="#222222")

    figs = Path(__file__).parent.parent / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    out = figs / "fig1_pipeline.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    plot_schematic()
