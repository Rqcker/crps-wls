"""Unified figure style for all MARBLE 2026 figures: single source of truth.

Every figure imports this module so the whole paper shares one visual language.
The style follows the clean line-chart aesthetic of Qiguang Chen's figures
(e.g. fig:learning panel (c) in arXiv:2601.06002, fig:decay_curve in
arXiv:2605.16508): muted seaborn-deep palette, one distinct marker per series,
a light y-only grid, framed legends, and restrained inline annotations.

Do not hard-code colours or markers in the plot functions. Use:
  - LINE charts:  plot_style.line(ax, x, y, i, label=...)
  - BAR  charts:  PASTEL fills + EDGE outline
  - cross-figure consistency: MIDX maps each method to a fixed colour/marker.
"""

import matplotlib.pyplot as plt
from cycler import cycler

# Geometry (single-column LNCS) and base font size
W, H, DPI, FS = 3.4, 2.6, 300, 9.0

# seaborn-deep family for LINE charts; index 0 is always the hero (our method)
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
           '#937860', '#DA8BC3', '#CCB974', '#64B5CD']
# seaborn-pastel family for BAR charts (use with a dark edge)
PASTEL  = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF',
           '#DEBB9B', '#FAB0E4', '#FFFEA3', '#B9F2F0']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<']
HERO    = PALETTE[0]
GREY    = '#9aa0a6'
EDGE    = '#3a3a3a'        # bar outline / axis ink

# Fixed colour/marker index per method, so a method looks identical in every
# figure it appears in (Boomitra is always orange squares, etc.).
MIDX = {
    'Ours (WLS)': 0, 'WLS (ours)': 0, 'Ours': 0,
    'Boomitra (B4)': 1, 'Boomitra': 1,
    'Naive mean (B3)': 2, 'Naive mean': 2,
    'Median (B6)': 3, 'Median': 3,
    'OFP (B1)': 4, 'OFP': 4,
    'WeatherXM (B2)': 5, 'WeatherXM': 5,
    'Peer-CRPS (B7)': 6, 'Peer-CRPS': 6,
    'WeatherXM CQC (B8)': 7, 'WeatherXM CQC': 7,
    'WLS+alpha noise (B5)': 8,
}


def apply() -> None:
    """Install the shared rcParams. Call once before plotting."""
    plt.rcParams.update({
        'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS,
        'xtick.labelsize': FS - 0.5, 'ytick.labelsize': FS - 0.5,
        'legend.fontsize': FS - 1.3,
        'axes.linewidth': 0.8, 'axes.edgecolor': EDGE,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'axes.prop_cycle': cycler(color=PALETTE),
        'grid.linewidth': 0.5, 'grid.color': GREY, 'grid.alpha': 0.30,
        'grid.linestyle': '-',
        'lines.linewidth': 1.7, 'lines.markersize': 4.4, 'lines.markeredgewidth': 0.6,
        'legend.frameon': True, 'legend.framealpha': 0.95,
        'legend.edgecolor': '#cfcfcf', 'legend.fancybox': False,
        'legend.borderpad': 0.45, 'legend.labelspacing': 0.35,
        'figure.dpi': DPI, 'savefig.dpi': DPI, 'savefig.bbox': 'tight',
    })


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def marker(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


def _markevery(n: int, k: int = 8):
    return max(1, n // k) if n and n > 12 else None


def line(ax, x, y, i, label=None, ls='-', hero=False, band=None, **kw):
    """Plot one styled series. i selects colour+marker; hero thickens it.

    band=(lo, hi) draws a translucent uncertainty ribbon in the same colour.
    """
    c, mk = color(i), marker(i)
    try:
        n = len(x)
    except TypeError:
        n = 0
    if band is not None:
        ax.fill_between(x, band[0], band[1], color=c, alpha=0.13, lw=0,
                        zorder=(4 if hero else 2))
    ax.plot(x, y, ls, color=c, marker=mk, markevery=_markevery(n),
            lw=2.3 if hero else 1.6, markersize=4.7 if hero else 4.0,
            markeredgecolor='white', markeredgewidth=0.6,
            zorder=(5 if hero else 3), label=label, **kw)
    return c, mk


def legend(ax, **kw):
    kw.setdefault('handlelength', 2.0)
    lg = ax.legend(**kw)
    if lg is not None:
        lg.get_frame().set_linewidth(0.7)
    return lg


def save(fig, name, figs_dir) -> None:
    fig.savefig(figs_dir / name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {name}")
