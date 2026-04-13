"""
Background Evolution Plots — NMDC Inflation
Thesis-ready individual figures: phi, phi', phi'', eps, H, H'
vs e-fold number N

Usage:
    python plot_background.py <datafile>
    python plot_background.py               # tries 'background.dat' by default

Columns expected (0-indexed):
    0:N  1:phi  2:phi'  3:phi''  4:eps  5:H  6:H'  7:H''  8:K  9:Kdot  10:t

Output: plots_background/  (PDF + PNG for each quantity)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys

# ─── Matplotlib global style ──────────────────────────────────────────────────
plt.rcParams.update(
    {
        # Font — Computer Modern gives LaTeX feel; falls back gracefully
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "text.usetex": False,  # flip to True if LaTeX is on your PATH
        "mathtext.fontset": "cm",
        # Axes
        "axes.linewidth": 1.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        # Ticks — all inward, all four sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # Lines
        "lines.linewidth": 1,
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

# ─── Journal-grade, colorblind-friendly palette ───────────────────────────────
COLORS = {
    "phi": "#1f77b4",  # steel blue
    "dphi": "#d62728",  # crimson
    "ddphi": "#2ca02c",  # forest green
    "eps": "#ff7f0e",  # burnt orange
    "H": "#9467bd",  # muted purple
    "dH": "#8c564b",  # warm brown
}

# ─── Per-plot configuration ───────────────────────────────────────────────────
# col: 0-indexed column in the data file
PLOTS = [
    {
        "col": 1,
        "key": "phi",
        "ylabel": r"$\phi \;[M_\mathrm{pl}]$",
        "fname": "phi_vs_N",
        "color": COLORS["phi"],
    },
    {
        "col": 2,
        "key": "dphi",
        "ylabel": r"$\dot\phi \;[M_\mathrm{pl}\,H]$",
        "fname": "dphi_vs_N",
        "color": COLORS["dphi"],
    },
    {
        "col": 3,
        "key": "ddphi",
        "ylabel": r"$\ddot\phi \;[M_\mathrm{pl}\,H^2]$",
        "fname": "ddphi_vs_N",
        "color": COLORS["ddphi"],
    },
    {
        "col": 4,
        "key": "eps",
        "ylabel": r"$\epsilon_1$",
        "fname": "eps_vs_N",
        "color": COLORS["eps"],
        "hline": 1.0,  # dashed line marking end of inflation
    },
    {
        "col": 5,
        "key": "H",
        "ylabel": r"$H \;[M_\mathrm{pl}]$",
        "fname": "H_vs_N",
        "color": COLORS["H"],
    },
    {
        "col": 6,
        "key": "dH",
        "ylabel": r"$\dot{H} \;[M_\mathrm{pl}^2]$",
        "fname": "dH_vs_N",
        "color": COLORS["dH"],
    },
]


# ─── Helper: load data ────────────────────────────────────────────────────────
def load_data(filepath: str) -> np.ndarray:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    try:
        data = np.loadtxt(path)
    except ValueError:
        data = np.loadtxt(path, delimiter=",")
    return data


# ─── Helper: single figure ────────────────────────────────────────────────────
def make_figure(N: np.ndarray, y: np.ndarray, cfg: dict, outdir: Path) -> None:

    # Auto log-scale for eps when it spans several decades (e.g. USR dip)
    use_log = (cfg["key"] == "eps") and (y > 0).all() and (y.max() / y.min() > 100)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    ax.plot(N, y, color=cfg["color"], lw=1.8, zorder=3)

    # Optional horizontal reference line
    if "hline" in cfg:
        ax.axhline(
            cfg["hline"],
            color="k",
            lw=0.9,
            ls="--",
            alpha=0.65,
            label=r"$\epsilon_1 = 1$",
            zorder=2,
        )
        ax.legend(fontsize=11, framealpha=0.85, edgecolor="0.7")

    if use_log:
        ax.set_yscale("log")

    ax.set_xlabel(r"$N$ (e-folds)", fontsize=13)
    ax.set_ylabel(cfg["ylabel"], fontsize=13)

    ax.set_xlim(N[0], N[-1] + 70)

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    if not use_log:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()

    for ext in ("pdf", "png"):
        dpi = 300 if ext == "pdf" else 200
        fig.savefig(
            outdir / f"{cfg['fname']}.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    plt.close(fig)
    print(f"  {cfg['fname']}.pdf  +  .png")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    datafile = sys.argv[1] if len(sys.argv) > 1 else "background.dat"
    outdir = Path("plots_background")
    outdir.mkdir(exist_ok=True)

    print(f"Loading: {datafile}")
    data = load_data(datafile)
    print(f"Shape  : {data.shape[0]} rows x {data.shape[1]} cols\n")

    N = data[:, 0]
    print(f"Output → ./{outdir}/")
    for cfg in PLOTS:
        make_figure(N, data[:, cfg["col"]], cfg, outdir)

    print(f"\nDone. Six figures in {outdir.resolve()}")


if __name__ == "__main__":
    main()
