"""Shared matplotlib style so every figure in the report reads as one system."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (colourblind-validated); assign in order, never cycle.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
NEG = "#e34948"   # losses
POS = "#2a78d6"   # gains

RESULTS = Path(__file__).resolve().parents[2] / "results"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    fig.savefig(path)
    plt.close(fig)
    return path
