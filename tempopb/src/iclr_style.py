"""One typography and colour contract for every figure in the paper.

Figures were previously styled per script. The inherited frontier plot used
matplotlib's default sans-serif while the newer figures used Times, and Equal
Shares was grey in one figure and green in another. This module keeps typography
and policy encodings consistent across the complete manuscript.

Everything here is deliberately boring: serif to match the ICLR body text, one
colour per policy used everywhere, embedded Type-42 fonts so the PDF is
searchable and selectable.

Usage:
    from iclr_style import POLICY, apply_style, panel_label
    apply_style()
    ax.plot(..., **POLICY["Equal Shares"].kw())
    panel_label(ax, "a")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

__all__ = [
    "CITY",
    "COLUMN",
    "FAIL",
    "GRID",
    "INK",
    "NEGATIVE",
    "PASS",
    "POLICY",
    "POSITIVE",
    "SPINE_PRIMARY",
    "SPINE_SECONDARY",
    "Style",
    "TEXT",
    "TIE",
    "apply_style",
    "apply_recessive_spines",
    "add_heatmap_sign_channel",
    "panel_label",
]

# ICLR single-column and full text widths in inches, so figures are never
# rescaled by \includegraphics and their fonts stay at the size set here.
COLUMN = 3.30
TEXT = 6.75

INK = "#1a1a1a"
GRID = "#dcdcdc"


@dataclass(frozen=True)
class Style:
    """How one policy is drawn, everywhere it appears."""

    color: str
    marker: str = "o"
    dashes: tuple = ()

    def kw(self, **extra) -> dict:
        out = {"color": self.color}
        if self.dashes:
            out["dashes"] = list(self.dashes)
        out.update(extra)
        return out


# One colour per policy. Blue is always "mechanism kept", red always
# "mechanism discarded", grey always the reference rule; the reader learns the
# mapping once in Figure 1 and it holds for the rest of the paper.
POLICY: Dict[str, Style] = {
    "Equal Shares":       Style("#6b7280", "^", (4, 2)),
    "RES":                Style("#8e7cc3", "v"),
    "Learned endowment":  Style("#2c6fbb", "s"),
    "Learned outcome":    Style("#c0392b", "o"),
    "Greedy (count)":     Style("#d1651a", "x"),
    "Greedy (per cost)":  Style("#c9a227", "D"),
}

# Non-policy semantics for external transfer and falsification figures. City
# identity always uses both colour and shape; pass/fail and signed effects have
# separate palettes so a policy colour is never overloaded as a verdict.
CITY: Dict[str, Style] = {
    "Poland/Katowice": Style("#0072B2", "o"),
    "Poland/Krakow": Style("#D55E00", "s"),
}
PASS = "#2A9D8F"
FAIL = "#B23A48"
TIE = "#F4F1DE"
NEGATIVE = "#3B6FB6"
POSITIVE = "#C45A3D"

# Soft-grey dashboard frame hierarchy.
# Left/bottom retain stronger scale anchoring; top/right close each panel
# without competing with data or the darker semantic reference lines.
SPINE_PRIMARY = "#777777"
SPINE_SECONDARY = "#AAAAAA"

# Semantic fills reused across figures.
FILL_MECH = "#e8f0f9"
FILL_DIRECT = "#fbeae8"
FILL_NEUTRAL = "#f3f4f6"
SHADE_BAD = "#f2d7d5"


def apply_style(base: float = 8.0) -> None:
    """Install the shared rcParams. Call once at the top of a figure script."""
    sns.set_theme(context="paper", style="ticks", palette="colorblind")
    plt.rcParams.update({
        # Type 42 keeps text selectable in the PDF rather than outlined.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 0.5,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })


def apply_recessive_spines(ax) -> None:
    """Keep a four-sided frame while making its hierarchy recessive."""
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE_PRIMARY)
        ax.spines[side].set_linewidth(0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE_SECONDARY)
        ax.spines[side].set_linewidth(0.5)


def add_heatmap_sign_channel(
    ax,
    matrix: np.ndarray,
    *,
    center: float,
    gid_prefix: str,
    marker_size: float = 10.0,
) -> None:
    """Overlay compact plus/minus markers at Seaborn heatmap cell centers."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("heatmap sign channel requires a two-dimensional matrix")
    finite = np.isfinite(values)
    deviations = np.abs(values[finite] - center)
    contrast_threshold = 0.45 * float(np.max(deviations)) if deviations.size else 0.0
    for name, mask, marker in (
        ("positive", finite & (values > center), "+"),
        ("negative", finite & (values < center), "_"),
    ):
        rows, columns = np.nonzero(mask)
        colors = [
            "white" if abs(values[row, column] - center) >= contrast_threshold else INK
            for row, column in zip(rows, columns, strict=True)
        ]
        collection = ax.scatter(
            columns + 0.5,
            rows + 0.5,
            marker=marker,
            s=marker_size,
            c=colors,
            linewidths=0.55,
            zorder=4,
        )
        collection.set_gid(f"{gid_prefix}:{name}")


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.04) -> None:
    """Bold (a)/(b) panel label in the conventional top-left position."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=plt.rcParams["font.size"] + 0.5, fontweight="bold",
            va="bottom", ha="left", color=INK)
