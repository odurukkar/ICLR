"""Headline figure: approver-funded payment buys coverage, not fairness.

Table~\\ref{tab:kernel-intervention} already carries these numbers, but a table
makes the reader compare twelve rows before the asymmetry appears. The claim is
about two effects moving differently, so it belongs in a figure.

Each panel shows paired per-series effects against direct greedy fill, holding
the learned score and its features fixed and changing only the allocation
kernel. Exclusion effects sit left of zero with intervals clear of it. Deficit
effects sit on or right of zero at every welfare target. That contrast is the
result: the mechanism supplies coverage, and the fairness metric never records
the gain.

Writes iclr_paper/tex/fig_coverage.{pdf,png}.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from iclr_style import COLUMN, GRID, INK, NEGATIVE, POSITIVE, apply_style, panel_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "iclr_payment_intervention_summary.csv"
OUT = ROOT.parent / "iclr_paper" / "tex" / "fig_coverage"

# Kernels in the order the allocation gains structure, direct fill excluded
# because it is the reference every effect is measured against.
KERNELS = (
    ("static-floor", "static floor"),
    ("payment-only", "payment"),
    ("payment+completion", "payment\n+ completion"),
)
TARGETS = (("0.0", "unpenalized"), ("0.85", "0.85"), ("1.0", "1.00"))
TARGET_SHADE = (1.00, 0.72, 0.48)


def _rows() -> Dict[Tuple[str, str], Dict[str, str]]:
    with SUMMARY.open() as handle:
        table = {}
        for row in csv.DictReader(handle):
            table[(f"{float(row['floor'])}", row["kernel"])] = row
    return table


def _series(table, field: str, lo: str, hi: str) -> Dict[str, List[Tuple[float, float, float]]]:
    out: Dict[str, List[Tuple[float, float, float]]] = {}
    for target, _ in TARGETS:
        points = []
        for kernel, _label in KERNELS:
            row = table[(target, kernel)]
            points.append((float(row[field]), float(row[lo]), float(row[hi])))
        out[target] = points
    return out


def _panel(ax, data, title: str, xlabel: str, colour: str) -> None:
    offsets = np.linspace(-0.22, 0.22, len(TARGETS))
    for (target, label), offset, shade in zip(TARGETS, offsets, TARGET_SHADE):
        ys = np.arange(len(KERNELS))[::-1] + offset
        values = np.array([p[0] for p in data[target]])
        lows = np.array([p[1] for p in data[target]])
        highs = np.array([p[2] for p in data[target]])
        ax.hlines(ys, lows, highs, color=colour, alpha=shade, linewidth=1.6,
                  zorder=2, capstyle="round")
        ax.plot(values, ys, "o", color=colour, alpha=shade, markersize=4.2,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3,
                label=label if title.startswith("Exclusion") else None)
    ax.axvline(0.0, color=INK, linewidth=0.9, zorder=1)
    for boundary in np.arange(len(KERNELS))[::-1][:-1] - 0.5:
        ax.axhline(boundary, color=GRID, linewidth=0.6, zorder=0)
    ax.set_yticks(np.arange(len(KERNELS))[::-1])
    ax.set_yticklabels([label for _, label in KERNELS])
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=6)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    apply_style()
    table = _rows()
    exclusion = _series(table, "exclusion_diff_vs_direct", "exclusion_ci_lo", "exclusion_ci_hi")
    deficit = _series(table, "diff_vs_direct", "ci_lo", "ci_hi")

    fig, axes = plt.subplots(1, 2, figsize=(2 * COLUMN, 2.35), sharey=True)
    _panel(axes[0], exclusion, "Exclusion", r"change vs direct fill", NEGATIVE)
    _panel(axes[1], deficit, "Worst-group deficit", r"change vs direct fill", POSITIVE)
    for ax, letter in zip(axes, "ab"):
        panel_label(ax, letter, dx=-0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0.10, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.35, columnspacing=1.6,
               title="welfare target")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=400)
    logger.info("wrote %s.{pdf,png}", OUT)
    for target, _ in TARGETS:
        logger.info(
            "target %-4s exclusion %+.4f  deficit %+.4f (payment+completion)",
            target, exclusion[target][-1][0], deficit[target][-1][0],
        )


if __name__ == "__main__":
    main()
