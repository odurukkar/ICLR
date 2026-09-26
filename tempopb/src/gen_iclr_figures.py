"""Figure 1: the fairness--welfare frontier, and Figure 2: the regularization gap.

Figure 1 is the paper's central object. It has to carry three facts at a glance:
every hand-designed rule sits in a tiny cluster; the learned frontier leaves it;
and a regularized fitted point beats the unconstrained corner on all three
reported held-out metrics.

Design constraints: vector output, colourblind-safe (Okabe-Ito), readable in
greyscale via distinct markers *and* line styles (verified by rendering the
PDF to 8-bit grey), no in-figure title (the caption does that job),
and a seed band rather than a single line, because the aggressive end of the
frontier is not seed-stable.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FRONTIER = ROOT / "results" / "iclr_frontier"
OUT_DIR = ROOT.parent / "iclr_paper" / "tex"
MANUSCRIPT_TEXT_WIDTH = 5.5
PDF_METADATA = {"CreationDate": None, "ModDate": None}

from iclr_style import (
    NEGATIVE,
    POLICY,
    POSITIVE,
    apply_recessive_spines,
    apply_style,
    panel_label,
)

apply_style(base=8.25)


def _load_seeds() -> Tuple[Dict[float, List[float]], Dict[float, List[float]], dict]:
    """Per-floor test CSD and welfare across seeds, plus one baselines block."""
    csd: Dict[float, List[float]] = {}
    welf: Dict[float, List[float]] = {}
    baselines: dict = {}
    for path in sorted(glob.glob(str(FRONTIER / "frontier_outcome_*seed*.json"))):
        d = json.loads(Path(path).read_text())
        baselines = baselines or d["baselines"]
        for p in d["points"]:
            csd.setdefault(p["floor"], []).append(p["test_worst_csd"])
            welf.setdefault(p["floor"], []).append(p["test_welfare"])
    return csd, welf, baselines


def build_frontier_figure() -> tuple[Figure, Axes]:
    """Build the frontier while exposing artists for fidelity regressions."""
    csd, welf, base = _load_seeds()
    if not csd:
        raise FileNotFoundError("no frontier json; run iclr_frontier.py first")
    floors = sorted(csd)

    x = np.array([np.mean(welf[f]) for f in floors]) / 1e6
    y = np.array([np.mean(csd[f]) for f in floors])
    ylo = np.array([np.min(csd[f]) for f in floors])
    yhi = np.array([np.max(csd[f]) for f in floors])

    # ``main.tex`` uses 0.72 text width; matching that natural width prevents
    # LaTeX from shrinking the 8-point type below the submission minimum.
    fig, ax = plt.subplots(figsize=(0.72 * MANUSCRIPT_TEXT_WIDTH, 3.25))

    # The paper's premise, drawn: every proportional / cost-aware hand-designed
    # rule lies inside a band of width ~0.004.
    hand = [
        ("Greedy (count)", "greedy-count", POLICY["Greedy (count)"]),
        ("Greedy (per cost)", "greedy-cost", POLICY["Greedy (per cost)"]),
        ("Equal Shares", "mes", POLICY["Equal Shares"]),
        ("RES", "res-1.0", POLICY["RES"]),
    ]
    band = [base[f"{k}/test"]["worst_csd"]
            for _, k, _ in hand if k != "greedy-count" and f"{k}/test" in base]
    if band:
        ax.axhspan(min(band) - 0.0015, max(band) + 0.0015, color="0.82",
                   alpha=0.55, lw=0, zorder=0)
        # No in-figure label for the band: the caption already states its width,
        # and duplicating it here only creates a collision with the legend.

    order = np.argsort(x)
    outcome_style = POLICY["Learned outcome"]
    ax.fill_between(
        x[order], ylo[order], yhi[order], color=outcome_style.color,
        alpha=0.14, lw=0, label="Learned outcome (seed range)",
    )
    sns.lineplot(
        x=x[order], y=y[order], color=outcome_style.color,
        marker=outcome_style.marker, markersize=4.8, linewidth=1.5,
        estimator=None, errorbar=None, sort=False,
        label="Learned outcome", zorder=3, ax=ax,
    )
    ax.lines[-1].set_gid("frontier-curve:learned-outcome")

    # The measured endowment frontier, drawn on the same axes. Its shortness is
    # empirical evidence for an inductive bias, not a universal reachability
    # claim about arbitrary endowment vectors.
    endow = FRONTIER / "frontier_endowment_temporal_2022_seed42.csv"
    if endow.exists():
        import csv as _csv
        with endow.open() as fh:
            rows = list(_csv.DictReader(fh))
        ex = np.array([float(r["test_welfare"]) for r in rows]) / 1e6
        ey = np.array([float(r["test_worst_csd"]) for r in rows])
        eo = np.argsort(ex)
        endowment_style = POLICY["Learned endowment"]
        sns.lineplot(
            x=ex[eo], y=ey[eo], color=endowment_style.color,
            marker=endowment_style.marker, linestyle="--", markersize=4.8,
            linewidth=1.5, estimator=None, errorbar=None, sort=False,
            label="Learned endowment", zorder=3, ax=ax,
        )
        ax.lines[-1].set_gid("frontier-curve:learned-endowment")
        ax.annotate(
            "endowment class:\nshort fitted frontier",
            xy=(ex.mean(), ey.mean()), xytext=(ex.mean() - 0.20, ey.mean() - 0.016),
            fontsize=8.25, color=endowment_style.color, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color=endowment_style.color, lw=0.8),
        )

    for label, key, style in hand:
        s = base.get(f"{key}/test")
        if not s:
            continue
        edge_kwargs = (
            {}
            if style.marker in ("x", "+")
            else {"edgecolor": "white"}
        )
        sns.scatterplot(
            x=[s["welfare"] / 1e6], y=[s["worst_csd"]],
            marker=style.marker, color=style.color, s=36,
            linewidth=0.6, label=label, zorder=4, ax=ax, **edge_kwargs,
        )
        ax.collections[-1].set_gid(f"frontier-policy:{key}")

    # Call out the degenerate corner: it is not the best point.
    i0 = floors.index(0.0) if 0.0 in floors else None
    if i0 is not None:
        ax.annotate(
            "unconstrained:\ndominated on all\n3 held-out metrics",
            xy=(x[i0], y[i0]), xytext=(x[i0] + 0.015, y[i0] - 0.012),
            fontsize=8.25, color="0.25", ha="left", va="bottom",
            arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8),
        )

    ax.set_xlabel("Utilitarian welfare on held-out years (millions of approvals)")
    ax.set_ylabel("Mean district worst-cohort deficit  (axis inverted)")
    ax.invert_yaxis()
    ax.grid(alpha=0.18, lw=0.6)
    apply_recessive_spines(ax)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for label, handle in zip(labels, handles, strict=True):
        unique.setdefault(label, handle)
    ax.legend(
        handles=list(unique.values()), labels=list(unique),
        loc="center left", bbox_to_anchor=(0.015, 0.30), frameon=False,
        ncol=1, handletextpad=0.5, borderaxespad=0.0,
    )
    fig.tight_layout(pad=0.35)
    return fig, ax


def figure_frontier(out: Path) -> None:
    fig, _ = build_frontier_figure()
    fig.savefig(
        out,
        format="pdf",
        bbox_inches=fig.bbox_inches,
        pad_inches=0,
        metadata=PDF_METADATA,
    )
    plt.close(fig)
    logger.info("wrote %s", out)


def build_regularization_figure() -> tuple[Figure, tuple[Axes, Axes]]:
    """Build regularization and all district effects without filtering rows."""
    path = FRONTIER / "frontier_outcome_temporal_2022_seed42.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run iclr_frontier.py first")
    pts = sorted(json.loads(path.read_text())["points"], key=lambda p: p["floor"])
    floors = [p["floor"] for p in pts]
    gap = [p["test_worst_csd"] - p["train_worst_csd"] for p in pts]

    effects_path = ROOT / "results" / "iclr_two_arm_per_series.csv"
    if not effects_path.exists():
        raise FileNotFoundError(
            f"missing {effects_path}; run iclr_endowment_stats.py first"
        )
    import csv as _csv
    with effects_path.open() as fh:
        effects = list(_csv.DictReader(fh))

    fig, (ax, ax_effect) = plt.subplots(
        1,
        2,
        figsize=(0.98 * MANUSCRIPT_TEXT_WIDTH, 3.25),
        gridspec_kw={"width_ratios": [0.82, 1.48]},
    )
    floor_labels = [f"{floor:g}" for floor in floors]
    gap_frame = pd.DataFrame({"floor": floors, "gap": gap})
    gap_palette = {
        floor: POSITIVE if value > 0.02 else NEGATIVE
        for floor, value in zip(floors, gap, strict=True)
    }
    sns.barplot(
        data=gap_frame,
        x="floor",
        y="gap",
        hue="floor",
        order=floors,
        hue_order=floors,
        palette=gap_palette,
        errorbar=None,
        dodge=False,
        legend=False,
        width=0.62,
        ax=ax,
    )
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(range(len(floors)))
    ax.set_xticklabels(floor_labels)
    ax.set_xlabel("Soft welfare target (relative to Equal Shares)")
    ax.set_ylabel("Test $-$ train deficit")
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    panel_label(ax, "a", dx=-0.13, dy=1.02)
    apply_recessive_spines(ax)

    # Full paired effect display: no district is filtered. Sort by the outcome
    # arm only, then show the mechanism-retaining arm on the same order.
    labels = [row["series"].split("/")[-1] for row in effects]
    endow_delta = np.array([float(row["endowment_minus_mes"]) for row in effects])
    outcome_delta = np.array([float(row["outcome_minus_mes"]) for row in effects])
    pos = np.arange(len(effects))
    ax_effect.axvline(0, color="0.35", lw=0.8, zorder=0)
    endowment_style = POLICY["Learned endowment"]
    outcome_style = POLICY["Learned outcome"]
    sns.scatterplot(
        x=endow_delta,
        y=pos + 0.14,
        marker=endowment_style.marker,
        s=4.5 ** 2,
        color=endowment_style.color,
        label="Learned endowment",
        zorder=3,
        ax=ax_effect,
    )
    ax_effect.collections[-1].set_gid("regularization-effect:learned-endowment")
    sns.scatterplot(
        x=outcome_delta,
        y=pos - 0.14,
        marker=outcome_style.marker,
        s=4.5 ** 2,
        color=outcome_style.color,
        label="Learned outcome",
        zorder=3,
        ax=ax_effect,
    )
    ax_effect.collections[-1].set_gid("regularization-effect:learned-outcome")
    ax_effect.set_yticks(pos)
    ax_effect.set_yticklabels(labels)
    ax_effect.invert_yaxis()
    ax_effect.set_xlabel(r"$\Delta$ deficit vs Equal Shares  (left is better)")
    ax_effect.grid(axis="x", alpha=0.18, lw=0.6)
    handles, labels = ax_effect.get_legend_handles_labels()
    unique = {}
    for label, handle in zip(labels, handles, strict=True):
        unique.setdefault(label, handle)
    ax_effect.legend(
        handles=list(unique.values()), labels=list(unique),
        loc="upper right", frameon=False, handletextpad=0.35, borderaxespad=0.2,
    )
    panel_label(ax_effect, "b", dx=-0.16, dy=1.02)
    apply_recessive_spines(ax_effect)

    fig.tight_layout(pad=0.35, w_pad=0.7)
    return fig, (ax, ax_effect)


def figure_regularization(out: Path) -> None:
    """Write the regularization and complete district-effect figure."""
    fig, _ = build_regularization_figure()
    fig.savefig(
        out,
        format="pdf",
        bbox_inches=fig.bbox_inches,
        pad_inches=0,
        metadata=PDF_METADATA,
    )
    plt.close(fig)
    logger.info("wrote %s", out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_frontier(OUT_DIR / "fig_frontier.pdf")
    figure_regularization(OUT_DIR / "fig_regularization.pdf")


if __name__ == "__main__":
    main()
