"""Figure 4: frozen-score allocation-kernel intervention.

The three panels hold score weights and feature definitions fixed while
comparing four allocation kernels. Realized score trajectories may diverge
through endogenous deficit histories. The panels show the floor-0 held-out
summaries over the same 18 Warsaw series; score-prioritized payment kernels are
diagnostics, not standard Equal Shares.

Writes ``iclr_paper/tex/fig_hack.{pdf,png}``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from iclr_analysis_rollout import held_out_selections
from iclr_env import EnvConfig
from iclr_outcome import project_features
from iclr_style import POLICY, Style, apply_recessive_spines, apply_style, panel_label

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "iclr_payment_intervention_summary.csv"
MANIFEST = RESULTS / "iclr_payment_intervention_manifest.json"
OUT = ROOT.parent / "iclr_paper" / "tex" / "fig_hack"
MANUSCRIPT_TEXT_WIDTH = 5.5
PDF_METADATA = {"CreationDate": None, "ModDate": None}

KERNELS: tuple[tuple[str, str, Style], ...] = (
    ("direct", "Direct fill", POLICY["Learned outcome"]),
    ("static-floor", "Static floor", POLICY["Greedy (per cost)"]),
    ("payment-only", "Payment only", POLICY["RES"]),
    ("payment+completion", "Payment + completion", POLICY["Learned endowment"]),
)

apply_style(base=8.2)


def _collect(selector, data, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return approval share, cost share, and raw spend for test winners.

    This retained helper protects the rollout-history semantics used by the
    original proxy-failure diagnostic and its regression test. Figure 4 now
    reads the immutable frozen-score intervention summaries directly.
    """
    xs: List[float] = []
    ys: List[float] = []
    ws: List[float] = []
    for selection in held_out_selections(selector, data, cfg):
        instance = selection.instance
        features = project_features(instance)
        index = {project_id: i for i, project_id in enumerate(features.project_ids)}
        voter_count = len(instance.votes) or 1
        counts = np.zeros(len(features.project_ids))
        for vote in instance.votes:
            for project_id in vote.projects:
                project_index = index.get(project_id)
                if project_index is not None:
                    counts[project_index] += 1
        for project_id in selection.winners:
            project_index = index.get(project_id)
            if project_index is None:
                continue
            xs.append(counts[project_index] / voter_count)
            ys.append(features.costs[project_index] / (instance.budget or 1.0))
            ws.append(float(features.costs[project_index]))
    return np.array(xs), np.array(ys), np.array(ws)


def _load_frozen_score_rows() -> Mapping[str, Mapping[str, float]]:
    """Load the immutable floor-0 summaries and enforce their sample identity."""
    if not SUMMARY.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("missing frozen-score intervention artifacts")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("analysis") != "frozen project-score allocation-kernel intervention":
        raise ValueError("unexpected intervention analysis identity")
    if manifest.get("n_series") != 18 or manifest.get("split") != "temporal_2022":
        raise ValueError("unexpected frozen-score sample identity")

    with SUMMARY.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if float(row["floor"]) == 0.0]
    by_kernel = {row["kernel"]: row for row in rows}
    expected = {kernel for kernel, _, _ in KERNELS}
    if len(rows) != len(expected) or set(by_kernel) != expected:
        raise ValueError("floor-0 intervention must contain each frozen kernel once")

    parsed: Dict[str, Mapping[str, float]] = {}
    for kernel, row in by_kernel.items():
        if int(row["n"]) != 18:
            raise ValueError(f"unexpected series count for {kernel}")
        parsed[kernel] = {
            "worst_csd": float(row["worst_csd"]),
            "welfare_millions": float(row["welfare"]) / 1e6,
            "exclusion": float(row["exclusion"]),
        }
    return parsed


def build_frozen_score_figure() -> tuple[Figure, tuple[Axes, Axes, Axes]]:
    """Build the three-estimand, four-kernel frozen-score figure."""
    rows = _load_frozen_score_rows()
    figure, axes_array = plt.subplots(
        1,
        3,
        figsize=(0.98 * MANUSCRIPT_TEXT_WIDTH, 3.2),
        sharey=True,
        gridspec_kw={"wspace": 0.20},
    )
    axes = tuple(axes_array)
    positions = np.array([3.0, 2.0, 1.0, 0.0])
    plot_frame = pd.DataFrame.from_records(
        {
            "kernel": kernel,
            "position": position,
            **rows[kernel],
        }
        for position, (kernel, _, _) in zip(positions, KERNELS, strict=True)
    )
    palette = {kernel: style.color for kernel, _, style in KERNELS}
    markers = {kernel: style.marker for kernel, _, style in KERNELS}
    panels = (
        (
            "worst_csd",
            "Mean district\nworst-cohort deficit",
            (0.105, 0.138),
            (0.11, 0.12, 0.13),
            lambda value: f"{value:.4f}",
        ),
        (
            "welfare_millions",
            "Total approval-count\nwelfare (millions)",
            (0.5, 1.27),
            (0.6, 0.9, 1.2),
            lambda value: f"{value:.3f}m",
        ),
        (
            "exclusion",
            "Mean district\nexclusion",
            (0.0, 0.165),
            (0.0, 0.05, 0.10, 0.15),
            lambda value: f"{100 * value:.1f}%",
        ),
    )

    for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
        metric, xlabel, limits, ticks, formatter = panel
        direct_value = rows["direct"][metric]
        axis.axvline(
            direct_value,
            color="#8a8a8a",
            lw=0.8,
            ls=(0, (3, 2)),
            zorder=0,
        )
        sns.scatterplot(
            data=plot_frame,
            x=metric,
            y="position",
            hue="kernel",
            style="kernel",
            hue_order=[kernel for kernel, _, _ in KERNELS],
            style_order=[kernel for kernel, _, _ in KERNELS],
            palette=palette,
            markers=markers,
            s=6.2 ** 2,
            edgecolor="white",
            linewidth=0.6,
            legend=False,
            zorder=3,
            ax=axis,
        )
        axis.collections[-1].set_gid(f"frozen-score:{metric}")
        for position, (kernel, label, style) in zip(positions, KERNELS):
            value = rows[kernel][metric]
            axis.annotate(
                formatter(value),
                (value, position),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color=style.color,
            )

        axis.set_xlim(*limits)
        axis.set_xticks(ticks)
        axis.set_ylim(-0.55, 3.55)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("")
        axis.set_yticks(positions)
        axis.grid(axis="x", alpha=0.35)
        axis.grid(axis="y", alpha=0.16)
        panel_label(axis, chr(ord("a") + panel_index), dx=-0.16, dy=1.02)
        apply_recessive_spines(axis)

    axes[0].set_yticklabels([label for _, label, _ in KERNELS])
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)

    figure.suptitle(
        "Frozen-score kernel intervention",
        x=0.5,
        y=0.975,
        fontsize=9.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.895,
        "score weights and feature definitions fixed (no welfare penalty) "
        "· n=18 Warsaw series",
        ha="center",
        va="center",
        fontsize=8.2,
    )
    figure.text(
        0.5,
        0.055,
        "realized score trajectories may diverge through endogenous deficit histories",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    figure.text(
        0.5,
        0.015,
        "static floor = one-shot support floor; score-prioritized diagnostic, "
        "not standard MES",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    figure.subplots_adjust(left=0.235, right=0.985, bottom=0.29, top=0.79)
    return figure, axes


def main() -> None:
    figure, _ = build_frozen_score_figure()
    for extension in ("pdf", "png"):
        metadata = PDF_METADATA if extension == "pdf" else None
        figure.savefig(
            f"{OUT}.{extension}",
            dpi=400,
            bbox_inches=figure.bbox_inches,
            pad_inches=0,
            metadata=metadata,
        )
    plt.close(figure)
    logger.info("wrote %s.{pdf,png}", OUT)


if __name__ == "__main__":
    main()
