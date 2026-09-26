"""Per-district forest plot: where each policy wins, and where it loses.

A mean over eighteen correlated Warsaw districts is weak evidence on its own,
and "14 of 18" hides which four and by how much. This draws the sorted
per-district contrasts against Equal Shares and adds a bootstrap interval for
their pooled mean, so the losses remain visible rather than being aggregated
away. It is the caterpillar form of the per-series table and makes the text
checkable.

Writes iclr_paper/tex/fig_forest.{pdf,png}.
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
import pandas as pd
import seaborn as sns

from iclr_corpus import build_series_index, load_split
from iclr_env import (
    EnvConfig,
    endowment_selector,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import score_selector
from iclr_policy import linear_policy
from iclr_style import GRID, INK, POLICY, TEXT, apply_style, panel_label
from iclr_train import _load_series_data
from gen_iclr_numbers import (
    _exact_frontier_point,
    _load_endowment_seed_artifact,
    _load_primary_frontier,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "iclr_paper" / "tex" / "fig_forest"
N_BOOT = 2000


def _load_policy_weights() -> Tuple[np.ndarray, np.ndarray]:
    """Load the declared seed-42 endowment and temporal outcome weights."""
    endowment = _load_endowment_seed_artifact(
        RESULTS / "iclr_train" / "run_main_seed42.json"
    )
    outcome = _exact_frontier_point(_load_primary_frontier(), 1.0)
    return np.array(endowment["best_weights"]), np.array(outcome["weights"])


def _per_series(sel, data, cfg) -> Dict[str, Dict[int, float]]:
    """Per-district, per-edition worst-group CSD on held-out editions."""
    out: Dict[str, Dict[int, float]] = {}
    for d in data:
        if not d.test_years:
            continue
        ep = rollout_selector(d.ref, sel, score_years=d.test_years, cfg=cfg,
                              instances=d.all_years)
        out[d.ref.key.split("/")[-1]] = {
            y.year: y for y in ep.years if y.scored
        } and {"worst": ep.worst_csd}
    return out


def main() -> None:
    apply_style()
    cfg = EnvConfig()
    data = _load_series_data(load_split("temporal_2022"), build_series_index())

    endow_w, head = _load_policy_weights()

    arms = [
        ("Learned endowment", endowment_selector(linear_policy(endow_w), cfg)),
        ("Learned outcome", score_selector(head)),
    ]
    mes = {k: v["worst"] for k, v in _per_series(
        endowment_selector(uniform_policy, cfg), data, cfg).items()}

    fig, axes = plt.subplots(1, 2, figsize=(TEXT, 3.5), sharey=True)
    order: List[str] = []
    for ax, (name, sel) in zip(axes, arms):
        got = {k: v["worst"] for k, v in _per_series(sel, data, cfg).items()}
        diffs = {k: got[k] - mes[k] for k in got}
        if not order:
            order = sorted(diffs, key=lambda k: diffs[k])
        style = POLICY[name]

        wins = sum(1 for k in order if diffs[k] < 0)
        ys = np.arange(len(order))
        vals = np.array([diffs[k] for k in order])
        effect_frame = pd.DataFrame(
            {
                "district": order,
                "effect": vals,
                "direction": ["improves" if value < 0 else "worsens" for value in vals],
            }
        )
        effect_palette = {
            "improves": style.color,
            "worsens": POLICY["Greedy (count)"].color,
        }

        ax.axvline(0, color=INK, lw=0.8, zorder=2)
        sns.barplot(
            data=effect_frame,
            x="effect",
            y="district",
            hue="direction",
            order=order,
            palette=effect_palette,
            errorbar=None,
            dodge=False,
            legend=False,
            orient="h",
            width=0.62,
            saturation=1,
            alpha=0.85,
            zorder=3,
            ax=ax,
        )
        sns.scatterplot(
            data=effect_frame,
            x="effect",
            y="district",
            s=11,
            color=INK,
            legend=False,
            zorder=4,
            ax=ax,
        )

        # pooled estimate with a bootstrap interval, drawn as a diamond
        rng = np.random.default_rng(42)
        boots = vals[rng.integers(0, vals.size, (N_BOOT, vals.size))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ymid = len(order) + 0.9
        ax.plot([lo, hi], [ymid, ymid], color=INK, lw=1.1, zorder=5)
        ax.plot(vals.mean(), ymid, marker="D", ms=5.5, color=style.color,
                markeredgecolor=INK, markeredgewidth=0.6, zorder=6)
        ax.text(vals.mean(), ymid + 0.75, f"pooled {vals.mean():+.3f}",
                ha="center", va="bottom", fontsize=6.8, color=INK)

        ax.set_ylim(-1.0, len(order) + 2.6)
        ax.set_yticks(ys)
        ax.set_yticklabels(order, fontsize=6.4)
        ax.set_xlabel(
            "change in worst-group $\\mathrm{CSD}$ vs Equal Shares",
            labelpad=14,
        )
        ax.set_ylabel("")
        ax.set_title(f"{name}  ({wins}/{len(order)} districts improve)", pad=5)
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.grid(axis="y", visible=False)
        ax.text(0.03, -0.115, "improves $\\leftarrow$", transform=ax.transAxes,
                fontsize=6.6, color=style.color, ha="left", va="top")
        ax.text(0.97, -0.115, "$\\rightarrow$ worsens", transform=ax.transAxes,
                fontsize=6.6, color=POLICY["Greedy (count)"].color, ha="right", va="top")
        logger.info("%-20s %d/%d improve, pooled %+.4f [%+.4f, %+.4f]",
                    name, wins, len(order), vals.mean(), lo, hi)

    panel_label(axes[0], "a", dx=-0.30)
    panel_label(axes[1], "b", dx=-0.06)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=400)
    logger.info("wrote %s.{pdf,png}", OUT)


if __name__ == "__main__":
    main()
