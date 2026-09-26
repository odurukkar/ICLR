"""Two figures the paper needs and did not have: a time axis, and behaviour.

(a) Deficit trajectories. The paper is about *repeated* budgeting and until now
    contained no time axis anywhere. This plots the worst-group cumulative share
    deficit as it accrues edition by edition, with the train/test boundary
    marked, so the reader can see that the learned policies separate from Equal
    Shares on editions the fit never saw.

(b) What the endowment map does. We refuse to interpret the fitted coefficients
    -- held-out behaviour is invariant to a fourfold change in them -- so we show
    the map's *behaviour* instead: the endowment multiple each group receives,
    relative to the uniform share, per edition. RES is drawn on the same scale
    and is substantially flatter overall, which is the visual form of the claim that its
    one-parameter tilt cannot express what the learned map does.

Writes iclr_paper/tex/fig_dynamics.{pdf,png} and fig_endowmap.{pdf,png}.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cohorts import cumulative_share_deficit, group_outcome
from iclr_corpus import build_series_index, load_split
from iclr_env import (
    EnvConfig,
    RolloutState,
    endowment_selector,
    res_policy,
    uniform_policy,
)
from iclr_outcome import score_selector
from iclr_policy import linear_policy
from iclr_train import _load_series_data
from rules import mes_with_endowments

from gen_iclr_numbers import (
    _exact_frontier_point,
    _load_endowment_seed_artifact,
    _load_primary_frontier,
)
from iclr_style import add_heatmap_sign_channel, apply_style

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TEX = ROOT.parent / "iclr_paper" / "tex"

INK = "#1a1a1a"
GRID = "#dcdcdc"
COLORS = {
    "Equal Shares": "#6b7280",
    "RES ($\\lambda=1$)": "#8e7cc3",
    "Learned endowment": "#2c6fbb",
    "Learned outcome": "#c0392b",
}

apply_style()


def _load_policy_weights() -> Tuple[np.ndarray, np.ndarray]:
    """Load the declared seed-42 endowment and temporal outcome weights."""
    endowment = _load_endowment_seed_artifact(
        RESULTS / "iclr_train" / "run_main_seed42.json"
    )
    outcome = _exact_frontier_point(_load_primary_frontier(), 1.0)
    return np.array(endowment["best_weights"]), np.array(outcome["weights"])


def _load_policies(cfg: EnvConfig):
    endow_w, head = _load_policy_weights()
    return [
        ("Equal Shares", endowment_selector(uniform_policy, cfg)),
        ("RES ($\\lambda=1$)", endowment_selector(res_policy(1.0), cfg)),
        ("Learned endowment", endowment_selector(linear_policy(endow_w), cfg)),
        ("Learned outcome", score_selector(head)),
    ], endow_w


def trajectories(data, cfg) -> Tuple[Dict[str, Dict[int, List[float]]], int]:
    """Worst-group CSD accumulated through each edition, per policy per series."""
    policies, _ = _load_policies(cfg)
    out: Dict[str, Dict[int, List[float]]] = {n: defaultdict(list) for n, _ in policies}
    boundary = 0
    common_start = max(min(d.ref.years) for d in data)
    for d in data:
        boundary = max(boundary, max(d.train_years))
        for name, sel in policies:
            state = RolloutState()
            per_year: Dict[int, dict] = {}
            for year in d.ref.years:
                inst = d.all_years.get(year)
                if inst is None:
                    continue
                winners = sel(inst, state)
                outcome = group_outcome(inst, winners, cfg.scheme)
                state.update(outcome)
                # Earlier Wawer editions warm the policy state but do not enter
                # the displayed ledger. Starting every series in 2019 prevents
                # a one-series 2016--2018 prefix from masquerading as a trend.
                if year < common_start:
                    continue
                per_year[year] = outcome
                years = sorted(per_year)
                cohorts = {c for o in per_year.values() for c in o}
                worst = max(
                    (cumulative_share_deficit((per_year[y] for y in years), c) or -9)
                    for c in cohorts
                )
                out[name][year].append(worst)
    return out, boundary


def fig_trajectories(data, cfg) -> None:
    traj, boundary = trajectories(data, cfg)
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    trajectory_rows: List[dict] = []
    envelopes: Dict[str, Tuple[List[int], List[float], List[float]]] = {}
    for name, series in traj.items():
        years = sorted(series)
        med = [float(np.median(series[y])) for y in years]
        lo = [float(np.percentile(series[y], 25)) for y in years]
        hi = [float(np.percentile(series[y], 75)) for y in years]
        trajectory_rows.extend(
            {"policy": name, "year": year, "median": median}
            for year, median in zip(years, med, strict=True)
        )
        envelopes[name] = (years, lo, hi)

    trajectory_frame = pd.DataFrame.from_records(trajectory_rows)
    for name in traj:
        policy_frame = trajectory_frame.loc[trajectory_frame["policy"] == name]
        years, lo, hi = envelopes[name]
        # RES sits almost exactly on Equal Shares -- which is the point -- so
        # dash the latter rather than let it vanish underneath.
        style = (
            {"linestyle": (0, (4, 2)), "linewidth": 1.6}
            if name == "Equal Shares"
            else {"linewidth": 1.3}
        )
        sns.lineplot(
            data=policy_frame,
            x="year",
            y="median",
            color=COLORS[name],
            label=name,
            estimator=None,
            errorbar=None,
            sort=False,
            zorder=3,
            ax=ax,
            **style,
        )
        ax.fill_between(years, lo, hi, color=COLORS[name], alpha=0.10, lw=0, zorder=2)
    ax.axvline(boundary + 0.5, color=INK, lw=0.8, ls=":", zorder=4)
    ax.text(boundary + 0.6, ax.get_ylim()[0], " held out ", fontsize=6.6,
            va="bottom", ha="left", color=INK)
    ax.set_xlabel("year"); ax.set_ylabel("worst-group $\\mathrm{CSD}$ to date")
    ax.grid(True, color=GRID, lw=0.4); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", handlelength=1.6,
              borderpad=0.1, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(TEX / f"fig_dynamics.{ext}", bbox_inches="tight", dpi=400)
    logger.info("wrote fig_dynamics (median and IQR over %d series)", len(data))


def fig_endowment_map(data, cfg) -> None:
    """Endowment multiple per group per edition: learned map against RES."""
    _, endow_w = _load_policies(cfg)
    maps = {"Learned endowment map": linear_policy(endow_w), "RES ($\\lambda=1$)": res_policy(1.0)}
    common_start = max(min(d.ref.years) for d in data)
    panels: Dict[str, Tuple[np.ndarray, List[str], List[int]]] = {}
    for name, pol in maps.items():
        acc: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        years: set = set()
        for d in data:
            state = RolloutState()
            for year in d.ref.years:
                inst = d.all_years.get(year)
                if inst is None:
                    continue
                beta = pol(inst, state)
                if year >= common_start:
                    base = inst.budget / len(inst.votes)
                    from cohorts import cohort_of
                    by_g: Dict[str, List[float]] = defaultdict(list)
                    for v, b in zip(inst.votes, beta):
                        c = cohort_of(v, cfg.scheme)
                        if c is not None:
                            by_g[c].append(b / base)
                    for c, vals in by_g.items():
                        acc[(c, year)].append(float(np.mean(vals)))
                        years.add(year)
                winners = mes_with_endowments(inst, endowments=beta, completion=cfg.completion)
                state.update(group_outcome(inst, winners, cfg.scheme))
        cohorts = sorted({c for c, _ in acc})
        yrs = sorted(years)
        grid = np.array([[np.mean(acc[(c, y)]) if acc.get((c, y)) else np.nan
                          for y in yrs] for c in cohorts])
        panels[name] = (grid, cohorts, yrs)

    spread = np.nanpercentile(
        np.concatenate([np.abs(g - 1).ravel() for g, _, _ in panels.values()]), 97
    )
    vmax = float(min(spread, 3.0))
    fig, axes = plt.subplots(
        2, 1, figsize=(3.35, 2.15), sharex=True, sharey=True, layout="constrained"
    )
    heatmap_artist = None
    for panel_index, (ax, (name, (grid, cohorts, yrs))) in enumerate(
        zip(axes, panels.items())
    ):
        sns.heatmap(
            grid,
            mask=np.isnan(grid),
            cmap="RdBu_r",
            vmin=max(0.0, 1 - vmax),
            vmax=1 + vmax,
            cbar=False,
            xticklabels=yrs if panel_index == len(axes) - 1 else False,
            yticklabels=[c.replace("|", " ") for c in cohorts],
            ax=ax,
        )
        heatmap_artist = ax.collections[0]
        add_heatmap_sign_channel(
            ax,
            grid,
            center=1.0,
            gid_prefix=f"endowment-signs:{panel_index}",
            marker_size=9.0,
        )
        ax.tick_params(axis="x", labelrotation=35)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.tick_params(axis="y", labelrotation=0)
        ax.set_title(name, pad=2, fontsize=7.2)
    assert heatmap_artist is not None
    cb = fig.colorbar(heatmap_artist, ax=axes, fraction=0.055, pad=0.02, shrink=0.96)
    cb.set_label(
        "endowment multiple $\\beta_g/(b/|N|)$\n(1 = uniform; 97th-pct clip)",
        fontsize=6.2,
    )
    cb.ax.tick_params(labelsize=6.2)
    for ext in ("pdf", "png"):
        fig.savefig(TEX / f"fig_endowmap.{ext}", bbox_inches="tight", dpi=400)
    logger.info("wrote fig_endowmap (spread learned %.3f vs RES %.3f)",
                float(np.nanstd(panels["Learned endowment map"][0])),
                float(np.nanstd(panels["RES ($\\lambda=1$)"][0])))


def main() -> None:
    cfg = EnvConfig()
    data = _load_series_data(load_split("temporal_2022"), build_series_index())
    fig_trajectories(data, cfg)
    fig_endowment_map(data, cfg)


if __name__ == "__main__":
    main()
