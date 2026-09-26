"""Render the external transfer-and-falsification evidence dashboard.

The dashboard reads only Task 2's hash-registered analysis tables. Panel (a)
keeps all failed city-level uncertainty visible, panel (b) shows all endowment
series and seeds without converting exact ties into epsilon effects, and panel
(c) shows the primary-seed CSD/exclusion tradeoff without a fitted relation.

Writes ``fig_external.{pdf,png}`` to both the paper tree and the registered
analysis bundle. Each analysis copy is copied from the paper rendering so the
paired assets are byte-identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_external_support_set_results import EFFECT_COLUMNS
import iclr_fig_external_contracts as external_contracts
import iclr_fig_external_io as external_io
from iclr_style import (
    CITY,
    GRID,
    INK,
    NEGATIVE,
    PASS,
    POSITIVE,
    TEXT,
    TIE,
    add_heatmap_sign_channel,
    apply_recessive_spines,
    apply_style,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = ROOT / "analysis-output" / "external-support-set"
DEFAULT_PAPER_TEX_DIR = ROOT.parent / "iclr_paper" / "tex"
FIGURE_STEM = "fig_external"
FIGURE_OUTPUTS = (
    "figures/fig_external.pdf",
    "figures/fig_external.png",
)
TRUSTED_TASK2_OUTPUT_SHA256 = external_contracts.TRUSTED_TASK2_OUTPUT_SHA256
CITY_ORDER = external_contracts.CITY_ORDER
SEED_ORDER = external_contracts.SEED_ORDER
SEED_OFFSETS = {1: -0.16, 2: 0.0, 42: 0.16}
SEED_FACES = {1: "white", 2: TIE, 42: None}
ANNOTATION_OFFSETS = {
    "Czyżyny": (5, -10),
    "Podlesie": (5, -10),
    "Bronowice": (-5, 7),
}


# Importing a figure generator establishes the same Type-42 contract as the
# other paper generators; build_transfer_dashboard reapplies it after callers
# alter rcParams.
apply_style()


def load_plot_tables(
    analysis_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    """Capture and load the three trusted Task 2 inputs used by the dashboard."""
    captured = external_contracts.capture_plot_inputs(
        Path(analysis_dir), trusted_hashes=TRUSTED_TASK2_OUTPUT_SHA256
    )
    return captured.effects, captured.intervals, captured.diagnostics


def _prepare_endowment_effects(effects: pd.DataFrame) -> pd.DataFrame:
    missing = set(EFFECT_COLUMNS) - set(effects.columns)
    if missing:
        raise ValueError(f"effects table is missing columns: {sorted(missing)}")
    endowment = effects.loc[effects["arm"] == "endowment"].copy()
    if endowment.empty:
        raise ValueError("effects table contains no endowment rows")
    if endowment.duplicated(["seed", "series"]).any():
        raise ValueError("endowment effects contain duplicate seed/series rows")
    per_series = endowment.groupby("series", sort=False)["seed"].agg(set)
    if any(seeds != set(SEED_ORDER) for seeds in per_series):
        raise ValueError("each retained endowment series must contain all three frozen seeds")
    return endowment


def _draw_interval_panel(ax: Axes, intervals: pd.DataFrame) -> None:
    if set(intervals["seed"]) != set(SEED_ORDER) or set(intervals["city"]) != set(CITY_ORDER):
        raise ValueError("interval panel requires both cities for all three seeds")
    if intervals.duplicated(["seed", "city"]).any() or len(intervals) != 6:
        raise ValueError("interval panel requires exactly six unique city/seed rows")

    city_y = {city: float(len(CITY_ORDER) - index - 1) for index, city in enumerate(CITY_ORDER)}
    for city in CITY_ORDER:
        style = CITY[city]
        for seed in SEED_ORDER:
            row = intervals.loc[(intervals["city"] == city) & (intervals["seed"] == seed)].iloc[0]
            estimate = float(row["difference"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            face = style.color if SEED_FACES[seed] is None else SEED_FACES[seed]
            sns.scatterplot(
                x=[estimate],
                y=[city_y[city] + SEED_OFFSETS[seed]],
                marker=style.marker,
                s=4.4 ** 2,
                facecolor=face,
                edgecolor=INK,
                linewidth=0.45,
                legend=False,
                zorder=4,
                ax=ax,
            )
            ax.errorbar(
                estimate,
                city_y[city] + SEED_OFFSETS[seed],
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt=style.marker,
                ms=0,
                mfc=face,
                mec=INK,
                mew=0.45,
                ecolor=style.color,
                elinewidth=0.9,
                capsize=1.7,
                capthick=0.7,
                color=style.color,
                zorder=4,
            )

    zero = ax.axvline(0.0, color=INK, lw=0.8, zorder=2)
    zero.set_gid("zero-reference")
    threshold = ax.axvline(-0.005, color=PASS, lw=0.9, ls=(0, (3, 2)), zorder=2)
    threshold.set_gid("macro-threshold")
    low = min(float(intervals["ci_low"].min()), -0.005)
    high = max(float(intervals["ci_high"].max()), 0.0)
    padding = 0.07 * (high - low)
    ax.set_xlim(low - padding, high + padding)
    ax.set_ylim(-0.42, 1.42)
    ax.set_yticks([city_y[city] for city in CITY_ORDER])
    ax.set_yticklabels([city.rsplit("/", 1)[-1] for city in CITY_ORDER])
    ax.set_xlabel(r"$\Delta$ CSD (learned endowment $-$ MES)")
    ax.set_title("(a) city intervals", loc="left", pad=5)
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.grid(axis="y", visible=False)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=4.0,
            markerfacecolor=(INK if SEED_FACES[seed] is None else SEED_FACES[seed]),
            markeredgecolor=INK,
            markeredgewidth=0.45,
            label=str(seed),
        )
        for seed in SEED_ORDER
    ]
    ax.legend(
        handles=handles,
        title="seed",
        ncol=3,
        loc="upper left",
        borderpad=0.1,
        handletextpad=0.15,
        columnspacing=0.45,
        labelspacing=0.2,
        fontsize=8.0,
        title_fontsize=8.0,
    )


def _heatmap_matrix(endowment: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    primary = endowment.loc[endowment["seed"] == 42, ["series", "city", "delta_csd"]]
    order = list(primary.sort_values(["delta_csd", "series"], kind="mergesort")["series"])
    seeds = list(SEED_ORDER)
    pivot = endowment.pivot(index="series", columns="seed", values="delta_csd")
    matrix = pivot.loc[order, seeds].to_numpy(dtype=float)
    return matrix, order, seeds


def _draw_heatmap_panel(fig: Figure, ax: Axes, endowment: pd.DataFrame) -> None:
    matrix, order, seeds = _heatmap_matrix(endowment)
    limit = float(np.max(np.abs(matrix)))
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("heatmap requires at least one finite nonzero effect")
    masked = np.ma.masked_where(matrix == 0.0, matrix)
    cmap = LinearSegmentedColormap.from_list(
        "external_effect",
        (NEGATIVE, "#FFFFFF", POSITIVE),
    )
    cmap = cmap.with_extremes(bad=TIE)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    sns.heatmap(
        masked,
        mask=np.ma.getmaskarray(masked),
        cmap=cmap,
        norm=norm,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    image = ax.collections[-1]
    image.set_gid("external-effect-heatmap")
    add_heatmap_sign_channel(
        ax,
        matrix,
        center=0.0,
        gid_prefix="external-effect-signs",
        marker_size=8.0,
    )

    cities = [series.rsplit("/", 1)[0] for series in order]
    if set(cities) != set(CITY_ORDER):
        raise ValueError("heatmap rows must belong to both frozen cities")
    strip_transform = blended_transform_factory(ax.transAxes, ax.transData)
    for row_index, (series, city) in enumerate(zip(order, cities, strict=True)):
        strip = Rectangle(
            (-0.075, row_index),
            0.035,
            1.0,
            transform=strip_transform,
            facecolor=CITY[city].color,
            edgecolor="none",
            clip_on=False,
            zorder=4,
        )
        strip.set_gid(f"city-strip:{series}")
        ax.add_patch(strip)

    ax.set_xticks(np.arange(len(seeds)) + 0.5, [str(seed) for seed in seeds])
    ax.set_yticks([])
    ax.set_xlabel("frozen seed")
    ax.set_title("(b) series effects", loc="left", pad=5)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.16, pad=0.05, aspect=18)
    colorbar.set_label(r"$\Delta$ CSD", fontsize=8.0, labelpad=1.5)
    colorbar.ax.tick_params(labelsize=8.0, width=0.5, length=2)
    colorbar.outline.set_linewidth(0.5)


def _axis_limits(values: np.ndarray, *, minimum_span: float = 0.01) -> tuple[float, float]:
    low = min(float(np.min(values)), 0.0)
    high = max(float(np.max(values)), 0.0)
    span = max(high - low, minimum_span)
    return low - 0.08 * span, high + 0.08 * span


def _draw_tradeoff_panel(ax: Axes, endowment: pd.DataFrame) -> None:
    primary = endowment.loc[endowment["seed"] == 42].copy()
    primary["district"] = primary["series"].str.rsplit("/", n=1).str[-1]
    for city in CITY_ORDER:
        rows = primary.loc[primary["city"] == city]
        if rows.empty:
            continue
        style = CITY[city]
        sns.scatterplot(
            data=rows,
            x="delta_csd",
            y="delta_exclusion",
            s=20,
            marker=style.marker,
            facecolor=style.color,
            edgecolor=INK,
            linewidth=0.45,
            alpha=0.9,
            label=city.rsplit("/", 1)[-1],
            zorder=4,
            ax=ax,
        )

    zero_x = ax.axvline(0.0, color=INK, lw=0.7, zorder=2)
    zero_x.set_gid("zero-csd")
    zero_y = ax.axhline(0.0, color=INK, lw=0.7, zorder=2)
    zero_y.set_gid("zero-exclusion")
    ax.set_xlim(*_axis_limits(primary["delta_csd"].to_numpy(dtype=float)))
    ax.set_ylim(*_axis_limits(primary["delta_exclusion"].to_numpy(dtype=float)))

    for district, offset in ANNOTATION_OFFSETS.items():
        rows = primary.loc[primary["district"] == district]
        if rows.empty:
            continue
        row = rows.iloc[0]
        ax.annotate(
            district,
            xy=(float(row["delta_csd"]), float(row["delta_exclusion"])),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            va="top" if offset[1] < 0 else "bottom",
            fontsize=8.0,
            color=INK,
            arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.45, "shrinkB": 2},
            zorder=5,
        )

    ax.set_xlabel(r"$\Delta$ CSD")
    ax.set_ylabel(r"$\Delta$ exclusion")
    ax.set_title("(c) seed-42 tradeoff", loc="left", pad=5)
    ax.grid(True, color=GRID, lw=0.35)
    ax.legend(
        loc="lower right",
        borderpad=0.1,
        handletextpad=0.35,
        labelspacing=0.25,
        fontsize=8.0,
    )


def build_transfer_dashboard(
    effects: pd.DataFrame,
    city_intervals: pd.DataFrame,
    diagnostics: Mapping[str, Any],
) -> tuple[Figure, Mapping[str, Axes]]:
    """Build the three evidence panels without changing any source table."""
    if diagnostics.get("decision") != "falsified":
        raise ValueError("dashboard requires the frozen falsified decision")
    apply_style(base=8.0)
    endowment = _prepare_endowment_effects(effects)
    figure = plt.figure(figsize=(TEXT, 3.35), constrained_layout=True)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.42, 0.76, 1.30))
    axes: dict[str, Axes] = {
        "intervals": figure.add_subplot(grid[0, 0]),
        "heatmap": figure.add_subplot(grid[0, 1]),
        "tradeoff": figure.add_subplot(grid[0, 2]),
    }
    _draw_interval_panel(axes["intervals"], city_intervals)
    _draw_heatmap_panel(figure, axes["heatmap"], endowment)
    _draw_tradeoff_panel(axes["tradeoff"], endowment)
    for ax in axes.values():
        apply_recessive_spines(ax)
    return figure, axes


def _assert_dashboard_layout(figure: Figure, axes: Mapping[str, Axes]) -> None:
    """Fail before export on clipping, overlap, missing margins, or tiny text."""
    if tuple(figure.get_size_inches()) != (TEXT, 3.35):
        raise ValueError("dashboard must retain the approved 6.75 by 3.35 inch size")
    if figure.get_layout_engine() is None:
        raise ValueError("dashboard must use constrained layout")

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox
    axis_boxes = {name: ax.get_window_extent(renderer) for name, ax in axes.items()}
    tolerance = 2.0
    for name, box in axis_boxes.items():
        if (
            box.x0 < figure_box.x0 - tolerance
            or box.y0 < figure_box.y0 - tolerance
            or box.x1 > figure_box.x1 + tolerance
            or box.y1 > figure_box.y1 + tolerance
        ):
            raise ValueError(f"{name} axes extend beyond the figure canvas")
    names = list(axis_boxes)
    for left, right in zip(names, names[1:]):
        if axis_boxes[left].x1 >= axis_boxes[right].x0:
            raise ValueError(f"dashboard axes overlap: {left} and {right}")

    layout_texts: list[matplotlib.text.Text] = []
    for ax in figure.axes:
        layout_texts.extend(
            [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        )
        layout_texts.extend(ax.texts)
        legend = ax.get_legend()
        if legend is not None:
            layout_texts.extend(legend.get_texts())
            layout_texts.append(legend.get_title())
        x_low, x_high = sorted(ax.get_xlim())
        y_low, y_high = sorted(ax.get_ylim())
        layout_texts.extend(
            label
            for value, label in zip(ax.get_xticks(), ax.get_xticklabels(), strict=True)
            if x_low <= value <= x_high
        )
        layout_texts.extend(
            label
            for value, label in zip(ax.get_yticks(), ax.get_yticklabels(), strict=True)
            if y_low <= value <= y_high
        )

    for text in layout_texts:
        if not text.get_visible() or not text.get_text():
            continue
        if text.get_fontsize() < 8.0:
            raise ValueError(f"dashboard text falls below 8 points: {text.get_text()!r}")
        box = text.get_window_extent(renderer)
        if (
            box.x0 < figure_box.x0 - tolerance
            or box.y0 < figure_box.y0 - tolerance
            or box.x1 > figure_box.x1 + tolerance
            or box.y1 > figure_box.y1 + tolerance
        ):
            raise ValueError(f"dashboard text is clipped: {text.get_text()!r}")

    interval_ax = axes["intervals"]
    interval_low, interval_high = interval_ax.get_xlim()
    for container in interval_ax.containers:
        if not isinstance(container, matplotlib.container.ErrorbarContainer):
            continue
        for collection in container.lines[2]:
            for segment in collection.get_segments():
                xs = np.asarray(segment)[:, 0]
                if xs.min() <= interval_low or xs.max() >= interval_high:
                    raise ValueError("an uncertainty interval touches or crosses the x boundary")

    tradeoff_ax = axes["tradeoff"]
    xlim = tradeoff_ax.get_xlim()
    ylim = tradeoff_ax.get_ylim()
    for collection in tradeoff_ax.collections:
        offsets = collection.get_offsets()
        if len(offsets) and (
            np.min(offsets[:, 0]) <= xlim[0]
            or np.max(offsets[:, 0]) >= xlim[1]
            or np.min(offsets[:, 1]) <= ylim[0]
            or np.max(offsets[:, 1]) >= ylim[1]
        ):
            raise ValueError("a tradeoff point touches or crosses an axis boundary")


def _render_figure_catalog(pdf_hash: str, png_hash: str) -> str:
    return "\n".join(
        [
            "# Figure Catalog",
            "",
            "## `fig_external` — transfer-and-falsification dashboard",
            "",
            "Purpose: show why a directionally favorable macro result still fails confirmation.",
            "",
            "Source tables:",
            "",
            "- `effects.csv`: policy-minus-MES per-series CSD and exclusion contrasts.",
            "- `city_intervals.csv`: authenticated city-by-seed point estimates and paired-bootstrap limits.",
            "- `diagnostics.json`: frozen primary seed and mechanical `falsified` classification.",
            "",
            "Panel estimands:",
            "",
            "- Panel (a): city mean endowment-minus-MES CSD difference with paired-bootstrap 95% interval.",
            "- Panel (b): per-series endowment-minus-MES CSD difference for each frozen seed; beige is an exact tie and the adjacent strip encodes city.",
            "- Panel (c): primary-seed per-series CSD difference versus exclusion difference; descriptive only.",
            "",
            "Registered outputs:",
            "",
            f"- `figures/fig_external.pdf`: SHA-256 `{pdf_hash}`.",
            f"- `figures/fig_external.png`: SHA-256 `{png_hash}`.",
            "- Paper copies under `iclr_paper/tex/` are byte-identical to these registered outputs.",
            "",
        ]
    )


def save_transfer_dashboard(
    analysis_dir: Path, paper_tex_dir: Path
) -> tuple[Path, Path]:
    """Capture, render, and transactionally register the dashboard."""
    paths = external_io.preflight_paths(
        Path(analysis_dir), Path(paper_tex_dir), root=ROOT
    )
    figure: Figure | None = None
    try:
        captured = external_contracts.capture_plot_inputs(
            paths.analysis_dir, trusted_hashes=TRUSTED_TASK2_OUTPUT_SHA256
        )
        external_io.validate_analysis_state(paths, captured)
        figure, axes = build_transfer_dashboard(
            captured.effects, captured.intervals, captured.diagnostics
        )
        _assert_dashboard_layout(figure, axes)

        def render(pdf_target: BinaryIO, png_target: BinaryIO) -> None:
            assert figure is not None
            with matplotlib.rc_context({"savefig.bbox": None}):
                figure.savefig(
                    pdf_target,
                    format="pdf",
                    dpi=400,
                    bbox_inches=None,
                    metadata={
                        "Creator": "iclr_fig_external.py",
                        "CreationDate": None,
                        "ModDate": None,
                    },
                )
                figure.savefig(
                    png_target,
                    format="png",
                    dpi=400,
                    bbox_inches=None,
                    metadata={"Software": "iclr_fig_external.py"},
                )

        return external_io.install_dashboard(
            paths,
            captured,
            render=render,
            render_catalog=_render_figure_catalog,
            recheck=lambda: external_contracts.recheck_captured_inputs(captured),
        )
    finally:
        if figure is not None:
            plt.close(figure)
        paths.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--paper-tex-dir", type=Path, default=DEFAULT_PAPER_TEX_DIR)
    args = parser.parse_args(argv)
    save_transfer_dashboard(args.analysis_dir, args.paper_tex_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
