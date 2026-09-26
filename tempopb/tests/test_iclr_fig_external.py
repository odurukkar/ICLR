"""Public façade, layout, and artist contracts for fig_external."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import to_hex
from matplotlib.container import ErrorbarContainer
import numpy as np
import pytest

from iclr_fig_external import (
    _assert_dashboard_layout,
    _heatmap_matrix,
    build_transfer_dashboard,
    load_plot_tables,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis-output" / "external-support-set"
GLOBAL_PRIMARY_ORDER = (
    "Poland/Krakow/Czyżyny",
    "Poland/Katowice/Podlesie",
    "Poland/Katowice/Murcki",
    "Poland/Krakow/Nowa Huta",
    "Poland/Katowice/Osiedle Witosa",
    "Poland/Krakow/Mistrzejowice",
    "Poland/Krakow/Bieńczyce",
    "Poland/Krakow/Bieżanów-Prokocim",
    "Poland/Katowice/Bogucice",
    "Poland/Katowice/Dąb",
    "Poland/Katowice/Dąbrówka Mała",
    "Poland/Katowice/Giszowiec",
    "Poland/Katowice/Kostuchna",
    "Poland/Katowice/Koszutka",
    "Poland/Katowice/Osiedle Tysiąclecia",
    "Poland/Katowice/Zarzecze",
    "Poland/Katowice/Zawodzie",
    "Poland/Katowice/Załęże",
    "Poland/Katowice/Śródmieście",
    "Poland/Krakow/Dębniki",
    "Poland/Krakow/Grzegórzki",
    "Poland/Krakow/Podgórze Duchackie",
    "Poland/Krakow/Prądnik Biały",
    "Poland/Krakow/Prądnik Czerwony",
    "Poland/Krakow/Stare Miasto",
    "Poland/Krakow/Swoszowice",
    "Poland/Krakow/Wzgórza Krzesławickie",
    "Poland/Krakow/Łagiewniki-Borek Fałęcki",
    "Poland/Krakow/Krowodrza",
    "Poland/Krakow/Zwierzyniec",
    "Poland/Krakow/Podgórze",
    "Poland/Krakow/Bronowice",
)


def _tradeoff_annotations(ax: plt.Axes) -> set[str]:
    expected = {"Czyżyny", "Podlesie", "Bronowice"}
    return {text.get_text() for text in ax.texts} & expected


def test_dashboard_contains_the_three_evidence_panels() -> None:
    effects, intervals, diagnostics = load_plot_tables(ANALYSIS_DIR)
    figure, axes = build_transfer_dashboard(effects, intervals, diagnostics)

    assert set(axes) == {"intervals", "heatmap", "tradeoff"}
    assert axes["intervals"].get_xlabel() == (
        r"$\Delta$ CSD (learned endowment $-$ MES)"
    )
    assert axes["heatmap"].get_xlabel() == "frozen seed"
    assert axes["tradeoff"].get_ylabel() == r"$\Delta$ exclusion"
    heatmaps = [
        collection
        for collection in axes["heatmap"].collections
        if isinstance(collection, QuadMesh)
        and collection.get_gid() == "external-effect-heatmap"
    ]
    assert len(heatmaps) == 1
    assert heatmaps[0].get_visible()
    assert not axes["heatmap"].images
    assert len(axes["tradeoff"].collections) >= 2
    assert tuple(figure.get_size_inches()) == pytest.approx((6.75, 3.35))
    assert figure.get_layout_engine() is not None
    for ax in axes.values():
        assert all(
            ax.spines[side].get_visible()
            for side in ("left", "bottom", "top", "right")
        )
        assert {
            side: to_hex(ax.spines[side].get_edgecolor())
            for side in ("left", "bottom", "top", "right")
        } == {
            "left": "#777777",
            "bottom": "#777777",
            "top": "#aaaaaa",
            "right": "#aaaaaa",
        }
        assert {
            side: ax.spines[side].get_linewidth()
            for side in ("left", "bottom", "top", "right")
        } == {"left": 0.6, "bottom": 0.6, "top": 0.5, "right": 0.5}
    plt.close(figure)


def test_interval_panel_keeps_all_six_estimates_and_uncertainty_intervals() -> None:
    effects, intervals, diagnostics = load_plot_tables(ANALYSIS_DIR)
    figure, axes = build_transfer_dashboard(effects, intervals, diagnostics)
    ax = axes["intervals"]
    errorbars = [
        container
        for container in ax.containers
        if isinstance(container, ErrorbarContainer)
    ]

    assert len(errorbars) == 6
    assert sum(len(container.lines[0].get_xdata()) for container in errorbars) == 6
    assert sum(
        len(segment)
        for container in errorbars
        for collection in container.lines[2]
        for segment in collection.get_segments()
    ) == 12
    plotted = sorted(
        float(container.lines[0].get_xdata()[0]) for container in errorbars
    )
    assert plotted == pytest.approx(sorted(intervals["difference"]))
    assert ax.get_xlim()[0] < float(intervals["ci_low"].min())
    assert ax.get_xlim()[1] > 0.0166219908152236

    references = {
        line.get_gid(): float(np.asarray(line.get_xdata())[0])
        for line in ax.lines
        if line.get_gid() in {"zero-reference", "macro-threshold"}
    }
    assert references == {"zero-reference": 0.0, "macro-threshold": -0.005}
    assert not any("*" in text.get_text() for text in ax.texts)
    plt.close(figure)


def test_heatmap_keeps_32_by_3_endowment_values_and_masks_only_exact_ties() -> None:
    effects, intervals, diagnostics = load_plot_tables(ANALYSIS_DIR)
    figure, axes = build_transfer_dashboard(effects, intervals, diagnostics)
    ax = axes["heatmap"]
    heatmaps = [
        collection
        for collection in ax.collections
        if isinstance(collection, QuadMesh)
        and collection.get_gid() == "external-effect-heatmap"
    ]
    assert len(heatmaps) == 1
    heatmap = heatmaps[0]
    plotted = heatmap.get_array()
    raw = np.ma.getdata(plotted)
    mask = np.ma.getmaskarray(plotted)

    assert raw.shape == (32, 3)
    assert np.array_equal(mask, raw == 0.0)
    assert np.any(raw == 0.0)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "1",
        "2",
        "42",
    ]
    assert ax.get_xticks() == pytest.approx([0.5, 1.5, 2.5])
    coordinates = heatmap.get_coordinates()
    assert coordinates.shape == (33, 4, 2)
    assert coordinates[0, :, 0] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert coordinates[:, 0, 1] == pytest.approx(np.arange(33))

    endowment = effects.loc[effects["arm"] == "endowment"]
    expected_order = list(GLOBAL_PRIMARY_ORDER)
    expected = (
        endowment.pivot(index="series", columns="seed", values="delta_csd")
        .loc[expected_order, [1, 2, 42]]
        .to_numpy()
    )
    assert raw == pytest.approx(expected)
    matrix, actual_order, _ = _heatmap_matrix(endowment)
    assert tuple(actual_order) == GLOBAL_PRIMARY_ORDER
    assert np.all(np.diff(matrix[:, 2]) >= 0.0)
    strips = [
        patch
        for patch in ax.patches
        if (patch.get_gid() or "").startswith("city-strip:")
    ]
    assert [
        patch.get_gid().removeprefix("city-strip:") for patch in strips
    ] == expected_order
    assert len({to_hex(patch.get_facecolor()) for patch in strips}) == 2
    assert [patch.get_y() + patch.get_height() / 2 for patch in strips] == pytest.approx(
        np.arange(32) + 0.5
    )

    sign_collections = {
        collection.get_gid(): collection
        for collection in ax.collections
        if isinstance(collection, PathCollection)
        and (collection.get_gid() or "").startswith("external-effect-signs:")
    }
    assert set(sign_collections) == {
        "external-effect-signs:positive",
        "external-effect-signs:negative",
    }
    expected_positive = {
        (column + 0.5, row + 0.5)
        for row, column in np.argwhere(expected > 0.0)
    }
    expected_negative = {
        (column + 0.5, row + 0.5)
        for row, column in np.argwhere(expected < 0.0)
    }
    assert {tuple(point) for point in sign_collections["external-effect-signs:positive"].get_offsets()} == expected_positive
    assert {tuple(point) for point in sign_collections["external-effect-signs:negative"].get_offsets()} == expected_negative
    plt.close(figure)


def test_dashboard_runtime_guard_enforces_eight_point_visible_text_floor() -> None:
    effects, intervals, diagnostics = load_plot_tables(ANALYSIS_DIR)
    figure, axes = build_transfer_dashboard(effects, intervals, diagnostics)
    _assert_dashboard_layout(figure, axes)
    label = axes["intervals"]._left_title
    label.set_fontsize(7.99)

    with pytest.raises(ValueError, match="below 8 points"):
        _assert_dashboard_layout(figure, axes)
    plt.close(figure)


def test_tradeoff_panel_keeps_all_primary_seed_points_and_named_leverage_rows() -> None:
    effects, intervals, diagnostics = load_plot_tables(ANALYSIS_DIR)
    figure, axes = build_transfer_dashboard(effects, intervals, diagnostics)
    ax = axes["tradeoff"]

    assert sum(len(collection.get_offsets()) for collection in ax.collections) == 32
    assert _tradeoff_annotations(ax) == {"Czyżyny", "Podlesie", "Bronowice"}
    assert all(collection.get_edgecolors().size for collection in ax.collections)
    assert not any(line.get_label() == "regression" for line in ax.lines)
    plt.close(figure)


@pytest.mark.parametrize("district", ["Czyżyny", "Podlesie", "Bronowice"])
def test_tradeoff_annotates_a_named_row_only_when_that_row_exists(
    district: str,
) -> None:
    effects, intervals, diagnostics = load_plot_tables(ANALYSIS_DIR)
    without_row = effects.loc[~effects["series"].str.endswith(f"/{district}")]
    figure, axes = build_transfer_dashboard(without_row, intervals, diagnostics)

    assert district not in _tradeoff_annotations(axes["tradeoff"])
    plt.close(figure)
