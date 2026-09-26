"""Terminology regressions for the shipped ICLR figure PDFs.

A generator edit that is never re-run leaves a stale label inside the shipped
PDF, and no test in the suite reads the artifact itself. That is exactly how
``fig_hack.pdf`` kept the retired phrase "no welfare floor" after the source had
been changed to "no welfare penalty". These checks read the shipped figures.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import time
import tomllib
import zlib
from itertools import combinations
from pathlib import Path
from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text

FIGURE_DIR = Path(__file__).resolve().parents[2] / "iclr_paper" / "tex"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _visible_collection(axis, gid: str) -> PathCollection:
    matches = [
        collection
        for collection in axis.collections
        if isinstance(collection, PathCollection) and collection.get_gid() == gid
    ]
    assert len(matches) == 1, gid
    assert matches[0].get_visible(), gid
    return matches[0]


def _assert_collection_marker(
    collection: PathCollection, marker: str, *, path_index: int = 0
) -> None:
    expected = MarkerStyle(marker)
    expected_path = expected.get_path().transformed(expected.get_transform())
    actual_path = collection.get_paths()[path_index]
    np.testing.assert_allclose(actual_path.vertices, expected_path.vertices)
    assert actual_path.codes is None or np.array_equal(actual_path.codes, expected_path.codes)

# The soft welfare target is a hinge penalty, not a constraint. "Support floor"
# is the only legitimate floor in this paper: the structural inequality of
# Proposition 2.
FORBIDDEN = (
    "welfare floor",
    "welfare constraint",
    "hard welfare",
    "successful transfer",
    "robust external",
)

_SHOW = re.compile(rb"\((?P<body>(?:[^()\\]|\\.)*)\)")
_STREAM = re.compile(rb"stream\r?\n(.*?)endstream", re.S)
_TEXT_OPERATION = re.compile(
    rb"/[^\s]+\s+(?P<size>-?(?:\d+(?:\.\d*)?|\.\d+))\s+Tf"
    rb"|(?P<tj>\((?:[^()\\]|\\.)*\)|<[0-9A-Fa-f\s]+>)\s*Tj"
    rb"|(?P<TJ>\[(?:[^\[\]]|\((?:[^()\\]|\\.)*\))*\])\s*TJ",
    re.S,
)


def figure_text(path: Path) -> str:
    """Concatenate the glyphs a matplotlib PDF draws, in content-stream order.

    Fragments from different text objects are joined, so this is only sound for
    "this phrase must (not) appear" checks, never for layout assertions.
    """
    chunks: List[str] = []
    for match in _STREAM.finditer(path.read_bytes()):
        try:
            content = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        for glyph in _SHOW.finditer(content):
            body = glyph.group("body").replace(b"\\", b"")
            chunks.append(body.decode("latin-1").replace("\x00", ""))
    return "".join(chunks)


def pdf_text_run_font_sizes(path: Path) -> List[float]:
    """Read actual ``Tf`` sizes used by visible ``Tj``/``TJ`` PDF runs."""
    sizes: List[float] = []
    for match in _STREAM.finditer(path.read_bytes()):
        try:
            content = zlib.decompress(match.group(1))
        except zlib.error:
            continue
        current_size = None
        for operation in _TEXT_OPERATION.finditer(content):
            if operation.group("size") is not None:
                current_size = float(operation.group("size"))
            elif current_size is not None:
                sizes.append(current_size)
    return sizes


def figure_paths() -> List[Path]:
    return sorted(FIGURE_DIR.glob("fig_*.pdf"))


def test_figure_runtime_declares_its_seaborn_dependency() -> None:
    """A fresh ``uv sync`` must install every direct plotting import."""
    dependencies = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    names = {
        re.split(r"[<>=!~ ]", dependency, maxsplit=1)[0].lower()
        for dependency in dependencies
    }
    assert "seaborn" in names


def test_figure_pdfs_exist() -> None:
    """Guard the glob: an empty parametrization would pass silently."""
    assert len(figure_paths()) >= 7


@pytest.mark.parametrize("path", figure_paths(), ids=lambda p: p.name)
def test_shipped_figure_avoids_retired_welfare_wording(path: Path) -> None:
    text = figure_text(path).lower()
    for phrase in FORBIDDEN:
        assert phrase not in text, f"{path.name} still says '{phrase}'"


def test_reward_hack_figure_labels_the_unpenalized_policy() -> None:
    """The panel title must name the soft penalty, not a constraint."""
    text = figure_text(FIGURE_DIR / "fig_hack.pdf")
    assert "no welfare penalty" in text
    assert "support floor" in text


def test_schematic_names_three_control_surfaces_and_intervention() -> None:
    """The shipped schematic must distinguish class and kernel evidence."""
    text = figure_text(FIGURE_DIR / "fig_schematic.pdf").lower()
    for phrase in (
        "endowment",
        "project priority",
        "direct score",
        "equal shares",
        "actuation",
        "frozen-score intervention",
    ):
        assert phrase in text, f"fig_schematic.pdf is missing '{phrase}'"
    assert "not guaranteed" in text
    assert "high reachability" not in text


def test_schematic_states_the_actuation_corpus_and_budget_normalization() -> None:
    """Figure 1 must not disguise projected ballots or an unnormalized map."""
    module = importlib.import_module("iclr_fig_schematic")
    figure = module.build_schematic()
    try:
        visible_text = "\n".join(text.get_text() for text in figure.axes[0].texts)
        assert "actuation evidence" in visible_text
        assert visible_text.count("projected-support series") == 2
        assert "not evaluated on projected support" in visible_text
        assert "22/58" not in visible_text
        assert r"\sum_i\beta_i=b" in visible_text
        assert r"\beta_i=(b/|N|)(1+w^{\top}f)_+" not in visible_text
    finally:
        plt.close(figure)


def test_schematic_endowment_heading_clears_equation() -> None:
    """The endowment-map heading must not collide with the tall math line."""
    module = importlib.import_module("iclr_fig_schematic")
    figure = module.build_schematic()
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        texts = {
            text.get_text(): text
            for text in figure.axes[0].texts
        }
        heading = texts["endowment map"].get_window_extent(renderer)
        equation = texts[
            r"$\beta_i\propto(1+w^{\top}f_i)_+,\ \sum_i\beta_i=b$"
        ].get_window_extent(renderer)
        nine_point_gap = 9.0 * figure.dpi / 72.0
        assert heading.y0 - equation.y1 >= nine_point_gap
    finally:
        plt.close(figure)


def test_schematic_endowment_heading_clears_equation_with_fallback_font() -> None:
    """The supported DejaVu fallback must retain the same safety margin."""
    module = importlib.import_module("iclr_fig_schematic")
    with plt.rc_context(
        {"font.family": "serif", "font.serif": ["DejaVu Serif"]}
    ):
        figure = module.build_schematic()
        try:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            texts = {text.get_text(): text for text in figure.axes[0].texts}
            heading = texts["endowment map"].get_window_extent(renderer)
            equation = texts[
                r"$\beta_i\propto(1+w^{\top}f_i)_+,\ \sum_i\beta_i=b$"
            ].get_window_extent(renderer)
            nine_point_gap = 9.0 * figure.dpi / 72.0
            assert heading.y0 - equation.y1 >= nine_point_gap
        finally:
            plt.close(figure)


@pytest.mark.parametrize("filename", ("fig_schematic.pdf", "fig_hack.pdf"))
def test_frozen_score_caveat_preserves_the_exact_fixed_objects(filename: str) -> None:
    """Kernel comparisons freeze parameters, not endogenous score paths."""
    text = figure_text(FIGURE_DIR / filename).lower()
    for phrase in (
        "score weights and feature definitions fixed",
        "realized score trajectories may diverge through endogenous deficit histories",
    ):
        assert phrase in text, f"{filename} is missing '{phrase}'"
    for overstatement in ("holds scores fixed", "allocation kernel only"):
        assert overstatement not in text, f"{filename} says '{overstatement}'"


def test_schematic_preserves_the_fitted_class_noncausal_warning() -> None:
    text = figure_text(FIGURE_DIR / "fig_schematic.pdf").lower()
    assert "fitted-class differences are not a causal estimate" in text


def test_frontier_preserves_points_and_uses_shared_policy_identity() -> None:
    """Dropping a fitted/baseline point or remapping policy identity must fail."""
    module = importlib.import_module("gen_iclr_figures")
    assert hasattr(module, "build_frontier_figure"), (
        "frontier generator must expose its scientific artists for regression QA"
    )
    figure, axis = module.build_frontier_figure()
    try:
        lines = {line.get_gid(): line for line in axis.lines if line.get_gid()}
        outcome = lines["frontier-curve:learned-outcome"]
        endowment = lines["frontier-curve:learned-endowment"]
        assert outcome.get_visible()
        assert endowment.get_visible()
        np.testing.assert_allclose(
            np.column_stack((outcome.get_xdata(), outcome.get_ydata())),
            [
                [0.8646725, 0.11645125],
                [1.18593625, 0.11761],
                [1.25169, 0.13594725],
                [1.30014675, 0.171506],
                [1.3095105, 0.18221525],
                [1.31274325, 0.18477075],
            ],
        )
        np.testing.assert_allclose(
            np.column_stack((endowment.get_xdata(), endowment.get_ydata())),
            [
                [1.274522, 0.17431],
                [1.277125, 0.173595],
                [1.284042, 0.187563],
            ],
        )

        from iclr_style import POLICY

        expected_policy_lines = {
            "frontier-curve:learned-outcome": "Learned outcome",
            "frontier-curve:learned-endowment": "Learned endowment",
        }
        for gid, policy in expected_policy_lines.items():
            artist = lines[gid]
            assert mcolors.to_hex(artist.get_color()) == POLICY[policy].color.lower()
            assert artist.get_marker() == POLICY[policy].marker

        expected_policy_points = {
            "frontier-policy:greedy-count": ("Greedy (count)", (0.918508, 0.2173287406045553)),
            "frontier-policy:greedy-cost": ("Greedy (per cost)", (1.319325, 0.1990698377094426)),
            "frontier-policy:mes": ("Equal Shares", (1.29361, 0.19915822342830186)),
            "frontier-policy:res-1.0": ("RES", (1.286793, 0.1951352518767799)),
        }
        for gid, (policy, point) in expected_policy_points.items():
            collection = _visible_collection(axis, gid)
            np.testing.assert_allclose(collection.get_offsets(), [point])
            assert mcolors.to_hex(collection.get_facecolors()[0]) == POLICY[policy].color.lower()
            _assert_collection_marker(collection, POLICY[policy].marker)

        assert all(line.get_visible() for line in axis.lines)

        assert axis.get_title() == ""
        assert {text.get_text() for text in axis.texts} == {
            "unconstrained:\ndominated on all\n3 held-out metrics",
            "endowment class:\nshort fitted frontier",
        }
        visible = [axis.xaxis.label, axis.yaxis.label, *axis.texts]
        visible.extend(axis.get_xticklabels())
        visible.extend(axis.get_yticklabels())
        visible.extend(axis.get_legend().get_texts())
        assert min(text.get_fontsize() for text in visible if text.get_text()) >= 8.0
    finally:
        plt.close(figure)


def test_regularization_preserves_all_effects_at_readable_scale() -> None:
    """The harmony pass must not filter districts or shrink visible text."""
    module = importlib.import_module("gen_iclr_figures")
    assert hasattr(module, "build_regularization_figure"), (
        "regularization generator must expose its scientific artists for QA"
    )
    figure, axes = module.build_regularization_figure()
    try:
        gap_axis, effect_axis = axes
        np.testing.assert_allclose(
            [patch.get_height() for patch in gap_axis.patches],
            [0.053065, 0.031179, 0.02374, 0.005691, -0.010293, -0.016858],
        )
        collections = {
            label: _visible_collection(effect_axis, f"regularization-effect:{label}")
            for label in ("learned-endowment", "learned-outcome")
        }
        np.testing.assert_allclose(
            collections["learned-endowment"].get_offsets()[:, 0],
            [
                -0.064663, -0.085952, -0.055952, -0.023614, -0.011983,
                0.0, 0.003035, 0.003074, -0.027595, -0.018401,
                -0.034155, -0.050003, 0.0, 0.037283, 0.016621,
                -0.135704, -0.01412, 0.002692,
            ],
        )
        np.testing.assert_allclose(
            collections["learned-outcome"].get_offsets()[:, 0],
            [
                -0.136781, -0.083338, -0.052438, -0.048199, -0.043156,
                -0.040087, -0.039742, -0.036292, -0.032775, -0.022647,
                -0.01185, -0.006235, -0.004611, -0.00213, 0.003976,
                0.006976, 0.010392, 0.064514,
            ],
        )

        from iclr_style import POLICY

        for gid_label, policy_label in (
            ("learned-endowment", "Learned endowment"),
            ("learned-outcome", "Learned outcome"),
        ):
            artist = collections[gid_label]
            style = POLICY[policy_label]
            assert mcolors.to_hex(artist.get_facecolors()[0]) == style.color.lower()
            _assert_collection_marker(artist, style.marker)

        assert all(line.get_visible() for line in effect_axis.lines)

        assert gap_axis.get_xlabel() == "Soft welfare target (relative to Equal Shares)"
        visible = []
        for axis in axes:
            visible.extend([axis.xaxis.label, axis.yaxis.label, *axis.texts])
            visible.extend(axis.get_xticklabels())
            visible.extend(axis.get_yticklabels())
            if axis.get_legend() is not None:
                visible.extend(axis.get_legend().get_texts())
        assert min(text.get_fontsize() for text in visible if text.get_text()) >= 8.0
    finally:
        plt.close(figure)


def test_frozen_score_figure_preserves_four_kernel_estimands() -> None:
    """The shipped three-panel figure must retain all four frozen kernels."""
    shipped_text = figure_text(FIGURE_DIR / "fig_hack.pdf").lower()
    for phrase in (
        "frozen-score kernel intervention",
        "direct fill",
        "static floor",
        "payment only",
        "payment + completion",
        "n=18",
    ):
        assert phrase in shipped_text, f"fig_hack.pdf is missing '{phrase}'"

    module = importlib.import_module("iclr_fig_hack")
    assert hasattr(module, "build_frozen_score_figure")
    figure, axes = module.build_frozen_score_figure()
    try:
        expected = {
            "Direct fill": (0.12284201495051265, 0.586114, 0.1470062083207223),
            "Static floor": (0.11645627739212576, 0.719168, 0.093514885304777),
            "Payment only": (0.11402625759138887, 1.021566, 0.05302672416921917),
            "Payment + completion": (
                0.1282197018070305,
                1.168671,
                0.04271770601627977,
            ),
        }
        metrics = ("worst_csd", "welfare_millions", "exclusion")
        from iclr_fig_hack import KERNELS

        for panel_index, (axis, metric) in enumerate(zip(axes, metrics, strict=True)):
            collection = _visible_collection(axis, f"frozen-score:{metric}")
            expected_offsets = [
                (values[panel_index], position)
                for position, values in zip((3.0, 2.0, 1.0, 0.0), expected.values(), strict=True)
            ]
            np.testing.assert_allclose(collection.get_offsets(), expected_offsets)
            for path_index, (_, _, style) in enumerate(KERNELS):
                assert mcolors.to_hex(collection.get_facecolors()[path_index]) == style.color.lower()
                _assert_collection_marker(collection, style.marker, path_index=path_index)
            assert all(line.get_visible() for line in axis.lines)

        assert [axis.get_xlim() for axis in axes] == [
            (0.105, 0.138),
            (0.5, 1.27),
            (0.0, 0.165),
        ]
        assert [axis.get_xlabel() for axis in axes] == [
            "Mean district\nworst-cohort deficit",
            "Total approval-count\nwelfare (millions)",
            "Mean district\nexclusion",
        ]
        assert any(
            "score-prioritized diagnostic, not standard MES" in text.get_text()
            for text in figure.texts
        )

        visible = list(figure.texts)
        for axis in axes:
            visible.extend([axis.xaxis.label, axis.yaxis.label, *axis.texts])
            visible.extend(axis.get_xticklabels())
            visible.extend(axis.get_yticklabels())
        assert min(text.get_fontsize() for text in visible if text.get_text()) >= 8.0
    finally:
        plt.close(figure)


def test_shared_heatmap_sign_channel_marks_values_relative_to_reference() -> None:
    style_module = importlib.import_module("iclr_style")
    add_sign_channel = getattr(style_module, "add_heatmap_sign_channel", None)
    assert callable(add_sign_channel)
    figure, axis = plt.subplots()
    matrix = np.array([[0.5, 1.0, 1.5], [2.0, np.nan, 0.75]])
    add_sign_channel(
        axis,
        matrix,
        center=1.0,
        gid_prefix="test-signs",
    )

    positive = _visible_collection(axis, "test-signs:positive")
    negative = _visible_collection(axis, "test-signs:negative")
    assert {tuple(point) for point in positive.get_offsets()} == {
        (2.5, 0.5),
        (0.5, 1.5),
    }
    assert {tuple(point) for point in negative.get_offsets()} == {
        (0.5, 0.5),
        (2.5, 1.5),
    }
    assert positive.get_paths()[0].vertices.shape != negative.get_paths()[0].vertices.shape
    plt.close(figure)


def test_frozen_score_text_stays_inside_canvas_without_label_collisions() -> None:
    """The exact-width three-panel layout must not crop or overlap any text."""
    module = importlib.import_module("iclr_fig_hack")
    figure, axes = module.build_frozen_score_figure()
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        canvas = figure.bbox
        visible = [
            text
            for text in figure.findobj(match=Text)
            if text.get_visible() and text.get_text()
        ]
        outside = []
        for text in visible:
            box = text.get_window_extent(renderer=renderer)
            if (
                box.x0 < canvas.x0
                or box.y0 < canvas.y0
                or box.x1 > canvas.x1
                or box.y1 > canvas.y1
            ):
                outside.append(text.get_text())
        assert outside == []

        overlaps = [
            (left.get_text(), right.get_text())
            for left, right in combinations(visible, 2)
            if left.get_window_extent(renderer=renderer).overlaps(
                right.get_window_extent(renderer=renderer)
            )
        ]
        assert overlaps == []
    finally:
        plt.close(figure)


def test_visible_text_is_at_least_eight_points_at_manuscript_scale() -> None:
    """PDF rescaling in LaTeX must not push visible type below eight points."""
    cases = (
        ("iclr_fig_schematic", "build_schematic", "fig_schematic.pdf", 1.0),
        ("gen_iclr_figures", "build_frontier_figure", "fig_frontier.pdf", 0.72),
        (
            "gen_iclr_figures",
            "build_regularization_figure",
            "fig_regularization.pdf",
            0.98,
        ),
        ("iclr_fig_hack", "build_frozen_score_figure", "fig_hack.pdf", 0.98),
    )
    text_width_points = 5.5 * 72.0
    for module_name, builder_name, filename, width_fraction in cases:
        module = importlib.reload(importlib.import_module(module_name))
        built = getattr(module, builder_name)()
        figure = built[0] if isinstance(built, tuple) else built
        try:
            visible = [
                text
                for text in figure.findobj(match=Text)
                if text.get_visible() and text.get_text()
            ]
            assert visible
            source_minimum = min(text.get_fontsize() for text in visible)
        finally:
            plt.close(figure)

        info = subprocess.check_output(
            ["pdfinfo", str(FIGURE_DIR / filename)], text=True
        )
        match = re.search(r"Page size:\s+([0-9.]+) x", info)
        assert match, f"could not read page width for {filename}"
        pdf_width_points = float(match.group(1))
        rendered_minimum = source_minimum * (
            width_fraction * text_width_points / pdf_width_points
        )
        assert rendered_minimum >= 8.0, (
            f"{filename} renders {source_minimum:g}-point source text at "
            f"{rendered_minimum:.3f} points"
        )


@pytest.mark.parametrize(
    ("filename", "width_fraction"),
    (
        ("fig_schematic.pdf", 1.0),
        ("fig_frontier.pdf", 0.72),
        ("fig_regularization.pdf", 0.98),
        ("fig_hack.pdf", 0.98),
    ),
)
def test_pdf_glyph_runs_are_at_least_eight_points_at_manuscript_scale(
    filename: str, width_fraction: float
) -> None:
    """Decoded PDF glyph operators, including math scripts, must stay readable."""
    path = FIGURE_DIR / filename
    source_sizes = pdf_text_run_font_sizes(path)
    assert source_sizes, f"no decoded Tj/TJ glyph runs found in {filename}"

    info = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"Page size:\s+([0-9.]+) x", info)
    assert match, f"could not read page width for {filename}"
    pdf_width_points = float(match.group(1))
    inclusion_ratio = width_fraction * (5.5 * 72.0) / pdf_width_points
    rendered_sizes = [size * inclusion_ratio for size in source_sizes]
    assert min(rendered_sizes) >= 8.0, (
        f"{filename} emits a {min(source_sizes):g}-point PDF glyph run at "
        f"{min(rendered_sizes):.3f} manuscript points"
    )


def test_task5_pdf_generation_is_byte_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All four PDFs must be independent of the wall-clock generation time."""
    schematic = importlib.import_module("iclr_fig_schematic")
    figures = importlib.import_module("gen_iclr_figures")
    frozen_score = importlib.import_module("iclr_fig_hack")
    monkeypatch.setattr(schematic, "OUT", tmp_path / "fig_schematic")
    monkeypatch.setattr(figures, "OUT_DIR", tmp_path)
    monkeypatch.setattr(frozen_score, "OUT", tmp_path / "fig_hack")

    def generate() -> dict[str, bytes]:
        schematic.main()
        figures.main()
        frozen_score.main()
        return {
            filename: (tmp_path / filename).read_bytes()
            for filename in (
                "fig_schematic.pdf",
                "fig_frontier.pdf",
                "fig_regularization.pdf",
                "fig_hack.pdf",
            )
        }

    first = generate()
    time.sleep(1.1)
    second = generate()
    changed = [name for name in first if first[name] != second[name]]
    assert changed == [], f"clock-dependent PDF bytes: {changed}"
    for filename, content in second.items():
        assert b"/CreationDate" not in content, filename
        assert b"/ModDate" not in content, filename


def test_external_dashboard_uses_descriptive_noncausal_labels() -> None:
    """The central dashboard must label contrasts without causal language."""
    text = figure_text(FIGURE_DIR / "fig_external.pdf").lower()
    assert "learned endowment" in text
    assert "mes" in text
    assert "exclusion" in text
    for phrase in ("causes", "causal effect", "treatment effect"):
        assert phrase not in text
