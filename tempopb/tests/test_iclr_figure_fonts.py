"""Submission-format regressions for generated ICLR figures."""

from __future__ import annotations

import importlib

import matplotlib
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "gen_iclr_figures",
        "iclr_fig_schematic",
        "iclr_fig_hack",
        "iclr_fig_dynamics",
        "iclr_fig_external",
    ),
)
def test_figure_generators_embed_truetype_fonts(module_name: str) -> None:
    """Generated PDFs must not fall back to Type 3 fonts."""
    original_pdf = matplotlib.rcParams["pdf.fonttype"]
    original_ps = matplotlib.rcParams["ps.fonttype"]
    try:
        module = importlib.import_module(module_name)
        matplotlib.rcParams["pdf.fonttype"] = 3
        matplotlib.rcParams["ps.fonttype"] = 3
        importlib.reload(module)
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42
    finally:
        matplotlib.rcParams["pdf.fonttype"] = original_pdf
        matplotlib.rcParams["ps.fonttype"] = original_ps


def test_external_dashboard_visible_text_is_at_least_eight_points() -> None:
    """The fixed-size dashboard must not solve layout by shrinking text."""
    module = importlib.import_module("iclr_fig_external")
    effects, intervals, diagnostics = module.load_plot_tables(module.DEFAULT_ANALYSIS_DIR)
    figure, _ = module.build_transfer_dashboard(effects, intervals, diagnostics)
    figure.canvas.draw()

    visible_text = []
    for ax in figure.axes:
        visible_text.extend(
            [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        )
        visible_text.extend(ax.texts)
        visible_text.extend(ax.get_xticklabels())
        visible_text.extend(ax.get_yticklabels())
        legend = ax.get_legend()
        if legend is not None:
            visible_text.extend(legend.get_texts())
            visible_text.append(legend.get_title())

    rendered = [text for text in visible_text if text.get_visible() and text.get_text()]
    assert rendered
    assert min(text.get_fontsize() for text in rendered) >= 8.0
    plt.close(figure)
