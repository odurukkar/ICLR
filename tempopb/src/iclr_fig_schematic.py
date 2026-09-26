"""Figure 1: three fairness control surfaces and their empirical actuation.

The lanes separate the learned object from the allocation kernel and from the
observed frequency with which the learned policy changes an external series.
The bottom callout also separates fitted-class comparisons from the narrower
frozen-score allocation-kernel intervention.

Writes ``iclr_paper/tex/fig_schematic.{pdf,png}``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from iclr_style import (
    FILL_DIRECT,
    FILL_MECH,
    FILL_NEUTRAL,
    INK,
    POLICY,
    apply_style,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT = Path(__file__).resolve().parent.parent.parent / "iclr_paper" / "tex" / "fig_schematic"
MANUSCRIPT_TEXT_WIDTH = 5.5
PDF_METADATA = {"CreationDate": None, "ModDate": None}

ENDOWMENT = POLICY["Learned endowment"].color
PRIORITY = POLICY["RES"].color
DIRECT = POLICY["Learned outcome"].color
MUTED = "#6b7280"

apply_style(base=8.1)


def _box_patch(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    edgecolor: str,
    *,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=0.9,
            linestyle=linestyle,
            edgecolor=edgecolor,
            facecolor=facecolor,
            zorder=2,
        )
    )


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    *,
    weight: str = "normal",
    linestyle: str = "-",
) -> None:
    _box_patch(ax, x, y, w, h, facecolor, edgecolor, linestyle=linestyle)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.1,
        color=INK,
        zorder=3,
        fontweight=weight,
        linespacing=1.18,
    )


def _equation_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    heading: str,
    equations: tuple[str, ...],
    facecolor: str,
    edgecolor: str,
) -> None:
    """Draw prose at body size and equations large enough for 8-point scripts."""
    _box_patch(ax, x, y, w, h, facecolor, edgecolor)
    if len(equations) == 1:
        heading_position = 0.84
        equation_positions = (0.25,)
    elif len(equations) == 2:
        heading_position = 0.80
        equation_positions = (0.50, 0.19)
    else:
        raise ValueError("equation box supports one or two equation lines")
    ax.text(
        x + w / 2,
        y + heading_position * h,
        heading,
        ha="center",
        va="center",
        fontsize=8.1,
        color=INK,
        zorder=3,
    )
    for equation, position in zip(equations, equation_positions):
        ax.text(
            x + w / 2,
            y + position * h,
            equation,
            ha="center",
            va="center",
            fontsize=11.5,
            color=INK,
            zorder=3,
        )


def _arrow(ax, x1: float, y: float, x2: float, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y),
            (x2, y),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            zorder=1,
            shrinkA=1,
            shrinkB=1,
        )
    )


def build_schematic() -> Figure:
    """Build the three-lane schematic without writing an artifact."""
    fig, ax = plt.subplots(figsize=(MANUSCRIPT_TEXT_WIDTH, 3.7))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    columns = (
        (0.03, 0.30, "learned object"),
        (0.375, 0.245, "allocation kernel"),
        (0.685, 0.285, "actuation evidence"),
    )
    for x, width, heading in columns:
        ax.text(
            x + width / 2,
            0.945,
            heading,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=INK,
        )

    lanes = (
        (
            0.765,
            ENDOWMENT,
            FILL_MECH,
            (
                "endowment map",
                r"$\beta_i\propto(1+w^{\top}f_i)_+,\ \sum_i\beta_i=b$",
            ),
            "Equal Shares\npayments",
            "actuation 12/32\nprojected-support series",
        ),
        (
            0.585,
            PRIORITY,
            FILL_NEUTRAL,
            ("project priority\nlearned purchase order",),
            "Equal Shares\npayments",
            "actuation 3/32\nprojected-support series",
        ),
        (
            0.405,
            DIRECT,
            FILL_DIRECT,
            ("direct score", r"$s(p)=v^{\top}\varphi(p,\mathcal{H})$"),
            "greedy fill\nto budget",
            "actuation not guaranteed\n"
            "not evaluated on projected support",
        ),
    )
    height = 0.12
    for y, color, fill, learned, kernel, actuation in lanes:
        if len(learned) == 2:
            learned_y = y - 0.02 if learned[0] == "endowment map" else y
            learned_height = (
                height + 0.04 if learned[0] == "endowment map" else height
            )
            _equation_box(
                ax,
                columns[0][0],
                learned_y,
                columns[0][1],
                learned_height,
                learned[0],
                (learned[1],),
                fill,
                color,
            )
        else:
            _box(
                ax,
                columns[0][0],
                y,
                columns[0][1],
                height,
                learned[0],
                fill,
                color,
            )
        _box(ax, columns[1][0], y, columns[1][1], height, kernel, fill, color)
        _box(ax, columns[2][0], y, columns[2][1], height, actuation, fill, color)
        midpoint = y + height / 2
        _arrow(ax, columns[0][0] + columns[0][1] + 0.005, midpoint, columns[1][0] - 0.005, color)
        _arrow(ax, columns[1][0] + columns[1][1] + 0.005, midpoint, columns[2][0] - 0.005, color)

    _equation_box(
        ax,
        0.03,
        0.235,
        0.94,
        0.15,
        "exact containment identities",
        (
            r"$w=0\Rightarrow \mathrm{MES};\quad "
            r"w=\lambda e_1\Rightarrow \mathrm{RES}(\lambda)$",
            r"$v=e_1\Rightarrow \mathrm{greedy\!-\!count};\quad "
            r"v=e_2\Rightarrow \mathrm{greedy\!-\!cost}$",
        ),
        "white",
        MUTED,
    )
    _box(
        ax,
        0.03,
        0.03,
        0.94,
        0.17,
        "frozen-score intervention\n"
        "score weights and feature definitions fixed\n"
        "realized score trajectories may diverge through endogenous deficit histories\n"
        "fitted-class differences are not a causal estimate",
        FILL_NEUTRAL,
        MUTED,
        weight="normal",
        linestyle="--",
    )
    return fig


def main() -> None:
    figure = build_schematic()
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
