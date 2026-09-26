from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATISTICAL_GENERATORS = (
    "gen_iclr_figures.py",
    "iclr_fig_forest.py",
    "iclr_fig_dynamics.py",
    "iclr_fig_hack.py",
    "iclr_fig_external.py",
)
SEABORN_MARKS = {
    "barplot",
    "heatmap",
    "lineplot",
    "scatterplot",
}
DIRECT_AXES_MARKS_COVERED_BY_SEABORN = {
    "bar",
    "barh",
    "boxplot",
    "hist",
    "imshow",
    "matshow",
    "pcolor",
    "pcolormesh",
    "scatter",
    "violinplot",
}


def test_statistical_generators_use_seaborn_marks() -> None:
    for filename in STATISTICAL_GENERATORS:
        tree = ast.parse((ROOT / "src" / filename).read_text())
        imports_sns = any(
            isinstance(node, ast.Import)
            and any(alias.name == "seaborn" and alias.asname == "sns" for alias in node.names)
            for node in ast.walk(tree)
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sns"
        ]
        assert imports_sns, filename
        assert calls, filename
        assert all(call.func.attr in SEABORN_MARKS for call in calls), (
            filename,
            {call.func.attr for call in calls},
        )
        assert all(any(keyword.arg == "ax" for keyword in call.keywords) for call in calls), (
            filename,
            "every Seaborn mark must target an explicit axes",
        )

        direct_axes_marks = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in DIRECT_AXES_MARKS_COVERED_BY_SEABORN
        }
        assert not direct_axes_marks, (filename, direct_axes_marks)


def test_schematic_remains_explicit_geometry() -> None:
    source = (ROOT / "src" / "iclr_fig_schematic.py").read_text()
    assert "FancyArrowPatch" in source
    assert "FancyBboxPatch" in source
