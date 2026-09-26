"""Transactional output and filesystem-safety contracts for fig_external."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from PIL import Image
import pytest

import iclr_fig_external as external_figure
from iclr_fig_external import save_transfer_dashboard


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis-output" / "external-support-set"
TRUSTED_TASK2_HASHES = {
    "effects.csv": "1d384ae167d88fae6a3b90db8ab358caaae7ef583ebde05012885aae62a424cf",
    "city_intervals.csv": "4dbbaa952ec65e2d2a49f23c65731fea8d70768a24552ac9acfe3dc42a3114ff",
    "diagnostics.json": "c2c34f16d3bf05a59f7c3a15a9fc049d3c0a60f40803c9f56d8e9d9fe0a58dcb",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_analysis_bundle(tmp_path: Path) -> Path:
    analysis_dir = tmp_path / "analysis"
    shutil.copytree(ANALYSIS_DIR, analysis_dir)
    return analysis_dir


def _make_pristine_task2_bundle(tmp_path: Path) -> Path:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    for asset in (analysis_dir / "figures").iterdir():
        asset.unlink()
    (analysis_dir / "figures").rmdir()
    catalog_path = analysis_dir / "figure-catalog.md"
    catalog_path.write_text(
        "\n".join(
            [
                "# Figure Catalog",
                "",
                "- External dashboard panel A: city-by-seed frozen bootstrap intervals.",
                "- External dashboard panel B: 32-series by three-seed CSD-effect heatmap.",
                "- External dashboard panel C: primary-seed CSD-versus-exclusion descriptive tradeoff.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"].pop("figures/fig_external.pdf")
    manifest["output_sha256"].pop("figures/fig_external.png")
    manifest["output_sha256"]["figure-catalog.md"] = _sha256(catalog_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return analysis_dir


def _hashes(paths: list[Path]) -> dict[Path, str]:
    return {path: _sha256(path) for path in paths}


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _tree_entries(root: Path) -> tuple[str, ...]:
    return tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*")))


def _tree_modes(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_mode & 0o777
        for path in sorted(root.rglob("*"))
        if not path.is_symlink()
    }


@contextmanager
def _temporary_umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _publication_files(analysis_dir: Path, paper_dir: Path) -> tuple[Path, ...]:
    return (
        paper_dir / "fig_external.pdf",
        paper_dir / "fig_external.png",
        analysis_dir / "figures/fig_external.pdf",
        analysis_dir / "figures/fig_external.png",
        analysis_dir / "figure-catalog.md",
        analysis_dir / "manifest.json",
    )


def test_task3_transaction_modules_stay_within_review_budget() -> None:
    """Keep the security-sensitive transaction surface independently auditable."""
    fs_path = ROOT / "src/iclr_fig_external_fs.py"
    io_path = ROOT / "src/iclr_fig_external_io.py"
    test_paths = (
        ROOT / "tests/test_iclr_fig_external.py",
        ROOT / "tests/test_iclr_fig_external_contracts.py",
        ROOT / "tests/test_iclr_fig_external_io.py",
    )

    assert fs_path.is_file()
    assert len(io_path.read_text(encoding="utf-8").splitlines()) < 650
    tree = ast.parse(io_path.read_text(encoding="utf-8"))
    function_lengths = {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert max(function_lengths.values()) <= 50
    assert all(path.is_file() for path in test_paths)
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) < 650
        for path in test_paths
    )


def test_save_registers_byte_identical_paper_and_analysis_assets(tmp_path: Path) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    protected = {
        name: (analysis_dir / name).read_bytes()
        for name in ("effects.csv", "city_intervals.csv", "diagnostics.json")
    }

    paper_pdf, paper_png = save_transfer_dashboard(analysis_dir, paper_dir)
    analysis_pdf = analysis_dir / "figures/fig_external.pdf"
    analysis_png = analysis_dir / "figures/fig_external.png"

    assert (paper_pdf, paper_png) == (
        paper_dir / "fig_external.pdf",
        paper_dir / "fig_external.png",
    )
    assert paper_pdf.read_bytes() == analysis_pdf.read_bytes()
    assert paper_png.read_bytes() == analysis_png.read_bytes()
    assert {name: (analysis_dir / name).read_bytes() for name in protected} == protected

    with Image.open(paper_png) as image:
        assert image.size == (2700, 1340)
        assert image.info["dpi"] == pytest.approx((400.0, 400.0), abs=0.01)
    media_box = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        paper_pdf.read_bytes(),
    )
    assert media_box is not None
    assert tuple(float(value) for value in media_box.groups()) == pytest.approx(
        (486.0, 241.2)
    )

    pdf_hash = _sha256(paper_pdf)
    png_hash = _sha256(paper_png)
    manifest = json.loads((analysis_dir / "manifest.json").read_text())
    catalog = (analysis_dir / "figure-catalog.md").read_text()
    assert manifest["output_sha256"]["figures/fig_external.pdf"] == pdf_hash
    assert manifest["output_sha256"]["figures/fig_external.png"] == png_hash
    for digest in (pdf_hash, png_hash):
        assert digest in catalog
    for phrase in (
        "directionally favorable macro result still fails confirmation",
        "effects.csv",
        "city_intervals.csv",
        "diagnostics.json",
        "Panel (a)",
        "Panel (b)",
        "Panel (c)",
    ):
        assert phrase in catalog
    for relative, digest in manifest["output_sha256"].items():
        assert _sha256(analysis_dir / relative) == digest
    for path in _publication_files(analysis_dir, paper_dir):
        assert path.stat().st_mode & 0o777 == 0o644


def test_save_accepts_pristine_task2_bundle_as_first_run(tmp_path: Path) -> None:
    analysis_dir = _make_pristine_task2_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    protected = {
        name: _sha256(analysis_dir / name)
        for name in ("effects.csv", "city_intervals.csv", "diagnostics.json")
    }

    paper_pdf, paper_png = save_transfer_dashboard(analysis_dir, paper_dir)

    assert paper_pdf.read_bytes() == (analysis_dir / "figures/fig_external.pdf").read_bytes()
    assert paper_png.read_bytes() == (analysis_dir / "figures/fig_external.png").read_bytes()
    manifest = json.loads((analysis_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["output_sha256"]) == {
        "analysis-report.md",
        "city_intervals.csv",
        "decision_conditions.csv",
        "diagnostics.json",
        "effects.csv",
        "figure-catalog.md",
        "figures/fig_external.pdf",
        "figures/fig_external.png",
        "stats-appendix.md",
    }
    assert {name: _sha256(analysis_dir / name) for name in protected} == protected


def test_save_rerun_is_byte_idempotent(tmp_path: Path) -> None:
    analysis_dir = _make_pristine_task2_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    save_transfer_dashboard(analysis_dir, paper_dir)
    first_entries = _tree_entries(tmp_path)
    first_hashes = _tree_hashes(tmp_path)

    save_transfer_dashboard(analysis_dir, paper_dir)

    assert _tree_entries(tmp_path) == first_entries
    assert _tree_hashes(tmp_path) == first_hashes


def test_rerun_does_not_create_parent_transaction_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    original_renderer = external_figure._render_figure_catalog
    inspected = False

    def inspect_parent_before_registration(pdf_hash: str, png_hash: str) -> str:
        nonlocal inspected
        inspected = True
        assert not list(analysis_dir.parent.glob(".fig_external-transaction-*"))
        return original_renderer(pdf_hash, png_hash)

    monkeypatch.setattr(
        external_figure, "_render_figure_catalog", inspect_parent_before_registration
    )

    save_transfer_dashboard(analysis_dir, paper_dir)

    assert inspected


@pytest.mark.parametrize(
    "partial_state",
    ("missing-asset", "unregistered-figures", "symlinked-figures", "extra-asset"),
)
def test_save_rejects_partial_or_symlinked_analysis_state_before_write(
    tmp_path: Path, partial_state: str
) -> None:
    if partial_state in {"unregistered-figures", "symlinked-figures"}:
        analysis_dir = _make_pristine_task2_bundle(tmp_path)
    else:
        analysis_dir = _copy_analysis_bundle(tmp_path)
    if partial_state == "missing-asset":
        (analysis_dir / "figures/fig_external.png").unlink()
    elif partial_state == "unregistered-figures":
        (analysis_dir / "figures").mkdir()
        (analysis_dir / "figures/fig_external.pdf").write_bytes(b"partial")
        (analysis_dir / "figures/fig_external.png").write_bytes(b"partial")
    elif partial_state == "symlinked-figures":
        target = tmp_path / "figure-target"
        target.mkdir()
        (analysis_dir / "figures").symlink_to(target, target_is_directory=True)
    else:
        (analysis_dir / "figures/unexpected.txt").write_text("extra", encoding="utf-8")
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    before_entries = _tree_entries(tmp_path)
    before_hashes = _tree_hashes(tmp_path)

    with pytest.raises(ValueError, match="partial|symlink|exactly|missing"):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert _tree_entries(tmp_path) == before_entries
    assert _tree_hashes(tmp_path) == before_hashes
    assert not list(paper_dir.iterdir())


def test_first_run_rolls_back_to_pristine_state_on_registration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _make_pristine_task2_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    for path in analysis_dir.iterdir():
        if path.is_file():
            path.chmod(0o644)
    before_entries = _tree_entries(tmp_path)
    before_hashes = _tree_hashes(tmp_path)
    before_modes = _tree_modes(tmp_path)
    real_replace = os.replace
    injected = False

    def fail_once_before_catalog(
        source: object, destination: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal injected
        destination_name = os.fspath(destination)
        catalog_commit = (
            destination_name == os.fspath(analysis_dir / "figure-catalog.md")
            or (
                destination_name == "figure-catalog.md"
                and kwargs.get("dst_dir_fd") is not None
            )
        )
        if not injected and catalog_commit:
            injected = True
            raise OSError("injected registration failure")
        real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_once_before_catalog)

    with pytest.raises(OSError, match="injected registration failure"):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert injected
    assert _tree_entries(tmp_path) == before_entries
    assert _tree_hashes(tmp_path) == before_hashes
    assert _tree_modes(tmp_path) == before_modes
    assert not (analysis_dir / "figures").exists()
    assert not list(paper_dir.iterdir())


def test_save_rejects_destination_symlink_before_any_write(tmp_path: Path) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    target = tmp_path / "target.pdf"
    target.write_bytes(b"do not overwrite")
    (paper_dir / "fig_external.pdf").symlink_to(target)
    protected_paths = [
        target,
        analysis_dir / "effects.csv",
        analysis_dir / "city_intervals.csv",
        analysis_dir / "diagnostics.json",
    ]
    before = _hashes(protected_paths)

    with pytest.raises(ValueError, match="symlink"):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert _hashes(protected_paths) == before


def test_save_rejects_symlinked_parent_before_any_write(tmp_path: Path) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    real_paper_dir = tmp_path / "real-paper"
    real_paper_dir.mkdir()
    paper_link = tmp_path / "paper-link"
    paper_link.symlink_to(real_paper_dir, target_is_directory=True)
    protected_paths = [
        analysis_dir / "effects.csv",
        analysis_dir / "city_intervals.csv",
        analysis_dir / "diagnostics.json",
    ]
    before = _hashes(protected_paths)

    with pytest.raises(ValueError, match="symlink"):
        save_transfer_dashboard(analysis_dir, paper_link)

    assert _hashes(protected_paths) == before
    assert not list(real_paper_dir.iterdir())


def test_save_rejects_paper_parent_replacement_after_preflight_without_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    captured_paper_dir = tmp_path / "paper-captured"
    attacker_target = tmp_path / "paper-attacker-target"
    attacker_target.mkdir()
    original_renderer = external_figure._render_figure_catalog
    swapped = False

    def swap_paper_parent(pdf_hash: str, png_hash: str) -> str:
        nonlocal swapped
        paper_dir.rename(captured_paper_dir)
        paper_dir.symlink_to(attacker_target, target_is_directory=True)
        swapped = True
        return original_renderer(pdf_hash, png_hash)

    monkeypatch.setattr(external_figure, "_render_figure_catalog", swap_paper_parent)

    with pytest.raises(ValueError, match="paper_tex_dir.*changed|symlink"):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert swapped
    assert paper_dir.is_symlink()
    assert not list(attacker_target.iterdir())
    assert not list(captured_paper_dir.iterdir())


def test_save_rejects_analysis_parent_replacement_at_commit_without_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    captured_analysis = tmp_path / "analysis-captured-at-commit"
    attacker_target = tmp_path / "analysis-attacker-target"
    attacker_target.mkdir()
    before_paper = _tree_hashes(paper_dir)
    real_replace = os.replace
    swapped = False

    def swap_analysis_on_first_commit(
        source: object, destination: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal swapped
        destination_name = os.fspath(destination)
        paper_pdf_commit = (
            destination_name == os.fspath(paper_dir / "fig_external.pdf")
            or (
                destination_name == "fig_external.pdf"
                and kwargs.get("dst_dir_fd") is not None
            )
        )
        if not swapped and paper_pdf_commit:
            analysis_dir.rename(captured_analysis)
            analysis_dir.symlink_to(attacker_target, target_is_directory=True)
            swapped = True
        real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", swap_analysis_on_first_commit)

    with pytest.raises(ValueError, match="analysis_dir.*changed|symlink"):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert swapped
    assert analysis_dir.is_symlink()
    assert not list(attacker_target.iterdir())
    assert _tree_hashes(paper_dir) == before_paper
    assert {
        name: _sha256(captured_analysis / name) for name in TRUSTED_TASK2_HASHES
    } == TRUSTED_TASK2_HASHES


def test_save_rejects_figures_parent_replacement_at_commit_without_target_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    figures_dir = analysis_dir / "figures"
    captured_figures = analysis_dir / "figures-captured-at-commit"
    attacker_target = tmp_path / "figures-attacker-target"
    attacker_target.mkdir()
    before_analysis = _tree_hashes(analysis_dir)
    real_replace = os.replace
    swapped = False
    pdf_commits = 0

    def swap_figures_on_asset_commit(
        source: object, destination: object, *args: object, **kwargs: object
    ) -> None:
        nonlocal pdf_commits, swapped
        destination_name = os.fspath(destination)
        if Path(destination_name).name == "fig_external.pdf":
            pdf_commits += 1
        if not swapped and pdf_commits == 2:
            figures_dir.rename(captured_figures)
            figures_dir.symlink_to(attacker_target, target_is_directory=True)
            swapped = True
        real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", swap_figures_on_asset_commit)

    with pytest.raises(ValueError, match="figures.*changed|symlink"):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert swapped
    assert figures_dir.is_symlink()
    assert not list(attacker_target.iterdir())
    assert _tree_hashes(captured_figures) == {
        key.removeprefix("figures/"): value
        for key, value in before_analysis.items()
        if key.startswith("figures/")
    }
    assert not list(paper_dir.iterdir())


def test_first_run_modes_ignore_umask_077(tmp_path: Path) -> None:
    analysis_dir = _make_pristine_task2_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()

    with _temporary_umask(0o077):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert (analysis_dir / "figures").stat().st_mode & 0o777 == 0o755
    for path in _publication_files(analysis_dir, paper_dir):
        assert path.stat().st_mode & 0o777 == 0o644


def test_rerun_modes_ignore_umask_077(tmp_path: Path) -> None:
    analysis_dir = _make_pristine_task2_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    save_transfer_dashboard(analysis_dir, paper_dir)
    (analysis_dir / "figures").chmod(0o700)
    for path in _publication_files(analysis_dir, paper_dir):
        path.chmod(0o600)

    with _temporary_umask(0o077):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert (analysis_dir / "figures").stat().st_mode & 0o777 == 0o755
    for path in _publication_files(analysis_dir, paper_dir):
        assert path.stat().st_mode & 0o777 == 0o644


def test_save_rejects_results_destination_without_mutating_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    fake_root = tmp_path / "repo"
    result_paper_dir = fake_root / "results/paper"
    result_paper_dir.mkdir(parents=True)
    marker = result_paper_dir / "immutable.txt"
    marker.write_text("immutable", encoding="utf-8")
    before = _tree_hashes(fake_root / "results")
    monkeypatch.setattr(external_figure, "ROOT", fake_root)

    with pytest.raises(ValueError, match="immutable result root"):
        save_transfer_dashboard(analysis_dir, result_paper_dir)

    assert _tree_hashes(fake_root / "results") == before


def test_save_rechecks_trusted_inputs_before_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    paper_paths = [paper_dir / "fig_external.pdf", paper_dir / "fig_external.png"]
    sources = (
        ANALYSIS_DIR / "figures/fig_external.pdf",
        ANALYSIS_DIR / "figures/fig_external.png",
    )
    for source, destination in zip(sources, paper_paths, strict=True):
        shutil.copyfile(source, destination)
    before = _hashes(paper_paths)
    original_renderer = external_figure._render_figure_catalog

    def tamper_before_registration(pdf_hash: str, png_hash: str) -> str:
        effects_path = analysis_dir / "effects.csv"
        effects_path.write_bytes(effects_path.read_bytes() + b"\n")
        return original_renderer(pdf_hash, png_hash)

    monkeypatch.setattr(
        external_figure, "_render_figure_catalog", tamper_before_registration
    )

    with pytest.raises(
        ValueError, match="trusted Task 2 SHA-256 mismatch for effects.csv"
    ):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert _hashes(paper_paths) == before


def test_save_rejects_analysis_directory_replacement_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    moved_analysis = tmp_path / "analysis-captured"
    original_renderer = external_figure._render_figure_catalog
    replaced = False

    def replace_directory_before_recheck(pdf_hash: str, png_hash: str) -> str:
        nonlocal replaced
        analysis_dir.rename(moved_analysis)
        analysis_dir.symlink_to(moved_analysis, target_is_directory=True)
        replaced = True
        return original_renderer(pdf_hash, png_hash)

    monkeypatch.setattr(
        external_figure, "_render_figure_catalog", replace_directory_before_recheck
    )

    with pytest.raises(ValueError, match="symlinked parent|changed after capture"):
        save_transfer_dashboard(analysis_dir, paper_dir)

    assert replaced
    assert analysis_dir.is_symlink()
    assert not list(paper_dir.iterdir())
    assert {
        name: _sha256(moved_analysis / name) for name in TRUSTED_TASK2_HASHES
    } == TRUSTED_TASK2_HASHES
