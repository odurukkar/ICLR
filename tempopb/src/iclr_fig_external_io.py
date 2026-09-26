"""Transaction state and manifest-last installation for ``fig_external``.

This module recognizes pristine Task 2 and complete Task 3 states, stages one
dashboard transaction, installs it in a fixed order, and rolls back failures.
Descriptor-relative/no-follow filesystem mechanics live in
:mod:`iclr_fig_external_fs`; scientific authentication and parsing live in
:mod:`iclr_fig_external_contracts`; plotting lives in :mod:`iclr_fig_external`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import BinaryIO, Callable, Mapping

import analyze_external_support_set_results as analysis_bundle
from iclr_fig_external_contracts import CapturedPlotInputs, FIGURE_OUTPUTS
import iclr_fig_external_fs as safe_fs


FIGURE_STEM = "fig_external"
BASE_NAMES = frozenset(analysis_bundle.OUTPUT_FILENAMES)
FIGURE_NAMES = frozenset({"fig_external.pdf", "fig_external.png"})

@dataclass
class TransactionPaths:
    """Exact destinations plus retained directory descriptors for one save."""

    analysis: safe_fs.BoundDirectory
    paper: safe_fs.BoundDirectory
    analysis_parent: safe_fs.BoundDirectory
    figures: safe_fs.BoundDirectory | None = None

    @property
    def analysis_dir(self) -> Path:
        return self.analysis.path

    @property
    def paper_dir(self) -> Path:
        return self.paper.path

    @property
    def figure_dir(self) -> Path:
        return self.analysis_dir / "figures"

    @property
    def paper_pdf(self) -> Path:
        return self.paper_dir / f"{FIGURE_STEM}.pdf"

    @property
    def paper_png(self) -> Path:
        return self.paper_dir / f"{FIGURE_STEM}.png"

    @property
    def analysis_pdf(self) -> Path:
        return self.figure_dir / f"{FIGURE_STEM}.pdf"

    @property
    def analysis_png(self) -> Path:
        return self.figure_dir / f"{FIGURE_STEM}.png"

    @property
    def catalog(self) -> Path:
        return self.analysis_dir / "figure-catalog.md"

    @property
    def manifest(self) -> Path:
        return self.analysis_dir / "manifest.json"

    def close(self) -> None:
        if self.figures is not None:
            self.figures.close()
            self.figures = None
        self.paper.close()
        self.analysis.close()
        self.analysis_parent.close()


@dataclass(frozen=True)
class FileBackup:
    """Recoverable state of one descriptor-relative destination file."""

    parent: safe_fs.BoundDirectory
    name: str
    existed: bool
    content: bytes
    mode: int


@dataclass
class StagedDashboard:
    """All transaction-owned staged names and retained directories."""

    directory: safe_fs.BoundDirectory
    figures: safe_fs.BoundDirectory | None
    directory_name: str | None
    owns_directory: bool
    owns_figures: bool
    pdf: str | None = None
    png: str | None = None
    catalog: str | None = None
    manifest: str | None = None
    paper_pdf: str | None = None
    paper_png: str | None = None


def _reject_result_path(path: Path, root: Path, label: str) -> None:
    immutable_root = Path(os.path.abspath(os.fspath(root / "results")))
    if path == immutable_root or immutable_root in path.parents:
        raise ValueError(f"{label} must be outside every immutable result root")


def _close_bindings(
    *bindings: safe_fs.BoundDirectory | None,
) -> None:
    for binding in bindings:
        if binding is not None:
            binding.close()


def preflight_paths(
    analysis_dir: Path, paper_dir: Path, *, root: Path
) -> TransactionPaths:
    """Bind safe existing destination directories without changing them."""
    analysis = paper = analysis_parent = None
    try:
        analysis = safe_fs.bind_directory(Path(analysis_dir), "analysis_dir")
        paper = safe_fs.bind_directory(Path(paper_dir), "paper_tex_dir")
        analysis_parent = safe_fs.bind_directory(
            analysis.path.parent, "analysis_dir parent"
        )
        _reject_result_path(analysis.path, root, "analysis_dir")
        _reject_result_path(paper.path, root, "paper_tex_dir")
        paths = TransactionPaths(analysis, paper, analysis_parent)
        for name in FIGURE_NAMES:
            safe_fs.validate_regular_or_absent(
                paths.paper, name, "paper destination asset"
            )
        return paths
    except BaseException:
        _close_bindings(analysis_parent, paper, analysis)
        raise


def _validate_top_level(
    paths: TransactionPaths, state: str, transaction_names: frozenset[str]
) -> None:
    top_level = set(safe_fs.list_names(paths.analysis))
    expected = set(BASE_NAMES) if state == "task2" else set(BASE_NAMES) | {"figures"}
    expected.update(transaction_names)
    if top_level == expected:
        return
    figure_identity = safe_fs.stat_at(paths.analysis, "figures")
    if figure_identity is not None and stat.S_ISLNK(figure_identity.st_mode):
        raise ValueError("analysis figures directory must not be a symlink")
    raise ValueError("partial analysis state: expected exactly Task 2 or Task 3 entries")


def _validate_task2_figures(paths: TransactionPaths) -> None:
    identity = safe_fs.stat_at(paths.analysis, "figures")
    if identity is None:
        return
    if stat.S_ISLNK(identity.st_mode):
        raise ValueError("analysis figures directory must not be a symlink")
    raise ValueError("partial analysis state: Task 2 manifest cannot have figures")


def _bind_task3_figures(
    paths: TransactionPaths, transaction_names: frozenset[str]
) -> None:
    identity = safe_fs.stat_at(paths.analysis, "figures")
    if identity is None:
        raise ValueError("partial analysis state: Task 3 figures directory is missing")
    if stat.S_ISLNK(identity.st_mode):
        raise ValueError("analysis figures directory must not be a symlink")
    if not stat.S_ISDIR(identity.st_mode):
        raise ValueError("analysis figures path must be a directory")
    if paths.figures is None:
        paths.figures = safe_fs.bind_child_directory(
            paths.analysis, "figures", paths.figure_dir, "analysis figures directory"
        )
    safe_fs.revalidate_directory(paths.figures)
    expected = set(FIGURE_NAMES) | set(transaction_names)
    if set(safe_fs.list_names(paths.figures)) != expected:
        raise ValueError("analysis figures directory must contain exactly two assets")
    for name in FIGURE_NAMES:
        safe_fs.require_regular(paths.figures, name, "analysis figure")


def validate_analysis_state(
    paths: TransactionPaths,
    captured: CapturedPlotInputs,
    *,
    analysis_transaction_names: frozenset[str] = frozenset(),
    figure_transaction_names: frozenset[str] = frozenset(),
) -> None:
    """Accept exactly pristine Task 2 or fully registered Task 3 state."""
    safe_fs.revalidate_bindings(paths.analysis, paths.paper, paths.figures)
    _validate_top_level(paths, captured.state, analysis_transaction_names)
    for name in BASE_NAMES:
        safe_fs.require_regular(paths.analysis, name, "analysis bundle entry")
    if captured.state == "task2":
        _validate_task2_figures(paths)
    else:
        _bind_task3_figures(paths, figure_transaction_names)
    for name in FIGURE_NAMES:
        safe_fs.validate_regular_or_absent(
            paths.paper, name, "paper destination asset"
        )


def _snapshot(parent: safe_fs.BoundDirectory, name: str) -> FileBackup:
    identity = safe_fs.stat_at(parent, name)
    if identity is None:
        return FileBackup(parent, name, False, b"", safe_fs.PUBLICATION_FILE_MODE)
    safe_fs.require_regular(parent, name, "destination")
    return FileBackup(
        parent,
        name,
        True,
        safe_fs.read_regular(parent, name),
        stat.S_IMODE(identity.st_mode),
    )


def _restore(backup: FileBackup) -> None:
    if backup.existed:
        staged = safe_fs.write_regular_temp(
            backup.parent,
            backup.name,
            backup.content,
            revalidate=False,
            mode=backup.mode,
        )
        safe_fs.replace_retained_at(
            backup.parent, staged, backup.parent, backup.name
        )
        return
    identity = safe_fs.stat_at(backup.parent, backup.name)
    if identity is None:
        return
    if not stat.S_ISREG(identity.st_mode) or stat.S_ISLNK(identity.st_mode):
        raise RuntimeError(f"rollback expected a regular file: {backup.name}")
    safe_fs.unlink_regular_if_present(backup.parent, backup.name)


def _remove_first_run_figures(paths: TransactionPaths) -> None:
    if paths.figures is None:
        return
    for name in safe_fs.list_names(paths.figures):
        safe_fs.require_regular(paths.figures, name, "rollback figure")
        safe_fs.unlink_regular_if_present(paths.figures, name)
    safe_fs.remove_empty_directory(paths.analysis, "figures")
    paths.figures.close()
    paths.figures = None


def _rollback(
    paths: TransactionPaths,
    captured: CapturedPlotInputs,
    backups: Mapping[str, FileBackup],
    figure_mode: int | None,
) -> None:
    _restore(backups["manifest"])
    _restore(backups["catalog"])
    if captured.state == "task2":
        _remove_first_run_figures(paths)
    else:
        _restore(backups["analysis_png"])
        _restore(backups["analysis_pdf"])
        if paths.figures is not None and figure_mode is not None:
            safe_fs.set_directory_mode(paths.figures, figure_mode)
    _restore(backups["paper_png"])
    _restore(backups["paper_pdf"])


def _manifest_bytes(
    captured: CapturedPlotInputs,
    *,
    catalog_sha256: str,
    pdf_sha256: str,
    png_sha256: str,
) -> bytes:
    manifest = dict(captured.manifest)
    hashes = {
        relative: digest
        for relative, digest in captured.manifest["output_sha256"].items()
        if relative not in FIGURE_OUTPUTS
    }
    hashes["figure-catalog.md"] = catalog_sha256
    hashes["figures/fig_external.pdf"] = pdf_sha256
    hashes["figures/fig_external.png"] = png_sha256
    manifest["output_sha256"] = hashes
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _create_first_run_stage(paths: TransactionPaths) -> StagedDashboard:
    safe_fs.revalidate_bindings(
        paths.analysis_parent, paths.analysis, paths.paper, paths.figures
    )
    for _ in range(100):
        name = f".fig_external-transaction-{secrets.token_hex(8)}"
        try:
            stage = safe_fs.create_bound_directory(
                paths.analysis_parent,
                name,
                paths.analysis_parent.path / name,
                "transaction staging directory",
            )
        except FileExistsError:
            continue
        figures = safe_fs.create_bound_directory(
            stage, "figures", stage.path / "figures", "staged figures directory"
        )
        return StagedDashboard(stage, figures, name, True, True)
    raise FileExistsError("could not allocate a unique transaction directory")


def _create_stage(
    paths: TransactionPaths, captured: CapturedPlotInputs
) -> StagedDashboard:
    if captured.state == "task2":
        return _create_first_run_stage(paths)
    assert paths.figures is not None
    safe_fs.revalidate_bindings(paths.analysis, paths.paper, paths.figures)
    return StagedDashboard(paths.analysis, paths.figures, None, False, False)


def _cleanup_staged_dashboard(
    paths: TransactionPaths, staged: StagedDashboard
) -> None:
    safe_fs.unlink_regular_if_present(paths.paper, staged.paper_pdf)
    safe_fs.unlink_regular_if_present(paths.paper, staged.paper_png)
    if staged.figures is not None:
        safe_fs.unlink_regular_if_present(staged.figures, staged.pdf)
        safe_fs.unlink_regular_if_present(staged.figures, staged.png)
    if staged.figures is not None and staged.owns_figures:
        staged.figures.close()
        try:
            safe_fs.remove_empty_directory(staged.directory, "figures")
        except FileNotFoundError:
            pass
    safe_fs.unlink_regular_if_present(staged.directory, staged.catalog)
    safe_fs.unlink_regular_if_present(staged.directory, staged.manifest)
    if staged.owns_directory:
        staged.directory.close()
        assert staged.directory_name is not None
        try:
            safe_fs.remove_empty_directory(paths.analysis_parent, staged.directory_name)
        except FileNotFoundError:
            pass


def _capture_backups(
    paths: TransactionPaths, captured: CapturedPlotInputs
) -> tuple[dict[str, FileBackup], int | None]:
    backups = {
        "paper_pdf": _snapshot(paths.paper, f"{FIGURE_STEM}.pdf"),
        "paper_png": _snapshot(paths.paper, f"{FIGURE_STEM}.png"),
        "catalog": _snapshot(paths.analysis, "figure-catalog.md"),
        "manifest": _snapshot(paths.analysis, "manifest.json"),
    }
    if captured.state == "task2":
        return backups, None
    assert paths.figures is not None
    backups["analysis_pdf"] = _snapshot(paths.figures, f"{FIGURE_STEM}.pdf")
    backups["analysis_png"] = _snapshot(paths.figures, f"{FIGURE_STEM}.png")
    mode = stat.S_IMODE(os.fstat(paths.figures.descriptor).st_mode)
    return backups, mode


def _close_render_handles(*handles: BinaryIO | None) -> None:
    for handle in handles:
        if handle is not None and not handle.closed:
            handle.close()


def _render_staged_assets(
    staged: StagedDashboard, render: Callable[[BinaryIO, BinaryIO], None]
) -> tuple[bytes, bytes, str, str]:
    assert staged.figures is not None
    pdf_handle = png_handle = None
    opener = (
        safe_fs.open_staged_output
        if staged.owns_figures
        else safe_fs.open_regular_temp
    )
    try:
        pdf_handle, staged.pdf = opener(
            staged.figures, f"{FIGURE_STEM}.pdf"
        )
        png_handle, staged.png = opener(
            staged.figures, f"{FIGURE_STEM}.png"
        )
        render(pdf_handle, png_handle)
        safe_fs.finish_staged_output(staged.figures, staged.pdf, pdf_handle)
        safe_fs.finish_staged_output(staged.figures, staged.png, png_handle)
        pdf_bytes = safe_fs.read_regular(staged.figures, staged.pdf)
        png_bytes = safe_fs.read_regular(staged.figures, staged.png)
        return (
            pdf_bytes,
            png_bytes,
            hashlib.sha256(pdf_bytes).hexdigest(),
            hashlib.sha256(png_bytes).hexdigest(),
        )
    except BaseException:
        _close_render_handles(pdf_handle, png_handle)
        raise


def _stage_metadata(
    staged: StagedDashboard,
    captured: CapturedPlotInputs,
    render_catalog: Callable[[str, str], str],
    pdf_hash: str,
    png_hash: str,
) -> None:
    catalog_bytes = render_catalog(pdf_hash, png_hash).encode("utf-8")
    staged.catalog = safe_fs.write_regular_temp(
        staged.directory, "figure-catalog.md", catalog_bytes
    )
    staged.manifest = safe_fs.write_regular_temp(
        staged.directory,
        "manifest.json",
        _manifest_bytes(
            captured,
            catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
            pdf_sha256=pdf_hash,
            png_sha256=png_hash,
        ),
    )


def _stage_paper_assets(
    paths: TransactionPaths,
    staged: StagedDashboard,
    pdf_bytes: bytes,
    png_bytes: bytes,
) -> None:
    staged.paper_pdf = safe_fs.write_regular_temp(
        paths.paper, paths.paper_pdf.name, pdf_bytes
    )
    staged.paper_png = safe_fs.write_regular_temp(
        paths.paper, paths.paper_png.name, png_bytes
    )


def _stage_dashboard(
    paths: TransactionPaths,
    captured: CapturedPlotInputs,
    *,
    render: Callable[[BinaryIO, BinaryIO], None],
    render_catalog: Callable[[str, str], str],
) -> StagedDashboard:
    staged = _create_stage(paths, captured)
    try:
        pdf_bytes, png_bytes, pdf_hash, png_hash = _render_staged_assets(
            staged, render
        )
        _stage_metadata(staged, captured, render_catalog, pdf_hash, png_hash)
        _stage_paper_assets(paths, staged, pdf_bytes, png_bytes)
        return staged
    except BaseException:
        _cleanup_staged_dashboard(paths, staged)
        raise


def _active_bindings(paths: TransactionPaths) -> tuple[safe_fs.BoundDirectory, ...]:
    return tuple(
        binding
        for binding in (paths.analysis, paths.paper, paths.figures)
        if binding is not None
    )


def _commit_paper_assets(paths: TransactionPaths, staged: StagedDashboard) -> None:
    assert staged.paper_pdf is not None and staged.paper_png is not None
    bindings = _active_bindings(paths)
    safe_fs.replace_at(
        paths.paper,
        staged.paper_pdf,
        paths.paper,
        paths.paper_pdf.name,
        bindings=bindings,
    )
    staged.paper_pdf = None
    safe_fs.replace_at(
        paths.paper,
        staged.paper_png,
        paths.paper,
        paths.paper_png.name,
        bindings=bindings,
    )
    staged.paper_png = None


def _commit_first_run_figures(
    paths: TransactionPaths, staged: StagedDashboard
) -> None:
    assert staged.figures is not None
    safe_fs.revalidate_bindings(
        paths.analysis_parent,
        paths.analysis,
        paths.paper,
        staged.directory,
        staged.figures,
    )
    safe_fs.replace_retained_at(
        staged.directory, "figures", paths.analysis, "figures"
    )
    staged.figures.path = paths.figure_dir
    staged.figures.label = "analysis figures directory"
    paths.figures = staged.figures
    staged.figures = None
    staged.pdf = staged.png = None
    safe_fs.revalidate_bindings(paths.analysis, paths.paper, paths.figures)


def _commit_rerun_figures(
    paths: TransactionPaths, staged: StagedDashboard
) -> None:
    assert paths.figures is not None and staged.figures is not None
    assert staged.pdf is not None and staged.png is not None
    bindings = _active_bindings(paths)
    safe_fs.replace_at(
        staged.figures,
        staged.pdf,
        paths.figures,
        paths.analysis_pdf.name,
        bindings=bindings,
    )
    staged.pdf = None
    safe_fs.replace_at(
        staged.figures,
        staged.png,
        paths.figures,
        paths.analysis_png.name,
        bindings=bindings,
    )
    staged.png = None


def _commit_figures(
    paths: TransactionPaths,
    captured: CapturedPlotInputs,
    staged: StagedDashboard,
) -> None:
    if captured.state == "task2":
        _commit_first_run_figures(paths, staged)
    else:
        _commit_rerun_figures(paths, staged)


def _commit_catalog(paths: TransactionPaths, staged: StagedDashboard) -> None:
    assert staged.catalog is not None
    safe_fs.replace_at(
        staged.directory,
        staged.catalog,
        paths.analysis,
        paths.catalog.name,
        bindings=_active_bindings(paths),
    )
    staged.catalog = None


def _commit_manifest(paths: TransactionPaths, staged: StagedDashboard) -> None:
    assert paths.figures is not None and staged.manifest is not None
    safe_fs.revalidate_bindings(paths.analysis, paths.paper, paths.figures)
    safe_fs.set_directory_mode(paths.figures)
    safe_fs.revalidate_bindings(paths.analysis, paths.paper, paths.figures)
    safe_fs.replace_at(
        staged.directory,
        staged.manifest,
        paths.analysis,
        paths.manifest.name,
        bindings=_active_bindings(paths),
    )
    staged.manifest = None


def _install_staged_dashboard(
    paths: TransactionPaths,
    captured: CapturedPlotInputs,
    staged: StagedDashboard,
) -> None:
    _commit_paper_assets(paths, staged)
    _commit_figures(paths, captured, staged)
    _commit_catalog(paths, staged)
    _commit_manifest(paths, staged)


def _commit_dashboard(
    paths: TransactionPaths,
    captured: CapturedPlotInputs,
    staged: StagedDashboard,
    backups: Mapping[str, FileBackup],
    figure_mode: int | None,
    recheck: Callable[[], None],
) -> None:
    recheck()
    analysis_names = (
        frozenset(name for name in (staged.catalog, staged.manifest) if name)
        if not staged.owns_directory
        else frozenset()
    )
    figure_names = (
        frozenset(name for name in (staged.pdf, staged.png) if name)
        if not staged.owns_figures
        else frozenset()
    )
    validate_analysis_state(
        paths,
        captured,
        analysis_transaction_names=analysis_names,
        figure_transaction_names=figure_names,
    )
    try:
        _install_staged_dashboard(paths, captured, staged)
    except BaseException:
        _rollback(paths, captured, backups, figure_mode)
        raise


def install_dashboard(
    paths: TransactionPaths,
    captured: CapturedPlotInputs,
    *,
    render: Callable[[BinaryIO, BinaryIO], None],
    render_catalog: Callable[[str, str], str],
    recheck: Callable[[], None],
) -> tuple[Path, Path]:
    """Stage, verify, atomically install, and roll back one dashboard save."""
    validate_analysis_state(paths, captured)
    backups, figure_mode = _capture_backups(paths, captured)
    staged = _stage_dashboard(
        paths, captured, render=render, render_catalog=render_catalog
    )
    try:
        _commit_dashboard(paths, captured, staged, backups, figure_mode, recheck)
    finally:
        _cleanup_staged_dashboard(paths, staged)
    return paths.paper_pdf, paths.paper_png
