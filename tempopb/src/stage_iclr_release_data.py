"""Stage protocol-locked ICLR corpus files from the verified source corpus."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK_PATHS: tuple[Path, ...] = (
    Path("results/iclr_multicity/protocol_lock.json"),
    Path("results/iclr_external_validation/protocol_lock.json"),
    Path("results/iclr_external_support_set/protocol_lock.json"),
    Path("results/iclr_external_support_set_amendment/protocol_lock.json"),
)
SUPPORTED_LOCK_SCHEMAS = frozenset({1, 2, 3})


class StageDataError(RuntimeError):
    """The locked corpus cannot be staged without changing declared bytes."""


@dataclass(frozen=True)
class StageReport:
    """Counts from one release-data staging pass."""

    lock_count: int
    reference_count: int
    unique_source_count: int
    target_count: int
    copied_count: int
    retained_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_relative(label: str) -> Path | None:
    parts = label.split("/")
    if "corpus" not in parts:
        return None
    if "\\" in label:
        raise StageDataError(f"unsafe corpus label: {label!r}")
    if parts.count("corpus") != 1:
        raise StageDataError(f"unsafe corpus label: {label!r}")
    suffix = parts[parts.index("corpus") + 1 :]
    if (
        not suffix
        or any(part in {"", ".", ".."} for part in suffix)
        or re.fullmatch(r"[A-Za-z]:", suffix[0]) is not None
    ):
        raise StageDataError(f"unsafe corpus label: {label!r}")
    return Path(*suffix)


def _locked_target(repo_root: Path, data_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise StageDataError(f"locked target path is invalid: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or "\\" in value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StageDataError(f"locked target is outside repository data directory: {value}")
    target = repo_root / Path(*relative.parts)
    resolved_target = target.resolve()
    try:
        resolved_target.relative_to(data_root)
    except ValueError as exc:
        raise StageDataError(
            f"locked target is outside repository data directory: {value}"
        ) from exc
    if resolved_target == data_root:
        raise StageDataError(f"locked target is outside repository data directory: {value}")
    if resolved_target != target:
        raise StageDataError(f"unsafe locked target uses a symbolic link: {value}")
    return target


def _validate_target_ancestors(target: Path, data_root: Path) -> None:
    ancestor = target.parent
    while ancestor != data_root:
        if ancestor.exists() and not ancestor.is_dir():
            raise StageDataError(
                f"target ancestor is not a directory: {ancestor}"
            )
        ancestor = ancestor.parent


FileIdentity = tuple[int, int]


def _file_identity(path: Path) -> FileIdentity:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _remove_if_identity(path: Path, identity: FileIdentity) -> bool:
    try:
        current = _file_identity(path)
    except FileNotFoundError:
        return True
    if current != identity:
        return False
    path.unlink()
    return True


def _copy_verified(
    source: Path, target: Path, expected_sha256: str
) -> FileIdentity | None:
    """Atomically install one verified file without overwriting a race winner."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    installed_identity: FileIdentity | None = None
    try:
        shutil.copyfile(source, temporary)
        shutil.copymode(source, temporary)
        observed = _sha256(temporary)
        if observed != expected_sha256:
            raise StageDataError(
                f"source bytes changed while staging {source}: found {observed}"
            )
        temporary_identity = _file_identity(temporary)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if (
                target.is_file()
                and not target.is_symlink()
                and _sha256(target) == expected_sha256
            ):
                return None
            raise StageDataError(f"conflicting existing target bytes: {target}")
        installed_identity = temporary_identity
        return installed_identity
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            if installed_identity is not None:
                _remove_if_identity(target, installed_identity)
            raise StageDataError(
                f"failed to clean temporary staged file {temporary}: {exc}"
            ) from exc


def _load_protocol_lock(lock: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageDataError(f"malformed protocol lock {lock}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StageDataError(f"malformed protocol lock {lock}: root must be a mapping")
    if payload.get("schema_version") not in SUPPORTED_LOCK_SCHEMAS:
        raise StageDataError(
            f"unsupported protocol lock schema in {lock}: "
            f"{payload.get('schema_version')!r}"
        )
    tracked = payload.get("tracked_files")
    if not isinstance(tracked, Mapping):
        raise StageDataError(
            f"protocol lock tracked_files must be a mapping: {lock}"
        )
    return tracked


def stage_release_data(
    *, repo_root: Path, source_dir: Path, lock_paths: Sequence[Path]
) -> StageReport:
    """Stage the corpus entries declared by ``lock_paths``."""

    root = Path(repo_root).resolve()
    data_root = (root / "data").resolve()
    source_root = (
        root / source_dir if not Path(source_dir).is_absolute() else Path(source_dir)
    ).resolve()
    try:
        source_root.relative_to(data_root)
    except ValueError as exc:
        raise StageDataError(
            f"source directory is outside repository data directory: {source_root}"
        ) from exc
    if not source_root.is_dir():
        raise StageDataError(f"source path is not a directory: {source_root}")
    resolved_locks = tuple(
        path if path.is_absolute() else root / path
        for path in (Path(value) for value in lock_paths)
    )
    if not resolved_locks:
        raise StageDataError("no protocol locks configured")
    missing_locks = tuple(
        path for path in resolved_locks if not path.is_file()
    )
    if missing_locks:
        names = ", ".join(
            (
                path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else str(path)
            )
            for path in missing_locks
        )
        raise StageDataError(f"configured protocol locks are missing: {names}")

    targets: dict[Path, tuple[Path, str]] = {}
    source_hashes: dict[Path, str] = {}
    references = 0
    for lock in resolved_locks:
        tracked = _load_protocol_lock(lock)
        lock_references = 0
        for label, row in tracked.items():
            source_relative = _corpus_relative(label)
            if source_relative is None:
                continue
            lock_references += 1
            references += 1
            if not isinstance(row, Mapping):
                raise StageDataError(
                    f"invalid tracked corpus record {label!r} in {lock}"
                )
            locked_sha256 = row.get("sha256")
            if not isinstance(locked_sha256, str) or re.fullmatch(
                r"[0-9a-f]{64}", locked_sha256
            ) is None:
                raise StageDataError(
                    f"invalid SHA-256 for tracked corpus record {label!r} in {lock}"
                )
            target = _locked_target(root, data_root, row.get("path"))
            requirement = (source_relative, locked_sha256)
            existing = targets.get(target)
            if existing is not None and existing != requirement:
                relative_target = target.relative_to(root).as_posix()
                if existing[1] != requirement[1]:
                    raise StageDataError(
                        f"inconsistent hashes for locked target {relative_target}"
                    )
                raise StageDataError(
                    f"inconsistent sources for locked target {relative_target}"
                )
            source_hash = source_hashes.get(source_relative)
            if source_hash is not None and source_hash != requirement[1]:
                raise StageDataError(
                    "inconsistent hashes for locked source "
                    f"{source_relative.as_posix()}"
                )
            source_hashes[source_relative] = requirement[1]
            targets[target] = requirement
        if lock_references == 0:
            raise StageDataError(f"protocol lock tracks no corpus files: {lock}")

    pending: list[tuple[Path, Path]] = []
    retained = 0
    for target, (source_relative, expected) in targets.items():
        source = source_root / source_relative
        resolved_source = source.resolve()
        try:
            resolved_source.relative_to(source_root)
        except ValueError as exc:
            raise StageDataError(
                f"unsafe locked source is outside source directory: {source}"
            ) from exc
        if resolved_source != source:
            raise StageDataError(
                f"unsafe locked source is not a regular in-tree file: {source}"
            )
        if not source.is_file():
            raise StageDataError(f"missing locked source file: {source}")
        if _sha256(source) != expected:
            raise StageDataError(f"source hash mismatch: {source}")
        _validate_target_ancestors(target, data_root)
        if target.exists():
            if not target.is_file():
                raise StageDataError(
                    f"locked target is not a regular file: {target}"
                )
            if _sha256(target) != expected:
                raise StageDataError(
                    f"conflicting existing target bytes: {target}"
                )
            retained += 1
            continue
        pending.append((source, target))

    copied = 0
    created: list[tuple[Path, FileIdentity]] = []
    try:
        for source, target in pending:
            expected = targets[target][1]
            identity = _copy_verified(source, target, expected)
            if identity is None:
                retained += 1
            else:
                created.append((target, identity))
                copied += 1
    except Exception as exc:
        rollback_failures: list[str] = []
        for target, identity in reversed(created):
            try:
                if not _remove_if_identity(target, identity):
                    rollback_failures.append(f"changed after creation: {target}")
            except OSError as rollback_exc:
                rollback_failures.append(f"{target}: {rollback_exc}")
        if rollback_failures:
            details = "; ".join(rollback_failures)
            raise StageDataError(
                f"staging failed and rollback was incomplete: {details}"
            ) from exc
        if isinstance(exc, StageDataError):
            raise
        raise StageDataError(f"staging failed after preflight: {exc}") from exc

    return StageReport(
        lock_count=len(resolved_locks),
        reference_count=references,
        unique_source_count=len(source_hashes),
        target_count=len(targets),
        copied_count=copied,
        retained_count=retained,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Stage every configured locked corpus target and report concise counts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--source-dir", type=Path, default=Path("data/pb"))
    parser.add_argument(
        "--lock",
        dest="lock_paths",
        action="append",
        type=Path,
        help="protocol lock path relative to --repo-root; repeat for each lock",
    )
    args = parser.parse_args(argv)
    lock_paths = tuple(args.lock_paths or DEFAULT_LOCK_PATHS)
    try:
        report = stage_release_data(
            repo_root=args.repo_root,
            source_dir=args.source_dir,
            lock_paths=lock_paths,
        )
    except StageDataError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        "ICLR RELEASE DATA STAGED: "
        f"locks={report.lock_count} "
        f"references={report.reference_count} "
        f"unique_sources={report.unique_source_count} "
        f"unique_targets={report.target_count} "
        f"copied={report.copied_count} "
        f"retained={report.retained_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
