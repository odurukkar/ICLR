"""Append-only protocol-v2 recovery for set-valued approval ballots."""

from __future__ import annotations

from collections import defaultdict
import errno
import json
import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from iclr_approval_semantics_v2 import (
    SEMANTICS_PROFILE,
    build_authenticated_semantics_receipt,
    load_series_authenticated_v2,
    validate_multicity_semantics_receipt,
)
from iclr_corpus import SeriesRef, Split
from iclr_env import RolloutState
from iclr_multicity_protocol import (
    PABULIB_COMMIT,
    LOCKED_SOURCE_FILES,
    LOCKED_TEST_FILES,
    build_multicity_splits,
    build_protocol_lock,
    evidence_gate_config,
    locked_fit_inventory,
    multicity_manifest_retained_sha256,
    protocol_config,
    split_payload,
    validate_canonical_counts,
)
from iclr_outcome import N_PROJECT_FEATURES
from iclr_priority_mes import priority_mes_outcome
from rules import mes_with_endowments


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT_V2 = ROOT / "results" / "iclr_multicity_v2"
LOCK_PROFILE_V2 = "multicity-priority-v2"
PARENT_LOCK_PROFILE = "multicity-priority-v1"
PARENT_PROTOCOL_LOCK_SHA256 = (
    "1db76ff89327277ec6e1f7b4f8af42d8aa0f4aa5158af82d457a64e2f813d039"
)
PARENT_HELDOUT_RECEIPT_SHA256 = (
    "da4d9bedf568f496ae54c5433e99988edd62b20b5217ef27a9cb972d28da34ee"
)

_DIRECTORY_FLAGS_V2 = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_NOFOLLOW_V2 = getattr(os, "O_NOFOLLOW", 0) | getattr(
    os, "O_CLOEXEC", 0
)


def _absolute_lexical_path_v2(path: Path) -> Path:
    """Return an absolute normalized path without resolving symbolic links."""

    return Path(os.path.abspath(os.fspath(path)))


def _require_nonsymlink_components_v2(
    path: Path,
    label: str,
    *,
    allow_missing: bool,
) -> Path:
    """Reject symbolic links and non-directory ancestors on a lexical path."""

    absolute = _absolute_lexical_path_v2(path)
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], start=1):
        current /= component
        try:
            identity = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                return absolute
            raise RuntimeError(f"{label} is missing: {current}") from None
        if stat.S_ISLNK(identity.st_mode):
            raise RuntimeError(f"{label} contains a symlink: {current}")
        if index < len(absolute.parts) - 1 and not stat.S_ISDIR(identity.st_mode):
            raise RuntimeError(f"{label} has a non-directory ancestor: {current}")
    return absolute


def _ensure_nonsymlink_directory_v2(path: Path, label: str) -> Path:
    """Create a directory tree while rejecting every symbolic-link component."""

    absolute = _absolute_lexical_path_v2(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            identity = os.lstat(current)
        except FileNotFoundError:
            try:
                os.mkdir(current, 0o755)
            except FileExistsError:
                pass
            identity = os.lstat(current)
        if stat.S_ISLNK(identity.st_mode):
            raise RuntimeError(f"{label} contains a symlink: {current}")
        if not stat.S_ISDIR(identity.st_mode):
            raise RuntimeError(f"{label} contains a non-directory: {current}")
    return absolute


def _open_nonsymlink_directory_v2(path: Path, label: str) -> int:
    """Bind an existing directory through descriptor-relative no-follow opens."""

    absolute = _absolute_lexical_path_v2(path)
    descriptor = -1
    try:
        descriptor = os.open(absolute.anchor, _DIRECTORY_FLAGS_V2)
        for component in absolute.parts[1:]:
            child = os.open(component, _DIRECTORY_FLAGS_V2, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if exc.errno in {
            errno.ELOOP,
            errno.EMLINK,
            errno.ENOTDIR,
            errno.ENOENT,
        }:
            raise RuntimeError(
                f"{label} must be an existing non-symlink directory"
            ) from exc
        raise


def _reopen_same_directory_v2(path: Path, bound_fd: int, label: str) -> int:
    """Reopen a lexical directory and require the originally bound inode."""

    bound = os.fstat(bound_fd)
    current_fd = _open_nonsymlink_directory_v2(path, label)
    current = os.fstat(current_fd)
    if (current.st_dev, current.st_ino) != (bound.st_dev, bound.st_ino):
        os.close(current_fd)
        raise RuntimeError(f"{label} directory identity changed")
    return current_fd


def _read_regular_at_v2(parent_fd: int, name: str, label: str) -> bytes | None:
    """Read one regular child without following links; return None when absent."""

    try:
        identity = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(identity.st_mode):
        kind = "symlink" if stat.S_ISLNK(identity.st_mode) else "non-regular file"
        raise RuntimeError(f"{label} is a {kind}: {name}")
    descriptor = os.open(
        name,
        os.O_RDONLY | _FILE_NOFOLLOW_V2,
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {name}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        or (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    ):
        raise RuntimeError(f"{label} changed during its authenticated read: {name}")
    return b"".join(chunks)


def write_immutable_bytes_artifact_v2(
    path: Path,
    content: bytes,
    *,
    label: str = "protocol-v2 immutable artifact",
) -> str:
    """Atomically create one regular file; allow only byte-identical reruns."""

    target = _absolute_lexical_path_v2(path)
    parent = _ensure_nonsymlink_directory_v2(target.parent, f"{label} parent")
    parent_fd = _open_nonsymlink_directory_v2(parent, f"{label} parent")
    temporary: str | None = None
    try:
        existing = _read_regular_at_v2(parent_fd, target.name, label)
        if existing is not None:
            if existing != content:
                raise RuntimeError(f"refusing to overwrite {label}: {target}")
            current_fd = _reopen_same_directory_v2(
                parent,
                parent_fd,
                f"{label} parent",
            )
            try:
                current = _read_regular_at_v2(current_fd, target.name, label)
            finally:
                os.close(current_fd)
            if current != existing:
                raise RuntimeError(f"{label} changed at its canonical path: {target}")
            return hashlib.sha256(existing).hexdigest()

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _FILE_NOFOLLOW_V2
        for _ in range(100):
            candidate = f".{target.name}.{secrets.token_hex(8)}.tmp"
            try:
                descriptor = os.open(candidate, flags, 0o644, dir_fd=parent_fd)
            except FileExistsError:
                continue
            temporary = candidate
            break
        else:
            raise RuntimeError(f"could not allocate staged file for {label}")

        try:
            view = memoryview(content)
            written = 0
            while written < len(view):
                written += os.write(descriptor, view[written:])
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o644)
        finally:
            os.close(descriptor)

        preinstall_fd = _reopen_same_directory_v2(
            parent,
            parent_fd,
            f"{label} parent",
        )
        os.close(preinstall_fd)
        try:
            os.link(
                temporary,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.fsync(parent_fd)
        except FileExistsError:
            existing = _read_regular_at_v2(parent_fd, target.name, label)
            if existing != content:
                raise RuntimeError(f"refusing to overwrite {label}: {target}")
        installed = _read_regular_at_v2(parent_fd, target.name, label)
        if installed != content:
            raise RuntimeError(f"{label} changed during atomic installation: {target}")
        current_fd = _reopen_same_directory_v2(
            parent,
            parent_fd,
            f"{label} parent",
        )
        try:
            current = _read_regular_at_v2(current_fd, target.name, label)
        finally:
            os.close(current_fd)
        if current != installed:
            raise RuntimeError(f"{label} changed at its canonical path: {target}")
        return hashlib.sha256(content).hexdigest()
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def read_regular_bytes_artifact_v2(
    path: Path,
    *,
    label: str,
    allow_missing: bool = False,
) -> bytes | None:
    """Read one exact regular artifact through a no-follow parent descriptor."""

    target = _absolute_lexical_path_v2(path)
    _require_nonsymlink_components_v2(
        target.parent,
        f"{label} parent",
        allow_missing=False,
    )
    parent_fd = _open_nonsymlink_directory_v2(target.parent, f"{label} parent")
    try:
        content = _read_regular_at_v2(parent_fd, target.name, label)
        current_fd = _reopen_same_directory_v2(
            target.parent,
            parent_fd,
            f"{label} parent",
        )
        try:
            current = _read_regular_at_v2(current_fd, target.name, label)
        finally:
            os.close(current_fd)
        if current != content:
            raise RuntimeError(f"{label} changed at its canonical path: {target}")
    finally:
        os.close(parent_fd)
    if content is None and not allow_missing:
        raise RuntimeError(f"{label} is missing: {target}")
    return content


def preflight_immutable_bytes_artifact_v2(
    path: Path,
    content: bytes,
    *,
    label: str,
) -> None:
    """Reject a conflicting target without creating any path or file."""

    target = _require_nonsymlink_components_v2(
        path,
        label,
        allow_missing=True,
    )
    try:
        parent_identity = os.lstat(target.parent)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(parent_identity.st_mode):
        raise RuntimeError(f"{label} parent is a symlink: {target.parent}")
    if not stat.S_ISDIR(parent_identity.st_mode):
        raise RuntimeError(f"{label} parent is not a directory: {target.parent}")
    existing = read_regular_bytes_artifact_v2(
        target,
        label=label,
        allow_missing=True,
    )
    if existing is not None and existing != content:
        raise RuntimeError(f"refusing to overwrite {label}: {target}")


def write_immutable_text_artifact_v2(path: Path, content: str) -> str:
    """Write an immutable UTF-8 protocol-v2 text artifact safely."""

    return write_immutable_bytes_artifact_v2(path, content.encode("utf-8"))


def write_immutable_json_artifact_v2(
    path: Path,
    payload: Mapping[str, object],
) -> str:
    """Write canonical JSON through the protocol-v2 no-follow writer."""

    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return write_immutable_text_artifact_v2(path, content)


def canonical_fit_path_v2(
    result_root: Path,
    spec: Mapping[str, object],
    raw_path: Path,
) -> Path:
    """Require one fit at its exact, symlink-free protocol-v2 location."""

    result = _absolute_lexical_path_v2(result_root)
    expected = result / "fits" / str(spec["split"]) / str(spec["arm"]) / (
        f"seed-{int(spec['seed'])}.json"
    )
    supplied = _absolute_lexical_path_v2(raw_path)
    if supplied != expected:
        raise RuntimeError(
            f"v2 fit is not at its canonical result path for {spec}: "
            f"{supplied} != {expected}"
        )
    _require_nonsymlink_components_v2(
        expected,
        "canonical v2 fits path",
        allow_missing=False,
    )
    if supplied.resolve() != supplied:
        raise RuntimeError(f"canonical v2 fits path contains a symlink: {supplied}")
    try:
        supplied.relative_to(result / "fits")
    except ValueError as exc:
        raise RuntimeError(f"v2 fit lies outside the canonical v2 fits root: {supplied}") from exc
    return supplied


def canonical_result_output_path_v2(
    result_root: Path,
    relative_path: Path,
) -> Path:
    """Return one exact symlink-free output name beneath the v2 result root."""

    result = _absolute_lexical_path_v2(result_root)
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"invalid protocol-v2 result path: {relative}")
    output = result / relative
    _require_nonsymlink_components_v2(
        output,
        "protocol-v2 result output",
        allow_missing=True,
    )
    return output


def canonical_result_root_v2(path: Path, *, allow_missing: bool = False) -> Path:
    """Retain the lexical v2 result root and reject symbolic-link components."""

    return _require_nonsymlink_components_v2(
        path,
        "protocol-v2 result root",
        allow_missing=allow_missing,
    )
SPLIT_NAMES = (
    "temporal_2022",
    "city_out_Poland_Warszawa",
    "city_out_Poland_Gdynia",
    "city_out_Poland_Łódź",
)
STRUCTURAL_SOURCE_FILES_V2 = (
    "cohorts.py",
    "iclr_approval_semantics_v2.py",
    "iclr_corpus.py",
    "iclr_env.py",
    "iclr_multicity_protocol.py",
    "iclr_multicity_protocol_v2.py",
    "iclr_outcome.py",
    "iclr_policy.py",
    "iclr_priority_mes.py",
    "parse_pb.py",
    "rules.py",
    "run_experiments.py",
)
LOCKED_SOURCE_FILES_V2 = (
    *LOCKED_SOURCE_FILES,
    "iclr_approval_semantics_v2.py",
    "iclr_multicity_protocol_v2.py",
    "iclr_multicity_train_v2.py",
    "iclr_multicity_evaluate_v2.py",
)
LOCKED_TEST_FILES_V2 = (
    *LOCKED_TEST_FILES,
    "test_iclr_approval_semantics_v2.py",
    "test_iclr_multicity_protocol_v2.py",
    "test_iclr_multicity_train_v2.py",
    "test_iclr_multicity_evaluate_v2.py",
)
FIT_SOURCE_FILES_V2 = LOCKED_SOURCE_FILES_V2


@dataclass(frozen=True)
class ParentProtocolSnapshotV2:
    """One authenticated byte snapshot of the complete parent lock."""

    payload: Mapping[str, object]
    content: bytes
    sha256: str


@dataclass(frozen=True)
class ProtocolLockSnapshotV2:
    """One fully verified byte snapshot of the protocol-v2 lock."""

    path: Path
    payload: Mapping[str, object]
    content: bytes
    sha256: str


@dataclass(frozen=True)
class JsonArtifactSnapshotV2:
    """One parsed JSON artifact whose hash and payload share the same bytes."""

    label: str
    path: Path
    content: bytes
    sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class ProtocolInputsSnapshotV2:
    """Manifest and split objects parsed only from authenticated snapshots."""

    manifest: JsonArtifactSnapshotV2
    split_artifacts: Mapping[str, JsonArtifactSnapshotV2]
    index: Mapping[str, SeriesRef]
    splits: Mapping[str, Split]


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be canonical SHA-256")
    return value


def exact_json_equal_v2(actual: object, expected: object) -> bool:
    """Compare JSON-shaped values without Python's bool/number aliases."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            exact_json_equal_v2(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_json_equal_v2(observed, wanted)
            for observed, wanted in zip(actual, expected)
        )
    return actual == expected


def _read_json_snapshot_v2(
    path: Path,
    label: str,
    *,
    expected_sha256: str | None = None,
) -> JsonArtifactSnapshotV2:
    """Read, hash, and parse one JSON artifact from the same byte image."""

    resolved = _absolute_lexical_path_v2(path)
    content = read_regular_bytes_artifact_v2(
        resolved,
        label=label,
    )
    assert content is not None
    observed = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and observed != _require_sha256(
        expected_sha256,
        f"locked digest for {label}",
    ):
        raise RuntimeError(
            f"locked digest differs for {label}: {observed} != {expected_sha256}"
        )
    payload = json.loads(content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"v2 JSON artifact must be an object for {label}")
    return JsonArtifactSnapshotV2(label, resolved, content, observed, payload)


def _index_from_manifest_snapshot_v2(
    snapshot: JsonArtifactSnapshotV2,
    data_dir: Path,
    *,
    enforce_expected_counts: bool = True,
) -> dict[str, SeriesRef]:
    """Reconstruct the canonical index without reopening the manifest path."""

    manifest = snapshot.payload
    rows = manifest.get("files", [])
    if not isinstance(rows, list) or len(rows) != int(
        manifest.get("n_files", -1)
    ):
        raise RuntimeError("corpus manifest file count is internally inconsistent")
    if enforce_expected_counts and manifest.get("source_commit") != PABULIB_COMMIT:
        raise RuntimeError(
            f"unexpected Pabulib commit {manifest.get('source_commit')!r}"
        )
    if any(not isinstance(row, Mapping) for row in rows):
        raise RuntimeError("corpus manifest contains a malformed file row")
    listed_names = [str(row["name"]) for row in rows]
    if len(listed_names) != len(set(listed_names)):
        raise RuntimeError("corpus manifest contains duplicate file names")
    data = _require_nonsymlink_components_v2(
        data_dir,
        "protocol-v2 corpus directory",
        allow_missing=False,
    )
    actual_names = sorted(path.name for path in data.glob("*.pb"))
    if sorted(listed_names) != actual_names:
        raise RuntimeError("staged corpus files differ from the corpus manifest")

    by_series: dict[str, dict[int, Path]] = defaultdict(dict)
    for row in rows:
        name = str(row["name"])
        path = data / name
        raw_bytes = read_regular_bytes_artifact_v2(
            path,
            label=f"protocol-v2 corpus file {name}",
        )
        assert raw_bytes is not None
        if len(raw_bytes) != int(row["bytes"]):
            raise RuntimeError(f"byte-size mismatch for corpus file {name}")
        observed_hash = hashlib.sha256(raw_bytes).hexdigest()
        if observed_hash != row["sha256"]:
            raise RuntimeError(
                f"hash mismatch for corpus file {name}: "
                f"expected {row['sha256']}, found {observed_hash}"
            )
        series = str(row["series"])
        year = int(row["year"])
        if year in by_series[series]:
            raise RuntimeError(f"duplicate corpus election {series} {year}")
        by_series[series][year] = path
    index = {
        series: SeriesRef(
            key=series,
            years=tuple(sorted(years)),
            paths=tuple(years[year] for year in sorted(years)),
        )
        for series, years in sorted(by_series.items())
    }
    if enforce_expected_counts:
        validate_canonical_counts(tuple(index.values()))
    if int(manifest.get("n_series", -1)) != len(index):
        raise RuntimeError("corpus manifest series count is internally inconsistent")
    if int(manifest.get("n_elections", -1)) != sum(
        len(ref.years) for ref in index.values()
    ):
        raise RuntimeError("corpus manifest election count is internally inconsistent")
    return index


def _split_from_snapshot_v2(
    snapshot: JsonArtifactSnapshotV2,
    index: Mapping[str, SeriesRef],
    *,
    expected_name: str,
) -> Split:
    """Regenerate and validate one split without reopening its JSON path."""

    name = str(snapshot.payload.get("name", ""))
    if name != expected_name:
        raise RuntimeError(
            f"frozen split name differs: {name!r} != {expected_name!r}"
        )
    generated = build_multicity_splits(index)
    if name not in generated:
        raise RuntimeError(f"unknown frozen split name {name!r}")
    expected_payload = split_payload(generated[name], index)
    if snapshot.payload != expected_payload:
        raise RuntimeError(f"frozen split differs from generated protocol: {name}")
    return generated[name]


def load_protocol_inputs_snapshot_v2(
    lock_payload: Mapping[str, object] | None,
    result_root: Path,
    data_dir: Path,
    split_names: Sequence[str],
    *,
    enforce_expected_counts: bool = True,
) -> ProtocolInputsSnapshotV2:
    """Parse manifest and splits only from one locked byte snapshot each."""

    names = tuple(split_names)
    if len(names) != len(set(names)) or any(
        name not in SPLIT_NAMES for name in names
    ):
        raise RuntimeError("protocol-v2 input snapshot names differ")
    tracked = (
        lock_payload.get("tracked_files") if lock_payload is not None else None
    )
    if lock_payload is not None and not isinstance(tracked, Mapping):
        raise RuntimeError("protocol-v2 lock tracked files are malformed")

    def expected_digest(label: str) -> str | None:
        if lock_payload is None:
            return None
        row = tracked.get(label) if isinstance(tracked, Mapping) else None
        if not isinstance(row, Mapping):
            raise RuntimeError(f"protocol-v2 lock is missing {label}")
        return _require_sha256(row.get("sha256"), f"protocol-v2 lock {label}")

    result = _require_nonsymlink_components_v2(
        result_root,
        "protocol-v2 input result root",
        allow_missing=False,
    )
    manifest = _read_json_snapshot_v2(
        result / "corpus_manifest.json",
        "artifact/corpus_manifest",
        expected_sha256=expected_digest("artifact/corpus_manifest"),
    )
    split_artifacts = {
        name: _read_json_snapshot_v2(
            result / "splits" / f"{name}.json",
            f"split/{name}",
            expected_sha256=expected_digest(f"split/{name}"),
        )
        for name in names
    }
    _validate_redacted_manifest_v2(manifest.payload)
    index = _index_from_manifest_snapshot_v2(
        manifest,
        Path(data_dir),
        enforce_expected_counts=enforce_expected_counts,
    )
    splits = {
        name: _split_from_snapshot_v2(
            split_artifacts[name],
            index,
            expected_name=name,
        )
        for name in names
    }
    return ProtocolInputsSnapshotV2(manifest, split_artifacts, index, splits)


def assert_protocol_inputs_unchanged_v2(
    snapshot: ProtocolInputsSnapshotV2,
) -> None:
    """Reject persistent path drift after computation from locked snapshots."""

    artifacts = (snapshot.manifest, *snapshot.split_artifacts.values())
    for artifact in artifacts:
        if read_regular_bytes_artifact_v2(
            artifact.path,
            label=artifact.label,
        ) != artifact.content:
            raise RuntimeError(f"locked v2 input changed during use: {artifact.label}")


def _canonical_source_sha256(
    source_sha256: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise ValueError("source_sha256 must be a non-empty mapping")
    canonical: dict[str, str] = {}
    for name, digest in sorted(source_sha256.items()):
        if not isinstance(name, str) or not name or name.startswith("/"):
            raise ValueError("source_sha256 names must be relative paths")
        parts = Path(name).parts
        if ".." in parts:
            raise ValueError("source_sha256 names must not traverse parents")
        canonical[name] = _require_sha256(digest, f"source_sha256[{name!r}]")
    return canonical


def structural_source_sha256_v2(repo_root: Path = ROOT) -> dict[str, str]:
    """Hash the exact source closure used by the v2 structural gate."""

    root = Path(repo_root).resolve()
    hashes: dict[str, str] = {}
    for name in STRUCTURAL_SOURCE_FILES_V2:
        path = root / "src" / name
        content = read_regular_bytes_artifact_v2(
            path,
            label=f"v2 structural source {name}",
        )
        assert content is not None
        hashes[f"src/{name}"] = hashlib.sha256(content).hexdigest()
    return _canonical_source_sha256(hashes)


def protocol_config_v2() -> dict[str, object]:
    """Retain every v1 decision while making ingestion semantics explicit."""

    config = protocol_config()
    config["approval_semantics"] = {
        "ballot_model": "set",
        "profile": SEMANTICS_PROFILE,
        "canonicalization": "retain each project's first occurrence",
        "applied_before": [
            "feature extraction",
            "supporter construction",
            "rule execution",
            "training",
            "evaluation",
            "trace validation",
        ],
        "raw_bytes_mutated": False,
    }
    # V1 inherited the CMA-ES library default implicitly. V2 freezes that
    # choice explicitly so a caller cannot silently widen the search budget.
    config["popsize"] = None
    return config


def locked_fit_inventory_v2() -> list[dict[str, object]]:
    """Use the complete v1 fit matrix under the corrected v2 input contract."""

    return locked_fit_inventory()


def _load_parent_protocol_lock_v2(
    parent_result_root: Path,
) -> ParentProtocolSnapshotV2:
    """Read, hash, parse, and validate the parent lock from the same bytes."""

    parent = _require_nonsymlink_components_v2(
        parent_result_root,
        "parent protocol result root",
        allow_missing=False,
    )
    lock_path = parent / "protocol_lock.json"
    content = read_regular_bytes_artifact_v2(
        lock_path,
        label="parent protocol lock",
    )
    assert content is not None
    observed = hashlib.sha256(content).hexdigest()
    if observed != PARENT_PROTOCOL_LOCK_SHA256:
        raise RuntimeError(
            "parent protocol lock differs: "
            f"{observed} != {PARENT_PROTOCOL_LOCK_SHA256}"
        )
    parent_lock = json.loads(content.decode("utf-8"))
    if not isinstance(parent_lock, dict):
        raise RuntimeError("parent protocol lock must be a JSON object")
    if (
        type(parent_lock.get("schema_version")) is not int
        or parent_lock.get("schema_version") != 1
        or parent_lock.get("lock_profile") != PARENT_LOCK_PROFILE
        or not exact_json_equal_v2(parent_lock.get("protocol"), protocol_config())
        or not exact_json_equal_v2(
            parent_lock.get("evidence_gates"), evidence_gate_config()
        )
        or not exact_json_equal_v2(
            parent_lock.get("fit_inventory"), locked_fit_inventory()
        )
    ):
        raise RuntimeError("parent protocol lock semantics differ")
    return ParentProtocolSnapshotV2(parent_lock, content, observed)


def parent_protocol_lock_sha256_v2(parent_result_root: Path) -> str:
    """Authenticate the exact v1 protocol whose decisions v2 inherits."""

    return _load_parent_protocol_lock_v2(parent_result_root).sha256


def _parent_fit_tail(spec: Mapping[str, object]) -> Path:
    return Path(
        "fits",
        str(spec["split"]),
        str(spec["arm"]),
        f"seed-{int(spec['seed'])}.json",
    )


def parent_heldout_receipt_sha256_v2(
    parent_result_root: Path,
    *,
    parent_snapshot: ParentProtocolSnapshotV2 | None = None,
) -> str:
    """Authenticate the completed v1 opening and its exact 24 fit bytes."""

    parent = _require_nonsymlink_components_v2(
        parent_result_root,
        "parent protocol result root",
        allow_missing=False,
    )
    snapshot = parent_snapshot or _load_parent_protocol_lock_v2(parent)
    receipt_path = parent / "heldout_opened.json"
    content = read_regular_bytes_artifact_v2(
        receipt_path,
        label="parent held-out receipt",
    )
    assert content is not None
    observed = hashlib.sha256(content).hexdigest()
    if observed != PARENT_HELDOUT_RECEIPT_SHA256:
        raise RuntimeError(
            "parent held-out receipt differs: "
            f"{observed} != {PARENT_HELDOUT_RECEIPT_SHA256}"
        )
    payload = json.loads(content.decode("utf-8"))
    expected_fields = {
        "schema_version",
        "lock_sha256",
        "fit_sha256",
        "structural_gates_sha256",
        "opened_at_utc",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise RuntimeError("parent held-out receipt schema differs")
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != 1:
        raise RuntimeError("parent held-out receipt schema version differs")
    if payload.get("lock_sha256") != snapshot.sha256:
        raise RuntimeError("parent held-out receipt lock binding differs")
    tracked = snapshot.payload.get("tracked_files")
    structural_row = (
        tracked.get("artifact/structural_gates")
        if isinstance(tracked, Mapping)
        else None
    )
    if (
        not isinstance(structural_row, Mapping)
        or payload.get("structural_gates_sha256") != structural_row.get("sha256")
    ):
        raise RuntimeError("parent held-out receipt structural binding differs")
    fit_sha256 = payload.get("fit_sha256")
    if not isinstance(fit_sha256, Mapping):
        raise RuntimeError("parent held-out receipt fit inventory differs")
    matched_labels: set[str] = set()
    for spec in locked_fit_inventory():
        tail = _parent_fit_tail(spec)
        labels = [
            label
            for label in fit_sha256
            if isinstance(label, str)
            and tuple(Path(label).parts[-4:]) == tuple(tail.parts)
        ]
        if len(labels) != 1:
            raise RuntimeError(f"parent held-out receipt fit path differs for {spec}")
        label = labels[0]
        path = parent / tail
        fit_bytes = read_regular_bytes_artifact_v2(
            path,
            label=f"parent fit {tail.as_posix()}",
        )
        assert fit_bytes is not None
        if fit_sha256[label] != hashlib.sha256(fit_bytes).hexdigest():
            raise RuntimeError(f"parent held-out receipt fit digest differs for {spec}")
        matched_labels.add(label)
    if set(fit_sha256) != matched_labels:
        raise RuntimeError("parent held-out receipt fit inventory differs")
    return observed


def _load_parent_lineage_v2(
    parent_result_root: Path,
) -> tuple[ParentProtocolSnapshotV2, str]:
    snapshot = _load_parent_protocol_lock_v2(parent_result_root)
    receipt_sha256 = parent_heldout_receipt_sha256_v2(
        parent_result_root,
        parent_snapshot=snapshot,
    )
    return snapshot, receipt_sha256


def build_protocol_lock_payload_v2(
    repo_root: Path,
    tracked_files: Mapping[str, Path],
    *,
    parent_result_root: Path,
    _parent_lineage: tuple[ParentProtocolSnapshotV2, str] | None = None,
) -> dict[str, object]:
    """Build a schema-v2 lock payload over an explicit mandatory inventory."""

    parent_lineage = _parent_lineage or _load_parent_lineage_v2(parent_result_root)
    parent_snapshot, parent_heldout_sha256 = parent_lineage
    payload = build_protocol_lock(repo_root, tracked_files)
    payload.update(
        {
            "schema_version": 2,
            "lock_profile": LOCK_PROFILE_V2,
            "mandatory_file_count": len(tracked_files),
            "parent_protocol_lock_sha256": parent_snapshot.sha256,
            "parent_heldout_receipt_sha256": parent_heldout_sha256,
            "fit_inventory": locked_fit_inventory_v2(),
            "protocol": protocol_config_v2(),
            "evidence_gates": evidence_gate_config(),
        }
    )
    validate_mandatory_lock_profile_v2(
        payload,
        repo_root,
        tracked_files,
        parent_result_root=parent_result_root,
        _parent_lineage=parent_lineage,
    )
    return payload


def validate_mandatory_lock_profile_v2(
    payload: Mapping[str, object],
    repo_root: Path,
    expected_files: Mapping[str, Path],
    *,
    parent_result_root: Path,
    _parent_lineage: tuple[ParentProtocolSnapshotV2, str] | None = None,
) -> None:
    """Require the exact v2 semantics, lineage, paths, and tracked-file set."""

    expected_top_level = {
        "schema_version",
        "lock_profile",
        "mandatory_file_count",
        "parent_protocol_lock_sha256",
        "parent_heldout_receipt_sha256",
        "tracked_files",
        "fit_inventory",
        "protocol",
        "evidence_gates",
    }
    if set(payload) != expected_top_level:
        raise RuntimeError("v2 protocol lock fields differ")
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != 2:
        raise RuntimeError("v2 protocol lock schema semantics differ")
    if payload.get("lock_profile") != LOCK_PROFILE_V2:
        raise RuntimeError("v2 protocol lock profile differs")
    if not exact_json_equal_v2(payload.get("protocol"), protocol_config_v2()):
        raise RuntimeError("v2 protocol semantics differ")
    if not exact_json_equal_v2(
        payload.get("evidence_gates"), evidence_gate_config()
    ):
        raise RuntimeError("v2 evidence-gate semantics differ")
    if not exact_json_equal_v2(
        payload.get("fit_inventory"), locked_fit_inventory_v2()
    ):
        raise RuntimeError("v2 fit inventory differs")
    parent_snapshot, parent_heldout_sha256 = (
        _parent_lineage or _load_parent_lineage_v2(parent_result_root)
    )
    if payload.get("parent_protocol_lock_sha256") != parent_snapshot.sha256:
        raise RuntimeError("v2 parent protocol lock differs")
    if payload.get("parent_heldout_receipt_sha256") != parent_heldout_sha256:
        raise RuntimeError("v2 parent held-out receipt differs")
    if (
        type(payload.get("mandatory_file_count")) is not int
        or payload.get("mandatory_file_count") != len(expected_files)
    ):
        raise RuntimeError("v2 mandatory-file count differs")

    tracked = payload.get("tracked_files")
    if not isinstance(tracked, dict) or set(tracked) != set(expected_files):
        raise RuntimeError("v2 mandatory tracked-file set differs")
    lineage_bindings = {
        "parent/protocol_lock": (
            parent_snapshot.sha256,
            "v2 tracked parent protocol lock differs from lineage snapshot",
        ),
        "parent/heldout_opened": (
            parent_heldout_sha256,
            "v2 tracked parent held-out receipt differs from lineage snapshot",
        ),
    }
    for label, (expected_digest, error_message) in lineage_bindings.items():
        if label not in expected_files:
            continue
        row = tracked.get(label)
        if not isinstance(row, dict) or row.get("sha256") != expected_digest:
            raise RuntimeError(error_message)
    root = Path(repo_root).resolve()
    for label, expected_path in expected_files.items():
        path = _require_nonsymlink_components_v2(
            expected_path,
            f"v2 mandatory file {label}",
            allow_missing=False,
        )
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"v2 mandatory file lies outside repository for {label}: {path}"
            ) from exc
        row = tracked.get(label)
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RuntimeError(f"v2 tracked-file row differs for {label}")
        content = read_regular_bytes_artifact_v2(
            path,
            label=f"v2 mandatory file {label}",
        )
        assert content is not None
        if (
            row.get("path") != relative
            or row.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise RuntimeError(f"v2 tracked-file binding differs for {label}")


def _validate_redacted_manifest_v2(manifest: Mapping[str, object]) -> None:
    """Require the two public, machine-independent staged path fields."""

    if manifest.get("source_dir") != "pabulib_files/pb_files":
        raise RuntimeError("v2 corpus manifest source_dir is not redacted")
    if manifest.get("destination") != "data/pb_multicity":
        raise RuntimeError("v2 corpus manifest destination is not canonical")


def stage_protocol_inputs_v2(
    parent_result_root: Path,
    result_root: Path,
    data_dir: Path,
    *,
    enforce_expected_counts: bool = True,
) -> dict[str, object]:
    """Copy immutable v1 boundaries and add the authenticated v2 receipt."""

    parent_result_root = Path(parent_result_root).resolve()
    result_root = _require_nonsymlink_components_v2(
        Path(result_root),
        "protocol-v2 result root",
        allow_missing=True,
    )
    if (
        parent_result_root == result_root
        or parent_result_root in result_root.parents
        or result_root in parent_result_root.parents
    ):
        raise ValueError(
            "protocol-v2 result root must be separate from and not nested in protocol v1"
        )

    parent_snapshot, parent_heldout_sha256 = _load_parent_lineage_v2(
        parent_result_root
    )
    parent_lock_sha256 = parent_snapshot.sha256
    parent_lock_path = parent_result_root / "protocol_lock.json"
    parent_lock = parent_snapshot.payload
    tracked = parent_lock.get("tracked_files")
    if not isinstance(tracked, dict):
        raise RuntimeError("parent protocol lock tracked files differ")

    parent_manifest = parent_result_root / "corpus_manifest.json"
    manifest_bytes = read_regular_bytes_artifact_v2(
        parent_manifest,
        label="parent corpus manifest",
    )
    assert manifest_bytes is not None
    manifest_row = tracked.get("artifact/corpus_manifest")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        not isinstance(manifest_row, dict)
        or manifest_row.get("sha256") != manifest_digest
    ):
        raise RuntimeError("parent corpus manifest differs from its protocol lock")
    split_bytes: dict[str, bytes] = {}
    for name in SPLIT_NAMES:
        source = parent_result_root / "splits" / f"{name}.json"
        content = read_regular_bytes_artifact_v2(
            source,
            label=f"parent split {name}",
        )
        assert content is not None
        row = tracked.get(f"split/{name}")
        if (
            not isinstance(row, dict)
            or row.get("sha256") != hashlib.sha256(content).hexdigest()
        ):
            raise RuntimeError(f"parent split differs from its lock: {name}")
        split_bytes[name] = content

    manifest_path = result_root / "corpus_manifest.json"
    manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest_payload, dict):
        raise RuntimeError("parent corpus manifest must be a JSON object")
    manifest_payload["source_dir"] = "pabulib_files/pb_files"
    manifest_payload["destination"] = "data/pb_multicity"
    _validate_redacted_manifest_v2(manifest_payload)
    manifest_sha256 = write_immutable_json_artifact_v2(
        manifest_path,
        manifest_payload,
    )
    split_hashes = {}
    for name in SPLIT_NAMES:
        target = result_root / "splits" / f"{name}.json"
        split_hashes[name] = write_immutable_text_artifact_v2(
            target,
            split_bytes[name].decode("utf-8"),
        )

    receipt = build_authenticated_semantics_receipt(
        manifest_path,
        Path(data_dir),
        enforce_expected_counts=enforce_expected_counts,
    )
    if enforce_expected_counts:
        validate_multicity_semantics_receipt(receipt)
    receipt_path = result_root / "approval_semantics_receipt.json"
    receipt_sha256 = write_immutable_json_artifact_v2(receipt_path, receipt)
    return {
        "corpus_manifest_sha256": manifest_sha256,
        "split_sha256": split_hashes,
        "approval_semantics_receipt_sha256": receipt_sha256,
        "parent_corpus_manifest_sha256": manifest_digest,
        "parent_protocol_lock_sha256": parent_lock_sha256,
        "parent_heldout_receipt_sha256": parent_heldout_sha256,
    }


def verify_priority_mes_corpus_v2(
    refs: Sequence[SeriesRef],
    *,
    semantics_receipt: Mapping[str, object],
    semantics_receipt_sha256: str,
    corpus_semantic_sha256: str,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    """Run structural MES gates only on normalized protocol-v2 instances."""

    semantics_receipt_sha256 = _require_sha256(
        semantics_receipt_sha256,
        "semantics receipt digest",
    )
    corpus_semantic_sha256 = _require_sha256(
        corpus_semantic_sha256,
        "corpus semantic digest",
    )
    canonical_source_sha256 = structural_source_sha256_v2(repo_root)
    zero = np.zeros(N_PROJECT_FEATURES)
    n_elections = 0
    normalized_ballots = 0
    duplicate_approvals = 0
    winner_identity_checks = 0
    determinism_checks = 0
    budget_checks = 0
    unique_payer_round_checks = 0
    payment_rounds = 0
    for ref in refs:
        instances = load_series_authenticated_v2(ref, semantics_receipt)
        for year in ref.years:
            instance = instances[year]
            n_elections += 1
            normalized_ballots += len(instance.votes)
            duplicate_approvals += sum(
                len(vote.projects) - len(set(vote.projects))
                for vote in instance.votes
            )
            for completion in (False, True):
                expected = mes_with_endowments(instance, completion=completion)
                first = priority_mes_outcome(
                    instance,
                    RolloutState(),
                    zero,
                    completion=completion,
                )
                second = priority_mes_outcome(
                    instance,
                    RolloutState(),
                    zero,
                    completion=completion,
                )
                winner_identity_checks += 1
                if set(first.winners) != expected:
                    raise RuntimeError(
                        f"v2 MES containment failed for {ref.key} {year} "
                        f"completion={completion}"
                    )
                determinism_checks += 1
                if first != second:
                    raise RuntimeError(
                        f"v2 determinism failed for {ref.key} {year} "
                        f"completion={completion}"
                    )
                spent = sum(
                    instance.projects[project_id].cost
                    for project_id in first.winners
                )
                budget_checks += 1
                if spent > instance.budget + 1e-6:
                    raise RuntimeError(
                        f"v2 budget feasibility failed for {ref.key} {year}: {spent}"
                    )
                for round_ in first.rounds:
                    unique_payer_round_checks += 1
                    if len(round_.supporter_ids) != len(set(round_.supporter_ids)):
                        raise RuntimeError(
                            f"v2 duplicate payer in {ref.key} {year} "
                            f"project {round_.project_id}"
                        )
                payment_rounds += len(first.rounds)
    if duplicate_approvals:
        raise RuntimeError(
            f"v2 ingestion retained {duplicate_approvals} duplicate approvals"
        )
    return {
        "schema_version": 2,
        "status": "pass",
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_receipt_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "source_sha256": canonical_source_sha256,
        "n_series": len(refs),
        "n_elections": n_elections,
        "approval_ballots_checked": normalized_ballots,
        "duplicate_tokens_remaining": duplicate_approvals,
        "winner_identity_checks": winner_identity_checks,
        "determinism_checks": determinism_checks,
        "budget_checks": budget_checks,
        "unique_payer_round_checks": unique_payer_round_checks,
        "payment_rounds_checked": payment_rounds,
        "completion_modes": [False, True],
        "zero_weights": zero.tolist(),
    }


def validate_structural_gates_v2(
    path: Path,
    *,
    semantics_receipt_sha256: str,
    corpus_semantic_sha256: str,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    """Require complete corpus counts and the exact v2 semantics receipt."""

    content = read_regular_bytes_artifact_v2(
        path,
        label="v2 structural gate",
    )
    assert content is not None
    payload = json.loads(content.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("v2 structural gate must be a JSON object")
    expected = {
        "schema_version": 2,
        "status": "pass",
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": _require_sha256(
            semantics_receipt_sha256,
            "semantics receipt digest",
        ),
        "corpus_semantic_sha256": _require_sha256(
            corpus_semantic_sha256,
            "corpus semantic digest",
        ),
        "source_sha256": structural_source_sha256_v2(repo_root),
        "n_series": 75,
        "n_elections": 397,
        "approval_ballots_checked": 1_126_949,
        "duplicate_tokens_remaining": 0,
        "winner_identity_checks": 794,
        "determinism_checks": 794,
        "budget_checks": 794,
        "unique_payer_round_checks": 11_078,
        "payment_rounds_checked": 11_078,
        "completion_modes": [False, True],
        "zero_weights": [0.0] * 5,
    }
    if set(payload) != set(expected):
        raise RuntimeError("v2 structural gate fields differ")
    integer_fields = (
        "n_series",
        "n_elections",
        "approval_ballots_checked",
        "duplicate_tokens_remaining",
        "winner_identity_checks",
        "determinism_checks",
        "budget_checks",
        "unique_payer_round_checks",
        "payment_rounds_checked",
    )
    if any(type(payload.get(field)) is not int for field in integer_fields):
        raise RuntimeError("v2 structural gate integer field has the wrong type")
    completion_modes = payload.get("completion_modes")
    if (
        not isinstance(completion_modes, list)
        or len(completion_modes) != 2
        or any(type(value) is not bool for value in completion_modes)
    ):
        raise RuntimeError("v2 structural gate completion_modes type differs")
    zero_weights = payload.get("zero_weights")
    if (
        not isinstance(zero_weights, list)
        or len(zero_weights) != 5
        or any(type(value) is not float for value in zero_weights)
    ):
        raise RuntimeError("v2 structural gate zero_weights type differs")
    for field, value in expected.items():
        if not exact_json_equal_v2(payload.get(field), value):
            raise RuntimeError(
                f"v2 structural gate field {field!r} differs: "
                f"{payload.get(field)!r} != {value!r}"
            )
    return payload


def build_structural_report_v2(
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Derive a structural report only from authenticated staged v2 inputs."""

    root = Path(repo_root).resolve()
    result = canonical_result_root_v2(result_root)
    data = _require_nonsymlink_components_v2(
        data_dir,
        "protocol-v2 data root",
        allow_missing=False,
    )
    manifest_path = result / "corpus_manifest.json"
    receipt_path = result / "approval_semantics_receipt.json"
    manifest_snapshot = _read_json_snapshot_v2(
        manifest_path,
        "artifact/corpus_manifest",
    )
    manifest = manifest_snapshot.payload
    receipt_bytes = read_regular_bytes_artifact_v2(
        receipt_path,
        label="v2 semantics receipt",
    )
    assert receipt_bytes is not None
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or not isinstance(receipt, dict):
        raise RuntimeError("v2 staged manifest and receipt must be JSON objects")
    _validate_redacted_manifest_v2(manifest)
    validate_multicity_semantics_receipt(receipt)
    if receipt.get(
        "parent_manifest_retained_sha256"
    ) != multicity_manifest_retained_sha256(manifest):
        raise RuntimeError("v2 semantics receipt manifest binding differs")

    index = _index_from_manifest_snapshot_v2(manifest_snapshot, data)
    if (
        read_regular_bytes_artifact_v2(
            manifest_path,
            label="artifact/corpus_manifest",
        )
        != manifest_snapshot.content
        or read_regular_bytes_artifact_v2(
            receipt_path,
            label="v2 semantics receipt",
        )
        != receipt_bytes
    ):
        raise RuntimeError("v2 staged inputs changed during structural verification")
    refs = tuple(index[key] for key in sorted(index))
    return verify_priority_mes_corpus_v2(
        refs,
        semantics_receipt=receipt,
        semantics_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        corpus_semantic_sha256=str(receipt.get("corpus_semantic_sha256")),
        repo_root=root,
    )


def mandatory_protocol_files_v2(
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
    *,
    parent_result_root: Path | None = None,
    _parent_lineage: tuple[ParentProtocolSnapshotV2, str] | None = None,
) -> dict[str, Path]:
    """Enumerate and verify every decision input required by protocol v2."""

    root = Path(repo_root).resolve()
    result = canonical_result_root_v2(result_root)
    data = _require_nonsymlink_components_v2(
        data_dir,
        "protocol-v2 data root",
        allow_missing=False,
    )
    parent = (
        _require_nonsymlink_components_v2(
            parent_result_root,
            "parent protocol result root",
            allow_missing=False,
        )
        if parent_result_root is not None
        else root / "results" / "iclr_multicity"
    )
    parent_snapshot, _ = _parent_lineage or _load_parent_lineage_v2(parent)
    parent_lock_path = parent / "protocol_lock.json"
    parent_lock = parent_snapshot.payload
    parent_tracked = parent_lock.get("tracked_files")
    if not isinstance(parent_tracked, dict):
        raise RuntimeError("parent protocol lock tracked files differ")

    manifest_path = result / "corpus_manifest.json"
    receipt_path = result / "approval_semantics_receipt.json"
    structural_path = result / "structural_gates.json"
    manifest_snapshot = _read_json_snapshot_v2(
        manifest_path,
        "artifact/corpus_manifest",
    )
    manifest_bytes = manifest_snapshot.content
    receipt_bytes = read_regular_bytes_artifact_v2(
        receipt_path,
        label="artifact/approval_semantics_receipt",
    )
    assert receipt_bytes is not None
    manifest = manifest_snapshot.payload
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or not isinstance(receipt, dict):
        raise RuntimeError("v2 mandatory manifest and receipt must be JSON objects")
    _validate_redacted_manifest_v2(manifest)
    validate_multicity_semantics_receipt(receipt)
    if receipt.get(
        "parent_manifest_retained_sha256"
    ) != multicity_manifest_retained_sha256(manifest):
        raise RuntimeError("v2 semantics receipt manifest binding differs")

    split_contents: dict[str, bytes] = {}
    for name in SPLIT_NAMES:
        split_path = result / "splits" / f"{name}.json"
        split_content = read_regular_bytes_artifact_v2(
            split_path,
            label=f"split/{name}",
        )
        assert split_content is not None
        split_contents[name] = split_content
        row = parent_tracked.get(f"split/{name}")
        if (
            not isinstance(row, dict)
            or row.get("sha256")
            != hashlib.sha256(split_content).hexdigest()
        ):
            raise RuntimeError(f"v2 split differs from parent protocol: {name}")

    validate_structural_gates_v2(
        structural_path,
        semantics_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        corpus_semantic_sha256=str(receipt.get("corpus_semantic_sha256")),
        repo_root=root,
    )
    index = _index_from_manifest_snapshot_v2(manifest_snapshot, data)
    if (
        read_regular_bytes_artifact_v2(
            manifest_path,
            label="artifact/corpus_manifest",
        )
        != manifest_bytes
        or read_regular_bytes_artifact_v2(
            receipt_path,
            label="artifact/approval_semantics_receipt",
        )
        != receipt_bytes
        or any(
            read_regular_bytes_artifact_v2(
                result / "splits" / f"{name}.json",
                label=f"split/{name}",
            )
            != content
            for name, content in split_contents.items()
        )
    ):
        raise RuntimeError("v2 mandatory inputs changed during verification")

    files: dict[str, Path] = {
        "artifact/corpus_manifest": manifest_path,
        "artifact/approval_semantics_receipt": receipt_path,
        "artifact/structural_gates": structural_path,
        "environment/pyproject": root / "pyproject.toml",
        "environment/uv_lock": root / "uv.lock",
        "parent/protocol_lock": parent_lock_path,
        "parent/heldout_opened": parent / "heldout_opened.json",
    }
    for name in SPLIT_NAMES:
        files[f"split/{name}"] = result / "splits" / f"{name}.json"
    for name in LOCKED_SOURCE_FILES_V2:
        files[f"source/{name}"] = root / "src" / name
    for name in LOCKED_TEST_FILES_V2:
        files[f"test/{name}"] = root / "tests" / name
    for ref in index.values():
        for path in ref.paths:
            files[f"corpus/{path.name}"] = Path(path)
    for label, path in files.items():
        checked = _require_nonsymlink_components_v2(
            path,
            f"v2 mandatory protocol file {label}",
            allow_missing=False,
        )
        if not checked.is_file():
            raise FileNotFoundError(
                f"v2 mandatory protocol file missing for {label}: {path}"
            )
    return files


def build_multicity_protocol_lock_v2(
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
    *,
    parent_result_root: Path | None = None,
) -> dict[str, object]:
    """Build the only protocol lock profile accepted by v2 entry points."""

    root = Path(repo_root).resolve()
    parent = (
        Path(parent_result_root).resolve()
        if parent_result_root is not None
        else root / "results" / "iclr_multicity"
    )
    parent_lineage = _load_parent_lineage_v2(parent)
    files = mandatory_protocol_files_v2(
        root,
        result_root,
        data_dir,
        parent_result_root=parent,
        _parent_lineage=parent_lineage,
    )
    return build_protocol_lock_payload_v2(
        root,
        files,
        parent_result_root=parent,
        _parent_lineage=parent_lineage,
    )


def verify_multicity_protocol_lock_snapshot_v2(
    lock_path: Path,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
    *,
    parent_result_root: Path | None = None,
) -> ProtocolLockSnapshotV2:
    """Verify and return one exact byte image of the protocol-v2 lock."""

    root = Path(repo_root).resolve()
    parent = (
        Path(parent_result_root).resolve()
        if parent_result_root is not None
        else root / "results" / "iclr_multicity"
    )
    parent_lineage = _load_parent_lineage_v2(parent)
    path = _require_nonsymlink_components_v2(
        lock_path,
        "v2 protocol lock",
        allow_missing=False,
    )
    lock_bytes = read_regular_bytes_artifact_v2(
        path,
        label="v2 protocol lock",
    )
    assert lock_bytes is not None
    payload = json.loads(lock_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("v2 protocol lock must be a JSON object")
    expected = mandatory_protocol_files_v2(
        root,
        result_root,
        data_dir,
        parent_result_root=parent,
        _parent_lineage=parent_lineage,
    )
    validate_mandatory_lock_profile_v2(
        payload,
        root,
        expected,
        parent_result_root=parent,
        _parent_lineage=parent_lineage,
    )
    final_bytes = read_regular_bytes_artifact_v2(
        path,
        label="v2 protocol lock",
    )
    if final_bytes != lock_bytes:
        raise RuntimeError("v2 protocol lock changed during verification")
    return ProtocolLockSnapshotV2(
        path=path,
        payload=payload,
        content=lock_bytes,
        sha256=hashlib.sha256(lock_bytes).hexdigest(),
    )


def verify_multicity_protocol_lock_v2(
    lock_path: Path,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
    *,
    parent_result_root: Path | None = None,
) -> dict[str, object]:
    """Compatibility wrapper returning the verified protocol-v2 lock payload."""

    return dict(
        verify_multicity_protocol_lock_snapshot_v2(
            lock_path,
            repo_root,
            result_root,
            data_dir,
            parent_result_root=parent_result_root,
        ).payload
    )


def open_heldout_once_v2(
    lock_path: Path,
    receipt_path: Path,
    required_fit_inventory: Sequence[tuple[Mapping[str, object], Path]],
    structural_gate_path: Path,
    semantics_receipt_path: Path,
    repo_root: Path,
    *,
    authenticated_fit_sha256: Mapping[str, str],
    data_dir: Path | None = None,
    lock_snapshot: ProtocolLockSnapshotV2 | None = None,
) -> dict[str, object]:
    """Create the one-way v2 opening receipt after every locked fit is present."""

    root = Path(repo_root).resolve()
    lock = _require_nonsymlink_components_v2(
        lock_path,
        "v2 protocol lock",
        allow_missing=False,
    )
    result = lock.parent
    receipt_target = _absolute_lexical_path_v2(receipt_path)
    structural = _require_nonsymlink_components_v2(
        structural_gate_path,
        "v2 structural gate",
        allow_missing=False,
    )
    semantics = _require_nonsymlink_components_v2(
        semantics_receipt_path,
        "v2 semantics receipt",
        allow_missing=False,
    )
    if (
        receipt_target != result / "heldout_opened.json"
        or structural != result / "structural_gates.json"
        or semantics != result / "approval_semantics_receipt.json"
    ):
        raise RuntimeError("v2 heldout-opening paths differ from the result root")
    _require_nonsymlink_components_v2(
        receipt_target,
        "v2 heldout-opening receipt",
        allow_missing=True,
    )
    data = (
        Path(data_dir)
        if data_dir is not None
        else root / "data" / "pb_multicity"
    )
    current_lock_snapshot = verify_multicity_protocol_lock_snapshot_v2(
        lock,
        root,
        result,
        data,
    )
    if lock_snapshot is not None and (
        not isinstance(lock_snapshot, ProtocolLockSnapshotV2)
        or lock_snapshot.path != current_lock_snapshot.path
        or lock_snapshot.content != current_lock_snapshot.content
        or lock_snapshot.sha256 != current_lock_snapshot.sha256
        or not exact_json_equal_v2(
            lock_snapshot.payload,
            current_lock_snapshot.payload,
        )
    ):
        raise RuntimeError("v2 protocol lock snapshot changed before held-out opening")
    authenticated_lock = lock_snapshot or current_lock_snapshot
    lock_payload = authenticated_lock.payload
    if (
        type(lock_payload.get("schema_version")) is not int
        or lock_payload.get("schema_version") != 2
        or lock_payload.get("lock_profile") != LOCK_PROFILE_V2
    ):
        raise RuntimeError("v2 heldout opening requires the v2 protocol lock")
    tracked = lock_payload.get("tracked_files")
    if not isinstance(tracked, dict):
        raise RuntimeError("v2 protocol lock tracked files differ")

    semantics_bytes = read_regular_bytes_artifact_v2(
        semantics,
        label="v2 semantics receipt",
    )
    assert semantics_bytes is not None
    semantics_digest = hashlib.sha256(semantics_bytes).hexdigest()
    semantics_row = tracked.get("artifact/approval_semantics_receipt")
    if (
        not isinstance(semantics_row, dict)
        or semantics_row.get("sha256") != semantics_digest
    ):
        raise RuntimeError("v2 semantics receipt differs from the protocol lock")
    semantics_payload = json.loads(semantics_bytes.decode("utf-8"))
    if not isinstance(semantics_payload, dict):
        raise RuntimeError("v2 semantics receipt must be a JSON object")
    validate_multicity_semantics_receipt(semantics_payload)
    if semantics_payload.get("semantics_profile") != SEMANTICS_PROFILE:
        raise RuntimeError("v2 semantics receipt profile differs")
    corpus_semantic_sha256 = _require_sha256(
        semantics_payload.get("corpus_semantic_sha256"),
        "corpus semantic digest",
    )

    structural_bytes = read_regular_bytes_artifact_v2(
        structural,
        label="v2 structural gate",
    )
    assert structural_bytes is not None
    structural_digest = hashlib.sha256(structural_bytes).hexdigest()
    structural_row = tracked.get("artifact/structural_gates")
    if (
        not isinstance(structural_row, dict)
        or structural_row.get("sha256") != structural_digest
    ):
        raise RuntimeError("v2 structural gate differs from the protocol lock")
    validate_structural_gates_v2(
        structural,
        semantics_receipt_sha256=semantics_digest,
        corpus_semantic_sha256=corpus_semantic_sha256,
        repo_root=root,
    )
    if read_regular_bytes_artifact_v2(
        structural,
        label="v2 structural gate",
    ) != structural_bytes:
        raise RuntimeError("v2 structural gate changed during held-out opening")

    def canonical(rows: Sequence[Mapping[str, object]]) -> list[tuple[str, str, int]]:
        canonical_rows = []
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or type(row.get("split")) is not str
                or type(row.get("arm")) is not str
                or type(row.get("seed")) is not int
            ):
                raise RuntimeError("v2 fit inventory has invalid coordinate types")
            canonical_rows.append((row["split"], row["arm"], row["seed"]))
        return sorted(canonical_rows)

    locked_inventory = lock_payload.get("fit_inventory")
    if not isinstance(locked_inventory, list):
        raise RuntimeError("v2 lock fit inventory differs")
    supplied_specs = [spec for spec, _ in required_fit_inventory]
    if canonical(locked_inventory) != canonical(supplied_specs):
        raise RuntimeError("supplied fit inventory differs from the protocol-v2 lock")
    resolved_inventory = sorted(
        (
            (spec, canonical_fit_path_v2(result, spec, path))
            for spec, path in required_fit_inventory
        ),
        key=lambda item: str(item[1]),
    )
    if len({path for _, path in resolved_inventory}) != len(resolved_inventory):
        raise RuntimeError("required v2 fit paths are not one-to-one")
    fit_hashes: dict[str, str] = {}
    for spec, path in resolved_inventory:
        if not path.is_file():
            raise RuntimeError(f"missing required v2 fit: {path}")
        try:
            label = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"required v2 fit lies outside repository: {path}") from exc
        fit_bytes = read_regular_bytes_artifact_v2(
            path,
            label="required protocol-v2 fit",
        )
        assert fit_bytes is not None
        fit_hashes[label] = hashlib.sha256(fit_bytes).hexdigest()
    authenticated_hashes = {
        str(label): _require_sha256(digest, f"authenticated fit {label!r}")
        for label, digest in authenticated_fit_sha256.items()
    }
    if fit_hashes != authenticated_hashes:
        raise RuntimeError(
            "authenticated v2 fit snapshots differ from held-out opening bytes"
        )

    stable = {
        "schema_version": 2,
        "lock_sha256": authenticated_lock.sha256,
        "fit_sha256": fit_hashes,
        "structural_gates_sha256": structural_digest,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_digest,
        "corpus_semantic_sha256": corpus_semantic_sha256,
    }
    existing_bytes = read_regular_bytes_artifact_v2(
        receipt_target,
        label="v2 heldout-opening receipt",
        allow_missing=True,
    )
    if existing_bytes is not None:
        existing = json.loads(existing_bytes.decode("utf-8"))
        if not isinstance(existing, dict) or set(existing) != {
            *stable,
            "opened_at_utc",
        }:
            raise RuntimeError("heldout receipt exists for a different v2 lock or fit inventory")
        if not exact_json_equal_v2(
            {key: existing.get(key) for key in stable},
            stable,
        ):
            raise RuntimeError("heldout receipt exists for a different v2 lock or fit inventory")
        opened_at = existing.get("opened_at_utc")
        try:
            parsed_opened_at = datetime.fromisoformat(opened_at)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("heldout receipt has an invalid opening timestamp") from exc
        if (
            type(opened_at) is not str
            or not opened_at
            or parsed_opened_at.tzinfo is None
            or parsed_opened_at.utcoffset() != timezone.utc.utcoffset(parsed_opened_at)
        ):
            raise RuntimeError("heldout receipt has an invalid opening timestamp")
        return existing
    payload = {
        **stable,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    final_lock_snapshot = verify_multicity_protocol_lock_snapshot_v2(
        lock,
        root,
        result,
        data,
    )
    if (
        final_lock_snapshot.path != authenticated_lock.path
        or final_lock_snapshot.content != authenticated_lock.content
        or final_lock_snapshot.sha256 != authenticated_lock.sha256
        or not exact_json_equal_v2(
            final_lock_snapshot.payload,
            authenticated_lock.payload,
        )
    ):
        raise RuntimeError(
            "v2 protocol locked closure changed before held-out receipt write"
        )
    if read_regular_bytes_artifact_v2(
        semantics,
        label="v2 semantics receipt",
    ) != semantics_bytes:
        raise RuntimeError("v2 semantics receipt changed before held-out receipt write")
    if read_regular_bytes_artifact_v2(
        structural,
        label="v2 structural gate",
    ) != structural_bytes:
        raise RuntimeError("v2 structural gate changed before held-out receipt write")
    for _, path in resolved_inventory:
        label = path.relative_to(root).as_posix()
        fit_bytes = read_regular_bytes_artifact_v2(
            path,
            label="required protocol-v2 fit",
        )
        assert fit_bytes is not None
        if hashlib.sha256(fit_bytes).hexdigest() != fit_hashes[label]:
            raise RuntimeError(
                f"required protocol-v2 fit changed before held-out receipt write: {label}"
            )
    write_immutable_json_artifact_v2(receipt_target, payload)
    return payload
