"""Release-data staging must preserve the bytes pinned by every protocol lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import stage_iclr_release_data as stage


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_lock(
    repo: Path,
    relative_path: str,
    tracked_files: dict[str, dict[str, str]],
) -> Path:
    lock = repo / relative_path
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps({"schema_version": 1, "tracked_files": tracked_files}),
        encoding="utf-8",
    )
    return lock


def test_stage_copies_missing_targets_deduplicates_locks_and_retains_existing(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    alpha = b"alpha locked bytes\n"
    beta = b"beta locked bytes\n"
    (source / "alpha.pb").write_bytes(alpha)
    (source / "beta.pb").write_bytes(beta)

    retained = repo / "data" / "pb_external_validation" / "beta.pb"
    retained.parent.mkdir(parents=True)
    retained.write_bytes(beta)
    retained_stat = retained.stat()

    first = _write_lock(
        repo,
        "results/study-a/protocol_lock.json",
        {
            "source/ignored.py": {
                "path": "src/ignored.py",
                "sha256": "0" * 64,
            },
            "corpus/alpha.pb": {
                "path": "data/pb_multicity/alpha.pb",
                "sha256": _sha256(alpha),
            },
            "corpus/beta.pb": {
                "path": "data/pb_external_validation/beta.pb",
                "sha256": _sha256(beta),
            },
        },
    )
    second = _write_lock(
        repo,
        "results/study-b/protocol_lock.json",
        {
            "v2_snapshot/corpus/alpha.pb": {
                "path": "data/pb_multicity/alpha.pb",
                "sha256": _sha256(alpha),
            }
        },
    )

    report = stage.stage_release_data(
        repo_root=repo,
        source_dir=Path("data/pb"),
        lock_paths=(first, second),
    )

    assert (repo / "data" / "pb_multicity" / "alpha.pb").read_bytes() == alpha
    assert retained.read_bytes() == beta
    assert retained.stat().st_ino == retained_stat.st_ino
    assert retained.stat().st_mtime_ns == retained_stat.st_mtime_ns
    assert report == stage.StageReport(
        lock_count=2,
        reference_count=3,
        unique_source_count=2,
        target_count=2,
        copied_count=1,
        retained_count=1,
    )


def test_missing_source_fails_before_any_target_is_written(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    good = b"available bytes\n"
    (source / "available.pb").write_bytes(good)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/available.pb": {
                "path": "data/pb_multicity/available.pb",
                "sha256": _sha256(good),
            },
            "corpus/missing.pb": {
                "path": "data/pb_multicity/missing.pb",
                "sha256": _sha256(b"missing bytes\n"),
            },
        },
    )

    with pytest.raises(RuntimeError, match=r"missing locked source.*missing\.pb"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert not (repo / "data" / "pb_multicity" / "available.pb").exists()


@pytest.mark.parametrize(
    "target_path",
    (
        "outside.pb",
        "results/escaped.pb",
        "data/../escaped.pb",
        "data/pb_multicity/../../escaped.pb",
    ),
)
def test_lock_target_must_stay_beneath_repository_data(
    tmp_path: Path, target_path: str
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    locked = b"locked bytes\n"
    (source / "locked.pb").write_bytes(locked)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/locked.pb": {
                "path": target_path,
                "sha256": _sha256(locked),
            }
        },
    )

    with pytest.raises(RuntimeError, match="outside repository data directory"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )


def test_corpus_label_cannot_traverse_outside_verified_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    secret = b"not a corpus file\n"
    (repo / "data" / "secret.pb").write_bytes(secret)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/../secret.pb": {
                "path": "data/pb_multicity/secret.pb",
                "sha256": _sha256(secret),
            }
        },
    )

    with pytest.raises(RuntimeError, match="unsafe corpus label"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert not (repo / "data" / "pb_multicity" / "secret.pb").exists()


def test_same_target_with_inconsistent_lock_hashes_fails_before_source_reads(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "pb").mkdir(parents=True)
    first = _write_lock(
        repo,
        "results/study-a/protocol_lock.json",
        {
            "corpus/shared.pb": {
                "path": "data/pb_multicity/shared.pb",
                "sha256": "1" * 64,
            }
        },
    )
    second = _write_lock(
        repo,
        "results/study-b/protocol_lock.json",
        {
            "v2_snapshot/corpus/shared.pb": {
                "path": "data/pb_multicity/shared.pb",
                "sha256": "2" * 64,
            }
        },
    )

    with pytest.raises(
        RuntimeError, match=r"inconsistent hashes.*data/pb_multicity/shared\.pb"
    ):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(first, second),
        )


def test_verified_source_directory_must_stay_beneath_repository_data(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    locked = b"locked bytes\n"
    (outside / "locked.pb").write_bytes(locked)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/locked.pb": {
                "path": "data/pb_multicity/locked.pb",
                "sha256": _sha256(locked),
            }
        },
    )

    with pytest.raises(
        RuntimeError, match="source directory is outside repository data directory"
    ):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=outside,
            lock_paths=(lock,),
        )


def test_same_source_with_inconsistent_hashes_across_targets_is_rejected(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    shared = b"one source file\n"
    (source / "shared.pb").write_bytes(shared)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/shared.pb": {
                "path": "data/pb_multicity/shared.pb",
                "sha256": _sha256(shared),
            },
            "v2_snapshot/corpus/shared.pb": {
                "path": "data/pb_external_validation/shared.pb",
                "sha256": "3" * 64,
            },
        },
    )

    with pytest.raises(RuntimeError, match=r"inconsistent hashes.*source shared\.pb"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert not (repo / "data" / "pb_multicity" / "shared.pb").exists()


def test_every_configured_protocol_lock_must_exist(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "pb").mkdir(parents=True)
    missing = (
        Path("results/study-a/protocol_lock.json"),
        Path("results/study-b/protocol_lock.json"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=missing,
        )

    message = str(exc_info.value)
    assert "configured protocol locks are missing" in message
    assert "results/study-a/protocol_lock.json" in message
    assert "results/study-b/protocol_lock.json" in message


def test_cli_stages_relative_paths_and_prints_concise_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    locked = b"release bytes\n"
    (source / "release.pb").write_bytes(locked)
    _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/release.pb": {
                "path": "data/pb_multicity/release.pb",
                "sha256": _sha256(locked),
            }
        },
    )

    status = stage.main(
        [
            "--repo-root",
            str(repo),
            "--source-dir",
            "data/pb",
            "--lock",
            "results/study/protocol_lock.json",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == (
        "ICLR RELEASE DATA STAGED: locks=1 references=1 unique_sources=1 "
        "unique_targets=1 copied=1 retained=0\n"
    )


def test_empty_protocol_lock_configuration_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "pb").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="no protocol locks configured"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(),
        )


def test_locked_source_must_be_a_regular_file_inside_source_directory(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    outside = tmp_path / "outside.pb"
    locked = b"outside bytes\n"
    outside.write_bytes(locked)
    (source / "linked.pb").symlink_to(outside)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/linked.pb": {
                "path": "data/pb_multicity/linked.pb",
                "sha256": _sha256(locked),
            }
        },
    )

    with pytest.raises(RuntimeError, match="unsafe locked source"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )


def test_existing_target_symlink_is_not_treated_as_a_retained_file(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    locked = b"locked bytes\n"
    (source / "locked.pb").write_bytes(locked)
    target = repo / "data" / "pb_multicity" / "locked.pb"
    target.parent.mkdir(parents=True)
    target.symlink_to(source / "locked.pb")
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/locked.pb": {
                "path": "data/pb_multicity/locked.pb",
                "sha256": _sha256(locked),
            }
        },
    )

    with pytest.raises(RuntimeError, match="unsafe locked target"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )


def test_source_hash_mismatch_never_reaches_the_target(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    (source / "mutated.pb").write_bytes(b"mutated bytes\n")
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/mutated.pb": {
                "path": "data/pb_multicity/mutated.pb",
                "sha256": _sha256(b"locked bytes\n"),
            }
        },
    )

    with pytest.raises(RuntimeError, match="source hash mismatch"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert not (repo / "data" / "pb_multicity" / "mutated.pb").exists()


def test_conflicting_existing_target_bytes_abort_before_other_copies(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    good = b"good bytes\n"
    locked = b"locked bytes\n"
    (source / "good.pb").write_bytes(good)
    (source / "locked.pb").write_bytes(locked)
    conflicting = repo / "data" / "pb_multicity" / "locked.pb"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"conflicting bytes\n")
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/good.pb": {
                "path": "data/pb_multicity/good.pb",
                "sha256": _sha256(good),
            },
            "corpus/locked.pb": {
                "path": "data/pb_multicity/locked.pb",
                "sha256": _sha256(locked),
            },
        },
    )

    with pytest.raises(RuntimeError, match="conflicting existing target bytes"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert conflicting.read_bytes() == b"conflicting bytes\n"
    assert not (repo / "data" / "pb_multicity" / "good.pb").exists()


def test_non_file_target_aborts_before_other_copies(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    good = b"good bytes\n"
    blocked = b"blocked bytes\n"
    (source / "good.pb").write_bytes(good)
    (source / "blocked.pb").write_bytes(blocked)
    blocked_target = repo / "data" / "pb_multicity" / "blocked.pb"
    blocked_target.mkdir(parents=True)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/good.pb": {
                "path": "data/pb_multicity/good.pb",
                "sha256": _sha256(good),
            },
            "corpus/blocked.pb": {
                "path": "data/pb_multicity/blocked.pb",
                "sha256": _sha256(blocked),
            },
        },
    )

    with pytest.raises(RuntimeError, match="locked target is not a regular file"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert not (repo / "data" / "pb_multicity" / "good.pb").exists()


@pytest.mark.parametrize(
    ("lock_text", "message"),
    (
        ("{not-json", "malformed protocol lock"),
        (json.dumps({"schema_version": 1}), "tracked_files must be a mapping"),
        (
            json.dumps({"schema_version": 99, "tracked_files": {}}),
            "unsupported protocol lock schema",
        ),
    ),
)
def test_malformed_protocol_lock_fails_with_controlled_error(
    tmp_path: Path, lock_text: str, message: str
) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "pb").mkdir(parents=True)
    lock = repo / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(lock_text, encoding="utf-8")

    with pytest.raises(stage.StageDataError, match=message):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )


def test_invalid_locked_digest_is_rejected_before_file_access(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "pb").mkdir(parents=True)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/file.pb": {
                "path": "data/pb_multicity/file.pb",
                "sha256": "not-a-sha256",
            }
        },
    )

    with pytest.raises(stage.StageDataError, match="invalid SHA-256"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )


@pytest.mark.parametrize("schema_version", (1, 2, 3))
def test_configured_protocol_lock_schema_versions_are_supported(
    tmp_path: Path, schema_version: int
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    locked = f"schema {schema_version}\n".encode()
    (source / "file.pb").write_bytes(locked)
    lock = repo / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "tracked_files": {
                    "corpus/file.pb": {
                        "path": "data/pb_multicity/file.pb",
                        "sha256": _sha256(locked),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = stage.stage_release_data(
        repo_root=repo,
        source_dir=Path("data/pb"),
        lock_paths=(lock,),
    )

    assert report.copied_count == 1


def test_lock_without_corpus_entries_cannot_report_staging_success(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "data" / "pb").mkdir(parents=True)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "source/ignored.py": {
                "path": "src/ignored.py",
                "sha256": "0" * 64,
            }
        },
    )

    with pytest.raises(stage.StageDataError, match="tracks no corpus files"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )


def test_verified_source_path_must_be_a_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not a directory\n")
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/file.pb": {
                "path": "data/pb_multicity/file.pb",
                "sha256": "4" * 64,
            }
        },
    )

    with pytest.raises(stage.StageDataError, match="source path is not a directory"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )


def test_non_directory_target_ancestor_aborts_before_any_copy(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    good = b"good bytes\n"
    blocked = b"blocked bytes\n"
    (source / "good.pb").write_bytes(good)
    (source / "blocked.pb").write_bytes(blocked)
    blocker = repo / "data" / "pb_multicity" / "not-a-directory"
    blocker.parent.mkdir(parents=True)
    blocker.write_bytes(b"blocking parent\n")
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/good.pb": {
                "path": "data/pb_multicity/good.pb",
                "sha256": _sha256(good),
            },
            "corpus/blocked.pb": {
                "path": "data/pb_multicity/not-a-directory/blocked.pb",
                "sha256": _sha256(blocked),
            },
        },
    )

    with pytest.raises(stage.StageDataError, match="target ancestor is not a directory"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert not (repo / "data" / "pb_multicity" / "good.pb").exists()
    assert blocker.read_bytes() == b"blocking parent\n"


@pytest.mark.parametrize("unsafe_relative", (r"C:\secret.pb", "C:/secret.pb"))
def test_windows_drive_and_backslash_corpus_labels_are_rejected(
    tmp_path: Path, unsafe_relative: str
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    locked = b"outside corpus namespace\n"
    unsafe_source = source.joinpath(*unsafe_relative.split("/"))
    unsafe_source.parent.mkdir(parents=True, exist_ok=True)
    unsafe_source.write_bytes(locked)
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            f"corpus/{unsafe_relative}": {
                "path": "data/pb_multicity/secret.pb",
                "sha256": _sha256(locked),
            }
        },
    )

    with pytest.raises(stage.StageDataError, match="unsafe corpus label"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert not (repo / "data" / "pb_multicity" / "secret.pb").exists()


def test_late_copy_failure_rolls_back_only_targets_created_by_this_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "pb"
    source.mkdir(parents=True)
    first = b"first bytes\n"
    second = b"second bytes\n"
    retained_bytes = b"retained bytes\n"
    (source / "first.pb").write_bytes(first)
    (source / "second.pb").write_bytes(second)
    (source / "retained.pb").write_bytes(retained_bytes)
    target_dir = repo / "data" / "pb_multicity"
    target_dir.mkdir(parents=True)
    retained = target_dir / "retained.pb"
    retained.write_bytes(retained_bytes)
    retained_stat = retained.stat()
    lock = _write_lock(
        repo,
        "results/study/protocol_lock.json",
        {
            "corpus/first.pb": {
                "path": "data/pb_multicity/first.pb",
                "sha256": _sha256(first),
            },
            "corpus/second.pb": {
                "path": "data/pb_multicity/second.pb",
                "sha256": _sha256(second),
            },
            "corpus/retained.pb": {
                "path": "data/pb_multicity/retained.pb",
                "sha256": _sha256(retained_bytes),
            },
        },
    )
    real_copy = stage._copy_verified
    calls = 0

    def fail_second_copy(
        source_path: Path, target: Path, expected: str
    ) -> stage.FileIdentity | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise stage.StageDataError("injected second-copy failure")
        return real_copy(source_path, target, expected)

    monkeypatch.setattr(stage, "_copy_verified", fail_second_copy)

    with pytest.raises(stage.StageDataError, match="injected second-copy failure"):
        stage.stage_release_data(
            repo_root=repo,
            source_dir=Path("data/pb"),
            lock_paths=(lock,),
        )

    assert calls == 2
    assert not (target_dir / "first.pb").exists()
    assert not (target_dir / "second.pb").exists()
    assert retained.read_bytes() == retained_bytes
    assert retained.stat().st_ino == retained_stat.st_ino
    assert retained.stat().st_mtime_ns == retained_stat.st_mtime_ns
    assert target_dir.is_dir()
