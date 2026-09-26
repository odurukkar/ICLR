"""The external decisions must stay verifiable while the paper keeps changing.

Every external lock pins the whole source tree, so a whole-tree comparison
fails as soon as a figure script or a macro generator is touched. These checks
assert the property that actually matters: no corpus file, fitted weight,
lineage record, or evaluation module has moved since a lock was published.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

import pytest

import iclr_lock_drift as drift
from iclr_multicity_protocol import LOCKED_SOURCE_FILES


@pytest.mark.parametrize("lock_path", drift.LOCK_PATHS, ids=lambda p: p.parent.name)
def test_no_decision_input_changed_since_the_lock(lock_path: Path) -> None:
    if not lock_path.is_file():
        pytest.skip(f"{lock_path.name} not present")
    report = drift.lock_drift(lock_path)
    critical_missing = [
        label for label in report["missing"] if drift.is_decision_critical(label)
    ]
    assert [
        label for label in critical_missing if not label.startswith("corpus/")
    ] == []
    if any(label.startswith("corpus/") for label in critical_missing):
        pytest.skip(
            "voter-level corpus files are not redistributed with the artifact; "
            "fetch them per REPRODUCE.md before auditing the locks"
        )
    assert report["decision_drifted"] == []
    assert report["intact"], "a lock with no verified files proves nothing"


def test_classification_distinguishes_evaluation_from_manuscript_code() -> None:
    for label in (
        "corpus/Poland_Katowice_2023.pb",
        "fit/endowment/seed-42",
        "split/temporal_2022",
        "artifact/corpus_manifest",
        "original/protocol_lock",
        "environment/pyproject",
        "environment/uv_lock",
        "source/iclr_env.py",
        "source/rules.py",
        "v2_snapshot/source/iclr_external_support_set.py",
        "v2_snapshot/split/city_out_Poland_Gdynia",
        "v2_snapshot/environment/pyproject",
        "v2_snapshot/environment/uv_lock",
    ):
        assert drift.is_decision_critical(label), label
    for label in (
        "source/gen_iclr_numbers.py",
        "source/iclr_fig_hack.py",
        "source/build_iclr_artifact.py",
        "v2_snapshot/source/gen_iclr_appendix.py",
    ):
        assert not drift.is_decision_critical(label), label

    assert {
        f"source/{name}" for name in LOCKED_SOURCE_FILES
    } <= drift.DECISION_SOURCES


def test_warning_only_registry_matches_tracked_sources_omitted_from_release() -> None:
    from build_iclr_artifact import RELEASE_SOURCE

    expected = set()
    for lock_path in drift.LOCK_PATHS:
        tracked = json.loads(lock_path.read_text(encoding="utf-8"))["tracked_files"]
        for label in tracked:
            base_label = label
            if base_label.startswith("v2_snapshot/"):
                base_label = base_label[len("v2_snapshot/") :]
            if (
                base_label.startswith("source/")
                and Path(base_label).name not in RELEASE_SOURCE
                and not drift.is_decision_critical(base_label)
            ):
                expected.add(base_label)

    assert expected
    assert drift.KNOWN_NONCRITICAL_MISSING_LABELS == expected
    assert all(
        not drift.is_decision_critical(label)
        for label in drift.KNOWN_NONCRITICAL_MISSING_LABELS
    )


def test_disclosed_drift_registry_matches_current_nondecision_lock_drift() -> None:
    expected = set()
    for lock_path in drift.LOCK_PATHS:
        report = drift.lock_drift(lock_path)
        for label in report["drifted"]:
            base_label = label.removeprefix("v2_snapshot/")
            if not drift.is_decision_critical(base_label):
                expected.add(base_label)

    assert expected
    assert drift.KNOWN_NONCRITICAL_DRIFT_LABELS == expected
    assert all(
        not drift.is_decision_critical(label)
        for label in drift.KNOWN_NONCRITICAL_DRIFT_LABELS
    )


@pytest.mark.parametrize(
    "label",
    (
        "source/iclr_cmaes.py",
        "source/iclr_corpus.py",
        "source/iclr_multicity_evaluate.py",
        "split/temporal_2022",
        "source/iclr_stats.py",
        "source/iclr_train.py",
        "source/run_experiments.py",
    ),
)
def test_multicity_scientific_source_mutation_is_decision_drift(
    label: str, tmp_path: Path
) -> None:
    tracked = tmp_path / "tracked.py"
    tracked.write_text("locked\n", encoding="utf-8")
    locked = hashlib.sha256(tracked.read_bytes()).hexdigest()
    lock = tmp_path / "protocol_lock.json"
    lock.write_text(
        json.dumps(
            {"tracked_files": {label: {"path": tracked.name, "sha256": locked}}}
        ),
        encoding="utf-8",
    )
    original_root = drift.ROOT
    try:
        drift.ROOT = tmp_path
        tracked.write_text("changed\n", encoding="utf-8")
        assert drift.lock_drift(lock)["decision_drifted"] == [label]
    finally:
        drift.ROOT = original_root


def test_audit_covers_every_lock_on_disk() -> None:
    locks_on_disk = set(drift.RESULTS.glob("**/protocol_lock.json"))
    assert set(drift.LOCK_PATHS) == locks_on_disk
    report = drift.audit()
    assert report, "no external lock was audited"
    for lock, entry in report.items():
        assert set(entry) == {
            "intact",
            "drifted",
            "missing",
            "decision_drifted",
            "amended",
        }
        assert [
            label
            for label in entry["missing"]
            if drift.is_decision_critical(label)
            and not label.startswith("corpus/")
        ] == [], lock
        assert entry["decision_drifted"] == [], lock


@pytest.mark.parametrize("absolute", (False, True))
def test_lock_drift_rejects_paths_outside_repository_root(
    absolute: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    relative = str(outside) if absolute else f"../{outside.name}"
    lock = tmp_path / "protocol_lock.json"
    lock.write_text(
        json.dumps(
            {
                "tracked_files": {
                    "source/gen_iclr_numbers.py": {
                        "path": relative,
                        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(drift, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="tracked path"):
        drift.lock_drift(lock)


def test_assert_audit_passes_rejects_mutated_decision_input_without_globals(
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "src" / "iclr_env.py"
    tracked.parent.mkdir()
    tracked.write_text("locked\n", encoding="utf-8")
    locked = hashlib.sha256(tracked.read_bytes()).hexdigest()
    lock = tmp_path / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "tracked_files": {
                    "source/iclr_env.py": {
                        "path": "src/iclr_env.py",
                        "sha256": locked,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    tracked.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="study is not verifiable"):
        drift.assert_audit_passes(
            (lock,),
            root=tmp_path,
            amendments_path=tmp_path / "results" / "iclr_lock_amendments.json",
        )


@pytest.mark.parametrize("locked_sha256", (None, True, "abc", "g" * 64))
def test_assert_audit_passes_rejects_malformed_locked_digest(
    locked_sha256: object,
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "src" / "iclr_env.py"
    tracked.parent.mkdir()
    tracked.write_text("current\n", encoding="utf-8")
    current_sha256 = hashlib.sha256(tracked.read_bytes()).hexdigest()
    lock = tmp_path / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "tracked_files": {
                    "source/iclr_env.py": {
                        "path": "src/iclr_env.py",
                        "sha256": locked_sha256,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    amendments = tmp_path / "results" / "iclr_lock_amendments.json"
    amendments.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "amendments": [
                    {
                        "path": "src/iclr_env.py",
                        "locked_sha256": locked_sha256,
                        "amended_sha256": current_sha256,
                        "affects_recorded_decision": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"SHA-256|sha256"):
        drift.assert_audit_passes(
            (lock,),
            root=tmp_path,
            amendments_path=amendments,
        )


@pytest.mark.parametrize("missing_count", (1, 2))
def test_audit_fails_closed_when_configured_locks_are_missing(
    missing_count: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting any configured lock must invalidate the audit itself."""
    present = tmp_path / "present" / "protocol_lock.json"
    present.parent.mkdir()
    present.write_text(json.dumps({"tracked_files": {}}), encoding="utf-8")
    missing = tuple(
        tmp_path / f"missing-{index}" / "protocol_lock.json"
        for index in range(missing_count)
    )
    monkeypatch.setattr(drift, "ROOT", tmp_path)

    with pytest.raises(drift.MissingConfiguredLocksError) as exc_info:
        drift.audit((present, *missing))

    assert exc_info.value.paths == missing
    message = str(exc_info.value)
    assert "configured protocol lock" in message
    for path in missing:
        assert path.relative_to(tmp_path).as_posix() in message


def test_cli_exits_nonzero_and_names_all_missing_configured_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = (
        tmp_path / "results" / "study-a" / "protocol_lock.json",
        tmp_path / "results" / "study-b" / "protocol_lock.json",
    )
    monkeypatch.setattr(drift, "ROOT", tmp_path)
    monkeypatch.setattr(drift, "LOCK_PATHS", missing)
    with pytest.raises(SystemExit) as exc_info:
        drift.main()

    assert exc_info.value.code != 0
    message = str(exc_info.value)
    assert "configured protocol locks are missing" in message
    assert "results/study-a/protocol_lock.json" in message
    assert "results/study-b/protocol_lock.json" in message


@pytest.mark.parametrize(
    "label",
    ("source/iclr_toy_model.py", "v2_snapshot/source/iclr_toy_model.py"),
)
def test_cli_discloses_noncritical_missing_tracked_file_without_failing(
    label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lock = tmp_path / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "tracked_files": {
                    label: {"path": "src/iclr_toy_model.py", "sha256": "0" * 64}
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(drift, "ROOT", tmp_path)
    monkeypatch.setattr(drift, "LOCK_PATHS", (lock,))
    monkeypatch.setattr(drift, "AMENDMENTS_PATH", tmp_path / "amendments.json")

    with caplog.at_level("INFO"):
        drift.main()

    assert "missing (non-decision-critical; disclosed)" in caplog.text
    assert label in caplog.text
    assert "every decision-critical locked input" in caplog.text


@pytest.mark.parametrize(
    "label",
    ("source/iclr_policy.py", "corpus/Poland_Warszawa_2023.pb"),
)
def test_cli_rejects_missing_decision_critical_tracked_file(
    label: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    lock = tmp_path / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "tracked_files": {
                    label: {"path": "omitted/critical-input", "sha256": "0" * 64}
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(drift, "ROOT", tmp_path)
    monkeypatch.setattr(drift, "LOCK_PATHS", (lock,))
    monkeypatch.setattr(drift, "AMENDMENTS_PATH", tmp_path / "amendments.json")

    with caplog.at_level("INFO"):
        with pytest.raises(SystemExit, match="changed or is missing"):
            drift.main()

    assert "DECISION-CRITICAL MISSING" in caplog.text
    assert label in caplog.text


def test_cli_rejects_missing_unclassified_tracked_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    label = "source/iclr_future_evaluator.py"
    lock = tmp_path / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "tracked_files": {
                    label: {"path": "src/iclr_future_evaluator.py", "sha256": "0" * 64}
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(drift, "ROOT", tmp_path)
    monkeypatch.setattr(drift, "LOCK_PATHS", (lock,))
    monkeypatch.setattr(drift, "AMENDMENTS_PATH", tmp_path / "amendments.json")

    with caplog.at_level("INFO"):
        with pytest.raises(SystemExit, match="changed or is missing"):
            drift.main()

    assert "UNCLASSIFIED TRACKED LABEL MISSING" in caplog.text
    assert label in caplog.text


def test_cli_rejects_drifted_unclassified_tracked_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    label = "source/iclr_future_evaluator.py"
    tracked = tmp_path / "src" / "iclr_future_evaluator.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("locked\n", encoding="utf-8")
    locked = hashlib.sha256(tracked.read_bytes()).hexdigest()
    lock = tmp_path / "results" / "study" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "tracked_files": {
                    label: {
                        "path": "src/iclr_future_evaluator.py",
                        "sha256": locked,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    tracked.write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(drift, "ROOT", tmp_path)
    monkeypatch.setattr(drift, "LOCK_PATHS", (lock,))
    monkeypatch.setattr(drift, "AMENDMENTS_PATH", tmp_path / "amendments.json")

    with caplog.at_level("INFO"):
        with pytest.raises(SystemExit, match="changed or is missing"):
            drift.main()

    assert "UNCLASSIFIED TRACKED LABEL DRIFTED" in caplog.text
    assert label in caplog.text


def test_amendment_record_converts_drift_into_a_disclosure(tmp_path: Path) -> None:
    """A recorded amendment is disclosed; an unrecorded change still fails."""
    import hashlib

    tracked = tmp_path / "cohorts.py"
    tracked.write_text("locked\n", encoding="utf-8")
    locked = hashlib.sha256(tracked.read_bytes()).hexdigest()
    tracked.write_text("amended\n", encoding="utf-8")
    amended = hashlib.sha256(tracked.read_bytes()).hexdigest()
    relative = tracked.relative_to(tmp_path).as_posix()

    lock = tmp_path / "protocol_lock.json"
    lock.write_text(
        json.dumps(
            {"tracked_files": {"source/cohorts.py": {"path": relative, "sha256": locked}}}
        ),
        encoding="utf-8",
    )
    record = tmp_path / "amendments.json"
    record.write_text(
        json.dumps(
            {
                "amendments": [
                    {
                        "path": relative,
                        "locked_sha256": locked,
                        "amended_sha256": amended,
                        "affects_recorded_decision": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    original_root, original_record = drift.ROOT, drift.AMENDMENTS_PATH
    try:
        drift.ROOT = tmp_path
        drift.AMENDMENTS_PATH = record
        report = drift.lock_drift(lock)
        assert report["decision_drifted"] == []
        assert report["amended"] == ["source/cohorts.py"]

        # A different change to the same file is not covered by the record.
        tracked.write_text("tampered\n", encoding="utf-8")
        assert drift.lock_drift(lock)["decision_drifted"] == ["source/cohorts.py"]

        # An amendment that admits it changes the decision is never accepted.
        tracked.write_text("amended\n", encoding="utf-8")
        payload = json.loads(record.read_text())
        payload["amendments"][0]["affects_recorded_decision"] = True
        record.write_text(json.dumps(payload), encoding="utf-8")
        assert drift.lock_drift(lock)["decision_drifted"] == ["source/cohorts.py"]

        # Missing is not an explicit scientific judgment and must also fail closed.
        del payload["amendments"][0]["affects_recorded_decision"]
        record.write_text(json.dumps(payload), encoding="utf-8")
        assert drift.lock_drift(lock)["decision_drifted"] == ["source/cohorts.py"]
    finally:
        drift.ROOT, drift.AMENDMENTS_PATH = original_root, original_record


def test_amendment_record_must_match_the_original_locked_digest(
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "cohorts.py"
    tracked.write_text("locked\n", encoding="utf-8")
    locked = hashlib.sha256(tracked.read_bytes()).hexdigest()
    tracked.write_text("amended\n", encoding="utf-8")
    amended = hashlib.sha256(tracked.read_bytes()).hexdigest()
    relative = tracked.relative_to(tmp_path).as_posix()
    lock = tmp_path / "protocol_lock.json"
    lock.write_text(
        json.dumps(
            {"tracked_files": {"source/cohorts.py": {"path": relative, "sha256": locked}}}
        ),
        encoding="utf-8",
    )
    record = tmp_path / "amendments.json"
    record.write_text(
        json.dumps(
            {
                "amendments": [
                    {
                        "path": relative,
                        "locked_sha256": "0" * 64,
                        "amended_sha256": amended,
                        "affects_recorded_decision": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    original_root, original_record = drift.ROOT, drift.AMENDMENTS_PATH
    try:
        drift.ROOT = tmp_path
        drift.AMENDMENTS_PATH = record
        report = drift.lock_drift(lock)
        assert report["amended"] == []
        assert report["decision_drifted"] == ["source/cohorts.py"]
    finally:
        drift.ROOT, drift.AMENDMENTS_PATH = original_root, original_record


def test_every_recorded_amendment_carries_its_evidence() -> None:
    """The shipped record must say what changed, why, and what was checked."""
    if not drift.AMENDMENTS_PATH.is_file():
        pytest.skip("no amendments recorded")
    payload = json.loads(drift.AMENDMENTS_PATH.read_text(encoding="utf-8"))
    assert payload["amendments"], "an empty record should be absent, not empty"
    for entry in payload["amendments"]:
        for field in ("path", "locked_sha256", "amended_sha256", "change", "reason"):
            assert entry.get(field), f"{field} missing from an amendment record"
        assert len(entry["verification"]) >= 2
        assert entry["affects_recorded_decision"] is False


def test_seaborn_dependency_amendments_reconstruct_the_locked_environment() -> None:
    """Only the already-used plotting dependency may differ from locked bytes."""
    payload = json.loads(drift.AMENDMENTS_PATH.read_text(encoding="utf-8"))
    by_path = {entry["path"]: entry for entry in payload["amendments"]}
    expected_locked = {
        "pyproject.toml": "31a6b7083e387e33fda0348de8d8f41c7ae96e70264ac0e0aaeb5e628148fa7d",
        "uv.lock": "035c2b3edb9a9387a77ebffa405e8748832dd74af1263ddf76b66be9534ddeb0",
    }
    for relative, locked_sha256 in expected_locked.items():
        entry = by_path[relative]
        current = drift.ROOT / relative
        assert entry["locked_sha256"] == locked_sha256
        assert entry["amended_sha256"] == hashlib.sha256(
            current.read_bytes()
        ).hexdigest()
        assert entry["affects_recorded_decision"] is False

    historical_project = (drift.ROOT / "pyproject.toml").read_text()
    dependency = '    "seaborn>=0.13.2",\n'
    assert historical_project.count(dependency) == 1
    historical_project = historical_project.replace(dependency, "")
    assert hashlib.sha256(historical_project.encode()).hexdigest() == expected_locked[
        "pyproject.toml"
    ]

    historical_lock = (drift.ROOT / "uv.lock").read_text()
    historical_lock, removed = re.subn(
        r'\n\[\[package\]\]\nname = "seaborn"\n.*?(?=\n\[\[package\]\])',
        "",
        historical_lock,
        flags=re.DOTALL,
    )
    assert removed == 1
    for reference in (
        '    { name = "seaborn" },\n',
        '    { name = "seaborn", specifier = ">=0.13.2" },\n',
    ):
        assert historical_lock.count(reference) == 1
        historical_lock = historical_lock.replace(reference, "")
    assert hashlib.sha256(historical_lock.encode()).hexdigest() == expected_locked[
        "uv.lock"
    ]

    seaborn_importers = set()
    for source in (drift.ROOT / "src").glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        if any(
            isinstance(node, ast.Import)
            and any(alias.name == "seaborn" for alias in node.names)
            for node in ast.walk(tree)
        ):
            seaborn_importers.add(source.name)
    assert seaborn_importers
    assert all(
        name == "iclr_style.py"
        or name == "gen_iclr_figures.py"
        or name.startswith("iclr_fig_")
        for name in seaborn_importers
    )


@pytest.mark.parametrize("lock_path", drift.LOCK_PATHS, ids=lambda p: p.parent.name)
def test_locked_environment_remains_critical_and_verifiable(
    lock_path: Path,
) -> None:
    tracked = json.loads(lock_path.read_text(encoding="utf-8"))["tracked_files"]
    expected = {
        label
        for label, row in tracked.items()
        if row["path"] in {"pyproject.toml", "uv.lock"}
    }
    assert len(expected) == 2
    assert all(drift.is_decision_critical(label) for label in expected)
    report = drift.lock_drift(lock_path)
    assert expected <= set(report["intact"]) | set(report["amended"])
    assert expected & set(report["drifted"]) <= set(report["amended"])
    assert expected.isdisjoint(report["decision_drifted"])


def test_drift_is_detected_when_a_tracked_file_changes(tmp_path: Path) -> None:
    """Guard against an audit that passes because it checks nothing."""
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("original\n", encoding="utf-8")
    lock = tmp_path / "protocol_lock.json"
    import hashlib

    digest = hashlib.sha256(tracked.read_bytes()).hexdigest()
    relative = tracked.relative_to(tmp_path).as_posix()
    lock.write_text(
        json.dumps({"tracked_files": {"corpus/x": {"path": relative, "sha256": digest}}}),
        encoding="utf-8",
    )
    original_root = drift.ROOT
    try:
        drift.ROOT = tmp_path
        assert drift.lock_drift(lock)["decision_drifted"] == []
        tracked.write_text("tampered\n", encoding="utf-8")
        assert drift.lock_drift(lock)["decision_drifted"] == ["corpus/x"]
    finally:
        drift.ROOT = original_root


def test_cli_describes_disclosed_amendments_without_calling_them_identical(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        drift,
        "audit",
        lambda: {
            "lock.json": {
                "intact": ["fit/x"],
                "drifted": ["environment/pyproject"],
                "missing": [],
                "decision_drifted": [],
                "amended": ["environment/pyproject"],
            }
        },
    )
    with caplog.at_level("INFO"):
        drift.main()
    assert "byte-identical or covered by a disclosed" in caplog.text
    assert "every decision-critical locked input is byte-identical\n" not in caplog.text
