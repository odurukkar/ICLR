from __future__ import annotations

import importlib
from functools import wraps
import hashlib
import json
from pathlib import Path
import shutil

import pytest

import iclr_external_validation as v1
from iclr_lock_drift import KNOWN_NONCRITICAL_MISSING_LABELS


class HistoricalWholeTreeBytesUnavailable(RuntimeError):
    """The frozen whole-tree lock references bytes that were not retained."""


HISTORICAL_WHOLE_TREE_BYTES_UNAVAILABLE = pytest.mark.xfail(
    strict=True,
    raises=HistoricalWholeTreeBytesUnavailable,
    reason=(
        "the v1 lock pinned a live whole-tree snapshot whose historical source "
        "bytes were not retained; scientific drift is enforced by "
        "test_iclr_lock_drift.py"
    ),
)
HISTORICAL_NONCRITICAL_SOURCE_NAMES = frozenset(
    label.removeprefix("source/")
    for label in KNOWN_NONCRITICAL_MISSING_LABELS
)


def _xfail_historical_whole_tree_bytes_unavailable(*expected_messages: str):
    def decorate(test):
        @wraps(test)
        def guarded(*args, **kwargs):
            try:
                return test(*args, **kwargs)
            except FileNotFoundError as error:
                filename = Path(error.filename).name if error.filename else ""
                if filename not in HISTORICAL_NONCRITICAL_SOURCE_NAMES and not any(
                    name in str(error)
                    for name in HISTORICAL_NONCRITICAL_SOURCE_NAMES
                ):
                    raise
                raise HistoricalWholeTreeBytesUnavailable(str(error)) from error
            except RuntimeError as error:
                if (
                    type(error) is not RuntimeError
                    or str(error) not in expected_messages
                ):
                    raise
                raise HistoricalWholeTreeBytesUnavailable(str(error)) from error

        return HISTORICAL_WHOLE_TREE_BYTES_UNAVAILABLE(guarded)

    return decorate


def test_historical_whole_tree_xfail_rejects_unrelated_runtime_errors() -> None:
    guard = globals().get("_xfail_historical_whole_tree_bytes_unavailable")
    exception_type = globals().get("HistoricalWholeTreeBytesUnavailable")
    assert guard is not None and exception_type is not None

    expected = (
        "v1 protocol tracked file differs for "
        "source/build_iclr_artifact.py"
    )

    @guard(expected)
    def fail_for_an_unrelated_reason() -> None:
        raise RuntimeError("unrelated runtime failure")

    xfail = next(
        mark
        for mark in fail_for_an_unrelated_reason.pytestmark
        if mark.name == "xfail"
    )
    assert xfail.kwargs["strict"] is True
    assert xfail.kwargs["raises"] is exception_type
    with pytest.raises(RuntimeError, match="unrelated runtime failure") as caught:
        fail_for_an_unrelated_reason()
    assert type(caught.value) is RuntimeError

    @guard(expected)
    def fail_for_the_declared_historical_mismatch() -> None:
        raise RuntimeError(expected)

    with pytest.raises(exception_type) as wrapped:
        fail_for_the_declared_historical_mismatch()
    assert str(wrapped.value) == expected
    assert type(wrapped.value.__cause__) is RuntimeError


def _v2():
    try:
        return importlib.import_module("iclr_external_support_set")
    except ModuleNotFoundError:
        pytest.fail("support-set production module is missing")


def _meta_fixture(path: Path, vote_type: str, has_demographics: bool = True) -> None:
    if vote_type == "cumulative":
        vote_columns = "voter_id;vote;points"
    elif has_demographics:
        vote_columns = "voter_id;vote"
    else:
        vote_columns = "voter_id;vote;voting_method"
    if has_demographics:
        vote_columns += ";age;sex"
    path.write_bytes(
        (
            "META\n"
            "key;value\n"
            f"vote_type;{vote_type}\n"
            "PROJECTS\n"
            "project_id;cost\n"
            "VOTES\n"
            f"{vote_columns}\n"
        ).encode("utf-8")
        + b"\xff\xfe ballot bytes must not be read during preparation"
    )


def _full_filename_corpus(root: Path) -> None:
    for city, row in v1.CITY_CONFIG.items():
        vote_type = "cumulative" if city == "Katowice" else "ordinal"
        for year in row["years"]:
            for token in row["tokens"]:
                _meta_fixture(
                    root / f"Poland_{city}_{year}_{token}.pb",
                    vote_type,
                    has_demographics=(city == "Katowice" or year >= 2023),
                )


def test_metadata_screen_stops_before_project_and_ballot_content(tmp_path: Path) -> None:
    v2 = _v2()
    path = tmp_path / "ballot.pb"
    _meta_fixture(path, "cumulative")

    assert v2.read_vote_type_from_meta(path) == "cumulative"


def test_metadata_screen_requires_a_projects_boundary(tmp_path: Path) -> None:
    v2 = _v2()
    path = tmp_path / "unbounded.pb"
    path.write_text(
        "META\nkey;value\nvote_type;cumulative\nVOTES\nvoter_id;vote;points\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="PROJECTS boundary"):
        v2.read_vote_type_from_meta(path)


def test_selection_requires_the_frozen_city_vote_types(tmp_path: Path) -> None:
    v2 = _v2()
    _full_filename_corpus(tmp_path)

    selected, file_meta = v2.select_support_set_files(tmp_path)
    assert len(selected) == 228
    assert {row["vote_type"] for row in file_meta.values()} == {
        "cumulative",
        "ordinal",
    }
    assert sum(row["has_age_sex"] for row in file_meta.values()) == 138
    assert sum(
        row["has_age_sex"]
        for name, row in file_meta.items()
        if "_2023_" in name or "_2024_" in name or "_2025_" in name
    ) == 96

    wrong = tmp_path / "Poland_Katowice_2020_Bogucice.pb"
    _meta_fixture(wrong, "approval")
    with pytest.raises(RuntimeError, match="vote type differs"):
        v2.select_support_set_files(tmp_path)


def test_v2_protocol_changes_only_ballot_semantics_and_lineage() -> None:
    v2 = _v2()
    v1_config = v1.external_protocol_config()
    v2_config = v2.external_protocol_config()

    assert v2_config["study"] == "fresh-city-support-set-confirmation-v2"
    assert v2_config["canonical_paths"]["result_root"] == (
        "results/iclr_external_support_set"
    )
    assert v2_config["ballot_projection"] == {
        "Katowice": "cumulative listed-project support; ignore point magnitude",
        "Krakow": "ordinal listed-project support; ignore list order",
    }
    assert v2_config["demographic_schema"] == {
        "all_scored_elections_have_age_and_sex": True,
        "demographic_warmup_elections": 42,
        "minimum_joint_valid_age_sex_coverage_per_scored_election": 0.5,
        "nondemographic_warmup_elections": 90,
    }
    assert v2_config["evidence_gates"] == v1_config["evidence_gates"]
    assert v2_config["statistics"] == v1_config["statistics"]
    assert v2_config["policies"] == v1_config["policies"]
    assert v2_config["expected_counts"] == v1_config["expected_counts"]


def test_v2_anchor_names_the_v2_study(tmp_path: Path) -> None:
    v2 = _v2()
    lock = tmp_path / "protocol_lock.json"
    lock.write_text("{}\n", encoding="utf-8")
    digest = v1._sha256(lock)
    anchor = tmp_path / "published_lock_anchor.json"

    v2.write_published_anchor(anchor, digest, f"confirm {digest}")
    payload = json.loads(anchor.read_text(encoding="utf-8"))
    assert payload["study"] == "fresh-city-support-set-confirmation-v2"
    v2.verify_published_anchor(anchor, lock)


def test_v2_tracked_files_bind_the_aborted_v1_lineage() -> None:
    v2 = _v2()
    tracked = v2.v1_lineage_files()

    assert tracked["v1/protocol_lock"].name == "protocol_lock.json"
    assert tracked["v1/published_anchor"].name == "published_lock_anchor.json"
    assert tracked["v1/heldout_receipt"].name == "heldout_opened.json"
    assert tracked["v1/abort_record"].name == "abort_record.json"
    assert tracked["v1/corpus_manifest"].name == "corpus_manifest.json"
    assert all(path.is_file() for path in tracked.values())


@pytest.mark.parametrize(
    ("vote_type", "header", "row"),
    (
        ("ordinal", "voter_id;vote;age;sex", "1;p2,p1;30;F"),
        ("cumulative", "voter_id;vote;points", "1;p2,p1;2,1"),
    ),
)
def test_projection_validation_accepts_listed_support_and_ignores_magnitude(
    tmp_path: Path, vote_type: str, header: str, row: str
) -> None:
    v2 = _v2()
    path = tmp_path / f"{vote_type}.pb"
    path.write_text(
        "META\nkey;value\n"
        f"vote_type;{vote_type}\n"
        "PROJECTS\nproject_id;cost\np1;1\np2;1\n"
        "VOTES\n"
        f"{header}\n{row}\n",
        encoding="utf-8",
    )

    assert v2.validate_support_set_rows(path, vote_type) == 1


def test_cumulative_projection_rejects_nonpositive_listed_points(tmp_path: Path) -> None:
    v2 = _v2()
    path = tmp_path / "bad.pb"
    path.write_text(
        "META\nkey;value\nvote_type;cumulative\n"
        "PROJECTS\nproject_id;cost\np1;1\n"
        "VOTES\nvoter_id;vote;points\n1;p1;0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="positive"):
        v2.validate_support_set_rows(path, "cumulative")


def test_projection_rejects_duplicate_project_ids(tmp_path: Path) -> None:
    v2 = _v2()
    path = tmp_path / "duplicate.pb"
    path.write_text(
        "META\nkey;value\nvote_type;cumulative\n"
        "PROJECTS\nproject_id;cost\np1;1\n"
        "VOTES\nvoter_id;vote;points\n1;p1,p1;1,1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate project"):
        v2.validate_support_set_rows(path, "cumulative")


def test_verified_ballot_bytes_are_the_bytes_parsed(tmp_path: Path) -> None:
    v2 = _v2()
    path = tmp_path / "locked.pb"
    original = (
        "META\nkey;value\ncountry;Poland\nunit;Katowice\nsubunit;Bogucice\n"
        "date_begin;2020\nbudget;10\nvote_type;cumulative\n"
        "PROJECTS\nproject_id;cost\np1;4\np2;6\n"
        "VOTES\nvoter_id;vote;points;age;sex\n1;p1,p2;2,1;30;F\n"
    ).encode("utf-8")
    replacement = original.replace(b"p1,p2", b"p2,p2")
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()

    locked = v2.read_locked_bytes(path, digest)
    path.write_bytes(replacement)
    instance = v2.parse_support_set_bytes(locked, path, "cumulative")

    assert instance.votes[0].projects == ("p1", "p2")


def test_verified_fit_json_is_consumed_from_locked_bytes(tmp_path: Path) -> None:
    v2 = _v2()
    path = tmp_path / "fit.json"
    original = b'{"config":{"arm":"endowment"},"result":{"best_weights":[0,0,0,0,0,0]}}\n'
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()

    payload = v2.read_locked_json(path, digest)
    path.write_text('{"config":{"arm":"priority"}}\n', encoding="utf-8")

    assert payload["config"]["arm"] == "endowment"
    assert payload["result"]["best_weights"] == [0, 0, 0, 0, 0, 0]


def test_schema_metadata_is_derived_from_authenticated_bytes(tmp_path: Path) -> None:
    v2 = _v2()
    path = tmp_path / "Poland_Katowice_2023_Bogucice.pb"
    _meta_fixture(path, "approval", has_demographics=False)
    selected = [v1.SelectedFile(path, "Katowice", "Bogucice", 2023)]
    authenticated = _meta_fixture_bytes("cumulative", has_demographics=True)

    metadata = v2.support_set_metadata_from_bytes(
        selected, {path.name: authenticated}
    )

    assert metadata[path.name]["vote_type"] == "cumulative"
    assert metadata[path.name]["has_age_sex"] is True


def _meta_fixture_bytes(vote_type: str, has_demographics: bool) -> bytes:
    if vote_type == "cumulative":
        columns = "voter_id;vote;points"
    elif has_demographics:
        columns = "voter_id;vote"
    else:
        columns = "voter_id;vote;voting_method"
    if has_demographics:
        columns += ";age;sex"
    return (
        "META\nkey;value\n"
        f"vote_type;{vote_type}\n"
        "PROJECTS\nproject_id;cost\n"
        "VOTES\n"
        f"{columns}\n"
    ).encode("utf-8")


def test_demographic_coverage_gate_uses_joint_valid_cohorts() -> None:
    v2 = _v2()
    instance = v2.PBInstance(
        path="memory://coverage",
        votes=[
            v2.Vote("1", ("p1",), age=30, sex="F"),
            v2.Vote("2", ("p1",), age=30, sex=""),
            v2.Vote("3", ("p1",), age=None, sex="M"),
            v2.Vote("4", ("p1",), age=65, sex="M"),
        ],
    )

    assert v2.joint_age_sex_coverage(instance) == 0.5
    v2.require_demographic_coverage(
        {"Poland/Katowice/Bogucice": {2023: instance}},
        {"Poland/Katowice/Bogucice": (2023,)},
    )

    instance.votes.pop()
    with pytest.raises(RuntimeError, match="demographic coverage"):
        v2.require_demographic_coverage(
            {"Poland/Katowice/Bogucice": {2023: instance}},
            {"Poland/Katowice/Bogucice": (2023,)},
        )


@_xfail_historical_whole_tree_bytes_unavailable(
    "v1 protocol tracked file differs for environment/pyproject",
    "v1 protocol tracked file differs for source/build_iclr_artifact.py",
)
def test_v1_abort_semantics_are_verified_not_only_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v2 = _v2()
    root = tmp_path / "v1"
    root.mkdir()
    for name in (
        "protocol_lock.json",
        "published_lock_anchor.json",
        "heldout_opened.json",
        "abort_record.json",
        "corpus_manifest.json",
    ):
        shutil.copy2(v2.V1_RESULT_ROOT / name, root / name)
    monkeypatch.setattr(v2, "V1_RESULT_ROOT", root)

    v2.verify_v1_abort()
    abort = json.loads((root / "abort_record.json").read_text(encoding="utf-8"))
    abort["metrics_computed"] = True
    (root / "abort_record.json").write_text(
        json.dumps(abort, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="abort record"):
        v2.verify_v1_abort()


def test_v1_receipt_requires_the_exact_manifest_and_fit_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v2 = _v2()
    root = tmp_path / "v1"
    root.mkdir()
    for name in (
        "protocol_lock.json",
        "published_lock_anchor.json",
        "heldout_opened.json",
        "abort_record.json",
        "corpus_manifest.json",
    ):
        shutil.copy2(v2.V1_RESULT_ROOT / name, root / name)
    monkeypatch.setattr(v2, "V1_RESULT_ROOT", root)
    lock_sha = hashlib.sha256((root / "protocol_lock.json").read_bytes()).hexdigest()
    (root / "heldout_opened.json").write_text(
        json.dumps({"protocol_lock_sha256": lock_sha}) + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="v1 held-out receipt"):
        v2.verify_v1_abort()


def test_v1_protocol_lock_semantics_are_fully_verified() -> None:
    v2 = _v2()
    lock_path = v2.V1_RESULT_ROOT / "protocol_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["protocol"]["study"] = "mutated"
    mutated = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")

    with pytest.raises(RuntimeError, match="v1 protocol"):
        v2.verify_v1_protocol_lock_bytes(mutated)


def test_v1_abort_rejects_late_evidence_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v2 = _v2()
    root = tmp_path / "v1"
    root.mkdir()
    for name in (
        "protocol_lock.json",
        "published_lock_anchor.json",
        "heldout_opened.json",
        "abort_record.json",
        "corpus_manifest.json",
    ):
        shutil.copy2(v2.V1_RESULT_ROOT / name, root / name)
    monkeypatch.setattr(v2, "V1_RESULT_ROOT", root)
    (root / "evidence_decision.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="v1 evidence output"):
        v2.verify_v1_abort()


def test_v2_lock_writer_does_not_mutate_v1_globals(tmp_path: Path) -> None:
    v2 = _v2()
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    lock = tmp_path / "lock.json"
    before = (
        v1.RESULT_ROOT,
        v1.external_protocol_config,
        v1.external_tracked_files,
        v1.load_external_corpus,
    )

    v2.write_external_lock(lock, tmp_path, {"artifact": tracked})

    after = (
        v1.RESULT_ROOT,
        v1.external_protocol_config,
        v1.external_tracked_files,
        v1.load_external_corpus,
    )
    assert after == before


@_xfail_historical_whole_tree_bytes_unavailable(
    "v1 protocol snapshot differs for environment/pyproject",
    "v1 protocol snapshot differs for source/build_iclr_artifact.py",
)
def test_v1_snapshot_verifier_requires_and_resolves_every_old_tracked_path() -> None:
    v2 = _v2()
    v1_lock_bytes = (v2.V1_RESULT_ROOT / "protocol_lock.json").read_bytes()
    v1_lock = json.loads(v1_lock_bytes.decode("utf-8"))
    tracked_paths = {
        label: v2.ROOT / row["path"]
        for label, row in v1_lock["tracked_files"].items()
    }
    tracked_paths.update(v2.v1_lineage_files())
    payload = {
        "tracked_files": {
            label: {
                "path": str(path.resolve().relative_to(v2.ROOT.resolve())),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for label, path in tracked_paths.items()
        }
    }
    locked = {label: path.read_bytes() for label, path in tracked_paths.items()}

    v2._verify_v1_against_locked_snapshot(payload, locked)

    locked.pop("v1/corpus_manifest")
    with pytest.raises(RuntimeError, match="v1 lineage labels"):
        v2._verify_v1_against_locked_snapshot(payload, locked)
