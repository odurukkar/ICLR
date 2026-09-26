"""Protocol-v2 approval-set normalization and provenance contracts."""

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path

import iclr_priority_mes
import pytest
from iclr_corpus import SeriesRef
from iclr_env import RolloutState
from parse_pb import PBInstance, Project, Vote, parse_pb_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ordered_set_canonicalization_preserves_first_occurrence_and_input() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    original = PBInstance(
        path="corpus/Wola.pb",
        meta={"budget": "10", "vote_type": "approval"},
        projects={
            "147": Project("147", 4.0, 0, name="first"),
            "1040": Project("1040", 6.0, 1, name="second"),
        },
        votes=[
            Vote(
                "37051",
                ("147", "1040", "1040", "147"),
                age=27,
                sex="M",
                neighborhood="internet",
            )
        ],
    )

    result = semantics.canonicalize_instance(original)

    assert result.instance is not original
    assert result.instance.meta == original.meta
    assert result.instance.meta is not original.meta
    assert result.instance.projects == original.projects
    assert result.instance.projects is not original.projects
    assert result.instance.votes[0] == Vote(
        "37051",
        ("147", "1040"),
        age=27,
        sex="M",
        neighborhood="internet",
    )
    assert original.votes[0].projects == ("147", "1040", "1040", "147")


def test_set_canonicalization_rejects_nonapproval_ballots() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    instance = PBInstance(
        path="ordinal.pb",
        meta={"vote_type": "ordinal"},
        projects={"a": Project("a", 1.0, None)},
        votes=[Vote("v", ("a", "a"))],
    )

    with pytest.raises(ValueError, match="approval ballots"):
        semantics.canonicalize_instance(instance)


def test_v2_instance_cannot_reuse_a_v1_supporter_cache_entry() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    raw = PBInstance(
        path="same-source.pb",
        meta={"budget": "1", "vote_type": "approval"},
        projects={"a": Project("a", 0.75, None)},
        votes=[Vote("voter", ("a", "a"))],
    )
    normalized = semantics.canonicalize_instance(raw).instance
    iclr_priority_mes._SUPPORTER_CACHE.clear()
    try:
        iclr_priority_mes.priority_mes_outcome(
            raw,
            RolloutState(),
            [0.0] * 5,
            completion=False,
        )
        after_v1 = iclr_priority_mes.priority_mes_outcome(
            normalized,
            RolloutState(),
            [0.0] * 5,
            completion=False,
        )
        iclr_priority_mes._SUPPORTER_CACHE.clear()
        clean_v2 = iclr_priority_mes.priority_mes_outcome(
            normalized,
            RolloutState(),
            [0.0] * 5,
            completion=False,
        )

        assert after_v1 == clean_v2
    finally:
        iclr_priority_mes._SUPPORTER_CACHE.clear()


def test_duplicate_receipt_records_each_removed_token_and_source_identity() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    instance = PBInstance(
        path="/machine-specific/staging/Wola.pb",
        meta={"budget": "10", "vote_type": "approval"},
        projects={},
        votes=[Vote("37051", ("147", "1040", "1040", "147"))],
    )

    result = semantics.canonicalize_instance(
        instance,
        source_id="Poland_Warszawa_2022_Wola.pb",
    )

    assert result.anomalies == (
        semantics.DuplicateApprovalAnomaly(
            source_id="Poland_Warszawa_2022_Wola.pb",
            vote_index_zero_based=0,
            voter_id="37051",
            original_projects=("147", "1040", "1040", "147"),
            canonical_projects=("147", "1040"),
            duplicate_projects=("1040", "147"),
        ),
    )


def test_v2_parser_normalizes_a_real_pb_file_without_changing_raw_bytes(
    tmp_path: Path,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    path = tmp_path / "duplicate.pb"
    path.write_text(
        "META\n"
        "key;value\n"
        "budget;10\n"
        "vote_type;approval\n"
        "PROJECTS\n"
        "project_id;cost;selected\n"
        "147;4;0\n"
        "1040;6;1\n"
        "VOTES\n"
        "voter_id;vote;age;sex;neighborhood\n"
        "37051;147,1040,1040;27;M;internet\n",
        encoding="utf-8",
    )
    before = path.read_bytes()
    before_sha256 = hashlib.sha256(before).hexdigest()

    instance = semantics.parse_pb_file_v2(path)

    assert instance.votes[0].projects == ("147", "1040")
    assert path.read_bytes() == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_sha256


def test_v2_series_loader_restricts_years_and_normalizes_every_loaded_ballot(
    tmp_path: Path,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    paths = []
    for year, vote in ((2021, "a,a,b"), (2022, "b,b")):
        path = tmp_path / f"election-{year}.pb"
        path.write_text(
            "META\n"
            "key;value\n"
            f"date_begin;{year}-01-01\n"
            "budget;2\n"
            "vote_type;approval\n"
            "PROJECTS\n"
            "project_id;cost;selected\n"
            "a;1;1\n"
            "b;1;0\n"
            "VOTES\n"
            "voter_id;vote\n"
            f"voter-{year};{vote}\n",
            encoding="utf-8",
        )
        paths.append(path)
    ref = SeriesRef(
        key="Poland/Warszawa/Wola",
        years=(2021, 2022),
        paths=tuple(paths),
    )

    loaded = semantics.load_series_v2(ref, years=(2021,))

    assert tuple(loaded) == (2021,)
    assert loaded[2021].votes[0].projects == ("a", "b")


def _authenticated_series_fixture(
    tmp_path: Path,
) -> tuple[SeriesRef, dict[str, object], Path]:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    path = tmp_path / "election-2021.pb"
    path.write_text(
        "META\n"
        "key;value\n"
        "date_begin;2021-01-01\n"
        "budget;2\n"
        "vote_type;approval\n"
        "PROJECTS\n"
        "project_id;cost;selected\n"
        "a;1;1\n"
        "b;1;0\n"
        "VOTES\n"
        "voter_id;vote\n"
        "voter-2021;a,a,b\n",
        encoding="utf-8",
    )
    ref = SeriesRef(
        key="Poland/Warszawa/Wola",
        years=(2021,),
        paths=(path,),
    )
    raw = parse_pb_file(path)
    normalized = semantics.canonicalize_instance(
        raw,
        source_id=path.name,
    ).instance
    receipt = {
        "semantics_profile": semantics.SEMANTICS_PROFILE,
        "files": [
            {
                "series": ref.key,
                "year": 2021,
                "name": path.name,
                "raw_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_semantic_sha256": semantics.canonical_instance_sha256(raw),
                "v2_semantic_sha256": semantics.canonical_instance_sha256(normalized),
                "changed": True,
            }
        ],
    }
    return ref, receipt, path


def test_authenticated_v2_loader_binds_raw_and_canonical_semantics(
    tmp_path: Path,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    ref, receipt, _ = _authenticated_series_fixture(tmp_path)

    loaded = semantics.load_series_authenticated_v2(ref, receipt)

    assert tuple(loaded) == (2021,)
    assert loaded[2021].votes[0].projects == ("a", "b")


def test_authenticated_v2_loader_rejects_raw_digest_mismatch(tmp_path: Path) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    ref, receipt, _ = _authenticated_series_fixture(tmp_path)
    receipt["files"][0]["raw_file_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="raw file digest differs"):
        semantics.load_series_authenticated_v2(ref, receipt)


def test_authenticated_v2_loader_rejects_mutation_during_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    ref, receipt, path = _authenticated_series_fixture(tmp_path)
    original_parse = semantics.parse_pb_file

    def mutate_after_parse(raw_path: Path) -> PBInstance:
        instance = original_parse(raw_path)
        path.write_bytes(path.read_bytes() + b"\n")
        return instance

    monkeypatch.setattr(semantics, "parse_pb_file", mutate_after_parse)

    with pytest.raises(RuntimeError, match="raw file changed during parse"):
        semantics.load_series_authenticated_v2(ref, receipt)


def test_authenticated_v2_loader_rejects_stale_semantic_receipt(
    tmp_path: Path,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    ref, receipt, _ = _authenticated_series_fixture(tmp_path)
    receipt["files"][0]["v2_semantic_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="canonical semantic digest differs"):
        semantics.load_series_authenticated_v2(ref, receipt)


def test_authenticated_v2_loader_rejects_remaining_duplicate_approvals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    ref, receipt, path = _authenticated_series_fixture(tmp_path)
    raw = parse_pb_file(path)
    receipt["files"][0]["v2_semantic_sha256"] = (
        semantics.canonical_instance_sha256(raw)
    )

    monkeypatch.setattr(
        semantics,
        "canonicalize_instance",
        lambda instance, *, source_id=None: semantics.CanonicalizationResult(
            instance=instance,
            anomalies=(),
        ),
    )

    with pytest.raises(RuntimeError, match="retained duplicate approvals"):
        semantics.load_series_authenticated_v2(ref, receipt)


def test_v2_manifest_receipt_identity_gate_rejects_coherent_city_out_omission(
    tmp_path: Path,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    protocol = importlib.import_module("iclr_multicity_protocol")
    gdynia_path = tmp_path / "gdynia.pb"
    warsaw_path = tmp_path / "warsaw.pb"
    original_index = {
        "Poland/Gdynia/District": SeriesRef(
            key="Poland/Gdynia/District",
            years=(2021,),
            paths=(gdynia_path,),
        ),
        "Poland/Warszawa/District": SeriesRef(
            key="Poland/Warszawa/District",
            years=(2021,),
            paths=(warsaw_path,),
        ),
    }
    receipt = {
        "semantics_profile": "approval-set-first-occurrence-v2",
        "files": [
            {
                "series": ref.key,
                "year": year,
                "name": path.name,
                "raw_file_sha256": "a" * 64,
                "v2_semantic_sha256": "b" * 64,
            }
            for ref in original_index.values()
            for year, path in zip(ref.years, ref.paths)
        ]
    }
    semantics.validate_manifest_receipt_identities_v2(original_index, receipt)

    renamed_key = "Poland/Warszawa/Renamed-Gdynia"
    changed_index = {
        renamed_key: SeriesRef(
            key=renamed_key,
            years=(2021,),
            paths=(gdynia_path,),
        ),
        original_index["Poland/Warszawa/District"].key: original_index[
            "Poland/Warszawa/District"
        ],
    }
    heldout_split = protocol.build_multicity_splits(changed_index)[
        "city_out_Poland_Warszawa"
    ]
    assert renamed_key not in {key for key, _ in heldout_split.train}

    with pytest.raises(RuntimeError, match="manifest identities differ"):
        semantics.validate_manifest_receipt_identities_v2(changed_index, receipt)


def test_known_wola_semantic_digest_changes_only_after_set_normalization() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    path = (
        PROJECT_ROOT
        / "data"
        / "pb_multicity"
        / "Poland_Warszawa_2022_Wola.pb"
    )
    raw = parse_pb_file(path)
    normalized = semantics.canonicalize_instance(raw, source_id=path.name).instance

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "9054e7b28c97b18a990d75baaea5777942a7703774863944324567fbec60f61a"
    )
    assert semantics.canonical_instance_sha256(raw) == (
        "1750e09bc13b584ec1a0066acb8861652ad4e0bbd68b2799923a0008ff867be1"
    )
    assert semantics.canonical_instance_sha256(normalized) == (
        "495b6b54e9049e62554f2f433b0cf14ee2fb7b2dd3279134a9fcfbafda37b312"
    )


def test_semantic_digest_rejects_a_project_stored_under_the_wrong_key() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    malformed = PBInstance(
        path="malformed.pb",
        meta={"budget": "1", "vote_type": "approval"},
        projects={"wrong-key": Project("actual-id", 1.0, None)},
        votes=[Vote("v", ("actual-id",))],
    )

    with pytest.raises(ValueError, match="project mapping key"):
        semantics.canonical_instance_sha256(malformed)


def test_authenticated_receipt_counts_and_identifies_only_changed_ballots(
    tmp_path: Path,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = []
    for year, name, voter_id, vote in (
        (2021, "first.pb", "v-1", "a,a,b"),
        (2022, "second.pb", "v-2", "b"),
    ):
        path = data_dir / name
        path.write_text(
            "META\n"
            "key;value\n"
            f"date_begin;{year}-01-01\n"
            "budget;2\n"
            "vote_type;approval\n"
            "PROJECTS\n"
            "project_id;cost;selected\n"
            "a;1;1\n"
            "b;1;0\n"
            "VOTES\n"
            "voter_id;vote\n"
            f"{voter_id};{vote}\n",
            encoding="utf-8",
        )
        raw = path.read_bytes()
        rows.append(
            {
                "name": name,
                "series": "Poland/Warszawa/Wola",
                "year": year,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "source_commit": "pinned-commit",
        "source_dir": "/upstream/location",
        "destination": str(data_dir),
        "n_series": 1,
        "n_elections": 2,
        "n_files": 2,
        "files": rows,
    }
    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    receipt = semantics.build_authenticated_semantics_receipt(
        manifest_path,
        data_dir,
        enforce_expected_counts=False,
    )

    assert receipt["semantics_profile"] == "approval-set-first-occurrence-v2"
    assert receipt["source_commit"] == "pinned-commit"
    assert receipt["counts"] == {
        "series": 1,
        "files": 2,
        "ballots": 2,
        "raw_approval_tokens": 4,
        "canonical_approval_tokens": 3,
        "affected_files": 1,
        "affected_ballots": 1,
        "removed_tokens": 1,
        "empty_ballots": 0,
        "unknown_project_tokens": 0,
    }
    assert [row["changed"] for row in receipt["files"]] == [True, False]
    assert receipt["anomalies"] == [
        {
            "series": "Poland/Warszawa/Wola",
            "year": 2021,
            "source_id": "first.pb",
            "vote_index_zero_based": 0,
            "voter_id": "v-1",
            "original_projects": ["a", "a", "b"],
            "canonical_projects": ["a", "b"],
            "duplicate_projects": ["a"],
        }
    ]
    assert len(receipt["parent_manifest_retained_sha256"]) == 64
    assert len(receipt["corpus_semantic_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(receipt)


def test_authenticated_multicity_corpus_passes_the_exact_v2_semantics_gate() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    result_root = PROJECT_ROOT / "results" / "iclr_multicity"
    receipt = semantics.build_authenticated_semantics_receipt(
        result_root / "corpus_manifest.json",
        PROJECT_ROOT / "data" / "pb_multicity",
    )

    validated = semantics.validate_multicity_semantics_receipt(receipt)

    assert validated is receipt
    assert receipt["source_commit"] == (
        "2f4321fec84069f50abf35fb5e90852013d17070"
    )
    assert receipt["counts"] == {
        "series": 75,
        "files": 397,
        "ballots": 1_126_949,
        "raw_approval_tokens": 7_057_192,
        "canonical_approval_tokens": 7_057_191,
        "affected_files": 1,
        "affected_ballots": 1,
        "removed_tokens": 1,
        "empty_ballots": 0,
        "unknown_project_tokens": 0,
    }
    assert receipt["anomalies"] == [
        {
            "series": "Poland/Warszawa/Wola",
            "year": 2021,
            "source_id": "Poland_Warszawa_2022_Wola.pb",
            "vote_index_zero_based": 2976,
            "voter_id": "37051",
            "original_projects": ["147", "1040", "1040"],
            "canonical_projects": ["147", "1040"],
            "duplicate_projects": ["1040"],
        }
    ]
    changed = [row for row in receipt["files"] if row["changed"]]
    assert changed == [
        {
            "series": "Poland/Warszawa/Wola",
            "year": 2021,
            "name": "Poland_Warszawa_2022_Wola.pb",
            "raw_file_sha256": (
                "9054e7b28c97b18a990d75baaea5777942a7703774863944324567fbec60f61a"
            ),
            "raw_semantic_sha256": (
                "1750e09bc13b584ec1a0066acb8861652ad4e0bbd68b2799923a0008ff867be1"
            ),
            "v2_semantic_sha256": (
                "495b6b54e9049e62554f2f433b0cf14ee2fb7b2dd3279134a9fcfbafda37b312"
            ),
            "changed": True,
        }
    ]


def test_authenticated_receipt_rejects_boolean_schema_and_noninteger_counts() -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    result_root = PROJECT_ROOT / "results" / "iclr_multicity"
    receipt = semantics.build_authenticated_semantics_receipt(
        result_root / "corpus_manifest.json",
        PROJECT_ROOT / "data" / "pb_multicity",
    )

    wrong_schema = json.loads(json.dumps(receipt))
    wrong_schema["schema_version"] = True
    with pytest.raises(RuntimeError, match="schema version"):
        semantics.validate_multicity_semantics_receipt(wrong_schema)

    for field, expected in receipt["counts"].items():
        wrong_count = json.loads(json.dumps(receipt))
        wrong_count["counts"][field] = True if expected == 1 else float(expected)
        with pytest.raises(RuntimeError, match="corpus counts"):
            semantics.validate_multicity_semantics_receipt(wrong_count)

    for field in ("year", "vote_index_zero_based"):
        wrong_anomaly = json.loads(json.dumps(receipt))
        wrong_anomaly["anomalies"][0][field] = float(
            wrong_anomaly["anomalies"][0][field]
        )
        with pytest.raises(RuntimeError, match="anomaly inventory"):
            semantics.validate_multicity_semantics_receipt(wrong_anomaly)

    wrong_file_year = json.loads(json.dumps(receipt))
    wrong_file_year["files"][0]["year"] = float(
        wrong_file_year["files"][0]["year"]
    )
    with pytest.raises(RuntimeError, match="malformed file row"):
        semantics.validate_multicity_semantics_receipt(wrong_file_year)


def test_authenticated_builder_applies_the_exact_gate_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    monkeypatch.setattr(
        semantics,
        "_EXPECTED_CORPUS_SEMANTIC_SHA256",
        "0" * 64,
    )
    result_root = PROJECT_ROOT / "results" / "iclr_multicity"

    with pytest.raises(RuntimeError, match="corpus digest differs"):
        semantics.build_authenticated_semantics_receipt(
            result_root / "corpus_manifest.json",
            PROJECT_ROOT / "data" / "pb_multicity",
        )
