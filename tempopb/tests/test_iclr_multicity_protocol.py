"""Locked corpus and split semantics for the three-city ICLR experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from iclr_corpus import SeriesRef
from iclr_multicity_protocol import (
    MultiCityProtocol,
    build_protocol_lock,
    build_multicity_splits,
    freeze_multicity_splits,
    load_canonical_index_from_manifest,
    load_frozen_split,
    locked_fit_inventory,
    select_canonical_series,
    split_payload,
    stage_canonical_corpus,
    validate_canonical_counts,
    validate_mandatory_lock_profile,
    validate_split_integrity,
    verify_protocol_lock,
    verify_structural_gates,
    open_heldout_once,
    write_json_artifact,
    write_immutable_json_artifact,
    write_immutable_text_artifact,
)


def _ref(key: str, years=(2022, 2023), budget: float = 1.0) -> SeriesRef:
    paths = tuple(Path(f"{key.replace('/', '_')}-{year}-{budget}.pb") for year in years)
    return SeriesRef(key=key, years=tuple(years), paths=paths)


def _index(*refs: SeriesRef):
    return {ref.key: ref for ref in refs}


def _write_amendments(root: Path, entries: list[dict]) -> None:
    path = root / "results" / "iclr_lock_amendments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"schema_version": 1, "amendments": entries},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _retained_manifest_sha256(payload: dict) -> str:
    retained = {
        key: value
        for key, value in payload.items()
        if key not in {"destination", "source_dir"}
    }
    canonical = json.dumps(
        retained,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_selector_keeps_only_canonical_local_electorates(monkeypatch) -> None:
    refs = _index(
        _ref("Poland/Warszawa/Wola"),
        _ref("Poland/Warszawa/CITYWIDE"),
        _ref("Poland/Warszawa/subunit Wawer"),
        _ref("Poland/Gdynia/Cisowa | small", budget=1.0),
        _ref("Poland/Gdynia/Cisowa | large", budget=9.0),
        _ref("Poland/Gdynia/Green Budget"),
        _ref("Poland/Łódź/Bałuty-Doły"),
        _ref("Poland/Łódź/CITYWIDE"),
        _ref("Poland/Wrocław/Psie Pole"),
    )
    budgets = {
        key: float(ref.paths[0].stem.rsplit("-", 1)[-1])
        for key, ref in refs.items()
    }
    monkeypatch.setattr(
        "iclr_multicity_protocol._series_budget",
        lambda ref: (_ for _ in ()).throw(AssertionError("budget probe is forbidden")),
    )

    selected = [ref.key for ref in select_canonical_series(refs)]

    assert selected == [
        "Poland/Gdynia/Cisowa | large",
        "Poland/Warszawa/Wola",
        "Poland/Łódź/Bałuty-Doły",
    ]


def test_manifest_index_verifies_every_staged_file_before_use(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first = data_dir / "a.pb"
    second = data_dir / "b.pb"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    import hashlib

    manifest = {
        "source_commit": "pinned",
        "n_series": 1,
        "n_elections": 2,
        "n_files": 2,
        "files": [
            {
                "name": "a.pb",
                "series": "Poland/Warszawa/Wola",
                "year": 2022,
                "bytes": first.stat().st_size,
                "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            },
            {
                "name": "b.pb",
                "series": "Poland/Warszawa/Wola",
                "year": 2023,
                "bytes": second.stat().st_size,
                "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    write_json_artifact(manifest_path, manifest)

    index = load_canonical_index_from_manifest(
        manifest_path, data_dir, enforce_expected_counts=False
    )
    assert index["Poland/Warszawa/Wola"].years == (2022, 2023)

    second.write_bytes(b"tamper\n")
    with pytest.raises(RuntimeError, match="hash mismatch.*b.pb"):
        load_canonical_index_from_manifest(
            manifest_path, data_dir, enforce_expected_counts=False
        )


def test_selector_requires_warmup_and_scored_years(monkeypatch) -> None:
    refs = _index(
        _ref("Poland/Gdynia/Only old | large", (2020, 2021, 2022)),
        _ref("Poland/Gdynia/Only new | large", (2023, 2024, 2025)),
        _ref("Poland/Gdynia/Both | large", (2022, 2023)),
    )
    monkeypatch.setattr("iclr_multicity_protocol._series_budget", lambda ref: 1.0)

    assert [ref.key for ref in select_canonical_series(refs)] == [
        "Poland/Gdynia/Both | large"
    ]


def test_selector_preserves_the_released_warsaw_year_window() -> None:
    refs = _index(
        _ref("Poland/Warszawa/Wola", tuple(range(2017, 2026))),
        _ref("Poland/Warszawa/Wawer", tuple(range(2016, 2026))),
    )

    selected = {ref.key: ref.years for ref in select_canonical_series(refs)}

    assert selected["Poland/Warszawa/Wola"] == tuple(range(2019, 2026))
    assert selected["Poland/Warszawa/Wawer"] == tuple(range(2016, 2026))


def test_count_validation_is_a_hard_assertion() -> None:
    protocol = MultiCityProtocol(
        expected_series=1,
        expected_elections=2,
        expected_train_elections=1,
        expected_test_elections=1,
    )
    refs = (_ref("Poland/Warszawa/Wola"),)
    validate_canonical_counts(refs, protocol)

    with pytest.raises(ValueError, match="expected 2 series"):
        validate_canonical_counts(
            refs,
            MultiCityProtocol(
                expected_series=2,
                expected_elections=2,
                expected_train_elections=1,
                expected_test_elections=1,
            ),
        )


def test_staging_copies_only_selected_files_and_records_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    selected_path = source / "selected.pb"
    rejected_path = source / "rejected.pb"
    selected_path.write_bytes(b"selected corpus bytes\n")
    rejected_path.write_bytes(b"citywide bytes\n")
    selected = SeriesRef(
        key="Poland/Warszawa/Wola",
        years=(2022, 2023),
        paths=(selected_path, selected_path),
    )
    rejected = SeriesRef(
        key="Poland/Warszawa/CITYWIDE",
        years=(2022, 2023),
        paths=(rejected_path, rejected_path),
    )
    monkeypatch.setattr(
        "iclr_multicity_protocol.build_series_index",
        lambda cfg: _index(selected, rejected),
    )

    manifest = stage_canonical_corpus(
        source,
        destination,
        source_commit="deadbeef",
        enforce_expected_counts=False,
    )

    assert sorted(path.name for path in destination.glob("*.pb")) == ["selected.pb"]
    assert manifest["source_commit"] == "deadbeef"
    assert manifest["n_series"] == 1
    assert manifest["n_files"] == 1
    row = manifest["files"][0]
    assert row["name"] == "selected.pb"
    assert row["bytes"] == len(b"selected corpus bytes\n")
    assert len(row["sha256"]) == 64


def test_staging_refuses_hash_conflicts(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    source_path = source / "same.pb"
    source_path.write_bytes(b"pinned upstream")
    (destination / "same.pb").write_bytes(b"different local bytes")
    ref = SeriesRef(
        key="Poland/Warszawa/Wola",
        years=(2022, 2023),
        paths=(source_path, source_path),
    )
    monkeypatch.setattr(
        "iclr_multicity_protocol.build_series_index", lambda cfg: _index(ref)
    )

    with pytest.raises(RuntimeError, match="hash conflict"):
        stage_canonical_corpus(
            source,
            destination,
            source_commit="deadbeef",
            enforce_expected_counts=False,
        )


def test_json_artifact_is_canonical_and_hash_addressed(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"

    digest = write_json_artifact(path, {"z": 1, "a": "Łódź"})

    assert path.read_text() == '{\n  "a": "Łódź",\n  "z": 1\n}\n'
    assert len(digest) == 64


def test_immutable_artifacts_allow_identical_reruns_only(tmp_path: Path) -> None:
    json_path = tmp_path / "result.json"
    text_path = tmp_path / "result.csv"
    first = write_immutable_json_artifact(json_path, {"a": 1})
    assert write_immutable_json_artifact(json_path, {"a": 1}) == first
    second = write_immutable_text_artifact(text_path, "a,b\n1,2\n")
    assert write_immutable_text_artifact(text_path, "a,b\n1,2\n") == second
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_immutable_json_artifact(json_path, {"a": 2})
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_immutable_text_artifact(text_path, "changed\n")


def test_temporal_split_uses_2022_boundary_and_disjoint_elections() -> None:
    index = _index(
        _ref("Poland/Warszawa/Wola", (2021, 2022, 2023, 2024)),
        _ref("Poland/Gdynia/Cisowa | large", (2020, 2021, 2022, 2023)),
        _ref("Poland/Łódź/Bałuty-Doły", (2022, 2023, 2024, 2025)),
    )

    splits = build_multicity_splits(index)
    temporal = splits["temporal_2022"]

    assert dict(temporal.train) == {
        "Poland/Gdynia/Cisowa | large": (2020, 2021, 2022),
        "Poland/Warszawa/Wola": (2021, 2022),
        "Poland/Łódź/Bałuty-Doły": (2022,),
    }
    assert dict(temporal.test) == {
        "Poland/Gdynia/Cisowa | large": (2023,),
        "Poland/Warszawa/Wola": (2023, 2024),
        "Poland/Łódź/Bałuty-Doły": (2023, 2024, 2025),
    }
    report = validate_split_integrity(splits, index)
    assert report["train_test_overlap"] == 0


@pytest.mark.parametrize(
    "held_out",
    ("Poland/Warszawa", "Poland/Gdynia", "Poland/Łódź"),
)
def test_city_out_fit_excludes_held_city_but_preserves_warmup(held_out: str) -> None:
    index = _index(
        _ref("Poland/Warszawa/Wola", (2021, 2022, 2023)),
        _ref("Poland/Gdynia/Cisowa | large", (2021, 2022, 2023)),
        _ref("Poland/Łódź/Bałuty-Doły", (2022, 2023, 2024)),
    )

    split = build_multicity_splits(index)[
        f"city_out_{held_out.replace('/', '_')}"
    ]
    assert all(not key.startswith(held_out + "/") for key, _ in split.train)
    assert all(key.startswith(held_out + "/") for key, _ in split.test)
    assert all(min(years) >= 2023 for _, years in split.test)
    for key, years in split.test:
        assert any(year <= 2022 for year in index[key].years)


def test_split_payload_separates_fit_warmup_and_score() -> None:
    index = _index(
        _ref("Poland/Warszawa/Wola", (2021, 2022, 2023)),
        _ref("Poland/Gdynia/Cisowa | large", (2021, 2022, 2023)),
        _ref("Poland/Łódź/Bałuty-Doły", (2022, 2023, 2024)),
    )
    split = build_multicity_splits(index)["city_out_Poland_Warszawa"]

    payload = split_payload(split, index)

    assert "Poland/Warszawa/Wola" not in payload["fit"]
    assert payload["warmup"] == {"Poland/Warszawa/Wola": [2021, 2022]}
    assert payload["score"] == {"Poland/Warszawa/Wola": [2023]}


def test_frozen_split_contents_are_deserialized_and_revalidated(tmp_path: Path) -> None:
    index = _index(
        _ref("Poland/Warszawa/Wola", (2022, 2023)),
        _ref("Poland/Gdynia/Cisowa | large", (2022, 2023)),
        _ref("Poland/Łódź/Bałuty-Doły", (2022, 2023)),
    )
    expected = build_multicity_splits(index)["temporal_2022"]
    path = tmp_path / "temporal_2022.json"
    write_json_artifact(path, split_payload(expected, index))

    observed = load_frozen_split(path, index)
    assert observed == expected

    payload = split_payload(expected, index)
    payload["fit"]["Poland/Warszawa/Wola"] = [2023]
    write_json_artifact(path, payload)
    with pytest.raises(RuntimeError, match="frozen split differs"):
        load_frozen_split(path, index)


def test_freeze_multicity_splits_writes_four_hash_addressed_files(
    tmp_path: Path,
) -> None:
    index = _index(
        _ref("Poland/Warszawa/Wola", (2022, 2023)),
        _ref("Poland/Gdynia/Cisowa | large", (2022, 2023)),
        _ref("Poland/Łódź/Bałuty-Doły", (2022, 2023)),
    )

    hashes = freeze_multicity_splits(index, tmp_path)

    assert sorted(hashes) == [
        "city_out_Poland_Gdynia",
        "city_out_Poland_Warszawa",
        "city_out_Poland_Łódź",
        "temporal_2022",
    ]
    assert all(len(digest) == 64 for digest in hashes.values())
    assert sorted(path.stem for path in tmp_path.glob("*.json")) == sorted(hashes)


def test_protocol_lock_detects_any_tracked_file_change(tmp_path: Path) -> None:
    tracked = tmp_path / "src" / "rule.py"
    tracked.parent.mkdir()
    tracked.write_text("frozen = True\n")
    payload = build_protocol_lock(tmp_path, {"rule": tracked})
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, payload)

    verified = verify_protocol_lock(lock_path, tmp_path)
    assert verified["protocol"]["seeds"] == [1, 2, 42]
    assert verified["protocol"]["primary_seed"] == 42
    assert verified["protocol"]["generations"] == 30
    assert verified["protocol"]["coefficient_bound"] == 10.0
    assert verified["evidence_gates"]["gold"]["city_macro_difference_max"] == -0.010
    assert len(verified["fit_inventory"]) == 24

    tracked.write_text("frozen = False\n")
    with pytest.raises(RuntimeError, match="hash mismatch.*rule"):
        verify_protocol_lock(lock_path, tmp_path)


def test_protocol_lock_rejects_tracked_path_traversal_before_hashing(
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "src" / "rule.py"
    tracked.parent.mkdir()
    tracked.write_text("frozen = True\n", encoding="utf-8")
    payload = build_protocol_lock(tmp_path, {"rule": tracked})
    payload["tracked_files"]["rule"]["path"] = "../outside.py"
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, payload)

    with pytest.raises(RuntimeError, match="malformed tracked file path"):
        verify_protocol_lock(lock_path, tmp_path)


def test_protocol_lock_rejects_tracked_symlink_escape_before_hashing(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("outside = True\n", encoding="utf-8")
    link = tmp_path / "src" / "rule.py"
    link.parent.mkdir()
    link.write_text("inside = True\n", encoding="utf-8")
    payload = build_protocol_lock(tmp_path, {"rule": link})
    link.unlink()
    link.symlink_to(outside)
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, payload)

    with pytest.raises(RuntimeError, match="tracked file escapes repository"):
        verify_protocol_lock(lock_path, tmp_path)


def test_protocol_lock_accepts_current_dev_amendment_shape_only_for_exact_triple(
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "src" / "cohorts.py"
    tracked.parent.mkdir()
    tracked.write_text("frozen = True\n", encoding="utf-8")
    payload = build_protocol_lock(tmp_path, {"source/cohorts.py": tracked})
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, payload)
    locked_sha256 = payload["tracked_files"]["source/cohorts.py"]["sha256"]
    tracked.write_text("frozen = False\n", encoding="utf-8")
    amended_sha256 = hashlib.sha256(tracked.read_bytes()).hexdigest()
    entry = {
        "path": "src/cohorts.py",
        "locked_sha256": locked_sha256,
        "amended_sha256": amended_sha256,
        "change": "Make deterministic accumulation order explicit.",
        "reason": "Remove hash-order dependence without changing frozen decisions.",
        "verification": ["Frozen decision artifacts are byte-identical."],
        "affects_recorded_decision": False,
        "disclosed_in": "appendix",
    }
    _write_amendments(tmp_path, [entry])

    assert verify_protocol_lock(lock_path, tmp_path) == payload

    for invalid_entry in (
        {**entry, "path": "src/unrelated.py"},
        {**entry, "locked_sha256": "0" * 64},
        {**entry, "amended_sha256": "1" * 64},
        {**entry, "affects_recorded_decision": True},
    ):
        _write_amendments(tmp_path, [invalid_entry])
        with pytest.raises(RuntimeError, match="hash mismatch.*source/cohorts.py"):
            verify_protocol_lock(lock_path, tmp_path)


@pytest.mark.parametrize(
    "malformed_entry",
    (
        {
            "path": "../src/rule.py",
            "locked_sha256": "0" * 64,
            "amended_sha256": "1" * 64,
            "affects_recorded_decision": False,
        },
        {
            "path": "src/rule.py",
            "locked_sha256": "not-a-digest",
            "amended_sha256": "1" * 64,
            "affects_recorded_decision": False,
        },
        {
            "path": "src/rule.py",
            "locked_sha256": "0" * 64,
            "amended_sha256": "1" * 64,
        },
    ),
)
def test_protocol_lock_rejects_malformed_amendment_records(
    tmp_path: Path,
    malformed_entry: dict,
) -> None:
    tracked = tmp_path / "src" / "rule.py"
    tracked.parent.mkdir()
    tracked.write_text("frozen = True\n", encoding="utf-8")
    payload = build_protocol_lock(tmp_path, {"rule": tracked})
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, payload)
    _write_amendments(tmp_path, [malformed_entry])

    with pytest.raises(RuntimeError, match="malformed amendment"):
        verify_protocol_lock(lock_path, tmp_path)


@pytest.mark.parametrize("schema_version", (True, 1.0, "1", None))
def test_protocol_lock_requires_integer_amendment_schema_version(
    tmp_path: Path,
    schema_version: object,
) -> None:
    tracked = tmp_path / "src" / "rule.py"
    tracked.parent.mkdir()
    tracked.write_text("frozen = True\n", encoding="utf-8")
    payload = build_protocol_lock(tmp_path, {"rule": tracked})
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, payload)
    amendments_path = tmp_path / "results" / "iclr_lock_amendments.json"
    amendments_path.parent.mkdir(parents=True)
    amendments_path.write_text(
        json.dumps({"schema_version": schema_version, "amendments": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="malformed amendment inventory"):
        verify_protocol_lock(lock_path, tmp_path)


def test_protocol_lock_rejects_unrelated_malformed_manifest_amendment(
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "src" / "rule.py"
    tracked.parent.mkdir()
    tracked.write_text("frozen = True\n", encoding="utf-8")
    payload = build_protocol_lock(tmp_path, {"rule": tracked})
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, payload)
    _write_amendments(
        tmp_path,
        [
            {
                "path": "results/iclr_multicity/corpus_manifest.json",
                "locked_sha256": "0" * 64,
                "amended_sha256": "1" * 64,
                "affects_recorded_decision": False,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="semantic constraint"):
        verify_protocol_lock(lock_path, tmp_path)


def test_release_manifest_amendment_verifies_only_location_redaction(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "results" / "iclr_multicity" / "corpus_manifest.json"
    manifest.parent.mkdir(parents=True)
    locked = {
        "schema_version": 1,
        "source_commit": "abc123",
        "source_dir": "/private/tmp/upstream",
        "destination": "/Users/author/project/data/pb_multicity",
        "n_series": 1,
        "n_elections": 1,
        "n_files": 1,
        "files": [{"name": "one.pb", "sha256": "a" * 64}],
    }
    manifest.write_text(
        json.dumps(locked, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock_payload = build_protocol_lock(
        tmp_path,
        {"artifact/corpus_manifest": manifest},
    )
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, lock_payload)
    locked_sha256 = lock_payload["tracked_files"]["artifact/corpus_manifest"][
        "sha256"
    ]
    redacted = {
        **locked,
        "source_dir": "pabulib_files/pb_files",
        "destination": "data/pb_multicity",
    }
    manifest.write_text(
        json.dumps(redacted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    amended_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    entry = {
        "path": "results/iclr_multicity/corpus_manifest.json",
        "locked_sha256": locked_sha256,
        "amended_sha256": amended_sha256,
        "affects_recorded_decision": False,
        "semantic_constraint": {
            "kind": "json-top-level-location-redaction-v1",
            "excluded_top_level_fields": ["destination", "source_dir"],
            "retained_payload_sha256": _retained_manifest_sha256(locked),
            "current_values": {
                "destination": "data/pb_multicity",
                "source_dir": "pabulib_files/pb_files",
            },
        },
    }
    _write_amendments(tmp_path, [entry])

    assert verify_protocol_lock(lock_path, tmp_path) == lock_payload

    semantic_drift = json.loads(manifest.read_text(encoding="utf-8"))
    semantic_drift["n_files"] = 2
    manifest.write_text(
        json.dumps(semantic_drift, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    drift_entry = {
        **entry,
        "amended_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    _write_amendments(tmp_path, [drift_entry])
    with pytest.raises(RuntimeError, match="semantic constraint"):
        verify_protocol_lock(lock_path, tmp_path)


@pytest.mark.parametrize(
    "constraint_update",
    (
        {"excluded_top_level_fields": ["source_dir"]},
        {"kind": "arbitrary-json-redaction-v1"},
        {"retained_payload_sha256": "0" * 64},
        {"current_values": {"destination": "elsewhere", "source_dir": "hidden"}},
    ),
)
def test_release_manifest_rejects_malformed_or_unbound_semantic_constraints(
    tmp_path: Path,
    constraint_update: dict,
) -> None:
    manifest = tmp_path / "results" / "iclr_multicity" / "corpus_manifest.json"
    manifest.parent.mkdir(parents=True)
    locked = {
        "source_dir": "/private/source",
        "destination": "/private/destination",
        "files": [],
        "n_files": 0,
    }
    manifest.write_text(json.dumps(locked), encoding="utf-8")
    lock_payload = build_protocol_lock(
        tmp_path,
        {"artifact/corpus_manifest": manifest},
    )
    lock_path = tmp_path / "protocol_lock.json"
    write_json_artifact(lock_path, lock_payload)
    current = {
        **locked,
        "source_dir": "pabulib_files/pb_files",
        "destination": "data/pb_multicity",
    }
    manifest.write_text(json.dumps(current), encoding="utf-8")
    constraint = {
        "kind": "json-top-level-location-redaction-v1",
        "excluded_top_level_fields": ["destination", "source_dir"],
        "retained_payload_sha256": _retained_manifest_sha256(locked),
        "current_values": {
            "destination": "data/pb_multicity",
            "source_dir": "pabulib_files/pb_files",
        },
        **constraint_update,
    }
    _write_amendments(
        tmp_path,
        [
            {
                "path": "results/iclr_multicity/corpus_manifest.json",
                "locked_sha256": lock_payload["tracked_files"][
                    "artifact/corpus_manifest"
                ]["sha256"],
                "amended_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "affects_recorded_decision": False,
                "semantic_constraint": constraint,
            }
        ],
    )

    with pytest.raises(RuntimeError, match="semantic constraint"):
        verify_protocol_lock(lock_path, tmp_path)


def test_heldout_receipt_inherits_amendment_check_and_remains_one_way(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    lock_path = tmp_path / "protocol_lock.json"
    lock_payload = build_protocol_lock(tmp_path, {"source": source})
    spec = {"split": "temporal_2022", "arm": "priority", "seed": 1}
    lock_payload["fit_inventory"] = [spec]
    write_json_artifact(lock_path, lock_payload)
    locked_source_sha256 = lock_payload["tracked_files"]["source"]["sha256"]
    source.write_text("version = 2\n", encoding="utf-8")
    _write_amendments(
        tmp_path,
        [
            {
                "path": "source.py",
                "locked_sha256": locked_source_sha256,
                "amended_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "affects_recorded_decision": False,
            }
        ],
    )
    fit = tmp_path / "fits" / "temporal_2022" / "priority" / "seed-1.json"
    structural = tmp_path / "structural_gates.json"
    write_json_artifact(
        structural,
        {
            "status": "pass",
            "n_series": 75,
            "n_elections": 397,
            "identity_checks": 794,
            "determinism_checks": 794,
            "budget_checks": 794,
            "payment_rounds_checked": 11078,
            "completion_modes": [False, True],
            "zero_weights": [0.0] * 5,
        },
    )
    receipt = tmp_path / "heldout_opened.json"

    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"training_only": true}\n')
    with pytest.raises(RuntimeError, match="fit path does not match"):
        open_heldout_once(
            lock_path,
            tmp_path / "wrong-receipt.json",
            [(spec, wrong)],
            structural,
            tmp_path,
        )

    with pytest.raises(RuntimeError, match="missing required fit"):
        open_heldout_once(lock_path, receipt, [(spec, fit)], structural, tmp_path)

    fit.parent.mkdir(parents=True)
    fit.write_text('{"training_only": true}\n')
    first = open_heldout_once(
        lock_path, receipt, [(spec, fit)], structural, tmp_path
    )
    second = open_heldout_once(
        lock_path, receipt, [(spec, fit)], structural, tmp_path
    )

    assert first == second
    assert first["lock_sha256"]
    assert first["fit_sha256"]["fits/temporal_2022/priority/seed-1.json"]
    assert first["structural_gates_sha256"]


def test_empty_or_mismatched_receipt_inventory_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("x=1\n")
    lock = tmp_path / "lock.json"
    write_json_artifact(lock, build_protocol_lock(tmp_path, {"source": source}))
    structural = tmp_path / "structural.json"
    write_json_artifact(
        structural,
        {
            "status": "pass",
            "n_series": 75,
            "n_elections": 397,
            "identity_checks": 794,
            "determinism_checks": 794,
            "budget_checks": 794,
            "payment_rounds_checked": 1,
            "completion_modes": [False, True],
            "zero_weights": [0.0] * 5,
        },
    )
    with pytest.raises(RuntimeError, match="fit inventory differs"):
        open_heldout_once(lock, tmp_path / "receipt.json", [], structural, tmp_path)


def test_structural_gate_requires_complete_corpus_counts(tmp_path: Path) -> None:
    path = tmp_path / "structural.json"
    path.write_text('{"status": "pass"}')
    with pytest.raises(RuntimeError, match="structural gate field"):
        verify_structural_gates(path)


def test_locked_fit_inventory_is_the_canonical_24_run_matrix() -> None:
    inventory = locked_fit_inventory()
    assert len(inventory) == 24
    assert len({(row["split"], row["arm"], row["seed"]) for row in inventory}) == 24


def test_mandatory_lock_profile_rejects_missing_or_extra_labels(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.write_text("a")
    second.write_text("b")
    payload = build_protocol_lock(tmp_path, {"a": first, "b": second})
    payload["lock_profile"] = "multicity-priority-v1"
    expected = {"a": first, "b": second}
    validate_mandatory_lock_profile(payload, tmp_path, expected)

    payload["tracked_files"].pop("b")
    with pytest.raises(RuntimeError, match="mandatory tracked-file set"):
        validate_mandatory_lock_profile(payload, tmp_path, expected)


def test_mandatory_lock_profile_rejects_threshold_or_protocol_edits(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.write_text("x")
    expected = {"source": source}
    payload = build_protocol_lock(tmp_path, expected)
    payload["lock_profile"] = "multicity-priority-v1"

    changed_gate = json.loads(json.dumps(payload))
    changed_gate["evidence_gates"]["gold"]["city_macro_difference_max"] = 1.0
    with pytest.raises(RuntimeError, match="evidence-gate semantics"):
        validate_mandatory_lock_profile(changed_gate, tmp_path, expected)

    changed_protocol = json.loads(json.dumps(payload))
    changed_protocol["protocol"]["generations"] = 1
    with pytest.raises(RuntimeError, match="protocol semantics"):
        validate_mandatory_lock_profile(changed_protocol, tmp_path, expected)
