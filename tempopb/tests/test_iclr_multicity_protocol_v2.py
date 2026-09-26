"""Append-only protocol-v2 lock and preparation contracts."""

from __future__ import annotations

import importlib
import copy
import hashlib
import json
from pathlib import Path

import pytest

from iclr_corpus import SeriesRef
from iclr_multicity_protocol import locked_fit_inventory, protocol_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v2_protocol_changes_only_ingestion_semantics_and_result_namespace() -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")

    config_v2 = protocol_v2.protocol_config_v2()
    approval_semantics = config_v2.pop("approval_semantics")
    popsize = config_v2.pop("popsize")

    assert config_v2 == protocol_config()
    assert popsize is None
    assert approval_semantics == {
        "ballot_model": "set",
        "profile": "approval-set-first-occurrence-v2",
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
    assert protocol_v2.LOCK_PROFILE_V2 == "multicity-priority-v2"
    assert protocol_v2.RESULT_ROOT_V2 == (
        PROJECT_ROOT / "results" / "iclr_multicity_v2"
    )
    assert protocol_v2.locked_fit_inventory_v2() == locked_fit_inventory()


def test_v2_input_staging_is_append_only_path_free_and_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = tmp_path / "v1"
    result_root = tmp_path / "v2"
    data_dir = tmp_path / "data"
    (parent / "splits").mkdir(parents=True)
    data_dir.mkdir()
    election = data_dir / "election.pb"
    election.write_text(
        "META\n"
        "key;value\n"
        "date_begin;2021-01-01\n"
        "budget;1\n"
        "vote_type;approval\n"
        "PROJECTS\n"
        "project_id;cost;selected\n"
        "a;1;1\n"
        "VOTES\n"
        "voter_id;vote\n"
        "v;a,a\n",
        encoding="utf-8",
    )
    raw = election.read_bytes()
    manifest = {
        "schema_version": 1,
        "source_commit": "pinned",
        "source_dir": "/unpublished/upstream",
        "destination": "/unpublished/staging",
        "n_series": 1,
        "n_elections": 1,
        "n_files": 1,
        "files": [
            {
                "name": election.name,
                "series": "Poland/Warszawa/Wola",
                "year": 2021,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    }
    (parent / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in protocol_v2.SPLIT_NAMES:
        (parent / "splits" / f"{name}.json").write_text(
            json.dumps({"name": name}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    tracked_files = {
        "artifact/corpus_manifest": {
            "path": "results/iclr_multicity/corpus_manifest.json",
            "sha256": hashlib.sha256(
                (parent / "corpus_manifest.json").read_bytes()
            ).hexdigest(),
        },
        **{
            f"split/{name}": {
                "path": f"results/iclr_multicity/splits/{name}.json",
                "sha256": hashlib.sha256(
                    (parent / "splits" / f"{name}.json").read_bytes()
                ).hexdigest(),
            }
            for name in protocol_v2.SPLIT_NAMES
        },
    }
    (parent / "protocol_lock.json").write_text(
        json.dumps({"tracked_files": tracked_files}), encoding="utf-8"
    )
    parent_lock_bytes = (parent / "protocol_lock.json").read_bytes()
    parent_snapshot = protocol_v2.ParentProtocolSnapshotV2(
        json.loads(parent_lock_bytes),
        parent_lock_bytes,
        "f" * 64,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_load_parent_lineage_v2",
        lambda path: (parent_snapshot, "e" * 64),
    )
    parent_hashes = {
        path.relative_to(parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parent.rglob("*.json")
    }

    first = protocol_v2.stage_protocol_inputs_v2(
        parent,
        result_root,
        data_dir,
        enforce_expected_counts=False,
    )
    second = protocol_v2.stage_protocol_inputs_v2(
        parent,
        result_root,
        data_dir,
        enforce_expected_counts=False,
    )

    assert first == second
    staged_manifest = json.loads(
        (result_root / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    parent_manifest = json.loads(
        (parent / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    assert staged_manifest["source_dir"] == "pabulib_files/pb_files"
    assert staged_manifest["destination"] == "data/pb_multicity"
    assert {
        key: value
        for key, value in staged_manifest.items()
        if key not in {"source_dir", "destination"}
    } == {
        key: value
        for key, value in parent_manifest.items()
        if key not in {"source_dir", "destination"}
    }
    assert set(first["split_sha256"]) == set(protocol_v2.SPLIT_NAMES)
    receipt_text = (result_root / "approval_semantics_receipt.json").read_text()
    assert str(tmp_path) not in receipt_text
    assert json.loads(receipt_text)["counts"]["removed_tokens"] == 1
    assert parent_hashes == {
        path.relative_to(parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in parent.rglob("*.json")
    }

    escaped_target = tmp_path / "escaped-split.json"
    staged_split = result_root / "splits" / "temporal_2022.json"
    staged_split.unlink()
    staged_split.symlink_to(escaped_target)
    with pytest.raises(RuntimeError, match="symlink|non-regular"):
        protocol_v2.stage_protocol_inputs_v2(
            parent,
            result_root,
            data_dir,
            enforce_expected_counts=False,
        )
    assert staged_split.is_symlink()
    assert not escaped_target.exists()


def test_v2_input_staging_rejects_parent_split_drift_before_writing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = tmp_path / "v1"
    result_root = tmp_path / "v2"
    (parent / "splits").mkdir(parents=True)
    (parent / "corpus_manifest.json").write_text(
        json.dumps({"files": []}), encoding="utf-8"
    )
    tracked = {
        "artifact/corpus_manifest": {
            "path": "results/iclr_multicity/corpus_manifest.json",
            "sha256": hashlib.sha256(
                (parent / "corpus_manifest.json").read_bytes()
            ).hexdigest(),
        }
    }
    for name in protocol_v2.SPLIT_NAMES:
        path = parent / "splits" / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        tracked[f"split/{name}"] = {
            "path": f"results/iclr_multicity/splits/{name}.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (parent / "protocol_lock.json").write_text(
        json.dumps({"tracked_files": tracked}), encoding="utf-8"
    )
    (parent / "splits" / "temporal_2022.json").write_text(
        '{"name":"drifted"}', encoding="utf-8"
    )
    parent_lock_bytes = (parent / "protocol_lock.json").read_bytes()
    parent_snapshot = protocol_v2.ParentProtocolSnapshotV2(
        json.loads(parent_lock_bytes),
        parent_lock_bytes,
        "f" * 64,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_load_parent_lineage_v2",
        lambda path: (parent_snapshot, "e" * 64),
    )

    with pytest.raises(RuntimeError, match="parent split differs"):
        protocol_v2.stage_protocol_inputs_v2(
            parent,
            result_root,
            tmp_path / "data",
            enforce_expected_counts=False,
        )

    assert not result_root.exists()


def test_v2_input_staging_rejects_nested_result_roots(tmp_path: Path) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = tmp_path / "iclr_multicity"

    with pytest.raises(ValueError, match="not nested"):
        protocol_v2.stage_protocol_inputs_v2(
            parent,
            parent / "iclr_multicity_v2",
            tmp_path / "data",
            enforce_expected_counts=False,
        )


def test_v2_immutable_writer_rejects_parent_rename_during_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    evaluation = tmp_path / "result" / "evaluation"
    evaluation.mkdir(parents=True)
    displaced = tmp_path / "displaced-evaluation"
    real_link = protocol_v2.os.link

    def rename_then_link(*args, **kwargs):
        evaluation.rename(displaced)
        evaluation.mkdir()
        return real_link(*args, **kwargs)

    monkeypatch.setattr(protocol_v2.os, "link", rename_then_link)

    with pytest.raises(RuntimeError, match="changed|directory identity"):
        protocol_v2.write_immutable_json_artifact_v2(
            evaluation / "summary.json",
            {"status": "pass"},
        )

    assert not (evaluation / "summary.json").exists()
    assert (displaced / "summary.json").is_file()


def test_v2_regular_reader_rejects_parent_rename_during_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    fits = tmp_path / "result" / "fits"
    fits.mkdir(parents=True)
    target = fits / "seed-1.json"
    target.write_bytes(b"A")
    displaced = tmp_path / "displaced-fits"
    real_read = protocol_v2._read_regular_at_v2
    calls = 0

    def read_then_rename(*args, **kwargs):
        nonlocal calls
        content = real_read(*args, **kwargs)
        calls += 1
        if calls == 1:
            fits.rename(displaced)
            fits.mkdir()
            (fits / "seed-1.json").write_bytes(b"B")
        return content

    monkeypatch.setattr(protocol_v2, "_read_regular_at_v2", read_then_rename)

    with pytest.raises(RuntimeError, match="changed|directory identity"):
        protocol_v2.read_regular_bytes_artifact_v2(
            target,
            label="test fit",
        )


def test_v2_json_snapshot_uses_one_authenticated_byte_image(tmp_path: Path) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    path = tmp_path / "input.json"
    original = b'{"version":"A"}\n'
    replacement = b'{"version":"B"}\n'
    path.write_bytes(original)
    expected_sha256 = hashlib.sha256(original).hexdigest()

    snapshot = protocol_v2._read_json_snapshot_v2(
        path,
        "artifact/test",
        expected_sha256=expected_sha256,
    )
    path.write_bytes(replacement)
    path.write_bytes(original)

    assert snapshot.content == original
    assert snapshot.payload == {"version": "A"}
    assert snapshot.sha256 == expected_sha256

    path.write_bytes(replacement)
    with pytest.raises(RuntimeError, match="locked digest differs"):
        protocol_v2._read_json_snapshot_v2(
            path,
            "artifact/test",
            expected_sha256=expected_sha256,
        )


def test_v2_protocol_input_loader_consumes_only_locked_snapshots(
    tmp_path: Path,
) -> None:
    protocol_v1 = importlib.import_module("iclr_multicity_protocol")
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result = tmp_path / "results" / "iclr_multicity_v2"
    data = tmp_path / "data" / "pb_multicity"
    (result / "splits").mkdir(parents=True)
    data.mkdir(parents=True)
    election = data / "one.pb"
    election.write_bytes(b"authenticated election bytes\n")
    manifest = {
        "schema_version": 1,
        "source_commit": "fixture",
        "source_dir": "pabulib_files/pb_files",
        "destination": "data/pb_multicity",
        "n_series": 1,
        "n_elections": 1,
        "n_files": 1,
        "files": [
            {
                "name": election.name,
                "series": "Poland/Gdynia/District",
                "year": 2021,
                "bytes": election.stat().st_size,
                "sha256": hashlib.sha256(election.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = result / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index = {
        "Poland/Gdynia/District": SeriesRef(
            key="Poland/Gdynia/District",
            years=(2021,),
            paths=(election,),
        )
    }
    split_name = "city_out_Poland_Warszawa"
    split_payload = protocol_v1.split_payload(
        protocol_v1.build_multicity_splits(index)[split_name],
        index,
    )
    split_path = result / "splits" / f"{split_name}.json"
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")
    lock_payload = {
        "tracked_files": {
            "artifact/corpus_manifest": {
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            },
            f"split/{split_name}": {
                "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()
            },
        }
    }

    snapshot = protocol_v2.load_protocol_inputs_snapshot_v2(
        lock_payload,
        result,
        data,
        (split_name,),
        enforce_expected_counts=False,
    )

    assert snapshot.index == index
    assert snapshot.splits[split_name] == protocol_v1.build_multicity_splits(index)[
        split_name
    ]

    displaced_manifest = tmp_path / "outside-manifest.json"
    manifest_path.rename(displaced_manifest)
    manifest_path.symlink_to(displaced_manifest)
    with pytest.raises(RuntimeError, match="symlink"):
        protocol_v2.load_protocol_inputs_snapshot_v2(
            lock_payload,
            result,
            data,
            (split_name,),
            enforce_expected_counts=False,
        )
    manifest_path.unlink()
    displaced_manifest.rename(manifest_path)

    displaced_split = tmp_path / "outside-split.json"
    split_path.rename(displaced_split)
    split_path.symlink_to(displaced_split)
    with pytest.raises(RuntimeError, match="symlink"):
        protocol_v2.load_protocol_inputs_snapshot_v2(
            lock_payload,
            result,
            data,
            (split_name,),
            enforce_expected_counts=False,
        )
    split_path.unlink()
    displaced_split.rename(split_path)

    manifest_path.write_text(json.dumps({**manifest, "n_series": 2}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked digest differs"):
        protocol_v2.load_protocol_inputs_snapshot_v2(
            lock_payload,
            result,
            data,
            (split_name,),
            enforce_expected_counts=False,
        )


def test_v2_snapshot_parsers_match_v1_on_the_authenticated_parent_inputs() -> None:
    protocol_v1 = importlib.import_module("iclr_multicity_protocol")
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = PROJECT_ROOT / "results" / "iclr_multicity"
    data_dir = PROJECT_ROOT / "data" / "pb_multicity"
    manifest_snapshot = protocol_v2._read_json_snapshot_v2(
        parent / "corpus_manifest.json",
        "artifact/corpus_manifest",
    )

    expected_index = protocol_v1.load_canonical_index_from_manifest(
        parent / "corpus_manifest.json",
        data_dir,
    )
    observed_index = protocol_v2._index_from_manifest_snapshot_v2(
        manifest_snapshot,
        data_dir,
    )
    assert observed_index == expected_index

    split_path = parent / "splits" / "city_out_Poland_Warszawa.json"
    split_snapshot = protocol_v2._read_json_snapshot_v2(
        split_path,
        "split/city_out_Poland_Warszawa",
    )
    assert protocol_v2._split_from_snapshot_v2(
        split_snapshot,
        observed_index,
        expected_name="city_out_Poland_Warszawa",
    ) == protocol_v1.load_frozen_split(split_path, expected_index)


def test_v2_manifest_path_redaction_is_mandatory() -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    valid = {
        "source_dir": "pabulib_files/pb_files",
        "destination": "data/pb_multicity",
    }
    protocol_v2._validate_redacted_manifest_v2(valid)

    for field, value in (
        ("source_dir", "/private/upstream"),
        ("destination", "/private/staging"),
    ):
        mutated = dict(valid)
        mutated[field] = value
        with pytest.raises(RuntimeError, match=field):
            protocol_v2._validate_redacted_manifest_v2(mutated)


def test_v2_structural_gate_uses_normalized_ballots_and_binds_the_receipt(
    tmp_path: Path,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    semantics_v2 = importlib.import_module("iclr_approval_semantics_v2")
    path = tmp_path / "duplicate.pb"
    path.write_text(
        "META\n"
        "key;value\n"
        "date_begin;2021-01-01\n"
        "budget;1\n"
        "vote_type;approval\n"
        "PROJECTS\n"
        "project_id;cost;selected\n"
        "a;0.75;1\n"
        "VOTES\n"
        "voter_id;vote\n"
        "v;a,a\n",
        encoding="utf-8",
    )
    ref = SeriesRef(
        key="Poland/Warszawa/Wola",
        years=(2021,),
        paths=(path,),
    )
    raw = semantics_v2.parse_pb_file(path)
    normalized = semantics_v2.canonicalize_instance(
        raw,
        source_id=path.name,
    ).instance
    semantics_receipt = {
        "semantics_profile": semantics_v2.SEMANTICS_PROFILE,
        "files": [
            {
                "series": ref.key,
                "year": 2021,
                "name": path.name,
                "raw_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "v2_semantic_sha256": semantics_v2.canonical_instance_sha256(
                    normalized
                ),
            }
        ],
    }

    report = protocol_v2.verify_priority_mes_corpus_v2(
        (ref,),
        semantics_receipt=semantics_receipt,
        semantics_receipt_sha256="a" * 64,
        corpus_semantic_sha256="b" * 64,
        repo_root=PROJECT_ROOT,
    )
    source_sha256 = protocol_v2.structural_source_sha256_v2(PROJECT_ROOT)

    assert report == {
        "schema_version": 2,
        "status": "pass",
        "semantics_profile": "approval-set-first-occurrence-v2",
        "semantics_receipt_sha256": "a" * 64,
        "corpus_semantic_sha256": "b" * 64,
        "source_sha256": source_sha256,
        "n_series": 1,
        "n_elections": 1,
        "approval_ballots_checked": 1,
        "duplicate_tokens_remaining": 0,
        "winner_identity_checks": 2,
        "determinism_checks": 2,
        "budget_checks": 2,
        "unique_payer_round_checks": 2,
        "payment_rounds_checked": 2,
        "completion_modes": [False, True],
        "zero_weights": [0.0] * 5,
    }


def test_v2_structural_gate_validator_requires_full_corpus_and_receipt_binding(
    tmp_path: Path,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    receipt_sha256 = "b" * 64
    corpus_semantic_sha256 = "c" * 64
    source_sha256 = {
        **protocol_v2.structural_source_sha256_v2(PROJECT_ROOT)
    }
    report = {
        "schema_version": 2,
        "status": "pass",
        "semantics_profile": "approval-set-first-occurrence-v2",
        "semantics_receipt_sha256": receipt_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "source_sha256": source_sha256,
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
    path = tmp_path / "structural_gates.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    observed = protocol_v2.validate_structural_gates_v2(
        path,
        semantics_receipt_sha256=receipt_sha256,
        corpus_semantic_sha256=corpus_semantic_sha256,
        repo_root=PROJECT_ROOT,
    )

    assert observed == report


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda row: row.update(schema_version=1), "schema_version"),
        (lambda row: row.update(schema_version=2.0), "schema_version"),
        (
            lambda row: row.update(corpus_semantic_sha256="0" * 64),
            "corpus_semantic_sha256",
        ),
        (
            lambda row: row["source_sha256"].update(
                {"src/iclr_multicity_protocol_v2.py": "0" * 64}
            ),
            "source_sha256",
        ),
        (lambda row: row.update(payment_rounds_checked=11_077), "payment"),
        (lambda row: row.update(n_series=75.0), "integer field"),
        (
            lambda row: row.update(duplicate_tokens_remaining=False),
            "integer field",
        ),
        (lambda row: row.update(completion_modes=[0, 1]), "completion_modes"),
        (lambda row: row.update(zero_weights=[False] * 5), "zero_weights"),
    ),
)
def test_v2_structural_gate_validator_rejects_provenance_or_coverage_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    receipt_sha256 = "a" * 64
    corpus_semantic_sha256 = "b" * 64
    source_sha256 = {
        **protocol_v2.structural_source_sha256_v2(PROJECT_ROOT)
    }
    report = {
        "schema_version": 2,
        "status": "pass",
        "semantics_profile": "approval-set-first-occurrence-v2",
        "semantics_receipt_sha256": receipt_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "source_sha256": dict(source_sha256),
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
    mutation(report)
    path = tmp_path / "structural_gates.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        protocol_v2.validate_structural_gates_v2(
            path,
            semantics_receipt_sha256=receipt_sha256,
            corpus_semantic_sha256=corpus_semantic_sha256,
            repo_root=PROJECT_ROOT,
        )


def test_v2_structural_source_inventory_is_exact_and_path_free() -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")

    observed = protocol_v2.structural_source_sha256_v2(PROJECT_ROOT)

    assert tuple(observed) == tuple(
        f"src/{name}" for name in protocol_v2.STRUCTURAL_SOURCE_FILES_V2
    )
    assert protocol_v2.STRUCTURAL_SOURCE_FILES_V2 == (
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
    assert all(len(digest) == 64 for digest in observed.values())
    assert not any(str(PROJECT_ROOT) in name for name in observed)


def test_v2_parent_lock_binding_authenticates_the_exact_v1_protocol() -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = PROJECT_ROOT / "results" / "iclr_multicity"

    observed = protocol_v2.parent_protocol_lock_sha256_v2(parent)

    assert observed == (
        "1db76ff89327277ec6e1f7b4f8af42d8aa0f4aa5158af82d457a64e2f813d039"
    )
    assert protocol_v2.parent_heldout_receipt_sha256_v2(parent) == (
        "da4d9bedf568f496ae54c5433e99988edd62b20b5217ef27a9cb972d28da34ee"
    )


def test_v2_lock_payload_and_validator_are_exact_and_fail_closed(
    tmp_path: Path,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = PROJECT_ROOT / "results" / "iclr_multicity"
    first = tmp_path / "src" / "first.py"
    second = tmp_path / "tests" / "second.py"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("FIRST = 1\n", encoding="utf-8")
    second.write_text("SECOND = 2\n", encoding="utf-8")
    expected_files = {"source/first": first, "test/second": second}

    payload = protocol_v2.build_protocol_lock_payload_v2(
        tmp_path,
        expected_files,
        parent_result_root=parent,
    )

    assert set(payload) == {
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
    assert payload["schema_version"] == 2
    assert payload["lock_profile"] == "multicity-priority-v2"
    assert payload["mandatory_file_count"] == 2
    assert payload["fit_inventory"] == protocol_v2.locked_fit_inventory_v2()
    assert payload["protocol"] == protocol_v2.protocol_config_v2()
    protocol_v2.validate_mandatory_lock_profile_v2(
        payload,
        tmp_path,
        expected_files,
        parent_result_root=parent,
    )

    changed_protocol = copy.deepcopy(payload)
    changed_protocol["protocol"]["approval_semantics"]["profile"] = "other"
    with pytest.raises(RuntimeError, match="protocol semantics"):
        protocol_v2.validate_mandatory_lock_profile_v2(
            changed_protocol,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )

    boolean_seed = copy.deepcopy(payload)
    boolean_seed["fit_inventory"][0]["seed"] = True
    with pytest.raises(RuntimeError, match="fit inventory"):
        protocol_v2.validate_mandatory_lock_profile_v2(
            boolean_seed,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )

    boolean_floor = copy.deepcopy(payload)
    boolean_floor["protocol"]["objective"]["welfare_floor_ratio"] = True
    with pytest.raises(RuntimeError, match="protocol semantics"):
        protocol_v2.validate_mandatory_lock_profile_v2(
            boolean_floor,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )

    integer_bound = copy.deepcopy(payload)
    integer_bound["protocol"]["coefficient_bound"] = 10
    with pytest.raises(RuntimeError, match="protocol semantics"):
        protocol_v2.validate_mandatory_lock_profile_v2(
            integer_bound,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )

    missing_file = copy.deepcopy(payload)
    missing_file["tracked_files"].pop("test/second")
    with pytest.raises(RuntimeError, match="tracked-file set"):
        protocol_v2.validate_mandatory_lock_profile_v2(
            missing_file,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )

    wrong_parent = copy.deepcopy(payload)
    wrong_parent["parent_protocol_lock_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="parent protocol lock"):
        protocol_v2.validate_mandatory_lock_profile_v2(
            wrong_parent,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )

    wrong_parent_receipt = copy.deepcopy(payload)
    wrong_parent_receipt["parent_heldout_receipt_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="parent held-out receipt"):
        protocol_v2.validate_mandatory_lock_profile_v2(
            wrong_parent_receipt,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )


@pytest.mark.parametrize(
    ("label", "filename", "message"),
    (
        (
            "parent/protocol_lock",
            "protocol_lock.json",
            "tracked parent protocol lock",
        ),
        (
            "parent/heldout_opened",
            "heldout_opened.json",
            "tracked parent held-out receipt",
        ),
    ),
)
def test_v2_lock_cross_binds_tracked_parent_rows_to_lineage_snapshots(
    tmp_path: Path,
    label: str,
    filename: str,
    message: str,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = PROJECT_ROOT / "results" / "iclr_multicity"
    copied_parent = tmp_path / "results" / "iclr_multicity"
    copied_parent.mkdir(parents=True)
    expected_files = {}
    for tracked_label, tracked_filename in (
        ("parent/protocol_lock", "protocol_lock.json"),
        ("parent/heldout_opened", "heldout_opened.json"),
    ):
        copied_path = copied_parent / tracked_filename
        copied_path.write_bytes((parent / tracked_filename).read_bytes())
        expected_files[tracked_label] = copied_path

    payload = protocol_v2.build_protocol_lock_payload_v2(
        tmp_path,
        expected_files,
        parent_result_root=parent,
    )
    decoy = tmp_path / "decoy" / filename
    decoy.parent.mkdir()
    decoy.write_text('{"different":"authenticated bytes"}\n', encoding="utf-8")
    expected_files[label] = decoy
    payload["tracked_files"][label] = {
        "path": decoy.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(decoy.read_bytes()).hexdigest(),
    }

    with pytest.raises(RuntimeError, match=message):
        protocol_v2.validate_mandatory_lock_profile_v2(
            payload,
            tmp_path,
            expected_files,
            parent_result_root=parent,
        )


def test_v2_lock_builder_rejects_symlinked_mandatory_source(tmp_path: Path) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = PROJECT_ROOT / "results" / "iclr_multicity"
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    real_source = source_dir / "real.py"
    real_source.write_text("VALUE = 1\n", encoding="utf-8")
    canonical_source = source_dir / "canonical.py"
    canonical_source.symlink_to(real_source)

    with pytest.raises(RuntimeError, match="symlink"):
        protocol_v2.build_protocol_lock_payload_v2(
            tmp_path,
            {"source/canonical.py": canonical_source},
            parent_result_root=parent,
        )


def test_v2_lineage_rejects_a_modified_parent_snapshot(tmp_path: Path) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = tmp_path / "iclr_multicity"
    parent.mkdir()
    (parent / "protocol_lock.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="parent protocol lock differs"):
        protocol_v2.parent_protocol_lock_sha256_v2(parent)


def test_v2_structural_report_builder_derives_refs_and_provenance_from_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result_root = tmp_path / "results"
    result_root.mkdir()
    (tmp_path / "data").mkdir()
    manifest = {
        "schema_version": 1,
        "source_commit": "pinned",
        "source_dir": "pabulib_files/pb_files",
        "destination": "data/pb_multicity",
        "n_series": 0,
        "n_elections": 0,
        "n_files": 0,
        "files": [],
    }
    manifest_path = result_root / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = {
        "parent_manifest_retained_sha256": (
            protocol_v2.multicity_manifest_retained_sha256(manifest)
        ),
        "corpus_semantic_sha256": "b" * 64,
    }
    receipt_path = result_root / "approval_semantics_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    ref = SeriesRef(key="derived/ref", years=(), paths=())
    calls = {}
    monkeypatch.setattr(
        protocol_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_index_from_manifest_snapshot_v2",
        lambda observed_manifest, observed_data: {ref.key: ref},
    )

    def fake_verify(refs, **kwargs):
        calls["refs"] = refs
        calls.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(protocol_v2, "verify_priority_mes_corpus_v2", fake_verify)

    report = protocol_v2.build_structural_report_v2(
        PROJECT_ROOT,
        result_root,
        tmp_path / "data",
    )

    assert report == {"status": "pass"}
    assert calls == {
        "refs": (ref,),
        "semantics_receipt": receipt,
        "semantics_receipt_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "corpus_semantic_sha256": "b" * 64,
        "repo_root": PROJECT_ROOT.resolve(),
    }


def test_v2_structural_report_builder_rejects_manifest_receipt_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result_root = tmp_path / "results"
    result_root.mkdir()
    (tmp_path / "data").mkdir()
    (result_root / "corpus_manifest.json").write_text(
        json.dumps(
            {
                "source_dir": "pabulib_files/pb_files",
                "destination": "data/pb_multicity",
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    (result_root / "approval_semantics_receipt.json").write_text(
        json.dumps(
            {
                "parent_manifest_retained_sha256": "0" * 64,
                "corpus_semantic_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )

    with pytest.raises(RuntimeError, match="manifest binding"):
        protocol_v2.build_structural_report_v2(
            PROJECT_ROOT,
            result_root,
            tmp_path / "data",
        )


def test_v2_mandatory_inventory_has_exact_protocol_closure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    repo_root = tmp_path / "repo"
    result_root = repo_root / "results" / "iclr_multicity_v2"
    parent_root = repo_root / "results" / "iclr_multicity"
    data_dir = repo_root / "data" / "pb_multicity"
    for directory in (
        repo_root / "src",
        repo_root / "tests",
        result_root / "splits",
        parent_root,
        data_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for name in set(protocol_v2.LOCKED_SOURCE_FILES_V2):
        (repo_root / "src" / name).write_text(f"# {name}\n", encoding="utf-8")
    for name in set(protocol_v2.LOCKED_TEST_FILES_V2):
        (repo_root / "tests" / name).write_text(f"# {name}\n", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (repo_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    election = data_dir / "one.pb"
    election.write_text("one\n", encoding="utf-8")
    manifest = {
        "source_dir": "pabulib_files/pb_files",
        "destination": "data/pb_multicity",
        "files": [],
    }
    (result_root / "corpus_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    receipt = {
        "parent_manifest_retained_sha256": (
            protocol_v2.multicity_manifest_retained_sha256(manifest)
        ),
        "corpus_semantic_sha256": "a" * 64,
    }
    receipt_path = result_root / "approval_semantics_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (result_root / "structural_gates.json").write_text("{}\n", encoding="utf-8")
    split_hashes = {}
    for name in protocol_v2.SPLIT_NAMES:
        path = result_root / "splits" / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        split_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    parent_tracked = {
        f"split/{name}": {
            "path": f"results/iclr_multicity/splits/{name}.json",
            "sha256": digest,
        }
        for name, digest in split_hashes.items()
    }
    (parent_root / "protocol_lock.json").write_text(
        json.dumps({"tracked_files": parent_tracked}), encoding="utf-8"
    )
    (parent_root / "heldout_opened.json").write_text("{}\n", encoding="utf-8")
    ref = SeriesRef(key="one", years=(2021,), paths=(election,))
    parent_lock_bytes = (parent_root / "protocol_lock.json").read_bytes()
    parent_snapshot = protocol_v2.ParentProtocolSnapshotV2(
        json.loads(parent_lock_bytes),
        parent_lock_bytes,
        "b" * 64,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_load_parent_lineage_v2",
        lambda path: (parent_snapshot, "c" * 64),
    )
    monkeypatch.setattr(
        protocol_v2,
        "_index_from_manifest_snapshot_v2",
        lambda manifest_snapshot, observed_data_dir: {ref.key: ref},
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )
    structural_calls = {}

    def fake_structural(path, **kwargs):
        structural_calls.update(kwargs)
        return {}

    monkeypatch.setattr(protocol_v2, "validate_structural_gates_v2", fake_structural)

    observed = protocol_v2.mandatory_protocol_files_v2(
        repo_root,
        result_root,
        data_dir,
        parent_result_root=parent_root,
    )

    assert len(observed) == 39
    assert set(observed) == {
        "artifact/corpus_manifest",
        "artifact/approval_semantics_receipt",
        "artifact/structural_gates",
        "environment/pyproject",
        "environment/uv_lock",
        "parent/protocol_lock",
        "parent/heldout_opened",
        "corpus/one.pb",
        *(f"split/{name}" for name in protocol_v2.SPLIT_NAMES),
        *(f"source/{name}" for name in protocol_v2.LOCKED_SOURCE_FILES_V2),
        *(f"test/{name}" for name in protocol_v2.LOCKED_TEST_FILES_V2),
    }
    assert structural_calls == {
        "semantics_receipt_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
        "corpus_semantic_sha256": "a" * 64,
        "repo_root": repo_root.resolve(),
    }
    assert 39 + 396 == 435


def test_v2_mandatory_inventory_rejects_symlinked_result_or_data_root(
    tmp_path: Path,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    repo_root = tmp_path / "repo"
    real_result = repo_root / "real-result"
    real_data = repo_root / "real-data"
    real_result.mkdir(parents=True)
    real_data.mkdir()
    linked_result = repo_root / "linked-result"
    linked_data = repo_root / "linked-data"
    linked_result.symlink_to(real_result, target_is_directory=True)
    linked_data.symlink_to(real_data, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        protocol_v2.mandatory_protocol_files_v2(
            repo_root,
            linked_result,
            real_data,
        )

    with pytest.raises(RuntimeError, match="symlink"):
        protocol_v2.mandatory_protocol_files_v2(
            repo_root,
            real_result,
            linked_data,
        )


def test_v2_lock_verifier_rejects_tracked_file_drift_without_amendments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    parent = PROJECT_ROOT / "results" / "iclr_multicity"
    source = tmp_path / "src" / "source.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    expected_files = {"source/source.py": source}
    payload = protocol_v2.build_protocol_lock_payload_v2(
        tmp_path,
        expected_files,
        parent_result_root=parent,
    )
    lock_path = tmp_path / "protocol_lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        protocol_v2,
        "mandatory_protocol_files_v2",
        lambda *args, **kwargs: expected_files,
    )

    assert protocol_v2.verify_multicity_protocol_lock_v2(
        lock_path,
        tmp_path,
        tmp_path / "results",
        tmp_path / "data",
        parent_result_root=parent,
    ) == payload

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked-file binding"):
        protocol_v2.verify_multicity_protocol_lock_v2(
            lock_path,
            tmp_path,
            tmp_path / "results",
            tmp_path / "data",
            parent_result_root=parent,
        )


def test_v2_heldout_receipt_is_one_way_and_binds_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    fit = result_root / "fits" / "temporal_2022" / "priority" / "seed-1.json"
    fit.parent.mkdir(parents=True)
    fit.write_text('{"training_only":true}\n', encoding="utf-8")
    lock_path = result_root / "protocol_lock.json"
    structural_path = result_root / "structural_gates.json"
    semantics_path = result_root / "approval_semantics_receipt.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    structural_path.write_text('{"status":"pass"}\n', encoding="utf-8")
    semantics_payload = {
        "semantics_profile": "approval-set-first-occurrence-v2",
        "corpus_semantic_sha256": "c" * 64,
    }
    semantics_path.write_text(json.dumps(semantics_payload), encoding="utf-8")
    spec = {"split": "temporal_2022", "arm": "priority", "seed": 1}
    lock_payload = {
        "schema_version": 2,
        "lock_profile": "multicity-priority-v2",
        "fit_inventory": [spec],
        "tracked_files": {
            "artifact/approval_semantics_receipt": {
                "sha256": hashlib.sha256(semantics_path.read_bytes()).hexdigest()
            },
            "artifact/structural_gates": {
                "sha256": hashlib.sha256(structural_path.read_bytes()).hexdigest()
            },
        },
    }
    lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")

    def fake_lock_snapshot(*_args, **_kwargs):
        content = lock_path.read_bytes()
        return protocol_v2.ProtocolLockSnapshotV2(
            path=lock_path,
            payload=json.loads(content.decode("utf-8")),
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
        )

    monkeypatch.setattr(
        protocol_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        fake_lock_snapshot,
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )
    structural_calls = []
    monkeypatch.setattr(
        protocol_v2,
        "validate_structural_gates_v2",
        lambda *args, **kwargs: structural_calls.append((args, kwargs)) or {},
    )
    receipt_path = result_root / "heldout_opened.json"
    fit_label = (
        "results/iclr_multicity_v2/fits/temporal_2022/priority/seed-1.json"
    )
    authenticated_fit_sha256 = {
        fit_label: hashlib.sha256(fit.read_bytes()).hexdigest()
    }

    impostor = (
        tmp_path
        / "impostor"
        / "fits"
        / "temporal_2022"
        / "priority"
        / "seed-1.json"
    )
    impostor.parent.mkdir(parents=True)
    impostor.write_bytes(fit.read_bytes())
    with pytest.raises(RuntimeError, match="canonical result path"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, impostor)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )
    structural_calls.clear()

    displaced_fits = tmp_path / "elsewhere" / "fits"
    displaced_fits.parent.mkdir()
    (result_root / "fits").rename(displaced_fits)
    (result_root / "fits").symlink_to(displaced_fits, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink|canonical v2 fits"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )
    (result_root / "fits").unlink()
    displaced_fits.rename(result_root / "fits")
    structural_calls.clear()

    receipt_alias = result_root / "receipt-alias.json"
    receipt_path.symlink_to(receipt_alias)
    with pytest.raises(RuntimeError, match="symlink|heldout-opening paths"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )
    assert receipt_path.is_symlink()
    assert not receipt_alias.exists()
    receipt_path.unlink()
    structural_calls.clear()

    authenticated_lock_snapshot = fake_lock_snapshot()
    lock_path.write_text(
        json.dumps({**lock_payload, "coherent_swap": "B"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lock snapshot|changed"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
            lock_snapshot=authenticated_lock_snapshot,
        )
    assert not receipt_path.exists()
    lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")
    structural_calls.clear()

    original_fit_bytes = fit.read_bytes()
    fit.write_bytes(original_fit_bytes + b"\n")
    with pytest.raises(RuntimeError, match="authenticated v2 fit snapshots"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )
    fit.write_bytes(original_fit_bytes)
    structural_calls.clear()

    first = protocol_v2.open_heldout_once_v2(
        lock_path,
        receipt_path,
        [(spec, fit)],
        structural_path,
        semantics_path,
        tmp_path,
        authenticated_fit_sha256=authenticated_fit_sha256,
    )
    second = protocol_v2.open_heldout_once_v2(
        lock_path,
        receipt_path,
        [(spec, fit)],
        structural_path,
        semantics_path,
        tmp_path,
        authenticated_fit_sha256=authenticated_fit_sha256,
    )

    assert first == second
    assert set(first) == {
        "schema_version",
        "lock_sha256",
        "fit_sha256",
        "structural_gates_sha256",
        "semantics_profile",
        "semantics_receipt_sha256",
        "corpus_semantic_sha256",
        "opened_at_utc",
    }
    assert first["schema_version"] == 2
    assert first["semantics_profile"] == "approval-set-first-occurrence-v2"
    assert first["semantics_receipt_sha256"] == lock_payload["tracked_files"][
        "artifact/approval_semantics_receipt"
    ]["sha256"]
    assert first["corpus_semantic_sha256"] == "c" * 64
    assert first["fit_sha256"] == {
        "results/iclr_multicity_v2/fits/temporal_2022/priority/seed-1.json": (
            hashlib.sha256(fit.read_bytes()).hexdigest()
        )
    }
    assert len(structural_calls) == 2

    schema_alias = dict(first)
    schema_alias["schema_version"] = 2.0
    receipt_path.write_text(json.dumps(schema_alias), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different v2 lock or fit inventory"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )

    invalid_timestamp = dict(first)
    invalid_timestamp["opened_at_utc"] = 7
    receipt_path.write_text(json.dumps(invalid_timestamp), encoding="utf-8")
    with pytest.raises(RuntimeError, match="opening timestamp"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )

    conflicting = dict(first)
    conflicting["semantics_receipt_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(conflicting), encoding="utf-8")
    with pytest.raises(RuntimeError, match="different v2 lock or fit inventory"):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )


def test_v2_heldout_receipt_rejects_semantics_hash_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    result_root.mkdir(parents=True)
    for name in (
        "protocol_lock.json",
        "structural_gates.json",
        "approval_semantics_receipt.json",
    ):
        (result_root / name).write_text("{}\n", encoding="utf-8")
    lock_payload = {
        "schema_version": 2,
        "lock_profile": "multicity-priority-v2",
        "fit_inventory": [],
        "tracked_files": {
            "artifact/approval_semantics_receipt": {"sha256": "0" * 64},
            "artifact/structural_gates": {
                "sha256": hashlib.sha256(
                    (result_root / "structural_gates.json").read_bytes()
                ).hexdigest()
            },
        },
    }
    lock_bytes = (result_root / "protocol_lock.json").read_bytes()
    monkeypatch.setattr(
        protocol_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        lambda *args, **kwargs: protocol_v2.ProtocolLockSnapshotV2(
            path=result_root / "protocol_lock.json",
            payload=lock_payload,
            content=lock_bytes,
            sha256=hashlib.sha256(lock_bytes).hexdigest(),
        ),
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )

    with pytest.raises(RuntimeError, match="semantics receipt differs"):
        protocol_v2.open_heldout_once_v2(
            result_root / "protocol_lock.json",
            result_root / "heldout_opened.json",
            [],
            result_root / "structural_gates.json",
            result_root / "approval_semantics_receipt.json",
            tmp_path,
            authenticated_fit_sha256={},
        )


@pytest.mark.parametrize("drift_kind", ["source", "fit"])
def test_v2_heldout_opening_revalidates_full_lock_before_receipt_write(
    tmp_path: Path,
    monkeypatch,
    drift_kind: str,
) -> None:
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    result_root.mkdir(parents=True)
    lock_path = result_root / "protocol_lock.json"
    structural_path = result_root / "structural_gates.json"
    semantics_path = result_root / "approval_semantics_receipt.json"
    receipt_path = result_root / "heldout_opened.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    structural_path.write_text("{}\n", encoding="utf-8")
    semantics_payload = {
        "semantics_profile": "approval-set-first-occurrence-v2",
        "corpus_semantic_sha256": "c" * 64,
    }
    semantics_path.write_text(json.dumps(semantics_payload), encoding="utf-8")
    tracked_source = tmp_path / "src" / "tracked.py"
    tracked_source.parent.mkdir()
    tracked_source.write_text("state = 'A'\n", encoding="utf-8")
    spec = {"split": "temporal_2022", "arm": "priority", "seed": 1}
    fit = result_root / "fits" / "temporal_2022" / "priority" / "seed-1.json"
    fit.parent.mkdir(parents=True)
    fit.write_text('{"state":"A"}\n', encoding="utf-8")
    fit_label = fit.relative_to(tmp_path).as_posix()
    authenticated_fit_sha256 = {
        fit_label: hashlib.sha256(fit.read_bytes()).hexdigest()
    }
    lock_payload = {
        "schema_version": 2,
        "lock_profile": "multicity-priority-v2",
        "fit_inventory": [spec],
        "tracked_files": {
            "artifact/approval_semantics_receipt": {
                "sha256": hashlib.sha256(semantics_path.read_bytes()).hexdigest()
            },
            "artifact/structural_gates": {
                "sha256": hashlib.sha256(structural_path.read_bytes()).hexdigest()
            },
        },
    }
    lock_content = lock_path.read_bytes()
    lock_snapshot = protocol_v2.ProtocolLockSnapshotV2(
        path=lock_path,
        payload=lock_payload,
        content=lock_content,
        sha256=hashlib.sha256(lock_content).hexdigest(),
    )
    verification_count = 0

    def fake_verify(*args, **kwargs):
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            if drift_kind == "source":
                assert tracked_source.read_text(encoding="utf-8") == "state = 'B'\n"
                raise RuntimeError("locked v2 digest differs for source/tracked.py")
            fit.write_text('{"state":"B"}\n', encoding="utf-8")
        return lock_snapshot

    def mutate_during_structural_validation(*args, **kwargs):
        if drift_kind == "source":
            tracked_source.write_text("state = 'B'\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        protocol_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        fake_verify,
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_structural_gates_v2",
        mutate_during_structural_validation,
    )

    message = "source/tracked.py" if drift_kind == "source" else "fit changed"
    with pytest.raises(RuntimeError, match=message):
        protocol_v2.open_heldout_once_v2(
            lock_path,
            receipt_path,
            [(spec, fit)],
            structural_path,
            semantics_path,
            tmp_path,
            authenticated_fit_sha256=authenticated_fit_sha256,
        )

    assert verification_count == 2
    assert not receipt_path.exists()
