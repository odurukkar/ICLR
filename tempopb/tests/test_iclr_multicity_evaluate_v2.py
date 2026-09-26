"""Protocol-v2 evaluation ingestion and provenance contracts."""

from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from iclr_corpus import SeriesRef, Split
from iclr_multicity_protocol import evidence_gate_config
from iclr_multicity_train import build_multicity_arm


def _write_approval_instance(path: Path, year: int) -> None:
    path.write_text(
        "META\n"
        "key;value\n"
        f"date_begin;{year}-01-01\n"
        "budget;1\n"
        "vote_type;approval\n"
        "PROJECTS\n"
        "project_id;cost;selected\n"
        "a;1;1\n"
        "VOTES\n"
        "voter_id;vote;age;sex\n"
        "v;a,a;30;F\n",
        encoding="utf-8",
    )


def _semantics_receipt_for_index(
    index: dict[str, SeriesRef],
) -> dict[str, object]:
    semantics = importlib.import_module("iclr_approval_semantics_v2")
    rows = []
    for key, ref in sorted(index.items()):
        for year, path in zip(ref.years, ref.paths):
            raw = semantics.parse_pb_file(path)
            normalized = semantics.canonicalize_instance(
                raw,
                source_id=Path(path).name,
            ).instance
            rows.append(
                {
                    "series": key,
                    "year": year,
                    "name": Path(path).name,
                    "raw_file_sha256": hashlib.sha256(
                        Path(path).read_bytes()
                    ).hexdigest(),
                    "v2_semantic_sha256": semantics.canonical_instance_sha256(
                        normalized
                    ),
                }
            )
    return {
        "semantics_profile": semantics.SEMANTICS_PROFILE,
        "files": rows,
    }


def test_v2_evaluation_loader_normalizes_full_warmup_and_score_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluate_v1 = importlib.import_module("iclr_multicity_evaluate")
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    env = importlib.import_module("iclr_env")
    warmup = tmp_path / "warmup.pb"
    scored = tmp_path / "scored.pb"
    _write_approval_instance(warmup, 2022)
    _write_approval_instance(scored, 2023)
    ref = SeriesRef(
        key="Poland/Test/Unit",
        years=(2022, 2023),
        paths=(warmup, scored),
    )

    def fail_v1_loader(*_args, **_kwargs):
        raise AssertionError("protocol v1 loader must not be called")

    monkeypatch.setattr(evaluate_v1, "load_series", fail_v1_loader)
    monkeypatch.setattr(env, "load_series", fail_v1_loader)

    index = {ref.key: ref}
    instances = evaluate_v2.load_evaluation_instances_v2(
        index,
        _semantics_receipt_for_index(index),
    )

    assert tuple(instances) == (ref.key,)
    assert tuple(instances[ref.key]) == (2022, 2023)
    for instance in instances[ref.key].values():
        assert instance.votes[0].projects == ("a",)
        assert instance.path.endswith("#approval-set-first-occurrence-v2")


def _valid_fit_fixture(tmp_path: Path):
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    spec = {"split": "temporal_2022", "arm": "priority", "seed": 1}
    source_sha256 = {
        name: f"{index:064x}"
        for index, name in enumerate(protocol_v2.FIT_SOURCE_FILES_V2, start=1)
    }
    tracked = {
        "artifact/corpus_manifest": {"sha256": "a" * 64},
        "artifact/approval_semantics_receipt": {"sha256": "b" * 64},
        "split/temporal_2022": {"sha256": "c" * 64},
        **{
            f"source/{name}": {"sha256": digest}
            for name, digest in source_sha256.items()
        },
    }
    lock_payload = {
        "schema_version": 2,
        "lock_profile": "multicity-priority-v2",
        "fit_inventory": [spec],
        "protocol": protocol_v2.protocol_config_v2(),
        "tracked_files": tracked,
    }
    arm = build_multicity_arm("priority")
    fit_payload = {
        "schema_version": 2,
        "training_only": True,
        "execution_mode": "serial",
        "config": {
            **spec,
            "generations": 30,
            "popsize": None,
            "sigma0": 0.4,
            "bound": 10.0,
            "welfare_penalty": 2.0,
            "welfare_floor": 1.0,
        },
        "arm": {
            "name": arm.name,
            "feature_names": list(arm.feature_names),
            "init_name": arm.init_name,
            "initial": arm.initial.tolist(),
        },
        "result": {"best_weights": [0.0] * 5},
        "provenance": {
            "protocol_lock_sha256": "d" * 64,
            "pabulib_commit": evaluate_v2.PABULIB_COMMIT,
            "corpus_manifest_sha256": "a" * 64,
            "split_sha256": "c" * 64,
            "source_sha256": source_sha256,
            "semantics_profile": "approval-set-first-occurrence-v2",
            "semantics_receipt_sha256": "b" * 64,
            "corpus_semantic_sha256": "e" * 64,
        },
    }
    path = (
        tmp_path
        / "results"
        / "iclr_multicity_v2"
        / "fits"
        / "temporal_2022"
        / "priority"
        / "seed-1.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(fit_payload), encoding="utf-8")
    return evaluate_v2, spec, path, fit_payload, lock_payload


def test_v2_fit_inventory_accepts_exact_schema_and_provenance(
    tmp_path: Path,
) -> None:
    evaluate_v2, spec, path, _, lock_payload = _valid_fit_fixture(tmp_path)

    snapshots = evaluate_v2.verify_fit_inventory_v2(
        [(spec, path)],
        lock_payload,
        "d" * 64,
        corpus_semantic_sha256="e" * 64,
        result_root=tmp_path / "results" / "iclr_multicity_v2",
        repo_root=tmp_path,
    )
    assert len(snapshots) == 1
    assert snapshots[0].spec == spec
    assert snapshots[0].sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    authenticated_payload = snapshots[0].payload
    path.write_text('{"changed": true}\n', encoding="utf-8")
    assert snapshots[0].payload is authenticated_payload
    assert snapshots[0].payload["result"]["best_weights"] == [0.0] * 5
    with pytest.raises(RuntimeError, match="fit changed"):
        evaluate_v2.assert_fit_snapshots_unchanged_v2(snapshots)


def test_v2_fit_inventory_accepts_two_fits_in_one_verification(
    tmp_path: Path,
) -> None:
    evaluate_v2, first_spec, first_path, fit_payload, lock_payload = (
        _valid_fit_fixture(tmp_path)
    )
    second_spec = {**first_spec, "seed": 2}
    second_payload = json.loads(json.dumps(fit_payload))
    second_payload["config"]["seed"] = 2
    second_path = (
        tmp_path
        / "results"
        / "iclr_multicity_v2"
        / "fits"
        / "temporal_2022"
        / "priority"
        / "seed-2.json"
    )
    second_path.write_text(json.dumps(second_payload), encoding="utf-8")
    lock_payload["fit_inventory"] = [first_spec, second_spec]

    snapshots = evaluate_v2.verify_fit_inventory_v2(
        [(first_spec, first_path), (second_spec, second_path)],
        lock_payload,
        "d" * 64,
        corpus_semantic_sha256="e" * 64,
        result_root=tmp_path / "results" / "iclr_multicity_v2",
        repo_root=tmp_path,
    )

    assert [snapshot.spec for snapshot in snapshots] == [
        first_spec,
        second_spec,
    ]


def test_v2_fit_inventory_rejects_symlinked_canonical_fits_root(
    tmp_path: Path,
) -> None:
    evaluate_v2, spec, path, _, lock_payload = _valid_fit_fixture(tmp_path)
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    displaced_fits = tmp_path / "elsewhere" / "fits"
    displaced_fits.parent.mkdir()
    (result_root / "fits").rename(displaced_fits)
    (result_root / "fits").symlink_to(displaced_fits, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink|canonical v2 fits"):
        evaluate_v2.verify_fit_inventory_v2(
            [(spec, path)],
            lock_payload,
            "d" * 64,
            corpus_semantic_sha256="e" * 64,
            result_root=result_root,
            repo_root=tmp_path,
        )


def test_v2_fit_inventory_rejects_v1_lock_and_fit_schemas(
    tmp_path: Path,
) -> None:
    evaluate_v2, spec, path, fit_payload, lock_payload = _valid_fit_fixture(
        tmp_path
    )
    lock_payload["schema_version"] = 1
    lock_payload["lock_profile"] = "multicity-priority-v1"

    with pytest.raises(RuntimeError, match="v2 protocol lock"):
        evaluate_v2.verify_fit_inventory_v2(
            [(spec, path)],
            lock_payload,
            "d" * 64,
            corpus_semantic_sha256="e" * 64,
            result_root=tmp_path / "results" / "iclr_multicity_v2",
            repo_root=tmp_path,
        )


def test_v2_fit_inventory_rejects_boolean_seed_alias(tmp_path: Path) -> None:
    evaluate_v2, spec, path, fit_payload, lock_payload = _valid_fit_fixture(
        tmp_path
    )
    fit_payload["config"]["seed"] = True
    path.write_text(json.dumps(fit_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="seed.*type|invalid.*seed"):
        evaluate_v2.verify_fit_inventory_v2(
            [(spec, path)],
            lock_payload,
            "d" * 64,
            corpus_semantic_sha256="e" * 64,
            result_root=tmp_path / "results" / "iclr_multicity_v2",
            repo_root=tmp_path,
        )


@pytest.mark.parametrize("alias_location", ("arm_initial", "best_weights"))
def test_v2_fit_inventory_rejects_boolean_numeric_aliases(
    tmp_path: Path,
    alias_location: str,
) -> None:
    evaluate_v2, spec, path, fit_payload, lock_payload = _valid_fit_fixture(
        tmp_path
    )
    if alias_location == "arm_initial":
        fit_payload["arm"]["initial"][0] = False
    else:
        fit_payload["result"]["best_weights"][0] = False
    path.write_text(json.dumps(fit_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="arm metadata|best_weights"):
        evaluate_v2.verify_fit_inventory_v2(
            [(spec, path)],
            lock_payload,
            "d" * 64,
            corpus_semantic_sha256="e" * 64,
            result_root=tmp_path / "results" / "iclr_multicity_v2",
            repo_root=tmp_path,
        )

    lock_payload["schema_version"] = 2
    lock_payload["lock_profile"] = "multicity-priority-v2"
    fit_payload["schema_version"] = 1
    path.write_text(json.dumps(fit_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema_version"):
        evaluate_v2.verify_fit_inventory_v2(
            [(spec, path)],
            lock_payload,
            "d" * 64,
            corpus_semantic_sha256="e" * 64,
            result_root=tmp_path / "results" / "iclr_multicity_v2",
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("protocol_lock_sha256", "f" * 64),
        ("semantics_profile", "raw-approval-multiplicity-v1"),
        ("semantics_receipt_sha256", "f" * 64),
        ("corpus_manifest_sha256", "f" * 64),
        ("split_sha256", "f" * 64),
        ("corpus_semantic_sha256", "f" * 64),
    ),
)
def test_v2_fit_inventory_rejects_each_wrong_provenance_binding(
    tmp_path: Path,
    field: str,
    bad_value: str,
) -> None:
    evaluate_v2, spec, path, fit_payload, lock_payload = _valid_fit_fixture(
        tmp_path
    )
    fit_payload["provenance"][field] = bad_value
    path.write_text(json.dumps(fit_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="provenance"):
        evaluate_v2.verify_fit_inventory_v2(
            [(spec, path)],
            lock_payload,
            "d" * 64,
            corpus_semantic_sha256="e" * 64,
            result_root=tmp_path / "results" / "iclr_multicity_v2",
            repo_root=tmp_path,
        )


def test_v2_fit_inventory_rejects_missing_v2_source_hash(
    tmp_path: Path,
) -> None:
    evaluate_v2, spec, path, fit_payload, lock_payload = _valid_fit_fixture(
        tmp_path
    )
    fit_payload["provenance"]["source_sha256"].pop(
        "iclr_approval_semantics_v2.py"
    )
    path.write_text(json.dumps(fit_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="provenance"):
        evaluate_v2.verify_fit_inventory_v2(
            [(spec, path)],
            lock_payload,
            "d" * 64,
            corpus_semantic_sha256="e" * 64,
            result_root=tmp_path / "results" / "iclr_multicity_v2",
            repo_root=tmp_path,
        )


def test_v2_required_fit_inventory_uses_all_locked_split_arm_seed_paths(
    tmp_path: Path,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")

    inventory = evaluate_v2.required_fit_inventory_v2(tmp_path)

    assert len(inventory) == 24
    assert sum(spec["arm"] == "priority" for spec, _ in inventory) == 12
    for spec, path in inventory:
        assert path == (
            tmp_path
            / "fits"
            / str(spec["split"])
            / str(spec["arm"])
            / f"seed-{spec['seed']}.json"
        )


def test_v2_evaluator_rejects_lock_with_a_different_fit_inventory() -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    lock_payload = {
        "fit_inventory": evaluate_v2.required_fit_specs_v2()[:-1]
    }

    with pytest.raises(RuntimeError, match="fit inventory differs"):
        evaluate_v2.verify_lock_fit_inventory_v2(lock_payload)


def test_compute_verified_evaluation_v2_rejects_incomplete_fit_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    monkeypatch.setattr(
        evaluate_v2,
        "load_evaluation_instances_v2",
        lambda *_args: pytest.fail("held-out data loaded before fit validation"),
    )

    with pytest.raises(RuntimeError, match="full v2 matrix"):
        evaluate_v2._compute_verified_evaluation_v2(
            {},
            {},
            [],
            True,
            evidence_gate_config(),
            {"semantics_profile": "approval-set-first-occurrence-v2", "files": []},
            provenance={
                "protocol_lock_sha256": "d" * 64,
                "semantics_profile": "approval-set-first-occurrence-v2",
                "semantics_receipt_sha256": "b" * 64,
                "corpus_semantic_sha256": "e" * 64,
            },
        )


def _write_evaluation_fit(path: Path, arm_name: str) -> None:
    arm = build_multicity_arm(arm_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": {"arm": arm_name},
                "result": {"best_weights": arm.initial.tolist()},
            }
        ),
        encoding="utf-8",
    )


def test_compute_verified_evaluation_v2_uses_snapshots_and_does_not_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluate_v1 = importlib.import_module("iclr_multicity_evaluate")
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    env = importlib.import_module("iclr_env")
    index = {}
    city_keys = {
        "Poland_Warszawa": "Poland/Warszawa/Unit",
        "Poland_Gdynia": "Poland/Gdynia/Unit",
        "Poland_Łódź": "Poland/Łódź/Unit",
    }
    for city, key in city_keys.items():
        path = tmp_path / f"{city}.pb"
        _write_approval_instance(path, 2023)
        index[key] = SeriesRef(key=key, years=(2023,), paths=(path,))
    temporal_test = tuple((key, (2023,)) for key in city_keys.values())
    splits = {
        "temporal_2022": Split(
            name="temporal_2022", train=(), test=temporal_test
        ),
        **{
            f"city_out_{city}": Split(
                name=f"city_out_{city}",
                train=(),
                test=((key, (2023,)),),
            )
            for city, key in city_keys.items()
        },
    }
    inventory = evaluate_v2.required_fit_inventory_v2(tmp_path)
    snapshots = []
    for spec, path in inventory:
        _write_evaluation_fit(path, str(spec["arm"]))
        content = path.read_bytes()
        snapshots.append(
            evaluate_v2.VerifiedFitSnapshotV2(
                split=str(spec["split"]),
                arm=str(spec["arm"]),
                seed=int(spec["seed"]),
                path=path,
                label=path.relative_to(tmp_path).as_posix(),
                sha256=hashlib.sha256(content).hexdigest(),
                payload=json.loads(content.decode("utf-8")),
            )
        )
    provenance = {
        "protocol_lock_sha256": "d" * 64,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "semantics_receipt_sha256": "b" * 64,
        "corpus_semantic_sha256": "e" * 64,
    }

    monkeypatch.setattr(
        evaluate_v1,
        "load_series",
        lambda *_args, **_kwargs: pytest.fail("v1 loader must remain unreachable"),
    )
    monkeypatch.setattr(
        env,
        "load_series",
        lambda *_args, **_kwargs: pytest.fail("v1 loader must remain unreachable"),
    )
    semantics_receipt = _semantics_receipt_for_index(index)
    decision, evidence, _, _, _ = evaluate_v2._compute_verified_evaluation_v2(
        index,
        splits,
        snapshots,
        True,
        evidence_gate_config(),
        semantics_receipt,
        provenance=provenance,
    )

    assert decision["provenance"] == provenance
    assert evidence["provenance"] == provenance
    assert not (tmp_path / "evaluation").exists()
    assert not (tmp_path / "evidence_decision.json").exists()


def test_v2_evaluation_outputs_reject_symlinked_evaluation_dir(
    tmp_path: Path,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    outside = tmp_path / "outside"
    result_root.mkdir(parents=True)
    outside.mkdir()
    (result_root / "evaluation").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink|result output"):
        evaluate_v2.write_evaluation_outputs_v2(
            result_root,
            decision={},
            evidence={},
            summaries={},
            per_series_payload={},
            evaluated={},
        )

    assert tuple(outside.iterdir()) == ()
    assert not (result_root / "evidence_decision.json").exists()


def test_v2_evaluation_orchestration_rejects_symlinked_result_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    real_result = tmp_path / "real-result"
    real_result.mkdir()
    linked_result = tmp_path / "iclr_multicity_v2"
    linked_result.symlink_to(real_result, target_is_directory=True)
    monkeypatch.setattr(
        evaluate_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        lambda *args, **kwargs: pytest.fail("symlinked result root reached verifier"),
    )

    with pytest.raises(RuntimeError, match="symlink"):
        evaluate_v2.execute_locked_evaluation_v2(
            linked_result,
            repo_root=tmp_path,
            data_dir=tmp_path / "data",
        )


def test_v2_evaluation_orchestration_rejects_symlinked_data_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    result_root.mkdir(parents=True)
    real_data = tmp_path / "real-data"
    real_data.mkdir()
    linked_data = tmp_path / "linked-data"
    linked_data.symlink_to(real_data, target_is_directory=True)
    monkeypatch.setattr(
        evaluate_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        lambda *args, **kwargs: pytest.fail("symlinked data root reached verifier"),
    )

    with pytest.raises(RuntimeError, match="symlink"):
        evaluate_v2.execute_locked_evaluation_v2(
            result_root,
            repo_root=tmp_path,
            data_dir=linked_data,
        )


def test_v2_evaluation_output_conflict_is_detected_before_any_new_write(
    tmp_path: Path,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    result_root.mkdir(parents=True)
    decision_path = result_root / "evidence_decision.json"
    decision_path.write_text('{"old":true}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        evaluate_v2.write_evaluation_outputs_v2(
            result_root,
            decision={"new": True},
            evidence={"new": True},
            summaries={"new": True},
            per_series_payload={"new": True},
            evaluated={},
        )

    assert not (result_root / "evaluation").exists()
    assert decision_path.read_text(encoding="utf-8") == '{"old":true}\n'


@pytest.mark.parametrize("drift_kind", ["none", "source", "fit"])
def test_execute_locked_evaluation_v2_verifies_then_receipts_then_loads(
    tmp_path: Path,
    monkeypatch,
    drift_kind: str,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    (result_root / "splits").mkdir(parents=True)
    (tmp_path / "data" / "pb_multicity").mkdir(parents=True)
    lock_path = result_root / "protocol_lock.json"
    semantics_path = result_root / "approval_semantics_receipt.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    semantics_path.write_text(
        json.dumps(
            {
                "semantics_profile": "approval-set-first-occurrence-v2",
                "corpus_semantic_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    (result_root / "corpus_manifest.json").write_text("{}\n", encoding="utf-8")
    (result_root / "structural_gates.json").write_text("{}\n", encoding="utf-8")
    for name in protocol_v2.SPLIT_NAMES:
        (result_root / "splits" / f"{name}.json").write_text(
            "{}\n", encoding="utf-8"
        )
    semantics_digest = hashlib.sha256(semantics_path.read_bytes()).hexdigest()
    events = []
    tracked_source = tmp_path / "src" / "tracked.py"
    tracked_source.parent.mkdir()
    tracked_source.write_text("state = 'A'\n", encoding="utf-8")
    tracked_fit = (
        result_root
        / "fits"
        / "temporal_2022"
        / "priority"
        / "seed-1.json"
    )
    tracked_fit.parent.mkdir(parents=True)
    tracked_fit.write_text('{"state":"A"}\n', encoding="utf-8")
    tracked_fit_content = tracked_fit.read_bytes()
    fit_snapshot = SimpleNamespace(
        path=tracked_fit,
        label=tracked_fit.relative_to(tmp_path).as_posix(),
        sha256=hashlib.sha256(tracked_fit_content).hexdigest(),
    )
    lock_payload = {
        "schema_version": 2,
        "lock_profile": "multicity-priority-v2",
        "fit_inventory": evaluate_v2.required_fit_specs_v2(),
        "evidence_gates": evidence_gate_config(),
        "tracked_files": {
            "artifact/approval_semantics_receipt": {
                "sha256": semantics_digest
            }
        },
    }
    lock_bytes = lock_path.read_bytes()
    lock_snapshot = SimpleNamespace(
        path=lock_path,
        payload=lock_payload,
        content=lock_bytes,
        sha256=hashlib.sha256(lock_bytes).hexdigest(),
    )

    def fake_verify_lock(*args, **kwargs):
        events.append("lock")
        if events.count("lock") == 2:
            assert "load" in events
            if drift_kind == "source":
                assert tracked_source.read_text(encoding="utf-8") == "state = 'B'\n"
                raise RuntimeError("locked v2 digest differs for source/tracked.py")
            if drift_kind == "fit":
                tracked_fit.write_text('{"state":"B"}\n', encoding="utf-8")
        return lock_snapshot

    monkeypatch.setattr(
        evaluate_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        fake_verify_lock,
    )
    monkeypatch.setattr(
        evaluate_v2,
        "verify_lock_fit_inventory_v2",
        lambda payload: events.append("fit-matrix"),
    )
    monkeypatch.setattr(
        evaluate_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: events.append("semantics"),
    )
    monkeypatch.setattr(
        evaluate_v2,
        "validate_structural_gates_v2",
        lambda *args, **kwargs: events.append("structural") or {"status": "pass"},
    )
    monkeypatch.setattr(
        evaluate_v2,
        "verify_fit_inventory_v2",
        lambda *args, **kwargs: events.append("fits") or (fit_snapshot,),
    )

    def fake_open(
        lock,
        receipt,
        inventory,
        structural,
        semantics,
        root,
        *,
        authenticated_fit_sha256,
        data_dir,
        lock_snapshot,
    ):
        events.append("receipt")
        assert lock_snapshot is not None
        assert lock_snapshot.content == lock.read_bytes()
        payload = {
            "schema_version": 2,
            "lock_sha256": lock_snapshot.sha256,
            "semantics_profile": "approval-set-first-occurrence-v2",
            "semantics_receipt_sha256": semantics_digest,
            "corpus_semantic_sha256": "e" * 64,
            "fit_sha256": authenticated_fit_sha256,
            "structural_gates_sha256": hashlib.sha256(
                structural.read_bytes()
            ).hexdigest(),
            "opened_at_utc": "2026-09-09T00:00:00+00:00",
        }
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(protocol_v2, "open_heldout_once_v2", fake_open)
    input_snapshot = SimpleNamespace(
        index={},
        splits={
            name: Split(name=name, train=(), test=())
            for name in protocol_v2.SPLIT_NAMES
        },
    )
    monkeypatch.setattr(
        evaluate_v2,
        "load_protocol_inputs_snapshot_v2",
        lambda *args, **kwargs: events.append("index") or input_snapshot,
    )
    monkeypatch.setattr(
        evaluate_v2,
        "load_canonical_index_from_manifest",
        lambda *args, **kwargs: pytest.fail("live manifest loader is forbidden"),
    )
    monkeypatch.setattr(
        evaluate_v2,
        "load_frozen_split",
        lambda *args, **kwargs: pytest.fail("live split loader is forbidden"),
    )
    monkeypatch.setattr(
        evaluate_v2,
        "validate_manifest_receipt_identities_v2",
        lambda index, receipt: None,
    )
    monkeypatch.setattr(
        evaluate_v2,
        "assert_protocol_inputs_unchanged_v2",
        lambda snapshot: None,
    )

    def fake_compute(*args, **kwargs):
        assert (result_root / "heldout_opened.json").is_file()
        events.append("load")
        if drift_kind == "source":
            tracked_source.write_text("state = 'B'\n", encoding="utf-8")
        provenance = kwargs["provenance"]
        decision = {"classification": "test-only", "provenance": provenance}
        evidence = {"provenance": provenance}
        return decision, evidence, {}, {}, {}

    monkeypatch.setattr(
        evaluate_v2,
        "_compute_verified_evaluation_v2",
        fake_compute,
    )

    if drift_kind != "none":
        message = "source/tracked.py" if drift_kind == "source" else "fit changed"
        with pytest.raises(RuntimeError, match=message):
            evaluate_v2.execute_locked_evaluation_v2(
                result_root,
                repo_root=tmp_path,
                data_dir=tmp_path / "data" / "pb_multicity",
            )
        assert not (result_root / "evidence_decision.json").exists()
        assert not (result_root / "evidence_payload.json").exists()
        assert not (result_root / "evaluation").exists()
    else:
        decision = evaluate_v2.execute_locked_evaluation_v2(
            result_root,
            repo_root=tmp_path,
            data_dir=tmp_path / "data" / "pb_multicity",
        )
        assert decision["classification"] == "test-only"
    assert events.index("lock") < events.index("fits")
    assert events.index("fits") < events.index("receipt")
    assert events.index("receipt") < events.index("index")
    assert events.index("index") < events.index("load")
    assert events.count("lock") == 2
    assert events[-1] == "lock"


def test_v2_evaluation_main_dispatches_to_locked_orchestration(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    evaluate_v2 = importlib.import_module("iclr_multicity_evaluate_v2")
    observed = []
    monkeypatch.setattr(
        evaluate_v2,
        "execute_locked_evaluation_v2",
        lambda result_root: observed.append(result_root)
        or {"classification": "test-only"},
    )

    evaluate_v2.main(["--result-root", str(tmp_path)])

    assert observed == [tmp_path.resolve()]
    assert json.loads(capsys.readouterr().out) == {"classification": "test-only"}


def test_v2_script_entrypoint_follows_all_runtime_definitions() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "iclr_multicity_evaluate_v2.py"
    ).read_text(encoding="utf-8")
    assert source.rfind('if __name__ == "__main__":') > source.index(
        "def execute_locked_evaluation_v2"
    )
