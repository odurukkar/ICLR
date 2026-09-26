"""Protocol-v2 training ingestion and provenance contracts."""

from __future__ import annotations

import importlib
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from iclr_corpus import SeriesRef, Split
from iclr_multicity_train import MultiCityTrainConfig


def test_v2_training_loader_opens_only_fit_years_with_set_semantics(
    tmp_path: Path,
) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    fit_path = tmp_path / "fit.pb"
    fit_path.write_text(
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
    ref = SeriesRef(
        key="Poland/Warszawa/Wola",
        years=(2021, 2023),
        paths=(fit_path, tmp_path / "heldout-must-not-open.pb"),
    )
    split = Split(
        name="temporal_2022",
        train=((ref.key, (2021,)),),
        test=((ref.key, (2023,)),),
    )
    semantics_v2 = importlib.import_module("iclr_approval_semantics_v2")
    normalized = semantics_v2.parse_pb_file_v2(fit_path)
    receipt = {
        "semantics_profile": semantics_v2.SEMANTICS_PROFILE,
        "files": [
            {
                "series": ref.key,
                "year": 2021,
                "name": fit_path.name,
                "raw_file_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                "v2_semantic_sha256": semantics_v2.canonical_instance_sha256(
                    normalized
                ),
            }
        ]
    }

    rows = train_v2.load_training_series_data_v2(
        split,
        {ref.key: ref},
        receipt,
    )

    assert len(rows) == 1
    assert rows[0].train_years == (2021,)
    assert rows[0].test_years == ()
    assert tuple(rows[0].all_years) == (2021,)
    assert rows[0].all_years[2021].votes[0].projects == ("a",)
    assert rows[0].all_years[2021].path.endswith(
        "#approval-set-first-occurrence-v2"
    )

    fit_path.write_text(fit_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="raw file digest"):
        train_v2.load_training_series_data_v2(split, {ref.key: ref}, receipt)


def test_v2_fit_authorization_uses_only_the_v2_lock_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    config = MultiCityTrainConfig(
        split="temporal_2022",
        arm="priority",
        seed=1,
    )
    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "protocol_lock.json").write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": 2,
        "lock_profile": "multicity-priority-v2",
        "fit_inventory": [
            {"split": "temporal_2022", "arm": "priority", "seed": 1}
        ],
        "protocol": protocol_v2.protocol_config_v2(),
    }
    snapshot = SimpleNamespace(
        path=result_root / "protocol_lock.json",
        payload=payload,
        content=b"{}\n",
        sha256=hashlib.sha256(b"{}\n").hexdigest(),
    )
    calls = []

    def fake_verify(*args, **kwargs):
        calls.append((args, kwargs))
        return snapshot

    monkeypatch.setattr(
        train_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        fake_verify,
    )

    assert train_v2.authorize_fit_v2(
        config,
        smoke=False,
        result_root=result_root,
        repo_root=tmp_path,
    ) is snapshot
    assert calls == [
        (
            (
                result_root / "protocol_lock.json",
                tmp_path,
                result_root,
                tmp_path / "data" / "pb_multicity",
            ),
            {},
        )
    ]

    calls.clear()
    assert train_v2.authorize_fit_v2(
        config,
        smoke=True,
        result_root=result_root,
        repo_root=tmp_path,
    ) is None
    assert calls == []

    payload["schema_version"] = 1
    with pytest.raises(RuntimeError, match="v2 protocol lock profile"):
        train_v2.authorize_fit_v2(
            config,
            smoke=False,
            result_root=result_root,
            repo_root=tmp_path,
        )

    payload["schema_version"] = 2
    with pytest.raises(RuntimeError, match="popsize"):
        train_v2.authorize_fit_v2(
            replace(config, popsize=999),
            smoke=False,
            result_root=result_root,
            repo_root=tmp_path,
        )

    with pytest.raises(RuntimeError, match="hyperparameters"):
        train_v2.authorize_fit_v2(
            replace(config, bound=10),
            smoke=False,
            result_root=result_root,
            repo_root=tmp_path,
        )

    with pytest.raises(RuntimeError, match="canonical v2 split"):
        train_v2.authorize_fit_v2(
            replace(config, split="../../escape"),
            smoke=True,
            result_root=result_root,
            repo_root=tmp_path,
        )


def test_v2_cli_rejects_noncanonical_split_before_path_construction() -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")

    with pytest.raises(SystemExit):
        train_v2.parse_train_args_v2(
            [
                "--split",
                "../../escape",
                "--arm",
                "priority",
                "--seed",
                "1",
                "--smoke",
            ]
        )

    args, config = train_v2.parse_train_args_v2(
        [
            "--split",
            "temporal_2022",
            "--arm",
            "priority",
            "--seed",
            "1",
            "--smoke",
        ]
    )
    assert args.smoke is True
    assert config.generations == 1
    assert config.popsize == 4


def test_v2_path_guard_rejects_symlink_escape(tmp_path: Path) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    result_root = tmp_path / "result"
    outside = tmp_path / "outside"
    result_root.mkdir()
    outside.mkdir()
    (result_root / "smoke").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="escapes its canonical v2 directory"):
        train_v2._resolved_under(
            result_root / "smoke",
            result_root,
            "smoke directory",
        )


def test_v2_training_output_rejects_in_root_symlink_alias(tmp_path: Path) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    result_root = tmp_path / "result"
    fit_dir = result_root / "fits" / "temporal_2022" / "priority"
    fit_dir.mkdir(parents=True)
    alias = fit_dir / "alias.json"
    alias.write_text("{}\n", encoding="utf-8")
    canonical = fit_dir / "seed-1.json"
    canonical.symlink_to(alias)

    with pytest.raises(RuntimeError, match="symlink|result output"):
        train_v2.canonical_result_output_path_v2(
            result_root,
            Path("fits/temporal_2022/priority/seed-1.json"),
        )


def test_v2_fit_provenance_binds_semantics_receipt_and_all_fit_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    protocol_v2 = importlib.import_module("iclr_multicity_protocol_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    (result_root / "splits").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    manifest = result_root / "corpus_manifest.json"
    split = result_root / "splits" / "temporal_2022.json"
    receipt = result_root / "approval_semantics_receipt.json"
    lock = result_root / "protocol_lock.json"
    manifest.write_text('{"manifest":true}\n', encoding="utf-8")
    split.write_text('{"split":true}\n', encoding="utf-8")
    receipt_payload = {
        "semantics_profile": "approval-set-first-occurrence-v2",
        "corpus_semantic_sha256": "c" * 64,
    }
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    monkeypatch.setattr(
        train_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )
    tracked = {
        "artifact/corpus_manifest": {
            "path": "results/iclr_multicity_v2/corpus_manifest.json",
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        },
        "artifact/approval_semantics_receipt": {
            "path": "results/iclr_multicity_v2/approval_semantics_receipt.json",
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        },
        "split/temporal_2022": {
            "path": "results/iclr_multicity_v2/splits/temporal_2022.json",
            "sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        },
    }
    expected_sources = {}
    for name in protocol_v2.FIT_SOURCE_FILES_V2:
        path = tmp_path / "src" / name
        path.write_text(f"# {name}\n", encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        tracked[f"source/{name}"] = {"path": f"src/{name}", "sha256": digest}
        expected_sources[name] = digest
    lock_payload = {
        "schema_version": 2,
        "lock_profile": "multicity-priority-v2",
        "tracked_files": tracked,
    }
    lock.write_text(json.dumps(lock_payload), encoding="utf-8")

    observed = train_v2.build_fit_provenance_v2(
        lock_payload,
        result_root=result_root,
        repo_root=tmp_path,
        split_name="temporal_2022",
    )

    assert observed == {
        "protocol_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "pabulib_commit": train_v2.PABULIB_COMMIT,
        "corpus_manifest_sha256": tracked["artifact/corpus_manifest"]["sha256"],
        "split_sha256": tracked["split/temporal_2022"]["sha256"],
        "source_sha256": expected_sources,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "semantics_receipt_sha256": tracked[
            "artifact/approval_semantics_receipt"
        ]["sha256"],
        "corpus_semantic_sha256": "c" * 64,
    }

    aliased_lock = {**lock_payload, "schema_version": 2.0}
    lock.write_text(json.dumps(aliased_lock), encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock payload changed"):
        train_v2.build_fit_provenance_v2(
            lock_payload,
            result_root=result_root,
            repo_root=tmp_path,
            split_name="temporal_2022",
        )
    lock.write_text(json.dumps(lock_payload), encoding="utf-8")

    receipt.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="approval semantics receipt"):
        train_v2.build_fit_provenance_v2(
            lock_payload,
            result_root=result_root,
            repo_root=tmp_path,
            split_name="temporal_2022",
        )


def test_v2_prewrite_lock_stability_rejects_persistent_swap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    result_root.mkdir(parents=True)
    lock_path = result_root / "protocol_lock.json"
    lock_payload = {"schema_version": 2, "state": "A"}
    lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")
    initial_content = lock_path.read_bytes()
    initial_snapshot = SimpleNamespace(
        path=lock_path,
        payload=lock_payload,
        content=initial_content,
        sha256=hashlib.sha256(initial_content).hexdigest(),
    )
    lock_path.write_text(
        json.dumps({"schema_version": 2, "state": "B"}),
        encoding="utf-8",
    )
    final_content = lock_path.read_bytes()
    final_snapshot = SimpleNamespace(
        path=lock_path,
        payload={"schema_version": 2, "state": "B"},
        content=final_content,
        sha256=hashlib.sha256(final_content).hexdigest(),
    )
    monkeypatch.setattr(
        train_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        lambda *args, **kwargs: final_snapshot,
    )

    with pytest.raises(RuntimeError, match="locked closure changed before fit write"):
        train_v2.assert_protocol_lock_unchanged_before_fit_write_v2(
            initial_snapshot,
            repo_root=tmp_path,
            result_root=result_root,
            data_dir=tmp_path / "data" / "pb_multicity",
        )


def test_v2_prewrite_lock_reverification_rejects_tracked_source_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    result_root.mkdir(parents=True)
    lock_path = result_root / "protocol_lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    content = lock_path.read_bytes()
    initial_snapshot = SimpleNamespace(
        path=lock_path,
        payload={},
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )

    def reject_drift(*args, **kwargs):
        raise RuntimeError("locked v2 digest differs for source/iclr_env.py")

    monkeypatch.setattr(
        train_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        reject_drift,
    )

    with pytest.raises(RuntimeError, match="source/iclr_env.py"):
        train_v2.assert_protocol_lock_unchanged_before_fit_write_v2(
            initial_snapshot,
            repo_root=tmp_path,
            result_root=result_root,
            data_dir=tmp_path / "data" / "pb_multicity",
        )


def test_v2_training_main_routes_only_normalized_data_to_the_reused_fitter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_v1 = importlib.import_module("iclr_multicity_train")
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    (result_root / "splits").mkdir(parents=True)
    for path in (
        result_root / "corpus_manifest.json",
        result_root / "approval_semantics_receipt.json",
        result_root / "splits" / "temporal_2022.json",
    ):
        path.write_text("{}\n", encoding="utf-8")
    config = MultiCityTrainConfig(
        split="temporal_2022",
        arm="priority",
        seed=1,
        generations=1,
        popsize=4,
    )
    split = Split(name="temporal_2022", train=(), test=())
    input_snapshot = SimpleNamespace(
        index={},
        splits={"temporal_2022": split},
    )
    normalized_data = [object()]
    captured = {}
    events = []
    monkeypatch.setattr(train_v2, "ROOT", tmp_path)
    monkeypatch.setattr(train_v2, "RESULT_ROOT", result_root)
    monkeypatch.setattr(
        train_v2,
        "parse_train_args_v2",
        lambda argv: (SimpleNamespace(smoke=True), config),
    )
    monkeypatch.setattr(train_v2, "authorize_fit_v2", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        train_v2,
        "load_protocol_inputs_snapshot_v2",
        lambda *args, **kwargs: input_snapshot,
    )
    monkeypatch.setattr(
        train_v2,
        "load_canonical_index_from_manifest",
        lambda *args, **kwargs: pytest.fail("live manifest loader is forbidden"),
    )
    monkeypatch.setattr(
        train_v2,
        "load_frozen_split",
        lambda *args, **kwargs: pytest.fail("live split loader is forbidden"),
    )
    monkeypatch.setattr(
        train_v2,
        "validate_manifest_receipt_identities_v2",
        lambda index, receipt: None,
    )
    monkeypatch.setattr(
        train_v2,
        "assert_protocol_inputs_unchanged_v2",
        lambda snapshot: events.append("snapshot-stability"),
    )
    monkeypatch.setattr(
        train_v2,
        "load_training_series_data_v2",
        lambda *args: normalized_data,
    )
    monkeypatch.setattr(
        train_v1,
        "load_series",
        lambda *args, **kwargs: pytest.fail("v1 loader must remain unreachable"),
    )

    def fake_fit(config_arg, data_arg, cfg):
        assert data_arg is normalized_data
        return {"schema_version": 1, "result": {"best_loss": 0.25}}

    monkeypatch.setattr(train_v2, "fit_one", fake_fit)
    def fake_provenance(*args, **kwargs):
        assert kwargs["input_snapshot"] is input_snapshot
        events.append("provenance")
        return {"semantics_profile": "v2"}

    monkeypatch.setattr(train_v2, "build_fit_provenance_v2", fake_provenance)

    def fake_write(path, payload):
        events.append("write")
        captured["path"] = path
        captured["payload"] = payload
        return "f" * 64

    monkeypatch.setattr(train_v2, "write_immutable_fit_v2", fake_write)

    train_v2.main([])

    assert captured["path"] == (
        result_root / "smoke" / "temporal_2022" / "priority" / "seed-1.json"
    )
    assert captured["payload"]["schema_version"] == 2
    assert captured["payload"]["provenance"] == {"semantics_profile": "v2"}
    assert events == ["provenance", "snapshot-stability", "write"]


def test_v2_training_revalidates_full_lock_after_fit_and_writes_nothing_on_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_v2 = importlib.import_module("iclr_multicity_train_v2")
    result_root = tmp_path / "results" / "iclr_multicity_v2"
    (result_root / "splits").mkdir(parents=True)
    lock_path = result_root / "protocol_lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    for path in (
        result_root / "corpus_manifest.json",
        result_root / "approval_semantics_receipt.json",
        result_root / "splits" / "temporal_2022.json",
    ):
        path.write_text("{}\n", encoding="utf-8")
    tracked_source = tmp_path / "src" / "tracked.py"
    tracked_source.parent.mkdir()
    tracked_source.write_text("state = 'A'\n", encoding="utf-8")
    lock_content = lock_path.read_bytes()
    lock_snapshot = SimpleNamespace(
        path=lock_path,
        payload={},
        content=lock_content,
        sha256=hashlib.sha256(lock_content).hexdigest(),
    )
    config = MultiCityTrainConfig(
        split="temporal_2022",
        arm="priority",
        seed=1,
    )
    input_snapshot = SimpleNamespace(
        index={},
        splits={"temporal_2022": Split(name="temporal_2022", train=(), test=())},
    )
    output = (
        result_root
        / "fits"
        / "temporal_2022"
        / "priority"
        / "seed-1.json"
    )

    monkeypatch.setattr(train_v2, "ROOT", tmp_path)
    monkeypatch.setattr(train_v2, "RESULT_ROOT", result_root)
    monkeypatch.setattr(
        train_v2,
        "parse_train_args_v2",
        lambda argv: (SimpleNamespace(smoke=False), config),
    )
    monkeypatch.setattr(
        train_v2,
        "authorize_fit_v2",
        lambda *args, **kwargs: lock_snapshot,
    )
    monkeypatch.setattr(
        train_v2,
        "load_protocol_inputs_snapshot_v2",
        lambda *args, **kwargs: input_snapshot,
    )
    monkeypatch.setattr(
        train_v2,
        "validate_multicity_semantics_receipt",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        train_v2,
        "validate_manifest_receipt_identities_v2",
        lambda *args: None,
    )
    monkeypatch.setattr(
        train_v2,
        "load_training_series_data_v2",
        lambda *args: [],
    )
    monkeypatch.setattr(
        train_v2,
        "assert_protocol_inputs_unchanged_v2",
        lambda snapshot: None,
    )

    def fake_fit(*args, **kwargs):
        tracked_source.write_text("state = 'B'\n", encoding="utf-8")
        return {"schema_version": 1, "result": {"best_loss": 0.25}}

    monkeypatch.setattr(train_v2, "fit_one", fake_fit)
    monkeypatch.setattr(
        train_v2,
        "build_fit_provenance_v2",
        lambda *args, **kwargs: {
            "protocol_lock_sha256": lock_snapshot.sha256,
        },
    )

    def reject_changed_closure(*args, **kwargs):
        assert tracked_source.read_text(encoding="utf-8") == "state = 'B'\n"
        raise RuntimeError("locked v2 digest differs for source/tracked.py")

    monkeypatch.setattr(
        train_v2,
        "verify_multicity_protocol_lock_snapshot_v2",
        reject_changed_closure,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_fit_v2",
        lambda *args, **kwargs: pytest.fail("fit write must not be reached"),
    )

    with pytest.raises(RuntimeError, match="source/tracked.py"):
        train_v2.main([])

    assert not output.exists()
