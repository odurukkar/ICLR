"""RED contracts for primary-Warsaw protocol-v2 training."""

from __future__ import annotations

import copy
from concurrent.futures import Future
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import iclr_primary_warsaw_protocol_v2 as protocol_v2
import iclr_primary_warsaw_train_v2 as train_v2
from iclr_corpus import SeriesRef, Split


def _fit_spec(family: str) -> dict[str, object]:
    return next(
        row
        for row in protocol_v2.locked_fit_inventory_v2()
        if row["family"] == family
    )


_SYNTHETIC_CORPUS_SEMANTIC_SHA256 = (
    protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
)
_SYNTHETIC_MANIFEST = {
    "schema_version": 1,
    "source_commit": protocol_v2.PABULIB_COMMIT,
    "source_dir": "data/pb",
    "destination": "data/pb",
    "n_series": 0,
    "n_elections": 0,
    "n_files": 0,
    "files": [],
}
_SYNTHETIC_RECEIPT = {
    "schema_version": 2,
    "semantics_profile": "approval-set-first-occurrence-v2",
    "corpus_semantic_sha256": _SYNTHETIC_CORPUS_SEMANTIC_SHA256,
    "files": [],
}


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


_SYNTHETIC_MANIFEST_SHA256 = hashlib.sha256(
    _canonical_bytes(_SYNTHETIC_MANIFEST)
).hexdigest()
_SYNTHETIC_RECEIPT_SHA256 = hashlib.sha256(
    _canonical_bytes(_SYNTHETIC_RECEIPT)
).hexdigest()
_SYNTHETIC_SOURCE_SHA256 = hashlib.sha256(b"synthetic primary trainer\n").hexdigest()


def _synthetic_split_payload(split_name: str) -> dict[str, object]:
    return {"name": split_name, "note": "synthetic", "train": [], "test": []}


def _synthetic_split_sha256(split_name: str) -> str:
    return hashlib.sha256(_canonical_bytes(_synthetic_split_payload(split_name))).hexdigest()


def _optimizer_config_v2(spec: dict[str, object]) -> dict[str, object]:
    return {
        "sigma0": spec["sigma0"],
        "popsize": spec["popsize"],
        "generations": spec["generations"],
        "seed": spec["seed"],
        "bound": spec["bound"],
    }


def _expected_n_objective_evals_v2(spec: dict[str, object]) -> int:
    dimensions = {
        "outcome": 5,
        "endowment": 6,
        "static_age_lookup": 4,
    }[str(spec["arm"])]
    population = spec["popsize"] or (4 + int(3 * math.log(dimensions)))
    cma_evals = int(spec["generations"]) * int(population)
    return cma_evals + (1 if spec["family"] == "static_age_lookup" else 0)


def _base_protocol_lock_snapshot_v2(
    result_root: Path,
) -> protocol_v2.PrimaryProtocolLockSnapshotV2:
    """Full honest lock schema with a future grid specification."""
    tracked_files = {
        "artifact/corpus_manifest": {
            "path": "results/iclr_primary_warsaw_v2/corpus_manifest.json",
            "sha256": _SYNTHETIC_MANIFEST_SHA256,
        },
        "artifact/approval_semantics_receipt": {
            "path": "results/iclr_primary_warsaw_v2/approval_semantics_receipt.json",
            "sha256": _SYNTHETIC_RECEIPT_SHA256,
        },
        "source/iclr_primary_warsaw_train_v2.py": {
            "path": "src/iclr_primary_warsaw_train_v2.py",
            "sha256": _SYNTHETIC_SOURCE_SHA256,
        },
        **{
            f"split/{name}": {
                "path": f"results/iclr_primary_warsaw_v2/splits/{name}.json",
                "sha256": _synthetic_split_sha256(name),
            }
            for name in protocol_v2.SPLIT_SHA256_V2
        },
    }
    lock_payload = {
        "schema_version": 2,
        "lock_profile": protocol_v2.LOCK_PROFILE_V2,
        "mandatory_file_count": len(tracked_files),
        "tracked_files": tracked_files,
        "fit_inventory": protocol_v2.locked_fit_inventory_v2(),
        "training_grid_spec": copy.deepcopy(protocol_v2.TRAINING_GRID_SPEC_V2),
        "protocol": protocol_v2.protocol_config_v2(),
        "corpus_semantic_sha256": _SYNTHETIC_CORPUS_SEMANTIC_SHA256,
    }
    lock_content = (
        json.dumps(lock_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=(result_root / "protocol_lock.json").absolute(),
        payload=lock_payload,
        content=lock_content,
        sha256=hashlib.sha256(lock_content).hexdigest(),
    )


def _training_grid_payload_v2(lock_sha256: str) -> dict[str, object]:
    alphas = list(protocol_v2.STATIC_AGE_GRID_VALUES)
    return {
        "schema_version": 2,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "split": "temporal_2022",
        "training_only": True,
        "alphas": alphas,
        "train_worst_csd": [1.0 + abs(alpha - 0.1) for alpha in alphas],
        "selected_alpha": 0.1,
        "selection_rule": (
            "minimum training worst-cohort CSD; ties choose smaller alpha"
        ),
        "provenance": {
            "protocol_lock_sha256": lock_sha256,
            "corpus_manifest_sha256": _SYNTHETIC_MANIFEST_SHA256,
            "split_sha256": _synthetic_split_sha256("temporal_2022"),
            "semantics_receipt_sha256": _SYNTHETIC_RECEIPT_SHA256,
            "corpus_semantic_sha256": _SYNTHETIC_CORPUS_SEMANTIC_SHA256,
            "source_sha256": {
                "iclr_primary_warsaw_train_v2.py": _SYNTHETIC_SOURCE_SHA256
            },
        },
    }


def _valid_nonstatic_fit_payload_v2(
    spec: dict[str, object],
    lock_sha256: str,
) -> dict[str, object]:
    feature_names = [
        "deficit_per_capita",
        "deficit_normalized",
        "share_deviation",
        "ballot_length_dev",
        "cost_focus_dev",
        "overlap_dev",
    ]
    weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return {
        "schema_version": 2,
        "fit_id": spec["fit_id"],
        "config": spec,
        "training_only": True,
        "execution_mode": "serial",
        "arm": {
            "name": "endowment",
            "feature_names": feature_names,
            "init_name": spec["init"],
        },
        "provenance": {
            "protocol_lock_sha256": lock_sha256,
            "corpus_manifest_sha256": _SYNTHETIC_MANIFEST_SHA256,
            "split_sha256": _synthetic_split_sha256(str(spec["split"])),
            "semantics_receipt_sha256": _SYNTHETIC_RECEIPT_SHA256,
            "corpus_semantic_sha256": _SYNTHETIC_CORPUS_SEMANTIC_SHA256,
            "source_sha256": {
                "iclr_primary_warsaw_train_v2.py": _SYNTHETIC_SOURCE_SHA256
            },
            "grid_sha256": None,
        },
        "result": {
            "best_loss": 0.25,
            "optimizer_best_loss": 0.25,
            "optimizer_best_weights": weights,
            "selected_weights": weights.copy(),
            "selection_source": "cmaes",
            "n_objective_evals": _expected_n_objective_evals_v2(spec),
            "optimizer_config": _optimizer_config_v2(spec),
        },
    }


def _valid_static_fit_payload_v2(
    spec: dict[str, object],
    lock_sha256: str,
    grid_snapshot: train_v2.JsonSnapshotV2,
) -> dict[str, object]:
    alpha = float(grid_snapshot.payload["selected_alpha"])
    initial = [0.0, 0.0, 0.0, math.log1p(alpha)]
    return {
        "schema_version": 2,
        "fit_id": spec["fit_id"],
        "config": spec,
        "training_only": True,
        "execution_mode": "serial",
        "arm": {
            "name": "static_age_lookup",
            "feature_names": ["age<25", "age25-39", "age40-59", "age60+"],
            "init_name": spec["init"],
        },
        "provenance": {
            "protocol_lock_sha256": lock_sha256,
            "corpus_manifest_sha256": _SYNTHETIC_MANIFEST_SHA256,
            "split_sha256": _synthetic_split_sha256(str(spec["split"])),
            "semantics_receipt_sha256": _SYNTHETIC_RECEIPT_SHA256,
            "corpus_semantic_sha256": _SYNTHETIC_CORPUS_SEMANTIC_SHA256,
            "source_sha256": {
                "iclr_primary_warsaw_train_v2.py": _SYNTHETIC_SOURCE_SHA256
            },
            "grid_sha256": grid_snapshot.sha256,
        },
        "result": {
            "best_loss": 1.0,
            "optimizer_best_loss": 1.25,
            "optimizer_best_weights": [1.0, 1.0, 1.0, 1.0],
            "selected_weights": initial,
            "selection_source": "initial",
            "n_objective_evals": _expected_n_objective_evals_v2(spec),
            "optimizer_config": _optimizer_config_v2(spec),
        },
    }


def _computation_input_snapshot_v2(
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    split_name: str,
    *,
    index: dict[str, SeriesRef] | None = None,
    split: Split | None = None,
    data_dir: Path | None = None,
) -> protocol_v2.PrimaryInputsSnapshotV2:
    tracked = lock_snapshot.payload["tracked_files"]
    manifest_path = (lock_snapshot.path.parent / "corpus_manifest.json").absolute()
    receipt_path = (
        lock_snapshot.path.parent / "approval_semantics_receipt.json"
    ).absolute()
    split_path = (lock_snapshot.path.parent / "splits" / f"{split_name}.json").absolute()
    manifest_content = _canonical_bytes(_SYNTHETIC_MANIFEST)
    receipt_content = _canonical_bytes(_SYNTHETIC_RECEIPT)
    split_payload = _synthetic_split_payload(split_name)
    split_content = _canonical_bytes(split_payload)
    resolved_data_dir = (
        (lock_snapshot.path.parent / "synthetic-data").absolute()
        if data_dir is None
        else data_dir.absolute()
    )
    lock_snapshot.path.parent.mkdir(parents=True, exist_ok=True)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_data_dir.mkdir(parents=True, exist_ok=True)
    lock_snapshot.path.write_bytes(lock_snapshot.content)
    manifest_path.write_bytes(manifest_content)
    receipt_path.write_bytes(receipt_content)
    split_path.write_bytes(split_content)
    return protocol_v2.PrimaryInputsSnapshotV2(
        result_root=lock_snapshot.path.parent.absolute(),
        data_dir=resolved_data_dir,
        protocol_lock_path=lock_snapshot.path,
        protocol_lock_content=lock_snapshot.content,
        protocol_lock_sha256=lock_snapshot.sha256,
        manifest=protocol_v2.JsonArtifactSnapshotV2(
            label="artifact/corpus_manifest",
            path=manifest_path,
            content=manifest_content,
            sha256=tracked["artifact/corpus_manifest"]["sha256"],
            payload=copy.deepcopy(_SYNTHETIC_MANIFEST),
        ),
        semantics_receipt=protocol_v2.JsonArtifactSnapshotV2(
            label="artifact/approval_semantics_receipt",
            path=receipt_path,
            content=receipt_content,
            sha256=tracked["artifact/approval_semantics_receipt"]["sha256"],
            payload=copy.deepcopy(_SYNTHETIC_RECEIPT),
        ),
        split_artifacts={
            split_name: protocol_v2.JsonArtifactSnapshotV2(
                label=f"split/{split_name}",
                path=split_path,
                content=split_content,
                sha256=tracked[f"split/{split_name}"]["sha256"],
                payload=split_payload,
            )
        },
        raw_artifacts={},
        index={} if index is None else dict(index),
        splits={
            split_name: split
            if split is not None
            else Split(name=split_name, train=(), test=(), note="synthetic")
        },
    )


def test_primary_v2_training_loader_opens_only_split_train_years(
    tmp_path: Path,
) -> None:
    semantics_v2 = __import__("iclr_approval_semantics_v2")
    train_path = tmp_path / "train.pb"
    train_path.write_text(
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
        "37051;a,a,b\n",
        encoding="utf-8",
    )
    heldout_path = tmp_path / "heldout-must-not-open.pb"
    ref = SeriesRef(
        key="Poland/Warszawa/Wola",
        years=(2021, 2023),
        paths=(train_path, heldout_path),
    )
    split = Split(
        name="temporal_2022",
        train=((ref.key, (2021,)),),
        test=((ref.key, (2023,)),),
    )
    normalized = semantics_v2.parse_pb_file_v2(train_path)
    receipt = {
        "semantics_profile": semantics_v2.SEMANTICS_PROFILE,
        "files": [
            {
                "series": ref.key,
                "year": 2021,
                "name": train_path.name,
                "raw_file_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "v2_semantic_sha256": semantics_v2.canonical_instance_sha256(
                    normalized
                ),
            }
        ],
    }

    rows = train_v2.load_training_series_data_v2(split, {ref.key: ref}, receipt)

    assert len(rows) == 1
    assert rows[0].train_years == (2021,)
    assert rows[0].test_years == ()
    assert rows[0].ref.years == (2021,)
    assert rows[0].ref.paths == (train_path,)
    assert tuple(rows[0].train_only) == (2021,)
    assert tuple(rows[0].all_years) == (2021,)
    assert rows[0].all_years[2021].votes[0].projects == ("a", "b")
    assert not heldout_path.exists()

    unavailable = Split(
        name="temporal_2022",
        train=((ref.key, (2022,)),),
        test=(),
    )
    with pytest.raises(RuntimeError, match="unavailable year|split.*year"):
        train_v2.load_training_series_data_v2(unavailable, {ref.key: ref}, receipt)

    missing_receipt = copy.deepcopy(receipt)
    missing_receipt["files"] = []
    with pytest.raises(RuntimeError, match="receipt|identity"):
        train_v2.load_training_series_data_v2(split, {ref.key: ref}, missing_receipt)

    drifted_ref = SeriesRef(
        key="Poland/Warszawa/Other",
        years=ref.years,
        paths=ref.paths,
    )
    with pytest.raises(RuntimeError, match="receipt|identity|series"):
        train_v2.load_training_series_data_v2(
            split,
            {ref.key: drifted_ref},
            receipt,
        )


def test_primary_v2_authorization_rejects_coordinate_or_lock_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    result_root.mkdir(parents=True)
    lock_path = result_root / "protocol_lock.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    inventory = protocol_v2.locked_fit_inventory_v2()
    requested = copy.deepcopy(inventory[0])
    lock_payload = {
        "schema_version": 2,
        "lock_profile": "primary-warsaw-corrected-post-hoc-v2",
        "fit_inventory": inventory,
    }
    lock_bytes = b"{}\n"
    snapshot = SimpleNamespace(
        path=lock_path,
        payload=lock_payload,
        content=lock_bytes,
        sha256=hashlib.sha256(lock_bytes).hexdigest(),
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_verify(*args, **kwargs):
        calls.append((args, kwargs))
        return snapshot

    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        fake_verify,
    )

    assert train_v2.authorize_fit_v2(
        requested,
        smoke=False,
        result_root=result_root,
        repo_root=tmp_path,
        data_dir=tmp_path / "data" / "pb",
    ) is snapshot
    assert calls == [
        (
            (
                lock_path,
                tmp_path,
                result_root,
                tmp_path / "data" / "pb",
            ),
            {},
        )
    ]

    changed = copy.deepcopy(requested)
    changed["generations"] = int(changed["generations"]) + 1
    with pytest.raises(RuntimeError, match="locked.*inventory|authorization"):
        train_v2.authorize_fit_v2(
            changed,
            smoke=False,
            result_root=result_root,
            repo_root=tmp_path,
            data_dir=tmp_path / "data" / "pb",
        )

    aliased = copy.deepcopy(requested)
    aliased["seed"] = True
    with pytest.raises(RuntimeError, match="seed|locked.*inventory|authorization"):
        train_v2.authorize_fit_v2(
            aliased,
            smoke=False,
            result_root=result_root,
            repo_root=tmp_path,
            data_dir=tmp_path / "data" / "pb",
        )

    lock_payload["schema_version"] = 2.0
    with pytest.raises(RuntimeError, match="lock profile|schema"):
        train_v2.authorize_fit_v2(
            requested,
            smoke=False,
            result_root=result_root,
            repo_root=tmp_path,
            data_dir=tmp_path / "data" / "pb",
        )


def test_static_age_fit_authenticates_the_training_grid_dependency(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    grid_path = protocol_v2.canonical_grid_path_v2(result_root)
    grid_path.parent.mkdir(parents=True)
    lock_snapshot = _base_protocol_lock_snapshot_v2(result_root)
    grid_payload = _training_grid_payload_v2(lock_snapshot.sha256)
    grid_bytes = (
        json.dumps(grid_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    grid_path.write_bytes(grid_bytes)
    grid_sha256 = hashlib.sha256(grid_bytes).hexdigest()
    static_age = _fit_spec("static_age_lookup")

    snapshot = train_v2.load_static_age_grid_dependency_v2(
        static_age,
        lock_snapshot,
        result_root=result_root,
    )

    assert "artifact/static_senior_alpha_grid" not in (
        lock_snapshot.payload["tracked_files"]
    )
    assert snapshot.path == grid_path
    assert snapshot.sha256 == grid_sha256
    assert snapshot.payload == grid_payload
    assert snapshot.payload["selected_alpha"] == 0.1
    assert snapshot.payload["provenance"]["protocol_lock_sha256"] == (
        lock_snapshot.sha256
    )

    wrong_lock = copy.deepcopy(grid_payload)
    wrong_lock["provenance"]["protocol_lock_sha256"] = "0" * 64
    grid_path.write_bytes(
        (
            json.dumps(wrong_lock, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    )
    with pytest.raises(RuntimeError, match="grid.*lock|protocol.*digest"):
        train_v2.load_static_age_grid_dependency_v2(
            static_age,
            lock_snapshot,
            result_root=result_root,
        )


def test_static_age_grid_rejects_heldout_fields_and_wrong_grid_values(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    grid_path = protocol_v2.canonical_grid_path_v2(result_root)
    grid_path.parent.mkdir(parents=True)
    lock_snapshot = _base_protocol_lock_snapshot_v2(result_root)
    payload = _training_grid_payload_v2(lock_snapshot.sha256)
    payload["heldout_worst_csd"] = 0.1
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    grid_path.write_bytes(content)

    with pytest.raises(RuntimeError, match="training-only grid|schema"):
        train_v2.load_static_age_grid_dependency_v2(
            _fit_spec("static_age_lookup"),
            lock_snapshot,
            result_root=result_root,
        )

    payload.pop("heldout_worst_csd")
    payload["alphas"][1] = 0.051
    changed = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    grid_path.write_bytes(changed)
    with pytest.raises(RuntimeError, match="61-value|alpha grid"):
        train_v2.load_static_age_grid_dependency_v2(
            _fit_spec("static_age_lookup"),
            lock_snapshot,
            result_root=result_root,
        )

    payload = _training_grid_payload_v2(lock_snapshot.sha256)
    payload["train_worst_csd"][0] = -0.01
    payload["selected_alpha"] = 0.0
    grid_path.write_bytes(_canonical_bytes(payload))
    with pytest.raises(RuntimeError, match="nonnegative|grid losses"):
        train_v2.load_static_age_grid_dependency_v2(
            _fit_spec("static_age_lookup"),
            lock_snapshot,
            result_root=result_root,
        )


def test_primary_v2_fit_schema_is_training_only_exact_and_deterministic() -> None:
    spec = _fit_spec("outcome_loo")
    feature_names = [
        "approval_share",
        "approvals_per_cost",
        "cost_share",
        "deficit_weighted",
        "cohort_concentration",
    ]
    masked_name = str(spec["masked_feature_names"][0])
    masked_index = feature_names.index(masked_name)
    optimizer_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    selected_weights = optimizer_weights.copy()
    selected_weights[masked_index] = 0.0
    lock_sha256 = "a" * 64
    payload = {
        "schema_version": 2,
        "fit_id": spec["fit_id"],
        "config": spec,
        "training_only": True,
        "execution_mode": "serial",
        "arm": {
            "name": "outcome",
            "feature_names": feature_names,
            "init_name": "res",
        },
        "provenance": {
            "protocol_lock_sha256": lock_sha256,
            "corpus_manifest_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "semantics_receipt_sha256": "d" * 64,
            "corpus_semantic_sha256": "e" * 64,
            "source_sha256": {"iclr_primary_warsaw_train_v2.py": "f" * 64},
            "grid_sha256": None,
        },
        "result": {
            "best_loss": 0.125,
            "optimizer_best_loss": 0.125,
            "optimizer_best_weights": optimizer_weights,
            "selected_weights": selected_weights,
            "selection_source": "cmaes",
            "n_objective_evals": 200,
            "optimizer_config": _optimizer_config_v2(spec),
        },
    }
    assert train_v2.validate_fit_payload_v2(
        payload,
        expected_spec=spec,
        expected_protocol_lock_sha256=lock_sha256,
    ) is payload
    expected_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    reordered = dict(reversed(list(payload.items())))
    assert train_v2.serialize_fit_payload_v2(payload) == expected_bytes
    assert train_v2.serialize_fit_payload_v2(reordered) == expected_bytes
    assert b"elapsed" not in expected_bytes
    assert b"heldout" not in expected_bytes

    leaked = copy.deepcopy(payload)
    leaked["heldout"] = {"worst_csd": 0.1}
    with pytest.raises(RuntimeError, match="fit schema|held-out|training-only"):
        train_v2.validate_fit_payload_v2(
            leaked,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_sha256,
        )

    wrong_mask = copy.deepcopy(payload)
    wrong_mask["result"]["selected_weights"][masked_index] = 1.0
    with pytest.raises(RuntimeError, match="masked|selected weights"):
        train_v2.validate_fit_payload_v2(
            wrong_mask,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_sha256,
        )

    nonfinite = copy.deepcopy(payload)
    nonfinite["result"]["best_loss"] = float("inf")
    with pytest.raises(RuntimeError, match="finite"):
        train_v2.validate_fit_payload_v2(
            nonfinite,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_sha256,
        )

    negative = copy.deepcopy(payload)
    negative["result"]["best_loss"] = -0.01
    negative["result"]["optimizer_best_loss"] = -0.01
    with pytest.raises(RuntimeError, match="nonnegative|loss"):
        train_v2.validate_fit_payload_v2(
            negative,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_sha256,
        )

    wrong_features = copy.deepcopy(payload)
    wrong_features["arm"]["feature_names"] = list(reversed(feature_names))
    with pytest.raises(RuntimeError, match="feature.*name|arm.*schema"):
        train_v2.validate_fit_payload_v2(
            wrong_features,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_sha256,
        )

    zero_evals = copy.deepcopy(payload)
    zero_evals["result"]["n_objective_evals"] = 0
    with pytest.raises(RuntimeError, match="objective.*eval|positive"):
        train_v2.validate_fit_payload_v2(
            zero_evals,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_sha256,
        )

    leaked_history = copy.deepcopy(payload)
    leaked_history["result"]["history"] = [{"heldout": {"score": 0.1}}]
    with pytest.raises(RuntimeError, match="history|fit result schema|held-out"):
        train_v2.validate_fit_payload_v2(
            leaked_history,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_sha256,
        )


def test_primary_v2_recipe_dispatch_masks_grid_and_static_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    by_id = {
        str(row["fit_id"]): row for row in protocol_v2.locked_fit_inventory_v2()
    }
    outcome_names = [
        "approval_share",
        "approvals_per_cost",
        "cost_share",
        "deficit_weighted",
        "cohort_concentration",
    ]
    endowment_names = [
        "deficit_per_capita",
        "deficit_normalized",
        "share_deviation",
        "ballot_length_dev",
        "cost_focus_dev",
        "overlap_dev",
    ]
    age_names = ["age<25", "age25-39", "age40-59", "age60+"]
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid_content = (
        json.dumps(grid_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        payload=grid_payload,
        content=grid_content,
        sha256=hashlib.sha256(grid_content).hexdigest(),
    )
    cases = {
        "outcome_frontier/temporal_2022/target-1/seed-1": (
            outcome_names, [0.0, 1.0, 0.0, 0.0, 0.0], [], 1.0, 2.0, None
        ),
        "endowment_frontier/temporal_2022/target-1/seed-42": (
            endowment_names, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [], 1.0, 2.0, None
        ),
        "temporal_endowment/temporal_2022/seed-42": (
            endowment_names, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [], 0.0, 0.0, None
        ),
        "district_endowment/district_out_f0of5/bound-10/seed-42": (
            endowment_names, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [], 0.0, 0.0, None
        ),
        "outcome_loo/temporal_2022/mask-approval_share/seed-42": (
            outcome_names, [0.0, 1.0, 0.0, 0.0, 0.0], [0], 1.0, 2.0, None
        ),
        "static_support_floor/temporal_2022/kappa-1/seed-42": (
            outcome_names, [0.0, 1.0, 0.0, 0.0, 0.0], [], 0.0, 0.0, 1.0
        ),
        "history_free/temporal_2022/seed-42": (
            endowment_names, [0.0] * 6, [0, 1], 0.0, 0.0, None
        ),
        "static_age_lookup/temporal_2022/seed-42": (
            age_names, [0.0, 0.0, 0.0, math.log1p(0.1)], [], 0.0, 0.0, None
        ),
    }
    recipes = {}
    for fit_id, expected in cases.items():
        spec = by_id[fit_id]
        recipe = train_v2.build_fit_recipe_v2(
            spec,
            grid_snapshot=grid if spec["family"] == "static_age_lookup" else None,
        )
        names, initial, masked, target, penalty, floor = expected
        assert recipe == {
            "family": spec["family"],
            "arm": spec["arm"],
            "feature_names": names,
            "init_name": spec["init"],
            "initial_weights": initial,
            "masked_feature_indices": masked,
            "soft_welfare_target": target,
            "welfare_penalty": penalty,
            "support_floor_kappa": floor,
            "grid_sha256": grid.sha256 if spec["family"] == "static_age_lookup" else None,
            "optimizer_config": {
                "seed": spec["seed"],
                "generations": spec["generations"],
                "popsize": spec["popsize"],
                "sigma0": spec["sigma0"],
                "bound": spec["bound"],
            },
        }
        recipes[str(spec["family"])] = recipe

    observed: list[tuple[float, ...]] = []

    def objective(weights):
        values = tuple(float(value) for value in weights)
        observed.append(values)
        return -sum(abs(value) for value in values)

    def fake_minimize(wrapped, initial, config):
        candidate = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0][: len(initial)]
        return SimpleNamespace(
            best_x=candidate,
            best_f=wrapped(candidate),
            n_evals=1,
            history=[],
        )

    monkeypatch.setattr(train_v2, "minimize", fake_minimize, raising=False)
    for family, zero_indices in (("outcome_loo", (0,)), ("history_free", (0, 1))):
        observed.clear()
        result = train_v2.optimize_fit_recipe_v2(
            recipes[family],
            objective,
        )
        assert observed and all(
            all(values[index] == 0.0 for index in zero_indices)
            for values in observed
        )
        assert all(result["selected_weights"][index] == 0.0 for index in zero_indices)

    with pytest.raises((TypeError, RuntimeError), match="CMA|config|argument"):
        train_v2.optimize_fit_recipe_v2(
            recipes["history_free"],
            objective,
            cma_config=train_v2.CMAESConfig(generations=1),
        )

    input_snapshot = _computation_input_snapshot_v2(lock, "temporal_2022")
    seen_initializers: list[tuple[float, ...]] = []
    training_rows = object()

    monkeypatch.setattr(
        train_v2,
        "load_training_series_data_v2",
        lambda split, index, receipt: training_rows,
    )

    def static_policy(weights):
        vector = tuple(float(value) for value in weights)
        seen_initializers.append(vector)
        return vector

    monkeypatch.setattr(train_v2, "_static_age_policy_v2", static_policy)
    monkeypatch.setattr(
        train_v2,
        "endowment_selector",
        lambda policy, env_config: policy,
    )

    def grid_training_loss(selector, rows, env_config):
        assert rows is training_rows
        alpha = protocol_v2.STATIC_AGE_GRID_VALUES[len(seen_initializers) - 1]
        assert selector[3] == math.log1p(alpha)
        return float(1.0 + abs(alpha - 0.1))

    monkeypatch.setattr(train_v2, "_training_worst_csd_v2", grid_training_loss)
    built = train_v2.build_static_age_grid_from_snapshot_v2(
        input_snapshot=input_snapshot,
        authorized_lock_snapshot=lock,
    )
    assert seen_initializers == [
        (0.0, 0.0, 0.0, math.log1p(alpha))
        for alpha in protocol_v2.STATIC_AGE_GRID_VALUES
    ]
    assert built == grid_payload
    initial = [0.0, 0.0, 0.0, math.log1p(0.1)]
    for optimizer_loss in (0.25, 0.5):
        selected = train_v2.select_best_with_initial_v2(
            initial,
            0.25,
            SimpleNamespace(best_x=[1.0] * 4, best_f=optimizer_loss),
        )
        assert selected == (initial, 0.25, "initial")


def test_primary_v2_immutable_fit_write_revalidates_lock_and_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    spec = _fit_spec("static_age_lookup")
    lock = _base_protocol_lock_snapshot_v2(result_root)
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid_content = (
        json.dumps(grid_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        payload=grid_payload,
        content=grid_content,
        sha256=hashlib.sha256(grid_content).hexdigest(),
    )
    initial = [0.0, 0.0, 0.0, math.log1p(0.1)]
    payload = {
        "schema_version": 2,
        "fit_id": spec["fit_id"],
        "config": spec,
        "training_only": True,
        "execution_mode": "serial",
        "arm": {
            "name": "static_age_lookup",
            "feature_names": ["age<25", "age25-39", "age40-59", "age60+"],
            "init_name": "training_grid",
        },
        "provenance": {
            "protocol_lock_sha256": lock.sha256,
            "corpus_manifest_sha256": _SYNTHETIC_MANIFEST_SHA256,
            "split_sha256": _synthetic_split_sha256("temporal_2022"),
            "semantics_receipt_sha256": _SYNTHETIC_RECEIPT_SHA256,
            "corpus_semantic_sha256": _SYNTHETIC_CORPUS_SEMANTIC_SHA256,
            "source_sha256": {
                "iclr_primary_warsaw_train_v2.py": _SYNTHETIC_SOURCE_SHA256
            },
            "grid_sha256": grid.sha256,
        },
        "result": {
            "best_loss": 1.0,
            "optimizer_best_loss": 1.25,
            "optimizer_best_weights": [1.0] * 4,
            "selected_weights": initial,
            "selection_source": "initial",
            "n_objective_evals": _expected_n_objective_evals_v2(spec),
            "optimizer_config": _optimizer_config_v2(spec),
        },
    }
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        str(spec["split"]),
        data_dir=data_dir,
    )
    events = ["compute"]
    current_grid = [grid]

    def verify(*args):
        events.append("lock")
        assert args == (lock.path, repo_root, result_root, data_dir)
        return lock

    def reload_grid(*args, **kwargs):
        events.append("grid")
        return current_grid[0]

    def rebuild_grid(**kwargs):
        events.append("recompute")
        assert kwargs == {
            "input_snapshot": input_snapshot,
            "authorized_lock_snapshot": lock,
        }
        return grid_payload

    def immutable_write(path, content, **kwargs):
        events.append("write")
        assert content == train_v2.serialize_fit_payload_v2(payload)
        return hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(
        train_v2, "verify_primary_warsaw_protocol_lock_snapshot_v2", verify
    )
    monkeypatch.setattr(train_v2, "load_static_age_grid_dependency_v2", reload_grid)
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        rebuild_grid,
    )
    monkeypatch.setattr(
        train_v2, "assert_protocol_inputs_unchanged_v2", lambda snapshot, current_lock: None,
        raising=False,
    )
    monkeypatch.setattr(
        train_v2, "write_immutable_bytes_artifact_v2", immutable_write, raising=False
    )
    output = protocol_v2.canonical_fit_path_v2(result_root, spec)
    train_v2.write_immutable_fit_v2(
        output,
        payload,
        expected_spec=spec,
        authorized_lock_snapshot=lock,
        input_snapshot=input_snapshot,
        grid_snapshot=grid,
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
    )
    assert events == ["compute", "recompute", "lock", "grid", "write"]

    current_grid[0] = SimpleNamespace(
        path=grid.path,
        payload=grid.payload,
        content=grid.content + b"\n",
        sha256="9" * 64,
    )
    events[:] = ["compute"]
    with pytest.raises(RuntimeError, match="grid.*drift|grid.*changed"):
        train_v2.write_immutable_fit_v2(
            output,
            payload,
            expected_spec=spec,
            authorized_lock_snapshot=lock,
            input_snapshot=input_snapshot,
            grid_snapshot=grid,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )
    assert events == ["compute", "recompute", "lock", "grid"]


def test_primary_v2_fit_one_orchestrates_locked_families_without_test_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_ids = (
        "outcome_frontier/temporal_2022/target-1/seed-1",
        "endowment_frontier/temporal_2022/target-1/seed-42",
        "temporal_endowment/temporal_2022/seed-42",
        "district_endowment/district_out_f0of5/bound-10/seed-42",
        "outcome_loo/temporal_2022/mask-approval_share/seed-42",
        "static_support_floor/temporal_2022/kappa-1/seed-42",
        "history_free/temporal_2022/seed-42",
        "static_age_lookup/temporal_2022/seed-42",
    )
    inventory = {
        str(row["fit_id"]): row for row in protocol_v2.locked_fit_inventory_v2()
    }
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid_content = (
        json.dumps(grid_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        payload=grid_payload,
        content=grid_content,
        sha256=hashlib.sha256(grid_content).hexdigest(),
    )
    class TrainingRow:
        train_years = (2021,)
        train_only = {2021: object()}

        @property
        def test_years(self):
            pytest.fail("fit_one_v2 must not inspect test years")

        @property
        def all_years(self):
            pytest.fail("fit_one_v2 must not inspect all-year data")

    data = [TrainingRow()]
    dispatches = []
    cma_configs = {}
    load_calls = []

    def fake_objective_builder(recipe, rows, env_config):
        assert rows is data
        assert rows[0].train_years == (2021,)
        assert set(rows[0].train_only) == {2021}
        dispatches.append(
            (
                recipe["family"],
                recipe["arm"],
                tuple(recipe["masked_feature_indices"]),
                recipe["soft_welfare_target"],
                recipe["welfare_penalty"],
                recipe["support_floor_kappa"],
            )
        )
        value = 1.0 if recipe["family"] == "static_age_lookup" else 0.25
        return lambda weights: value

    def fake_minimize(objective, initial, config):
        candidate = [1.0] * len(initial)
        objective(candidate)
        cma_configs[len(cma_configs)] = (
            config.seed,
            config.generations,
            config.popsize,
            config.sigma0,
            config.bound,
        )
        return SimpleNamespace(
            best_x=candidate,
            best_f=1.25 if len(initial) == 4 else 0.25,
            n_evals=(
                config.generations
                * (config.popsize or (4 + int(3 * math.log(len(initial)))))
            ),
            history=[],
        )

    monkeypatch.setattr(
        train_v2,
        "build_training_objective_v2",
        fake_objective_builder,
        raising=False,
    )
    monkeypatch.setattr(train_v2, "minimize", fake_minimize)
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        lambda **kwargs: grid_payload,
    )
    def fake_training_loader(split, index, receipt):
        assert split.test == ()
        load_calls.append((split.name, index, receipt))
        return data

    monkeypatch.setattr(
        train_v2, "load_training_series_data_v2", fake_training_loader
    )
    payloads = []
    for fit_id in fit_ids:
        spec = inventory[fit_id]
        split_name = str(spec["split"])
        input_snapshot = _computation_input_snapshot_v2(lock, split_name)
        payload = train_v2.fit_one_v2(
            spec,
            input_snapshot=input_snapshot,
            authorized_lock_snapshot=lock,
            grid_snapshot=(
                grid if spec["family"] == "static_age_lookup" else None
            ),
        )
        payloads.append(payload)
        assert payload["fit_id"] == fit_id
        assert payload["config"] == spec
        assert payload["training_only"] is True
        assert payload["provenance"] == {
            "protocol_lock_sha256": lock.sha256,
            "corpus_manifest_sha256": _SYNTHETIC_MANIFEST_SHA256,
            "split_sha256": _synthetic_split_sha256(split_name),
            "semantics_receipt_sha256": _SYNTHETIC_RECEIPT_SHA256,
            "corpus_semantic_sha256": _SYNTHETIC_CORPUS_SEMANTIC_SHA256,
            "source_sha256": {
                "iclr_primary_warsaw_train_v2.py": _SYNTHETIC_SOURCE_SHA256
            },
            "grid_sha256": (
                grid.sha256 if spec["family"] == "static_age_lookup" else None
            ),
        }
        expected_features = (
            ["age<25", "age25-39", "age40-59", "age60+"]
            if spec["family"] == "static_age_lookup"
            else (
                [
                    "approval_share", "approvals_per_cost", "cost_share",
                    "deficit_weighted", "cohort_concentration",
                ]
                if spec["arm"] == "outcome"
                else [
                    "deficit_per_capita", "deficit_normalized", "share_deviation",
                    "ballot_length_dev", "cost_focus_dev", "overlap_dev",
                ]
            )
        )
        assert payload["arm"]["feature_names"] == expected_features
        assert payload["result"]["n_objective_evals"] > 0
        assert payload["result"]["selection_source"] == (
            "initial" if spec["family"] == "static_age_lookup" else "cmaes"
        )
        assert "history" not in payload["result"]

    assert [row[0] for row in dispatches] == [
        str(inventory[fit_id]["family"]) for fit_id in fit_ids
    ]
    assert [call[0] for call in load_calls] == [
        str(inventory[fit_id]["split"]) for fit_id in fit_ids
    ]
    assert list(cma_configs.values()) == [
        (
            int(inventory[fit_id]["seed"]),
            int(inventory[fit_id]["generations"]),
            inventory[fit_id]["popsize"],
            float(inventory[fit_id]["sigma0"]),
            float(inventory[fit_id]["bound"]),
        )
        for fit_id in fit_ids
    ]
    static_payload = payloads[-1]
    assert static_payload["result"]["selected_weights"] == [
        0.0,
        0.0,
        0.0,
        math.log1p(0.1),
    ]
    assert static_payload["result"]["selection_source"] == "initial"
    assert static_payload["result"]["best_loss"] == 1.0
    assert static_payload["result"]["optimizer_best_loss"] == 1.25
    assert train_v2.serialize_fit_payload_v2(static_payload) == (
        train_v2.serialize_fit_payload_v2(copy.deepcopy(static_payload))
    )


def test_primary_v2_grid_write_revalidates_lock_before_immutable_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    payload = _training_grid_payload_v2(lock.sha256)
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    output = protocol_v2.canonical_grid_path_v2(result_root)
    events = ["compute"]
    current_lock = [lock]

    def verify(*args):
        events.append("lock")
        assert args == (lock.path, repo_root, result_root, data_dir)
        return current_lock[0]

    def immutable_write(path, content, **kwargs):
        events.append("write")
        assert path == output
        assert json.loads(content) == payload
        return hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(
        train_v2, "verify_primary_warsaw_protocol_lock_snapshot_v2", verify
    )
    monkeypatch.setattr(
        train_v2, "write_immutable_bytes_artifact_v2", immutable_write
    )
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        lambda **kwargs: payload,
    )
    monkeypatch.setattr(
        train_v2, "assert_protocol_inputs_unchanged_v2", lambda snapshot, current_lock: None,
        raising=False,
    )
    train_v2.write_static_age_grid_v2(
        output,
        payload,
        authorized_lock_snapshot=lock,
        input_snapshot=input_snapshot,
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
    )
    assert events == ["compute", "lock", "write"]

    current_lock[0] = protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=lock.path,
        payload=lock.payload,
        content=lock.content + b"\n",
        sha256="9" * 64,
    )
    events[:] = ["compute"]
    with pytest.raises(RuntimeError, match="lock.*changed|lock.*drift"):
        train_v2.write_static_age_grid_v2(
            output,
            payload,
            authorized_lock_snapshot=lock,
            input_snapshot=input_snapshot,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )
    assert events == ["compute", "lock"]


def test_primary_v2_fit_write_rejects_spec_absent_from_authorized_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    spec = _fit_spec("temporal_endowment")
    base = _base_protocol_lock_snapshot_v2(result_root)
    lock_payload = copy.deepcopy(base.payload)
    lock_payload["fit_inventory"] = [
        row
        for row in lock_payload["fit_inventory"]
        if row["fit_id"] != spec["fit_id"]
    ]
    lock_content = (
        json.dumps(lock_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    unauthorized_lock = protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=base.path,
        payload=lock_payload,
        content=lock_content,
        sha256=hashlib.sha256(lock_content).hexdigest(),
    )
    payload = _valid_nonstatic_fit_payload_v2(spec, unauthorized_lock.sha256)
    input_snapshot = _computation_input_snapshot_v2(
        unauthorized_lock,
        str(spec["split"]),
        data_dir=data_dir,
    )
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: unauthorized_lock,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        train_v2, "assert_protocol_inputs_unchanged_v2", lambda snapshot, current_lock: None,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="lock.*inventory|authorized.*spec"):
        train_v2.write_immutable_fit_v2(
            protocol_v2.canonical_fit_path_v2(result_root, spec),
            payload,
            expected_spec=spec,
            authorized_lock_snapshot=unauthorized_lock,
            input_snapshot=input_snapshot,
            grid_snapshot=None,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )


def test_primary_v2_fit_and_grid_provenance_equal_lock_tracked_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    spec = _fit_spec("temporal_endowment")
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        str(spec["split"]),
        data_dir=data_dir,
    )
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        train_v2, "assert_protocol_inputs_unchanged_v2", lambda snapshot, current_lock: None,
        raising=False,
    )
    mutations = {
        "corpus_manifest_sha256": "0" * 64,
        "split_sha256": "1" * 64,
        "semantics_receipt_sha256": "2" * 64,
        "corpus_semantic_sha256": "3" * 64,
        "source_sha256": {"iclr_primary_warsaw_train_v2.py": "4" * 64},
    }
    for field, wrong_value in mutations.items():
        fit_payload = _valid_nonstatic_fit_payload_v2(spec, lock.sha256)
        fit_payload["provenance"][field] = wrong_value
        with pytest.raises(RuntimeError, match="provenance|tracked.*digest|lock"):
            train_v2.write_immutable_fit_v2(
                protocol_v2.canonical_fit_path_v2(result_root, spec),
                fit_payload,
                expected_spec=spec,
                authorized_lock_snapshot=lock,
                input_snapshot=input_snapshot,
                grid_snapshot=None,
                repo_root=repo_root,
                result_root=result_root,
                data_dir=data_dir,
            )

        grid_payload = _training_grid_payload_v2(lock.sha256)
        grid_payload["provenance"][field] = wrong_value
        with pytest.raises(RuntimeError, match="provenance|tracked.*digest|lock"):
            train_v2.write_static_age_grid_v2(
                protocol_v2.canonical_grid_path_v2(result_root),
                grid_payload,
                authorized_lock_snapshot=lock,
                input_snapshot=input_snapshot,
                repo_root=repo_root,
                result_root=result_root,
                data_dir=data_dir,
            )


def test_primary_v2_fit_one_rejects_caller_injected_data_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    spec = _fit_spec("temporal_endowment")
    lock = _base_protocol_lock_snapshot_v2(result_root)
    input_snapshot = _computation_input_snapshot_v2(lock, str(spec["split"]))
    forged_row = SimpleNamespace(
        ref=SeriesRef(
            key="Poland/Warszawa/Forged",
            years=(2021,),
            paths=(tmp_path / "forged.pb",),
        ),
        train_years=(2021,),
        train_only={2021: object()},
    )

    monkeypatch.setattr(
        train_v2,
        "build_training_objective_v2",
        lambda *args: (lambda weights: 0.25),
    )
    monkeypatch.setattr(
        train_v2,
        "minimize",
        lambda objective, initial, config: SimpleNamespace(
            best_x=list(initial), best_f=objective(initial), n_evals=1, history=[]
        ),
    )
    monkeypatch.setattr(
        train_v2,
        "load_training_series_data_v2",
        lambda split, index, receipt: [],
    )
    with pytest.raises(TypeError, match="argument|positional|source_sha256"):
        train_v2.fit_one_v2(
            spec,
            [forged_row],
            input_snapshot=input_snapshot,
            authorized_lock_snapshot=lock,
            source_sha256={
                "iclr_primary_warsaw_train_v2.py": _SYNTHETIC_SOURCE_SHA256
            },
        )

    other_payload = copy.deepcopy(lock.payload)
    other_payload["corpus_semantic_sha256"] = "a" * 64
    other_content = _canonical_bytes(other_payload)
    cross_lock_snapshot = replace(
        input_snapshot,
        protocol_lock_content=other_content,
        protocol_lock_sha256=hashlib.sha256(other_content).hexdigest(),
    )
    with pytest.raises(RuntimeError, match="input snapshot|protocol lock|cross-lock"):
        train_v2.fit_one_v2(
            spec,
            input_snapshot=cross_lock_snapshot,
            authorized_lock_snapshot=lock,
        )


def test_primary_v2_writers_assert_exact_computation_inputs_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    spec = _fit_spec("temporal_endowment")
    fit_payload = _valid_nonstatic_fit_payload_v2(spec, lock.sha256)
    grid_payload = _training_grid_payload_v2(lock.sha256)
    events: list[str] = []

    def verify(*args):
        events.append("lock")
        return lock

    def inputs_unchanged(snapshot, current_lock):
        assert snapshot is input_snapshot
        assert current_lock is lock
        assert snapshot.protocol_lock_sha256 == current_lock.sha256
        assert snapshot.protocol_lock_content == current_lock.content
        events.append("inputs")

    def install(*args, **kwargs):
        events.append("write")
        return "a" * 64

    monkeypatch.setattr(
        train_v2, "verify_primary_warsaw_protocol_lock_snapshot_v2", verify
    )
    monkeypatch.setattr(
        train_v2, "assert_protocol_inputs_unchanged_v2", inputs_unchanged,
        raising=False,
    )
    monkeypatch.setattr(train_v2, "write_immutable_bytes_artifact_v2", install)
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        lambda **kwargs: grid_payload,
    )

    train_v2.write_immutable_fit_v2(
        protocol_v2.canonical_fit_path_v2(result_root, spec),
        fit_payload,
        expected_spec=spec,
        authorized_lock_snapshot=lock,
        input_snapshot=input_snapshot,
        grid_snapshot=None,
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
    )
    assert events == ["lock", "inputs", "write"]

    events.clear()
    train_v2.write_static_age_grid_v2(
        protocol_v2.canonical_grid_path_v2(result_root),
        grid_payload,
        authorized_lock_snapshot=lock,
        input_snapshot=input_snapshot,
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
    )
    assert events == ["lock", "inputs", "write"]

    def reject_drift(snapshot, current_lock):
        assert current_lock is lock
        events.append("inputs")
        raise RuntimeError("primary-v2 input changed after computation")

    monkeypatch.setattr(
        train_v2, "assert_protocol_inputs_unchanged_v2", reject_drift,
        raising=False,
    )
    for writer, path, payload, kwargs in (
        (
            train_v2.write_immutable_fit_v2,
            protocol_v2.canonical_fit_path_v2(result_root, spec),
            fit_payload,
            {"expected_spec": spec, "grid_snapshot": None},
        ),
        (
            train_v2.write_static_age_grid_v2,
            protocol_v2.canonical_grid_path_v2(result_root),
            grid_payload,
            {},
        ),
    ):
        events.clear()
        with pytest.raises(RuntimeError, match="input changed"):
            writer(
                path,
                payload,
                authorized_lock_snapshot=lock,
                input_snapshot=input_snapshot,
                repo_root=repo_root,
                result_root=result_root,
                data_dir=data_dir,
                **kwargs,
            )
        assert "write" not in events


@pytest.mark.parametrize("writer_kind", ("fit", "grid"))
def test_primary_v2_writers_require_exact_computation_split_scope(
    writer_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    fit_spec = _fit_spec("temporal_endowment")
    temporal = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    extra = _computation_input_snapshot_v2(
        lock,
        "district_out_f0of5",
        data_dir=data_dir,
    )
    overbroad = replace(
        temporal,
        split_artifacts={
            **temporal.split_artifacts,
            **extra.split_artifacts,
        },
        splits={**temporal.splits, **extra.splits},
    )
    installed: list[Path] = []
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda path, *args, **kwargs: installed.append(path) or "a" * 64,
    )

    with pytest.raises(RuntimeError, match="snapshot.*wrong split|exact.*split"):
        if writer_kind == "fit":
            train_v2.write_immutable_fit_v2(
                protocol_v2.canonical_fit_path_v2(result_root, fit_spec),
                _valid_nonstatic_fit_payload_v2(fit_spec, lock.sha256),
                expected_spec=fit_spec,
                authorized_lock_snapshot=lock,
                input_snapshot=overbroad,
                grid_snapshot=None,
                repo_root=repo_root,
                result_root=result_root,
                data_dir=data_dir,
            )
        else:
            train_v2.write_static_age_grid_v2(
                protocol_v2.canonical_grid_path_v2(result_root),
                _training_grid_payload_v2(lock.sha256),
                authorized_lock_snapshot=lock,
                input_snapshot=overbroad,
                repo_root=repo_root,
                result_root=result_root,
                data_dir=data_dir,
            )
    assert installed == []


def test_primary_v2_grid_writer_recomputes_authenticated_losses_before_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    handcrafted = _training_grid_payload_v2(lock.sha256)
    recomputed = copy.deepcopy(handcrafted)
    recomputed["train_worst_csd"][-1] += 0.125
    recompute_calls: list[tuple[object, object]] = []
    installed: list[Path] = []

    def rebuild(*, input_snapshot, authorized_lock_snapshot):
        recompute_calls.append((input_snapshot, authorized_lock_snapshot))
        return recomputed

    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        rebuild,
    )
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda path, *args, **kwargs: installed.append(path) or "a" * 64,
    )

    with pytest.raises(RuntimeError, match="authenticated recomputation|grid.*differ"):
        train_v2.write_static_age_grid_v2(
            protocol_v2.canonical_grid_path_v2(result_root),
            handcrafted,
            authorized_lock_snapshot=lock,
            input_snapshot=input_snapshot,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )
    assert recompute_calls == [(input_snapshot, lock)]
    assert installed == []


@pytest.mark.parametrize(
    "api",
    (train_v2.fit_one_v2, train_v2.build_static_age_grid_from_snapshot_v2),
)
def test_primary_v2_production_training_api_has_no_unlocked_env_override(api) -> None:
    assert "env_config" not in inspect.signature(api).parameters


@pytest.mark.parametrize("generations", (25, 30))
def test_primary_v2_fit_validator_rejects_impossible_cma_evaluation_count(
    generations: int,
) -> None:
    spec = next(
        row
        for row in protocol_v2.locked_fit_inventory_v2()
        if row["family"] == "temporal_endowment"
        and row["generations"] == generations
    )
    payload = _valid_nonstatic_fit_payload_v2(spec, "a" * 64)
    payload["result"]["n_objective_evals"] -= 1
    with pytest.raises(RuntimeError, match="evaluation count|CMA contract"):
        train_v2.validate_fit_payload_v2(
            payload,
            expected_spec=spec,
            expected_protocol_lock_sha256="a" * 64,
        )


def test_primary_v2_fit_validator_enforces_selected_weight_source_coherence(
    tmp_path: Path,
) -> None:
    nonstatic_spec = _fit_spec("temporal_endowment")
    nonstatic = _valid_nonstatic_fit_payload_v2(nonstatic_spec, "a" * 64)
    nonstatic["result"]["selected_weights"][2] = 0.5
    with pytest.raises(RuntimeError, match="selected weights|masked CMA"):
        train_v2.validate_fit_payload_v2(
            nonstatic,
            expected_spec=nonstatic_spec,
            expected_protocol_lock_sha256="a" * 64,
        )

    static_spec = _fit_spec("static_age_lookup")
    lock = _base_protocol_lock_snapshot_v2(
        tmp_path / "results" / "iclr_primary_warsaw_v2"
    )
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid_content = _canonical_bytes(grid_payload)
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(lock.path.parent),
        content=grid_content,
        sha256=hashlib.sha256(grid_content).hexdigest(),
        payload=grid_payload,
    )
    static_cma = _valid_static_fit_payload_v2(static_spec, lock.sha256, grid)
    static_cma["result"]["selection_source"] = "cmaes"
    with pytest.raises(RuntimeError, match="selected weights|CMA result"):
        train_v2.validate_fit_payload_v2(
            static_cma,
            expected_spec=static_spec,
            expected_protocol_lock_sha256=lock.sha256,
        )

    static_initial = _valid_static_fit_payload_v2(static_spec, lock.sha256, grid)
    static_initial["result"]["selected_weights"] = [0.0, 0.0, 0.0, 0.123456]
    with pytest.raises(RuntimeError, match="initial weights|locked grid"):
        train_v2.validate_fit_payload_v2(
            static_initial,
            expected_spec=static_spec,
            expected_protocol_lock_sha256=lock.sha256,
        )

    containment_violation = _valid_static_fit_payload_v2(
        static_spec,
        lock.sha256,
        grid,
    )
    containment_violation["result"]["optimizer_best_loss"] = 0.5
    with pytest.raises(RuntimeError, match="initializer loss|optimizer best loss"):
        train_v2.validate_fit_payload_v2(
            containment_violation,
            expected_spec=static_spec,
            expected_protocol_lock_sha256=lock.sha256,
        )


def test_primary_v2_static_initial_must_equal_authenticated_grid_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    spec = _fit_spec("static_age_lookup")
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid_content = _canonical_bytes(grid_payload)
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        content=grid_content,
        sha256=hashlib.sha256(grid_content).hexdigest(),
        payload=grid_payload,
    )
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    payload = _valid_static_fit_payload_v2(spec, lock.sha256, grid)
    payload["result"]["selected_weights"] = [
        0.0,
        0.0,
        0.0,
        math.log1p(0.2),
    ]
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        train_v2,
        "load_static_age_grid_dependency_v2",
        lambda *args, **kwargs: grid,
    )
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        lambda **kwargs: grid_payload,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: pytest.fail("incoherent fit must not be installed"),
    )

    with pytest.raises(RuntimeError, match="locked grid initializer"):
        train_v2.write_immutable_fit_v2(
            protocol_v2.canonical_fit_path_v2(result_root, spec),
            payload,
            expected_spec=spec,
            authorized_lock_snapshot=lock,
            input_snapshot=input_snapshot,
            grid_snapshot=grid,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )


def test_primary_v2_fit_compute_rejects_provenance_correct_forged_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    spec = _fit_spec("static_age_lookup")
    authentic = _training_grid_payload_v2(lock.sha256)
    forged = copy.deepcopy(authentic)
    forged["train_worst_csd"] = [0.0] * len(forged["alphas"])
    forged["selected_alpha"] = 0.0
    forged_content = _canonical_bytes(forged)
    forged_snapshot = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        content=forged_content,
        sha256=hashlib.sha256(forged_content).hexdigest(),
        payload=forged,
    )
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
    )
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        lambda **kwargs: authentic,
    )
    monkeypatch.setattr(
        train_v2,
        "load_training_series_data_v2",
        lambda *args: [],
    )
    monkeypatch.setattr(
        train_v2,
        "optimize_fit_recipe_v2",
        lambda *args, **kwargs: pytest.fail(
            "a forged grid must be rejected before optimization"
        ),
    )

    with pytest.raises(RuntimeError, match="authenticated.*grid|grid.*recomputation"):
        train_v2.fit_one_v2(
            spec,
            input_snapshot=input_snapshot,
            authorized_lock_snapshot=lock,
            grid_snapshot=forged_snapshot,
        )


def test_primary_v2_fit_writer_rechecks_grid_after_authenticated_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    spec = _fit_spec("static_age_lookup")
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid_content = _canonical_bytes(grid_payload)
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        content=grid_content,
        sha256=hashlib.sha256(grid_content).hexdigest(),
        payload=grid_payload,
    )
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    payload = _valid_static_fit_payload_v2(spec, lock.sha256, grid)
    drifted_grid = train_v2.JsonSnapshotV2(
        path=grid.path,
        content=grid.content + b"\n",
        sha256=hashlib.sha256(grid.content + b"\n").hexdigest(),
        payload=grid.payload,
    )
    current_grid = [grid]
    events: list[str] = []

    def rebuild(**kwargs):
        events.append("recompute")
        current_grid[0] = drifted_grid
        return grid_payload

    def verify(*args):
        events.append("lock")
        return lock

    def reload_grid(*args, **kwargs):
        events.append("grid")
        return current_grid[0]

    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        rebuild,
    )
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        verify,
    )
    monkeypatch.setattr(
        train_v2,
        "load_static_age_grid_dependency_v2",
        reload_grid,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: pytest.fail(
            "post-recomputation grid drift must prevent installation"
        ),
    )

    with pytest.raises(RuntimeError, match="grid.*changed|grid.*drift"):
        train_v2.write_immutable_fit_v2(
            protocol_v2.canonical_fit_path_v2(result_root, spec),
            payload,
            expected_spec=spec,
            authorized_lock_snapshot=lock,
            input_snapshot=input_snapshot,
            grid_snapshot=grid,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )
    assert events == ["recompute", "lock", "grid"]


def test_primary_v2_fit_writer_rejects_provenance_correct_forged_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    spec = _fit_spec("static_age_lookup")
    authentic = _training_grid_payload_v2(lock.sha256)
    forged = copy.deepcopy(authentic)
    forged["train_worst_csd"] = [0.0] * len(forged["alphas"])
    forged["selected_alpha"] = 0.0
    forged_content = _canonical_bytes(forged)
    forged_snapshot = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        content=forged_content,
        sha256=hashlib.sha256(forged_content).hexdigest(),
        payload=forged,
    )
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    payload = _valid_static_fit_payload_v2(spec, lock.sha256, forged_snapshot)
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        train_v2,
        "load_static_age_grid_dependency_v2",
        lambda *args, **kwargs: forged_snapshot,
    )
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        lambda **kwargs: authentic,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: pytest.fail("a forged grid must not be installed"),
    )

    with pytest.raises(RuntimeError, match="authenticated.*grid|grid.*recomputation"):
        train_v2.write_immutable_fit_v2(
            protocol_v2.canonical_fit_path_v2(result_root, spec),
            payload,
            expected_spec=spec,
            authorized_lock_snapshot=lock,
            input_snapshot=input_snapshot,
            grid_snapshot=forged_snapshot,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )


def test_primary_v2_static_initial_loss_must_equal_authenticated_grid_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    spec = _fit_spec("static_age_lookup")
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid_content = _canonical_bytes(grid_payload)
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        content=grid_content,
        sha256=hashlib.sha256(grid_content).hexdigest(),
        payload=grid_payload,
    )
    input_snapshot = _computation_input_snapshot_v2(
        lock,
        "temporal_2022",
        data_dir=data_dir,
    )
    payload = _valid_static_fit_payload_v2(spec, lock.sha256, grid)
    payload["result"]["best_loss"] = 123.0
    payload["result"]["optimizer_best_loss"] = 124.0
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        train_v2,
        "load_static_age_grid_dependency_v2",
        lambda *args, **kwargs: grid,
    )
    monkeypatch.setattr(
        train_v2,
        "build_static_age_grid_from_snapshot_v2",
        lambda **kwargs: grid_payload,
    )
    monkeypatch.setattr(
        train_v2,
        "write_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: pytest.fail("an incoherent fit must not be installed"),
    )

    with pytest.raises(RuntimeError, match="selected loss|grid.*loss"):
        train_v2.write_immutable_fit_v2(
            protocol_v2.canonical_fit_path_v2(result_root, spec),
            payload,
            expected_spec=spec,
            authorized_lock_snapshot=lock,
            input_snapshot=input_snapshot,
            grid_snapshot=grid,
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )


def test_primary_v2_fit_id_and_worker_count_are_closed_coordinates() -> None:
    spec = protocol_v2.locked_fit_inventory_v2()[0]
    assert train_v2.fit_spec_by_id_v2(str(spec["fit_id"])) == spec
    with pytest.raises(RuntimeError, match="fit id|inventory|coordinate"):
        train_v2.fit_spec_by_id_v2("not-a-locked-fit")

    assert train_v2.validate_max_workers_v2(1) == 1
    assert train_v2.validate_max_workers_v2(4) == 4
    assert train_v2.validate_max_workers_v2(5) == 5
    for invalid in (True, 0, -1, 6, 4.0):
        with pytest.raises((TypeError, ValueError, RuntimeError), match="worker"):
            train_v2.validate_max_workers_v2(invalid)  # type: ignore[arg-type]


def test_primary_v2_existing_fit_authentication_is_strict_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    lock = _base_protocol_lock_snapshot_v2(result_root)
    spec = _fit_spec("endowment_frontier")
    payload = _valid_nonstatic_fit_payload_v2(spec, lock.sha256)
    path = protocol_v2.canonical_fit_path_v2(result_root, spec)
    path.parent.mkdir(parents=True)
    path.write_bytes(_canonical_bytes(payload))

    snapshot = train_v2.authenticate_existing_fit_v2(
        spec,
        authorized_lock_snapshot=lock,
        result_root=result_root,
    )
    assert snapshot is not None
    assert snapshot.path == path
    assert snapshot.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    path.unlink()
    assert (
        train_v2.authenticate_existing_fit_v2(
            spec,
            authorized_lock_snapshot=lock,
            result_root=result_root,
        )
        is None
    )

    path.write_bytes(_canonical_bytes({**payload, "training_only": False}))
    with pytest.raises(RuntimeError, match="training|fit|schema"):
        train_v2.authenticate_existing_fit_v2(
            spec,
            authorized_lock_snapshot=lock,
            result_root=result_root,
        )


def test_primary_v2_complete_matrix_rejects_missing_and_extra_fit_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    result_root.mkdir(parents=True)
    lock = _base_protocol_lock_snapshot_v2(result_root)
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        content=_canonical_bytes(grid_payload),
        sha256=hashlib.sha256(_canonical_bytes(grid_payload)).hexdigest(),
        payload=grid_payload,
    )
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        train_v2,
        "load_static_age_grid_dependency_v2",
        lambda *args, **kwargs: grid,
    )
    monkeypatch.setattr(
        train_v2,
        "authenticate_existing_fit_v2",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RuntimeError, match="missing.*49|49.*missing|incomplete"):
        train_v2.verify_complete_matrix_v2(
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )

    snapshots = {
        str(spec["fit_id"]): train_v2.JsonSnapshotV2(
            path=protocol_v2.canonical_fit_path_v2(result_root, spec),
            content=b"{}\n",
            sha256=hashlib.sha256(b"{}\n").hexdigest(),
            payload={},
        )
        for spec in protocol_v2.locked_fit_inventory_v2()
    }
    monkeypatch.setattr(
        train_v2,
        "authenticate_existing_fit_v2",
        lambda spec, **kwargs: snapshots[str(spec["fit_id"])],
    )
    extra = result_root / "fits" / "not-in-lock.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="extra|unexpected|tree"):
        train_v2.verify_complete_matrix_v2(
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )


def test_primary_v2_matrix_requires_grid_then_process_parallelizes_missing_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    repo_root = tmp_path / "repo"
    data_dir = tmp_path / "data" / "pb"
    result_root.mkdir(parents=True)
    lock = _base_protocol_lock_snapshot_v2(result_root)
    grid_payload = _training_grid_payload_v2(lock.sha256)
    grid = train_v2.JsonSnapshotV2(
        path=protocol_v2.canonical_grid_path_v2(result_root),
        content=_canonical_bytes(grid_payload),
        sha256=hashlib.sha256(_canonical_bytes(grid_payload)).hexdigest(),
        payload=grid_payload,
    )
    events: list[str] = []
    submitted: list[str] = []

    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )

    def fake_grid(*args, **kwargs):
        events.append("grid")
        return grid

    monkeypatch.setattr(train_v2, "load_static_age_grid_dependency_v2", fake_grid)
    monkeypatch.setattr(
        train_v2,
        "authenticate_existing_fit_v2",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(train_v2, "_assert_fit_tree_paths_v2", lambda *args: None)
    monkeypatch.setattr(
        train_v2,
        "verify_complete_matrix_v2",
        lambda **kwargs: {
            "schema_version": 2,
            "status": "complete",
            "fit_count": 49,
        },
    )

    class FakePool:
        def __init__(self, *, max_workers: int):
            assert max_workers == 4

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, fit_id, **kwargs):
            assert events and events[0] == "grid"
            submitted.append(fit_id)
            future: Future[dict[str, object]] = Future()
            future.set_result({"fit_id": fit_id, "status": "written"})
            return future

    monkeypatch.setattr(train_v2, "ProcessPoolExecutor", FakePool)

    summary = train_v2.run_fit_matrix_v2(
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
        max_workers=4,
    )
    assert summary["status"] == "complete"
    assert len(submitted) == 49
    assert set(submitted) == {
        str(spec["fit_id"]) for spec in protocol_v2.locked_fit_inventory_v2()
    }
    assert "ProcessPoolExecutor" in inspect.getsource(train_v2.run_fit_matrix_v2)


def test_primary_v2_training_cli_exposes_grid_fit_matrix_and_verifier(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        train_v2,
        "run_static_age_grid_v2",
        lambda **kwargs: calls.append("grid") or {"status": "written"},
    )
    monkeypatch.setattr(
        train_v2,
        "run_one_fit_v2",
        lambda **kwargs: calls.append("one-fit") or {"status": "written"},
    )
    monkeypatch.setattr(
        train_v2,
        "run_fit_matrix_v2",
        lambda **kwargs: calls.append("matrix") or {"status": "complete"},
    )
    monkeypatch.setattr(
        train_v2,
        "verify_complete_matrix_v2",
        lambda **kwargs: calls.append("verify-complete")
        or {"status": "complete"},
    )
    fit_id = str(protocol_v2.locked_fit_inventory_v2()[0]["fit_id"])

    for argv in (
        ["grid"],
        ["one-fit", "--fit-id", fit_id],
        ["matrix", "--max-workers", "4"],
        ["verify-complete"],
    ):
        train_v2.main(argv)
        assert json.loads(capsys.readouterr().out)["status"] in {
            "written",
            "complete",
        }
    assert calls == ["grid", "one-fit", "matrix", "verify-complete"]
