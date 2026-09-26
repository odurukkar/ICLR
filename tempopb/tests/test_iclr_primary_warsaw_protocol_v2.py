"""RED contracts for the append-only primary-Warsaw protocol-v2 lineage."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import iclr_primary_warsaw_protocol_v2 as protocol_v2
import iclr_primary_warsaw_train_v2 as train_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SPLIT_SHA256 = {
    "temporal_2022": (
        "baa81f876f81a5018c6e72a5be38bb14863ce74ffb50536c99c214fe06124cdd"
    ),
    "district_out_f0of5": (
        "174a12304e6bd46d08e9dd372eed8fe5fb175d52cdc90241481ada7a6d8b125d"
    ),
    "district_out_f1of5": (
        "718ea241efdf806fb83ea71e3a90028a7d78b37fca572bc304bff1c93a03ceaf"
    ),
    "district_out_f2of5": (
        "a06162820b8ea5a6a860d713bab965d78ba99ea25e150db78358d5c802b255e5"
    ),
    "district_out_f3of5": (
        "e1918c2ca49b20934dc302172d4347adf2109b5b4952c00a4f0fd08b3466348c"
    ),
    "district_out_f4of5": (
        "0e802c8ca7b04a39b7c2ba02468a7ebca71a890b1c233d80f5bbf6f8881a7943"
    ),
    "city_out_Poland_Łódź": (
        "25ea66d2e85a05669711a87664ec3e744c5c2be2df3973979b316aa665d5ed2c"
    ),
}

EXPECTED_STATIC_AGE_GRID = (
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    1.15,
    1.2,
    1.25,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
    1.55,
    1.6,
    1.65,
    1.7,
    1.75,
    1.8,
    1.85,
    1.9,
    1.95,
    2.0,
    2.05,
    2.1,
    2.15,
    2.2,
    2.25,
    2.3,
    2.35,
    2.4,
    2.45,
    2.5,
    2.55,
    2.6,
    2.65,
    2.7,
    2.75,
    2.8,
    2.85,
    2.9,
    2.95,
    3.0,
)

INVENTORY_FIELDS = {
    "fit_id",
    "family",
    "split",
    "arm",
    "seed",
    "generations",
    "popsize",
    "sigma0",
    "bound",
    "init",
    "soft_welfare_target",
    "welfare_penalty",
    "masked_feature_names",
    "support_floor_kappa",
    "grid_dependency",
    "wola_2021_in_training",
    "relative_path",
}


def _by_family(
    inventory: list[dict[str, object]], family: str
) -> list[dict[str, object]]:
    return [row for row in inventory if row["family"] == family]


def test_primary_v2_namespace_and_authenticated_constants_are_exact() -> None:
    assert protocol_v2.RESULT_ROOT_V2 == (
        PROJECT_ROOT / "results" / "iclr_primary_warsaw_v2"
    )
    assert protocol_v2.LOCK_PROFILE_V2 == "primary-warsaw-corrected-post-hoc-v2"
    assert protocol_v2.PARENT_FROZEN_LIST_SHA256 == (
        "5fae438226ede5b3e365c5ee5d161231e0db26adab7e7d470cce3a6e491f3d05"
    )
    assert protocol_v2.EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256 == (
        "5e38dd79b07b6b7dc233f3fd14e6fadc1840b3e2e7825d289b0935ddf179a4fa"
    )
    assert protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256 == (
        "368dc78da2f998b53c07358a781777effd24fae4a813333b4dd7f8d54b342bd6"
    )
    assert protocol_v2.SPLIT_SHA256_V2 == EXPECTED_SPLIT_SHA256


def test_primary_v2_inventory_is_the_exact_49_fit_matrix() -> None:
    inventory = protocol_v2.locked_fit_inventory_v2()

    assert len(inventory) == 49
    assert all(type(row) is dict and set(row) == INVENTORY_FIELDS for row in inventory)
    assert [row["fit_id"] for row in inventory] == sorted(
        row["fit_id"] for row in inventory
    )
    assert len({row["fit_id"] for row in inventory}) == 49
    assert len({row["relative_path"] for row in inventory}) == 49
    assert {row["family"] for row in inventory} == {
        "outcome_frontier",
        "endowment_frontier",
        "temporal_endowment",
        "district_endowment",
        "outcome_loo",
        "static_support_floor",
        "history_free",
        "static_age_lookup",
    }
    assert {
        family: len(_by_family(inventory, family))
        for family in {row["family"] for row in inventory}
    } == {
        "outcome_frontier": 24,
        "endowment_frontier": 3,
        "temporal_endowment": 3,
        "district_endowment": 10,
        "outcome_loo": 5,
        "static_support_floor": 2,
        "history_free": 1,
        "static_age_lookup": 1,
    }

    for row in inventory:
        assert type(row["fit_id"]) is str
        assert type(row["split"]) is str
        assert type(row["arm"]) is str
        assert type(row["seed"]) is int
        assert type(row["generations"]) is int
        assert row["popsize"] is None
        assert type(row["sigma0"]) is float and row["sigma0"] == 0.4
        assert type(row["bound"]) is float
        assert type(row["soft_welfare_target"]) is float
        assert type(row["welfare_penalty"]) is float
        assert type(row["masked_feature_names"]) is list
        assert type(row["wola_2021_in_training"]) is bool
        assert row["relative_path"] == f"fits/{row['fit_id']}.json"

    outcome = _by_family(inventory, "outcome_frontier")
    assert {
        (row["seed"], row["soft_welfare_target"], row["welfare_penalty"])
        for row in outcome
    } == {
        (seed, target, 0.0 if target == 0.0 else 2.0)
        for seed in (1, 2, 3, 42)
        for target in (0.0, 0.85, 0.95, 1.0, 1.02, 1.05)
    }
    assert {
        (row["split"], row["arm"], row["generations"], row["bound"], row["init"])
        for row in outcome
    } == {("temporal_2022", "outcome", 25, 10.0, "res")}

    endowment_frontier = _by_family(inventory, "endowment_frontier")
    assert {
        (row["seed"], row["soft_welfare_target"], row["welfare_penalty"])
        for row in endowment_frontier
    } == {(42, 0.0, 0.0), (42, 0.99, 2.0), (42, 1.0, 2.0)}
    assert {
        (row["split"], row["arm"], row["generations"], row["bound"], row["init"])
        for row in endowment_frontier
    } == {("temporal_2022", "endowment", 25, 10.0, "res")}

    temporal_endowment = _by_family(inventory, "temporal_endowment")
    assert {
        (
            row["seed"],
            row["generations"],
            row["soft_welfare_target"],
            row["welfare_penalty"],
        )
        for row in temporal_endowment
    } == {(1, 25, 0.0, 0.0), (2, 25, 0.0, 0.0), (42, 30, 0.0, 0.0)}

    district = _by_family(inventory, "district_endowment")
    assert {
        (row["split"], row["bound"])
        for row in district
    } == {
        (f"district_out_f{fold}of5", bound)
        for fold in range(5)
        for bound in (10.0, 40.0)
    }
    assert {
        (
            row["arm"],
            row["seed"],
            row["generations"],
            row["soft_welfare_target"],
            row["welfare_penalty"],
            row["init"],
        )
        for row in district
    } == {("endowment", 42, 25, 0.0, 0.0, "res")}

    outcome_features = {
        "approval_share",
        "approvals_per_cost",
        "cost_share",
        "deficit_weighted",
        "cohort_concentration",
    }
    loo = _by_family(inventory, "outcome_loo")
    assert {tuple(row["masked_feature_names"]) for row in loo} == {
        (name,) for name in outcome_features
    }
    assert {
        (
            row["split"],
            row["arm"],
            row["seed"],
            row["generations"],
            row["soft_welfare_target"],
            row["welfare_penalty"],
        )
        for row in loo
    } == {("temporal_2022", "outcome", 42, 25, 1.0, 2.0)}

    support = _by_family(inventory, "static_support_floor")
    assert {row["support_floor_kappa"] for row in support} == {1.0, 2.0}
    assert {
        (
            row["split"],
            row["arm"],
            row["seed"],
            row["generations"],
            row["soft_welfare_target"],
            row["welfare_penalty"],
        )
        for row in support
    } == {("temporal_2022", "outcome", 42, 25, 0.0, 0.0)}

    history_free = _by_family(inventory, "history_free")
    assert [
        (
            history_free[0]["split"],
            history_free[0]["arm"],
            history_free[0]["seed"],
            history_free[0]["generations"],
            history_free[0]["init"],
            history_free[0]["masked_feature_names"],
        )
    ] == [
        (
            "temporal_2022",
            "endowment",
            42,
            30,
            "zeros",
            ["deficit_per_capita", "deficit_normalized"],
        )
    ]

    static_age = _by_family(inventory, "static_age_lookup")
    assert [
        (
            static_age[0]["split"],
            static_age[0]["arm"],
            static_age[0]["seed"],
            static_age[0]["generations"],
            static_age[0]["bound"],
            static_age[0]["init"],
            static_age[0]["grid_dependency"],
        )
    ] == [
        (
            "temporal_2022",
            "static_age_lookup",
            42,
            30,
            3.0,
            "training_grid",
            "training_grid/static_senior_alpha.json",
        )
    ]


def test_primary_v2_fit_paths_are_canonical_and_wola_exposure_is_47_of_49(
    tmp_path: Path,
) -> None:
    inventory = protocol_v2.locked_fit_inventory_v2()
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"

    assert {
        protocol_v2.canonical_fit_path_v2(result_root, row)
        for row in inventory
    } == {result_root / str(row["relative_path"]) for row in inventory}
    assert sum(bool(row["wola_2021_in_training"]) for row in inventory) == 47
    assert {
        row["fit_id"] for row in inventory if not row["wola_2021_in_training"]
    } == {
        "district_endowment/district_out_f4of5/bound-10/seed-42",
        "district_endowment/district_out_f4of5/bound-40/seed-42",
    }


def test_primary_v2_grid_and_seven_split_paths_are_exact(tmp_path: Path) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"

    assert protocol_v2.STATIC_AGE_GRID_VALUES == EXPECTED_STATIC_AGE_GRID
    assert len(protocol_v2.STATIC_AGE_GRID_VALUES) == 61
    assert protocol_v2.canonical_grid_path_v2(result_root) == (
        result_root / "training_grid" / "static_senior_alpha.json"
    )
    assert {
        name: protocol_v2.canonical_split_path_v2(result_root, name)
        for name in EXPECTED_SPLIT_SHA256
    } == {
        name: result_root / "splits" / f"{name}.json"
        for name in EXPECTED_SPLIT_SHA256
    }


@pytest.mark.parametrize(
    "content",
    (
        b'{"schema_version":2,"schema_version":2}',
        b'{"value":NaN}',
        b'{"value":Infinity}',
        b'{"value":-Infinity}',
        b'{"value":1e309}',
        b'{"nested":[0,{"value":1e309}]}',
    ),
)
def test_primary_v2_json_reader_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(content)

    with pytest.raises(RuntimeError, match="duplicate|finite"):
        protocol_v2.load_json_object_strict_v2(path, label="synthetic JSON")


def test_primary_v2_inventory_validator_rejects_python_type_aliases() -> None:
    inventory = protocol_v2.locked_fit_inventory_v2()
    protocol_v2.validate_locked_fit_inventory_v2(inventory)

    boolean_seed = copy.deepcopy(inventory)
    boolean_seed[0]["seed"] = True
    with pytest.raises(RuntimeError, match="fit inventory"):
        protocol_v2.validate_locked_fit_inventory_v2(boolean_seed)

    integer_bound = copy.deepcopy(inventory)
    integer_bound[0]["bound"] = 10
    with pytest.raises(RuntimeError, match="fit inventory"):
        protocol_v2.validate_locked_fit_inventory_v2(integer_bound)

    float_generations = copy.deepcopy(inventory)
    float_generations[0]["generations"] = 25.0
    with pytest.raises(RuntimeError, match="fit inventory"):
        protocol_v2.validate_locked_fit_inventory_v2(float_generations)


def test_primary_v2_immutable_writer_is_create_once_and_no_follow(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "result" / "artifact.bin"
    expected = hashlib.sha256(b"first").hexdigest()

    assert protocol_v2.write_immutable_bytes_artifact_v2(artifact, b"first") == expected
    assert protocol_v2.write_immutable_bytes_artifact_v2(artifact, b"first") == expected
    with pytest.raises(RuntimeError, match="immutable|overwrite|conflict"):
        protocol_v2.write_immutable_bytes_artifact_v2(artifact, b"second")
    assert artifact.read_bytes() == b"first"

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink|non-regular"):
        protocol_v2.write_immutable_bytes_artifact_v2(alias, b"replacement")
    assert outside.read_bytes() == b"outside"

    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        protocol_v2.write_immutable_bytes_artifact_v2(
            linked_parent / "escaped.bin", b"escaped"
        )
    assert not (outside_dir / "escaped.bin").exists()


def test_repository_primary_manifest_splits_and_semantics_match_contract() -> None:
    manifest = protocol_v2.build_primary_manifest_v2(
        PROJECT_ROOT / "results" / "iclr_multicity_v2" / "corpus_manifest.json",
        PROJECT_ROOT / "results" / "frozen_instances.txt",
        PROJECT_ROOT / "data" / "pb",
    )

    assert manifest["n_series"] == 19
    assert manifest["n_elections"] == 132
    assert manifest["n_files"] == 132
    assert len(manifest["files"]) == 132
    assert [row["name"] for row in manifest["files"]] == sorted(
        row["name"] for row in manifest["files"]
    )
    assert protocol_v2.primary_manifest_retained_sha256_v2(manifest) == (
        "5e38dd79b07b6b7dc233f3fd14e6fadc1840b3e2e7825d289b0935ddf179a4fa"
    )
    assert protocol_v2.validate_primary_split_sources_v2(
        PROJECT_ROOT / "results" / "iclr_splits"
    ) == EXPECTED_SPLIT_SHA256

    receipt = protocol_v2.build_primary_semantics_receipt_v2(
        manifest,
        PROJECT_ROOT / "data" / "pb",
    )
    assert protocol_v2.validate_primary_semantics_receipt_v2(receipt) is receipt
    assert receipt["semantics_profile"] == "approval-set-first-occurrence-v2"
    assert receipt["parent_manifest_retained_sha256"] == (
        "5e38dd79b07b6b7dc233f3fd14e6fadc1840b3e2e7825d289b0935ddf179a4fa"
    )
    assert receipt["counts"] == {
        "series": 19,
        "files": 132,
        "ballots": 821_572,
        "raw_approval_tokens": 6_577_176,
        "canonical_approval_tokens": 6_577_175,
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
    assert receipt["corpus_semantic_sha256"] == (
        "368dc78da2f998b53c07358a781777effd24fae4a813333b4dd7f8d54b342bd6"
    )
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


def test_primary_v2_staging_is_create_once_and_preserves_all_split_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    split_source_dir = tmp_path / "protected-splits"
    split_source_dir.mkdir()
    split_bytes = {
        name: (
            json.dumps(
                {"name": name, "train": [], "test": []},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for name in EXPECTED_SPLIT_SHA256
    }
    split_hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in split_bytes.items()
    }
    for name, content in split_bytes.items():
        (split_source_dir / f"{name}.json").write_bytes(content)

    manifest = {
        "schema_version": 1,
        "source_commit": "2f4321fec84069f50abf35fb5e90852013d17070",
        "source_dir": "data/pb",
        "destination": "data/pb",
        "n_series": 19,
        "n_elections": 132,
        "n_files": 132,
        "files": [],
    }
    receipt = {
        "schema_version": 2,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "corpus_semantic_sha256": "e" * 64,
        "parent_manifest_retained_sha256": (
            "5e38dd79b07b6b7dc233f3fd14e6fadc1840b3e2e7825d289b0935ddf179a4fa"
        ),
        "corpus_semantic_sha256": (
            "368dc78da2f998b53c07358a781777effd24fae4a813333b4dd7f8d54b342bd6"
        ),
    }
    monkeypatch.setattr(protocol_v2, "SPLIT_SHA256_V2", split_hashes)
    monkeypatch.setattr(
        protocol_v2,
        "build_primary_manifest_v2",
        lambda *args, **kwargs: copy.deepcopy(manifest),
    )
    monkeypatch.setattr(
        protocol_v2,
        "primary_manifest_retained_sha256_v2",
        lambda payload: protocol_v2.EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256,
    )
    monkeypatch.setattr(
        protocol_v2,
        "build_primary_semantics_receipt_v2",
        lambda *args, **kwargs: copy.deepcopy(receipt),
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_primary_semantics_receipt_v2",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_primary_split_sources_v2",
        lambda path: split_hashes,
    )

    first = protocol_v2.stage_protocol_inputs_v2(
        multicity_manifest_path=tmp_path / "multicity-manifest.json",
        frozen_list_path=tmp_path / "frozen_instances.txt",
        split_source_dir=split_source_dir,
        data_dir=tmp_path / "data" / "pb",
        result_root=result_root,
    )
    second = protocol_v2.stage_protocol_inputs_v2(
        multicity_manifest_path=tmp_path / "multicity-manifest.json",
        frozen_list_path=tmp_path / "frozen_instances.txt",
        split_source_dir=split_source_dir,
        data_dir=tmp_path / "data" / "pb",
        result_root=result_root,
    )

    manifest_content = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    receipt_content = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert first == second == {
        "corpus_manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
        "approval_semantics_receipt_sha256": hashlib.sha256(
            receipt_content
        ).hexdigest(),
        "primary_manifest_retained_sha256": (
            "5e38dd79b07b6b7dc233f3fd14e6fadc1840b3e2e7825d289b0935ddf179a4fa"
        ),
        "corpus_semantic_sha256": (
            "368dc78da2f998b53c07358a781777effd24fae4a813333b4dd7f8d54b342bd6"
        ),
        "split_sha256": split_hashes,
    }
    assert (result_root / "corpus_manifest.json").read_bytes() == manifest_content
    assert (
        result_root / "approval_semantics_receipt.json"
    ).read_bytes() == receipt_content
    assert {
        name: (result_root / "splits" / f"{name}.json").read_bytes()
        for name in EXPECTED_SPLIT_SHA256
    } == split_bytes

    conflicted = result_root / "splits" / "temporal_2022.json"
    conflicted.write_bytes(b'{"conflict":true}\n')
    with pytest.raises(RuntimeError, match="immutable|overwrite|conflict"):
        protocol_v2.stage_protocol_inputs_v2(
            multicity_manifest_path=tmp_path / "multicity-manifest.json",
            frozen_list_path=tmp_path / "frozen_instances.txt",
            split_source_dir=split_source_dir,
            data_dir=tmp_path / "data" / "pb",
            result_root=result_root,
        )
    assert conflicted.read_bytes() == b'{"conflict":true}\n'


def _synthetic_lock_files(repo_root: Path) -> dict[str, Path]:
    files = {
        "artifact/corpus_manifest": (
            repo_root / "results" / "iclr_primary_warsaw_v2" / "corpus_manifest.json"
        ),
        "artifact/approval_semantics_receipt": (
            repo_root
            / "results"
            / "iclr_primary_warsaw_v2"
            / "approval_semantics_receipt.json"
        ),
        "split/temporal_2022": (
            repo_root
            / "results"
            / "iclr_primary_warsaw_v2"
            / "splits"
            / "temporal_2022.json"
        ),
        "source/core.py": repo_root / "src" / "core.py",
        "test/test_core.py": repo_root / "tests" / "test_core.py",
        "environment/pyproject": repo_root / "pyproject.toml",
        "environment/uv_lock": repo_root / "uv.lock",
    }
    for label, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if label == "artifact/corpus_manifest":
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_commit": protocol_v2.PABULIB_COMMIT,
                        "source_dir": "data/pb",
                        "destination": "data/pb",
                        "n_series": 0,
                        "n_elections": 0,
                        "n_files": 0,
                        "files": [],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        elif label == "artifact/approval_semantics_receipt":
            path.write_text(
                json.dumps(
                    {
                        "corpus_semantic_sha256": (
                            protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
                        )
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"{label}\n", encoding="utf-8")
    return files


def test_primary_v2_lock_binds_closure_and_declares_grid_without_circular_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    result_root = repo_root / "results" / "iclr_primary_warsaw_v2"
    data_dir = repo_root / "data" / "pb"
    data_dir.mkdir(parents=True)
    mandatory = _synthetic_lock_files(repo_root)
    grid_spec = {
        "artifact_label": "artifact/static_senior_alpha_grid",
        "relative_path": "training_grid/static_senior_alpha.json",
        "split": "temporal_2022",
        "training_only": True,
        "alphas": list(EXPECTED_STATIC_AGE_GRID),
        "selection_metric": "training worst-cohort CSD",
        "tie_break": "smaller alpha",
        "created_after_lock": True,
    }

    payload = protocol_v2.build_protocol_lock_payload_v2(repo_root, mandatory)

    assert set(payload) == {
        "schema_version",
        "lock_profile",
        "mandatory_file_count",
        "tracked_files",
        "fit_inventory",
        "training_grid_spec",
        "protocol",
        "corpus_semantic_sha256",
    }
    assert payload["schema_version"] == 2
    assert payload["lock_profile"] == "primary-warsaw-corrected-post-hoc-v2"
    assert payload["mandatory_file_count"] == len(mandatory)
    assert payload["fit_inventory"] == protocol_v2.locked_fit_inventory_v2()
    assert payload["training_grid_spec"] == grid_spec
    assert payload["corpus_semantic_sha256"] == (
        protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
    )
    assert set(payload["tracked_files"]) == set(mandatory)
    assert "artifact/static_senior_alpha_grid" not in payload["tracked_files"]
    assert {
        label.split("/", 1)[0] for label in payload["tracked_files"]
    } >= {"artifact", "split", "source", "test", "environment"}
    protocol_v2.validate_protocol_lock_payload_v2(payload, repo_root, mandatory)

    lock_path = result_root / "protocol_lock.json"
    protocol_v2.write_immutable_bytes_artifact_v2(
        lock_path,
        (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    )
    monkeypatch.setattr(
        protocol_v2,
        "mandatory_protocol_files_v2",
        lambda *args, **kwargs: mandatory,
    )
    snapshot = protocol_v2.verify_primary_warsaw_protocol_lock_snapshot_v2(
        lock_path,
        repo_root,
        result_root,
        data_dir,
    )
    assert snapshot.payload == payload


def test_primary_v2_lock_rejects_drift_symlinks_and_json_type_aliases(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    mandatory = _synthetic_lock_files(repo_root)
    payload = protocol_v2.build_protocol_lock_payload_v2(repo_root, mandatory)

    schema_alias = copy.deepcopy(payload)
    schema_alias["schema_version"] = 2.0
    with pytest.raises(RuntimeError, match="schema|lock"):
        protocol_v2.validate_protocol_lock_payload_v2(
            schema_alias, repo_root, mandatory
        )

    count_alias = copy.deepcopy(payload)
    count_alias["mandatory_file_count"] = float(len(mandatory))
    with pytest.raises(RuntimeError, match="count|lock"):
        protocol_v2.validate_protocol_lock_payload_v2(
            count_alias, repo_root, mandatory
        )

    grid_alias = copy.deepcopy(payload)
    grid_alias["training_grid_spec"]["alphas"][0] = 0
    with pytest.raises(RuntimeError, match="grid|protocol"):
        protocol_v2.validate_protocol_lock_payload_v2(
            grid_alias, repo_root, mandatory
        )

    inventory_alias = copy.deepcopy(payload)
    inventory_alias["fit_inventory"][0]["seed"] = True
    with pytest.raises(RuntimeError, match="fit inventory|protocol"):
        protocol_v2.validate_protocol_lock_payload_v2(
            inventory_alias, repo_root, mandatory
        )

    mandatory["source/core.py"].write_text("drifted\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest|drift"):
        protocol_v2.validate_protocol_lock_payload_v2(payload, repo_root, mandatory)

    real_source = repo_root / "src" / "real.py"
    real_source.write_text("real\n", encoding="utf-8")
    alias = repo_root / "src" / "alias.py"
    alias.symlink_to(real_source)
    aliased_files = {**mandatory, "source/core.py": alias}
    with pytest.raises(RuntimeError, match="symlink"):
        protocol_v2.build_protocol_lock_payload_v2(repo_root, aliased_files)


def test_training_lock_verification_uses_manifest_metadata_not_raw_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    mandatory = _synthetic_lock_files(repo_root)
    data_dir = repo_root / "data" / "pb"
    data_dir.mkdir(parents=True)
    manifest_rows = []
    raw_paths = []
    for year in (2021, 2023):
        path = data_dir / f"election-{year}.pb"
        path.write_bytes(f"raw-{year}\n".encode())
        raw_paths.append(path.absolute())
        manifest_rows.append(
            {
                "name": path.name,
                "source_relative": path.name,
                "series": "Poland/Warszawa/Wola",
                "year": year,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        mandatory[f"data/{path.name}"] = path
    manifest_path = mandatory["artifact/corpus_manifest"]
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_commit": protocol_v2.PABULIB_COMMIT,
                "source_dir": "data/pb",
                "destination": "data/pb",
                "n_series": 1,
                "n_elections": 2,
                "n_files": 2,
                "files": manifest_rows,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = protocol_v2.build_protocol_lock_payload_v2(repo_root, mandatory)
    reads: list[Path] = []
    real_read = protocol_v2.read_regular_bytes_artifact_v2

    def recording_read(path, **kwargs):
        reads.append(Path(path).absolute())
        return real_read(path, **kwargs)

    monkeypatch.setattr(
        protocol_v2, "read_regular_bytes_artifact_v2", recording_read
    )
    protocol_v2.validate_protocol_lock_payload_metadata_v2(
        payload,
        repo_root,
        mandatory,
    )

    assert not set(raw_paths) & set(reads)


def test_primary_v2_input_snapshot_is_single_read_identity_bound_and_train_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    data_dir = tmp_path / "data" / "pb"
    split_dir = result_root / "splits"
    split_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    series = "Poland/Warszawa/Wola"
    raw_files = {}
    for year, name in (
        (2021, "Poland_Warszawa_2022_Wola.pb"),
        (2023, "Poland_Warszawa_2024_Wola.pb"),
    ):
        content = f"raw-{year}\n".encode()
        (data_dir / name).write_bytes(content)
        raw_files[year] = (name, content)
    # The primary manifest selects a subset of the shared data directory.
    (data_dir / "unlisted-parent-superset.pb").write_bytes(b"allowed\n")
    manifest = {
        "schema_version": 1,
        "source_commit": "2f4321fec84069f50abf35fb5e90852013d17070",
        "source_dir": "data/pb",
        "destination": "data/pb",
        "n_series": 1,
        "n_elections": 2,
        "n_files": 2,
        "files": [
            {
                "name": raw_files[year][0],
                "source_relative": raw_files[year][0],
                "series": series,
                "year": year,
                "bytes": len(raw_files[year][1]),
                "sha256": hashlib.sha256(raw_files[year][1]).hexdigest(),
            }
            for year in (2021, 2023)
        ],
    }
    receipt = {
        "schema_version": 2,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "corpus_semantic_sha256": (
            protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
        ),
        "files": [
            {
                "series": series,
                "year": year,
                "name": raw_files[year][0],
                "raw_file_sha256": hashlib.sha256(raw_files[year][1]).hexdigest(),
                "v2_semantic_sha256": str(year)[-1] * 64,
            }
            for year in (2021, 2023)
        ],
    }
    split_payload = {
        "name": "temporal_2022",
        "note": "synthetic inherited v1 split",
        "train": [[series, [2021]]],
        "test": [[series, [2023]]],
    }
    paths = {
        "artifact/corpus_manifest": result_root / "corpus_manifest.json",
        "artifact/approval_semantics_receipt": (
            result_root / "approval_semantics_receipt.json"
        ),
        "split/temporal_2022": split_dir / "temporal_2022.json",
    }
    payloads = {
        "artifact/corpus_manifest": manifest,
        "artifact/approval_semantics_receipt": receipt,
        "split/temporal_2022": split_payload,
    }
    for label, path in paths.items():
        path.write_text(
            json.dumps(payloads[label], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    tracked_files = {
        label: {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for label, path in paths.items()
    }
    lock_payload = {
        "schema_version": 2,
        "lock_profile": protocol_v2.LOCK_PROFILE_V2,
        "mandatory_file_count": len(tracked_files),
        "tracked_files": tracked_files,
        "fit_inventory": protocol_v2.locked_fit_inventory_v2(),
        "training_grid_spec": copy.deepcopy(protocol_v2.TRAINING_GRID_SPEC_V2),
        "protocol": protocol_v2.protocol_config_v2(),
        "corpus_semantic_sha256": (
            protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
        ),
    }
    lock_content = (
        json.dumps(lock_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    lock_snapshot = protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=result_root / "protocol_lock.json",
        content=lock_content,
        sha256=hashlib.sha256(lock_content).hexdigest(),
        payload=lock_payload,
    )
    lock_snapshot.path.write_bytes(lock_content)
    reads: list[Path] = []
    real_read = protocol_v2.read_regular_bytes_artifact_v2

    def recording_read(path, **kwargs):
        reads.append(Path(path).absolute())
        return real_read(path, **kwargs)

    monkeypatch.setattr(
        protocol_v2, "read_regular_bytes_artifact_v2", recording_read
    )
    with pytest.raises(RuntimeError, match="verified.*lock|lock.*snapshot"):
        protocol_v2.load_protocol_inputs_snapshot_v2(
            lock_payload,
            result_root,
            data_dir,
            ("temporal_2022",),
            enforce_expected_counts=False,
        )

    snapshot = protocol_v2.load_protocol_inputs_snapshot_v2(
        lock_snapshot,
        result_root,
        data_dir,
        ("temporal_2022",),
        enforce_expected_counts=False,
    )

    assert all(reads.count(path.absolute()) == 1 for path in paths.values())
    training_raw_path = (data_dir / raw_files[2021][0]).absolute()
    heldout_raw_path = (data_dir / raw_files[2023][0]).absolute()
    assert reads.count(training_raw_path) == 1
    assert heldout_raw_path not in reads
    assert set(snapshot.raw_artifacts) == {
        (series, 2021, raw_files[2021][0])
    }
    assert set(snapshot.index) == {series}
    assert snapshot.index[series].years == (2021, 2023)
    assert set(snapshot.split_artifacts["temporal_2022"].payload) == {
        "name", "note", "train", "test"
    }
    training_split = snapshot.splits["temporal_2022"]
    assert training_split.train == ((series, (2021,)),)
    assert training_split.test == ()
    assert snapshot.semantics_receipt.payload == receipt

    protocol_v2.assert_protocol_inputs_unchanged_v2(snapshot, lock_snapshot)
    training_raw_path = data_dir / raw_files[2021][0]
    original_training_raw = training_raw_path.read_bytes()
    training_raw_path.write_bytes(original_training_raw + b"drift")
    with pytest.raises(RuntimeError, match="raw.*changed|input.*changed"):
        protocol_v2.assert_protocol_inputs_unchanged_v2(snapshot, lock_snapshot)
    training_raw_path.write_bytes(original_training_raw)

    paths["split/temporal_2022"].write_bytes(
        paths["split/temporal_2022"].read_bytes() + b"\n"
    )
    with pytest.raises(RuntimeError, match="input.*changed|split.*changed"):
        protocol_v2.assert_protocol_inputs_unchanged_v2(snapshot, lock_snapshot)

    # Identity equality is exact even when the changed receipt has a fresh lock hash.
    paths["split/temporal_2022"].write_text(
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bad_receipt = copy.deepcopy(receipt)
    bad_receipt["files"][1]["name"] = "different.pb"
    paths["artifact/approval_semantics_receipt"].write_text(
        json.dumps(bad_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lock_payload["tracked_files"]["artifact/approval_semantics_receipt"][
        "sha256"
    ] = hashlib.sha256(
        paths["artifact/approval_semantics_receipt"].read_bytes()
    ).hexdigest()
    changed_lock_content = (
        json.dumps(lock_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    changed_lock_snapshot = protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=lock_snapshot.path,
        content=changed_lock_content,
        sha256=hashlib.sha256(changed_lock_content).hexdigest(),
        payload=lock_payload,
    )
    changed_lock_snapshot.path.write_bytes(changed_lock_content)
    with pytest.raises(RuntimeError, match="manifest.*receipt|identit"):
        protocol_v2.load_protocol_inputs_snapshot_v2(
            changed_lock_snapshot,
            result_root,
            data_dir,
            ("temporal_2022",),
            enforce_expected_counts=False,
        )

    forged = protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=lock_snapshot.path,
        content=lock_snapshot.content,
        sha256="0" * 64,
        payload=lock_snapshot.payload,
    )
    with pytest.raises(
        RuntimeError,
        match="lock.*digest|lock.*snapshot|lock.*byte identity",
    ):
        protocol_v2.load_protocol_inputs_snapshot_v2(
            forged,
            result_root,
            data_dir,
            ("temporal_2022",),
            enforce_expected_counts=False,
        )


def test_postreceipt_structural_replay_checks_both_completion_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = (
        protocol_v2.SeriesRef(
            key="Poland/Warszawa/Wola",
            years=(2021,),
            paths=(Path("Wola.pb"),),
        ),
    )
    instance = SimpleNamespace(
        votes=(SimpleNamespace(projects=("a", "b")),),
        projects={"a": SimpleNamespace(cost=4.0)},
        budget=5.0,
    )
    calls: list[bool] = []

    monkeypatch.setattr(
        protocol_v2,
        "load_series_authenticated_v2",
        lambda ref, receipt: {2021: instance},
    )
    monkeypatch.setattr(
        protocol_v2,
        "mes_with_endowments",
        lambda election, *, completion: {"a"},
    )

    def fake_priority(election, state, weights, *, completion):
        calls.append(completion)
        return SimpleNamespace(
            winners=("a",),
            rounds=(SimpleNamespace(project_id="a", supporter_ids=("v1",)),),
        )

    monkeypatch.setattr(protocol_v2, "priority_mes_outcome", fake_priority)
    monkeypatch.setattr(
        protocol_v2,
        "primary_structural_source_sha256_v2",
        lambda root: {"src/rules.py": "a" * 64},
    )

    report = protocol_v2._verify_primary_priority_mes_corpus_v2(
        refs,
        semantics_receipt={"files": []},
        semantics_receipt_sha256="b" * 64,
        corpus_semantic_sha256="c" * 64,
        repo_root=Path("repo"),
    )

    assert calls == [False, False, True, True]
    assert report["n_series"] == 1
    assert report["n_elections"] == 1
    assert report["approval_ballots_checked"] == 1
    assert report["winner_identity_checks"] == 2
    assert report["determinism_checks"] == 2
    assert report["budget_checks"] == 2
    assert report["unique_payer_round_checks"] == 2
    assert report["payment_rounds_checked"] == 2
    assert report["completion_modes"] == [False, True]
    assert report["zero_weights"] == [0.0] * 5


def test_prefit_structural_gate_never_reads_raw_or_calls_parser_or_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    source_hashes = {"src/iclr_primary_warsaw_protocol_v2.py": "a" * 64}
    index = {
        f"Poland/Warszawa/S{i:02d}": protocol_v2.SeriesRef(
            key=f"Poland/Warszawa/S{i:02d}",
            years=(),
            paths=(),
        )
        for i in range(19)
    }
    metadata = {
        (f"series-{i}", i, f"file-{i}.pb"): SimpleNamespace()
        for i in range(132)
    }
    manifest = protocol_v2.JsonArtifactSnapshotV2(
        label="artifact/corpus_manifest",
        path=result_root / "corpus_manifest.json",
        content=b"manifest\n",
        sha256="b" * 64,
        payload={"n_series": 19, "n_elections": 132, "n_files": 132},
    )
    receipt = protocol_v2.JsonArtifactSnapshotV2(
        label="artifact/approval_semantics_receipt",
        path=result_root / "approval_semantics_receipt.json",
        content=b"receipt\n",
        sha256="c" * 64,
        payload={
            "corpus_semantic_sha256": (
                protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
            ),
            "counts": {"ballots": 821_572, "removed_tokens": 1},
        },
    )

    def fake_artifact(path, label, lock):
        del lock
        if Path(path).name == "corpus_manifest.json":
            return manifest
        if Path(path).name == "approval_semantics_receipt.json":
            return receipt
        return protocol_v2.JsonArtifactSnapshotV2(
            label=label,
            path=Path(path),
            content=b"split\n",
            sha256=protocol_v2.SPLIT_SHA256_V2[Path(path).stem],
            payload={"name": Path(path).stem, "train": [], "test": []},
        )

    monkeypatch.setattr(protocol_v2, "_locked_artifact_snapshot_v2", fake_artifact)
    monkeypatch.setattr(
        protocol_v2, "validate_primary_semantics_receipt_v2", lambda value: value
    )
    monkeypatch.setattr(
        protocol_v2,
        "_primary_index_metadata_from_snapshot_v2",
        lambda *args, **kwargs: (index, metadata),
        raising=False,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_primary_index_from_snapshot_v2",
        lambda *args, **kwargs: (index, metadata),
    )
    monkeypatch.setattr(
        protocol_v2, "_validate_manifest_receipt_identity_v2", lambda *args: None
    )
    monkeypatch.setattr(
        protocol_v2,
        "_training_split_from_snapshot_v2",
        lambda artifact, index, expected_name: protocol_v2.Split(
            name=expected_name, train=(), test=(), note="synthetic"
        ),
    )
    monkeypatch.setattr(
        protocol_v2,
        "primary_structural_source_sha256_v2",
        lambda root: source_hashes,
    )
    monkeypatch.setattr(
        protocol_v2,
        "read_regular_bytes_artifact_v2",
        lambda *args, **kwargs: pytest.fail("pre-fit gate read a raw byte path"),
    )
    monkeypatch.setattr(
        protocol_v2,
        "parse_pb_file",
        lambda *args, **kwargs: pytest.fail("pre-fit gate called the parser"),
    )
    monkeypatch.setattr(
        protocol_v2,
        "priority_mes_outcome",
        lambda *args, **kwargs: pytest.fail("pre-fit gate called an outcome solver"),
    )
    monkeypatch.setattr(
        protocol_v2,
        "mes_with_endowments",
        lambda *args, **kwargs: pytest.fail("pre-fit gate called MES"),
    )

    report = protocol_v2.build_primary_structural_report_v2(
        tmp_path,
        result_root,
        tmp_path / "data" / "pb",
    )
    assert report["gate_scope"] == (
        "pre-fit metadata and approval-semantics authentication only"
    )
    assert report["raw_election_files_opened"] == 0
    assert report["election_parser_calls"] == 0
    assert report["outcome_solver_calls"] == 0
    assert report["heldout_outcomes_scored"] is False


def test_evaluation_loader_requires_authenticated_receipt_before_raw_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reject_receipt(*args, **kwargs):
        events.append("receipt")
        raise RuntimeError("receipt authorization differs")

    monkeypatch.setattr(
        protocol_v2,
        "_validate_evaluation_replay_receipt_snapshot_v2",
        reject_receipt,
        raising=False,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_load_protocol_inputs_snapshot_scoped_v2",
        lambda *args, **kwargs: pytest.fail(
            "evaluation raw scope opened before receipt authorization"
        ),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="receipt authorization"):
        protocol_v2.load_evaluation_inputs_snapshot_v2(
            SimpleNamespace(),
            Path("results"),
            Path("data"),
            ("temporal_2022",),
            replay_receipt_snapshot=SimpleNamespace(),
        )
    assert events == ["receipt"]


def _matrix_boundary_lock_v2(
    result_root: Path,
) -> protocol_v2.PrimaryProtocolLockSnapshotV2:
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    tracked = {
        "artifact/corpus_manifest": {
            "path": "results/iclr_primary_warsaw_v2/corpus_manifest.json",
            "sha256": digest("manifest"),
        },
        "artifact/approval_semantics_receipt": {
            "path": "results/iclr_primary_warsaw_v2/approval_semantics_receipt.json",
            "sha256": digest("semantics"),
        },
        "artifact/structural_gates": {
            "path": "results/iclr_primary_warsaw_v2/structural_gates.json",
            "sha256": digest("structural"),
        },
        "source/iclr_primary_warsaw_train_v2.py": {
            "path": "src/iclr_primary_warsaw_train_v2.py",
            "sha256": digest("trainer"),
        },
        **{
            f"split/{name}": {
                "path": f"results/iclr_primary_warsaw_v2/splits/{name}.json",
                "sha256": digest(f"split:{name}"),
            }
            for name in protocol_v2.SPLIT_SHA256_V2
        },
    }
    payload = {
        "schema_version": 2,
        "lock_profile": protocol_v2.LOCK_PROFILE_V2,
        "mandatory_file_count": len(tracked),
        "tracked_files": tracked,
        "fit_inventory": protocol_v2.locked_fit_inventory_v2(),
        "training_grid_spec": copy.deepcopy(protocol_v2.TRAINING_GRID_SPEC_V2),
        "protocol": protocol_v2.protocol_config_v2(),
        "corpus_semantic_sha256": (
            protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
        ),
    }
    content = protocol_v2._canonical_json_bytes_v2(payload)
    path = result_root / "protocol_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=path.absolute(),
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
    )


def _matrix_boundary_provenance_v2(
    lock: protocol_v2.PrimaryProtocolLockSnapshotV2,
    split_name: str,
) -> dict[str, object]:
    tracked = lock.payload["tracked_files"]
    return {
        "protocol_lock_sha256": lock.sha256,
        "corpus_manifest_sha256": tracked["artifact/corpus_manifest"]["sha256"],
        "split_sha256": tracked[f"split/{split_name}"]["sha256"],
        "semantics_receipt_sha256": tracked[
            "artifact/approval_semantics_receipt"
        ]["sha256"],
        "corpus_semantic_sha256": lock.payload["corpus_semantic_sha256"],
        "source_sha256": {
            label.removeprefix("source/"): row["sha256"]
            for label, row in sorted(tracked.items())
            if label.startswith("source/")
        },
    }


def _matrix_boundary_grid_v2(
    lock: protocol_v2.PrimaryProtocolLockSnapshotV2,
) -> dict[str, object]:
    alphas = list(protocol_v2.STATIC_AGE_GRID_VALUES)
    return {
        "schema_version": 2,
        "semantics_profile": protocol_v2.SEMANTICS_PROFILE,
        "split": "temporal_2022",
        "training_only": True,
        "alphas": alphas,
        "train_worst_csd": [1.0 + alpha for alpha in alphas],
        "selected_alpha": 0.0,
        "selection_rule": train_v2.GRID_SELECTION_RULE,
        "provenance": _matrix_boundary_provenance_v2(lock, "temporal_2022"),
    }


def _matrix_boundary_fit_v2(
    spec: dict[str, object],
    lock: protocol_v2.PrimaryProtocolLockSnapshotV2,
    grid_sha256: str,
) -> dict[str, object]:
    arm = str(spec["arm"])
    feature_names = {
        "outcome": list(train_v2.PROJECT_FEATURES),
        "endowment": list(train_v2.FEATURE_NAMES),
        "static_age_lookup": list(train_v2.STATIC_AGE_FEATURE_NAMES),
    }[arm]
    weights = [0.0] * len(feature_names)
    provenance = _matrix_boundary_provenance_v2(lock, str(spec["split"]))
    provenance["grid_sha256"] = (
        grid_sha256 if spec["family"] == "static_age_lookup" else None
    )
    population = int(spec["popsize"] or (4 + int(3 * math.log(len(weights)))))
    evaluations = int(spec["generations"]) * population
    if spec["family"] == "static_age_lookup":
        evaluations += 1
    return {
        "schema_version": 2,
        "fit_id": spec["fit_id"],
        "config": spec,
        "training_only": True,
        "execution_mode": "serial",
        "arm": {
            "name": arm,
            "feature_names": feature_names,
            "init_name": spec["init"],
        },
        "provenance": provenance,
        "result": {
            "best_loss": 0.25,
            "optimizer_best_loss": 0.25,
            "optimizer_best_weights": weights,
            "selected_weights": list(weights),
            "selection_source": "cmaes",
            "n_objective_evals": evaluations,
            "optimizer_config": {
                "sigma0": spec["sigma0"],
                "popsize": spec["popsize"],
                "generations": spec["generations"],
                "seed": spec["seed"],
                "bound": spec["bound"],
            },
        },
    }


def _matrix_boundary_receipt_v2(
    result_root: Path,
    lock: protocol_v2.PrimaryProtocolLockSnapshotV2,
    *,
    grid_sha256: str,
    fit_sha256: dict[str, str],
) -> protocol_v2.JsonArtifactSnapshotV2:
    payload = {
        "schema_version": 2,
        "event": "evaluation replay started",
        "classification": "corrected post-hoc replay",
        "heldout_outcomes_already_known": True,
        "fresh_holdout": False,
        "preregistered": False,
        "disclosure": protocol_v2.EVALUATION_REPLAY_DISCLOSURE_V2,
        "protocol_lock_sha256": lock.sha256,
        "semantics_profile": protocol_v2.SEMANTICS_PROFILE,
        "semantics_receipt_sha256": lock.payload["tracked_files"][
            "artifact/approval_semantics_receipt"
        ]["sha256"],
        "corpus_semantic_sha256": lock.payload["corpus_semantic_sha256"],
        "structural_gates_sha256": lock.payload["tracked_files"][
            "artifact/structural_gates"
        ]["sha256"],
        "training_grid_sha256": grid_sha256,
        "fit_count": 49,
        "fit_sha256": dict(sorted(fit_sha256.items())),
    }
    content = protocol_v2._canonical_json_bytes_v2(payload)
    path = result_root / protocol_v2.EVALUATION_REPLAY_RECEIPT_NAME_V2
    path.write_bytes(content)
    return protocol_v2.JsonArtifactSnapshotV2(
        label="artifact/evaluation_replay_started",
        path=path.absolute(),
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
    )


def _stage_matrix_boundary_v2(
    tmp_path: Path,
) -> tuple[
    Path,
    protocol_v2.PrimaryProtocolLockSnapshotV2,
    protocol_v2.JsonArtifactSnapshotV2,
    Path,
    dict[str, Path],
]:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    lock = _matrix_boundary_lock_v2(result_root)
    grid_payload = _matrix_boundary_grid_v2(lock)
    grid_content = protocol_v2._canonical_json_bytes_v2(grid_payload)
    grid_path = protocol_v2.canonical_grid_path_v2(result_root)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_bytes(grid_content)
    grid_sha256 = hashlib.sha256(grid_content).hexdigest()
    fit_sha256: dict[str, str] = {}
    fit_paths: dict[str, Path] = {}
    for spec in protocol_v2.locked_fit_inventory_v2():
        fit_id = str(spec["fit_id"])
        path = protocol_v2.canonical_fit_path_v2(result_root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = protocol_v2._canonical_json_bytes_v2(
            _matrix_boundary_fit_v2(spec, lock, grid_sha256)
        )
        path.write_bytes(content)
        fit_paths[fit_id] = path
        fit_sha256[fit_id] = hashlib.sha256(content).hexdigest()
    receipt = _matrix_boundary_receipt_v2(
        result_root,
        lock,
        grid_sha256=grid_sha256,
        fit_sha256=fit_sha256,
    )
    return result_root, lock, receipt, grid_path, fit_paths


def _guard_matrix_boundary_before_raw_v2(
    monkeypatch: pytest.MonkeyPatch,
    lock: protocol_v2.PrimaryProtocolLockSnapshotV2,
) -> None:
    protocol_read = protocol_v2.read_regular_bytes_artifact_v2
    train_read = train_v2.read_regular_bytes_artifact_v2

    def guard(reader):
        def checked(path, **kwargs):
            if Path(path).suffix == ".pb":
                pytest.fail("matrix authorization opened a raw election")
            return reader(path, **kwargs)

        return checked

    monkeypatch.setattr(
        protocol_v2, "read_regular_bytes_artifact_v2", guard(protocol_read)
    )
    monkeypatch.setattr(train_v2, "read_regular_bytes_artifact_v2", guard(train_read))
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_load_protocol_inputs_snapshot_scoped_v2",
        lambda *args, **kwargs: pytest.fail(
            "raw evaluation scope opened before matrix authentication"
        ),
    )
    monkeypatch.setattr(
        protocol_v2,
        "parse_pb_file",
        lambda *args, **kwargs: pytest.fail("matrix authorization called parser"),
    )
    monkeypatch.setattr(
        protocol_v2,
        "priority_mes_outcome",
        lambda *args, **kwargs: pytest.fail("matrix authorization called solver"),
    )


def test_forged_receipt_without_grid_or_fits_cannot_unlock_raw_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    lock = _matrix_boundary_lock_v2(result_root)
    receipt = _matrix_boundary_receipt_v2(
        result_root,
        lock,
        grid_sha256="a" * 64,
        fit_sha256={
            str(spec["fit_id"]): "b" * 64
            for spec in protocol_v2.locked_fit_inventory_v2()
        },
    )
    _guard_matrix_boundary_before_raw_v2(monkeypatch, lock)

    with pytest.raises(RuntimeError, match="grid|matrix|missing|exist"):
        protocol_v2.load_evaluation_inputs_snapshot_v2(
            lock,
            result_root,
            tmp_path / "data" / "pb",
            tuple(protocol_v2.SPLIT_SHA256_V2),
            replay_receipt_snapshot=receipt,
        )


@pytest.mark.parametrize("mutation", ("missing", "mutated"))
def test_receipt_rejects_missing_or_mutated_referenced_grid(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root, lock, receipt, grid_path, _ = _stage_matrix_boundary_v2(tmp_path)
    if mutation == "missing":
        grid_path.unlink()
    else:
        grid_path.write_bytes(grid_path.read_bytes() + b"\n")
    _guard_matrix_boundary_before_raw_v2(monkeypatch, lock)

    with pytest.raises(RuntimeError, match="grid|matrix|missing|exist|provenance"):
        protocol_v2.load_evaluation_inputs_snapshot_v2(
            lock,
            result_root,
            tmp_path / "data" / "pb",
            tuple(protocol_v2.SPLIT_SHA256_V2),
            replay_receipt_snapshot=receipt,
        )


@pytest.mark.parametrize("mutation", ("missing", "mutated"))
def test_receipt_rejects_missing_or_mutated_referenced_fit(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root, lock, receipt, _, fit_paths = _stage_matrix_boundary_v2(tmp_path)
    fit_path = fit_paths[sorted(fit_paths)[-1]]
    if mutation == "missing":
        fit_path.unlink()
    else:
        fit_path.write_bytes(fit_path.read_bytes() + b"\n")
    _guard_matrix_boundary_before_raw_v2(monkeypatch, lock)

    with pytest.raises(RuntimeError, match="fit|matrix|missing|incomplete|digest"):
        protocol_v2.load_evaluation_inputs_snapshot_v2(
            lock,
            result_root,
            tmp_path / "data" / "pb",
            tuple(protocol_v2.SPLIT_SHA256_V2),
            replay_receipt_snapshot=receipt,
        )


def test_receipt_accepts_exact_canonical_complete_on_disk_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_root, lock, receipt, _, _ = _stage_matrix_boundary_v2(tmp_path)
    sentinel = object()
    monkeypatch.setattr(
        train_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args: lock,
    )
    monkeypatch.setattr(
        protocol_v2,
        "_load_protocol_inputs_snapshot_scoped_v2",
        lambda *args, **kwargs: sentinel,
    )

    assert protocol_v2.load_evaluation_inputs_snapshot_v2(
        lock,
        result_root,
        tmp_path / "data" / "pb",
        tuple(protocol_v2.SPLIT_SHA256_V2),
        replay_receipt_snapshot=receipt,
    ) is sentinel


def test_primary_structural_gate_validator_is_exact_and_type_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_hashes = {"src/rules.py": "a" * 64}
    monkeypatch.setattr(
        protocol_v2,
        "primary_structural_source_sha256_v2",
        lambda root: source_hashes,
    )
    payload = {
        "schema_version": 2,
        "status": "pass",
        "gate_scope": "pre-fit metadata and approval-semantics authentication only",
        "semantics_profile": protocol_v2.SEMANTICS_PROFILE,
        "semantics_receipt_sha256": "b" * 64,
        "corpus_semantic_sha256": (
            protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
        ),
        "source_sha256": source_hashes,
        "n_series": 19,
        "n_elections": 132,
        "manifest_file_bindings": 132,
        "approval_ballots_authenticated": 821_572,
        "duplicate_tokens_remaining": 0,
        "split_artifacts_authenticated": 7,
        "raw_election_files_opened": 0,
        "election_parser_calls": 0,
        "outcome_solver_calls": 0,
        "heldout_outcomes_scored": False,
    }
    path = tmp_path / "structural_gates.json"
    path.write_bytes(protocol_v2._canonical_json_bytes_v2(payload))

    assert protocol_v2.validate_structural_gates_v2(
        path,
        semantics_receipt_sha256="b" * 64,
        corpus_semantic_sha256=(
            protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
        ),
        repo_root=tmp_path,
    ) == payload

    wrong_type = copy.deepcopy(payload)
    wrong_type["n_elections"] = 132.0
    path.write_bytes(protocol_v2._canonical_json_bytes_v2(wrong_type))
    with pytest.raises(RuntimeError, match="integer|count|field"):
        protocol_v2.validate_structural_gates_v2(
            path,
            semantics_receipt_sha256="b" * 64,
            corpus_semantic_sha256=(
                protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
            ),
            repo_root=tmp_path,
        )

    leaked_solver = copy.deepcopy(payload)
    leaked_solver["outcome_solver_calls"] = 1
    path.write_bytes(protocol_v2._canonical_json_bytes_v2(leaked_solver))
    with pytest.raises(RuntimeError, match="solver|field|gate"):
        protocol_v2.validate_structural_gates_v2(
            path,
            semantics_receipt_sha256="b" * 64,
            corpus_semantic_sha256=(
                protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
            ),
            repo_root=tmp_path,
        )


def test_primary_structural_and_lock_writers_are_double_built_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    result_root = repo_root / "results" / "iclr_primary_warsaw_v2"
    data_dir = repo_root / "data" / "pb"
    data_dir.mkdir(parents=True)
    structural_payload = {"schema_version": 2, "status": "pass"}
    structural_calls: list[int] = []

    def fake_structural(*args):
        structural_calls.append(1)
        return copy.deepcopy(structural_payload)

    monkeypatch.setattr(
        protocol_v2,
        "build_primary_structural_report_v2",
        fake_structural,
    )
    monkeypatch.setattr(
        protocol_v2,
        "validate_structural_gates_v2",
        lambda *args, **kwargs: structural_payload,
    )
    structural = protocol_v2.write_primary_structural_gates_v2(
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
    )
    assert structural_calls == [1, 1]
    assert structural["path"] == str(result_root / "structural_gates.json")
    assert structural["sha256"] == hashlib.sha256(
        protocol_v2._canonical_json_bytes_v2(structural_payload)
    ).hexdigest()

    mandatory = _synthetic_lock_files(repo_root)
    monkeypatch.setattr(
        protocol_v2,
        "mandatory_protocol_files_v2",
        lambda *args, **kwargs: mandatory,
    )
    built = protocol_v2.write_primary_protocol_lock_v2(
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
    )
    assert built["path"] == str(result_root / "protocol_lock.json")
    snapshot = protocol_v2.verify_primary_warsaw_protocol_lock_snapshot_v2(
        result_root / "protocol_lock.json",
        repo_root,
        result_root,
        data_dir,
    )
    assert built["sha256"] == snapshot.sha256

    (result_root / "protocol_lock.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="overwrite|conflict|lock"):
        protocol_v2.write_primary_protocol_lock_v2(
            repo_root=repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )


def test_primary_protocol_cli_exposes_only_ordered_production_actions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        protocol_v2,
        "stage_protocol_inputs_v2",
        lambda **kwargs: calls.append("stage") or {"status": "staged"},
    )
    monkeypatch.setattr(
        protocol_v2,
        "write_primary_structural_gates_v2",
        lambda **kwargs: calls.append("structural-gate") or {"status": "pass"},
    )
    monkeypatch.setattr(
        protocol_v2,
        "write_primary_protocol_lock_v2",
        lambda **kwargs: calls.append("lock") or {"status": "locked"},
    )
    monkeypatch.setattr(
        protocol_v2,
        "verify_primary_warsaw_full_protocol_lock_snapshot_v2",
        lambda *args: calls.append("verify-lock")
        or SimpleNamespace(
            path=Path("protocol_lock.json"),
            sha256="d" * 64,
            payload={
                "mandatory_file_count": 1,
                "fit_inventory": protocol_v2.locked_fit_inventory_v2(),
            },
        ),
    )

    for command in ("stage", "structural-gate", "lock", "verify-lock"):
        protocol_v2.main([command])
        assert json.loads(capsys.readouterr().out)["status"] in {
            "staged",
            "pass",
            "locked",
            "verified",
        }
    assert calls == ["stage", "structural-gate", "lock", "verify-lock"]
