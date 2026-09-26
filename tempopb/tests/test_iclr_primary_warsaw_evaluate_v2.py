"""Contracts for the corrected post-hoc primary-Warsaw v2 evaluator.

The fixtures are deliberately small, but the inventory is the real locked
49-coordinate matrix. No test reads a held-out result from the historical
lineage and no test writes below ``results/``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import iclr_primary_warsaw_protocol_v2 as protocol_v2
import iclr_primary_warsaw_train_v2 as train_v2
from iclr_corpus import SeriesRef, Split
from iclr_env import EnvConfig
from iclr_outcome import PROJECT_FEATURES
from iclr_policy import FEATURE_NAMES
from iclr_train import SeriesData
from parse_pb import PBInstance, Project, Vote


EXPECTED_CONSUMER_IDS = (
    "baseline.primary",
    "baseline.prior_rules",
    "control.history_free",
    "control.senior_scalar",
    "control.static_age_lookup",
    "cross_district.bound10",
    "cross_district.bound40",
    "failure.outcome",
    "frontier.endowment",
    "frontier.outcome",
    "mechanism.attribution_coverage",
    "mechanism.payment_kernel",
    "mechanism.support_floor",
    "mechanism.theory_floor",
    "outcome.ablation",
    "outcome.per_series",
    "outcome.seed_stability",
    "outcome.significance",
    "primary.corpus",
    "robustness.demographic_partition",
    "summary.composition",
    "summary.endowment_seed_stability",
    "summary.endowment_significance",
    "summary.matched_target",
    "transfer.lodz",
)


def _evaluate_v2():
    return importlib.import_module("iclr_primary_warsaw_evaluate_v2")


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _synthetic_lock(result_root: Path):
    source_sha = {
        "iclr_primary_warsaw_evaluate_v2.py": _sha(b"evaluator source\n"),
        "iclr_primary_warsaw_train_v2.py": _sha(b"trainer source\n"),
    }
    tracked = {
        "artifact/corpus_manifest": {
            "path": "results/iclr_primary_warsaw_v2/corpus_manifest.json",
            "sha256": _sha(b"manifest\n"),
        },
        "artifact/approval_semantics_receipt": {
            "path": "results/iclr_primary_warsaw_v2/approval_semantics_receipt.json",
            "sha256": _sha(b"semantics\n"),
        },
        "artifact/structural_gates": {
            "path": "results/iclr_primary_warsaw_v2/structural_gates.json",
            "sha256": _sha(b"structural\n"),
        },
        **{
            f"split/{name}": {
                "path": f"results/iclr_primary_warsaw_v2/splits/{name}.json",
                "sha256": _sha(f"split:{name}\n".encode()),
            }
            for name in protocol_v2.SPLIT_SHA256_V2
        },
        **{
            f"source/{name}": {"path": f"src/{name}", "sha256": digest}
            for name, digest in source_sha.items()
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
        "corpus_semantic_sha256": protocol_v2.EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256,
    }
    content = _canonical_bytes(payload)
    return protocol_v2.PrimaryProtocolLockSnapshotV2(
        path=(result_root / "protocol_lock.json").absolute(),
        content=content,
        sha256=_sha(content),
        payload=payload,
    )


def _locked_provenance(lock_snapshot, split: str) -> dict[str, object]:
    tracked = lock_snapshot.payload["tracked_files"]
    return {
        "protocol_lock_sha256": lock_snapshot.sha256,
        "corpus_manifest_sha256": tracked["artifact/corpus_manifest"]["sha256"],
        "split_sha256": tracked[f"split/{split}"]["sha256"],
        "semantics_receipt_sha256": tracked[
            "artifact/approval_semantics_receipt"
        ]["sha256"],
        "corpus_semantic_sha256": lock_snapshot.payload[
            "corpus_semantic_sha256"
        ],
        "source_sha256": {
            label.removeprefix("source/"): row["sha256"]
            for label, row in sorted(tracked.items())
            if label.startswith("source/")
        },
    }


def _grid_payload(lock_snapshot) -> dict[str, object]:
    alphas = list(protocol_v2.STATIC_AGE_GRID_VALUES)
    losses = [float(1.0 + alpha) for alpha in alphas]
    return {
        "schema_version": 2,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "split": "temporal_2022",
        "training_only": True,
        "alphas": alphas,
        "train_worst_csd": losses,
        "selected_alpha": 0.0,
        "selection_rule": train_v2.GRID_SELECTION_RULE,
        "provenance": _locked_provenance(lock_snapshot, "temporal_2022"),
    }


def _fit_payload(spec: dict[str, object], lock_snapshot, grid_sha: str):
    arm = str(spec["arm"])
    feature_names = {
        "outcome": list(PROJECT_FEATURES),
        "endowment": list(FEATURE_NAMES),
        "static_age_lookup": list(train_v2.STATIC_AGE_FEATURE_NAMES),
    }[arm]
    weights = [0.0] * len(feature_names)
    population = 4 + int(3 * math.log(len(feature_names)))
    n_evals = int(spec["generations"]) * population
    if spec["family"] == "static_age_lookup":
        n_evals += 1
    provenance = _locked_provenance(lock_snapshot, str(spec["split"]))
    provenance["grid_sha256"] = (
        grid_sha if spec["family"] == "static_age_lookup" else None
    )
    payload = {
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
            "selected_weights": weights.copy(),
            "selection_source": "cmaes",
            "n_objective_evals": n_evals,
            "optimizer_config": {
                "sigma0": spec["sigma0"],
                "popsize": spec["popsize"],
                "generations": spec["generations"],
                "seed": spec["seed"],
                "bound": spec["bound"],
            },
        },
    }
    train_v2.validate_fit_payload_v2(
        payload,
        expected_spec=spec,
        expected_protocol_lock_sha256=lock_snapshot.sha256,
    )
    return payload


def _stage_complete_inventory(result_root: Path):
    lock_snapshot = _synthetic_lock(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    grid = _grid_payload(lock_snapshot)
    grid_bytes = _canonical_bytes(grid)
    grid_path = protocol_v2.canonical_grid_path_v2(result_root)
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_bytes(grid_bytes)
    for spec in protocol_v2.locked_fit_inventory_v2():
        path = protocol_v2.canonical_fit_path_v2(result_root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            _canonical_bytes(_fit_payload(spec, lock_snapshot, _sha(grid_bytes)))
        )
    return lock_snapshot, grid


def _minimal_row(fit_id: str) -> dict[str, object]:
    return {
        "fit_id": fit_id,
        "source_kind": "fit",
        "split": "temporal_2022",
        "view": "test",
        "scheme": "age_sex",
        "series": "Poland/Test/Unit",
        "scored_years": [2023],
        "year_outcomes": [
            {
                "year": 2022,
                "scored": False,
                "winners": ["1"],
                "spent": 1.0,
                "welfare": 1.0,
                "cost_welfare": 1.0,
                "exclusion": 0.0,
            },
            {
                "year": 2023,
                "scored": True,
                "winners": ["1", "2"],
                "spent": 2.0,
                "welfare": 2.0,
                "cost_welfare": 2.0,
                "exclusion": 0.0,
            },
        ],
        "metrics": {
            "worst_csd": 0.1,
            "worst_cohort": "age18-39|F",
            "mean_csd": 0.0,
            "welfare": 2.0,
            "cost_welfare": 2.0,
            "exclusion": 0.0,
        },
    }


def _tiny_instance(
    year: int,
    key: str = "Poland/Test/Unit",
    *,
    project_cost: float = 1.0,
) -> PBInstance:
    country, unit, subunit = key.split("/", 2)
    return PBInstance(
        path=f"test://primary-v2/{country}_{unit}_{subunit}_{year}.pb",
        meta={
            "country": country,
            "unit": unit,
            "subunit": subunit,
            "date_begin": f"{year}-01-01",
            "budget": "1",
            "vote_type": "approval",
        },
        projects={
            "p1": Project("p1", project_cost, 1),
            "p2": Project("p2", project_cost, 0),
        },
        votes=[
            Vote("v1", ("p1",), age=30, sex="F"),
            Vote("v2", ("p2",), age=70, sex="M"),
        ],
    )


def _tiny_series_data() -> SeriesData:
    instances = {2022: _tiny_instance(2022), 2023: _tiny_instance(2023)}
    ref = SeriesRef(
        key="Poland/Test/Unit",
        years=(2022, 2023),
        paths=(Path("2022.pb"), Path("2023.pb")),
    )
    return SeriesData(
        ref=ref,
        train_years=(2022,),
        test_years=(2023,),
        train_only={2022: instances[2022]},
        all_years=instances,
    )


def _contract_series_data(
    key: str,
    years: tuple[int, ...],
    *,
    train_years: tuple[int, ...],
    test_years: tuple[int, ...],
) -> SeriesData:
    instances = {
        year: _tiny_instance(year, key, project_cost=0.5) for year in years
    }
    ref = SeriesRef(
        key=key,
        years=years,
        paths=tuple(Path(f"{key.replace('/', '_')}_{year}.pb") for year in years),
    )
    return SeriesData(
        ref=ref,
        train_years=train_years,
        test_years=test_years,
        train_only={year: instances[year] for year in train_years},
        all_years=instances,
    )


def _contract_data_by_split() -> dict[str, list[SeriesData]]:
    warsaw = [f"Poland/Warszawa/District-{index:02d}" for index in range(18)]
    lodz = "Poland/Łódź/CITYWIDE"
    years_by_series = {
        key: tuple(range(2019, 2026)) for key in warsaw
    }
    years_by_series[warsaw[0]] = tuple(range(2016, 2026))
    years_by_series[lodz] = (2023, 2024, 2025)

    temporal = [
        _contract_series_data(
            key,
            years_by_series[key],
            train_years=tuple(year for year in years_by_series[key] if year <= 2022),
            test_years=(2023, 2024, 2025),
        )
        for key in warsaw
    ]
    city_out = [
        _contract_series_data(
            key,
            years_by_series[key],
            train_years=years_by_series[key] if key != lodz else (),
            test_years=years_by_series[key] if key == lodz else (),
        )
        for key in (*warsaw, lodz)
    ]
    result: dict[str, list[SeriesData]] = {
        "temporal_2022": temporal,
        "city_out_Poland_Łódź": city_out,
    }
    all_keys = (*warsaw, lodz)
    for fold in range(5):
        result[f"district_out_f{fold}of5"] = [
            _contract_series_data(
                key,
                years_by_series[key],
                train_years=years_by_series[key] if index % 5 != fold else (),
                test_years=years_by_series[key] if index % 5 == fold else (),
            )
            for index, key in enumerate(all_keys)
        ]
    assert set(result) == set(protocol_v2.SPLIT_SHA256_V2)
    assert sum(len(row.all_years) for row in city_out) == 132
    return result


def _minimal_per_series_payload(lock_snapshot) -> dict[str, object]:
    return {
        "schema_version": 2,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "rows": [
            _minimal_row(str(spec["fit_id"]))
            for spec in protocol_v2.locked_fit_inventory_v2()
        ],
        "provenance": {"protocol_lock_sha256": lock_snapshot.sha256},
    }


def _expected_consumer_sources() -> dict[str, list[str]]:
    rows = protocol_v2.locked_fit_inventory_v2()

    def ids(family: str, **filters: object) -> list[str]:
        return sorted(
            str(row["fit_id"])
            for row in rows
            if row["family"] == family
            and all(row.get(key) == value for key, value in filters.items())
        )

    headline = ids("outcome_frontier", seed=42, soft_welfare_target=1.0)
    corner = ids("outcome_frontier", seed=42, soft_welfare_target=0.0)
    main_endow = ids("temporal_endowment", seed=42)
    return {
        "baseline.primary": [],
        "baseline.prior_rules": [],
        "control.history_free": ids("history_free") + main_endow,
        "control.senior_scalar": [],
        "control.static_age_lookup": ids("static_age_lookup") + ids("history_free"),
        "cross_district.bound10": ids("district_endowment", bound=10.0),
        "cross_district.bound40": ids("district_endowment", bound=40.0),
        "failure.outcome": headline,
        "frontier.endowment": ids("endowment_frontier"),
        "frontier.outcome": ids("outcome_frontier"),
        "mechanism.attribution_coverage": sorted(main_endow + headline + corner),
        "mechanism.payment_kernel": corner,
        "mechanism.support_floor": ids("static_support_floor"),
        "mechanism.theory_floor": sorted(main_endow + headline + corner),
        "outcome.ablation": sorted(headline + ids("outcome_loo")),
        "outcome.per_series": headline,
        "outcome.seed_stability": ids(
            "outcome_frontier", soft_welfare_target=1.0
        ),
        "outcome.significance": ids("outcome_frontier", seed=42),
        "primary.corpus": [],
        "robustness.demographic_partition": sorted(main_endow + headline),
        "summary.composition": sorted(main_endow + headline + corner),
        "summary.endowment_seed_stability": ids("temporal_endowment"),
        "summary.endowment_significance": sorted(main_endow + headline),
        "summary.matched_target": sorted(
            headline + ids("endowment_frontier", soft_welfare_target=1.0)
        ),
        "transfer.lodz": sorted(main_endow + headline),
    }


def _minimal_consumers_payload(lock_snapshot) -> dict[str, object]:
    module = _evaluate_v2()
    sources = _expected_consumer_sources()
    consumers = {
        consumer_id: {
            "consumer_id": consumer_id,
            "legacy_paths": list(
                module.PAPER_CONSUMER_CONTRACT_V2[consumer_id]["legacy_paths"]
            ),
            "source_ids": sources[consumer_id],
            "policy_ids": [],
            "values": {"status": "synthetic"},
        }
        for consumer_id in EXPECTED_CONSUMER_IDS
    }
    return {
        "schema_version": 2,
        "classification": "corrected post-hoc replay",
        "consumer_ids": list(EXPECTED_CONSUMER_IDS),
        "consumers": consumers,
        "manuscript_macros": {"SyntheticPrimaryValue": "0.0000"},
        "provenance": {"protocol_lock_sha256": lock_snapshot.sha256},
    }


def _receipt_snapshot(result_root: Path, lock_snapshot, inventory):
    module = _evaluate_v2()
    return module.write_evaluation_replay_receipt_v2(
        result_root,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
        structural_gates_sha256=_sha(b"structural\n"),
        semantics_receipt_sha256=lock_snapshot.payload["tracked_files"][
            "artifact/approval_semantics_receipt"
        ]["sha256"],
    )


@pytest.fixture(scope="module")
def complete_evaluation_fixture(tmp_path_factory):
    module = _evaluate_v2()
    result_root = tmp_path_factory.mktemp("typed-primary-v2") / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    receipt = _receipt_snapshot(result_root, lock_snapshot, inventory)
    data_by_split = _contract_data_by_split()
    per_series = module.build_per_series_payload_v2(
        data_by_split,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
        replay_receipt_snapshot=receipt,
    )
    consumers = module.build_paper_consumers_payload_v2(
        per_series,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
        replay_receipt_snapshot=receipt,
        data_by_split=data_by_split,
        corpus_values={
            "n_series": 19,
            "n_elections": 132,
            "approval_ballots_checked": 264,
            "duplicate_tokens_remaining": 0,
            "winner_identity_checks": 264,
            "determinism_checks": 264,
            "budget_checks": 264,
            "unique_payer_round_checks": 264,
            "payment_rounds_checked": 264,
        },
    )
    return {
        "module": module,
        "result_root": result_root,
        "lock": lock_snapshot,
        "inventory": inventory,
        "receipt": receipt,
        "data": data_by_split,
        "per_series": per_series,
        "consumers": consumers,
    }


def test_primary_warsaw_v2_evaluator_contract_is_explicit() -> None:
    """Break caught: a renamed receipt or consumer silently drops a paper table."""

    module = _evaluate_v2()
    assert module.REPLAY_RECEIPT_RELATIVE_PATH == "evaluation_replay_started.json"
    assert tuple(sorted(module.PAPER_CONSUMER_CONTRACT_V2)) == EXPECTED_CONSUMER_IDS
    assert len(module.PAPER_CONSUMER_CONTRACT_V2) == 25
    assert all(
        tuple(row) == ("legacy_paths",)
        and type(row["legacy_paths"]) is tuple
        and row["legacy_paths"]
        for row in module.PAPER_CONSUMER_CONTRACT_V2.values()
    )


def test_complete_inventory_authenticates_exact_fit_and_grid_bytes(tmp_path: Path) -> None:
    """Break caught: opening proceeds from counts without authenticating bytes."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    snapshot = module.verify_complete_inventory_v2(lock_snapshot, result_root)

    assert snapshot.grid.sha256 == _sha(
        protocol_v2.canonical_grid_path_v2(result_root).read_bytes()
    )
    assert len(snapshot.fits) == 49
    assert tuple(item.spec["fit_id"] for item in snapshot.fits) == tuple(
        row["fit_id"] for row in protocol_v2.locked_fit_inventory_v2()
    )
    assert snapshot.fit_sha256 == {
        str(item.spec["fit_id"]): item.sha256 for item in snapshot.fits
    }


def test_complete_inventory_rejects_missing_and_extra_fit_before_opening(
    tmp_path: Path,
) -> None:
    """Break caught: a partial or stale fit tree is treated as complete."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    missing = protocol_v2.canonical_fit_path_v2(
        result_root, protocol_v2.locked_fit_inventory_v2()[0]
    )
    missing.unlink()
    with pytest.raises(RuntimeError, match="missing required.*fit"):
        module.verify_complete_inventory_v2(lock_snapshot, result_root)

    missing.write_bytes(
        _canonical_bytes(
            _fit_payload(
                protocol_v2.locked_fit_inventory_v2()[0],
                lock_snapshot,
                _sha(protocol_v2.canonical_grid_path_v2(result_root).read_bytes()),
            )
        )
    )
    extra = result_root / "fits" / "stale.json"
    extra.write_text("{}\n")
    with pytest.raises(RuntimeError, match="extra.*fit"):
        module.verify_complete_inventory_v2(lock_snapshot, result_root)


def test_complete_inventory_rejects_duplicate_json_and_grid_provenance_drift(
    tmp_path: Path,
) -> None:
    """Break caught: permissive JSON or a caller-authored grid enters the lineage."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    fit_path = protocol_v2.canonical_fit_path_v2(
        result_root, protocol_v2.locked_fit_inventory_v2()[0]
    )
    fit_path.write_bytes(b'{"schema_version":2,"schema_version":2}\n')
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        module.verify_complete_inventory_v2(lock_snapshot, result_root)

    result_root_2 = tmp_path / "primary_v2_grid"
    lock_snapshot_2, grid = _stage_complete_inventory(result_root_2)
    grid["provenance"]["protocol_lock_sha256"] = "0" * 64
    protocol_v2.canonical_grid_path_v2(result_root_2).write_bytes(
        _canonical_bytes(grid)
    )
    with pytest.raises(RuntimeError, match="grid|protocol-lock"):
        module.verify_complete_inventory_v2(lock_snapshot_2, result_root_2)


def test_replay_receipt_discloses_post_opening_status_and_binds_inventory(
    tmp_path: Path,
) -> None:
    """Break caught: corrected evidence is mislabeled as a fresh holdout."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    payload = module.build_evaluation_replay_receipt_payload_v2(
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
        structural_gates_sha256=_sha(b"structural\n"),
        semantics_receipt_sha256=lock_snapshot.payload["tracked_files"][
            "artifact/approval_semantics_receipt"
        ]["sha256"],
    )

    assert set(payload) == {
        "schema_version",
        "event",
        "classification",
        "heldout_outcomes_already_known",
        "fresh_holdout",
        "preregistered",
        "disclosure",
        "protocol_lock_sha256",
        "semantics_profile",
        "semantics_receipt_sha256",
        "corpus_semantic_sha256",
        "structural_gates_sha256",
        "training_grid_sha256",
        "fit_count",
        "fit_sha256",
    }
    assert payload["classification"] == "corrected post-hoc replay"
    assert payload["heldout_outcomes_already_known"] is True
    assert payload["fresh_holdout"] is False
    assert payload["preregistered"] is False
    assert "already known" in str(payload["disclosure"]).lower()
    assert payload["fit_count"] == 49
    assert payload["fit_sha256"] == inventory.fit_sha256


def test_replay_receipt_is_create_once_and_conflict_rejecting(tmp_path: Path) -> None:
    """Break caught: opening disclosure can be overwritten after seeing results."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    kwargs = {
        "lock_snapshot": lock_snapshot,
        "inventory_snapshot": inventory,
        "structural_gates_sha256": _sha(b"structural\n"),
        "semantics_receipt_sha256": lock_snapshot.payload["tracked_files"][
            "artifact/approval_semantics_receipt"
        ]["sha256"],
    }
    first = module.write_evaluation_replay_receipt_v2(result_root, **kwargs)
    second = module.write_evaluation_replay_receipt_v2(result_root, **kwargs)
    assert first.content == second.content
    assert first.sha256 == second.sha256

    changed = json.loads(first.content)
    changed["fresh_holdout"] = True
    first.path.write_bytes(_canonical_bytes(changed))
    with pytest.raises(RuntimeError, match="conflict|differs"):
        module.write_evaluation_replay_receipt_v2(result_root, **kwargs)


def test_per_series_schema_requires_all_fits_sorted_winners_and_unique_keys(
    tmp_path: Path,
) -> None:
    """Break caught: an evaluated fit or winner coordinate disappears silently."""

    module = _evaluate_v2()
    lock_snapshot = _synthetic_lock(tmp_path)
    payload = _minimal_per_series_payload(lock_snapshot)
    fit_ids = [row["fit_id"] for row in lock_snapshot.payload["fit_inventory"]]
    assert module.validate_per_series_payload_v2(
        payload, expected_fit_ids=fit_ids
    ) is payload

    missing = copy.deepcopy(payload)
    missing["rows"].pop()
    with pytest.raises(RuntimeError, match="fit coverage"):
        module.validate_per_series_payload_v2(missing, expected_fit_ids=fit_ids)

    unsorted = copy.deepcopy(payload)
    unsorted["rows"][0]["year_outcomes"][1]["winners"] = ["2", "1"]
    with pytest.raises(RuntimeError, match="winner.*sorted"):
        module.validate_per_series_payload_v2(unsorted, expected_fit_ids=fit_ids)

    duplicate = copy.deepcopy(payload)
    duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
    with pytest.raises(RuntimeError, match="duplicate.*row"):
        module.validate_per_series_payload_v2(duplicate, expected_fit_ids=fit_ids)


def test_paper_consumer_schema_rejects_missing_consumer_and_fit_source(
    complete_evaluation_fixture,
) -> None:
    """Break caught: a green run omits a manuscript consumer or fitted source."""

    module = complete_evaluation_fixture["module"]
    payload = copy.deepcopy(complete_evaluation_fixture["consumers"])
    inventory = [
        item.spec for item in complete_evaluation_fixture["inventory"].fits
    ]
    assert module.validate_paper_consumers_payload_v2(
        payload, inventory=inventory
    ) is payload

    missing = copy.deepcopy(payload)
    missing["consumer_ids"].remove("outcome.ablation")
    missing["consumers"].pop("outcome.ablation")
    with pytest.raises(RuntimeError, match="consumer coverage"):
        module.validate_paper_consumers_payload_v2(missing, inventory=inventory)

    extra = copy.deepcopy(payload)
    extra["consumer_ids"].append("unexpected.consumer")
    extra["consumers"]["unexpected.consumer"] = copy.deepcopy(
        extra["consumers"]["outcome.ablation"]
    )
    with pytest.raises(RuntimeError, match="consumer coverage"):
        module.validate_paper_consumers_payload_v2(extra, inventory=inventory)

    source_missing = copy.deepcopy(payload)
    source_missing["consumers"]["frontier.outcome"]["source_ids"].pop()
    with pytest.raises(RuntimeError, match="source coverage"):
        module.validate_paper_consumers_payload_v2(
            source_missing, inventory=inventory
        )


def test_typed_consumer_contract_rejects_generic_rows_summary_payload() -> None:
    """Break caught: named consumers still contain an untyped placeholder bag."""

    module = _evaluate_v2()
    assert set(module.PAPER_CONSUMER_VALUE_CONTRACT_V2) == set(
        EXPECTED_CONSUMER_IDS
    )
    assert len(
        {
            contract["schema"]
            for contract in module.PAPER_CONSUMER_VALUE_CONTRACT_V2.values()
        }
    ) == len(EXPECTED_CONSUMER_IDS)
    with pytest.raises(RuntimeError, match="typed|schema"):
        module.validate_consumer_values_v2(
            "outcome.ablation", {"rows": [], "summary": {}}
        )


def test_specialized_ablation_and_transfer_reconstruct_independent_tables(
    tmp_path: Path,
) -> None:
    """Break caught: specialized tables are labels over a generic summary."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    receipt = _receipt_snapshot(result_root, lock_snapshot, inventory)
    data_by_split = {
        name: [_tiny_series_data()] for name in protocol_v2.SPLIT_SHA256_V2
    }
    payload = module.build_per_series_payload_v2(
        data_by_split,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
        replay_receipt_snapshot=receipt,
    )
    rows = payload["rows"]
    specs = [item.spec for item in inventory.fits]

    ablation = module.build_outcome_ablation_values_v2(rows, specs)
    full_id = next(
        str(spec["fit_id"])
        for spec in specs
        if spec["family"] == "outcome_frontier"
        and spec["seed"] == 42
        and spec["soft_welfare_target"] == 1.0
    )
    full = next(
        row
        for row in rows
        if row["fit_id"] == full_id
        and row["split"] == "temporal_2022"
        and row["view"] == "test"
        and row["scheme"] == "age_sex"
    )
    assert ablation["schema"] == "outcome-ablation-table-v2"
    assert len(ablation["ablation_rows"]) == 6
    assert ablation["ablation_rows"][0] == {
        "ablated": "none (full model)",
        "source_id": full_id,
        "test_worst_csd": full["metrics"]["worst_csd"],
        "test_welfare": full["metrics"]["welfare"],
        "delta_vs_full": 0.0,
    }

    transfer = module.build_transfer_lodz_values_v2(rows, specs)
    assert transfer["schema"] == "lodz-transfer-table-v2"
    assert len(transfer["transfer_rows"]) == 8
    source_by_policy = {
        row["policy"]: row["source_id"] for row in transfer["transfer_rows"]
    }
    assert source_by_policy["historical"] == "historical"
    assert source_by_policy["learned-outcome-f1.00"].startswith("transfer/")
    for record in transfer["transfer_rows"]:
        reference = next(
            row
            for row in rows
            if row["fit_id"] == record["source_id"]
            and row["split"] == "city_out_Poland_Łódź"
            and row["view"] == "test"
            and row["scheme"] == "age_sex"
        )
        assert record["worst_csd"] == reference["metrics"]["worst_csd"]
        assert record["welfare"] == reference["metrics"]["welfare"]
        assert record["exclusion"] == reference["metrics"]["exclusion"]
        assert record["worst_cohort"] == reference["metrics"]["worst_cohort"]


def test_prepare_bundle_is_in_memory_hash_bound_and_summary_last(
    complete_evaluation_fixture,
) -> None:
    """Break caught: outputs leak early or the final commit does not bind them."""

    module = complete_evaluation_fixture["module"]
    result_root = complete_evaluation_fixture["result_root"]
    lock_snapshot = complete_evaluation_fixture["lock"]
    inventory = complete_evaluation_fixture["inventory"]
    receipt = complete_evaluation_fixture["receipt"]
    prepared = module.prepare_evaluation_bundle_v2(
        result_root,
        per_series_payload=complete_evaluation_fixture["per_series"],
        paper_consumers_payload=complete_evaluation_fixture["consumers"],
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
        replay_receipt_snapshot=receipt,
    )

    relative = [
        path.relative_to(result_root).as_posix() for path, _ in prepared.artifacts
    ]
    assert relative == [
        "evaluation/per_series.json",
        "evaluation/per_series.csv",
        "evaluation/paper_consumers.json",
        "evaluation/artifact_manifest.json",
        "evaluation/summary.json",
    ]
    assert prepared.summary_path == result_root / "evaluation" / "summary.json"
    assert not any(path.exists() for path, _ in prepared.artifacts)
    summary = json.loads(prepared.artifacts[-1][1])
    assert summary["status"] == "complete"
    assert summary["fit_count"] == 49
    assert summary["consumer_count"] == 25
    assert summary["replay_receipt_sha256"] == receipt.sha256
    assert set(summary["output_sha256"]) == set(relative[:-1])


def test_independent_rebuild_rejects_any_byte_difference(tmp_path: Path) -> None:
    """Break caught: one nondeterministic rebuild is sealed as official output."""

    module = _evaluate_v2()
    calls = {"count": 0}

    def stable() -> tuple[tuple[Path, bytes], ...]:
        calls["count"] += 1
        return ((tmp_path / "evaluation" / "summary.json", b"stable\n"),)

    expected = stable()
    assert module.require_independent_byte_rebuild_v2(stable) == expected
    assert calls["count"] == 3

    sequence = iter(
        [
            ((tmp_path / "a", b"first"),),
            ((tmp_path / "a", b"second"),),
        ]
    )
    with pytest.raises(RuntimeError, match="independent.*byte"):
        module.require_independent_byte_rebuild_v2(lambda: next(sequence))


def test_bundle_install_preflights_all_conflicts_and_check_only_never_writes(
    tmp_path: Path,
) -> None:
    """Break caught: a late conflict leaves a partial official bundle."""

    module = _evaluate_v2()
    paths = tuple(
        (tmp_path / "evaluation" / name, content)
        for name, content in (
            ("per_series.json", b"one\n"),
            ("per_series.csv", b"two\n"),
            ("paper_consumers.json", b"three\n"),
            ("artifact_manifest.json", b"four\n"),
            ("summary.json", b"commit\n"),
        )
    )
    prepared = module.PreparedEvaluationBundleV2(
        artifacts=paths,
        summary_path=paths[-1][0],
    )
    paths[-2][0].parent.mkdir(parents=True)
    paths[-2][0].write_bytes(b"conflict\n")
    with pytest.raises(RuntimeError, match="conflict"):
        module.install_prepared_evaluation_bundle_v2(prepared)
    assert not paths[0][0].exists()
    assert not paths[-1][0].exists()

    paths[-2][0].unlink()
    with pytest.raises(RuntimeError, match="missing.*check-only"):
        module.verify_installed_evaluation_bundle_v2(prepared)
    assert not paths[0][0].exists()

    module.install_prepared_evaluation_bundle_v2(prepared)
    module.verify_installed_evaluation_bundle_v2(prepared)
    paths[1][0].write_bytes(b"drift\n")
    with pytest.raises(RuntimeError, match="differs.*check-only"):
        module.verify_installed_evaluation_bundle_v2(prepared)


def test_authenticated_snapshots_detect_post_verification_drift(tmp_path: Path) -> None:
    """Break caught: a fit changes between pre-opening authentication and commit."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    module.assert_inventory_snapshots_unchanged_v2(inventory)
    inventory.fits[0].path.write_bytes(inventory.fits[0].content + b" ")
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        module.assert_inventory_snapshots_unchanged_v2(inventory)


def test_main_exposes_non_mutating_check_only_mode(capsys) -> None:
    """Break caught: the production command has no safe inspection mode."""

    module = _evaluate_v2()
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "--check-only" in output
    assert "corrected post-hoc" in output.lower()


def test_selector_episode_rows_preserve_warmup_and_exact_winner_trace() -> None:
    """Break caught: aggregate-only replay loses the auditable winner coordinate."""

    module = _evaluate_v2()
    data = _tiny_series_data()

    def choose_p2(instance, state):
        del instance, state
        return {"p2"}

    rows = module.evaluate_selector_rows_v2(
        source_id="fixed/unit",
        source_kind="fixed",
        split_name="temporal_2022",
        selector=choose_p2,
        data=[data],
        cfg=EnvConfig(),
    )

    assert [(row["view"], row["scored_years"]) for row in rows] == [
        ("test", [2023]),
        ("train", [2022]),
    ]
    test_row = rows[0]
    assert [year["year"] for year in test_row["year_outcomes"]] == [2022, 2023]
    assert [year["scored"] for year in test_row["year_outcomes"]] == [False, True]
    assert all(year["winners"] == ["p2"] for year in test_row["year_outcomes"])
    assert test_row["metrics"]["welfare"] == 1.0
    assert test_row["metrics"]["cost_welfare"] == 1.0


def test_fit_selector_factory_replays_all_three_locked_arms(tmp_path: Path) -> None:
    """Break caught: a fit family is evaluated through the wrong allocation arm."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    families = {
        "outcome_frontier": "outcome",
        "temporal_endowment": "endowment",
        "static_age_lookup": "static_age_lookup",
    }
    instance = _tiny_instance(2023)
    for family, arm in families.items():
        fit = next(item for item in inventory.fits if item.spec["family"] == family)
        assert fit.payload["arm"]["name"] == arm
        selector = module.selector_for_fit_v2(fit, EnvConfig())
        winners = selector(instance, module.RolloutState())
        assert winners <= set(instance.projects)


def test_complete_per_series_rebuild_covers_every_fit_view_and_fixed_policy(
    tmp_path: Path,
) -> None:
    """Break caught: the evaluator authenticates 49 fits but numerically skips one."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    receipt = _receipt_snapshot(result_root, lock_snapshot, inventory)
    split_names = set(protocol_v2.SPLIT_SHA256_V2)
    data_by_split = {name: [_tiny_series_data()] for name in split_names}

    payload = module.build_per_series_payload_v2(
        data_by_split,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
        replay_receipt_snapshot=receipt,
    )
    module.validate_per_series_payload_v2(
        payload,
        expected_fit_ids=[item.spec["fit_id"] for item in inventory.fits],
    )
    fit_rows = [row for row in payload["rows"] if row["source_kind"] == "fit"]
    coordinates = {
        (row["fit_id"], row["split"], row["view"], row["scheme"], row["series"])
        for row in fit_rows
    }
    assert len({row["fit_id"] for row in fit_rows}) == 49
    assert all(
        any(coord[0] == fit_id and coord[2] == view for coord in coordinates)
        for fit_id in inventory.fit_sha256
        for view in ("train", "test")
    )
    assert {
        row["fit_id"] for row in payload["rows"] if row["source_kind"] == "fixed"
    } >= {"historical", "greedy-count", "greedy-cost", "mes", "res-1.0"}
    canonical_split = {
        str(item.spec["fit_id"]): str(item.spec["split"])
        for item in inventory.fits
    }
    assert all(
        row["split"] == canonical_split[row["fit_id"]]
        for row in fit_rows
    )
    assert {
        row["fit_id"]
        for row in payload["rows"]
        if row["source_kind"] == "diagnostic"
        and row["split"] == "city_out_Poland_Łódź"
    } == {
        f"transfer/{item.spec['fit_id']}"
        for item in inventory.fits
        if (
            item.spec["family"] == "temporal_endowment"
            and item.spec["seed"] == 42
        )
        or (
            item.spec["family"] == "outcome_frontier"
            and item.spec["seed"] == 42
            and item.spec["soft_welfare_target"] in {0.85, 1.0}
        )
    }


def test_paper_consumers_are_computed_from_normalized_rows_not_placeholders(
    complete_evaluation_fixture,
) -> None:
    """Break caught: consumer coverage is nominal but carries no paper values."""

    module = complete_evaluation_fixture["module"]
    inventory = complete_evaluation_fixture["inventory"]
    consumers = complete_evaluation_fixture["consumers"]

    module.validate_paper_consumers_payload_v2(
        consumers, inventory=[item.spec for item in inventory.fits]
    )
    assert consumers["consumer_ids"] == list(EXPECTED_CONSUMER_IDS)
    assert consumers["manuscript_macros"]
    assert {
        row["values"]["schema"] for row in consumers["consumers"].values()
    } == {
        contract["schema"]
        for contract in module.PAPER_CONSUMER_VALUE_CONTRACT_V2.values()
    }
    assert all(
        module.validate_consumer_values_v2(consumer_id, row["values"])
        is row["values"]
        for consumer_id, row in consumers["consumers"].items()
    )
    assert consumers["consumers"]["baseline.primary"]["policy_ids"] == [
        "greedy-cost",
        "greedy-count",
        "llmrule-card",
        "llmrule-cost",
        "mes",
        "res-0.25",
        "res-0.5",
        "res-0.75",
        "res-1.0",
    ]
    main_endow = next(
        item.spec["fit_id"]
        for item in inventory.fits
        if item.spec["family"] == "temporal_endowment" and item.spec["seed"] == 42
    )
    outcome_transfers = sorted(
        item.spec["fit_id"]
        for item in inventory.fits
        if item.spec["family"] == "outcome_frontier"
        and item.spec["seed"] == 42
        and item.spec["soft_welfare_target"] in {0.85, 1.0}
    )
    assert consumers["consumers"]["transfer.lodz"]["policy_ids"] == sorted(
        [
            "greedy-cost",
            "greedy-count",
            "historical",
            "mes",
            "res-1.0",
            f"transfer/{main_endow}",
            *[f"transfer/{fit_id}" for fit_id in outcome_transfers],
        ]
    )
    protected_names = {
        line.split("{\\", 1)[1].split("}", 1)[0]
        for line in (Path(__file__).parents[2] / "iclr_paper/tex/numbers.tex")
        .read_text()
        .splitlines()
        if line.startswith("\\newcommand{\\")
    }
    assert set(consumers["manuscript_macros"]) <= protected_names


def test_receipt_authorized_loader_restores_full_test_views(
    monkeypatch, tmp_path: Path
) -> None:
    """Break caught: the training-safe split projection empties official tests."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    receipt = _receipt_snapshot(result_root, lock_snapshot, inventory)
    parent_ref = SeriesRef(
        key="Poland/Test/Unit",
        years=(2022, 2023),
        paths=(Path("2022.pb"), Path("2023.pb")),
    )
    training_only = Split(
        name="training-only",
        train=((parent_ref.key, (2022,)),),
        test=(),
    )
    split_artifacts = {
        name: SimpleNamespace(
            payload={
                "name": name,
                "note": "full authenticated split",
                "train": [[parent_ref.key, [2022]]],
                "test": [[parent_ref.key, [2023]]],
            }
        )
        for name in protocol_v2.SPLIT_SHA256_V2
    }
    snapshot = SimpleNamespace(
        splits={name: training_only for name in protocol_v2.SPLIT_SHA256_V2},
        split_artifacts=split_artifacts,
        index={parent_ref.key: parent_ref},
        semantics_receipt=SimpleNamespace(payload={}),
    )
    instances = {2022: _tiny_instance(2022), 2023: _tiny_instance(2023)}
    monkeypatch.setattr(
        protocol_v2,
        "validate_protocol_inputs_snapshot_v2",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "load_series_authenticated_v2",
        lambda ref, receipt_payload, years=None: {
            year: instances[year] for year in years
        },
    )

    loaded = module.load_evaluation_series_data_v2(
        snapshot,
        receipt,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory,
    )

    assert set(loaded) == set(protocol_v2.SPLIT_SHA256_V2)
    assert all(rows[0].train_years == (2022,) for rows in loaded.values())
    assert all(rows[0].test_years == (2023,) for rows in loaded.values())


def test_heldout_authorization_rejects_missing_forged_stale_and_wrong_path_receipts(
    tmp_path: Path,
) -> None:
    """Break caught: an in-memory dataclass bypasses the on-disk opening event."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    receipt = _receipt_snapshot(result_root, lock_snapshot, inventory)

    receipt.path.unlink()
    with pytest.raises(RuntimeError, match="receipt.*missing|missing.*receipt"):
        module._validate_replay_authorization_v2(
            receipt,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory,
        )

    receipt.path.write_bytes(receipt.content)
    forged_hash = module.ReplayReceiptSnapshotV2(
        path=receipt.path,
        content=receipt.content,
        sha256="0" * 64,
        payload=receipt.payload,
    )
    with pytest.raises(RuntimeError, match="receipt"):
        module._validate_replay_authorization_v2(
            forged_hash,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory,
        )

    wrong_path = module.ReplayReceiptSnapshotV2(
        path=result_root / "forged.json",
        content=receipt.content,
        sha256=receipt.sha256,
        payload=receipt.payload,
    )
    with pytest.raises(RuntimeError, match="receipt"):
        module._validate_replay_authorization_v2(
            wrong_path,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory,
        )

    stale = receipt.content + b" "
    receipt.path.write_bytes(stale)
    with pytest.raises(RuntimeError, match="receipt"):
        module._validate_replay_authorization_v2(
            receipt,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory,
        )

    changed_payload = copy.deepcopy(receipt.payload)
    changed_payload["extra"] = "forged"
    changed_content = _canonical_bytes(changed_payload)
    receipt.path.write_bytes(changed_content)
    forged_schema = module.ReplayReceiptSnapshotV2(
        path=receipt.path,
        content=changed_content,
        sha256=_sha(changed_content),
        payload=changed_payload,
    )
    with pytest.raises(RuntimeError, match="receipt"):
        module._validate_replay_authorization_v2(
            forged_schema,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory,
        )


def test_directory_lock_rejects_a_second_evaluation_writer(tmp_path: Path) -> None:
    """Break caught: two evaluators can both pass preflight and partially race."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    result_root.mkdir()
    with module.evaluation_writer_lock_v2(result_root):
        with pytest.raises(RuntimeError, match="single writer|already running"):
            with module.evaluation_writer_lock_v2(result_root):
                pass


def test_check_only_before_receipt_authenticates_inventory_without_opening(
    monkeypatch, tmp_path: Path
) -> None:
    """Break caught: a harmless readiness check creates the post-opening receipt."""

    module = _evaluate_v2()
    result_root = tmp_path / "primary_v2"
    lock_snapshot, _ = _stage_complete_inventory(result_root)
    inventory = module.verify_complete_inventory_v2(lock_snapshot, result_root)
    structural_path = result_root / "structural_gates.json"
    structural_path.write_bytes(b"structural\n")
    monkeypatch.setattr(
        protocol_v2,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args, **kwargs: lock_snapshot,
    )
    monkeypatch.setattr(
        protocol_v2, "validate_structural_gates_v2", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        train_v2,
        "verify_complete_matrix_v2",
        lambda **kwargs: {
            "schema_version": 2,
            "status": "complete",
            "protocol_lock_sha256": lock_snapshot.sha256,
            "grid_sha256": inventory.grid.sha256,
            "fit_count": 49,
            "fit_sha256": inventory.fit_sha256,
        },
    )
    monkeypatch.setattr(module, "verify_complete_inventory_v2", lambda *args: inventory)
    monkeypatch.setattr(
        protocol_v2,
        "load_protocol_inputs_snapshot_v2",
        lambda *args, **kwargs: pytest.fail("check-only opened held-out inputs"),
    )

    result = module.execute_locked_evaluation_v2(
        result_root,
        repo_root=tmp_path,
        data_dir=tmp_path / "pb",
        check_only=True,
    )

    assert result["status"] == "ready_for_corrected_posthoc_replay"
    assert result["fit_count"] == 49
    assert not (result_root / module.REPLAY_RECEIPT_RELATIVE_PATH).exists()
