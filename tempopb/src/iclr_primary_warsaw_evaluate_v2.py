"""Evaluate the corrected post-hoc primary-Warsaw protocol-v2 lineage.

The evaluator is intentionally append-only.  It authenticates the complete
49-fit matrix and the training-only scalar grid before creating the replay
receipt, evaluates only protocol-v2 instances, builds every output in memory
twice, and installs the final summary last.  Warsaw held-out outcomes were
already known before this lineage was created; nothing here is a fresh holdout
or a preregistration.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from cohorts import (
    AGE_BRACKETS,
    age_bracket,
    cohort_of,
    cumulative_share_deficit,
    group_outcome,
    historical_winners,
)
from iclr_approval_semantics_v2 import SEMANTICS_PROFILE, load_series_authenticated_v2
from iclr_corpus import SeriesRef
from iclr_env import EnvConfig, RolloutState, endowment_selector, res_policy, uniform_policy
from iclr_multicity_protocol_v2 import (
    preflight_immutable_bytes_artifact_v2,
    read_regular_bytes_artifact_v2,
    write_immutable_bytes_artifact_v2,
)
from iclr_outcome import (
    clear_project_cache,
    cost_effective_weights,
    greedy_equivalent_weights,
    llmrule_card_selector,
    llmrule_cost_selector,
    payment_gated_score_selector,
    project_features,
    score_selector,
)
from iclr_policy import clear_feature_cache, instance_features, linear_policy
import iclr_primary_warsaw_protocol_v2 as protocol_v2
import iclr_primary_warsaw_train_v2 as train_v2
from iclr_stats import exact_paired_sign_flip_pvalue
from iclr_train import SeriesData
from rules import mes_with_endowments
from run_experiments import approval_welfare, cost_welfare, exclusion_rate


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results" / "iclr_primary_warsaw_v2"
REPLAY_RECEIPT_RELATIVE_PATH = "evaluation_replay_started.json"

# This registry is the fail-closed bridge between the 49-coordinate fit matrix
# and every current primary-Warsaw manuscript input.  ``legacy_paths`` are
# identities for the delta audit; this evaluator never writes to them.
PAPER_CONSUMER_CONTRACT_V2: dict[str, dict[str, tuple[str, ...]]] = {
    "baseline.primary": {"legacy_paths": ("results/iclr_baselines.csv",)},
    "baseline.prior_rules": {
        "legacy_paths": ("results/iclr_llmrule_baseline.csv",)
    },
    "control.history_free": {
        "legacy_paths": (
            "results/iclr_static_tilt.csv",
            "analysis-output/ml-contribution-audit/history_free_seed42_g30.json",
        )
    },
    "control.senior_scalar": {
        "legacy_paths": ("results/iclr_senior_tilt.csv",)
    },
    "control.static_age_lookup": {
        "legacy_paths": (
            "analysis-output/ml-contribution-audit/static_age_lookup_seed42_g30.json",
        )
    },
    "cross_district.bound10": {
        "legacy_paths": (
            "results/iclr_cross_district_summary.csv",
            "results/iclr_cross_district_per_series.csv",
            "results/iclr_cross_district_manifest.json",
        )
    },
    "cross_district.bound40": {
        "legacy_paths": (
            "results/iclr_cross_district_b40_summary.csv",
            "results/iclr_cross_district_b40_per_series.csv",
            "results/iclr_cross_district_b40_manifest.json",
        )
    },
    "failure.outcome": {
        "legacy_paths": (
            "results/iclr_failure_analysis.csv",
            "results/iclr_worst_cohort_mix.json",
        )
    },
    "frontier.endowment": {
        "legacy_paths": (
            "results/iclr_frontier/frontier_endowment_temporal_2022_seed42.json",
            "results/iclr_frontier/frontier_endowment_temporal_2022_seed42.csv",
        )
    },
    "frontier.outcome": {
        "legacy_paths": tuple(
            path
            for seed in (1, 2, 3, 42)
            for path in (
                f"results/iclr_frontier/frontier_outcome_temporal_2022_seed{seed}.json",
                f"results/iclr_frontier/frontier_outcome_temporal_2022_seed{seed}.csv",
            )
        )
    },
    "mechanism.attribution_coverage": {
        "legacy_paths": ("results/iclr_attribution_coverage.csv",)
    },
    "mechanism.payment_kernel": {
        "legacy_paths": (
            "results/iclr_payment_intervention_summary.csv",
            "results/iclr_payment_intervention_per_series.csv",
            "results/iclr_payment_intervention_manifest.json",
        )
    },
    "mechanism.support_floor": {
        "legacy_paths": ("results/iclr_identification.csv",)
    },
    "mechanism.theory_floor": {
        "legacy_paths": (
            "results/iclr_theory_check.csv",
            "results/iclr_infeasible_spend.csv",
        )
    },
    "outcome.ablation": {"legacy_paths": ("results/iclr_ablation.csv",)},
    "outcome.per_series": {
        "legacy_paths": (
            "results/iclr_per_series.csv",
            "results/iclr_two_arm_per_series.csv",
        )
    },
    "outcome.seed_stability": {
        "legacy_paths": (
            "results/iclr_frontier/frontier_outcome_temporal_2022_seed1.json",
            "results/iclr_frontier/frontier_outcome_temporal_2022_seed2.json",
            "results/iclr_frontier/frontier_outcome_temporal_2022_seed3.json",
            "results/iclr_frontier/frontier_outcome_temporal_2022_seed42.json",
        )
    },
    "outcome.significance": {
        "legacy_paths": ("results/iclr_significance.csv",)
    },
    "primary.corpus": {
        "legacy_paths": (
            "results/frozen_instances.txt",
            "results/iclr_instance_granularity.csv",
        )
    },
    "robustness.demographic_partition": {
        "legacy_paths": ("results/iclr_scheme_robustness.csv",)
    },
    "summary.composition": {
        "legacy_paths": ("results/iclr_composition.csv",)
    },
    "summary.endowment_seed_stability": {
        "legacy_paths": (
            "results/iclr_train/run_endow_seed1.json",
            "results/iclr_train/run_endow_seed2.json",
            "results/iclr_train/run_main_seed42.json",
        )
    },
    "summary.endowment_significance": {
        "legacy_paths": (
            "results/iclr_endow_significance.csv",
            "results/iclr_two_arm_per_series.csv",
            "results/iclr_composition.csv",
        )
    },
    "summary.matched_target": {
        "legacy_paths": ("results/iclr_matched_target.csv",)
    },
    "transfer.lodz": {"legacy_paths": ("results/iclr_transfer.csv",)},
}


# Every consumer has a distinct value schema.  Counts describe the current
# protected primary-Warsaw interfaces (not whatever rows happen to be present
# in a rebuild), so incomplete or over-broad reconstructions fail closed.
PAPER_CONSUMER_VALUE_CONTRACT_V2: dict[str, dict[str, object]] = {
    "baseline.primary": {
        "schema": "primary-baseline-table-v2",
        "lists": {"series_rows": 19},
    },
    "baseline.prior_rules": {
        "schema": "prior-rule-table-v2",
        "lists": {"rule_view_rows": 76},
    },
    "control.history_free": {
        "schema": "history-free-control-v2",
        "lists": {"policy_rows": 4, "fit_records": 2},
    },
    "control.senior_scalar": {
        "schema": "senior-scalar-control-v2",
        "lists": {},
    },
    "control.static_age_lookup": {
        "schema": "static-age-lookup-control-v2",
        "lists": {"policy_rows": 3, "contrast_rows": 2},
    },
    "cross_district.bound10": {
        "schema": "cross-district-bound10-v2",
        "lists": {
            "series_policy_rows": 133,
            "contrast_rows": 6,
            "fold_records": 5,
        },
    },
    "cross_district.bound40": {
        "schema": "cross-district-bound40-v2",
        "lists": {
            "series_policy_rows": 133,
            "contrast_rows": 6,
            "fold_records": 5,
        },
    },
    "failure.outcome": {
        "schema": "outcome-failure-diagnostics-v2",
        "lists": {"series_diagnostics": 18},
    },
    "frontier.endowment": {
        "schema": "endowment-frontier-v2",
        "lists": {"point_rows": 3, "baseline_rows": 18},
    },
    "frontier.outcome": {
        "schema": "outcome-frontier-v2",
        "lists": {"point_rows": 24, "baseline_rows": 18},
    },
    "mechanism.attribution_coverage": {
        "schema": "attribution-coverage-v2",
        "lists": {"policy_rows": 4},
    },
    "mechanism.payment_kernel": {
        "schema": "payment-kernel-intervention-v2",
        "lists": {
            "series_kernel_rows": 216,
            "kernel_summary_rows": 12,
        },
    },
    "mechanism.support_floor": {
        "schema": "static-support-floor-identification-v2",
        "lists": {"identification_rows": 2},
    },
    "mechanism.theory_floor": {
        "schema": "empirical-support-floor-v2",
        "lists": {"theory_rows": 324, "infeasible_spend_rows": 6},
    },
    "outcome.ablation": {
        "schema": "outcome-ablation-table-v2",
        "lists": {"ablation_rows": 6},
    },
    "outcome.per_series": {
        "schema": "outcome-series-tables-v2",
        "lists": {"learned_rows": 18, "two_arm_rows": 18},
    },
    "outcome.seed_stability": {
        "schema": "outcome-seed-stability-v2",
        "lists": {"seed_rows": 4},
    },
    "outcome.significance": {
        "schema": "outcome-significance-v2",
        "lists": {"contrast_rows": 36},
    },
    "primary.corpus": {
        "schema": "primary-corpus-inventory-v2",
        "lists": {
            "instance_rows": 132,
            "granularity_rows": 1,
            "excluded_legacy_rows": 1,
        },
    },
    "robustness.demographic_partition": {
        "schema": "demographic-partition-robustness-v2",
        "lists": {"scheme_rows": 15},
    },
    "summary.composition": {
        "schema": "allocation-composition-v2",
        "lists": {"composition_rows": 7},
    },
    "summary.endowment_seed_stability": {
        "schema": "endowment-seed-stability-v2",
        "lists": {"seed_rows": 3},
    },
    "summary.endowment_significance": {
        "schema": "endowment-significance-suite-v2",
        "lists": {
            "contrast_rows": 5,
            "two_arm_rows": 18,
            "composition_rows": 7,
        },
    },
    "summary.matched_target": {
        "schema": "matched-target-contrast-v2",
        "lists": {"matched_rows": 1},
    },
    "transfer.lodz": {
        "schema": "lodz-transfer-table-v2",
        "lists": {"transfer_rows": 8},
    },
}


@dataclass(frozen=True)
class VerifiedFitSnapshotV2:
    """One exact fit byte image authenticated against its locked coordinate."""

    spec: Mapping[str, object]
    path: Path
    content: bytes
    sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class VerifiedEvaluationInventoryV2:
    """The complete pre-opening grid and 49-fit byte snapshot."""

    grid: train_v2.JsonSnapshotV2
    fits: tuple[VerifiedFitSnapshotV2, ...]

    @property
    def fit_sha256(self) -> dict[str, str]:
        return {
            str(snapshot.spec["fit_id"]): snapshot.sha256
            for snapshot in self.fits
        }


@dataclass(frozen=True)
class ReplayReceiptSnapshotV2:
    """The immutable disclosure receipt that precedes held-out replay."""

    path: Path
    content: bytes
    sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class PreparedEvaluationBundleV2:
    """Conflict-preflighted bytes; the final tuple entry is the commit record."""

    artifacts: tuple[tuple[Path, bytes], ...]
    summary_path: Path


def _canonical_json_bytes_v2(payload: object) -> bytes:
    _require_finite_json_v2(payload, label="primary-v2 evaluation JSON")
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_finite_json_v2(value: object, *, label: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{label} contains a non-finite number")
        return
    if type(value) is list:
        for child in value:
            _require_finite_json_v2(child, label=label)
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise RuntimeError(f"{label} contains a non-string key")
            _require_finite_json_v2(child, label=label)
        return
    raise RuntimeError(f"{label} contains a non-JSON value")


def _tracked_digest_v2(lock_payload: Mapping[str, object], label: str) -> str:
    tracked = lock_payload.get("tracked_files")
    row = tracked.get(label) if isinstance(tracked, Mapping) else None
    digest = row.get("sha256") if isinstance(row, Mapping) else None
    if not _is_sha256(digest):
        raise RuntimeError(f"primary-v2 lock is missing tracked digest {label}")
    return str(digest)


def _fit_ids(
    inventory: Sequence[Mapping[str, object]],
    family: str,
    **filters: object,
) -> list[str]:
    return sorted(
        str(row["fit_id"])
        for row in inventory
        if row.get("family") == family
        and all(row.get(key) == value for key, value in filters.items())
    )


def expected_consumer_source_ids_v2(
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, list[str]]:
    """Map every manuscript consumer to its exact locked fit dependencies."""

    rows = list(inventory)
    protocol_v2.validate_locked_fit_inventory_v2(rows)
    headline = _fit_ids(
        rows, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    aggressive = _fit_ids(
        rows, "outcome_frontier", seed=42, soft_welfare_target=0.85
    )
    corner = _fit_ids(
        rows, "outcome_frontier", seed=42, soft_welfare_target=0.0
    )
    main_endow = _fit_ids(rows, "temporal_endowment", seed=42)
    result = {
        "baseline.primary": [],
        "baseline.prior_rules": [],
        "control.history_free": _fit_ids(rows, "history_free") + main_endow,
        "control.senior_scalar": [],
        "control.static_age_lookup": _fit_ids(rows, "static_age_lookup")
        + _fit_ids(rows, "history_free"),
        "cross_district.bound10": _fit_ids(
            rows, "district_endowment", bound=10.0
        ),
        "cross_district.bound40": _fit_ids(
            rows, "district_endowment", bound=40.0
        ),
        "failure.outcome": headline,
        "frontier.endowment": _fit_ids(rows, "endowment_frontier"),
        "frontier.outcome": _fit_ids(rows, "outcome_frontier"),
        "mechanism.attribution_coverage": sorted(main_endow + headline + corner),
        "mechanism.payment_kernel": sorted(corner + aggressive + headline),
        "mechanism.support_floor": _fit_ids(rows, "static_support_floor"),
        "mechanism.theory_floor": sorted(main_endow + headline + corner),
        "outcome.ablation": sorted(headline + _fit_ids(rows, "outcome_loo")),
        "outcome.per_series": sorted(main_endow + headline),
        "outcome.seed_stability": _fit_ids(
            rows, "outcome_frontier", soft_welfare_target=1.0
        ),
        "outcome.significance": _fit_ids(rows, "outcome_frontier", seed=42),
        "primary.corpus": [],
        "robustness.demographic_partition": sorted(main_endow + headline),
        "summary.composition": sorted(main_endow + headline + corner),
        "summary.endowment_seed_stability": _fit_ids(rows, "temporal_endowment"),
        "summary.endowment_significance": sorted(main_endow + headline),
        "summary.matched_target": sorted(
            headline
            + _fit_ids(
                rows,
                "endowment_frontier",
                soft_welfare_target=1.0,
            )
        ),
        "transfer.lodz": sorted(main_endow + aggressive + headline),
    }
    if set(result) != set(PAPER_CONSUMER_CONTRACT_V2):
        raise RuntimeError("primary-v2 internal consumer registry is incomplete")
    covered = {fit_id for source_ids in result.values() for fit_id in source_ids}
    expected = {str(row["fit_id"]) for row in rows}
    if covered != expected:
        raise RuntimeError("primary-v2 consumer registry does not cover all 49 fits")
    return result


def verify_complete_inventory_v2(
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    result_root: Path,
) -> VerifiedEvaluationInventoryV2:
    """Authenticate the exact grid and 49-fit tree without reading outcomes."""

    payload = lock_snapshot.payload
    if (
        type(payload) is not dict
        or payload.get("schema_version") != 2
        or type(payload.get("schema_version")) is not int
        or payload.get("lock_profile") != protocol_v2.LOCK_PROFILE_V2
        or not protocol_v2.exact_json_equal_v2(
            payload.get("fit_inventory"), protocol_v2.locked_fit_inventory_v2()
        )
    ):
        raise RuntimeError("primary-v2 evaluation requires the exact protocol lock")
    inventory = payload["fit_inventory"]
    protocol_v2.validate_locked_fit_inventory_v2(inventory)
    static_spec = next(
        row for row in inventory if row["family"] == "static_age_lookup"
    )
    grid = train_v2.load_static_age_grid_dependency_v2(
        static_spec,
        lock_snapshot,
        result_root=Path(result_root),
    )

    expected_paths = {
        protocol_v2.canonical_fit_path_v2(Path(result_root), spec).absolute()
        for spec in inventory
    }
    fits_root = Path(result_root).absolute() / "fits"
    if not fits_root.is_dir() or fits_root.is_symlink():
        raise RuntimeError("missing required primary-v2 fit tree")
    actual_paths: set[Path] = set()
    for path in fits_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"primary-v2 fit tree contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(path.absolute())
    extras = sorted(actual_paths - expected_paths)
    if extras:
        raise RuntimeError(f"extra primary-v2 fit artifact: {extras[0]}")

    snapshots: list[VerifiedFitSnapshotV2] = []
    for spec in inventory:
        path = protocol_v2.canonical_fit_path_v2(Path(result_root), spec).absolute()
        if path not in actual_paths:
            raise RuntimeError(f"missing required primary-v2 fit: {path}")
        content = read_regular_bytes_artifact_v2(
            path, label=f"primary-v2 fit {spec['fit_id']}"
        )
        assert content is not None
        fit_payload = protocol_v2.load_json_object_bytes_strict_v2(
            content, label=f"primary-v2 fit {spec['fit_id']}"
        )
        train_v2.validate_fit_payload_v2(
            fit_payload,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock_snapshot.sha256,
        )
        grid_sha = grid.sha256 if spec["family"] == "static_age_lookup" else None
        train_v2._validate_locked_provenance_v2(
            fit_payload.get("provenance"),
            lock_snapshot,
            split_name=str(spec["split"]),
            grid_sha256=grid_sha,
        )
        snapshots.append(
            VerifiedFitSnapshotV2(
                spec=spec,
                path=path,
                content=content,
                sha256=hashlib.sha256(content).hexdigest(),
                payload=fit_payload,
            )
        )
    if len(snapshots) != 49:
        raise RuntimeError("primary-v2 complete fit inventory count differs")
    return VerifiedEvaluationInventoryV2(grid=grid, fits=tuple(snapshots))


def assert_inventory_snapshots_unchanged_v2(
    snapshot: VerifiedEvaluationInventoryV2,
) -> None:
    """Reject fit or grid drift after the pre-opening byte snapshot."""

    artifacts = ((snapshot.grid.path, snapshot.grid.content, "training grid"),) + tuple(
        (fit.path, fit.content, f"fit {fit.spec['fit_id']}") for fit in snapshot.fits
    )
    for path, expected, label in artifacts:
        current = read_regular_bytes_artifact_v2(
            path, label=f"primary-v2 {label}"
        )
        if current != expected:
            raise RuntimeError(f"primary-v2 {label} changed during evaluation")


def build_evaluation_replay_receipt_payload_v2(
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    structural_gates_sha256: str,
    semantics_receipt_sha256: str,
) -> dict[str, object]:
    """Build the explicit already-known-outcome disclosure receipt."""

    if not _is_sha256(structural_gates_sha256) or not _is_sha256(
        semantics_receipt_sha256
    ):
        raise RuntimeError("primary-v2 replay receipt digests are malformed")
    if semantics_receipt_sha256 != _tracked_digest_v2(
        lock_snapshot.payload, "artifact/approval_semantics_receipt"
    ):
        raise RuntimeError("primary-v2 semantics receipt differs from the lock")
    if structural_gates_sha256 != _tracked_digest_v2(
        lock_snapshot.payload, "artifact/structural_gates"
    ):
        raise RuntimeError("primary-v2 structural gates differ from the lock")
    if len(inventory_snapshot.fits) != 49:
        raise RuntimeError("primary-v2 replay receipt requires all 49 fits")
    payload: dict[str, object] = {
        "schema_version": 2,
        "event": "evaluation replay started",
        "classification": "corrected post-hoc replay",
        "heldout_outcomes_already_known": True,
        "fresh_holdout": False,
        "preregistered": False,
        "disclosure": (
            "Warsaw held-out outcomes were already known before this corrected "
            "post-hoc replay; this receipt is not a fresh holdout or preregistration."
        ),
        "protocol_lock_sha256": lock_snapshot.sha256,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_receipt_sha256,
        "corpus_semantic_sha256": lock_snapshot.payload[
            "corpus_semantic_sha256"
        ],
        "structural_gates_sha256": structural_gates_sha256,
        "training_grid_sha256": inventory_snapshot.grid.sha256,
        "fit_count": 49,
        "fit_sha256": inventory_snapshot.fit_sha256,
    }
    _require_finite_json_v2(payload, label="primary-v2 replay receipt")
    return payload


def _validate_replay_receipt_payload_v2(
    payload: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    if not protocol_v2.exact_json_equal_v2(payload, expected):
        raise RuntimeError("primary-v2 evaluation replay receipt differs")


def write_evaluation_replay_receipt_v2(
    result_root: Path,
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    structural_gates_sha256: str,
    semantics_receipt_sha256: str,
) -> ReplayReceiptSnapshotV2:
    """Create once (or byte-confirm) the post-opening replay receipt."""

    payload = build_evaluation_replay_receipt_payload_v2(
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
        structural_gates_sha256=structural_gates_sha256,
        semantics_receipt_sha256=semantics_receipt_sha256,
    )
    content = _canonical_json_bytes_v2(payload)
    path = Path(result_root).absolute() / REPLAY_RECEIPT_RELATIVE_PATH
    try:
        preflight_immutable_bytes_artifact_v2(
            path, content, label="primary-v2 evaluation replay receipt"
        )
        write_immutable_bytes_artifact_v2(
            path, content, label="primary-v2 evaluation replay receipt"
        )
    except RuntimeError as exc:
        raise RuntimeError(f"primary-v2 replay receipt conflict: {exc}") from exc
    observed = read_regular_bytes_artifact_v2(
        path, label="primary-v2 evaluation replay receipt"
    )
    assert observed is not None
    parsed = protocol_v2.load_json_object_bytes_strict_v2(
        observed, label="primary-v2 evaluation replay receipt"
    )
    _validate_replay_receipt_payload_v2(parsed, payload)
    return ReplayReceiptSnapshotV2(
        path=path,
        content=observed,
        sha256=hashlib.sha256(observed).hexdigest(),
        payload=parsed,
    )


def load_evaluation_replay_receipt_v2(
    result_root: Path,
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    structural_gates_sha256: str,
    semantics_receipt_sha256: str,
    allow_missing: bool = False,
) -> ReplayReceiptSnapshotV2 | None:
    """Read and authenticate an existing receipt without creating it."""

    expected = build_evaluation_replay_receipt_payload_v2(
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
        structural_gates_sha256=structural_gates_sha256,
        semantics_receipt_sha256=semantics_receipt_sha256,
    )
    path = Path(result_root).absolute() / REPLAY_RECEIPT_RELATIVE_PATH
    content = read_regular_bytes_artifact_v2(
        path,
        label="primary-v2 evaluation replay receipt",
        allow_missing=allow_missing,
    )
    if content is None:
        return None
    payload = protocol_v2.load_json_object_bytes_strict_v2(
        content, label="primary-v2 evaluation replay receipt"
    )
    _validate_replay_receipt_payload_v2(payload, expected)
    return ReplayReceiptSnapshotV2(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
    )


def _validate_replay_authorization_v2(
    replay_receipt_snapshot: ReplayReceiptSnapshotV2,
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
) -> None:
    if type(replay_receipt_snapshot) is not ReplayReceiptSnapshotV2:
        raise RuntimeError("primary-v2 replay receipt snapshot type differs")
    payload = replay_receipt_snapshot.payload
    canonical_path = (
        lock_snapshot.path.absolute().parent / REPLAY_RECEIPT_RELATIVE_PATH
    )
    if replay_receipt_snapshot.path.absolute() != canonical_path:
        raise RuntimeError("primary-v2 replay receipt path is not canonical")
    observed = read_regular_bytes_artifact_v2(
        canonical_path, label="primary-v2 evaluation replay receipt"
    )
    if observed != replay_receipt_snapshot.content:
        raise RuntimeError("primary-v2 replay receipt bytes are stale or forged")
    digest = hashlib.sha256(replay_receipt_snapshot.content).hexdigest()
    if digest != replay_receipt_snapshot.sha256:
        raise RuntimeError("primary-v2 replay receipt digest is forged")
    parsed = protocol_v2.load_json_object_bytes_strict_v2(
        replay_receipt_snapshot.content,
        label="primary-v2 evaluation replay receipt",
    )
    if not protocol_v2.exact_json_equal_v2(parsed, payload):
        raise RuntimeError("primary-v2 replay receipt payload differs from its bytes")
    expected = build_evaluation_replay_receipt_payload_v2(
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
        structural_gates_sha256=_tracked_digest_v2(
            lock_snapshot.payload, "artifact/structural_gates"
        ),
        semantics_receipt_sha256=_tracked_digest_v2(
            lock_snapshot.payload, "artifact/approval_semantics_receipt"
        ),
    )
    if (
        not protocol_v2.exact_json_equal_v2(payload, expected)
        or payload.get("classification") != "corrected post-hoc replay"
        or payload.get("heldout_outcomes_already_known") is not True
        or payload.get("fresh_holdout") is not False
        or payload.get("preregistered") is not False
        or payload.get("protocol_lock_sha256") != lock_snapshot.sha256
        or payload.get("training_grid_sha256") != inventory_snapshot.grid.sha256
        or not protocol_v2.exact_json_equal_v2(
            payload.get("fit_sha256"), inventory_snapshot.fit_sha256
        )
    ):
        raise RuntimeError("primary-v2 held-out replay is not receipt-authorized")


def _static_age_policy_v2(log_multipliers: Sequence[float]):
    values = tuple(float(value) for value in log_multipliers)
    if len(values) != len(AGE_BRACKETS) or any(
        not math.isfinite(value) for value in values
    ):
        raise RuntimeError("primary-v2 static-age fit is not a finite four-vector")
    factors = {
        label: math.exp(value)
        for (_, _, label), value in zip(AGE_BRACKETS, values, strict=True)
    }

    def policy(instance, state):
        del state
        count = len(instance.votes)
        if count == 0:
            return []
        base = float(instance.budget) / count
        raw = [
            base * factors.get(age_bracket(vote.age), 1.0)
            for vote in instance.votes
        ]
        total = math.fsum(raw)
        if total <= 0.0:
            return [base] * count
        scale = float(instance.budget) / total
        return [float(value * scale) for value in raw]

    return policy


def selector_for_fit_v2(
    fit_snapshot: VerifiedFitSnapshotV2,
    cfg: EnvConfig | None = None,
):
    """Reconstruct the exact selector encoded by one authenticated fit."""

    if type(fit_snapshot) is not VerifiedFitSnapshotV2:
        raise RuntimeError("primary-v2 fit selector requires an authenticated snapshot")
    cfg = cfg or EnvConfig()
    spec = fit_snapshot.spec
    payload = fit_snapshot.payload
    if not protocol_v2.exact_json_equal_v2(payload.get("config"), spec):
        raise RuntimeError("primary-v2 fit selector config differs")
    result = payload.get("result")
    if type(result) is not dict or type(result.get("selected_weights")) is not list:
        raise RuntimeError("primary-v2 fit selector weights differ")
    weights = result["selected_weights"]
    if any(type(value) is not float or not math.isfinite(value) for value in weights):
        raise RuntimeError("primary-v2 fit selector weights are non-finite or aliased")
    family = str(spec.get("family"))
    arm = str(spec.get("arm"))
    if family == "static_age_lookup":
        return endowment_selector(_static_age_policy_v2(weights), cfg)
    if arm == "endowment":
        return endowment_selector(linear_policy(weights, cfg.scheme), cfg)
    if arm == "outcome":
        floor = spec.get("support_floor_kappa")
        support_floor = float(floor) if family == "static_support_floor" else None
        return score_selector(weights, cfg.scheme, support_floor=support_floor)
    raise RuntimeError(f"primary-v2 fit arm is unsupported: {arm}")


def _episode_row_v2(
    *,
    source_id: str,
    source_kind: str,
    split_name: str,
    view: str,
    selector,
    row: SeriesData,
    cfg: EnvConfig,
) -> dict[str, object]:
    score_years = row.train_years if view == "train" else row.test_years
    instances = row.train_only if view == "train" else row.all_years
    scored_set = set(score_years)
    state = RolloutState()
    scored_outcomes: dict[int, dict[str, dict[str, float]]] = {}
    years: list[dict[str, object]] = []
    for year in row.ref.years:
        instance = instances.get(year)
        if instance is None:
            continue
        selected = selector(instance, state)
        if not isinstance(selected, set) or any(type(pid) is not str for pid in selected):
            raise RuntimeError("primary-v2 selector returned a noncanonical winner set")
        if not selected <= set(instance.projects):
            raise RuntimeError("primary-v2 selector returned an unknown project")
        winners = set(selected)
        outcome = group_outcome(instance, winners, cfg.scheme)
        state.update(outcome)
        is_scored = year in scored_set
        if is_scored:
            scored_outcomes[year] = outcome
        spent = math.fsum(float(instance.projects[pid].cost) for pid in winners)
        if spent > float(instance.budget) + 1e-7:
            raise RuntimeError("primary-v2 selector exceeded the election budget")
        years.append(
            {
                "year": int(year),
                "scored": bool(is_scored),
                "winners": sorted(winners),
                "spent": float(spent),
                "welfare": float(approval_welfare(instance, winners)),
                "cost_welfare": float(cost_welfare(instance, winners)),
                "exclusion": float(exclusion_rate(instance, winners)),
            }
        )
    observed_scored = [year["year"] for year in years if year["scored"]]
    if observed_scored != list(score_years):
        raise RuntimeError(
            f"primary-v2 scored years differ for {row.ref.key}: "
            f"{observed_scored} != {list(score_years)}"
        )
    cohorts = sorted(
        {cohort for outcome in scored_outcomes.values() for cohort in outcome}
    )
    csds: dict[str, float] = {}
    for cohort in cohorts:
        value = cumulative_share_deficit(
            (scored_outcomes[year] for year in score_years), cohort
        )
        if value is not None:
            csds[cohort] = float(value)
    worst_cohort = max(csds, key=lambda cohort: csds[cohort]) if csds else None
    scored_rows = [year for year in years if year["scored"]]
    metrics = {
        "worst_csd": csds[worst_cohort] if worst_cohort is not None else None,
        "worst_cohort": worst_cohort,
        "mean_csd": (
            float(math.fsum(csds.values()) / len(csds)) if csds else None
        ),
        "welfare": float(math.fsum(float(year["welfare"]) for year in scored_rows)),
        "cost_welfare": float(
            math.fsum(float(year["cost_welfare"]) for year in scored_rows)
        ),
        "exclusion": float(
            math.fsum(float(year["exclusion"]) for year in scored_rows)
            / (len(scored_rows) or 1)
        ),
    }
    return {
        "fit_id": source_id,
        "source_kind": source_kind,
        "split": split_name,
        "view": view,
        "scheme": cfg.scheme,
        "series": row.ref.key,
        "scored_years": list(score_years),
        "year_outcomes": years,
        "metrics": metrics,
    }


def evaluate_selector_rows_v2(
    *,
    source_id: str,
    source_kind: str,
    split_name: str,
    selector,
    data: Sequence[SeriesData],
    cfg: EnvConfig | None = None,
    views: Sequence[str] = ("test", "train"),
) -> list[dict[str, object]]:
    """Pure replay API used by the evaluator and the protected-lineage audit."""

    if type(source_id) is not str or not source_id:
        raise RuntimeError("primary-v2 replay source id differs")
    if source_kind not in {"fit", "fixed", "diagnostic"}:
        raise RuntimeError("primary-v2 replay source kind differs")
    if tuple(views) not in {("test",), ("train",), ("test", "train")}:
        raise RuntimeError("primary-v2 replay view request differs")
    cfg = cfg or EnvConfig()
    rows: list[dict[str, object]] = []
    for view in views:
        for series in sorted(data, key=lambda item: item.ref.key):
            years = series.train_years if view == "train" else series.test_years
            if not years:
                continue
            rows.append(
                _episode_row_v2(
                    source_id=source_id,
                    source_kind=source_kind,
                    split_name=split_name,
                    view=view,
                    selector=selector,
                    row=series,
                    cfg=cfg,
                )
            )
    return rows


def load_evaluation_series_data_v2(
    input_snapshot: protocol_v2.PrimaryInputsSnapshotV2,
    replay_receipt_snapshot: ReplayReceiptSnapshotV2,
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
) -> dict[str, list[SeriesData]]:
    """Open every frozen train/test view only after the replay receipt exists."""

    _validate_replay_authorization_v2(
        replay_receipt_snapshot,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
    )
    protocol_v2.validate_protocol_inputs_snapshot_v2(input_snapshot, lock_snapshot)
    expected_splits = set(protocol_v2.SPLIT_SHA256_V2)
    if set(input_snapshot.splits) != expected_splits:
        raise RuntimeError("primary-v2 evaluation split coverage differs")

    def full_split(name: str):
        artifact = input_snapshot.split_artifacts.get(name)
        payload = getattr(artifact, "payload", None)
        if (
            type(payload) is not dict
            or set(payload) not in (
                {"name", "note", "train", "test"},
                {"name", "train", "test"},
            )
            or payload.get("name") != name
        ):
            raise RuntimeError(f"primary-v2 full evaluation split differs: {name}")

        def parse_view(field: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
            raw = payload.get(field)
            if type(raw) is not list:
                raise RuntimeError(f"primary-v2 evaluation split {field} differs")
            parsed: list[tuple[str, tuple[int, ...]]] = []
            seen_series: set[str] = set()
            seen_elections: set[tuple[str, int]] = set()
            for item in raw:
                if (
                    type(item) is not list
                    or len(item) != 2
                    or type(item[0]) is not str
                    or type(item[1]) is not list
                    or any(type(year) is not int for year in item[1])
                ):
                    raise RuntimeError(
                        f"primary-v2 evaluation split {field} row differs"
                    )
                series = item[0]
                years = tuple(item[1])
                parent = input_snapshot.index.get(series)
                if (
                    series in seen_series
                    or parent is None
                    or not years
                    or years != tuple(sorted(set(years)))
                    or any(year not in parent.years for year in years)
                ):
                    raise RuntimeError(
                        f"primary-v2 evaluation split {field} identity differs"
                    )
                seen_series.add(series)
                for year in years:
                    if (series, year) in seen_elections:
                        raise RuntimeError(
                            f"primary-v2 evaluation split {field} duplicates an election"
                        )
                    seen_elections.add((series, year))
                parsed.append((series, years))
            if parsed != sorted(parsed, key=lambda item: item[0]):
                raise RuntimeError(
                    f"primary-v2 evaluation split {field} ordering differs"
                )
            return tuple(parsed)

        train = parse_view("train")
        test = parse_view("test")
        overlap = {
            (series, year) for series, years in train for year in years
        } & {(series, year) for series, years in test for year in years}
        if overlap:
            raise RuntimeError("primary-v2 evaluation split train/test overlap")
        if train != input_snapshot.splits[name].train:
            raise RuntimeError("primary-v2 training projection and full split differ")
        return train, test

    result: dict[str, list[SeriesData]] = {}
    for split_name in sorted(expected_splits):
        train, test = full_split(split_name)
        train_map = dict(train)
        test_map = dict(test)
        rows: list[SeriesData] = []
        for key in sorted(set(train_map) | set(test_map)):
            parent = input_snapshot.index.get(key)
            if parent is None:
                raise RuntimeError(f"primary-v2 evaluation split names unknown series {key}")
            train_years = tuple(train_map.get(key, ()))
            test_years = tuple(test_map.get(key, ()))
            requested = tuple(
                year for year in parent.years if year in set(train_years + test_years)
            )
            if set(requested) != set(train_years + test_years):
                raise RuntimeError(f"primary-v2 split years differ for {key}")
            ref = SeriesRef(
                key=parent.key,
                years=requested,
                paths=tuple(
                    parent.paths[parent.years.index(year)] for year in requested
                ),
            )
            instances = load_series_authenticated_v2(
                ref,
                input_snapshot.semantics_receipt.payload,
                years=requested,
            )
            if tuple(instances) != requested:
                raise RuntimeError(f"primary-v2 loaded years differ for {key}")
            rows.append(
                SeriesData(
                    ref=ref,
                    train_years=train_years,
                    test_years=test_years,
                    train_only={year: instances[year] for year in train_years},
                    all_years=dict(instances),
                )
            )
        result[split_name] = rows
    return result


def _historical_selector_v2(instance, state) -> set[str]:
    del state
    winners = historical_winners(instance)
    if not winners:
        raise RuntimeError(f"primary-v2 historical winners are missing: {instance.path}")
    return winners


def _fixed_selectors_v2(cfg: EnvConfig) -> dict[str, object]:
    return {
        "historical": _historical_selector_v2,
        "greedy-count": score_selector(greedy_equivalent_weights(), cfg.scheme),
        "greedy-cost": score_selector(cost_effective_weights(), cfg.scheme),
        "llmrule-card": llmrule_card_selector(cfg.scheme),
        "llmrule-cost": llmrule_cost_selector(cfg.scheme),
        "mes": endowment_selector(uniform_policy, cfg),
        "res-0.25": endowment_selector(res_policy(0.25, cfg.scheme), cfg),
        "res-0.5": endowment_selector(res_policy(0.5, cfg.scheme), cfg),
        "res-0.75": endowment_selector(res_policy(0.75, cfg.scheme), cfg),
        "res-1.0": endowment_selector(res_policy(1.0, cfg.scheme), cfg),
    }


def build_per_series_payload_v2(
    data_by_split: Mapping[str, Sequence[SeriesData]],
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    replay_receipt_snapshot: ReplayReceiptSnapshotV2,
) -> dict[str, object]:
    """Recompute every locked fit, reference, robustness, and kernel trace."""

    _validate_replay_authorization_v2(
        replay_receipt_snapshot,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
    )
    expected_splits = set(protocol_v2.SPLIT_SHA256_V2)
    if set(data_by_split) != expected_splits:
        raise RuntimeError("primary-v2 numerical rebuild split coverage differs")
    clear_feature_cache()
    clear_project_cache()
    rows: list[dict[str, object]] = []

    # Every fit is evaluated on both views of its own frozen split.
    for fit in inventory_snapshot.fits:
        split_name = str(fit.spec["split"])
        cfg = EnvConfig(scheme="age_sex")
        rows.extend(
            evaluate_selector_rows_v2(
                source_id=str(fit.spec["fit_id"]),
                source_kind="fit",
                split_name=split_name,
                selector=selector_for_fit_v2(fit, cfg),
                data=data_by_split[split_name],
                cfg=cfg,
            )
        )

    # Fixed references are regenerated for every frozen split, not copied from v1.
    for split_name in sorted(expected_splits):
        cfg = EnvConfig(scheme="age_sex")
        for policy_id, selector in sorted(_fixed_selectors_v2(cfg).items()):
            rows.extend(
                evaluate_selector_rows_v2(
                    source_id=policy_id,
                    source_kind="fixed",
                    split_name=split_name,
                    selector=selector,
                    data=data_by_split[split_name],
                    cfg=cfg,
                )
            )

    temporal = "temporal_2022"
    main_endowment = next(
        fit
        for fit in inventory_snapshot.fits
        if fit.spec["family"] == "temporal_endowment" and fit.spec["seed"] == 42
    )
    headline = next(
        fit
        for fit in inventory_snapshot.fits
        if fit.spec["family"] == "outcome_frontier"
        and fit.spec["seed"] == 42
        and fit.spec["soft_welfare_target"] == 1.0
    )
    aggressive = next(
        fit
        for fit in inventory_snapshot.fits
        if fit.spec["family"] == "outcome_frontier"
        and fit.spec["seed"] == 42
        and fit.spec["soft_welfare_target"] == 0.85
    )
    corner = next(
        fit
        for fit in inventory_snapshot.fits
        if fit.spec["family"] == "outcome_frontier"
        and fit.spec["seed"] == 42
        and fit.spec["soft_welfare_target"] == 0.0
    )

    # Same frozen fits under coarser demographic ledgers; no refitting occurs.
    for scheme in ("age", "sex"):
        cfg = EnvConfig(scheme=scheme)
        for fit in (main_endowment, headline):
            rows.extend(
                evaluate_selector_rows_v2(
                    source_id=str(fit.spec["fit_id"]),
                    source_kind="fit",
                    split_name=temporal,
                    selector=selector_for_fit_v2(fit, cfg),
                    data=data_by_split[temporal],
                    cfg=cfg,
                )
            )
        for policy_id in ("mes", "res-1.0", "greedy-cost"):
            rows.extend(
                evaluate_selector_rows_v2(
                    source_id=policy_id,
                    source_kind="fixed",
                    split_name=temporal,
                    selector=_fixed_selectors_v2(cfg)[policy_id],
                    data=data_by_split[temporal],
                    cfg=cfg,
                )
            )

    # Training-selected scalar control, evaluated without consulting test results.
    selected_alpha = inventory_snapshot.grid.payload.get("selected_alpha")
    if type(selected_alpha) is not float or not math.isfinite(selected_alpha):
        raise RuntimeError("primary-v2 selected scalar grid value differs")
    senior_cfg = EnvConfig(scheme="age_sex")
    senior_weights = [0.0, 0.0, 0.0, float(math.log1p(selected_alpha))]
    rows.extend(
        evaluate_selector_rows_v2(
            source_id="static-senior-grid",
            source_kind="fixed",
            split_name=temporal,
            selector=endowment_selector(_static_age_policy_v2(senior_weights), senior_cfg),
            data=data_by_split[temporal],
            cfg=senior_cfg,
        )
    )

    # Hold each protected score vector fixed and intervene only on the
    # allocation kernel.  The three floors are all paper-facing coordinates.
    payment_fits = tuple(
        next(
            fit
            for fit in inventory_snapshot.fits
            if fit.spec["family"] == "outcome_frontier"
            and fit.spec["seed"] == 42
            and fit.spec["soft_welfare_target"] == target
        )
        for target in (0.0, 0.85, 1.0)
    )
    for fit in payment_fits:
        weights = fit.payload["result"]["selected_weights"]
        source = str(fit.spec["fit_id"])
        interventions = (
            ("static-floor", score_selector(
                weights, senior_cfg.scheme, support_floor=1.0
            )),
            ("payment+completion", payment_gated_score_selector(
                weights, senior_cfg.scheme, completion=True
            )),
            ("payment-only", payment_gated_score_selector(
                weights, senior_cfg.scheme, completion=False
            )),
        )
        for kernel, selector in interventions:
            rows.extend(
                evaluate_selector_rows_v2(
                    source_id=f"payment/{source}/{kernel}",
                    source_kind="diagnostic",
                    split_name=temporal,
                    selector=selector,
                    data=data_by_split[temporal],
                    cfg=senior_cfg,
                )
            )

    # Transfer the three protected-table fits to the locked evaluation-only Lodz split.
    transfer_split = "city_out_Poland_Łódź"
    for fit in (main_endowment, headline, aggressive):
        rows.extend(
            evaluate_selector_rows_v2(
                source_id=f"transfer/{fit.spec['fit_id']}",
                source_kind="diagnostic",
                split_name=transfer_split,
                selector=selector_for_fit_v2(fit, senior_cfg),
                data=data_by_split[transfer_split],
                cfg=senior_cfg,
                views=("test",),
            )
        )

    rows.sort(
        key=lambda row: (
            str(row["fit_id"]),
            str(row["split"]),
            str(row["view"]),
            str(row["scheme"]),
            str(row["series"]),
        )
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "semantics_profile": SEMANTICS_PROFILE,
        "rows": rows,
        "provenance": {
            "classification": "corrected post-hoc replay",
            "protocol_lock_sha256": lock_snapshot.sha256,
            "replay_receipt_sha256": replay_receipt_snapshot.sha256,
            "training_grid_sha256": inventory_snapshot.grid.sha256,
            "fit_sha256": inventory_snapshot.fit_sha256,
            "split_sha256": {
                name: _tracked_digest_v2(lock_snapshot.payload, f"split/{name}")
                for name in sorted(expected_splits)
            },
        },
    }
    validate_per_series_payload_v2(
        payload,
        expected_fit_ids=[fit.spec["fit_id"] for fit in inventory_snapshot.fits],
    )
    return payload


def _summary_for_rows_v2(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["split"]), str(row["view"]), str(row["scheme"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for (split, view, scheme), group in sorted(groups.items()):
        worst = [
            float(row["metrics"]["worst_csd"])
            for row in group
            if row["metrics"]["worst_csd"] is not None
        ]
        mean = [
            float(row["metrics"]["mean_csd"])
            for row in group
            if row["metrics"]["mean_csd"] is not None
        ]
        output.append(
            {
                "split": split,
                "view": view,
                "scheme": scheme,
                "n_series": len(group),
                "n_scoreable_series": len(worst),
                "worst_csd": float(math.fsum(worst) / len(worst)) if worst else None,
                "mean_csd": float(math.fsum(mean) / len(mean)) if mean else None,
                "welfare": float(
                    math.fsum(float(row["metrics"]["welfare"]) for row in group)
                ),
                "cost_welfare": float(
                    math.fsum(
                        float(row["metrics"]["cost_welfare"]) for row in group
                    )
                ),
                "exclusion": float(
                    math.fsum(float(row["metrics"]["exclusion"]) for row in group)
                    / (len(group) or 1)
                ),
            }
        )
    return output


def _contrast_rows_v2(
    rows: Sequence[Mapping[str, object]],
    source_ids: Sequence[str],
    policy_ids: Sequence[str],
) -> list[dict[str, object]]:
    index = {
        (
            str(row["fit_id"]),
            str(row["split"]),
            str(row["view"]),
            str(row["scheme"]),
            str(row["series"]),
        ): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for source_id in source_ids:
        coordinates = sorted(
            key[1:] for key in index if key[0] == source_id and key[2] == "test"
        )
        for policy_id in policy_ids:
            pairs = [
                (index[(source_id, *coordinate)], index[(policy_id, *coordinate)])
                for coordinate in coordinates
                if (policy_id, *coordinate) in index
                and index[(source_id, *coordinate)]["metrics"]["worst_csd"] is not None
                and index[(policy_id, *coordinate)]["metrics"]["worst_csd"] is not None
            ]
            by_view: dict[tuple[str, str], list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
            for learned, baseline in pairs:
                by_view.setdefault(
                    (str(learned["split"]), str(learned["scheme"])), []
                ).append((learned, baseline))
            for (split, scheme), paired in sorted(by_view.items()):
                differences = [
                    float(learned["metrics"]["worst_csd"])
                    - float(baseline["metrics"]["worst_csd"])
                    for learned, baseline in paired
                ]
                output.append(
                    {
                        "source_id": source_id,
                        "policy_id": policy_id,
                        "split": split,
                        "scheme": scheme,
                        "n": len(differences),
                        "mean_worst_csd_delta": float(
                            math.fsum(differences) / len(differences)
                        ),
                        "wins": sum(value < 0.0 for value in differences),
                        "ties": sum(value == 0.0 for value in differences),
                        "p_two_sided": float(
                            exact_paired_sign_flip_pvalue(differences)
                        ),
                    }
                )
    return output


def _consumer_policy_ids_v2(
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, list[str]]:
    primary = [
        "greedy-count",
        "greedy-cost",
        "llmrule-card",
        "llmrule-cost",
        "mes",
        "res-0.25",
        "res-0.5",
        "res-0.75",
        "res-1.0",
    ]
    headline = _fit_ids(
        inventory,
        "outcome_frontier",
        seed=42,
        soft_welfare_target=1.0,
    )[0]
    payment_fits = sorted(
        _fit_ids(
            inventory,
            "outcome_frontier",
            seed=42,
            soft_welfare_target=target,
        )[0]
        for target in (0.0, 0.85, 1.0)
    )
    transfer_sources = (
        _fit_ids(inventory, "temporal_endowment", seed=42)
        + _fit_ids(
            inventory,
            "outcome_frontier",
            seed=42,
            soft_welfare_target=0.85,
        )
        + [headline]
    )
    return {
        "baseline.primary": primary,
        "baseline.prior_rules": ["llmrule-card", "llmrule-cost"],
        "control.history_free": ["mes", "res-1.0"],
        "control.senior_scalar": ["mes", "static-senior-grid"],
        "control.static_age_lookup": ["mes", "static-senior-grid"],
        "cross_district.bound10": [
            "greedy-cost", "greedy-count", "llmrule-card", "llmrule-cost", "mes", "res-1.0"
        ],
        "cross_district.bound40": [
            "greedy-cost", "greedy-count", "llmrule-card", "llmrule-cost", "mes", "res-1.0"
        ],
        "failure.outcome": ["mes"],
        "frontier.endowment": ["mes", "res-1.0"],
        "frontier.outcome": primary,
        "mechanism.attribution_coverage": ["mes"],
        "mechanism.payment_kernel": sorted(
            f"payment/{fit_id}/{kernel}"
            for fit_id in payment_fits
            for kernel in ("payment+completion", "payment-only", "static-floor")
        ),
        "mechanism.support_floor": ["mes"],
        "mechanism.theory_floor": ["greedy-cost", "mes", "res-1.0"],
        "outcome.ablation": [],
        "outcome.per_series": ["greedy-count", "greedy-cost", "mes", "res-1.0"],
        "outcome.seed_stability": [],
        "outcome.significance": primary + ["llmrule-card", "llmrule-cost"],
        "primary.corpus": [],
        "robustness.demographic_partition": ["greedy-cost", "mes", "res-1.0"],
        "summary.composition": ["greedy-cost", "llmrule-card", "llmrule-cost", "mes"],
        "summary.endowment_seed_stability": ["mes"],
        "summary.endowment_significance": ["llmrule-card", "llmrule-cost", "mes", "res-1.0"],
        "summary.matched_target": ["mes"],
        "transfer.lodz": sorted(
            [
                "greedy-cost",
                "greedy-count",
                "historical",
                "mes",
                "res-1.0",
                *[f"transfer/{source_id}" for source_id in transfer_sources],
            ]
        ),
    }


def _render_macro_value_v2(value: object) -> str:
    if value is None:
        return "NA"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError("primary-v2 manuscript value is non-finite")
        rendered = f"{value:.6f}"
        return "0.000000" if rendered == "-0.000000" else rendered
    if type(value) is str:
        return value.replace("_", "\\_")
    raise RuntimeError("primary-v2 manuscript value has an unsupported type")


def _manuscript_macros_v2(
    rows: Sequence[Mapping[str, object]],
    corpus_values: Mapping[str, object],
    inventory: Sequence[Mapping[str, object]],
) -> tuple[dict[str, str], dict[str, object]]:
    macros: dict[str, str] = {}
    coordinates: dict[str, object] = {}

    def summary(
        source_id: str,
        *,
        split: str = "temporal_2022",
        view: str = "test",
        scheme: str = "age_sex",
    ) -> Mapping[str, object]:
        selected = [
            row
            for row in rows
            if row["fit_id"] == source_id
            and row["split"] == split
            and row["view"] == view
            and row["scheme"] == scheme
        ]
        reduced = _summary_for_rows_v2(selected)
        if len(reduced) != 1:
            raise RuntimeError(
                f"primary-v2 manuscript coordinate is missing: {source_id}/{split}/{view}/{scheme}"
            )
        return reduced[0]

    def put(
        name: str,
        value: str,
        *,
        source_id: str | None = None,
        split: str = "temporal_2022",
        view: str = "test",
        scheme: str = "age_sex",
        corpus_field: str | None = None,
    ) -> None:
        if name in macros:
            raise RuntimeError(f"primary-v2 duplicate manuscript macro: {name}")
        macros[name] = value
        coordinates[name] = (
            {"corpus_field": corpus_field}
            if corpus_field is not None
            else {
                "source_id": source_id,
                "split": split,
                "view": view,
                "scheme": scheme,
            }
        )

    def four(value: object) -> str:
        if value is None:
            return "NA"
        if type(value) is not float:
            raise RuntimeError("primary-v2 manuscript decimal differs")
        rendered = f"{value:.4f}"
        return "0.0000" if rendered == "-0.0000" else rendered

    def integer(value: object) -> str:
        if type(value) not in (int, float):
            raise RuntimeError("primary-v2 manuscript integer differs")
        return f"{round(float(value)):,}"

    def fit_id(family: str, **filters: object) -> str:
        matches = _fit_ids(inventory, family, **filters)
        if len(matches) != 1:
            raise RuntimeError(f"primary-v2 manuscript fit coordinate differs: {family}")
        return matches[0]

    temporal_sources = {
        "LearnedAggr": fit_id(
            "outcome_frontier", seed=42, soft_welfare_target=0.85
        ),
        "LearnedCorner": fit_id(
            "outcome_frontier", seed=42, soft_welfare_target=0.0
        ),
        "LearnedHead": fit_id(
            "outcome_frontier", seed=42, soft_welfare_target=1.0
        ),
        "Endow": fit_id("temporal_endowment", seed=42),
        "AgeLookup": fit_id("static_age_lookup"),
    }
    baseline_names = {
        "greedy-count": "TestGreedy",
        "greedy-cost": "TestGreedyCost",
        "mes": "TestMES",
        "res-1.0": "TestRES",
    }
    for source_id, prefix in baseline_names.items():
        values = summary(source_id)
        put(f"{prefix}CSD", four(values["worst_csd"]), source_id=source_id)
        put(f"{prefix}Welfare", integer(values["welfare"]), source_id=source_id)
        if prefix in {"TestGreedy", "TestGreedyCost"}:
            put(f"{prefix}Excl", four(values["exclusion"]), source_id=source_id)
    historical = summary("historical")
    put(
        "TestHistoricalWelfare",
        integer(historical["welfare"]),
        source_id="historical",
    )
    put(
        "EvalSeriesN",
        str(summary("mes")["n_series"]),
        source_id="mes",
    )

    target_by_prefix = {
        "LearnedAggr": 0.85,
        "LearnedCorner": 0.0,
        "LearnedHead": 1.0,
    }
    for prefix, source_id in temporal_sources.items():
        test_values = summary(source_id)
        put(f"{prefix}CSD", four(test_values["worst_csd"]), source_id=source_id)
        put(f"{prefix}Excl", four(test_values["exclusion"]), source_id=source_id)
        put(f"{prefix}Welfare", integer(test_values["welfare"]), source_id=source_id)
        if prefix in target_by_prefix:
            train_values = summary(source_id, view="train")
            put(
                f"{prefix}TrainCSD",
                four(train_values["worst_csd"]),
                source_id=source_id,
                view="train",
            )
            put(
                f"{prefix}Floor",
                f"{target_by_prefix[prefix]:.2f}",
                source_id=source_id,
            )
            gap = float(test_values["worst_csd"]) - float(train_values["worst_csd"])
            put(
                f"{prefix}Gap",
                f"{gap:+.4f}" if round(gap, 4) != 0 else "+0.0000",
                source_id=source_id,
            )

    kernel_sources = {
        "KernelDirect": temporal_sources["LearnedCorner"],
        "KernelPaymentOnly": (
            f"payment/{temporal_sources['LearnedCorner']}/payment-only"
        ),
        "KernelPaymentComplete": (
            f"payment/{temporal_sources['LearnedCorner']}/payment+completion"
        ),
    }
    for prefix, source_id in kernel_sources.items():
        values = summary(source_id)
        put(f"{prefix}CSD", four(values["worst_csd"]), source_id=source_id)
        put(f"{prefix}Excl", four(values["exclusion"]), source_id=source_id)
        put(f"{prefix}Welfare", integer(values["welfare"]), source_id=source_id)

    transfer_sources = {
        "TransferHist": "historical",
        "TransferGreedyCost": "greedy-cost",
        "TransferMES": "mes",
        "TransferEndow": f"transfer/{temporal_sources['Endow']}",
        "TransferOutcome": f"transfer/{temporal_sources['LearnedHead']}",
    }
    transfer_split = "city_out_Poland_Łódź"
    for prefix, source_id in transfer_sources.items():
        values = summary(source_id, split=transfer_split)
        put(
            f"{prefix}CSD",
            four(values["worst_csd"]),
            source_id=source_id,
            split=transfer_split,
        )
        put(
            f"{prefix}Welfare",
            integer(values["welfare"]),
            source_id=source_id,
            split=transfer_split,
        )

    partition_names = {"age_sex": "AgeSex", "age": "Age", "sex": "Sex"}
    for scheme, label in partition_names.items():
        for suffix, source_id in (
            ("Endow", temporal_sources["Endow"]),
            ("MES", "mes"),
            ("Outcome", temporal_sources["LearnedHead"]),
            ("RES", "res-1.0"),
        ):
            values = summary(source_id, scheme=scheme)
            put(
                f"Part{label}{suffix}CSD",
                four(values["worst_csd"]),
                source_id=source_id,
                scheme=scheme,
            )

    if type(corpus_values.get("n_elections")) is int:
        put(
            "PrimaryElectionN",
            integer(corpus_values["n_elections"]),
            corpus_field="n_elections",
        )
    if type(corpus_values.get("approval_ballots_checked")) is int:
        put(
            "PrimaryBallotN",
            integer(corpus_values["approval_ballots_checked"]),
            corpus_field="approval_ballots_checked",
        )
    return dict(sorted(macros.items())), coordinates


def _one_fit_id_v2(
    inventory: Sequence[Mapping[str, object]],
    family: str,
    **filters: object,
) -> str:
    matches = _fit_ids(inventory, family, **filters)
    if len(matches) != 1:
        raise RuntimeError(
            f"primary-v2 consumer fit coordinate differs: {family}/{filters}"
        )
    return matches[0]


def _coordinate_rows_v2(
    rows: Sequence[Mapping[str, object]],
    source_id: str,
    *,
    split: str = "temporal_2022",
    view: str = "test",
    scheme: str = "age_sex",
) -> list[Mapping[str, object]]:
    selected = sorted(
        (
            row
            for row in rows
            if row["fit_id"] == source_id
            and row["split"] == split
            and row["view"] == view
            and row["scheme"] == scheme
        ),
        key=lambda row: str(row["series"]),
    )
    keys = [str(row["series"]) for row in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"primary-v2 duplicate consumer coordinate: {source_id}")
    return selected


def _one_summary_v2(
    rows: Sequence[Mapping[str, object]],
    source_id: str,
    *,
    split: str = "temporal_2022",
    view: str = "test",
    scheme: str = "age_sex",
) -> Mapping[str, object]:
    reduced = _summary_for_rows_v2(
        _coordinate_rows_v2(
            rows, source_id, split=split, view=view, scheme=scheme
        )
    )
    if len(reduced) != 1:
        raise RuntimeError(
            f"primary-v2 consumer coordinate is missing: "
            f"{source_id}/{split}/{view}/{scheme}"
        )
    return reduced[0]


def _paired_statistics_v2(
    learned: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 42,
    bootstrap_replicates: int = 10_000,
) -> dict[str, object]:
    a = np.asarray(learned, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise RuntimeError("primary-v2 consumer paired vectors differ")
    keep = np.isfinite(a) & np.isfinite(b)
    diff = a[keep] - b[keep]
    if diff.size == 0:
        raise RuntimeError("primary-v2 consumer has no finite paired values")
    rng = np.random.default_rng(seed)
    draws = diff[
        rng.integers(
            0,
            diff.size,
            size=(bootstrap_replicates, diff.size),
        )
    ].mean(axis=1)
    low, high = np.percentile(draws, (2.5, 97.5))
    return {
        "learned_mean": float(a[keep].mean()),
        "baseline_mean": float(b[keep].mean()),
        "diff": float(diff.mean()),
        "ci_lo": float(low),
        "ci_hi": float(high),
        "p_two_sided": float(exact_paired_sign_flip_pvalue(diff)),
        "wins": int(np.count_nonzero(diff < 0.0)),
        "ties": int(np.count_nonzero(diff == 0.0)),
        "n": int(diff.size),
    }


def _paired_source_values_v2(
    rows: Sequence[Mapping[str, object]],
    learned_id: str,
    baseline_id: str,
    *,
    split: str = "temporal_2022",
    scheme: str = "age_sex",
    metric: str = "worst_csd",
    seed: int = 42,
) -> dict[str, object]:
    learned = {
        str(row["series"]): row
        for row in _coordinate_rows_v2(
            rows, learned_id, split=split, view="test", scheme=scheme
        )
    }
    baseline = {
        str(row["series"]): row
        for row in _coordinate_rows_v2(
            rows, baseline_id, split=split, view="test", scheme=scheme
        )
    }
    keys = sorted(set(learned) & set(baseline))
    if keys != sorted(learned) or keys != sorted(baseline):
        raise RuntimeError("primary-v2 paired consumer series coverage differs")
    a = [learned[key]["metrics"][metric] for key in keys]
    b = [baseline[key]["metrics"][metric] for key in keys]
    if any(value is None for value in a + b):
        raise RuntimeError("primary-v2 paired consumer metric is missing")
    return _paired_statistics_v2(
        [float(value) for value in a],
        [float(value) for value in b],
        seed=seed,
    )


def _fit_record_v2(fit: VerifiedFitSnapshotV2) -> dict[str, object]:
    result = fit.payload.get("result")
    if type(result) is not dict:
        raise RuntimeError("primary-v2 consumer fit result differs")
    weights = result.get("selected_weights")
    best_loss = result.get("best_loss")
    if (
        type(weights) is not list
        or any(type(value) is not float for value in weights)
        or type(best_loss) is not float
    ):
        raise RuntimeError("primary-v2 consumer fit record differs")
    return {
        "fit_id": str(fit.spec["fit_id"]),
        "family": str(fit.spec["family"]),
        "split": str(fit.spec["split"]),
        "seed": int(fit.spec["seed"]),
        "soft_welfare_target": float(fit.spec["soft_welfare_target"]),
        "support_floor_kappa": fit.spec["support_floor_kappa"],
        "best_train_loss": float(best_loss),
        "selected_weights": list(weights),
    }


def build_outcome_ablation_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct the six-row protected leave-one-feature-out table."""

    full_id = _one_fit_id_v2(
        inventory,
        "outcome_frontier",
        seed=42,
        soft_welfare_target=1.0,
    )
    full = _one_summary_v2(rows, full_id)
    records: list[dict[str, object]] = [
        {
            "ablated": "none (full model)",
            "source_id": full_id,
            "test_worst_csd": full["worst_csd"],
            "test_welfare": full["welfare"],
            "delta_vs_full": 0.0,
        }
    ]
    loo = sorted(
        (spec for spec in inventory if spec["family"] == "outcome_loo"),
        key=lambda spec: str(spec["masked_feature_names"][0]),
    )
    if len(loo) != 5:
        raise RuntimeError("primary-v2 outcome ablation inventory differs")
    for spec in loo:
        source_id = str(spec["fit_id"])
        summary = _one_summary_v2(rows, source_id)
        records.append(
            {
                "ablated": str(spec["masked_feature_names"][0]),
                "source_id": source_id,
                "test_worst_csd": summary["worst_csd"],
                "test_welfare": summary["welfare"],
                "delta_vs_full": float(summary["worst_csd"])
                - float(full["worst_csd"]),
            }
        )
    return {
        "schema": "outcome-ablation-table-v2",
        "row_count": len(records),
        "ablation_rows": records,
    }


def build_transfer_lodz_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct the exact eight-policy Lodz transfer table."""

    endow = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    direct = _one_fit_id_v2(
        inventory,
        "outcome_frontier",
        seed=42,
        soft_welfare_target=1.0,
    )
    aggressive = _one_fit_id_v2(
        inventory,
        "outcome_frontier",
        seed=42,
        soft_welfare_target=0.85,
    )
    coordinates = (
        ("historical", "historical"),
        ("greedy-count", "greedy-count"),
        ("greedy-cost", "greedy-cost"),
        ("mes", "mes"),
        ("res-1.0", "res-1.0"),
        ("learned-endowment", f"transfer/{endow}"),
        ("learned-outcome-f1.00", f"transfer/{direct}"),
        ("learned-outcome-f0.85", f"transfer/{aggressive}"),
    )
    output: list[dict[str, object]] = []
    for policy, source_id in coordinates:
        selected = _coordinate_rows_v2(
            rows,
            source_id,
            split="city_out_Poland_Łódź",
            view="test",
            scheme="age_sex",
        )
        if len(selected) != 1:
            raise RuntimeError(f"primary-v2 Lodz transfer coordinate differs: {policy}")
        row = selected[0]
        metrics = row["metrics"]
        output.append(
            {
                "series": str(row["series"]),
                "policy": policy,
                "source_id": source_id,
                "worst_csd": metrics["worst_csd"],
                "welfare": metrics["welfare"],
                "exclusion": metrics["exclusion"],
                "worst_cohort": metrics["worst_cohort"],
            }
        )
    return {
        "schema": "lodz-transfer-table-v2",
        "row_count": len(output),
        "transfer_rows": output,
    }


def _baseline_primary_values_v2(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    policies = (
        "greedy-count",
        "greedy-cost",
        "llmrule-cost",
        "llmrule-card",
        "mes",
        "res-0.25",
        "res-0.5",
        "res-0.75",
        "res-1.0",
    )
    by_policy: dict[str, dict[str, Mapping[str, object]]] = {}
    views: dict[str, str] = {}
    for policy in policies:
        selected = [
            *_coordinate_rows_v2(
                rows,
                policy,
                split="city_out_Poland_Łódź",
                view="train",
            ),
            *_coordinate_rows_v2(
                rows,
                policy,
                split="city_out_Poland_Łódź",
                view="test",
            ),
        ]
        by_policy[policy] = {str(row["series"]): row for row in selected}
        for row in selected:
            series = str(row["series"])
            observed = views.setdefault(series, str(row["view"]))
            if observed != row["view"]:
                raise RuntimeError("primary-v2 full-corpus baseline view overlaps")
    series_keys = sorted(views)
    if any(sorted(by_policy[policy]) != series_keys for policy in policies):
        raise RuntimeError("primary-v2 baseline series coverage differs")
    output: list[dict[str, object]] = []
    for series in series_keys:
        record: dict[str, object] = {"series": series, "split_view": views[series]}
        for policy in policies:
            metrics = by_policy[policy][series]["metrics"]
            record[f"{policy}_worst"] = metrics["worst_csd"]
            record[f"{policy}_welfare"] = metrics["welfare"]
        output.append(record)
    return {
        "schema": "primary-baseline-table-v2",
        "series_count": len(output),
        "series_rows": output,
    }


def _prior_rule_values_v2(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    formulas = {
        "llmrule-cost": "sqrt(approval_share)/(1+cost_share)",
        "llmrule-card": "approval_share/cost_share*log1p(approval_share)",
    }
    output: list[dict[str, object]] = []
    for rule in ("llmrule-cost", "llmrule-card"):
        for view in ("train", "test"):
            selected = _coordinate_rows_v2(rows, rule, view=view)
            for row in selected:
                metrics = row["metrics"]
                output.append(
                    {
                        "rule": rule,
                        "series": str(row["series"]),
                        "view": view,
                        "worst_csd": metrics["worst_csd"],
                        "mean_csd": metrics["mean_csd"],
                        "welfare": metrics["welfare"],
                        "cost_welfare": metrics["cost_welfare"],
                        "exclusion": metrics["exclusion"],
                        "n_series": 1,
                        "source_doi": "10.65109/LGEP3560",
                        "score": formulas[rule],
                    }
                )
            summary = _one_summary_v2(rows, rule, view=view)
            output.append(
                {
                    "rule": rule,
                    "series": "__aggregate__",
                    "view": view,
                    "worst_csd": summary["worst_csd"],
                    "mean_csd": summary["mean_csd"],
                    "welfare": summary["welfare"],
                    "cost_welfare": summary["cost_welfare"],
                    "exclusion": summary["exclusion"],
                    "n_series": summary["n_series"],
                    "source_doi": "10.65109/LGEP3560",
                    "score": formulas[rule],
                }
            )
    return {
        "schema": "prior-rule-table-v2",
        "row_count": len(output),
        "rule_view_rows": output,
    }


def _history_free_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    fits: Mapping[str, VerifiedFitSnapshotV2],
) -> dict[str, object]:
    history_free = _one_fit_id_v2(inventory, "history_free")
    full = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    policies = (
        ("static-only (no history)", history_free),
        ("MES", "mes"),
        ("RES(1.0)", "res-1.0"),
        ("full (with history)", full),
    )
    policy_rows = []
    for policy, source_id in policies:
        summary = _one_summary_v2(rows, source_id)
        policy_rows.append(
            {
                "policy": policy,
                "source_id": source_id,
                "test_worst_csd": summary["worst_csd"],
                "test_welfare": summary["welfare"],
                "test_exclusion": summary["exclusion"],
            }
        )
    return {
        "schema": "history-free-control-v2",
        "policy_count": len(policy_rows),
        "policy_rows": policy_rows,
        "fit_records": [_fit_record_v2(fits[source]) for source in (history_free, full)],
    }


def _senior_scalar_values_v2(
    rows: Sequence[Mapping[str, object]],
    grid: train_v2.JsonSnapshotV2,
) -> dict[str, object]:
    alpha = grid.payload.get("selected_alpha")
    alphas = grid.payload.get("alphas")
    losses = grid.payload.get("train_worst_csd")
    if (
        type(alpha) is not float
        or type(alphas) is not list
        or type(losses) is not list
        or len(alphas) != 61
        or len(losses) != 61
    ):
        raise RuntimeError("primary-v2 senior training grid differs")
    selected = _one_summary_v2(rows, "static-senior-grid")
    train = _one_summary_v2(rows, "static-senior-grid", view="train")
    mes = _one_summary_v2(rows, "mes")
    paired = _paired_source_values_v2(
        rows, "static-senior-grid", "mes", seed=20260812
    )
    return {
        "schema": "senior-scalar-control-v2",
        "selected": {
            "alpha": alpha,
            "train_worst_csd": train["worst_csd"],
            "test_worst_csd": selected["worst_csd"],
            "mes_worst_csd": mes["worst_csd"],
            "difference": paired["diff"],
            "ci_low": paired["ci_lo"],
            "ci_high": paired["ci_hi"],
            "p_value": paired["p_two_sided"],
            "wins": paired["wins"],
            "n_series": paired["n"],
            "test_welfare": selected["welfare"],
            "test_exclusion": selected["exclusion"],
        },
        "training_grid": {
            "candidate_count": len(alphas),
            "alphas": list(alphas),
            "train_worst_csd": list(losses),
            "selection_rule": grid.payload.get("selection_rule"),
        },
    }


def _static_age_lookup_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    fits: Mapping[str, VerifiedFitSnapshotV2],
) -> dict[str, object]:
    age_lookup = _one_fit_id_v2(inventory, "static_age_lookup")
    history_free = _one_fit_id_v2(inventory, "history_free")
    sources = (
        ("age_lookup", age_lookup),
        ("history_free", history_free),
        ("static_60plus", "static-senior-grid"),
    )
    policy_rows = []
    for policy, source in sources:
        summary = _one_summary_v2(rows, source)
        policy_rows.append(
            {
                "policy": policy,
                "source_id": source,
                "mean_csd": summary["worst_csd"],
                "welfare": summary["welfare"],
                "mean_exclusion": summary["exclusion"],
            }
        )
    contrast_rows = []
    for contrast, learned, baseline in (
        ("age_lookup_minus_static_60plus", age_lookup, "static-senior-grid"),
        ("history_free_minus_age_lookup", history_free, age_lookup),
    ):
        contrast_rows.append(
            {
                "contrast": contrast,
                "learned_source_id": learned,
                "baseline_source_id": baseline,
                **_paired_source_values_v2(rows, learned, baseline),
            }
        )
    return {
        "schema": "static-age-lookup-control-v2",
        "policy_count": len(policy_rows),
        "policy_rows": policy_rows,
        "contrast_count": len(contrast_rows),
        "contrast_rows": contrast_rows,
        "fit_record": _fit_record_v2(fits[age_lookup]),
    }


def _cross_district_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    fits: Mapping[str, VerifiedFitSnapshotV2],
    *,
    bound: float,
) -> dict[str, object]:
    fit_ids = _fit_ids(inventory, "district_endowment", bound=bound)
    if len(fit_ids) != 5:
        raise RuntimeError("primary-v2 district fold fit count differs")
    baselines = (
        "greedy-cost",
        "greedy-count",
        "llmrule-card",
        "llmrule-cost",
        "mes",
        "res-1.0",
    )
    series_rows: list[dict[str, object]] = []
    learned_by_series: dict[str, Mapping[str, object]] = {}
    fold_by_series: dict[str, int] = {}
    fold_records: list[dict[str, object]] = []
    for fit_id in fit_ids:
        fit = fits[fit_id]
        split = str(fit.spec["split"])
        fold = int(split.split("_f", 1)[1].split("of", 1)[0])
        learned_rows = _coordinate_rows_v2(rows, fit_id, split=split)
        fold_records.append(
            {
                "fold": fold,
                "split": split,
                "fit": _fit_record_v2(fit),
                "test_series": sorted(str(row["series"]) for row in learned_rows),
                "boundary_features": sorted(
                    name
                    for name, value in zip(
                        fit.payload["arm"]["feature_names"],
                        fit.payload["result"]["selected_weights"],
                        strict=True,
                    )
                    if abs(abs(float(value)) - bound) <= 1e-6
                ),
            }
        )
        for source_id in (fit_id, *baselines):
            policy = "learned" if source_id == fit_id else source_id
            selected = _coordinate_rows_v2(rows, source_id, split=split)
            for row in selected:
                series = str(row["series"])
                metrics = row["metrics"]
                series_rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "series": series,
                        "policy": policy,
                        "source_id": source_id,
                        "n_years": len(row["scored_years"]),
                        "worst_csd": metrics["worst_csd"],
                        "mean_csd": metrics["mean_csd"],
                        "welfare": metrics["welfare"],
                        "cost_welfare": metrics["cost_welfare"],
                        "exclusion": metrics["exclusion"],
                    }
                )
                if policy == "learned":
                    if series in learned_by_series:
                        raise RuntimeError("primary-v2 district test fold overlap")
                    learned_by_series[series] = row
                    fold_by_series[series] = fold
    series_rows.sort(key=lambda row: (int(row["fold"]), str(row["series"]), str(row["policy"])))
    contrast_rows = []
    keys = sorted(learned_by_series)
    for baseline in baselines:
        baseline_by_series: dict[str, Mapping[str, object]] = {}
        for record in series_rows:
            if record["policy"] == baseline:
                baseline_by_series[str(record["series"])] = record
        if sorted(baseline_by_series) != keys:
            raise RuntimeError("primary-v2 district baseline coverage differs")
        learned_values = [
            float(learned_by_series[key]["metrics"]["worst_csd"]) for key in keys
        ]
        baseline_values = [
            float(baseline_by_series[key]["worst_csd"]) for key in keys
        ]
        paired = _paired_statistics_v2(learned_values, baseline_values)
        fold_diffs = [
            float(
                np.mean(
                    [
                        learned_values[index] - baseline_values[index]
                        for index, key in enumerate(keys)
                        if fold_by_series[key] == fold
                    ]
                )
            )
            for fold in range(5)
        ]
        learned_welfare = math.fsum(
            float(learned_by_series[key]["metrics"]["welfare"]) for key in keys
        )
        baseline_welfare = math.fsum(
            float(baseline_by_series[key]["welfare"]) for key in keys
        )
        contrast_rows.append(
            {
                "baseline": baseline,
                **paired,
                "fold_mean_diff": float(np.mean(fold_diffs)),
                "fold_p_two_sided": float(exact_paired_sign_flip_pvalue(fold_diffs)),
                "fold_wins": sum(value < 0 for value in fold_diffs),
                "n_folds": len(fold_diffs),
                "learned_welfare": learned_welfare,
                "baseline_welfare": baseline_welfare,
                "welfare_ratio": learned_welfare / baseline_welfare,
                "learned_exclusion": float(
                    np.mean(
                        [
                            learned_by_series[key]["metrics"]["exclusion"]
                            for key in keys
                        ]
                    )
                ),
                "baseline_exclusion": float(
                    np.mean([baseline_by_series[key]["exclusion"] for key in keys])
                ),
            }
        )
    return {
        "schema": f"cross-district-bound{int(bound)}-v2",
        "bound": bound,
        "n_folds": len(fold_records),
        "n_series": len(keys),
        "n_policy_series_rows": len(series_rows),
        "series_policy_rows": series_rows,
        "contrast_rows": contrast_rows,
        "fold_records": sorted(fold_records, key=lambda row: int(row["fold"])),
        "hypothesis_tests": {
            "pooled_series": "exact two-sided paired sign-flip over series",
            "paired_series_interval": "10000-draw paired percentile bootstrap, seed 42",
            "fold_block": "exact two-sided sign-flip over five fold-mean effects",
        },
    }


def _frontier_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    fits: Mapping[str, VerifiedFitSnapshotV2],
    *,
    family: str,
) -> dict[str, object]:
    arm = "endowment" if family == "endowment_frontier" else "outcome"
    fit_ids = _fit_ids(inventory, family)
    mes_train = _one_summary_v2(rows, "mes", view="train")
    point_rows = []
    for fit_id in fit_ids:
        fit = fits[fit_id]
        train = _one_summary_v2(rows, fit_id, view="train")
        test = _one_summary_v2(rows, fit_id)
        point_rows.append(
            {
                "fit_id": fit_id,
                "arm": arm,
                "floor": float(fit.spec["soft_welfare_target"]),
                "seed": int(fit.spec["seed"]),
                "penalty": float(fit.spec["welfare_penalty"]),
                "train_worst_csd": train["worst_csd"],
                "train_welfare_ratio": float(train["welfare"])
                / float(mes_train["welfare"]),
                "test_worst_csd": test["worst_csd"],
                "test_mean_csd": test["mean_csd"],
                "test_welfare": test["welfare"],
                "test_cost_welfare": test["cost_welfare"],
                "test_exclusion": test["exclusion"],
                "best_train_loss": fit.payload["result"]["best_loss"],
                "weights": list(fit.payload["result"]["selected_weights"]),
            }
        )
    point_rows.sort(key=lambda row: (int(row["seed"]), float(row["floor"])))
    baseline_rows = []
    baseline_coordinates = (
        ("mes", "mes"),
        ("res-0.25", "res-0.25"),
        ("res-0.5", "res-0.5"),
        ("res-0.75", "res-0.75"),
        ("res-1.0", "res-1.0"),
        ("greedy-count", "greedy-count"),
        ("greedy-cost", "greedy-cost"),
        ("historical", "historical"),
        ("greedy", "greedy-count"),
    )
    for policy, source in baseline_coordinates:
        for view in ("train", "test"):
            summary = _one_summary_v2(rows, source, view=view)
            baseline_rows.append(
                {
                    "policy": policy,
                    "source_id": source,
                    "view": view,
                    "n_series": summary["n_series"],
                    "worst_csd": summary["worst_csd"],
                    "mean_csd": summary["mean_csd"],
                    "welfare": summary["welfare"],
                    "cost_welfare": summary["cost_welfare"],
                    "exclusion": summary["exclusion"],
                }
            )
    return {
        "schema": f"{arm}-frontier-v2",
        "arm": arm,
        "point_count": len(point_rows),
        "point_rows": point_rows,
        "baseline_count": len(baseline_rows),
        "baseline_rows": baseline_rows,
    }


def _support_floor_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    mes_train = _one_summary_v2(rows, "mes", view="train")
    output = []
    for fit_id in _fit_ids(inventory, "static_support_floor"):
        spec = next(spec for spec in inventory if spec["fit_id"] == fit_id)
        train = _one_summary_v2(rows, fit_id, view="train")
        test = _one_summary_v2(rows, fit_id)
        output.append(
            {
                "fit_id": fit_id,
                "kappa": float(spec["support_floor_kappa"]),
                "welfare_floor": "none",
                "train_worst_csd": train["worst_csd"],
                "test_worst_csd": test["worst_csd"],
                "test_welfare": test["welfare"],
                "test_exclusion": test["exclusion"],
                "welfare_ratio_vs_mes": float(test["welfare"])
                / float(mes_train["welfare"]),
                "ratio_denominator_view": "mes/train",
            }
        )
    output.sort(key=lambda row: float(row["kappa"]))
    return {
        "schema": "static-support-floor-identification-v2",
        "row_count": len(output),
        "identification_rows": output,
    }


def _outcome_per_series_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    outcome = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    endow = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    sources = {
        source: {
            str(row["series"]): row
            for row in _coordinate_rows_v2(rows, source)
        }
        for source in (
            "greedy-count",
            "greedy-cost",
            "mes",
            "res-1.0",
            endow,
            outcome,
        )
    }
    keys = sorted(sources[outcome])
    if any(sorted(table) != keys for table in sources.values()):
        raise RuntimeError("primary-v2 outcome per-series coverage differs")
    learned_rows = []
    two_arm_rows = []
    for key in keys:
        learned = sources[outcome][key]["metrics"]
        endowment = sources[endow][key]["metrics"]
        mes = sources["mes"][key]["metrics"]
        learned_delta = float(learned["worst_csd"]) - float(mes["worst_csd"])
        endow_delta = float(endowment["worst_csd"]) - float(mes["worst_csd"])
        learned_rows.append(
            {
                "series": key,
                "test_years": list(sources[outcome][key]["scored_years"]),
                "greedy-count": sources["greedy-count"][key]["metrics"]["worst_csd"],
                "greedy-cost": sources["greedy-cost"][key]["metrics"]["worst_csd"],
                "mes": mes["worst_csd"],
                "res-1.0": sources["res-1.0"][key]["metrics"]["worst_csd"],
                "learned": learned["worst_csd"],
                "worst_cohort": learned["worst_cohort"],
                "learned_minus_mes": learned_delta,
                "learned_wins": int(learned_delta < 0.0),
            }
        )
        two_arm_rows.append(
            {
                "series": key,
                "mes": mes["worst_csd"],
                "learned_endowment": endowment["worst_csd"],
                "learned_outcome": learned["worst_csd"],
                "endowment_minus_mes": endow_delta,
                "outcome_minus_mes": learned_delta,
                "endowment_wins": int(endow_delta < 0.0),
                "outcome_wins": int(learned_delta < 0.0),
            }
        )
    learned_rows.sort(key=lambda row: float(row["learned_minus_mes"]))
    two_arm_rows.sort(key=lambda row: float(row["outcome_minus_mes"]))
    return {
        "schema": "outcome-series-tables-v2",
        "row_count": len(keys),
        "learned_rows": learned_rows,
        "two_arm_rows": two_arm_rows,
    }


def _outcome_seed_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    fits: Mapping[str, VerifiedFitSnapshotV2],
) -> dict[str, object]:
    output = []
    for fit_id in _fit_ids(
        inventory, "outcome_frontier", soft_welfare_target=1.0
    ):
        fit = fits[fit_id]
        summary = _one_summary_v2(rows, fit_id)
        output.append(
            {
                "fit_id": fit_id,
                "seed": int(fit.spec["seed"]),
                "test_worst_csd": summary["worst_csd"],
                "test_mean_csd": summary["mean_csd"],
                "test_welfare": summary["welfare"],
                "test_cost_welfare": summary["cost_welfare"],
                "test_exclusion": summary["exclusion"],
                "best_train_loss": fit.payload["result"]["best_loss"],
                "weights": list(fit.payload["result"]["selected_weights"]),
            }
        )
    output.sort(key=lambda row: int(row["seed"]))
    return {
        "schema": "outcome-seed-stability-v2",
        "seed_count": len(output),
        "seed_rows": output,
    }


def _outcome_significance_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baselines = (
        "greedy-count",
        "greedy-cost",
        "llmrule-cost",
        "llmrule-card",
        "mes",
        "res-1.0",
    )
    output = []
    for fit_id in _fit_ids(inventory, "outcome_frontier", seed=42):
        spec = next(spec for spec in inventory if spec["fit_id"] == fit_id)
        for baseline in baselines:
            paired = _paired_source_values_v2(rows, fit_id, baseline)
            output.append(
                {
                    "arm": "outcome",
                    "floor": float(spec["soft_welfare_target"]),
                    "source_id": fit_id,
                    "baseline": baseline,
                    "learned_mean": paired["learned_mean"],
                    "baseline_mean": paired["baseline_mean"],
                    "diff": paired["diff"],
                    "ci_lo": paired["ci_lo"],
                    "ci_hi": paired["ci_hi"],
                    "p_two_sided": paired["p_two_sided"],
                    "wins": paired["wins"],
                    "n": paired["n"],
                }
            )
    output.sort(key=lambda row: (float(row["floor"]), str(row["baseline"])))
    return {
        "schema": "outcome-significance-v2",
        "contrast_count": len(output),
        "contrast_rows": output,
    }


def _demographic_partition_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    endow = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    outcome = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    sources = (
        ("mes", "mes"),
        ("res-1.0", "res-1.0"),
        ("greedy-cost", "greedy-cost"),
        ("learned-endowment", endow),
        ("learned-outcome", outcome),
    )
    group_counts = {"age_sex": 8, "age": 4, "sex": 2}
    output = []
    for scheme in ("age_sex", "age", "sex"):
        mes_rows = {
            str(row["series"]): row
            for row in _coordinate_rows_v2(rows, "mes", scheme=scheme)
        }
        for policy, source in sources:
            selected = _coordinate_rows_v2(rows, source, scheme=scheme)
            summary = _one_summary_v2(rows, source, scheme=scheme)
            record: dict[str, object] = {
                "scheme": scheme,
                "n_groups": group_counts[scheme],
                "policy": policy,
                "source_id": source,
                "worst_csd": summary["worst_csd"],
                "vs_mes_diff": None,
                "vs_mes_p": None,
                "vs_mes_wins": None,
                "n": None,
            }
            if policy.startswith("learned"):
                current = {str(row["series"]): row for row in selected}
                keys = sorted(current)
                paired = _paired_statistics_v2(
                    [float(current[key]["metrics"]["worst_csd"]) for key in keys],
                    [float(mes_rows[key]["metrics"]["worst_csd"]) for key in keys],
                )
                record.update(
                    {
                        "vs_mes_diff": paired["diff"],
                        "vs_mes_p": paired["p_two_sided"],
                        "vs_mes_wins": paired["wins"],
                        "n": paired["n"],
                    }
                )
            output.append(record)
    return {
        "schema": "demographic-partition-robustness-v2",
        "row_count": len(output),
        "scheme_rows": output,
    }


def _endowment_seed_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    fits: Mapping[str, VerifiedFitSnapshotV2],
) -> dict[str, object]:
    output = []
    for fit_id in _fit_ids(inventory, "temporal_endowment"):
        fit = fits[fit_id]
        train = _one_summary_v2(rows, fit_id, view="train")
        test = _one_summary_v2(rows, fit_id)
        output.append(
            {
                "fit_id": fit_id,
                "seed": int(fit.spec["seed"]),
                "train_worst_csd": train["worst_csd"],
                "test_worst_csd": test["worst_csd"],
                "test_mean_csd": test["mean_csd"],
                "test_welfare": test["welfare"],
                "test_cost_welfare": test["cost_welfare"],
                "test_exclusion": test["exclusion"],
                "best_train_loss": fit.payload["result"]["best_loss"],
                "weights": list(fit.payload["result"]["selected_weights"]),
            }
        )
    output.sort(key=lambda row: int(row["seed"]))
    return {
        "schema": "endowment-seed-stability-v2",
        "seed_count": len(output),
        "seed_rows": output,
    }


def _matched_target_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    endow = _one_fit_id_v2(
        inventory, "endowment_frontier", soft_welfare_target=1.0
    )
    direct = _one_fit_id_v2(
        inventory,
        "outcome_frontier",
        seed=42,
        soft_welfare_target=1.0,
    )
    paired = _paired_source_values_v2(rows, endow, direct, seed=20260812)
    endow_summary = _one_summary_v2(rows, endow)
    direct_summary = _one_summary_v2(rows, direct)
    record = {
        "soft_target": 1.0,
        "endowment_source_id": endow,
        "direct_source_id": direct,
        "n_series": paired["n"],
        "endowment_csd": paired["learned_mean"],
        "direct_csd": paired["baseline_mean"],
        "difference": paired["diff"],
        "ci_low": paired["ci_lo"],
        "ci_high": paired["ci_hi"],
        "p_value": paired["p_two_sided"],
        "endowment_wins": paired["wins"],
        "endowment_welfare": endow_summary["welfare"],
        "direct_welfare": direct_summary["welfare"],
        "endowment_exclusion": endow_summary["exclusion"],
        "direct_exclusion": direct_summary["exclusion"],
    }
    return {
        "schema": "matched-target-contrast-v2",
        "row_count": 1,
        "matched_rows": [record],
    }


def _series_data_index_v2(
    data_by_split: Mapping[str, Sequence[SeriesData]],
    split: str,
) -> dict[str, SeriesData]:
    data = data_by_split.get(split)
    if not isinstance(data, Sequence):
        raise RuntimeError(f"primary-v2 consumer data split is missing: {split}")
    result = {row.ref.key: row for row in data}
    if len(result) != len(data):
        raise RuntimeError(f"primary-v2 consumer data duplicates a series: {split}")
    return result


def _scored_instances_v2(
    row: Mapping[str, object],
    series_data: SeriesData,
) -> list[tuple[int, object, set[str]]]:
    outcomes = {
        int(outcome["year"]): outcome for outcome in row["year_outcomes"]
    }
    output = []
    for year in row["scored_years"]:
        instance = series_data.all_years.get(year)
        outcome = outcomes.get(year)
        if instance is None or outcome is None or outcome["scored"] is not True:
            raise RuntimeError("primary-v2 consumer scored instance differs")
        output.append((int(year), instance, set(outcome["winners"])))
    return output


def _failure_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    data_by_split: Mapping[str, Sequence[SeriesData]],
) -> dict[str, object]:
    outcome = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    learned = {
        str(row["series"]): row for row in _coordinate_rows_v2(rows, outcome)
    }
    mes = {str(row["series"]): row for row in _coordinate_rows_v2(rows, "mes")}
    data = _series_data_index_v2(data_by_split, "temporal_2022")
    if sorted(learned) != sorted(mes) or sorted(learned) != sorted(data):
        raise RuntimeError("primary-v2 failure-diagnostic series coverage differs")
    output = []
    for key in sorted(learned):
        overlaps: list[float] = []
        ballot_dispersion: list[float] = []
        voters: list[int] = []
        for year in learned[key]["scored_years"]:
            instance = data[key].all_years.get(year)
            if instance is None:
                raise RuntimeError("primary-v2 failure-diagnostic year differs")
            features = instance_features(instance, "age_sex")
            if not features.cohort_order:
                continue
            overlaps.append(float(np.std(features.static["overlap_dev"])))
            ballot_dispersion.append(
                float(np.std(features.static["ballot_length_dev"]))
            )
            voters.append(len(instance.votes))
        learned_metric = learned[key]["metrics"]
        mes_metric = mes[key]["metrics"]
        delta = float(learned_metric["worst_csd"]) - float(
            mes_metric["worst_csd"]
        )
        worst = learned_metric["worst_cohort"]
        output.append(
            {
                "series": key.split("/")[-1],
                "source_series": key,
                "learned_source_id": outcome,
                "learned_minus_mes": delta,
                "loses": int(delta >= 0.0),
                "worst_cohort": worst,
                "is_senior_worst": int("60+" in str(worst)),
                "mes_csd": mes_metric["worst_csd"],
                "cohort_divergence": float(np.mean(overlaps)),
                "ballot_len_dispersion": float(np.mean(ballot_dispersion)),
                "mean_voters": float(np.mean(voters)),
                "n_test_years": len(learned[key]["scored_years"]),
            }
        )
    output.sort(key=lambda row: float(row["learned_minus_mes"]))
    senior = sum("60+" in str(row["worst_cohort"]) for row in output)
    male = sum("|M" in str(row["worst_cohort"]) for row in output)
    senior_male = sum(
        "60+" in str(row["worst_cohort"])
        and "|M" in str(row["worst_cohort"])
        for row in output
    )
    denominator = len(output)
    return {
        "schema": "outcome-failure-diagnostics-v2",
        "series_count": denominator,
        "series_diagnostics": output,
        "worst_cohort_mix": {
            "WorstCohortSeniorMale": {
                "numerator": senior_male,
                "denominator": denominator,
            },
            "WorstCohortSenior": {
                "numerator": senior,
                "denominator": denominator,
            },
            "WorstCohortMale": {
                "numerator": male,
                "denominator": denominator,
            },
        },
    }


def _uncovered_exposure_v2(
    instance: object, winners: set[str]
) -> tuple[int, float, float]:
    approvers: dict[str, int] = {}
    for vote in instance.votes:
        if cohort_of(vote, "age_sex") is None:
            continue
        for project in vote.projects:
            approvers[project] = approvers.get(project, 0) + 1
    uncovered = 0
    uncovered_spend = 0.0
    total_spend = 0.0
    for project in sorted(winners):
        item = instance.projects.get(project)
        if item is None:
            raise RuntimeError("primary-v2 attribution winner is unknown")
        total_spend += float(item.cost)
        if approvers.get(project, 0) == 0:
            uncovered += 1
            uncovered_spend += float(item.cost)
    return uncovered, uncovered_spend, total_spend


def _attribution_coverage_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    data_by_split: Mapping[str, Sequence[SeriesData]],
) -> dict[str, object]:
    endow = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    direct = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    corner = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=0.0
    )
    policies = (
        ("mes", "mes"),
        ("learned-endowment", endow),
        ("learned-direct", direct),
        ("learned-direct-unpenalized", corner),
    )
    data = _series_data_index_v2(data_by_split, "temporal_2022")
    clean_by_policy: dict[str, set[str]] = {}
    output = []
    for policy, source in policies:
        selected = _coordinate_rows_v2(rows, source)
        editions = affected = funded = uncovered = 0
        uncovered_spend = total_spend = 0.0
        clean: set[str] = set()
        for row in selected:
            series = str(row["series"])
            series_clean = True
            if series not in data:
                raise RuntimeError("primary-v2 attribution series differs")
            for _, instance, winners in _scored_instances_v2(row, data[series]):
                count, spend, total = _uncovered_exposure_v2(instance, winners)
                editions += 1
                affected += int(count > 0)
                funded += len(winners)
                uncovered += count
                uncovered_spend += spend
                total_spend += total
                series_clean = series_clean and count == 0
            if series_clean:
                clean.add(series)
        clean_by_policy[policy] = clean
        output.append(
            {
                "policy": policy,
                "source_id": source,
                "scored_editions": editions,
                "editions_with_uncovered_funded_project": affected,
                "funded_projects": funded,
                "uncovered_funded_projects": uncovered,
                "uncovered_budget_share": (
                    uncovered_spend / total_spend if total_spend else 0.0
                ),
            }
        )
    all_series = set(data)
    joint = all_series & clean_by_policy["mes"] & clean_by_policy["learned-endowment"]
    return {
        "schema": "attribution-coverage-v2",
        "policy_count": len(output),
        "policy_rows": output,
        "joint_clean_series": {
            "policies": ["mes", "learned-endowment"],
            "n_series": len(joint),
            "series": sorted(joint),
        },
    }


def _payment_kernel_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    data_by_split: Mapping[str, Sequence[SeriesData]],
) -> dict[str, object]:
    data = _series_data_index_v2(data_by_split, "temporal_2022")
    total_budget = math.fsum(
        float(row.all_years[year].budget)
        for row in data.values()
        for year in row.test_years
    )
    per_series = []
    summaries = []
    for floor in (0.0, 0.85, 1.0):
        direct = _one_fit_id_v2(
            inventory,
            "outcome_frontier",
            seed=42,
            soft_welfare_target=floor,
        )
        sources = (
            ("direct", direct),
            ("static-floor", f"payment/{direct}/static-floor"),
            ("payment+completion", f"payment/{direct}/payment+completion"),
            ("payment-only", f"payment/{direct}/payment-only"),
        )
        direct_rows = {
            str(row["series"]): row for row in _coordinate_rows_v2(rows, direct)
        }
        for kernel, source in sources:
            selected = _coordinate_rows_v2(rows, source)
            for row in selected:
                metrics = row["metrics"]
                per_series.append(
                    {
                        "floor": floor,
                        "kernel": kernel,
                        "source_id": source,
                        "series": str(row["series"]),
                        "n_years": len(row["scored_years"]),
                        "worst_csd": metrics["worst_csd"],
                        "mean_csd": metrics["mean_csd"],
                        "welfare": metrics["welfare"],
                        "exclusion": metrics["exclusion"],
                        "spent": math.fsum(
                            float(year["spent"])
                            for year in row["year_outcomes"]
                            if year["scored"]
                        ),
                    }
                )
            summary = _one_summary_v2(rows, source)
            by_series = {str(row["series"]): row for row in selected}
            keys = sorted(by_series)
            direct_keys = sorted(direct_rows)
            if keys != direct_keys:
                raise RuntimeError("primary-v2 payment series coverage differs")
            paired = _paired_statistics_v2(
                [float(by_series[key]["metrics"]["worst_csd"]) for key in keys],
                [float(direct_rows[key]["metrics"]["worst_csd"]) for key in keys],
            )
            exclusion = _paired_statistics_v2(
                [float(by_series[key]["metrics"]["exclusion"]) for key in keys],
                [float(direct_rows[key]["metrics"]["exclusion"]) for key in keys],
            )
            spent = math.fsum(
                float(year["spent"])
                for row in selected
                for year in row["year_outcomes"]
                if year["scored"]
            )
            direct_summary = _one_summary_v2(rows, direct)
            summaries.append(
                {
                    "floor": floor,
                    "kernel": kernel,
                    "source_id": source,
                    "worst_csd": summary["worst_csd"],
                    "mean_csd": summary["mean_csd"],
                    "welfare": summary["welfare"],
                    "exclusion": summary["exclusion"],
                    "spent": spent,
                    "budget_utilization": spent / total_budget,
                    "welfare_ratio_vs_direct": float(summary["welfare"])
                    / float(direct_summary["welfare"]),
                    "diff_vs_direct": paired["diff"],
                    "ci_lo": paired["ci_lo"],
                    "ci_hi": paired["ci_hi"],
                    "p_two_sided": paired["p_two_sided"],
                    "wins_vs_direct": paired["wins"],
                    "n": paired["n"],
                    "exclusion_diff_vs_direct": exclusion["diff"],
                    "exclusion_ci_lo": exclusion["ci_lo"],
                    "exclusion_ci_hi": exclusion["ci_hi"],
                    "exclusion_p_two_sided": exclusion["p_two_sided"],
                    "exclusion_wins_vs_direct": exclusion["wins"],
                }
            )
    per_series.sort(key=lambda row: (float(row["floor"]), str(row["kernel"]), str(row["series"])))
    summaries.sort(key=lambda row: (float(row["floor"]), str(row["kernel"])))
    return {
        "schema": "payment-kernel-intervention-v2",
        "floor_count": 3,
        "kernel_count": 4,
        "series_kernel_rows": per_series,
        "kernel_summary_rows": summaries,
        "analysis_metadata": {
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": 42,
            "hypothesis_test": "exact paired two-sided sign-flip over series",
            "warning": "payment-prioritized score is a diagnostic, not standard MES",
        },
    }


def _winner_support_stats_v2(
    instance: object, winners: set[str]
) -> tuple[float | None, float | None]:
    if not winners:
        return None, None
    counts = {project: 0 for project in sorted(winners)}
    for vote in instance.votes:
        for project in vote.projects:
            if project in counts:
                counts[project] += 1
    voters = len(instance.votes) or 1
    project = min(counts, key=lambda item: (counts[item] / voters, item))
    return (
        float(counts[project] / voters),
        float(instance.projects[project].cost / (instance.budget or 1.0)),
    )


def _outside_uniform_floor_v2(
    records: Sequence[tuple[object, set[str]]],
) -> dict[str, object]:
    violating_projects = total_projects = 0
    violating_cost = total_cost = 0.0
    for instance, winners in records:
        counts: dict[str, int] = {}
        for vote in instance.votes:
            for project in vote.projects:
                counts[project] = counts.get(project, 0) + 1
        voters = len(instance.votes) or 1
        for project in sorted(winners):
            cost = float(instance.projects[project].cost)
            total_projects += 1
            total_cost += cost
            if counts.get(project, 0) / voters < cost / instance.budget - 1e-12:
                violating_projects += 1
                violating_cost += cost
    return {
        "pct_projects": 100.0 * violating_projects / max(total_projects, 1),
        "pct_budget": 100.0 * violating_cost / max(total_cost, 1e-9),
    }


def _theory_floor_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    fits: Mapping[str, VerifiedFitSnapshotV2],
    data_by_split: Mapping[str, Sequence[SeriesData]],
) -> dict[str, object]:
    endow = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    direct = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    corner = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=0.0
    )
    policies = (
        ("mes", "mes", uniform_policy),
        ("res-1.0", "res-1.0", res_policy(1.0, "age_sex")),
        (
            "learned-endowment",
            endow,
            linear_policy(
                fits[endow].payload["result"]["selected_weights"], "age_sex"
            ),
        ),
        ("greedy-cost", "greedy-cost", None),
        ("learned-outcome-f1.0", direct, None),
        ("learned-outcome-f0.0", corner, None),
    )
    data = _series_data_index_v2(data_by_split, "temporal_2022")
    theory_rows: list[dict[str, object]] = []
    winner_records: dict[str, list[tuple[object, set[str]]]] = {
        policy: [] for policy, _, _ in policies
    }
    for policy, source, raw_endowment in policies:
        selected = _coordinate_rows_v2(rows, source)
        for row in selected:
            series = str(row["series"])
            series_data = data.get(series)
            if series_data is None:
                raise RuntimeError("primary-v2 theory series differs")
            by_year = {
                int(outcome["year"]): outcome for outcome in row["year_outcomes"]
            }
            state = RolloutState()
            for year in series_data.ref.years:
                instance = series_data.all_years.get(year)
                outcome = by_year.get(year)
                if instance is None or outcome is None:
                    continue
                winners = set(outcome["winners"])
                beta: Sequence[float] | None = None
                payment_winners: set[str] = set()
                if raw_endowment is not None:
                    beta = raw_endowment(instance, state)
                    payment_winners = mes_with_endowments(
                        instance, endowments=beta, completion=False
                    )
                state.update(group_outcome(instance, winners, "age_sex"))
                if outcome["scored"] is not True:
                    continue
                winner_records[policy].append((instance, winners))
                min_share, min_cost_share = _winner_support_stats_v2(
                    instance, winners
                )
                if beta is None or not beta or not instance.votes:
                    kappa = None
                    implied_floor = None
                    violations = 0
                else:
                    base = float(instance.budget) / len(instance.votes)
                    kappa = float(max(beta) / base) if base else None
                    implied_floor = (
                        float(min_cost_share / kappa)
                        if min_cost_share is not None and kappa and kappa > 0.0
                        else None
                    )
                    counts = {project: 0 for project in payment_winners}
                    for vote in instance.votes:
                        for project in vote.projects:
                            if project in counts:
                                counts[project] += 1
                    violations = sum(
                        counts[project] / (len(instance.votes) or 1) + 1e-9
                        < float(instance.projects[project].cost)
                        / ((instance.budget or 1.0) * float(kappa))
                        for project in payment_winners
                    )
                theory_rows.append(
                    {
                        "series": series.split("/")[-1],
                        "source_series": series,
                        "year": int(year),
                        "policy": policy,
                        "source_id": source,
                        "kappa": kappa,
                        "min_approval_share": min_share,
                        "implied_floor": implied_floor,
                        "violates_floor": int(violations),
                    }
                )
    theory_rows.sort(key=lambda row: (str(row["source_series"]), int(row["year"]), str(row["policy"])))
    infeasible = [
        {
            "policy": policy,
            "source_id": source,
            **_outside_uniform_floor_v2(winner_records[policy]),
        }
        for policy, source, _ in policies
    ]
    return {
        "schema": "empirical-support-floor-v2",
        "policy_count": len(policies),
        "edition_row_count": len(theory_rows),
        "theory_rows": theory_rows,
        "infeasible_spend_rows": infeasible,
    }


def _project_representation_tv_v2(
    instance: object, project: str, scheme: str = "age_sex"
) -> float | None:
    features = project_features(instance, scheme)
    index = {
        project_id: row for row, project_id in enumerate(features.project_ids)
    }
    row = index.get(project)
    if row is None:
        return None
    supporter_share = features.cohort_share[row]
    if supporter_share.sum() <= 0:
        return None
    election = instance_features(instance, scheme)
    electorate_share = np.asarray(
        [election.cohort_sizes[group] for group in features.cohort_order],
        dtype=float,
    )
    if electorate_share.sum() <= 0:
        return None
    electorate_share /= electorate_share.sum()
    return float(0.5 * np.abs(supporter_share - electorate_share).sum())


def _composition_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    data_by_split: Mapping[str, Sequence[SeriesData]],
) -> dict[str, object]:
    endow = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    direct = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    corner = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=0.0
    )
    policies = (
        ("mes", "mes"),
        ("learned-endowment", endow),
        ("greedy-cost", "greedy-cost"),
        ("llmrule-cost", "llmrule-cost"),
        ("llmrule-card", "llmrule-card"),
        ("outcome-f0.00", corner),
        ("outcome-f1.00", direct),
    )
    data = _series_data_index_v2(data_by_split, "temporal_2022")
    output = []
    for policy, source in policies:
        funded_counts: list[int] = []
        mean_cost_shares: list[float] = []
        mean_concentrations: list[float] = []
        weighted_tv = total_cost = 0.0
        for row in _coordinate_rows_v2(rows, source):
            series = str(row["series"])
            if series not in data:
                raise RuntimeError("primary-v2 composition series differs")
            for _, instance, winners in _scored_instances_v2(row, data[series]):
                features = project_features(instance, "age_sex")
                index = {
                    project_id: item
                    for item, project_id in enumerate(features.project_ids)
                }
                selected = [index[project] for project in sorted(winners) if project in index]
                if not selected:
                    continue
                funded_counts.append(len(selected))
                mean_cost_shares.append(float(np.mean(features.static[selected, 2])))
                mean_concentrations.append(
                    float(np.mean(features.static[selected, 4]))
                )
                for project in sorted(winners):
                    tv = _project_representation_tv_v2(instance, project)
                    if tv is None:
                        continue
                    cost = float(instance.projects[project].cost)
                    total_cost += cost
                    weighted_tv += cost * tv
        if not funded_counts or total_cost <= 0.0:
            raise RuntimeError("primary-v2 composition has no funded projects")
        output.append(
            {
                "policy": policy,
                "source_id": source,
                "mean_projects_funded": float(np.mean(funded_counts)),
                "mean_cost_share": float(np.mean(mean_cost_shares)),
                "mean_concentration": float(np.mean(mean_concentrations)),
                "cost_weighted_representation_tv": weighted_tv / total_cost,
            }
        )
    return {
        "schema": "allocation-composition-v2",
        "row_count": len(output),
        "composition_rows": output,
    }


def _corpus_values_v2(
    data_by_split: Mapping[str, Sequence[SeriesData]],
    corpus_values: Mapping[str, object],
) -> dict[str, object]:
    full = _series_data_index_v2(data_by_split, "city_out_Poland_Łódź")
    instance_rows = []
    for series in sorted(full):
        row = full[series]
        for year in row.ref.years:
            instance = row.all_years.get(year)
            if instance is None:
                raise RuntimeError("primary-v2 corpus instance differs")
            instance_rows.append(
                {
                    "file": Path(str(instance.path)).name,
                    "series": series,
                    "year": int(year),
                    "budget": float(instance.budget),
                    "n_projects": len(instance.projects),
                    "n_voters": len(instance.votes),
                }
            )
    temporal = _series_data_index_v2(data_by_split, "temporal_2022")
    project_counts = [
        len(row.all_years[year].projects)
        for row in temporal.values()
        for year in row.test_years
    ]
    if not project_counts:
        raise RuntimeError("primary-v2 corpus granularity has no scored elections")
    return {
        "schema": "primary-corpus-inventory-v2",
        "corpus_inventory": dict(sorted(corpus_values.items())),
        "instance_rows": instance_rows,
        "granularity_rows": [
            {
                "corpus": "warsaw_fitting",
                "scored_elections": len(project_counts),
                "median_projects": float(median(project_counts)),
                "mean_projects": float(math.fsum(project_counts) / len(project_counts)),
            }
        ],
        "excluded_legacy_rows": [
            {
                "corpus": "external_projected",
                "reason": "outside the locked primary-Warsaw v2 corpus",
            }
        ],
    }


def _endowment_significance_values_v2(
    rows: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    composition: Mapping[str, object],
    per_series: Mapping[str, object],
) -> dict[str, object]:
    endow = _one_fit_id_v2(inventory, "temporal_endowment", seed=42)
    outcome = _one_fit_id_v2(
        inventory, "outcome_frontier", seed=42, soft_welfare_target=1.0
    )
    contrasts = (
        ("learned-endowment vs mes", "mes"),
        ("learned-endowment vs res-1.0", "res-1.0"),
        ("learned-endowment vs llmrule-cost", "llmrule-cost"),
        ("learned-endowment vs llmrule-card", "llmrule-card"),
        ("learned-endowment vs learned-outcome", outcome),
    )
    output = []
    for label, baseline in contrasts:
        paired = _paired_source_values_v2(rows, endow, baseline)
        output.append(
            {
                "contrast": label,
                "learned_source_id": endow,
                "baseline_source_id": baseline,
                "learned_mean": paired["learned_mean"],
                "baseline_mean": paired["baseline_mean"],
                "diff": paired["diff"],
                "ci_lo": paired["ci_lo"],
                "ci_hi": paired["ci_hi"],
                "p_two_sided": paired["p_two_sided"],
                "wins": paired["wins"],
                "n": paired["n"],
            }
        )
    return {
        "schema": "endowment-significance-suite-v2",
        "contrast_count": len(output),
        "contrast_rows": output,
        "two_arm_rows": list(per_series["two_arm_rows"]),
        "composition_rows": list(composition["composition_rows"]),
    }


_CONSUMER_VALUE_KEYS_V2: dict[str, set[str]] = {
    "baseline.primary": {"schema", "series_count", "series_rows"},
    "baseline.prior_rules": {"schema", "row_count", "rule_view_rows"},
    "control.history_free": {"schema", "policy_count", "policy_rows", "fit_records"},
    "control.senior_scalar": {"schema", "selected", "training_grid"},
    "control.static_age_lookup": {
        "schema", "policy_count", "policy_rows", "contrast_count", "contrast_rows", "fit_record"
    },
    "cross_district.bound10": {
        "schema", "bound", "n_folds", "n_series", "n_policy_series_rows",
        "series_policy_rows", "contrast_rows", "fold_records", "hypothesis_tests",
    },
    "cross_district.bound40": {
        "schema", "bound", "n_folds", "n_series", "n_policy_series_rows",
        "series_policy_rows", "contrast_rows", "fold_records", "hypothesis_tests",
    },
    "failure.outcome": {"schema", "series_count", "series_diagnostics", "worst_cohort_mix"},
    "frontier.endowment": {"schema", "arm", "point_count", "point_rows", "baseline_count", "baseline_rows"},
    "frontier.outcome": {"schema", "arm", "point_count", "point_rows", "baseline_count", "baseline_rows"},
    "mechanism.attribution_coverage": {"schema", "policy_count", "policy_rows", "joint_clean_series"},
    "mechanism.payment_kernel": {
        "schema", "floor_count", "kernel_count", "series_kernel_rows",
        "kernel_summary_rows", "analysis_metadata",
    },
    "mechanism.support_floor": {"schema", "row_count", "identification_rows"},
    "mechanism.theory_floor": {
        "schema", "policy_count", "edition_row_count", "theory_rows", "infeasible_spend_rows"
    },
    "outcome.ablation": {"schema", "row_count", "ablation_rows"},
    "outcome.per_series": {"schema", "row_count", "learned_rows", "two_arm_rows"},
    "outcome.seed_stability": {"schema", "seed_count", "seed_rows"},
    "outcome.significance": {"schema", "contrast_count", "contrast_rows"},
    "primary.corpus": {
        "schema", "corpus_inventory", "instance_rows", "granularity_rows", "excluded_legacy_rows"
    },
    "robustness.demographic_partition": {"schema", "row_count", "scheme_rows"},
    "summary.composition": {"schema", "row_count", "composition_rows"},
    "summary.endowment_seed_stability": {"schema", "seed_count", "seed_rows"},
    "summary.endowment_significance": {
        "schema", "contrast_count", "contrast_rows", "two_arm_rows", "composition_rows"
    },
    "summary.matched_target": {"schema", "row_count", "matched_rows"},
    "transfer.lodz": {"schema", "row_count", "transfer_rows"},
}


_CONSUMER_ROW_FIELDS_V2: dict[tuple[str, str], set[str]] = {
    ("baseline.primary", "series_rows"): {
        "series", "split_view",
        *{
            f"{policy}_{metric}"
            for policy in (
                "greedy-count", "greedy-cost", "llmrule-cost", "llmrule-card",
                "mes", "res-0.25", "res-0.5", "res-0.75", "res-1.0",
            )
            for metric in ("worst", "welfare")
        },
    },
    ("baseline.prior_rules", "rule_view_rows"): {
        "rule", "series", "view", "worst_csd", "mean_csd", "welfare",
        "cost_welfare", "exclusion", "n_series", "source_doi", "score",
    },
    ("control.history_free", "policy_rows"): {
        "policy", "source_id", "test_worst_csd", "test_welfare", "test_exclusion"
    },
    ("control.history_free", "fit_records"): {
        "fit_id", "family", "split", "seed", "soft_welfare_target",
        "support_floor_kappa", "best_train_loss", "selected_weights",
    },
    ("control.static_age_lookup", "policy_rows"): {
        "policy", "source_id", "mean_csd", "welfare", "mean_exclusion"
    },
    ("control.static_age_lookup", "contrast_rows"): {
        "contrast", "learned_source_id", "baseline_source_id", "learned_mean",
        "baseline_mean", "diff", "ci_lo", "ci_hi", "p_two_sided", "wins", "ties", "n",
    },
    **{
        (consumer, "series_policy_rows"): {
            "fold", "split", "series", "policy", "source_id", "n_years",
            "worst_csd", "mean_csd", "welfare", "cost_welfare", "exclusion",
        }
        for consumer in ("cross_district.bound10", "cross_district.bound40")
    },
    **{
        (consumer, "contrast_rows"): {
            "baseline", "learned_mean", "baseline_mean", "diff", "ci_lo", "ci_hi",
            "p_two_sided", "wins", "ties", "n", "fold_mean_diff", "fold_p_two_sided",
            "fold_wins", "n_folds", "learned_welfare", "baseline_welfare", "welfare_ratio",
            "learned_exclusion", "baseline_exclusion",
        }
        for consumer in ("cross_district.bound10", "cross_district.bound40")
    },
    **{
        (consumer, "fold_records"): {"fold", "split", "fit", "test_series", "boundary_features"}
        for consumer in ("cross_district.bound10", "cross_district.bound40")
    },
    ("failure.outcome", "series_diagnostics"): {
        "series", "source_series", "learned_source_id", "learned_minus_mes", "loses",
        "worst_cohort", "is_senior_worst", "mes_csd", "cohort_divergence",
        "ballot_len_dispersion", "mean_voters", "n_test_years",
    },
    **{
        (consumer, "point_rows"): {
            "fit_id", "arm", "floor", "seed", "penalty", "train_worst_csd",
            "train_welfare_ratio", "test_worst_csd", "test_mean_csd", "test_welfare",
            "test_cost_welfare", "test_exclusion", "best_train_loss", "weights",
        }
        for consumer in ("frontier.endowment", "frontier.outcome")
    },
    **{
        (consumer, "baseline_rows"): {
            "policy", "source_id", "view", "n_series", "worst_csd", "mean_csd",
            "welfare", "cost_welfare", "exclusion",
        }
        for consumer in ("frontier.endowment", "frontier.outcome")
    },
    ("mechanism.attribution_coverage", "policy_rows"): {
        "policy", "source_id", "scored_editions", "editions_with_uncovered_funded_project",
        "funded_projects", "uncovered_funded_projects", "uncovered_budget_share",
    },
    ("mechanism.payment_kernel", "series_kernel_rows"): {
        "floor", "kernel", "source_id", "series", "n_years", "worst_csd",
        "mean_csd", "welfare", "exclusion", "spent",
    },
    ("mechanism.payment_kernel", "kernel_summary_rows"): {
        "floor", "kernel", "source_id", "worst_csd", "mean_csd", "welfare", "exclusion",
        "spent", "budget_utilization", "welfare_ratio_vs_direct", "diff_vs_direct",
        "ci_lo", "ci_hi", "p_two_sided", "wins_vs_direct", "n",
        "exclusion_diff_vs_direct", "exclusion_ci_lo", "exclusion_ci_hi",
        "exclusion_p_two_sided", "exclusion_wins_vs_direct",
    },
    ("mechanism.support_floor", "identification_rows"): {
        "fit_id", "kappa", "welfare_floor", "train_worst_csd", "test_worst_csd",
        "test_welfare", "test_exclusion", "welfare_ratio_vs_mes", "ratio_denominator_view",
    },
    ("mechanism.theory_floor", "theory_rows"): {
        "series", "source_series", "year", "policy", "source_id", "kappa",
        "min_approval_share", "implied_floor", "violates_floor",
    },
    ("mechanism.theory_floor", "infeasible_spend_rows"): {
        "policy", "source_id", "pct_projects", "pct_budget"
    },
    ("outcome.ablation", "ablation_rows"): {
        "ablated", "source_id", "test_worst_csd", "test_welfare", "delta_vs_full"
    },
    ("outcome.per_series", "learned_rows"): {
        "series", "test_years", "greedy-count", "greedy-cost", "mes", "res-1.0",
        "learned", "worst_cohort", "learned_minus_mes", "learned_wins",
    },
    ("outcome.per_series", "two_arm_rows"): {
        "series", "mes", "learned_endowment", "learned_outcome", "endowment_minus_mes",
        "outcome_minus_mes", "endowment_wins", "outcome_wins",
    },
    ("outcome.seed_stability", "seed_rows"): {
        "fit_id", "seed", "test_worst_csd", "test_mean_csd", "test_welfare",
        "test_cost_welfare", "test_exclusion", "best_train_loss", "weights",
    },
    ("outcome.significance", "contrast_rows"): {
        "arm", "floor", "source_id", "baseline", "learned_mean", "baseline_mean",
        "diff", "ci_lo", "ci_hi", "p_two_sided", "wins", "n",
    },
    ("primary.corpus", "instance_rows"): {
        "file", "series", "year", "budget", "n_projects", "n_voters"
    },
    ("primary.corpus", "granularity_rows"): {
        "corpus", "scored_elections", "median_projects", "mean_projects"
    },
    ("primary.corpus", "excluded_legacy_rows"): {"corpus", "reason"},
    ("robustness.demographic_partition", "scheme_rows"): {
        "scheme", "n_groups", "policy", "source_id", "worst_csd", "vs_mes_diff",
        "vs_mes_p", "vs_mes_wins", "n",
    },
    ("summary.composition", "composition_rows"): {
        "policy", "source_id", "mean_projects_funded", "mean_cost_share",
        "mean_concentration", "cost_weighted_representation_tv",
    },
    ("summary.endowment_seed_stability", "seed_rows"): {
        "fit_id", "seed", "train_worst_csd", "test_worst_csd", "test_mean_csd",
        "test_welfare", "test_cost_welfare", "test_exclusion", "best_train_loss", "weights",
    },
    ("summary.endowment_significance", "contrast_rows"): {
        "contrast", "learned_source_id", "baseline_source_id", "learned_mean",
        "baseline_mean", "diff", "ci_lo", "ci_hi", "p_two_sided", "wins", "n",
    },
    ("summary.endowment_significance", "two_arm_rows"): {
        "series", "mes", "learned_endowment", "learned_outcome", "endowment_minus_mes",
        "outcome_minus_mes", "endowment_wins", "outcome_wins",
    },
    ("summary.endowment_significance", "composition_rows"): {
        "policy", "source_id", "mean_projects_funded", "mean_cost_share",
        "mean_concentration", "cost_weighted_representation_tv",
    },
    ("summary.matched_target", "matched_rows"): {
        "soft_target", "endowment_source_id", "direct_source_id", "n_series",
        "endowment_csd", "direct_csd", "difference", "ci_low", "ci_high", "p_value",
        "endowment_wins", "endowment_welfare", "direct_welfare", "endowment_exclusion",
        "direct_exclusion",
    },
    ("transfer.lodz", "transfer_rows"): {
        "series", "policy", "source_id", "worst_csd", "welfare", "exclusion", "worst_cohort"
    },
}


_CONSUMER_OBJECT_FIELDS_V2: dict[tuple[str, str], set[str]] = {
    ("control.senior_scalar", "selected"): {
        "alpha", "train_worst_csd", "test_worst_csd", "mes_worst_csd",
        "difference", "ci_low", "ci_high", "p_value", "wins", "n_series",
        "test_welfare", "test_exclusion",
    },
    ("control.senior_scalar", "training_grid"): {
        "candidate_count", "alphas", "train_worst_csd", "selection_rule"
    },
    ("control.static_age_lookup", "fit_record"): {
        "fit_id", "family", "split", "seed", "soft_welfare_target",
        "support_floor_kappa", "best_train_loss", "selected_weights",
    },
    ("cross_district.bound10", "hypothesis_tests"): {
        "pooled_series", "paired_series_interval", "fold_block"
    },
    ("cross_district.bound40", "hypothesis_tests"): {
        "pooled_series", "paired_series_interval", "fold_block"
    },
    ("failure.outcome", "worst_cohort_mix"): {
        "WorstCohortSeniorMale", "WorstCohortSenior", "WorstCohortMale"
    },
    ("mechanism.attribution_coverage", "joint_clean_series"): {
        "policies", "n_series", "series"
    },
    ("mechanism.payment_kernel", "analysis_metadata"): {
        "bootstrap_replicates", "bootstrap_seed", "hypothesis_test", "warning"
    },
}


def validate_consumer_values_v2(
    consumer_id: str, values: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate one specialized consumer's exact keys, rows, and quantities."""

    contract = PAPER_CONSUMER_VALUE_CONTRACT_V2.get(consumer_id)
    if contract is None or type(values) is not dict:
        raise RuntimeError("primary-v2 typed consumer schema differs")
    if set(values) != _CONSUMER_VALUE_KEYS_V2[consumer_id]:
        raise RuntimeError(f"primary-v2 typed consumer key schema differs: {consumer_id}")
    if values.get("schema") != contract["schema"]:
        raise RuntimeError(f"primary-v2 typed consumer schema differs: {consumer_id}")
    lists = contract["lists"]
    assert isinstance(lists, Mapping)
    for field, expected_count in lists.items():
        records = values.get(field)
        if type(records) is not list or len(records) != expected_count:
            raise RuntimeError(
                f"primary-v2 typed consumer quantity differs: {consumer_id}/{field}"
            )
        expected_fields = _CONSUMER_ROW_FIELDS_V2[(consumer_id, str(field))]
        for record in records:
            if type(record) is not dict or set(record) != expected_fields:
                raise RuntimeError(
                    f"primary-v2 typed consumer row schema differs: {consumer_id}/{field}"
                )
    for (owner, field), expected_fields in _CONSUMER_OBJECT_FIELDS_V2.items():
        if owner != consumer_id:
            continue
        record = values.get(field)
        if type(record) is not dict or set(record) != expected_fields:
            raise RuntimeError(
                f"primary-v2 typed consumer object schema differs: {consumer_id}/{field}"
            )
    if consumer_id.startswith("cross_district."):
        fit_fields = _CONSUMER_ROW_FIELDS_V2[("control.history_free", "fit_records")]
        for record in values["fold_records"]:
            if type(record["fit"]) is not dict or set(record["fit"]) != fit_fields:
                raise RuntimeError("primary-v2 typed district fit schema differs")
    if consumer_id == "failure.outcome":
        for key in _CONSUMER_OBJECT_FIELDS_V2[(consumer_id, "worst_cohort_mix")]:
            record = values["worst_cohort_mix"][key]
            if type(record) is not dict or set(record) != {"numerator", "denominator"}:
                raise RuntimeError("primary-v2 typed worst-cohort mix schema differs")
    count_fields = {
        "series_rows": "series_count",
        "rule_view_rows": "row_count",
        "policy_rows": "policy_count",
        "contrast_rows": "contrast_count",
        "point_rows": "point_count",
        "identification_rows": "row_count",
        "theory_rows": "edition_row_count",
        "ablation_rows": "row_count",
        "learned_rows": "row_count",
        "seed_rows": "seed_count",
        "scheme_rows": "row_count",
        "composition_rows": "row_count",
        "matched_rows": "row_count",
        "transfer_rows": "row_count",
    }
    for list_field, count_field in count_fields.items():
        if list_field in lists and count_field in values:
            if type(values[count_field]) is not int or values[count_field] != len(values[list_field]):
                raise RuntimeError(
                    f"primary-v2 typed consumer declared count differs: {consumer_id}"
                )
    if consumer_id.startswith("cross_district."):
        if (
            values["n_folds"] != len(values["fold_records"])
            or values["n_series"] != 19
            or values["n_policy_series_rows"] != len(values["series_policy_rows"])
        ):
            raise RuntimeError("primary-v2 typed district quantities differ")
    if consumer_id.startswith("frontier.") and values["baseline_count"] != len(values["baseline_rows"]):
        raise RuntimeError("primary-v2 typed frontier baseline count differs")
    if consumer_id == "mechanism.payment_kernel" and (
        values["floor_count"] != 3 or values["kernel_count"] != 4
    ):
        raise RuntimeError("primary-v2 typed payment quantity differs")
    if consumer_id == "mechanism.theory_floor" and values["policy_count"] != 6:
        raise RuntimeError("primary-v2 typed theory policy count differs")
    _require_finite_json_v2(values, label=f"primary-v2 typed consumer {consumer_id}")
    return values


def reconstruct_paper_consumer_values_v2(
    per_series_rows: Sequence[Mapping[str, object]],
    *,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    data_by_split: Mapping[str, Sequence[SeriesData]],
    corpus_values: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Build all 25 paper interfaces from authenticated replay and fit state."""

    inventory = [fit.spec for fit in inventory_snapshot.fits]
    fits = {str(fit.spec["fit_id"]): fit for fit in inventory_snapshot.fits}
    outcome_series = _outcome_per_series_values_v2(per_series_rows, inventory)
    composition = _composition_values_v2(
        per_series_rows, inventory, data_by_split
    )
    values = {
        "baseline.primary": _baseline_primary_values_v2(per_series_rows),
        "baseline.prior_rules": _prior_rule_values_v2(per_series_rows),
        "control.history_free": _history_free_values_v2(
            per_series_rows, inventory, fits
        ),
        "control.senior_scalar": _senior_scalar_values_v2(
            per_series_rows, inventory_snapshot.grid
        ),
        "control.static_age_lookup": _static_age_lookup_values_v2(
            per_series_rows, inventory, fits
        ),
        "cross_district.bound10": _cross_district_values_v2(
            per_series_rows, inventory, fits, bound=10.0
        ),
        "cross_district.bound40": _cross_district_values_v2(
            per_series_rows, inventory, fits, bound=40.0
        ),
        "failure.outcome": _failure_values_v2(
            per_series_rows, inventory, data_by_split
        ),
        "frontier.endowment": _frontier_values_v2(
            per_series_rows, inventory, fits, family="endowment_frontier"
        ),
        "frontier.outcome": _frontier_values_v2(
            per_series_rows, inventory, fits, family="outcome_frontier"
        ),
        "mechanism.attribution_coverage": _attribution_coverage_values_v2(
            per_series_rows, inventory, data_by_split
        ),
        "mechanism.payment_kernel": _payment_kernel_values_v2(
            per_series_rows, inventory, data_by_split
        ),
        "mechanism.support_floor": _support_floor_values_v2(
            per_series_rows, inventory
        ),
        "mechanism.theory_floor": _theory_floor_values_v2(
            per_series_rows, inventory, fits, data_by_split
        ),
        "outcome.ablation": build_outcome_ablation_values_v2(
            per_series_rows, inventory
        ),
        "outcome.per_series": outcome_series,
        "outcome.seed_stability": _outcome_seed_values_v2(
            per_series_rows, inventory, fits
        ),
        "outcome.significance": _outcome_significance_values_v2(
            per_series_rows, inventory
        ),
        "primary.corpus": _corpus_values_v2(data_by_split, corpus_values),
        "robustness.demographic_partition": _demographic_partition_values_v2(
            per_series_rows, inventory
        ),
        "summary.composition": composition,
        "summary.endowment_seed_stability": _endowment_seed_values_v2(
            per_series_rows, inventory, fits
        ),
        "summary.endowment_significance": _endowment_significance_values_v2(
            per_series_rows, inventory, composition, outcome_series
        ),
        "summary.matched_target": _matched_target_values_v2(
            per_series_rows, inventory
        ),
        "transfer.lodz": build_transfer_lodz_values_v2(
            per_series_rows, inventory
        ),
    }
    if set(values) != set(PAPER_CONSUMER_VALUE_CONTRACT_V2):
        raise RuntimeError("primary-v2 specialized consumer coverage differs")
    for consumer_id, consumer_values in values.items():
        validate_consumer_values_v2(consumer_id, consumer_values)
    return values


def build_paper_consumers_payload_v2(
    per_series_payload: Mapping[str, object],
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    replay_receipt_snapshot: ReplayReceiptSnapshotV2,
    data_by_split: Mapping[str, Sequence[SeriesData]],
    corpus_values: Mapping[str, object],
) -> dict[str, object]:
    """Materialize every manuscript consumer from the normalized replay rows."""

    fit_ids = [fit.spec["fit_id"] for fit in inventory_snapshot.fits]
    validate_per_series_payload_v2(per_series_payload, expected_fit_ids=fit_ids)
    _validate_replay_authorization_v2(
        replay_receipt_snapshot,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
    )
    rows = per_series_payload["rows"]
    assert isinstance(rows, list)
    source_contract = expected_consumer_source_ids_v2(
        [fit.spec for fit in inventory_snapshot.fits]
    )
    policy_contract = _consumer_policy_ids_v2(
        [fit.spec for fit in inventory_snapshot.fits]
    )
    if set(policy_contract) != set(PAPER_CONSUMER_CONTRACT_V2):
        raise RuntimeError("primary-v2 consumer policy registry is incomplete")
    values_by_consumer = reconstruct_paper_consumer_values_v2(
        rows,
        inventory_snapshot=inventory_snapshot,
        data_by_split=data_by_split,
        corpus_values=corpus_values,
    )
    consumers: dict[str, object] = {}
    for consumer_id in sorted(PAPER_CONSUMER_CONTRACT_V2):
        source_ids = source_contract[consumer_id]
        policy_ids = sorted(policy_contract[consumer_id])
        consumers[consumer_id] = {
            "consumer_id": consumer_id,
            "legacy_paths": list(
                PAPER_CONSUMER_CONTRACT_V2[consumer_id]["legacy_paths"]
            ),
            "source_ids": source_ids,
            "policy_ids": policy_ids,
            "values": values_by_consumer[consumer_id],
        }
    macros, macro_coordinates = _manuscript_macros_v2(
        rows,
        corpus_values,
        [fit.spec for fit in inventory_snapshot.fits],
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "classification": "corrected post-hoc replay",
        "consumer_ids": sorted(PAPER_CONSUMER_CONTRACT_V2),
        "consumers": consumers,
        "manuscript_macros": macros,
        "provenance": {
            "protocol_lock_sha256": lock_snapshot.sha256,
            "replay_receipt_sha256": replay_receipt_snapshot.sha256,
            "per_series_sha256": hashlib.sha256(
                _canonical_json_bytes_v2(per_series_payload)
            ).hexdigest(),
            "macro_coordinates": macro_coordinates,
        },
    }
    validate_paper_consumers_payload_v2(
        payload, inventory=[fit.spec for fit in inventory_snapshot.fits]
    )
    return payload


_YEAR_FIELDS = {
    "year",
    "scored",
    "winners",
    "spent",
    "welfare",
    "cost_welfare",
    "exclusion",
}
_METRIC_FIELDS = {
    "worst_csd",
    "worst_cohort",
    "mean_csd",
    "welfare",
    "cost_welfare",
    "exclusion",
}
_ROW_FIELDS = {
    "fit_id",
    "source_kind",
    "split",
    "view",
    "scheme",
    "series",
    "scored_years",
    "year_outcomes",
    "metrics",
}


def validate_per_series_payload_v2(
    payload: Mapping[str, object], *, expected_fit_ids: Sequence[object]
) -> Mapping[str, object]:
    """Validate normalized episode rows and complete locked-fit coverage."""

    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "semantics_profile",
        "rows",
        "provenance",
    }:
        raise RuntimeError("primary-v2 per-series schema differs")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("semantics_profile") != SEMANTICS_PROFILE
        or type(payload.get("provenance")) is not dict
    ):
        raise RuntimeError("primary-v2 per-series identity differs")
    rows = payload.get("rows")
    if type(rows) is not list or not rows:
        raise RuntimeError("primary-v2 per-series rows are empty")
    keys: list[tuple[str, str, str, str, str]] = []
    covered: set[str] = set()
    expected = {str(value) for value in expected_fit_ids}
    if len(expected) != len(expected_fit_ids):
        raise RuntimeError("primary-v2 expected fit ids contain duplicates")
    for row in rows:
        if type(row) is not dict or set(row) != _ROW_FIELDS:
            raise RuntimeError("primary-v2 per-series row schema differs")
        for field in ("fit_id", "source_kind", "split", "view", "scheme", "series"):
            if type(row.get(field)) is not str or not row[field]:
                raise RuntimeError(f"primary-v2 per-series {field} differs")
        if row["source_kind"] not in {"fit", "fixed", "diagnostic"}:
            raise RuntimeError("primary-v2 per-series source kind differs")
        if row["view"] not in {"train", "test"}:
            raise RuntimeError("primary-v2 per-series view differs")
        if row["source_kind"] == "fit":
            if row["fit_id"] not in expected:
                raise RuntimeError("primary-v2 per-series names an unknown fit")
            covered.add(row["fit_id"])
        scored_years = row.get("scored_years")
        outcomes = row.get("year_outcomes")
        if (
            type(scored_years) is not list
            or any(type(year) is not int for year in scored_years)
            or scored_years != sorted(set(scored_years))
            or type(outcomes) is not list
        ):
            raise RuntimeError("primary-v2 per-series year inventory differs")
        outcome_years: list[int] = []
        observed_scored: list[int] = []
        for outcome in outcomes:
            if type(outcome) is not dict or set(outcome) != _YEAR_FIELDS:
                raise RuntimeError("primary-v2 year outcome schema differs")
            year = outcome.get("year")
            winners = outcome.get("winners")
            if type(year) is not int or type(outcome.get("scored")) is not bool:
                raise RuntimeError("primary-v2 year outcome identity differs")
            if (
                type(winners) is not list
                or any(type(winner) is not str for winner in winners)
                or winners != sorted(set(winners))
            ):
                raise RuntimeError("primary-v2 winner ids are not sorted and unique")
            for field in ("spent", "welfare", "cost_welfare", "exclusion"):
                value = outcome.get(field)
                if type(value) is not float or not math.isfinite(value):
                    raise RuntimeError("primary-v2 year metric is non-finite or aliased")
            outcome_years.append(year)
            if outcome["scored"]:
                observed_scored.append(year)
        if outcome_years != sorted(set(outcome_years)) or observed_scored != scored_years:
            raise RuntimeError("primary-v2 scored-year trace differs")
        metrics = row.get("metrics")
        if type(metrics) is not dict or set(metrics) != _METRIC_FIELDS:
            raise RuntimeError("primary-v2 episode metric schema differs")
        for field in ("worst_csd", "mean_csd"):
            value = metrics.get(field)
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise RuntimeError("primary-v2 episode CSD is non-finite or aliased")
        worst = metrics.get("worst_cohort")
        if worst is not None and type(worst) is not str:
            raise RuntimeError("primary-v2 worst cohort type differs")
        for field in ("welfare", "cost_welfare", "exclusion"):
            value = metrics.get(field)
            if type(value) is not float or not math.isfinite(value):
                raise RuntimeError("primary-v2 episode metric is non-finite or aliased")
        keys.append(
            (
                str(row["fit_id"]),
                str(row["split"]),
                str(row["view"]),
                str(row["scheme"]),
                str(row["series"]),
            )
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("primary-v2 per-series payload contains a duplicate row")
    if covered != expected:
        raise RuntimeError("primary-v2 per-series fit coverage is incomplete")
    _require_finite_json_v2(payload, label="primary-v2 per-series payload")
    return payload


def validate_paper_consumers_payload_v2(
    payload: Mapping[str, object], *, inventory: Sequence[Mapping[str, object]]
) -> Mapping[str, object]:
    """Require exact consumer IDs, paths, and locked fit-source coverage."""

    if type(payload) is not dict or set(payload) != {
        "schema_version",
        "classification",
        "consumer_ids",
        "consumers",
        "manuscript_macros",
        "provenance",
    }:
        raise RuntimeError("primary-v2 paper consumer schema differs")
    expected_ids = sorted(PAPER_CONSUMER_CONTRACT_V2)
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("classification") != "corrected post-hoc replay"
        or payload.get("consumer_ids") != expected_ids
        or type(payload.get("consumers")) is not dict
        or set(payload["consumers"]) != set(expected_ids)
    ):
        raise RuntimeError("primary-v2 paper consumer coverage differs")
    expected_sources = expected_consumer_source_ids_v2(inventory)
    expected_policies = _consumer_policy_ids_v2(inventory)
    for consumer_id in expected_ids:
        consumer = payload["consumers"][consumer_id]
        if type(consumer) is not dict or set(consumer) != {
            "consumer_id",
            "legacy_paths",
            "source_ids",
            "policy_ids",
            "values",
        }:
            raise RuntimeError("primary-v2 paper consumer row schema differs")
        if (
            consumer.get("consumer_id") != consumer_id
            or consumer.get("legacy_paths")
            != list(PAPER_CONSUMER_CONTRACT_V2[consumer_id]["legacy_paths"])
            or consumer.get("source_ids") != expected_sources[consumer_id]
            or type(consumer.get("policy_ids")) is not list
            or any(type(value) is not str for value in consumer["policy_ids"])
            or consumer["policy_ids"]
            != sorted(expected_policies[consumer_id])
            or type(consumer.get("values")) is not dict
            or not consumer["values"]
        ):
            raise RuntimeError(
                f"primary-v2 paper consumer source coverage differs: {consumer_id}"
            )
        validate_consumer_values_v2(consumer_id, consumer["values"])
    macros = payload.get("manuscript_macros")
    if (
        type(macros) is not dict
        or not macros
        or any(type(key) is not str or not key for key in macros)
        or any(type(value) is not str for value in macros.values())
    ):
        raise RuntimeError("primary-v2 manuscript macro coverage differs")
    if type(payload.get("provenance")) is not dict:
        raise RuntimeError("primary-v2 paper consumer provenance differs")
    _require_finite_json_v2(payload, label="primary-v2 paper consumers")
    return payload


def _per_series_csv_bytes_v2(payload: Mapping[str, object]) -> bytes:
    fields = (
        "fit_id",
        "source_kind",
        "split",
        "view",
        "scheme",
        "series",
        "scored_years",
        "worst_csd",
        "worst_cohort",
        "mean_csd",
        "welfare",
        "cost_welfare",
        "exclusion",
        "winner_trace_sha256",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    rows = payload["rows"]
    assert isinstance(rows, list)
    for row in rows:
        metrics = row["metrics"]
        trace_bytes = json.dumps(
            row["year_outcomes"],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        writer.writerow(
            {
                "fit_id": row["fit_id"],
                "source_kind": row["source_kind"],
                "split": row["split"],
                "view": row["view"],
                "scheme": row["scheme"],
                "series": row["series"],
                "scored_years": "|".join(map(str, row["scored_years"])),
                "worst_csd": metrics["worst_csd"],
                "worst_cohort": metrics["worst_cohort"],
                "mean_csd": metrics["mean_csd"],
                "welfare": metrics["welfare"],
                "cost_welfare": metrics["cost_welfare"],
                "exclusion": metrics["exclusion"],
                "winner_trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
            }
        )
    return handle.getvalue().encode("utf-8")


def prepare_evaluation_bundle_v2(
    result_root: Path,
    *,
    per_series_payload: Mapping[str, object],
    paper_consumers_payload: Mapping[str, object],
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    replay_receipt_snapshot: ReplayReceiptSnapshotV2,
) -> PreparedEvaluationBundleV2:
    """Serialize and conflict-preflight the complete five-file bundle in memory."""

    fit_ids = [snapshot.spec["fit_id"] for snapshot in inventory_snapshot.fits]
    validate_per_series_payload_v2(
        per_series_payload, expected_fit_ids=fit_ids
    )
    validate_paper_consumers_payload_v2(
        paper_consumers_payload,
        inventory=[snapshot.spec for snapshot in inventory_snapshot.fits],
    )
    result = Path(result_root).absolute()
    paths = {
        "evaluation/per_series.json": result / "evaluation" / "per_series.json",
        "evaluation/per_series.csv": result / "evaluation" / "per_series.csv",
        "evaluation/paper_consumers.json": result
        / "evaluation"
        / "paper_consumers.json",
        "evaluation/artifact_manifest.json": result
        / "evaluation"
        / "artifact_manifest.json",
        "evaluation/summary.json": result / "evaluation" / "summary.json",
    }
    content: dict[str, bytes] = {
        "evaluation/per_series.json": _canonical_json_bytes_v2(per_series_payload),
        "evaluation/per_series.csv": _per_series_csv_bytes_v2(per_series_payload),
        "evaluation/paper_consumers.json": _canonical_json_bytes_v2(
            paper_consumers_payload
        ),
    }
    provenance = {
        "protocol_lock_sha256": lock_snapshot.sha256,
        "replay_receipt_sha256": replay_receipt_snapshot.sha256,
        "training_grid_sha256": inventory_snapshot.grid.sha256,
        "fit_sha256": inventory_snapshot.fit_sha256,
    }
    manifest = {
        "schema_version": 2,
        "classification": "corrected post-hoc replay",
        "files": {
            relative: {
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
            for relative, encoded in sorted(content.items())
        },
        "provenance": provenance,
    }
    content["evaluation/artifact_manifest.json"] = _canonical_json_bytes_v2(manifest)
    output_sha256 = {
        relative: hashlib.sha256(encoded).hexdigest()
        for relative, encoded in sorted(content.items())
    }
    summary = {
        "schema_version": 2,
        "status": "complete",
        "classification": "corrected post-hoc replay",
        "successful_process_exit_required_before_consumption": True,
        "fit_count": len(inventory_snapshot.fits),
        "consumer_count": len(PAPER_CONSUMER_CONTRACT_V2),
        "per_series_row_count": len(per_series_payload["rows"]),
        "protocol_lock_sha256": lock_snapshot.sha256,
        "replay_receipt_sha256": replay_receipt_snapshot.sha256,
        "training_grid_sha256": inventory_snapshot.grid.sha256,
        "fit_sha256": inventory_snapshot.fit_sha256,
        "output_sha256": output_sha256,
    }
    content["evaluation/summary.json"] = _canonical_json_bytes_v2(summary)
    ordered = tuple((paths[relative], content[relative]) for relative in paths)
    for path, encoded in ordered:
        try:
            preflight_immutable_bytes_artifact_v2(
                path, encoded, label="primary-v2 evaluation bundle"
            )
        except RuntimeError as exc:
            raise RuntimeError(f"primary-v2 evaluation bundle conflict: {exc}") from exc
    return PreparedEvaluationBundleV2(
        artifacts=ordered,
        summary_path=paths["evaluation/summary.json"],
    )


def require_independent_byte_rebuild_v2(
    builder: Callable[[], tuple[tuple[Path, bytes], ...]],
) -> tuple[tuple[Path, bytes], ...]:
    """Run two independent builders and require identical paths and bytes."""

    first = builder()
    second = builder()
    if first != second:
        raise RuntimeError("primary-v2 independent evaluation byte rebuild differs")
    return first


def _validate_prepared_order_v2(prepared: PreparedEvaluationBundleV2) -> None:
    if type(prepared) is not PreparedEvaluationBundleV2 or not prepared.artifacts:
        raise RuntimeError("primary-v2 prepared evaluation bundle differs")
    paths = [path for path, _ in prepared.artifacts]
    if len(paths) != len(set(paths)) or paths[-1] != prepared.summary_path:
        raise RuntimeError("primary-v2 final summary is not the last bundle artifact")
    if prepared.summary_path.name != "summary.json":
        raise RuntimeError("primary-v2 final summary path differs")


_HELD_WRITER_LOCKS_V2: set[Path] = set()


@contextmanager
def evaluation_writer_lock_v2(result_root: Path) -> Iterator[None]:
    """Hold a nonblocking process lock across receipt, rebuild, and commit."""

    root = Path(result_root).absolute()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("primary-v2 result root is missing or unsafe")
    if root in _HELD_WRITER_LOCKS_V2:
        raise RuntimeError("primary-v2 evaluation single writer is already running")
    descriptor = os.open(root, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "primary-v2 evaluation single writer is already running"
            ) from exc
        _HELD_WRITER_LOCKS_V2.add(root)
        try:
            yield
        finally:
            _HELD_WRITER_LOCKS_V2.remove(root)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def install_prepared_evaluation_bundle_v2(
    prepared: PreparedEvaluationBundleV2,
    *,
    _lock_held: bool = False,
) -> None:
    """Preflight every byte, then immutably install with summary last."""

    _validate_prepared_order_v2(prepared)
    if not _lock_held:
        result_root = prepared.summary_path.parent.parent
        with evaluation_writer_lock_v2(result_root):
            install_prepared_evaluation_bundle_v2(prepared, _lock_held=True)
        return
    for path, content in prepared.artifacts:
        try:
            preflight_immutable_bytes_artifact_v2(
                path, content, label="primary-v2 evaluation output"
            )
        except RuntimeError as exc:
            raise RuntimeError(f"primary-v2 evaluation bundle conflict: {exc}") from exc
    for path, content in prepared.artifacts:
        write_immutable_bytes_artifact_v2(
            path, content, label="primary-v2 evaluation output"
        )


def verify_installed_evaluation_bundle_v2(
    prepared: PreparedEvaluationBundleV2,
) -> None:
    """Check-only path: require every installed byte without writing anything."""

    _validate_prepared_order_v2(prepared)
    for path, expected in prepared.artifacts:
        try:
            observed = read_regular_bytes_artifact_v2(
                path,
                label="primary-v2 check-only evaluation output",
                allow_missing=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"primary-v2 check-only output differs: {path}") from exc
        if observed is None:
            raise RuntimeError(f"missing primary-v2 check-only output: {path}")
        if observed != expected:
            raise RuntimeError(f"primary-v2 output differs in check-only mode: {path}")


def assert_complete_evaluation_inputs_unchanged_v2(
    input_snapshot: protocol_v2.PrimaryInputsSnapshotV2,
    *,
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    replay_receipt_snapshot: ReplayReceiptSnapshotV2,
) -> None:
    """Reject drift in the lock, every split/raw byte, grid, fit, or receipt."""

    protocol_v2.assert_protocol_inputs_unchanged_v2(input_snapshot, lock_snapshot)
    for artifact in input_snapshot.raw_artifacts.values():
        content = read_regular_bytes_artifact_v2(
            artifact.path, label=f"primary-v2 raw evaluation input {artifact.name}"
        )
        assert content is not None
        if (
            len(content) != artifact.size
            or hashlib.sha256(content).hexdigest() != artifact.sha256
        ):
            raise RuntimeError(
                f"primary-v2 raw evaluation input changed: {artifact.name}"
            )
    assert_inventory_snapshots_unchanged_v2(inventory_snapshot)
    receipt = read_regular_bytes_artifact_v2(
        replay_receipt_snapshot.path,
        label="primary-v2 evaluation replay receipt",
    )
    if receipt != replay_receipt_snapshot.content:
        raise RuntimeError("primary-v2 replay receipt changed during evaluation")


def recompute_evaluation_bundle_v2(
    result_root: Path,
    *,
    data_by_split: Mapping[str, Sequence[SeriesData]],
    corpus_values: Mapping[str, object],
    lock_snapshot: protocol_v2.PrimaryProtocolLockSnapshotV2,
    inventory_snapshot: VerifiedEvaluationInventoryV2,
    replay_receipt_snapshot: ReplayReceiptSnapshotV2,
) -> PreparedEvaluationBundleV2:
    """Pure public rebuild of all numerical payloads and their exact bytes."""

    per_series = build_per_series_payload_v2(
        data_by_split,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
        replay_receipt_snapshot=replay_receipt_snapshot,
    )
    consumers = build_paper_consumers_payload_v2(
        per_series,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
        replay_receipt_snapshot=replay_receipt_snapshot,
        data_by_split=data_by_split,
        corpus_values=corpus_values,
    )
    return prepare_evaluation_bundle_v2(
        result_root,
        per_series_payload=per_series,
        paper_consumers_payload=consumers,
        lock_snapshot=lock_snapshot,
        inventory_snapshot=inventory_snapshot,
        replay_receipt_snapshot=replay_receipt_snapshot,
    )


def execute_locked_evaluation_v2(
    result_root: Path,
    *,
    repo_root: Path = ROOT,
    data_dir: Path | None = None,
    check_only: bool = False,
) -> dict[str, object]:
    """Authenticate, disclose, double-rebuild, and immutably commit the replay."""

    result = Path(result_root).absolute()
    root = Path(repo_root).absolute()
    data = Path(data_dir).absolute() if data_dir is not None else root / "data" / "pb"
    if type(check_only) is not bool:
        raise RuntimeError("primary-v2 check-only flag type differs")
    with evaluation_writer_lock_v2(result):
        lock_snapshot = protocol_v2.verify_primary_warsaw_protocol_lock_snapshot_v2(
            result / "protocol_lock.json", root, result, data
        )
        semantics_sha = _tracked_digest_v2(
            lock_snapshot.payload, "artifact/approval_semantics_receipt"
        )
        corpus_semantic_sha = lock_snapshot.payload.get("corpus_semantic_sha256")
        if not _is_sha256(corpus_semantic_sha):
            raise RuntimeError("primary-v2 lock corpus semantic digest differs")
        structural_path = result / "structural_gates.json"
        structural_content = read_regular_bytes_artifact_v2(
            structural_path, label="primary-v2 structural gates"
        )
        assert structural_content is not None
        structural_sha = hashlib.sha256(structural_content).hexdigest()
        if structural_sha != _tracked_digest_v2(
            lock_snapshot.payload, "artifact/structural_gates"
        ):
            raise RuntimeError("primary-v2 structural gate bytes differ from lock")
        structural = protocol_v2.validate_structural_gates_v2(
            structural_path,
            semantics_receipt_sha256=semantics_sha,
            corpus_semantic_sha256=str(corpus_semantic_sha),
            repo_root=root,
        )

        matrix = train_v2.verify_complete_matrix_v2(
            repo_root=root, result_root=result, data_dir=data
        )
        inventory_snapshot = verify_complete_inventory_v2(lock_snapshot, result)
        expected_matrix = {
            "schema_version": 2,
            "status": "complete",
            "protocol_lock_sha256": lock_snapshot.sha256,
            "grid_sha256": inventory_snapshot.grid.sha256,
            "fit_count": 49,
            "fit_sha256": inventory_snapshot.fit_sha256,
        }
        if not protocol_v2.exact_json_equal_v2(matrix, expected_matrix):
            raise RuntimeError("primary-v2 matrix verifier and evaluator differ")

        existing_receipt = load_evaluation_replay_receipt_v2(
            result,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory_snapshot,
            structural_gates_sha256=structural_sha,
            semantics_receipt_sha256=semantics_sha,
            allow_missing=True,
        )
        if check_only and existing_receipt is None:
            assert_inventory_snapshots_unchanged_v2(inventory_snapshot)
            return {
                "schema_version": 2,
                "status": "ready_for_corrected_posthoc_replay",
                "classification": "corrected post-hoc replay",
                "heldout_opened": False,
                "fit_count": 49,
                "protocol_lock_sha256": lock_snapshot.sha256,
                "training_grid_sha256": inventory_snapshot.grid.sha256,
                "fit_sha256": inventory_snapshot.fit_sha256,
            }
        receipt = (
            existing_receipt
            if existing_receipt is not None
            else write_evaluation_replay_receipt_v2(
                result,
                lock_snapshot=lock_snapshot,
                inventory_snapshot=inventory_snapshot,
                structural_gates_sha256=structural_sha,
                semantics_receipt_sha256=semantics_sha,
            )
        )
        assert receipt is not None

        # Full raw access is a capability granted only by the authenticated
        # on-disk opening receipt.  The pre-lock training loader is never used
        # for evaluation.
        input_snapshot = protocol_v2.load_evaluation_inputs_snapshot_v2(
            lock_snapshot,
            result,
            data,
            sorted(protocol_v2.SPLIT_SHA256_V2),
            replay_receipt_snapshot=receipt,
        )
        postreceipt_structural = protocol_v2.build_postreceipt_structural_replay_v2(
            input_snapshot,
            lock_snapshot,
        )
        data_by_split = load_evaluation_series_data_v2(
            input_snapshot,
            receipt,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory_snapshot,
        )
        corpus_values = {
            **{
                key: structural[key]
                for key in (
                    "n_series",
                    "n_elections",
                    "manifest_file_bindings",
                    "approval_ballots_authenticated",
                    "duplicate_tokens_remaining",
                    "split_artifacts_authenticated",
                    "raw_election_files_opened",
                    "election_parser_calls",
                    "outcome_solver_calls",
                    "heldout_outcomes_scored",
                )
                if key in structural
            },
            **dict(postreceipt_structural),
        }

        def rebuild() -> tuple[tuple[Path, bytes], ...]:
            return recompute_evaluation_bundle_v2(
                result,
                data_by_split=data_by_split,
                corpus_values=corpus_values,
                lock_snapshot=lock_snapshot,
                inventory_snapshot=inventory_snapshot,
                replay_receipt_snapshot=receipt,
            ).artifacts

        artifacts = require_independent_byte_rebuild_v2(rebuild)
        prepared = PreparedEvaluationBundleV2(
            artifacts=artifacts,
            summary_path=result / "evaluation" / "summary.json",
        )
        assert_complete_evaluation_inputs_unchanged_v2(
            input_snapshot,
            lock_snapshot=lock_snapshot,
            inventory_snapshot=inventory_snapshot,
            replay_receipt_snapshot=receipt,
        )
        if read_regular_bytes_artifact_v2(
            structural_path, label="primary-v2 structural gates"
        ) != structural_content:
            raise RuntimeError("primary-v2 structural gates changed during evaluation")
        summary = protocol_v2.load_json_object_bytes_strict_v2(
            artifacts[-1][1], label="primary-v2 evaluation summary"
        )
        if check_only:
            verify_installed_evaluation_bundle_v2(prepared)
            assert_complete_evaluation_inputs_unchanged_v2(
                input_snapshot,
                lock_snapshot=lock_snapshot,
                inventory_snapshot=inventory_snapshot,
                replay_receipt_snapshot=receipt,
            )
        else:
            install_prepared_evaluation_bundle_v2(prepared, _lock_held=True)
        return dict(summary)


def main(argv: Sequence[str] | None = None) -> None:
    """Run or non-mutatingly verify the corrected post-hoc replay."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the primary-Warsaw corrected post-hoc replay. Held-out outcomes "
            "were already known; this is not a fresh holdout."
        )
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="authenticated primary corpus"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify without creating a receipt or evaluation output",
    )
    args = parser.parse_args(argv)
    summary = execute_locked_evaluation_v2(
        args.result_root,
        data_dir=args.data_dir,
        check_only=args.check_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
