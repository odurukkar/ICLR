"""Evaluation entry point for the append-only multicity protocol v2."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import iclr_multicity_protocol_v2 as protocol_v2
from iclr_approval_semantics_v2 import (
    SEMANTICS_PROFILE,
    load_series_authenticated_v2,
    validate_manifest_receipt_identities_v2,
    validate_multicity_semantics_receipt,
)
from iclr_corpus import SeriesRef, Split
from iclr_env import (
    EnvConfig,
    SelectorPolicy,
    endowment_selector,
    res_policy,
    uniform_policy,
)
from iclr_multicity_evaluate import (
    PRIMARY_SEED,
    _evidence_payload,
    _evaluated_payload,
    evaluate_gold_gate,
    evaluate_policy_on_split,
    holm_adjust,
    paired_city_summary,
)
from iclr_multicity_protocol import (
    PABULIB_COMMIT,
    _sha256,
    load_canonical_index_from_manifest,
    load_frozen_split,
)
from iclr_multicity_protocol_v2 import (
    FIT_SOURCE_FILES_V2,
    LOCK_PROFILE_V2,
    SPLIT_NAMES,
    assert_protocol_inputs_unchanged_v2,
    canonical_fit_path_v2,
    canonical_result_output_path_v2,
    canonical_result_root_v2,
    exact_json_equal_v2,
    load_protocol_inputs_snapshot_v2,
    preflight_immutable_bytes_artifact_v2,
    read_regular_bytes_artifact_v2,
    validate_structural_gates_v2,
    verify_multicity_protocol_lock_snapshot_v2,
    write_immutable_bytes_artifact_v2,
    locked_fit_inventory_v2,
)
from iclr_multicity_train import build_multicity_arm
from iclr_outcome import (
    cost_effective_weights,
    greedy_equivalent_weights,
    score_selector,
)
from parse_pb import PBInstance


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results" / "iclr_multicity_v2"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerifiedFitSnapshotV2:
    """One fit parsed and authenticated from a single immutable byte snapshot."""

    split: str
    arm: str
    seed: int
    path: Path
    label: str
    sha256: str
    payload: Mapping[str, object]

    @property
    def spec(self) -> dict[str, object]:
        return {"split": self.split, "arm": self.arm, "seed": self.seed}


@dataclass(frozen=True)
class PreparedEvaluationOutputsV2:
    """Serialized, conflict-preflighted evaluation outputs ready to install."""

    artifacts: tuple[tuple[Path, bytes], ...]

def assert_fit_snapshots_unchanged_v2(
    snapshots: Sequence[VerifiedFitSnapshotV2],
) -> None:
    """Reject persistent drift of any authenticated fit before output writes."""

    for snapshot in snapshots:
        content = read_regular_bytes_artifact_v2(
            snapshot.path,
            label=f"protocol-v2 fit {snapshot.label}",
        )
        assert content is not None
        if hashlib.sha256(content).hexdigest() != snapshot.sha256:
            raise RuntimeError(f"protocol-v2 fit changed during use: {snapshot.label}")


def assert_protocol_lock_snapshot_unchanged_v2(
    initial_snapshot: protocol_v2.ProtocolLockSnapshotV2,
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> None:
    """Revalidate the complete locked closure immediately before output."""

    final_snapshot = verify_multicity_protocol_lock_snapshot_v2(
        canonical_result_output_path_v2(
            result_root,
            Path("protocol_lock.json"),
        ),
        repo_root,
        result_root,
        data_dir,
    )
    if (
        final_snapshot.path != initial_snapshot.path
        or final_snapshot.content != initial_snapshot.content
        or final_snapshot.sha256 != initial_snapshot.sha256
        or not exact_json_equal_v2(final_snapshot.payload, initial_snapshot.payload)
    ):
        raise RuntimeError("protocol-v2 locked closure changed during evaluation")


def _fit_selector_from_snapshot(
    snapshot: VerifiedFitSnapshotV2,
    cfg: EnvConfig,
) -> SelectorPolicy:
    arm = build_multicity_arm(snapshot.arm)
    result = snapshot.payload["result"]
    if not isinstance(result, Mapping):
        raise RuntimeError(f"fit snapshot has malformed result: {snapshot.path}")
    weights = np.asarray(result["best_weights"], dtype=float)
    return arm.selector(weights, cfg)


def required_fit_specs_v2() -> list[dict[str, object]]:
    """Return the complete protocol-v2 fit matrix."""

    return locked_fit_inventory_v2()


def required_fit_inventory_v2(
    result_root: Path,
) -> list[tuple[dict[str, object], Path]]:
    """Pair every v2 fit specification with its deterministic path."""

    root = Path(result_root)
    return [
        (
            spec,
            root
            / "fits"
            / str(spec["split"])
            / str(spec["arm"])
            / f"seed-{spec['seed']}.json",
        )
        for spec in required_fit_specs_v2()
    ]


def verify_lock_fit_inventory_v2(lock_payload: Mapping[str, object]) -> None:
    """Require the lock and evaluator to name the same v2 fit matrix."""

    def canonical(rows) -> list[tuple[str, str, int]]:
        canonical_rows = []
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or type(row.get("split")) is not str
                or type(row.get("arm")) is not str
                or type(row.get("seed")) is not int
            ):
                raise RuntimeError("v2 fit inventory has invalid coordinate types")
            canonical_rows.append((row["split"], row["arm"], row["seed"]))
        return sorted(canonical_rows)

    locked = canonical(lock_payload.get("fit_inventory", []))
    expected = canonical(required_fit_specs_v2())
    if locked != expected:
        raise RuntimeError(
            "protocol-v2 lock fit inventory differs from evaluator: "
            f"locked={locked}, evaluator={expected}"
        )


def load_evaluation_instances_v2(
    index: Mapping[str, SeriesRef],
    semantics_receipt: Mapping[str, object],
) -> dict[str, dict[int, PBInstance]]:
    """Load complete warm-up and scored histories under v2 set semantics."""

    return {
        key: load_series_authenticated_v2(ref, semantics_receipt)
        for key, ref in sorted(index.items())
    }


def _compute_verified_evaluation_v2(
    index: Mapping[str, SeriesRef],
    splits: Mapping[str, Split],
    fit_snapshots: Sequence[VerifiedFitSnapshotV2],
    containment_pass: bool,
    evidence_gates: Mapping[str, object],
    semantics_receipt: Mapping[str, object],
    *,
    provenance: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Compute from verified in-memory inputs without writing official artifacts."""

    expected_specs = sorted(
        (str(row["split"]), str(row["arm"]), int(row["seed"]))
        for row in required_fit_specs_v2()
    )
    observed_specs = sorted(
        (snapshot.split, snapshot.arm, snapshot.seed)
        for snapshot in fit_snapshots
    )
    if observed_specs != expected_specs:
        raise RuntimeError("verified fit snapshots differ from the full v2 matrix")
    if set(splits) != set(SPLIT_NAMES):
        raise RuntimeError("evaluation splits differ from the full v2 inventory")
    cfg = EnvConfig()
    instances = load_evaluation_instances_v2(index, semantics_receipt)
    inventory_by_split: dict[str, list[VerifiedFitSnapshotV2]] = defaultdict(list)
    for snapshot in fit_snapshots:
        inventory_by_split[snapshot.split].append(snapshot)

    evaluated = {}
    fixed = {
        "mes": endowment_selector(uniform_policy, cfg),
        "res-1.0": endowment_selector(res_policy(1.0), cfg),
        "greedy-count": score_selector(greedy_equivalent_weights()),
        "greedy-cost": score_selector(cost_effective_weights()),
    }
    for split_name, split in sorted(splits.items()):
        policies: dict[str, SelectorPolicy] = dict(fixed)
        for snapshot in inventory_by_split[split_name]:
            name = f"{snapshot.arm}/seed-{snapshot.seed}"
            policies[name] = _fit_selector_from_snapshot(snapshot, cfg)
        evaluated[split_name] = {}
        for name, selector in sorted(policies.items()):
            logger.info("evaluating %s on %s under protocol v2", name, split_name)
            evaluated[split_name][name] = evaluate_policy_on_split(
                split,
                index,
                instances,
                selector,
                cfg,
            )

    summaries = {}
    for split_name, policies in evaluated.items():
        baseline = [row.episode for row in policies["mes"]]
        summaries[split_name] = {}
        for name, rows in policies.items():
            if name == "mes":
                continue
            summaries[split_name][name] = paired_city_summary(
                [row.episode for row in rows],
                baseline,
                include_inference=(split_name == "temporal_2022"),
            )

    primary_rows = summaries["temporal_2022"][
        f"priority/seed-{PRIMARY_SEED}"
    ]
    primary_pvalues = {
        city: float(row["p_value"])
        for city, row in primary_rows.items()
        if city != "CITY_MACRO" and "p_value" in row
    }
    for city, value in holm_adjust(primary_pvalues).items():
        primary_rows[city]["holm_p_value"] = value

    evaluation_provenance = dict(provenance)
    evidence = _evidence_payload(evaluated, summaries, containment_pass)
    evidence["provenance"] = evaluation_provenance
    decision = evaluate_gold_gate(evidence, evidence_gates)
    decision["provenance"] = evaluation_provenance
    per_series_payload = {
        split_name: {
            policy: [_evaluated_payload(row) for row in rows]
            for policy, rows in sorted(policies.items())
        }
        for split_name, policies in sorted(evaluated.items())
    }
    return decision, evidence, summaries, per_series_payload, evaluated


def _per_series_csv_text_v2(
    evaluated: Mapping[str, Mapping[str, Sequence[object]]],
) -> str:
    """Serialize per-series evidence in the exact v1 column format."""

    fields = (
        "split",
        "policy",
        "series",
        "city",
        "scored_years",
        "worst_csd",
        "worst_cohort",
        "mean_csd",
        "welfare",
        "cost_welfare",
        "exclusion",
        "outcome_sha256",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for split_name, policies in sorted(evaluated.items()):
        for policy, rows in sorted(policies.items()):
            for row in rows:
                episode = row.episode
                encoded = json.dumps(
                    row.scored_outcomes,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                parts = episode.series.split("/")
                city = "/".join(parts[:2]) if len(parts) >= 2 else episode.series
                writer.writerow(
                    {
                        "split": split_name,
                        "policy": policy,
                        "series": episode.series,
                        "city": city,
                        "scored_years": " ".join(map(str, episode.scored_years)),
                        "worst_csd": episode.worst_csd,
                        "worst_cohort": episode.worst_cohort,
                        "mean_csd": episode.mean_csd,
                        "welfare": episode.welfare,
                        "cost_welfare": episode.cost_welfare,
                        "exclusion": episode.exclusion,
                        "outcome_sha256": hashlib.sha256(encoded).hexdigest(),
                    }
                )
    return handle.getvalue()


def prepare_evaluation_outputs_v2(
    result_root: Path,
    *,
    decision: Mapping[str, object],
    evidence: Mapping[str, object],
    summaries: Mapping[str, object],
    per_series_payload: Mapping[str, object],
    evaluated: Mapping[str, Mapping[str, Sequence[object]]],
) -> PreparedEvaluationOutputsV2:
    """Serialize and preflight every official v2 evaluation output."""

    relative_paths = {
        "per_series_json": Path("evaluation/per_series.json"),
        "per_series_csv": Path("evaluation/per_series.csv"),
        "summary": Path("evaluation/summary.json"),
        "evidence": Path("evaluation/evidence_payload.json"),
        "decision": Path("evidence_decision.json"),
    }
    paths = {
        name: canonical_result_output_path_v2(result_root, relative)
        for name, relative in relative_paths.items()
    }

    def json_bytes(payload: Mapping[str, object]) -> bytes:
        return (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")

    serialized = {
        "per_series_json": json_bytes(per_series_payload),
        "per_series_csv": _per_series_csv_text_v2(evaluated).encode("utf-8"),
        "summary": json_bytes(summaries),
        "evidence": json_bytes(evidence),
        "decision": json_bytes(decision),
    }
    for name, path in paths.items():
        preflight_immutable_bytes_artifact_v2(
            path,
            serialized[name],
            label="protocol-v2 evaluation output",
        )
    return PreparedEvaluationOutputsV2(
        tuple((path, serialized[name]) for name, path in paths.items())
    )


def install_prepared_evaluation_outputs_v2(
    prepared: PreparedEvaluationOutputsV2,
) -> None:
    """Install one already serialized and conflict-preflighted output bundle."""

    for path, content in prepared.artifacts:
        write_immutable_bytes_artifact_v2(
            path,
            content,
            label="protocol-v2 evaluation output",
        )


def write_evaluation_outputs_v2(
    result_root: Path,
    *,
    decision: Mapping[str, object],
    evidence: Mapping[str, object],
    summaries: Mapping[str, object],
    per_series_payload: Mapping[str, object],
    evaluated: Mapping[str, Mapping[str, Sequence[object]]],
) -> None:
    """Preflight and immutably install every official v2 evaluation output."""

    prepared = prepare_evaluation_outputs_v2(
        result_root,
        decision=decision,
        evidence=evidence,
        summaries=summaries,
        per_series_payload=per_series_payload,
        evaluated=evaluated,
    )
    install_prepared_evaluation_outputs_v2(prepared)


def execute_locked_evaluation_v2(
    result_root: Path,
    *,
    repo_root: Path = ROOT,
    data_dir: Path | None = None,
) -> dict[str, object]:
    """Verify v2 evidence, create the one-way receipt, then load held-out data."""

    root = Path(repo_root).resolve()
    result = canonical_result_root_v2(result_root)
    data = protocol_v2._require_nonsymlink_components_v2(
        Path(data_dir)
        if data_dir is not None
        else root / "data" / "pb_multicity",
        "protocol-v2 evaluation data root",
        allow_missing=False,
    )
    lock_path = result / "protocol_lock.json"
    structural_path = result / "structural_gates.json"
    semantics_path = result / "approval_semantics_receipt.json"
    heldout_receipt_path = result / "heldout_opened.json"
    inventory = required_fit_inventory_v2(result)

    lock_snapshot = verify_multicity_protocol_lock_snapshot_v2(
        lock_path,
        root,
        result,
        data,
    )
    lock_payload = lock_snapshot.payload
    verify_lock_fit_inventory_v2(lock_payload)

    semantics_bytes = read_regular_bytes_artifact_v2(
        semantics_path,
        label="v2 semantics receipt",
    )
    assert semantics_bytes is not None
    semantics_sha256 = hashlib.sha256(semantics_bytes).hexdigest()
    semantics_payload = json.loads(semantics_bytes.decode("utf-8"))
    if not isinstance(semantics_payload, dict):
        raise RuntimeError("v2 semantics receipt must be a JSON object")
    validate_multicity_semantics_receipt(semantics_payload)
    if semantics_payload.get("semantics_profile") != SEMANTICS_PROFILE:
        raise RuntimeError("v2 semantics receipt profile differs")
    corpus_semantic_sha256 = semantics_payload.get("corpus_semantic_sha256")
    if (
        not isinstance(corpus_semantic_sha256, str)
        or len(corpus_semantic_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in corpus_semantic_sha256
        )
    ):
        raise RuntimeError("v2 semantics receipt corpus digest differs")
    tracked = lock_payload.get("tracked_files")
    semantics_row = (
        tracked.get("artifact/approval_semantics_receipt")
        if isinstance(tracked, Mapping)
        else None
    )
    if (
        not isinstance(semantics_row, Mapping)
        or semantics_row.get("sha256") != semantics_sha256
    ):
        raise RuntimeError("v2 semantics receipt differs from protocol lock")

    structural_bytes = read_regular_bytes_artifact_v2(
        structural_path,
        label="v2 structural gate",
    )
    assert structural_bytes is not None
    structural_sha256 = hashlib.sha256(structural_bytes).hexdigest()
    structural = validate_structural_gates_v2(
        structural_path,
        semantics_receipt_sha256=semantics_sha256,
        corpus_semantic_sha256=corpus_semantic_sha256,
        repo_root=root,
    )
    if read_regular_bytes_artifact_v2(
        structural_path,
        label="v2 structural gate",
    ) != structural_bytes:
        raise RuntimeError("v2 structural gate changed during evaluation verification")
    lock_sha256 = lock_snapshot.sha256
    fit_snapshots = verify_fit_inventory_v2(
        inventory,
        lock_payload,
        lock_sha256,
        corpus_semantic_sha256=corpus_semantic_sha256,
        result_root=result,
        repo_root=root,
    )
    authenticated_fit_sha256 = {
        snapshot.label: snapshot.sha256 for snapshot in fit_snapshots
    }
    opening = protocol_v2.open_heldout_once_v2(
        lock_path,
        heldout_receipt_path,
        inventory,
        structural_path,
        semantics_path,
        root,
        authenticated_fit_sha256=authenticated_fit_sha256,
        data_dir=data,
        lock_snapshot=lock_snapshot,
    )
    expected_opening = {
        "lock_sha256": lock_sha256,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "fit_sha256": authenticated_fit_sha256,
        "structural_gates_sha256": structural_sha256,
    }
    if any(opening.get(key) != value for key, value in expected_opening.items()):
        raise RuntimeError("v2 held-out receipt lineage differs")
    if read_regular_bytes_artifact_v2(
        semantics_path,
        label="v2 semantics receipt",
    ) != semantics_bytes:
        raise RuntimeError("v2 semantics receipt changed before held-out loading")

    input_snapshot = load_protocol_inputs_snapshot_v2(
        lock_payload,
        result,
        data,
        SPLIT_NAMES,
    )
    index = input_snapshot.index
    splits = input_snapshot.splits
    validate_manifest_receipt_identities_v2(index, semantics_payload)
    opening_bytes = read_regular_bytes_artifact_v2(
        heldout_receipt_path,
        label="v2 heldout-opening receipt",
    )
    assert opening_bytes is not None
    if not exact_json_equal_v2(
        json.loads(opening_bytes.decode("utf-8")),
        opening,
    ):
        raise RuntimeError("v2 held-out receipt changed before evaluation")
    heldout_receipt_sha256 = hashlib.sha256(opening_bytes).hexdigest()
    provenance = {
        "protocol_lock_sha256": lock_sha256,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "heldout_receipt_sha256": heldout_receipt_sha256,
    }
    decision, evidence, summaries, per_series_payload, evaluated = (
        _compute_verified_evaluation_v2(
            index,
            splits,
            fit_snapshots,
            structural.get("status") == "pass",
            lock_payload["evidence_gates"],
            semantics_payload,
            provenance=provenance,
        )
    )
    prepared_outputs = prepare_evaluation_outputs_v2(
        result,
        decision=decision,
        evidence=evidence,
        summaries=summaries,
        per_series_payload=per_series_payload,
        evaluated=evaluated,
    )
    assert_protocol_inputs_unchanged_v2(input_snapshot)
    if (
        read_regular_bytes_artifact_v2(
            lock_path,
            label="v2 protocol lock",
        )
        != lock_snapshot.content
        or read_regular_bytes_artifact_v2(
            semantics_path,
            label="v2 semantics receipt",
        )
        != semantics_bytes
        or read_regular_bytes_artifact_v2(
            structural_path,
            label="v2 structural gate",
        )
        != structural_bytes
        or read_regular_bytes_artifact_v2(
            heldout_receipt_path,
            label="v2 heldout-opening receipt",
        )
        != opening_bytes
    ):
        raise RuntimeError("locked v2 inputs changed during evaluation")
    assert_protocol_lock_snapshot_unchanged_v2(
        lock_snapshot,
        repo_root=root,
        result_root=result,
        data_dir=data,
    )
    if read_regular_bytes_artifact_v2(
        heldout_receipt_path,
        label="v2 heldout-opening receipt",
    ) != opening_bytes:
        raise RuntimeError("v2 held-out receipt changed before evaluation output")
    assert_fit_snapshots_unchanged_v2(fit_snapshots)
    install_prepared_evaluation_outputs_v2(prepared_outputs)
    return decision


def verify_fit_inventory_v2(
    inventory: Sequence[tuple[Mapping[str, object], Path]],
    lock_payload: Mapping[str, object],
    lock_sha256: str,
    *,
    corpus_semantic_sha256: str,
    result_root: Path,
    repo_root: Path,
) -> tuple[VerifiedFitSnapshotV2, ...]:
    """Authenticate every fit from one byte snapshot before held-out opening."""

    if (
        type(lock_payload.get("schema_version")) is not int
        or lock_payload.get("schema_version") != 2
        or lock_payload.get("lock_profile") != LOCK_PROFILE_V2
    ):
        raise RuntimeError("fit verification requires a v2 protocol lock")
    tracked = lock_payload.get("tracked_files")
    if not isinstance(tracked, Mapping):
        raise RuntimeError("v2 protocol lock tracked files are malformed")
    protocol = lock_payload.get("protocol")
    if not exact_json_equal_v2(protocol, protocol_v2.protocol_config_v2()):
        raise RuntimeError("fit verification requires the exact v2 protocol")
    locked_inventory = lock_payload.get("fit_inventory")
    if not isinstance(locked_inventory, list):
        raise RuntimeError("v2 protocol lock fit inventory is malformed")

    def canonical(rows) -> list[tuple[str, str, int]]:
        canonical_rows = []
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or type(row.get("split")) is not str
                or type(row.get("arm")) is not str
                or type(row.get("seed")) is not int
            ):
                raise RuntimeError("v2 fit inventory has invalid coordinate types")
            canonical_rows.append((row["split"], row["arm"], row["seed"]))
        return sorted(canonical_rows)

    supplied_specs = [spec for spec, _ in inventory]
    if canonical(supplied_specs) != canonical(locked_inventory):
        raise RuntimeError("supplied fit inventory differs from protocol-v2 lock")
    if len(canonical(supplied_specs)) != len(set(canonical(supplied_specs))):
        raise RuntimeError("supplied fit inventory contains duplicate specifications")
    expected_fixed = {
        "generations": protocol["generations"],
        "popsize": protocol["popsize"],
        "sigma0": protocol["sigma0"],
        "bound": protocol["coefficient_bound"],
        "welfare_penalty": protocol["objective"]["welfare_penalty"],
        "welfare_floor": protocol["objective"]["welfare_floor_ratio"],
    }
    feature_counts = {"priority": 5, "endowment": 6, "outcome": 5}
    expected_sources = {
        name: tracked.get(f"source/{name}", {}).get("sha256")
        for name in FIT_SOURCE_FILES_V2
    }
    if any(value is None for value in expected_sources.values()):
        raise RuntimeError("v2 protocol lock is missing fit source hashes")

    resolved_result_root = Path(result_root).resolve()
    root = Path(repo_root).resolve()
    snapshots = []
    for spec, raw_path in inventory:
        path = canonical_fit_path_v2(resolved_result_root, spec, raw_path)
        if not path.is_file():
            raise RuntimeError(f"missing required v2 fit: {path}")
        try:
            label = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"fit lies outside the v2 repository: {path}") from exc
        fit_bytes = read_regular_bytes_artifact_v2(
            path,
            label="protocol-v2 fit",
        )
        assert fit_bytes is not None
        fit_sha256 = hashlib.sha256(fit_bytes).hexdigest()
        payload = json.loads(fit_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"fit {path} must be a JSON object")
        if type(payload.get("schema_version")) is not int or payload.get(
            "schema_version"
        ) != 2:
            raise RuntimeError(f"fit {path} has invalid schema_version")
        if payload.get("training_only") is not True:
            raise RuntimeError(f"fit is not training-only: {path}")
        if payload.get("execution_mode") != "serial":
            raise RuntimeError(f"fit has invalid execution_mode: {path}")

        config = payload.get("config")
        if not isinstance(config, Mapping):
            raise RuntimeError(f"fit {path} has malformed config")
        if type(config.get("seed")) is not int:
            raise RuntimeError(f"fit {path} has invalid seed type")
        for key in ("split", "arm", "seed"):
            if config.get(key) != spec.get(key):
                raise RuntimeError(
                    f"fit {path} has {key}={config.get(key)!r}, "
                    f"expected {spec.get(key)!r}"
                )
        for key, expected in expected_fixed.items():
            if (
                type(config.get(key)) is not type(expected)
                or config.get(key) != expected
            ):
                raise RuntimeError(
                    f"fit {path} has {key}={config.get(key)!r}, "
                    f"expected {expected!r}"
                )

        arm_name = str(spec.get("arm"))
        if arm_name not in feature_counts:
            raise RuntimeError(f"fit {path} names unknown arm {arm_name!r}")
        fit_result = payload.get("result")
        if not isinstance(fit_result, Mapping):
            raise RuntimeError(f"fit {path} has malformed result")
        raw_weights = fit_result.get("best_weights", [])
        if not isinstance(raw_weights, list) or any(
            type(value) not in {int, float} for value in raw_weights
        ):
            raise RuntimeError(f"fit {path} has invalid best_weights")
        weights = np.asarray(raw_weights, dtype=float)
        if (
            weights.shape != (feature_counts[arm_name],)
            or not np.all(np.isfinite(weights))
        ):
            raise RuntimeError(f"fit {path} has invalid best_weights")
        arm = build_multicity_arm(arm_name)
        expected_arm = {
            "name": arm.name,
            "feature_names": list(arm.feature_names),
            "init_name": arm.init_name,
            "initial": arm.initial.tolist(),
        }
        if not exact_json_equal_v2(payload.get("arm"), expected_arm):
            raise RuntimeError(f"fit {path} has invalid arm metadata")

        expected_provenance = {
            "protocol_lock_sha256": lock_sha256,
            "pabulib_commit": PABULIB_COMMIT,
            "corpus_manifest_sha256": tracked.get(
                "artifact/corpus_manifest", {}
            ).get("sha256"),
            "split_sha256": tracked.get(
                f"split/{spec.get('split')}", {}
            ).get("sha256"),
            "source_sha256": expected_sources,
            "semantics_profile": SEMANTICS_PROFILE,
            "semantics_receipt_sha256": tracked.get(
                "artifact/approval_semantics_receipt", {}
            ).get("sha256"),
            "corpus_semantic_sha256": corpus_semantic_sha256,
        }
        if payload.get("provenance") != expected_provenance:
            raise RuntimeError(
                f"fit {path} provenance differs from protocol-v2 lock"
            )
        snapshots.append(
            VerifiedFitSnapshotV2(
                split=str(spec["split"]),
                arm=str(spec["arm"]),
                seed=int(spec["seed"]),
                path=path,
                label=label,
                sha256=fit_sha256,
                payload=payload,
            )
        )
    return tuple(snapshots)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the locked protocol-v2 evaluation after its one-way receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    decision = execute_locked_evaluation_v2(args.result_root)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
