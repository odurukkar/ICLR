"""Training primitives for the append-only primary-Warsaw v2 replay.

The production entry point is deliberately narrower than the historical
scripts: callers request one coordinate from the frozen inventory, and the
loader exposes only that split's training years.  Held-out evaluation belongs
to :mod:`iclr_primary_warsaw_evaluate_v2` and is never serialized in a fit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Callable, Mapping, Sequence

import numpy as np

from cohorts import AGE_BRACKETS, age_bracket
from iclr_approval_semantics_v2 import (
    SEMANTICS_PROFILE,
    load_series_authenticated_v2,
)
from iclr_corpus import SeriesRef, Split
from iclr_cmaes import CMAESConfig, minimize
from iclr_env import (
    EnvConfig,
    RolloutState,
    aggregate,
    endowment_selector,
    rollout_selector,
    uniform_policy,
)
from iclr_multicity_protocol_v2 import (
    read_regular_bytes_artifact_v2,
    write_immutable_bytes_artifact_v2,
)
from iclr_outcome import (
    PROJECT_FEATURES,
    cost_effective_weights,
    score_selector,
)
from iclr_policy import FEATURE_NAMES, linear_policy, res_equivalent_weights
from iclr_primary_warsaw_protocol_v2 import (
    LOCK_PROFILE_V2,
    PrimaryInputsSnapshotV2,
    PrimaryProtocolLockSnapshotV2,
    RESULT_ROOT_V2,
    ROOT,
    STATIC_AGE_GRID_VALUES,
    assert_protocol_inputs_unchanged_v2,
    canonical_fit_path_v2,
    canonical_grid_path_v2,
    exact_json_equal_v2,
    load_json_object_bytes_strict_v2,
    load_protocol_inputs_snapshot_v2,
    locked_fit_inventory_v2,
    validate_protocol_inputs_snapshot_v2,
    validate_locked_fit_inventory_v2,
    verify_primary_warsaw_protocol_lock_snapshot_v2,
)
from iclr_train import SeriesData, TrainConfig, _evaluate, _objective, build_arm
from parse_pb import PBInstance


GRID_ARTIFACT_LABEL = "artifact/static_senior_alpha_grid"
GRID_FIELDS = {
    "schema_version",
    "semantics_profile",
    "split",
    "training_only",
    "alphas",
    "train_worst_csd",
    "selected_alpha",
    "selection_rule",
    "provenance",
}
GRID_SELECTION_RULE = (
    "minimum training worst-cohort CSD; ties choose smaller alpha"
)
STATIC_AGE_FEATURE_NAMES = tuple(label for _, _, label in AGE_BRACKETS)
DEFAULT_MATRIX_WORKERS_V2 = 4
MAX_MATRIX_WORKERS_V2 = 5
EVALUATION_REPLAY_STARTED_NAME_V2 = "evaluation_replay_started.json"


@dataclass(frozen=True)
class JsonSnapshotV2:
    """One JSON value bound to the exact bytes read from its path."""

    path: Path
    content: bytes
    sha256: str
    payload: dict[str, object]


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_finite_json(value: object, *, label: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{label} must contain only finite numbers")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _require_finite_json(child, label=f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise RuntimeError(f"{label} contains a non-string JSON key")
            _require_finite_json(child, label=f"{label}.{key}")
        return
    raise RuntimeError(f"{label} contains a non-JSON value")


def load_training_series_data_v2(
    split: Split,
    index: Mapping[str, SeriesRef],
    semantics_receipt: Mapping[str, object],
) -> list[SeriesData]:
    """Load exactly the years in ``split.train`` under set-valued semantics."""

    rows: list[SeriesData] = []
    seen: set[str] = set()
    for key, years in split.train:
        if key in seen:
            raise RuntimeError(f"duplicate training series in split: {key}")
        seen.add(key)
        if key not in index:
            raise RuntimeError(f"unknown training series in split: {key}")
        if not years:
            continue
        parent_ref = index[key]
        requested = tuple(years)
        path_by_year = dict(zip(parent_ref.years, parent_ref.paths))
        if any(year not in path_by_year for year in requested):
            raise RuntimeError(f"training split names an unavailable year for {key}")
        ref = SeriesRef(
            key=parent_ref.key,
            years=requested,
            paths=tuple(path_by_year[year] for year in requested),
        )
        instances = load_series_authenticated_v2(
            ref,
            semantics_receipt,
            years=requested,
        )
        if tuple(instances) != requested:
            raise RuntimeError(
                f"training years differ for {key}: {tuple(instances)} != {requested}"
            )
        rows.append(
            SeriesData(
                ref=ref,
                train_years=requested,
                test_years=(),
                train_only=dict(instances),
                all_years=dict(instances),
            )
        )
    return rows


def _canonical_inventory_match(spec: Mapping[str, object]) -> dict[str, object]:
    inventory = locked_fit_inventory_v2()
    validate_locked_fit_inventory_v2(inventory)
    matches = [
        row for row in inventory if exact_json_equal_v2(row, spec)
    ]
    if len(matches) != 1:
        raise RuntimeError("fit authorization is outside the locked inventory")
    return matches[0]


def authorize_fit_v2(
    spec: Mapping[str, object],
    *,
    smoke: bool,
    result_root: Path,
    repo_root: Path,
    data_dir: Path,
):
    """Authorize one exact inventory coordinate against the complete lock."""

    _canonical_inventory_match(spec)
    if smoke:
        return None
    lock_path = Path(result_root) / "protocol_lock.json"
    snapshot = verify_primary_warsaw_protocol_lock_snapshot_v2(
        lock_path,
        Path(repo_root),
        Path(result_root),
        Path(data_dir),
    )
    payload = snapshot.payload
    if (
        type(payload) is not dict
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("lock_profile") != LOCK_PROFILE_V2
    ):
        raise RuntimeError("primary-v2 lock profile or schema differs")
    locked_inventory = payload.get("fit_inventory")
    if type(locked_inventory) is not list:
        raise RuntimeError("primary-v2 locked fit inventory is malformed")
    validate_locked_fit_inventory_v2(locked_inventory)
    if not any(exact_json_equal_v2(row, spec) for row in locked_inventory):
        raise RuntimeError("fit authorization is absent from locked inventory")
    return snapshot


def _locked_inventory_match_v2(
    spec: Mapping[str, object],
    lock_snapshot,
) -> dict[str, object]:
    """Return one exact coordinate authorized by the supplied lock snapshot."""

    payload = getattr(lock_snapshot, "payload", None)
    if type(payload) is not dict:
        raise RuntimeError("primary-v2 authorized lock payload is malformed")
    inventory = payload.get("fit_inventory")
    try:
        validate_locked_fit_inventory_v2(inventory)
    except RuntimeError as exc:
        raise RuntimeError("primary-v2 locked fit inventory differs") from exc
    matches = [row for row in inventory if exact_json_equal_v2(row, spec)]
    if len(matches) != 1:
        raise RuntimeError("primary-v2 authorized spec is absent from lock inventory")
    return matches[0]


def _tracked_digest_v2(
    tracked: Mapping[str, object],
    label: str,
) -> str:
    row = tracked.get(label)
    digest = row.get("sha256") if isinstance(row, Mapping) else None
    if not _is_sha256(digest):
        raise RuntimeError(f"primary-v2 lock tracked digest is missing: {label}")
    return digest


def _locked_provenance_v2(
    lock_snapshot,
    *,
    split_name: str,
) -> dict[str, object]:
    """Derive all result provenance from the authenticated lock, never callers."""

    payload = getattr(lock_snapshot, "payload", None)
    lock_sha256 = getattr(lock_snapshot, "sha256", None)
    tracked = payload.get("tracked_files") if isinstance(payload, Mapping) else None
    if not _is_sha256(lock_sha256) or not isinstance(tracked, Mapping):
        raise RuntimeError("primary-v2 authorized lock provenance is malformed")
    corpus_semantic_sha256 = payload.get("corpus_semantic_sha256")
    if not _is_sha256(corpus_semantic_sha256):
        raise RuntimeError("primary-v2 lock corpus semantic provenance is missing")
    source_rows = {
        label.removeprefix("source/"): _tracked_digest_v2(tracked, label)
        for label in sorted(tracked)
        if type(label) is str and label.startswith("source/")
    }
    if not source_rows:
        raise RuntimeError("primary-v2 lock source provenance is empty")
    return {
        "protocol_lock_sha256": lock_sha256,
        "corpus_manifest_sha256": _tracked_digest_v2(
            tracked, "artifact/corpus_manifest"
        ),
        "split_sha256": _tracked_digest_v2(tracked, f"split/{split_name}"),
        "semantics_receipt_sha256": _tracked_digest_v2(
            tracked, "artifact/approval_semantics_receipt"
        ),
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "source_sha256": source_rows,
    }


def _validate_locked_provenance_v2(
    provenance: object,
    lock_snapshot,
    *,
    split_name: str,
    grid_sha256: object,
) -> None:
    expected = _locked_provenance_v2(lock_snapshot, split_name=split_name)
    expected["grid_sha256"] = grid_sha256
    if not exact_json_equal_v2(provenance, expected):
        raise RuntimeError(
            "primary-v2 provenance does not equal protocol-lock tracked digests"
        )


def _validate_training_grid_payload_v2(
    payload: Mapping[str, object],
    *,
    protocol_lock_sha256: str,
) -> None:
    if type(payload) is not dict or set(payload) != GRID_FIELDS:
        raise RuntimeError("training-only grid schema differs")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("semantics_profile") != SEMANTICS_PROFILE
        or payload.get("split") != "temporal_2022"
        or type(payload.get("training_only")) is not bool
        or payload.get("training_only") is not True
        or payload.get("selection_rule") != GRID_SELECTION_RULE
    ):
        raise RuntimeError("training-only grid schema differs")
    alphas = payload.get("alphas")
    losses = payload.get("train_worst_csd")
    if (
        type(alphas) is not list
        or len(alphas) != 61
        or any(type(value) is not float for value in alphas)
        or tuple(alphas) != STATIC_AGE_GRID_VALUES
    ):
        raise RuntimeError("training-only 61-value alpha grid differs")
    if (
        type(losses) is not list
        or len(losses) != len(alphas)
        or any(
            type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            for value in losses
        )
    ):
        raise RuntimeError("training-only grid losses must be finite and nonnegative")
    selected = payload.get("selected_alpha")
    if type(selected) is not float or not math.isfinite(selected):
        raise RuntimeError("training-only grid selected alpha is malformed")
    expected = min(zip(losses, alphas), key=lambda pair: (pair[0], pair[1]))[1]
    if selected != expected:
        raise RuntimeError("training-only grid selection differs")
    provenance = payload.get("provenance")
    provenance_fields = {
        "protocol_lock_sha256",
        "corpus_manifest_sha256",
        "split_sha256",
        "semantics_receipt_sha256",
        "corpus_semantic_sha256",
        "source_sha256",
    }
    if type(provenance) is not dict or set(provenance) != provenance_fields:
        raise RuntimeError("training-only grid provenance schema differs")
    if (
        not _is_sha256(protocol_lock_sha256)
        or provenance.get("protocol_lock_sha256") != protocol_lock_sha256
        or any(
            not _is_sha256(provenance.get(field))
            for field in (
                "corpus_manifest_sha256",
                "split_sha256",
                "semantics_receipt_sha256",
                "corpus_semantic_sha256",
            )
        )
        or type(provenance.get("source_sha256")) is not dict
        or not provenance["source_sha256"]
        or any(
            type(name) is not str or not _is_sha256(digest)
            for name, digest in provenance["source_sha256"].items()
        )
    ):
        raise RuntimeError("training grid protocol-lock digest differs")


def load_static_age_grid_dependency_v2(
    spec: Mapping[str, object],
    lock_snapshot,
    *,
    result_root: Path,
) -> JsonSnapshotV2:
    """Authenticate the static-age initializer's training-only scalar grid."""

    if spec.get("family") != "static_age_lookup" or spec.get(
        "grid_dependency"
    ) != "training_grid/static_senior_alpha.json":
        raise RuntimeError("static-age fit grid dependency differs")
    lock_payload = getattr(lock_snapshot, "payload", None)
    lock_sha256 = getattr(lock_snapshot, "sha256", None)
    if type(lock_payload) is not dict or not _is_sha256(lock_sha256):
        raise RuntimeError("training grid protocol-lock snapshot is malformed")
    tracked = lock_payload.get("tracked_files")
    if not isinstance(tracked, Mapping) or GRID_ARTIFACT_LABEL in tracked:
        raise RuntimeError("base lock must declare, not pre-hash, the training grid")
    expected_grid_spec = {
        "artifact_label": GRID_ARTIFACT_LABEL,
        "relative_path": "training_grid/static_senior_alpha.json",
        "split": "temporal_2022",
        "training_only": True,
        "alphas": list(STATIC_AGE_GRID_VALUES),
        "selection_metric": "training worst-cohort CSD",
        "tie_break": "smaller alpha",
        "created_after_lock": True,
    }
    if not exact_json_equal_v2(
        lock_payload.get("training_grid_spec"), expected_grid_spec
    ):
        raise RuntimeError("base lock training-grid specification differs")
    path = canonical_grid_path_v2(Path(result_root))
    content = read_regular_bytes_artifact_v2(path, label="training-only grid")
    assert content is not None
    observed = hashlib.sha256(content).hexdigest()
    from iclr_primary_warsaw_protocol_v2 import load_json_object_bytes_strict_v2

    payload = load_json_object_bytes_strict_v2(content, label="training-only grid")
    _validate_training_grid_payload_v2(
        payload,
        protocol_lock_sha256=lock_sha256,
    )
    expected_provenance = _locked_provenance_v2(
        lock_snapshot,
        split_name="temporal_2022",
    )
    if not exact_json_equal_v2(payload.get("provenance"), expected_provenance):
        raise RuntimeError("training grid provenance differs from protocol lock")
    return JsonSnapshotV2(path, content, observed, payload)


def _validate_grid_snapshot_for_compute_v2(
    grid_snapshot: JsonSnapshotV2,
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
    input_snapshot: PrimaryInputsSnapshotV2,
) -> dict[str, object]:
    if type(grid_snapshot) is not JsonSnapshotV2:
        raise RuntimeError("static-age computation requires an authenticated grid snapshot")
    if grid_snapshot.path.absolute() != canonical_grid_path_v2(
        input_snapshot.result_root
    ):
        raise RuntimeError("static-age training grid path differs")
    if (
        type(grid_snapshot.content) is not bytes
        or hashlib.sha256(grid_snapshot.content).hexdigest() != grid_snapshot.sha256
    ):
        raise RuntimeError("static-age training grid byte identity differs")
    from iclr_primary_warsaw_protocol_v2 import load_json_object_bytes_strict_v2

    parsed = load_json_object_bytes_strict_v2(
        grid_snapshot.content,
        label="static-age authenticated training grid",
    )
    if not exact_json_equal_v2(parsed, grid_snapshot.payload):
        raise RuntimeError("static-age training grid payload differs from its bytes")
    _validate_training_grid_payload_v2(
        grid_snapshot.payload,
        protocol_lock_sha256=lock_snapshot.sha256,
    )
    expected = _locked_provenance_v2(
        lock_snapshot,
        split_name="temporal_2022",
    )
    if not exact_json_equal_v2(grid_snapshot.payload.get("provenance"), expected):
        raise RuntimeError("static-age training grid provenance differs")
    recomputed = build_static_age_grid_from_snapshot_v2(
        input_snapshot=input_snapshot,
        authorized_lock_snapshot=lock_snapshot,
    )
    if not exact_json_equal_v2(grid_snapshot.payload, recomputed):
        raise RuntimeError(
            "static-age grid differs from authenticated training-data recomputation"
        )
    return recomputed


def _selected_training_grid_loss_v2(
    grid_payload: Mapping[str, object],
) -> float:
    """Return the loss paired with the grid's selected alpha."""

    alphas = grid_payload["alphas"]
    losses = grid_payload["train_worst_csd"]
    selected_alpha = grid_payload["selected_alpha"]
    if not isinstance(alphas, list) or not isinstance(losses, list):
        raise RuntimeError("authenticated static-age grid loss arrays are malformed")
    selected_index = alphas.index(selected_alpha)
    return float(losses[selected_index])


def build_fit_recipe_v2(
    spec: Mapping[str, object],
    *,
    grid_snapshot=None,
) -> dict[str, object]:
    """Translate one locked coordinate into its exact model-space recipe."""

    canonical = _canonical_inventory_match(spec)
    family = str(canonical["family"])
    arm = str(canonical["arm"])
    if arm == "outcome":
        feature_names = list(PROJECT_FEATURES)
        initial = cost_effective_weights().astype(float).tolist()
    elif arm == "endowment":
        feature_names = list(FEATURE_NAMES)
        if canonical["init"] == "zeros":
            initial = [0.0] * len(feature_names)
        else:
            initial = res_equivalent_weights(1.0).astype(float).tolist()
    elif arm == "static_age_lookup":
        feature_names = list(STATIC_AGE_FEATURE_NAMES)
        if grid_snapshot is None:
            raise RuntimeError("static-age recipe requires its training grid")
        grid_payload = getattr(grid_snapshot, "payload", None)
        grid_sha256 = getattr(grid_snapshot, "sha256", None)
        selected = grid_payload.get("selected_alpha") if isinstance(grid_payload, Mapping) else None
        if type(selected) is not float or not math.isfinite(selected) or not _is_sha256(grid_sha256):
            raise RuntimeError("static-age training grid is malformed")
        initial = [0.0, 0.0, 0.0, float(math.log1p(selected))]
    else:
        raise RuntimeError(f"unsupported primary-v2 arm: {arm}")

    masked_names = canonical["masked_feature_names"]
    masked_indices: list[int] = []
    for name in masked_names:
        if name not in feature_names:
            raise RuntimeError(f"masked feature is absent from {family}: {name}")
        masked_indices.append(feature_names.index(name))
    return {
        "family": family,
        "arm": arm,
        "feature_names": feature_names,
        "init_name": canonical["init"],
        "initial_weights": initial,
        "masked_feature_indices": masked_indices,
        "soft_welfare_target": canonical["soft_welfare_target"],
        "welfare_penalty": canonical["welfare_penalty"],
        "support_floor_kappa": canonical["support_floor_kappa"],
        "grid_sha256": (
            getattr(grid_snapshot, "sha256")
            if family == "static_age_lookup"
            else None
        ),
        "optimizer_config": {
            "sigma0": float(canonical["sigma0"]),
            "popsize": canonical["popsize"],
            "generations": int(canonical["generations"]),
            "seed": int(canonical["seed"]),
            "bound": float(canonical["bound"]),
        },
    }


def _masked_weights_v2(
    weights: Sequence[float],
    masked_indices: Sequence[int],
) -> np.ndarray:
    vector = np.asarray(weights, dtype=float).copy()
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise RuntimeError("optimizer weights must be a finite vector")
    for index in masked_indices:
        vector[index] = 0.0
    return vector


def optimize_fit_recipe_v2(
    recipe: Mapping[str, object],
    objective: Callable[[np.ndarray], float],
    *,
    cma_config: CMAESConfig | None = None,
) -> dict[str, object]:
    """Run the recipe's exact CMA-ES contract with masks on every evaluation."""

    initial = np.asarray(recipe["initial_weights"], dtype=float)
    masked_indices = tuple(int(index) for index in recipe["masked_feature_indices"])

    def wrapped(candidate: np.ndarray) -> float:
        value = float(objective(_masked_weights_v2(candidate, masked_indices)))
        if not math.isfinite(value):
            raise RuntimeError("primary-v2 objective returned a non-finite value")
        return value

    raw_config = recipe.get("optimizer_config")
    expected_fields = {"sigma0", "popsize", "generations", "seed", "bound"}
    if type(raw_config) is not dict or set(raw_config) != expected_fields:
        raise RuntimeError("locked CMA configuration is missing from fit recipe")
    if (
        type(raw_config.get("sigma0")) is not float
        or raw_config.get("popsize") is not None
        or type(raw_config.get("generations")) is not int
        or type(raw_config.get("seed")) is not int
        or type(raw_config.get("bound")) is not float
    ):
        raise RuntimeError("locked CMA configuration types differ")
    expected_config = CMAESConfig(
        sigma0=raw_config["sigma0"],
        popsize=raw_config["popsize"],
        generations=raw_config["generations"],
        seed=raw_config["seed"],
        bound=raw_config["bound"],
    )
    if cma_config is not None and cma_config != expected_config:
        raise RuntimeError("explicit CMA configuration differs from locked recipe")
    config = expected_config
    fitted = minimize(wrapped, initial, config)
    selected = _masked_weights_v2(fitted.best_x, masked_indices)
    return {
        "best_loss": float(fitted.best_f),
        "optimizer_best_loss": float(fitted.best_f),
        "optimizer_best_weights": np.asarray(fitted.best_x, dtype=float).tolist(),
        "selected_weights": selected.tolist(),
        "selection_source": "cmaes",
        "n_objective_evals": int(fitted.n_evals),
        "optimizer_config": dict(raw_config),
    }


def build_static_age_grid_payload_v2(
    training_loss: Callable[[float], float],
    *,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate and serialize all 61 prespecified training-only scalar points."""

    losses = [float(training_loss(alpha)) for alpha in STATIC_AGE_GRID_VALUES]
    if any(not math.isfinite(value) for value in losses):
        raise RuntimeError("training-only grid contains a non-finite loss")
    selected = min(
        zip(losses, STATIC_AGE_GRID_VALUES),
        key=lambda pair: (pair[0], pair[1]),
    )[1]
    payload: dict[str, object] = {
        "schema_version": 2,
        "semantics_profile": SEMANTICS_PROFILE,
        "split": "temporal_2022",
        "training_only": True,
        "alphas": list(STATIC_AGE_GRID_VALUES),
        "train_worst_csd": losses,
        "selected_alpha": float(selected),
        "selection_rule": GRID_SELECTION_RULE,
        "provenance": dict(provenance),
    }
    lock_sha256 = payload["provenance"].get("protocol_lock_sha256") if isinstance(payload["provenance"], Mapping) else None
    _validate_training_grid_payload_v2(
        payload,
        protocol_lock_sha256=str(lock_sha256),
    )
    return payload


def build_static_age_grid_from_snapshot_v2(
    *,
    input_snapshot: PrimaryInputsSnapshotV2,
    authorized_lock_snapshot: PrimaryProtocolLockSnapshotV2,
) -> dict[str, object]:
    """Build the prespecified grid solely from authenticated temporal training data."""

    validate_protocol_inputs_snapshot_v2(input_snapshot, authorized_lock_snapshot)
    if set(input_snapshot.splits) != {"temporal_2022"}:
        raise RuntimeError("static-age grid requires only the temporal training snapshot")
    assert_protocol_inputs_unchanged_v2(input_snapshot, authorized_lock_snapshot)
    data = load_training_series_data_v2(
        input_snapshot.splits["temporal_2022"],
        input_snapshot.index,
        input_snapshot.semantics_receipt.payload,
    )
    cfg = EnvConfig()

    def training_loss(alpha: float) -> float:
        weights = np.asarray(
            [0.0, 0.0, 0.0, float(math.log1p(alpha))],
            dtype=float,
        )
        selector = endowment_selector(_static_age_policy_v2(weights), cfg)
        return _training_worst_csd_v2(selector, data, cfg)

    payload = build_static_age_grid_payload_v2(
        training_loss,
        provenance=_locked_provenance_v2(
            authorized_lock_snapshot,
            split_name="temporal_2022",
        ),
    )
    assert_protocol_inputs_unchanged_v2(input_snapshot, authorized_lock_snapshot)
    return payload


def select_best_with_initial_v2(
    initial_weights: Sequence[float],
    initial_loss: float,
    optimizer_result,
) -> tuple[list[float], float, str]:
    """Retain the contained static initializer on a tie or worse CMA result."""

    initial = [float(value) for value in initial_weights]
    initial_value = float(initial_loss)
    optimizer_value = float(optimizer_result.best_f)
    if not math.isfinite(initial_value) or not math.isfinite(optimizer_value):
        raise RuntimeError("static-age selection loss must be finite")
    if initial_value <= optimizer_value:
        return initial, initial_value, "initial"
    selected = [float(value) for value in optimizer_result.best_x]
    if any(not math.isfinite(value) for value in selected):
        raise RuntimeError("static-age optimizer weights must be finite")
    return selected, optimizer_value, "cmaes"


def _static_age_policy_v2(log_multipliers: Sequence[float]):
    logits = np.asarray(log_multipliers, dtype=float)
    if logits.shape != (len(STATIC_AGE_FEATURE_NAMES),) or not np.all(
        np.isfinite(logits)
    ):
        raise RuntimeError("static-age logits must be a finite four-vector")
    factors = {
        label: float(math.exp(value))
        for label, value in zip(STATIC_AGE_FEATURE_NAMES, logits)
    }

    def policy(instance: PBInstance, state: RolloutState) -> list[float]:
        del state
        count = len(instance.votes)
        if count == 0:
            return []
        base = instance.budget / count
        raw = [
            base * factors.get(age_bracket(vote.age), 1.0)
            for vote in instance.votes
        ]
        total = math.fsum(raw)
        if total <= 0.0:
            return [base] * count
        scale = instance.budget / total
        return [value * scale for value in raw]

    return policy


def _training_worst_csd_v2(selector, data: Sequence[SeriesData], cfg: EnvConfig) -> float:
    episodes = [
        rollout_selector(
            row.ref,
            selector,
            score_years=row.train_years,
            cfg=cfg,
            instances=row.train_only,
        )
        for row in data
        if row.train_years
    ]
    stats = aggregate(episodes)
    if not stats.get("n_series"):
        raise RuntimeError("primary-v2 training view contains no scored series")
    value = float(stats["worst_csd"])
    if not math.isfinite(value):
        raise RuntimeError("primary-v2 training objective is non-finite")
    return value


def build_training_objective_v2(
    recipe: Mapping[str, object],
    data: Sequence[SeriesData],
    env_config: EnvConfig,
) -> Callable[[np.ndarray], float]:
    """Build the exact training-only objective for one fit family."""

    family = recipe["family"]
    if family == "static_support_floor":
        kappa = recipe["support_floor_kappa"]
        return lambda weights: _training_worst_csd_v2(
            score_selector(np.asarray(weights, dtype=float), support_floor=float(kappa)),
            data,
            env_config,
        )
    if family == "static_age_lookup":
        return lambda weights: _training_worst_csd_v2(
            endowment_selector(_static_age_policy_v2(weights), env_config),
            data,
            env_config,
        )

    arm = build_arm(str(recipe["arm"]))
    mes_train = _evaluate(
        endowment_selector(uniform_policy, env_config),
        data,
        env_config,
        on_test=False,
    )
    mes_welfare = float(mes_train["welfare"])
    train_config = TrainConfig(
        arm=str(recipe["arm"]),
        welfare_penalty=float(recipe["welfare_penalty"]),
        welfare_floor=float(recipe["soft_welfare_target"]),
    )

    def objective(weights: np.ndarray) -> float:
        return _objective(
            arm.selector(np.asarray(weights, dtype=float), env_config),
            data,
            env_config,
            train_config,
            mes_welfare,
        )

    return objective


def fit_one_v2(
    spec: Mapping[str, object],
    *,
    input_snapshot: PrimaryInputsSnapshotV2,
    authorized_lock_snapshot: PrimaryProtocolLockSnapshotV2,
    grid_snapshot=None,
) -> dict[str, object]:
    """Compute one locked coordinate from authenticated training inputs only."""

    canonical = _locked_inventory_match_v2(spec, authorized_lock_snapshot)
    validate_protocol_inputs_snapshot_v2(input_snapshot, authorized_lock_snapshot)
    split_name = str(canonical["split"])
    if set(input_snapshot.splits) != {split_name}:
        raise RuntimeError("fit input snapshot must contain exactly its locked split")
    assert_protocol_inputs_unchanged_v2(input_snapshot, authorized_lock_snapshot)
    data = load_training_series_data_v2(
        input_snapshot.splits[split_name],
        input_snapshot.index,
        input_snapshot.semantics_receipt.payload,
    )
    if canonical["family"] == "static_age_lookup":
        if grid_snapshot is None:
            raise RuntimeError("static-age fit is missing its authenticated grid")
        authenticated_grid = _validate_grid_snapshot_for_compute_v2(
            grid_snapshot,
            authorized_lock_snapshot,
            input_snapshot,
        )
    elif grid_snapshot is not None:
        raise RuntimeError("non-static fit must not receive a training grid")
    recipe = build_fit_recipe_v2(canonical, grid_snapshot=grid_snapshot)
    cfg = EnvConfig()
    objective = build_training_objective_v2(recipe, data, cfg)  # type: ignore[arg-type]
    optimized = optimize_fit_recipe_v2(
        recipe,
        objective,
    )
    if canonical["family"] == "static_age_lookup":
        initial = list(recipe["initial_weights"])
        initial_loss = float(objective(np.asarray(initial, dtype=float)))
        expected_initial_loss = _selected_training_grid_loss_v2(authenticated_grid)
        if initial_loss != expected_initial_loss:
            raise RuntimeError(
                "static-age initializer loss differs from the authenticated grid loss"
            )
        selected, best_loss, source = select_best_with_initial_v2(
            initial,
            initial_loss,
            type(
                "OptimizerSelection",
                (),
                {
                    "best_x": optimized["optimizer_best_weights"],
                    "best_f": optimized["best_loss"],
                },
            )(),
        )
        optimized["selected_weights"] = selected
        optimized["best_loss"] = best_loss
        optimized["selection_source"] = source
        optimized["n_objective_evals"] = int(optimized["n_objective_evals"]) + 1

    split_artifact = input_snapshot.split_artifacts.get(split_name)
    if split_artifact is None:
        raise RuntimeError(f"primary-v2 input snapshot lacks split {split_name}")
    provenance = _locked_provenance_v2(
        authorized_lock_snapshot,
        split_name=split_name,
    )
    provenance["grid_sha256"] = recipe["grid_sha256"]
    payload: dict[str, object] = {
        "schema_version": 2,
        "fit_id": canonical["fit_id"],
        "config": canonical,
        "training_only": True,
        "execution_mode": "serial",
        "arm": {
            "name": canonical["arm"],
            "feature_names": list(recipe["feature_names"]),
            "init_name": canonical["init"],
        },
        "provenance": provenance,
        "result": optimized,
    }
    validate_fit_payload_v2(
        payload,
        expected_spec=canonical,
        expected_protocol_lock_sha256=authorized_lock_snapshot.sha256,
    )
    assert_protocol_inputs_unchanged_v2(input_snapshot, authorized_lock_snapshot)
    return payload


def validate_fit_payload_v2(
    payload: Mapping[str, object],
    *,
    expected_spec: Mapping[str, object],
    expected_protocol_lock_sha256: str,
) -> Mapping[str, object]:
    """Require the exact deterministic, training-only fit schema."""

    top_fields = {
        "schema_version",
        "fit_id",
        "config",
        "training_only",
        "execution_mode",
        "arm",
        "provenance",
        "result",
    }
    if type(payload) is not dict or set(payload) != top_fields:
        raise RuntimeError("training-only fit schema differs or leaks held-out data")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("fit_id") != expected_spec.get("fit_id")
        or not exact_json_equal_v2(payload.get("config"), expected_spec)
        or type(payload.get("training_only")) is not bool
        or payload.get("training_only") is not True
        or payload.get("execution_mode") != "serial"
    ):
        raise RuntimeError("training-only fit schema differs")
    arm = payload.get("arm")
    if type(arm) is not dict or set(arm) != {"name", "feature_names", "init_name"}:
        raise RuntimeError("fit arm schema differs")
    expected_arm = expected_spec.get("arm")
    expected_feature_names = {
        "outcome": list(PROJECT_FEATURES),
        "endowment": list(FEATURE_NAMES),
        "static_age_lookup": list(STATIC_AGE_FEATURE_NAMES),
    }.get(expected_arm)
    if (
        expected_feature_names is None
        or arm.get("name") != expected_arm
        or arm.get("init_name") != expected_spec.get("init")
        or not exact_json_equal_v2(arm.get("feature_names"), expected_feature_names)
    ):
        raise RuntimeError("fit arm schema differs")
    provenance = payload.get("provenance")
    provenance_fields = {
        "protocol_lock_sha256",
        "corpus_manifest_sha256",
        "split_sha256",
        "semantics_receipt_sha256",
        "corpus_semantic_sha256",
        "source_sha256",
        "grid_sha256",
    }
    if type(provenance) is not dict or set(provenance) != provenance_fields:
        raise RuntimeError("fit provenance schema differs")
    if (
        not _is_sha256(expected_protocol_lock_sha256)
        or provenance.get("protocol_lock_sha256") != expected_protocol_lock_sha256
        or any(
            not _is_sha256(provenance.get(field))
            for field in (
                "corpus_manifest_sha256",
                "split_sha256",
                "semantics_receipt_sha256",
                "corpus_semantic_sha256",
            )
        )
        or type(provenance.get("source_sha256")) is not dict
        or not provenance["source_sha256"]
        or any(
            type(name) is not str or not _is_sha256(digest)
            for name, digest in provenance["source_sha256"].items()
        )
    ):
        raise RuntimeError("fit provenance differs")
    grid_sha = provenance.get("grid_sha256")
    expects_grid = expected_spec.get("grid_dependency") is not None
    if (expects_grid and not _is_sha256(grid_sha)) or (
        not expects_grid and grid_sha is not None
    ):
        raise RuntimeError("fit grid provenance differs")
    result = payload.get("result")
    result_fields = {
        "best_loss",
        "optimizer_best_loss",
        "optimizer_best_weights",
        "selected_weights",
        "selection_source",
        "n_objective_evals",
        "optimizer_config",
    }
    if type(result) is not dict or set(result) != result_fields:
        raise RuntimeError("fit result schema differs")
    optimizer = result.get("optimizer_best_weights")
    selected = result.get("selected_weights")
    if (
        type(result.get("best_loss")) is not float
        or not math.isfinite(result["best_loss"])
        or result["best_loss"] < 0.0
        or type(result.get("optimizer_best_loss")) is not float
        or not math.isfinite(result["optimizer_best_loss"])
        or result["optimizer_best_loss"] < 0.0
        or type(optimizer) is not list
        or type(selected) is not list
        or len(optimizer) != len(arm["feature_names"])
        or len(selected) != len(optimizer)
        or any(type(value) is not float or not math.isfinite(value) for value in optimizer)
        or any(type(value) is not float or not math.isfinite(value) for value in selected)
        or type(result.get("selection_source")) is not str
    ):
        raise RuntimeError(
            "fit result must contain finite nonnegative losses and deterministic values"
        )
    expected_optimizer_config = {
        "sigma0": expected_spec.get("sigma0"),
        "popsize": expected_spec.get("popsize"),
        "generations": expected_spec.get("generations"),
        "seed": expected_spec.get("seed"),
        "bound": expected_spec.get("bound"),
    }
    if not exact_json_equal_v2(
        result.get("optimizer_config"),
        expected_optimizer_config,
    ):
        raise RuntimeError("fit result optimizer configuration differs")
    configured_popsize = expected_spec.get("popsize")
    effective_popsize = (
        configured_popsize
        if type(configured_popsize) is int
        else 4 + int(3 * math.log(len(optimizer)))
    )
    expected_evals = int(expected_spec["generations"]) * effective_popsize
    if expected_spec.get("family") == "static_age_lookup":
        expected_evals += 1
    if (
        type(result.get("n_objective_evals")) is not int
        or result["n_objective_evals"] != expected_evals
    ):
        raise RuntimeError(
            "fit objective evaluation count differs from the locked CMA contract"
        )
    allowed_selection_sources = (
        {"initial", "cmaes"}
        if expected_spec.get("family") == "static_age_lookup"
        else {"cmaes"}
    )
    if result["selection_source"] not in allowed_selection_sources:
        raise RuntimeError("fit result selection source differs for its family")
    masked = expected_spec.get("masked_feature_names")
    if type(masked) is not list:
        raise RuntimeError("fit masked-feature contract differs")
    masked_indices: list[int] = []
    for name in masked:
        if name not in arm["feature_names"]:
            raise RuntimeError("masked feature is absent from the fit arm")
        index = arm["feature_names"].index(name)
        masked_indices.append(index)
        if selected[index] != 0.0:
            raise RuntimeError("masked selected weights must be exactly zero")
    masked_optimizer = list(optimizer)
    for index in masked_indices:
        masked_optimizer[index] = 0.0
    if result["selection_source"] == "cmaes":
        if not exact_json_equal_v2(selected, masked_optimizer):
            raise RuntimeError("selected weights differ from the masked CMA result")
        if result["best_loss"] != result["optimizer_best_loss"]:
            raise RuntimeError("selected CMA loss differs from the optimizer best loss")
    else:
        static_initializers = [
            [0.0, 0.0, 0.0, float(math.log1p(alpha))]
            for alpha in STATIC_AGE_GRID_VALUES
        ]
        if not any(exact_json_equal_v2(selected, row) for row in static_initializers):
            raise RuntimeError("selected initial weights differ from the locked grid")
        if result["best_loss"] > result["optimizer_best_loss"]:
            raise RuntimeError(
                "selected initializer loss exceeds the optimizer best loss"
            )
    bound = float(expected_spec["bound"])
    if any(abs(value) > bound for value in (*optimizer, *selected)):
        raise RuntimeError("fit weights exceed the locked CMA bound")
    _require_finite_json(payload, label="training-only fit")
    return payload


def serialize_fit_payload_v2(payload: Mapping[str, object]) -> bytes:
    """Serialize a validated JSON-shaped fit without machine-local metadata."""

    _require_finite_json(payload, label="training-only fit")
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_immutable_fit_v2(
    path: Path,
    payload: Mapping[str, object],
    *,
    expected_spec: Mapping[str, object],
    authorized_lock_snapshot,
    input_snapshot: PrimaryInputsSnapshotV2,
    grid_snapshot,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> str:
    """Revalidate every dependency after compute, then install one fit once."""

    if type(input_snapshot) is not PrimaryInputsSnapshotV2:
        raise RuntimeError("primary-v2 fit requires an authenticated input snapshot")
    if (
        input_snapshot.result_root != Path(result_root).absolute()
        or input_snapshot.data_dir != Path(data_dir).absolute()
    ):
        raise RuntimeError("primary-v2 fit computation input roots differ")
    locked_spec = _locked_inventory_match_v2(
        expected_spec,
        authorized_lock_snapshot,
    )
    if set(input_snapshot.splits) != {str(locked_spec["split"])}:
        raise RuntimeError("primary-v2 fit computation snapshot has the wrong split")
    expected_path = Path(result_root).absolute() / str(locked_spec["relative_path"])
    if Path(path).absolute() != expected_path:
        raise RuntimeError("primary-v2 fit output path is not canonical")
    validate_fit_payload_v2(
        payload,
        expected_spec=locked_spec,
        expected_protocol_lock_sha256=authorized_lock_snapshot.sha256,
    )
    expected_grid_sha256 = (
        getattr(grid_snapshot, "sha256", None)
        if locked_spec.get("grid_dependency") is not None
        else None
    )
    _validate_locked_provenance_v2(
        payload.get("provenance"),
        authorized_lock_snapshot,
        split_name=str(locked_spec["split"]),
        grid_sha256=expected_grid_sha256,
    )
    authenticated_grid: dict[str, object] | None = None
    if locked_spec.get("grid_dependency") is not None:
        if grid_snapshot is None:
            raise RuntimeError("static-age fit is missing its training grid")
        authenticated_grid = _validate_grid_snapshot_for_compute_v2(
            grid_snapshot,
            authorized_lock_snapshot,
            input_snapshot,
        )
    elif grid_snapshot is not None:
        raise RuntimeError("non-static fit must not depend on a training grid")

    current_lock = verify_primary_warsaw_protocol_lock_snapshot_v2(
        authorized_lock_snapshot.path,
        Path(repo_root),
        Path(result_root),
        Path(data_dir),
    )
    if (
        current_lock.sha256 != authorized_lock_snapshot.sha256
        or current_lock.content != authorized_lock_snapshot.content
        or not exact_json_equal_v2(
            current_lock.payload,
            authorized_lock_snapshot.payload,
        )
    ):
        raise RuntimeError("primary-v2 protocol lock changed after fit computation")

    _locked_inventory_match_v2(locked_spec, current_lock)
    _validate_locked_provenance_v2(
        payload.get("provenance"),
        current_lock,
        split_name=str(locked_spec["split"]),
        grid_sha256=expected_grid_sha256,
    )

    if locked_spec.get("grid_dependency") is not None:
        if grid_snapshot is None or authenticated_grid is None:
            raise RuntimeError("static-age grid authentication did not complete")
        current_grid = load_static_age_grid_dependency_v2(
            locked_spec,
            current_lock,
            result_root=Path(result_root),
        )
        if (
            current_grid.path != grid_snapshot.path
            or current_grid.sha256 != grid_snapshot.sha256
            or current_grid.content != grid_snapshot.content
            or not exact_json_equal_v2(current_grid.payload, grid_snapshot.payload)
            or not exact_json_equal_v2(current_grid.payload, authenticated_grid)
        ):
            raise RuntimeError("training grid changed after fit computation")
        selected_grid_loss = _selected_training_grid_loss_v2(authenticated_grid)
        result = payload["result"]
        if result["selection_source"] == "initial":
            selected_alpha = current_grid.payload["selected_alpha"]
            expected_initial = [
                0.0,
                0.0,
                0.0,
                float(math.log1p(selected_alpha)),
            ]
            if not exact_json_equal_v2(
                payload["result"]["selected_weights"],
                expected_initial,
            ):
                raise RuntimeError(
                    "static-age selected weights differ from the locked grid initializer"
                )
            if result["best_loss"] != selected_grid_loss:
                raise RuntimeError(
                    "static-age selected loss differs from the authenticated grid loss"
                )
        elif result["best_loss"] >= selected_grid_loss:
            raise RuntimeError(
                "static-age CMA selection does not improve on the authenticated grid loss"
            )

    assert_protocol_inputs_unchanged_v2(input_snapshot, current_lock)
    content = serialize_fit_payload_v2(payload)
    return write_immutable_bytes_artifact_v2(
        Path(path),
        content,
        label="primary-v2 immutable fit",
    )


def write_static_age_grid_v2(
    path: Path,
    payload: Mapping[str, object],
    *,
    authorized_lock_snapshot,
    input_snapshot: PrimaryInputsSnapshotV2,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> str:
    """Revalidate the base lock after grid computation, then install once."""

    if type(input_snapshot) is not PrimaryInputsSnapshotV2:
        raise RuntimeError("primary-v2 grid requires an authenticated input snapshot")
    if (
        input_snapshot.result_root != Path(result_root).absolute()
        or input_snapshot.data_dir != Path(data_dir).absolute()
    ):
        raise RuntimeError("primary-v2 grid computation input roots differ")
    if set(input_snapshot.splits) != {"temporal_2022"}:
        raise RuntimeError("primary-v2 grid computation snapshot has the wrong split")
    expected_path = canonical_grid_path_v2(Path(result_root))
    if Path(path).absolute() != expected_path:
        raise RuntimeError("training-grid output path is not canonical")
    _validate_training_grid_payload_v2(
        payload,
        protocol_lock_sha256=authorized_lock_snapshot.sha256,
    )
    expected_provenance = _locked_provenance_v2(
        authorized_lock_snapshot,
        split_name="temporal_2022",
    )
    if not exact_json_equal_v2(payload.get("provenance"), expected_provenance):
        raise RuntimeError(
            "primary-v2 grid provenance does not equal protocol-lock tracked digests"
        )
    recomputed = build_static_age_grid_from_snapshot_v2(
        input_snapshot=input_snapshot,
        authorized_lock_snapshot=authorized_lock_snapshot,
    )
    if not exact_json_equal_v2(payload, recomputed):
        raise RuntimeError(
            "primary-v2 grid payload differs from authenticated recomputation"
        )
    current_lock = verify_primary_warsaw_protocol_lock_snapshot_v2(
        authorized_lock_snapshot.path,
        Path(repo_root),
        Path(result_root),
        Path(data_dir),
    )
    if (
        current_lock.sha256 != authorized_lock_snapshot.sha256
        or current_lock.content != authorized_lock_snapshot.content
        or not exact_json_equal_v2(
            current_lock.payload,
            authorized_lock_snapshot.payload,
        )
    ):
        raise RuntimeError("primary-v2 protocol lock changed after grid computation")
    current_provenance = _locked_provenance_v2(
        current_lock,
        split_name="temporal_2022",
    )
    if not exact_json_equal_v2(payload.get("provenance"), current_provenance):
        raise RuntimeError(
            "primary-v2 grid provenance does not equal current lock tracked digests"
        )
    assert_protocol_inputs_unchanged_v2(input_snapshot, current_lock)
    content = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return write_immutable_bytes_artifact_v2(
        expected_path,
        content,
        label="primary-v2 immutable training grid",
    )


def fit_spec_by_id_v2(fit_id: str) -> dict[str, object]:
    """Resolve one opaque fit id; no free hyperparameter surface is exposed."""

    if type(fit_id) is not str or not fit_id:
        raise RuntimeError("primary-v2 fit id must be a non-empty string")
    inventory = locked_fit_inventory_v2()
    validate_locked_fit_inventory_v2(inventory)
    matches = [row for row in inventory if row["fit_id"] == fit_id]
    if len(matches) != 1:
        raise RuntimeError(f"primary-v2 fit id is outside the locked inventory: {fit_id}")
    return matches[0]


def validate_max_workers_v2(max_workers: int) -> int:
    """Bound host-level parallelism while every fit stays internally serial."""

    if type(max_workers) is not int:
        raise TypeError("primary-v2 worker count must be an integer")
    if not 1 <= max_workers <= MAX_MATRIX_WORKERS_V2:
        raise ValueError(
            "primary-v2 worker count must be between 1 and "
            f"{MAX_MATRIX_WORKERS_V2}"
        )
    return max_workers


def _optional_regular_bytes_v2(path: Path, *, label: str) -> bytes | None:
    """Read an optional file without treating an absent parent as an error."""

    target = Path(path).absolute()
    current = Path(target.anchor)
    for component in target.parent.parts[1:]:
        current /= component
        try:
            identity = os.lstat(current)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(identity.st_mode):
            raise RuntimeError(f"{label} parent contains a symlink: {current}")
        if not stat.S_ISDIR(identity.st_mode):
            raise RuntimeError(f"{label} parent contains a non-directory: {current}")
    return read_regular_bytes_artifact_v2(
        target,
        label=label,
        allow_missing=True,
    )


def _static_fit_grid_coherence_v2(
    payload: Mapping[str, object],
    grid_snapshot: JsonSnapshotV2,
) -> None:
    result = payload.get("result")
    if type(result) is not dict:
        raise RuntimeError("primary-v2 static fit result is malformed")
    selected_grid_loss = _selected_training_grid_loss_v2(grid_snapshot.payload)
    if result.get("selection_source") == "initial":
        alpha = grid_snapshot.payload["selected_alpha"]
        expected_weights = [0.0, 0.0, 0.0, float(math.log1p(alpha))]
        if not exact_json_equal_v2(result.get("selected_weights"), expected_weights):
            raise RuntimeError(
                "primary-v2 static fit initializer differs from its training grid"
            )
        if result.get("best_loss") != selected_grid_loss:
            raise RuntimeError(
                "primary-v2 static fit selected loss differs from its training grid"
            )
    elif (
        type(result.get("best_loss")) is not float
        or result["best_loss"] >= selected_grid_loss
    ):
        raise RuntimeError(
            "primary-v2 static CMA fit does not improve on its training grid"
        )


def authenticate_existing_fit_v2(
    spec: Mapping[str, object],
    *,
    authorized_lock_snapshot: PrimaryProtocolLockSnapshotV2,
    result_root: Path,
    grid_snapshot: JsonSnapshotV2 | None = None,
) -> JsonSnapshotV2 | None:
    """Authenticate one existing canonical fit, or return ``None`` if absent."""

    locked = _locked_inventory_match_v2(spec, authorized_lock_snapshot)
    path = canonical_fit_path_v2(Path(result_root), locked)
    content = _optional_regular_bytes_v2(
        path,
        label=f"primary-v2 fit {locked['fit_id']}",
    )
    if content is None:
        return None
    payload = load_json_object_bytes_strict_v2(
        content,
        label=f"primary-v2 fit {locked['fit_id']}",
    )
    validate_fit_payload_v2(
        payload,
        expected_spec=locked,
        expected_protocol_lock_sha256=authorized_lock_snapshot.sha256,
    )
    if locked.get("grid_dependency") is not None:
        current_grid = grid_snapshot or load_static_age_grid_dependency_v2(
            locked,
            authorized_lock_snapshot,
            result_root=Path(result_root),
        )
        expected_grid_sha256: object = current_grid.sha256
        _static_fit_grid_coherence_v2(payload, current_grid)
    else:
        if grid_snapshot is not None:
            raise RuntimeError("primary-v2 non-static fit received a training grid")
        expected_grid_sha256 = None
    _validate_locked_provenance_v2(
        payload.get("provenance"),
        authorized_lock_snapshot,
        split_name=str(locked["split"]),
        grid_sha256=expected_grid_sha256,
    )
    if read_regular_bytes_artifact_v2(
        path,
        label=f"primary-v2 fit {locked['fit_id']}",
    ) != content:
        raise RuntimeError(f"primary-v2 fit changed during read: {locked['fit_id']}")
    return JsonSnapshotV2(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
    )


def _assert_fit_tree_paths_v2(
    result_root: Path,
    expected_paths: Sequence[Path],
    allow_missing: bool,
) -> None:
    """Reject symlinks, special files, and coordinates outside the lock."""

    fit_root = Path(result_root).absolute() / "fits"
    expected = {Path(path).absolute() for path in expected_paths}
    try:
        fit_root_identity = os.lstat(fit_root)
    except FileNotFoundError:
        if allow_missing:
            return
        raise RuntimeError("primary-v2 fit matrix is incomplete: all 49 fits are missing")
    if stat.S_ISLNK(fit_root_identity.st_mode) or not stat.S_ISDIR(
        fit_root_identity.st_mode
    ):
        raise RuntimeError("primary-v2 fit tree root is not a regular directory")
    observed: set[Path] = set()
    pending = [fit_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path).absolute()
                if entry.is_symlink():
                    raise RuntimeError(f"primary-v2 fit tree contains a symlink: {path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    observed.add(path)
                else:
                    raise RuntimeError(
                        f"primary-v2 fit tree contains a special file: {path}"
                    )
    extra = sorted(str(path) for path in observed - expected)
    if extra:
        raise RuntimeError(f"primary-v2 fit tree has unexpected paths: {extra}")
    missing = expected - observed
    if missing and not allow_missing:
        raise RuntimeError(
            f"primary-v2 fit matrix is incomplete: {len(missing)} of 49 fits missing"
        )


def _evaluation_replay_started_v2(result_root: Path) -> bool:
    path = Path(result_root).absolute() / EVALUATION_REPLAY_STARTED_NAME_V2
    return (
        _optional_regular_bytes_v2(
            path,
            label="primary-v2 evaluation replay receipt",
        )
        is not None
    )


def run_static_age_grid_v2(
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Create or authenticate the single locked 61-point training grid."""

    root = Path(repo_root).absolute()
    result = Path(result_root).absolute()
    data = Path(data_dir).absolute()
    lock = verify_primary_warsaw_protocol_lock_snapshot_v2(
        result / "protocol_lock.json",
        root,
        result,
        data,
    )
    static_spec = fit_spec_by_id_v2(
        str(
            next(
                row["fit_id"]
                for row in locked_fit_inventory_v2()
                if row["family"] == "static_age_lookup"
            )
        )
    )
    grid_path = canonical_grid_path_v2(result)
    if _optional_regular_bytes_v2(
        grid_path,
        label="primary-v2 training grid",
    ) is not None:
        grid = load_static_age_grid_dependency_v2(
            static_spec,
            lock,
            result_root=result,
        )
        return {
            "status": "existing",
            "path": str(grid.path),
            "sha256": grid.sha256,
            "selected_alpha": grid.payload["selected_alpha"],
            "grid_points": len(grid.payload["alphas"]),
        }
    if _evaluation_replay_started_v2(result):
        raise RuntimeError(
            "primary-v2 evaluation already started; missing grid cannot be created"
        )
    inputs = load_protocol_inputs_snapshot_v2(
        lock,
        result,
        data,
        ("temporal_2022",),
    )
    payload = build_static_age_grid_from_snapshot_v2(
        input_snapshot=inputs,
        authorized_lock_snapshot=lock,
    )
    digest = write_static_age_grid_v2(
        grid_path,
        payload,
        authorized_lock_snapshot=lock,
        input_snapshot=inputs,
        repo_root=root,
        result_root=result,
        data_dir=data,
    )
    grid = load_static_age_grid_dependency_v2(
        static_spec,
        lock,
        result_root=result,
    )
    if grid.sha256 != digest or not exact_json_equal_v2(grid.payload, payload):
        raise RuntimeError("primary-v2 training grid changed after installation")
    return {
        "status": "written",
        "path": str(grid.path),
        "sha256": grid.sha256,
        "selected_alpha": grid.payload["selected_alpha"],
        "grid_points": len(grid.payload["alphas"]),
    }


def run_one_fit_v2(
    *,
    fit_id: str,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Create or authenticate one exact immutable coordinate from the lock."""

    root = Path(repo_root).absolute()
    result = Path(result_root).absolute()
    data = Path(data_dir).absolute()
    spec = fit_spec_by_id_v2(fit_id)
    lock = verify_primary_warsaw_protocol_lock_snapshot_v2(
        result / "protocol_lock.json",
        root,
        result,
        data,
    )
    locked = _locked_inventory_match_v2(spec, lock)
    grid = (
        load_static_age_grid_dependency_v2(locked, lock, result_root=result)
        if locked.get("grid_dependency") is not None
        else None
    )
    existing = authenticate_existing_fit_v2(
        locked,
        authorized_lock_snapshot=lock,
        result_root=result,
        grid_snapshot=grid,
    )
    if existing is not None:
        return {
            "status": "existing",
            "fit_id": fit_id,
            "path": str(existing.path),
            "sha256": existing.sha256,
        }
    if _evaluation_replay_started_v2(result):
        raise RuntimeError(
            "primary-v2 evaluation already started; a missing fit cannot be created"
        )
    inputs = load_protocol_inputs_snapshot_v2(
        lock,
        result,
        data,
        (str(locked["split"]),),
    )
    payload = fit_one_v2(
        locked,
        input_snapshot=inputs,
        authorized_lock_snapshot=lock,
        grid_snapshot=grid,
    )
    path = canonical_fit_path_v2(result, locked)
    digest = write_immutable_fit_v2(
        path,
        payload,
        expected_spec=locked,
        authorized_lock_snapshot=lock,
        input_snapshot=inputs,
        grid_snapshot=grid,
        repo_root=root,
        result_root=result,
        data_dir=data,
    )
    installed = authenticate_existing_fit_v2(
        locked,
        authorized_lock_snapshot=lock,
        result_root=result,
        grid_snapshot=grid,
    )
    if installed is None or installed.sha256 != digest:
        raise RuntimeError(f"primary-v2 fit did not authenticate after write: {fit_id}")
    return {
        "status": "written",
        "fit_id": fit_id,
        "path": str(installed.path),
        "sha256": installed.sha256,
    }


def _fit_worker_v2(
    fit_id: str,
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Pickle-safe process worker for one distinct immutable fit path."""

    return run_one_fit_v2(
        fit_id=fit_id,
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
    )


def verify_complete_matrix_v2(
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Authenticate the grid, exact 49-fit tree, schemas, and provenance."""

    root = Path(repo_root).absolute()
    result = Path(result_root).absolute()
    data = Path(data_dir).absolute()
    lock = verify_primary_warsaw_protocol_lock_snapshot_v2(
        result / "protocol_lock.json",
        root,
        result,
        data,
    )
    inventory = locked_fit_inventory_v2()
    validate_locked_fit_inventory_v2(inventory)
    static_spec = next(
        row for row in inventory if row["family"] == "static_age_lookup"
    )
    grid = load_static_age_grid_dependency_v2(
        static_spec,
        lock,
        result_root=result,
    )
    snapshots: dict[str, JsonSnapshotV2] = {}
    missing: list[str] = []
    for spec in inventory:
        snapshot = authenticate_existing_fit_v2(
            spec,
            authorized_lock_snapshot=lock,
            result_root=result,
            grid_snapshot=grid if spec["family"] == "static_age_lookup" else None,
        )
        if snapshot is None:
            missing.append(str(spec["fit_id"]))
        else:
            snapshots[str(spec["fit_id"])] = snapshot
    if missing:
        raise RuntimeError(
            f"primary-v2 fit matrix is incomplete: {len(missing)} of 49 fits missing"
        )
    expected_paths = [canonical_fit_path_v2(result, spec) for spec in inventory]
    _assert_fit_tree_paths_v2(result, expected_paths, False)
    return {
        "schema_version": 2,
        "status": "complete",
        "protocol_lock_sha256": lock.sha256,
        "grid_sha256": grid.sha256,
        "fit_count": len(snapshots),
        "fit_sha256": {
            fit_id: snapshots[fit_id].sha256 for fit_id in sorted(snapshots)
        },
    }


def run_fit_matrix_v2(
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
    max_workers: int = DEFAULT_MATRIX_WORKERS_V2,
) -> dict[str, object]:
    """Resume missing coordinates in separate processes, then verify all 49."""

    workers = validate_max_workers_v2(max_workers)
    root = Path(repo_root).absolute()
    result = Path(result_root).absolute()
    data = Path(data_dir).absolute()
    lock = verify_primary_warsaw_protocol_lock_snapshot_v2(
        result / "protocol_lock.json",
        root,
        result,
        data,
    )
    inventory = locked_fit_inventory_v2()
    validate_locked_fit_inventory_v2(inventory)
    static_spec = next(
        row for row in inventory if row["family"] == "static_age_lookup"
    )
    grid = load_static_age_grid_dependency_v2(
        static_spec,
        lock,
        result_root=result,
    )
    expected_paths = [canonical_fit_path_v2(result, spec) for spec in inventory]
    _assert_fit_tree_paths_v2(result, expected_paths, True)
    missing: list[str] = []
    for spec in inventory:
        snapshot = authenticate_existing_fit_v2(
            spec,
            authorized_lock_snapshot=lock,
            result_root=result,
            grid_snapshot=grid if spec["family"] == "static_age_lookup" else None,
        )
        if snapshot is None:
            missing.append(str(spec["fit_id"]))
    preexisting = len(inventory) - len(missing)
    if missing and _evaluation_replay_started_v2(result):
        raise RuntimeError(
            "primary-v2 evaluation already started; incomplete fits cannot resume"
        )

    futures = {}
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fit_id in missing:
                future = pool.submit(
                    _fit_worker_v2,
                    fit_id,
                    repo_root=root,
                    result_root=result,
                    data_dir=data,
                )
                futures[future] = fit_id
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
    complete = verify_complete_matrix_v2(
        repo_root=root,
        result_root=result,
        data_dir=data,
    )
    return {
        **complete,
        "preexisting_fit_count": preexisting,
        "executed_fit_count": len(missing),
        "max_workers": workers,
    }


def _add_runtime_paths_v2(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT_V2)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "pb")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the grid, one closed fit, the resumable matrix, or verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    grid = commands.add_parser("grid", help="create/authenticate the 61-point grid")
    _add_runtime_paths_v2(grid)
    one = commands.add_parser("one-fit", help="create/authenticate one locked fit")
    _add_runtime_paths_v2(one)
    one.add_argument("--fit-id", required=True)
    matrix = commands.add_parser("matrix", help="resume the exact 49-fit matrix")
    _add_runtime_paths_v2(matrix)
    matrix.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MATRIX_WORKERS_V2,
    )
    verify = commands.add_parser(
        "verify-complete",
        help="authenticate the grid and complete 49-fit matrix",
    )
    _add_runtime_paths_v2(verify)
    args = parser.parse_args(argv)
    common = {
        "repo_root": args.repo_root,
        "result_root": args.result_root,
        "data_dir": args.data_dir,
    }
    if args.command == "grid":
        summary = run_static_age_grid_v2(**common)
    elif args.command == "one-fit":
        summary = run_one_fit_v2(fit_id=args.fit_id, **common)
    elif args.command == "matrix":
        summary = run_fit_matrix_v2(max_workers=args.max_workers, **common)
    else:
        summary = verify_complete_matrix_v2(**common)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
