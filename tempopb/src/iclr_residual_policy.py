"""Nested static endowment anchors with a fold-local contextual residual."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from cohorts import AGE_BRACKETS, cohort_of
from iclr_env import EndowmentPolicy, RolloutState
from iclr_policy import instance_features
from parse_pb import PBInstance


AnchorFamily = Literal["mes", "senior", "age", "age_sex"]
AGE_CELLS = tuple(label for _, _, label in AGE_BRACKETS)
AGE_SEX_CELLS = tuple(
    f"{age}|{sex}"
    for age in AGE_CELLS
    for sex in ("F", "M")
)
AGE_REFERENCE = "age<25"
AGE_SEX_REFERENCE = "age<25|F"
FEATURE_NAMES = (
    "share_deviation",
    "ballot_length_dev",
    "cost_focus_dev",
    "overlap_dev",
)
_FREE_DIMENSIONS = {"mes": 0, "senior": 1, "age": 3, "age_sex": 7}
_REFERENCE_CELLS = {
    "mes": None,
    "senior": AGE_REFERENCE,
    "age": AGE_REFERENCE,
    "age_sex": AGE_SEX_REFERENCE,
}

__all__ = [
    "AGE_CELLS",
    "AGE_SEX_CELLS",
    "AGE_REFERENCE",
    "AGE_SEX_REFERENCE",
    "FEATURE_NAMES",
    "AnchorFamily",
    "AnchorSpec",
    "FeatureScaler",
    "anchor_logit_map",
    "anchor_multipliers",
    "anchor_policy",
    "fit_context_scaler",
    "standardized_context",
    "residual_policy",
]


def _checked_exp(logit: float) -> float:
    """Return a finite, strictly positive exponential or reject it clearly."""
    if not math.isfinite(logit):
        raise ValueError("logits must be finite")
    try:
        value = math.exp(logit)
    except OverflowError as exc:
        raise ValueError("logit produces a non-representable multiplier") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("logit produces a non-representable multiplier")
    return value


def _residual_log_factor(score: float) -> float:
    """The approved bounded residual in log space."""
    if not math.isfinite(score):
        raise ValueError("residual score must be finite")
    return math.log(2.0) * math.tanh(score)


def _logsumexp(log_values: np.ndarray) -> float:
    largest = float(log_values.max())
    total = math.fsum(
        math.exp(float(value) - largest)
        for value in log_values
    )
    return largest + math.log(total)


def _final_shares_from_logs(log_values: np.ndarray, budget: float) -> list[float]:
    """Normalize finite log factors into finite positive budget shares."""
    values = np.asarray(log_values, dtype=float)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("finite and positive budget is required")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("positive budget and at least one voter are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("log endowment multipliers must be finite")

    if np.all(values == values[0]):
        shares = np.full(values.size, budget / values.size)
        dominant = 0
    else:
        log_total = _logsumexp(values)
        log_budget_scale = math.log(budget) - log_total
        log_shares = values + log_budget_scale
        if not np.all(np.isfinite(log_shares)):
            raise ValueError("normalized log endowments are not representable")
        shares = np.asarray([_checked_exp(float(value)) for value in log_shares])
        dominant = int(np.argmax(log_shares))
    shares[dominant] = budget - math.fsum(
        float(value) for index, value in enumerate(shares) if index != dominant
    )
    if not np.all(np.isfinite(shares)) or np.any(shares <= 0.0):
        raise ValueError("positive endowments are not representable for this budget")
    tolerance = 1e-10 * max(1.0, budget)
    if abs(math.fsum(float(value) for value in shares) - budget) > tolerance:
        raise ValueError("normalization failed to conserve the budget")
    return shares.tolist()


@dataclass(frozen=True)
class AnchorSpec:
    """A canonical static endowment anchor in a nested policy family."""

    family: AnchorFamily
    free_logits: Sequence[float]
    reference_cell: str | None
    alpha: float | None = field(init=False)

    def __post_init__(self) -> None:
        if self.family not in _FREE_DIMENSIONS:
            raise ValueError(f"unknown anchor family: {self.family!r}")
        logits = np.asarray(self.free_logits, dtype=float)
        expected = _FREE_DIMENSIONS[self.family]
        if logits.shape != (expected,):
            raise ValueError(
                f"{self.family} requires {expected} free logits, got {logits.shape}"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError("free logits must be finite")
        if self.reference_cell != _REFERENCE_CELLS[self.family]:
            raise ValueError(
                f"{self.family} reference cell must be "
                f"{_REFERENCE_CELLS[self.family]!r}"
            )

        values = tuple(float(value) for value in logits)
        for value in values:
            _checked_exp(value)
        object.__setattr__(self, "free_logits", values)
        if self.family == "senior":
            try:
                alpha = math.expm1(values[0])
            except OverflowError as exc:
                raise ValueError(
                    "senior free logit must produce a positive multiplier"
                ) from exc
            if not math.isfinite(alpha) or alpha <= -1.0:
                raise ValueError("senior free logit must produce a positive multiplier")
            object.__setattr__(self, "alpha", alpha)
        else:
            object.__setattr__(self, "alpha", None)


@dataclass(frozen=True)
class FeatureScaler:
    """Population standardization fitted from training election-cohort rows."""

    feature_names: Sequence[str]
    mean: Sequence[float]
    scale: Sequence[float]
    clip: float
    row_count: int = 0
    instance_count: int = 0
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(self.feature_names)
        mean = np.array(self.mean, dtype=float, copy=True)
        scale = np.array(self.scale, dtype=float, copy=True)
        clip = float(self.clip)
        if names != FEATURE_NAMES:
            raise ValueError(f"feature names must be {FEATURE_NAMES!r}")
        if mean.shape != (len(FEATURE_NAMES),) or scale.shape != mean.shape:
            raise ValueError("mean and scale must have one value per static feature")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("scaler values must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("scaler scales must be positive")
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("clip must be finite and positive")
        if self.row_count < 0 or self.instance_count < 0:
            raise ValueError("scaler counts must be nonnegative")

        mean = np.frombuffer(mean.tobytes(), dtype=float)
        scale = np.frombuffer(scale.tobytes(), dtype=float)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "clip", clip)
        object.__setattr__(self, "sha256", _canonical_scaler_sha256(self))


def _canonical_scaler_sha256(scaler: FeatureScaler) -> str:
    payload = {
        "clip": scaler.clip,
        "feature_names": list(scaler.feature_names),
        "instance_count": scaler.instance_count,
        "mean": scaler.mean.tolist(),
        "row_count": scaler.row_count,
        "scale": scaler.scale.tolist(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def anchor_logit_map(anchor: AnchorSpec) -> dict[str, float]:
    """Expand an anchor's fitted free logits into its full demographic map."""
    if anchor.family == "mes":
        return {}
    if anchor.family == "senior":
        return {
            cell: (anchor.free_logits[0] if cell == "age60+" else 0.0)
            for cell in AGE_CELLS
        }

    cells = AGE_CELLS if anchor.family == "age" else AGE_SEX_CELLS
    reference = _REFERENCE_CELLS[anchor.family]
    free_cells = tuple(cell for cell in cells if cell != reference)
    return {
        cell: 0.0 if cell == reference else anchor.free_logits[free_cells.index(cell)]
        for cell in cells
    }


def anchor_multipliers(inst: PBInstance, anchor: AnchorSpec) -> np.ndarray:
    """Return unnormalized per-voter static factors for an anchor."""
    return np.asarray([_checked_exp(value) for value in _anchor_log_factors(inst, anchor)])


def _anchor_log_factors(inst: PBInstance, anchor: AnchorSpec) -> np.ndarray:
    """Return unnormalized static factors in log space."""
    if anchor.family == "mes":
        return np.zeros(len(inst.votes), dtype=float)

    logits = anchor_logit_map(anchor)
    if anchor.family in ("senior", "age"):
        cells = (
            cohort.split("|", 1)[0] if cohort is not None else None
            for cohort in (cohort_of(vote, "age_sex") for vote in inst.votes)
        )
    else:
        cells = (cohort_of(vote, "age_sex") for vote in inst.votes)
    return np.asarray([logits.get(cell, 0.0) for cell in cells], dtype=float)


def _budget_normalize(raw: np.ndarray, budget: float) -> list[float]:
    values = np.asarray(raw, dtype=float)
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("finite and positive budget is required")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("positive budget and at least one voter are required")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("raw endowment multipliers must be finite and positive")
    return _final_shares_from_logs(np.log(values), budget)


def anchor_policy(anchor: AnchorSpec) -> EndowmentPolicy:
    """Build the selected static anchor as an endowment policy."""
    def _policy(inst: PBInstance, state: RolloutState) -> list[float]:
        del state
        if not inst.votes:
            return []
        return _final_shares_from_logs(_anchor_log_factors(inst, anchor), inst.budget)

    return _policy


def _content_keyed_feature_view(inst: PBInstance) -> tuple[str, PBInstance]:
    """Snapshot every static-feature input behind a deterministic cache key."""
    projects = dict(inst.projects)
    votes = tuple(inst.votes)
    payload = {
        "path": inst.path,
        "projects": [
            [pid, projects[pid].cost]
            for pid in sorted(projects)
        ],
        "votes": [
            [vote.vid, list(vote.projects), vote.age, vote.sex]
            for vote in votes
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return digest, PBInstance(
        path=f"residual-static://{digest}",
        meta=dict(inst.meta),
        projects=projects,
        votes=list(votes),
    )


def _static_features(inst: PBInstance):
    _, view = _content_keyed_feature_view(inst)
    return instance_features(view, scheme="age_sex")


def fit_context_scaler(instances: Sequence[PBInstance]) -> FeatureScaler:
    """Fit a population scaler from one row per observed election cohort."""
    records = []
    for inst in instances:
        digest, view = _content_keyed_feature_view(inst)
        records.append((digest, instance_features(view, scheme="age_sex")))
    rows: list[list[float]] = []
    for _, features in sorted(records, key=lambda record: record[0]):
        cohort_index = features.cohort_index
        for cohort in AGE_SEX_CELLS:
            index = cohort_index.get(cohort)
            if index is None:
                continue
            rows.append([features.static[name][index] for name in FEATURE_NAMES])
    if not rows:
        raise ValueError("at least one observed age-sex cohort is required")

    matrix = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("source feature rows must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = matrix.mean(axis=0)
    if not np.all(np.isfinite(mean)):
        raise ValueError("fitted feature means must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        scale = matrix.std(axis=0, ddof=0)
    scale[(scale == 0.0) | ~np.isfinite(scale)] = 1.0
    return FeatureScaler(
        FEATURE_NAMES,
        mean,
        scale,
        clip=3.0,
        row_count=len(rows),
        instance_count=len(instances),
    )


def standardized_context(inst: PBInstance, scaler: FeatureScaler) -> dict[str, np.ndarray]:
    """Return clipped standardized static feature vectors by observed cohort."""
    features = _static_features(inst)
    context: dict[str, np.ndarray] = {}
    for index, cohort in enumerate(features.cohort_order):
        transformed = []
        for feature_index, name in enumerate(scaler.feature_names):
            value = float(features.static[name][index])
            if not math.isfinite(value):
                raise ValueError(f"non-finite source feature {name} for cohort {cohort}")
            standardized = (value - float(scaler.mean[feature_index])) / float(
                scaler.scale[feature_index]
            )
            if not math.isfinite(standardized):
                raise ValueError(
                    f"non-finite standardized feature {name} for cohort {cohort}"
                )
            transformed.append(standardized)
        context[cohort] = np.clip(
            np.asarray(transformed), -scaler.clip, scaler.clip
        )
    return context


def residual_policy(
    anchor: AnchorSpec, scaler: FeatureScaler, weights: Sequence[float]
) -> EndowmentPolicy:
    """Build a bounded contextual residual on top of a static anchor."""
    values = np.array(weights, dtype=float, copy=True)
    if values.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"weights must have shape ({len(FEATURE_NAMES)},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite")
    values.setflags(write=False)

    def _policy(inst: PBInstance, state: RolloutState) -> list[float]:
        del state
        if not inst.votes:
            return []
        z_by_cohort = standardized_context(inst, scaler)
        residual_logs = np.zeros(len(inst.votes), dtype=float)
        for index, vote in enumerate(inst.votes):
            cohort = cohort_of(vote, "age_sex")
            if cohort in z_by_cohort:
                score = float(np.dot(values, z_by_cohort[cohort]))
                residual_logs[index] = _residual_log_factor(score)
        return _final_shares_from_logs(
            _anchor_log_factors(inst, anchor) + residual_logs, inst.budget
        )

    return _policy
