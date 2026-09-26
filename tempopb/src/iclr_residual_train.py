"""Fit and select a deterministic, training-only static endowment anchor."""

from __future__ import annotations

import argparse
import ctypes
import copy
import csv
from collections.abc import Iterator
from collections import defaultdict
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from fnmatch import fnmatchcase
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
from numbers import Real
import re
import resource
import secrets
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

import numpy as np

from iclr_cmaes import CMAESConfig
from iclr_corpus import (
    CorpusConfig,
    SeriesRef,
    build_series_index,
    leave_district_out,
    load_series,
    load_split,
)
from iclr_env import EnvConfig, EpisodeResult, endowment_selector, rollout_selector
from iclr_history_free_audit import minimize_batch
from iclr_residual_policy import (
    AGE_SEX_CELLS,
    AGE_REFERENCE,
    AGE_SEX_REFERENCE,
    FEATURE_NAMES,
    AnchorSpec,
    FeatureScaler,
    anchor_logit_map,
    anchor_policy,
    fit_context_scaler,
    residual_policy,
)
from iclr_train import SeriesData
from iclr_wide_senior_audit import alpha_grid
from parse_pb import PBInstance, Project, Vote


ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = ROOT / "results" / "iclr_residual_upgrade" / "smoke"
SMOKE_ARTIFACT = SMOKE_ROOT / "static-anchor.json"
SMOKE_RESIDUAL_ARTIFACT = SMOKE_ROOT / "residual-fit.json"
SMOKE_DEVELOPMENT_ARTIFACT = SMOKE_ROOT / "development.json"
DEVELOPMENT_ROOT = ROOT / "results" / "iclr_residual_upgrade" / "development"
DEVELOPMENT_ANALYSIS_ROOT = (
    ROOT / "analysis-output" / "iclr_residual_upgrade" / "development"
)
MULTICITY_RESULT_ROOT = ROOT / "results" / "iclr_multicity"
MULTICITY_DATA_DIR = ROOT / "data" / "pb_multicity"
DISTRICT_SPLIT_ROOT = ROOT / "results" / "iclr_splits"
DISTRICT_DATA_DIR = ROOT / "data" / "pb"
RAY_AMPLITUDES = (0.0, 0.25, 0.5, 0.75, 1.0)

_CITY_DEVELOPMENT_FOLDS = (
    "city_out_Poland_Gdynia",
    "city_out_Poland_Warszawa",
    "city_out_Poland_Łódź",
)
_DISTRICT_DEVELOPMENT_FOLDS = tuple(
    f"district_out_f{fold}of5" for fold in range(5)
)
_DEVELOPMENT_SEEDS = (1, 2, 42)
_DEVELOPMENT_METRICS = ("csd", "welfare", "exclusion")


class _ImmutableJSONDict(dict):
    """Defensive immutable dict that remains ordinary-JSON and pickle safe."""

    def __init__(self, values=(), **kwargs) -> None:
        copied = {}
        for key, value in dict(values, **kwargs).items():
            copied[str(key)] = _freeze_development_value(value)
        dict.__init__(self, copied)

    def _immutable(self, *args, **kwargs) -> None:
        del args, kwargs
        raise TypeError("immutable development payload cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __reduce__(self):
        return type(self), (dict(self),)


def _freeze_development_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _ImmutableJSONDict(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_development_value(item) for item in value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("development payload values must be finite")
        return value
    raise TypeError(f"unsupported development payload value: {type(value).__name__}")


def _sha256_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 identity")
    return value


class _ImmutableSnapshotDict(dict):
    """Immutable pickle/JSON-safe mapping that preserves integer year keys."""

    def __init__(self, values=()) -> None:
        dict.__init__(self, values)

    def _immutable(self, *args, **kwargs) -> None:
        del args, kwargs
        raise TypeError("immutable development snapshot cannot be modified")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __reduce__(self):
        return type(self), (dict(self),)


@dataclass(frozen=True)
class _FrozenPBInstance:
    path: str
    meta: Mapping[str, str]
    projects: Mapping[str, Project]
    votes: tuple[Vote, ...]


@dataclass(frozen=True)
class _FrozenSeriesData:
    ref: SeriesRef
    train_years: tuple[int, ...]
    test_years: tuple[int, ...]
    train_only: Mapping[int, _FrozenPBInstance]
    all_years: Mapping[int, _FrozenPBInstance]


def _canonical_persisted_path(
    value: object, *, approved_prefixes: Sequence[str]
) -> str:
    """Return a non-private logical POSIX path for a persisted identity."""

    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("persisted path must be a nonempty canonical POSIX path")
    if value.startswith(("synthetic://", "smoke-training://", "smoke-development://")):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve(strict=False).relative_to(ROOT.resolve())
        except (OSError, ValueError) as exc:
            raise ValueError("persisted path is outside the approved repository root") from exc
    posix = candidate.as_posix()
    if (
        posix.startswith("/")
        or any(part in {"", ".", ".."} for part in posix.split("/"))
        or not any(posix.startswith(prefix) for prefix in approved_prefixes)
    ):
        raise ValueError("persisted path is outside an approved POSIX root")
    return posix


def _snapshot_development_instance(instance: PBInstance) -> _FrozenPBInstance:
    copied = copy.deepcopy(instance)
    return _FrozenPBInstance(
        path=_canonical_persisted_path(
            str(copied.path),
            approved_prefixes=("data/pb/", "data/pb_multicity/"),
        ),
        meta=_ImmutableJSONDict(copied.meta),
        projects=_ImmutableSnapshotDict(
            (str(project), copy.deepcopy(value))
            for project, value in sorted(copied.projects.items())
        ),
        votes=tuple(copy.deepcopy(copied.votes)),
    )


def _development_year_tuple(
    values: object, name: str, *, require_nonempty: bool = False
) -> tuple[int, ...]:
    try:
        years = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} years must be an iterable of integers") from exc
    if any(type(year) is not int for year in years):
        raise TypeError(f"{name} years must contain only integers")
    if len(years) != len(set(years)):
        raise ValueError(f"{name} years must be unique")
    if any(current >= following for current, following in zip(years, years[1:])):
        raise ValueError(f"{name} years must be strictly increasing")
    if require_nonempty and not years:
        raise ValueError(f"{name} years must be nonempty")
    return years


def _development_instance_years(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} year map must be a mapping")
    years = tuple(values)
    if any(type(year) is not int for year in years):
        raise TypeError(f"{name} year map keys must be integers")
    if any(not isinstance(values[year], PBInstance) for year in years):
        raise TypeError(f"{name} year map values must be PBInstance objects")
    return years


def _snapshot_development_series(
    row: SeriesData, *, role: str
) -> _FrozenSeriesData:
    if not isinstance(row, SeriesData):
        raise TypeError("development fold views must contain SeriesData")
    if role not in {"training", "scored"}:
        raise ValueError("development series role must be training or scored")
    ref_years = _development_year_tuple(row.ref.years, "reference")
    train_years = _development_year_tuple(
        row.train_years, "training", require_nonempty=role == "training"
    )
    test_years = _development_year_tuple(
        row.test_years, "scored", require_nonempty=role == "scored"
    )
    train_instance_years = _development_instance_years(
        row.train_only, "training"
    )
    all_instance_years = _development_instance_years(row.all_years, "all")
    if role == "training":
        if test_years:
            raise ValueError("training development rows cannot declare scored years")
        expected = set(train_years)
        if (
            set(ref_years) != expected
            or set(train_instance_years) != expected
            or set(all_instance_years) != expected
        ):
            raise ValueError(
                "training years, reference years, train_only, and all_years must agree"
            )
    else:
        if train_years or train_instance_years:
            raise ValueError("scored development rows cannot contain training data")
        if set(all_instance_years) != set(ref_years):
            raise ValueError("scored all_years must exactly match reference years")
        prefix_length = len(ref_years) - len(test_years)
        if prefix_length < 0 or ref_years[prefix_length:] != test_years:
            raise ValueError("scored years must be the terminal reference-year suffix")
        warmup_years = ref_years[:prefix_length]
        if any(year >= test_years[0] for year in warmup_years):
            raise ValueError("scored warm-up years must strictly precede scored years")
    instances = {
        year: _snapshot_development_instance(instance)
        for year, instance in sorted(row.all_years.items())
    }
    training = {
        year: instances[year]
        for year in sorted(row.train_only)
        if year in instances
    }
    if set(training) != set(row.train_only):
        raise ValueError("development training instances must occur in all_years")
    return _FrozenSeriesData(
        ref=SeriesRef(row.ref.key, ref_years, ()),
        train_years=train_years,
        test_years=test_years,
        train_only=_ImmutableSnapshotDict(training),
        all_years=_ImmutableSnapshotDict(instances),
    )


def _materialize_development_instance(instance: _FrozenPBInstance) -> PBInstance:
    return PBInstance(
        path=instance.path,
        meta=dict(instance.meta),
        projects=copy.deepcopy(dict(instance.projects)),
        votes=copy.deepcopy(list(instance.votes)),
    )


def _materialize_development_series(row: _FrozenSeriesData) -> SeriesData:
    instances = {
        year: _materialize_development_instance(instance)
        for year, instance in row.all_years.items()
    }
    return SeriesData(
        ref=SeriesRef(row.ref.key, tuple(row.ref.years), ()),
        train_years=tuple(row.train_years),
        test_years=tuple(row.test_years),
        train_only={year: instances[year] for year in row.train_only},
        all_years=instances,
    )


@dataclass(frozen=True)
class DevelopmentFold:
    """One normalized outer fold with disjoint fit and scored series views."""

    name: str
    kind: str
    train: Sequence[SeriesData]
    test: Sequence[SeriesData]
    source_manifest_sha256: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("development fold name must be a nonempty string")
        if self.kind not in {"city", "district"}:
            raise ValueError("development fold kind must be city or district")
        train = tuple(
            _snapshot_development_series(row, role="training") for row in self.train
        )
        test = tuple(
            _snapshot_development_series(row, role="scored") for row in self.test
        )
        train_keys = tuple(row.ref.key for row in train)
        test_keys = tuple(row.ref.key for row in test)
        if len(train_keys) != len(set(train_keys)) or len(test_keys) != len(set(test_keys)):
            raise ValueError("development fold series must be unique in each view")
        if set(train_keys) & set(test_keys):
            raise ValueError("development fold train and test series must be disjoint")
        identity = _sha256_identity(
            self.source_manifest_sha256, "source manifest"
        )
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)
        object.__setattr__(self, "source_manifest_sha256", identity)
        object.__setattr__(self, "provenance", _ImmutableJSONDict(self.provenance))


def _metric_snapshot(values: Mapping[str, object], name: str) -> _ImmutableJSONDict:
    if not isinstance(values, Mapping) or set(values) != set(_DEVELOPMENT_METRICS):
        raise ValueError(
            f"{name} development metrics must contain exactly {_DEVELOPMENT_METRICS!r}"
        )
    normalized = {}
    for key in _DEVELOPMENT_METRICS:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}.{key} must be a finite real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}.{key} must be finite")
        normalized[key] = 0.0 if number == 0.0 else number
    return _ImmutableJSONDict(normalized)


@dataclass(frozen=True)
class DevelopmentSeriesResult:
    """Immutable held-out metrics for one fold, seed, and series."""

    fold: str
    seed: int
    series: str
    city: str
    cluster: str
    residual: Mapping[str, object]
    anchor: Mapping[str, object]
    mes: Mapping[str, object]
    signatures: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in ("fold", "series", "city", "cluster"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a nonempty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("development seed must be an integer")
        expected_city = "/".join(self.series.split("/")[:2])
        if self.city != expected_city:
            raise ValueError("development city contradicts series identifier")
        if self.cluster != development_cluster_key(self.series):
            raise ValueError("development cluster is not canonical")
        signature_keys = {"residual", "anchor", "actuated"}
        if not isinstance(self.signatures, Mapping) or set(self.signatures) != signature_keys:
            raise ValueError("development signatures must have the exact writer schema")
        signatures = _ImmutableJSONDict(self.signatures)
        if type(signatures["actuated"]) is not bool:
            raise TypeError("development actuation signature must be a bool")
        expected_residual = (
            "actuated" if signatures["actuated"] else "anchor-equivalent"
        )
        if (
            signatures["anchor"] != "anchor-equivalent"
            or signatures["residual"] != expected_residual
        ):
            raise ValueError("development signatures contradict actuation state")
        object.__setattr__(self, "residual", _metric_snapshot(self.residual, "residual"))
        object.__setattr__(self, "anchor", _metric_snapshot(self.anchor, "anchor"))
        object.__setattr__(self, "mes", _metric_snapshot(self.mes, "mes"))
        object.__setattr__(self, "signatures", signatures)


@dataclass(frozen=True)
class CityMetrics:
    """One macro observation per training city."""

    city_csd: float
    city_welfare: float
    city_exclusion: float


class _ImmutableMetrics(Mapping[str, CityMetrics]):
    """A defensive, pickle-safe immutable snapshot of city metrics."""

    __slots__ = ("_items",)

    def __init__(self, metrics: Mapping[str, CityMetrics]) -> None:
        object.__setattr__(self, "_items", tuple(sorted(metrics.items())))

    def __getitem__(self, key: str) -> CityMetrics:
        for city, metrics in self._items:
            if city == key:
                return metrics
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (city for city, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("immutable metrics cannot be modified")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("immutable metrics cannot be modified")

    def __reduce__(self):
        return type(self), (dict(self._items),)


@dataclass(frozen=True)
class AnchorCandidate:
    """One canonically scored static anchor."""

    anchor: AnchorSpec
    objective: float
    metrics: Mapping[str, CityMetrics]
    safe: bool
    source: str
    seed: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _ImmutableMetrics(self.metrics))


@dataclass(frozen=True)
class AnchorSearchConfig:
    """All search and safety constants for static-anchor fitting."""

    seeds: tuple[int, ...]
    generations: int
    popsize: int
    sigma0: float
    bound: float
    tau: float
    welfare_floor: float
    exclusion_delta: float
    tie_tolerance: float


@dataclass(frozen=True)
class AnchorFit:
    """Selected anchor and its complete deduplicated candidate archive."""

    selected: AnchorCandidate
    candidates: tuple[AnchorCandidate, ...]
    mes_metrics: Mapping[str, CityMetrics]
    config: AnchorSearchConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "mes_metrics", _ImmutableMetrics(self.mes_metrics))


@dataclass(frozen=True)
class SeriesMetrics:
    """Raw training metrics for one series under one residual policy."""

    series: str
    city: str
    csd: float | None
    welfare: float
    exclusion: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "series", str(self.series))
        object.__setattr__(self, "city", str(self.city))
        object.__setattr__(
            self, "csd", None if self.csd is None else float(self.csd)
        )
        object.__setattr__(self, "welfare", float(self.welfare))
        object.__setattr__(self, "exclusion", float(self.exclusion))


def _series_metric_failures(rows: Sequence[SeriesMetrics]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        if row.csd is None:
            failures.append(f"{row.series}: missing CSD")
        elif not math.isfinite(row.csd):
            failures.append(f"{row.series}: non-finite CSD")
        if not math.isfinite(row.welfare):
            failures.append(f"{row.series}: non-finite welfare")
        if not math.isfinite(row.exclusion):
            failures.append(f"{row.series}: non-finite exclusion")
    return tuple(failures)


def _canonical_city_metrics(
    rows: Sequence[SeriesMetrics],
) -> dict[str, CityMetrics]:
    grouped: dict[str, list[SeriesMetrics]] = defaultdict(list)
    for row in rows:
        parts = row.series.split("/")
        expected_city = "/".join(parts[:2]) if len(parts) >= 2 else row.series
        if row.city != expected_city:
            raise ValueError(f"series city does not match series ID: {row.series}")
        values = (row.csd, row.welfare, row.exclusion)
        if row.csd is None or not all(
            math.isfinite(float(value)) for value in values if value is not None
        ):
            continue
        grouped[row.city].append(row)

    result: dict[str, CityMetrics] = {}
    for city, city_rows in sorted(grouped.items()):
        result[city] = CityMetrics(
            city_csd=math.fsum(float(row.csd) for row in city_rows) / len(city_rows),
            city_welfare=math.fsum(row.welfare for row in city_rows),
            city_exclusion=(
                math.fsum(row.exclusion for row in city_rows) / len(city_rows)
            ),
        )
    return result


@dataclass(frozen=True)
class ResidualMetrics:
    """Immutable raw series/city metrics and explicit safety failures."""

    series: Sequence[SeriesMetrics]
    cities: Mapping[str, CityMetrics]
    failed_safety: Sequence[str]

    def __post_init__(self) -> None:
        source_rows = tuple(self.series)
        series_ids = tuple(row.series for row in source_rows)
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("duplicate residual series IDs are forbidden")
        rows = tuple(sorted(source_rows, key=lambda row: row.series))
        canonical_cities = _canonical_city_metrics(rows)
        supplied_cities = dict(self.cities)
        if supplied_cities != canonical_cities:
            raise ValueError("city aggregates do not match raw series evidence")
        failures: list[str] = []
        for failure in (*_series_metric_failures(rows), *tuple(self.failed_safety)):
            text = str(failure)
            if text not in failures:
                failures.append(text)
        object.__setattr__(self, "series", rows)
        object.__setattr__(self, "cities", _ImmutableMetrics(canonical_cities))
        object.__setattr__(self, "failed_safety", tuple(failures))


def _guard_optimizer_history_dtype(value: np.ndarray | np.generic) -> None:
    dtype = value.dtype
    if dtype.fields is not None or dtype.kind in {"V", "M", "m"}:
        raise TypeError(f"unsupported optimizer history value: {dtype}")


def _freeze_history_value(value: object) -> object:
    if isinstance(value, (np.ndarray, np.generic)):
        _guard_optimizer_history_dtype(value)
    if isinstance(value, np.ndarray):
        return _freeze_history_value(value.tolist())
    if isinstance(value, np.bool_):
        return _freeze_history_value(bool(value))
    if isinstance(value, np.integer):
        return _freeze_history_value(int(value))
    if isinstance(value, np.floating):
        return _freeze_history_value(float(value))
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, np.generic):
            raise TypeError(
                f"unsupported optimizer history value: {type(value).__name__}"
            )
        return _freeze_history_value(item)
    if isinstance(value, Mapping):
        return _ImmutableRecord(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_history_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("finite optimizer history values are required")
        return value
    raise TypeError(
        f"unsupported optimizer history value: {type(value).__name__}"
    )


class _ImmutableRecord(Mapping[str, object]):
    """Small recursively immutable, pickle-safe optimizer record."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, object]) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple(
                (str(key), _freeze_history_value(value))
                for key, value in sorted(values.items())
            ),
        )

    def __getitem__(self, key: str) -> object:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("immutable optimizer history cannot be modified")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("immutable optimizer history cannot be modified")

    def __reduce__(self):
        return type(self), (dict(self._items),)


@dataclass(frozen=True)
class ResidualCandidate:
    """One fully evaluated point on a globally scaled residual ray."""

    weights: Sequence[float]
    endpoint: Sequence[float]
    amplitude: float
    loss: float
    metrics: ResidualMetrics
    safe: bool
    source: str

    def __post_init__(self) -> None:
        weights = tuple(
            0.0 if float(value) == 0.0 else float(value) for value in self.weights
        )
        endpoint = tuple(
            0.0 if float(value) == 0.0 else float(value) for value in self.endpoint
        )
        if len(weights) != len(FEATURE_NAMES) or len(endpoint) != len(FEATURE_NAMES):
            raise ValueError(
                f"residual vectors must have shape ({len(FEATURE_NAMES)},)"
            )
        if not all(math.isfinite(value) for value in (*weights, *endpoint)):
            raise ValueError("residual vectors must be finite")
        amplitude = float(self.amplitude)
        if amplitude not in RAY_AMPLITUDES:
            raise ValueError(f"amplitude must be one of {RAY_AMPLITUDES!r}")
        if amplitude == 0.0:
            amplitude = 0.0
            if any(weights):
                raise ValueError("weights must equal amplitude times endpoint")
            weights = tuple(0.0 for _ in FEATURE_NAMES)
            endpoint = tuple(0.0 for _ in FEATURE_NAMES)
        else:
            expected = tuple(
                0.0 if amplitude * value == 0.0 else amplitude * value
                for value in endpoint
            )
            if weights != expected:
                raise ValueError("weights must equal amplitude times endpoint")
        metrics = ResidualMetrics(
            self.metrics.series,
            self.metrics.cities,
            self.metrics.failed_safety,
        )
        loss = float(self.loss)
        safe_state = bool(self.safe)
        consistent = (
            safe_state
            and not metrics.failed_safety
            and math.isfinite(loss)
        ) or (
            not safe_state
            and bool(metrics.failed_safety)
            and math.isinf(loss)
            and loss > 0.0
        )
        if not consistent:
            raise ValueError("candidate safety state is inconsistent")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "safe", safe_state)


def _is_canonical_zero(candidate: ResidualCandidate) -> bool:
    values = (*candidate.weights, *candidate.endpoint)
    return candidate.amplitude == 0.0 and all(
        value == 0.0 and math.copysign(1.0, value) == 1.0 for value in values
    )


def _has_nonzero_policy_weights(candidate: ResidualCandidate) -> bool:
    return any(value != 0.0 for value in candidate.weights)


@dataclass(frozen=True)
class ResidualSeedFit:
    """One seed's selected residual and complete deterministic archive."""

    seed: int
    selected: ResidualCandidate
    archive: Sequence[ResidualCandidate]
    optimizer_history: Sequence[Mapping[str, object]]

    def __post_init__(self) -> None:
        archive = tuple(self.archive)
        if not archive:
            raise ValueError("nonempty residual archive is required")
        payload_hashes = tuple(residual_candidate_sha256(row) for row in archive)
        if len(payload_hashes) != len(set(payload_hashes)):
            raise ValueError("residual archive must be payload-unique")
        zeros = [
            row
            for row in archive
            if row.safe
            and math.isfinite(row.loss)
            and _is_canonical_zero(row)
        ]
        if len(zeros) != 1:
            raise ValueError("residual archive must contain exactly one canonical safe zero")
        selected_hash = residual_candidate_sha256(self.selected)
        if selected_hash not in set(payload_hashes):
            raise ValueError("selected candidate must occur in archive")
        if not self.selected.safe or not math.isfinite(self.selected.loss):
            raise ValueError("selected candidate must be safe with finite loss")
        if (
            not _has_nonzero_policy_weights(self.selected)
            and not _is_canonical_zero(self.selected)
        ):
            raise ValueError("selected zero-weight candidate must be canonical")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "archive", archive)
        object.__setattr__(
            self,
            "optimizer_history",
            tuple(_ImmutableRecord(record) for record in self.optimizer_history),
        )


@dataclass(frozen=True)
class ResidualSearchConfig:
    """Frozen optimizer and residual-safety constants."""

    seeds: tuple[int, ...]
    generations: int
    popsize: int
    sigma0: float
    bound: float
    tau: float
    welfare_floor: float
    replacement_delta: float
    norm_penalty: float


@dataclass(frozen=True)
class FoldFit:
    """Static anchor, scaler, and all residual seed fits for one fold."""

    anchor_fit: AnchorFit
    scaler: FeatureScaler
    seeds: Sequence[ResidualSeedFit]
    primary_seed: int
    config: ResidualSearchConfig

    def __post_init__(self) -> None:
        seed_fits = tuple(self.seeds)
        if tuple(row.seed for row in seed_fits) != tuple(self.config.seeds):
            raise ValueError("residual seed order must equal configured seeds")
        expected_primary = select_primary_seed(seed_fits)
        if self.primary_seed != expected_primary:
            raise ValueError("primary seed must equal training selection")
        object.__setattr__(self, "seeds", seed_fits)


_PRODUCTION_RESIDUAL_CONFIG = ResidualSearchConfig(
    seeds=(1, 2, 42),
    generations=40,
    popsize=8,
    sigma0=0.4,
    bound=3.0,
    tau=0.02,
    welfare_floor=0.99,
    replacement_delta=0.002,
    norm_penalty=1e-3,
)


def residual_search_config() -> ResidualSearchConfig:
    """Return the fixed production residual search configuration."""

    return _PRODUCTION_RESIDUAL_CONFIG


def smoke_residual_search_config() -> ResidualSearchConfig:
    """Return the reduced synthetic-smoke configuration, never production CLI input."""

    return replace(_PRODUCTION_RESIDUAL_CONFIG, generations=1, popsize=4)


def anchor_search_config() -> AnchorSearchConfig:
    """Return the frozen production search; the CLI exposes no overrides."""

    return AnchorSearchConfig(
        seeds=(1, 2, 42),
        generations=60,
        popsize=12,
        sigma0=0.4,
        bound=6.0,
        tau=0.02,
        welfare_floor=0.99,
        exclusion_delta=0.0,
        tie_tolerance=1e-4,
    )


def smoke_anchor_search_config() -> AnchorSearchConfig:
    """Return a reduced config used only by tests and ``smoke-anchor``."""

    return replace(anchor_search_config(), seeds=(1,), generations=1, popsize=4)


def soft_worst_city(values: Sequence[float], tau: float = 0.02) -> float:
    """Stable log-mean-exp approximation to the worst city CSD."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("at least one city value is required")
    if not np.all(np.isfinite(array)):
        raise ValueError("city values must be finite")
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    peak = float(array.max())
    return peak + tau * math.log(
        float(np.exp((array - peak) / tau).mean())
    )


def residual_training_loss(
    candidate_metrics: ResidualMetrics,
    anchor_metrics: ResidualMetrics,
    weights: Sequence[float],
) -> float:
    """City-balanced training excess CSD plus the exact squared-norm penalty."""

    values = np.asarray(weights, dtype=float)
    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise ValueError(f"weights must be finite with shape ({len(FEATURE_NAMES)},)")
    candidate_by_series = {row.series: row for row in candidate_metrics.series}
    anchor_by_series = {row.series: row for row in anchor_metrics.series}
    if not candidate_by_series or set(candidate_by_series) != set(anchor_by_series):
        raise ValueError("candidate and anchor must contain the same training series")

    grouped: dict[str, list[float]] = defaultdict(list)
    for series in sorted(candidate_by_series):
        candidate = candidate_by_series[series]
        anchor = anchor_by_series[series]
        if candidate.city != anchor.city:
            raise ValueError(f"training city mismatch for series {series!r}")
        if candidate.csd is None or anchor.csd is None:
            raise ValueError(f"missing training CSD for series {series!r}")
        if not math.isfinite(float(candidate.csd)) or not math.isfinite(float(anchor.csd)):
            raise ValueError(f"non-finite training CSD for series {series!r}")
        grouped[candidate.city].append(float(candidate.csd) - float(anchor.csd))

    city_excess = tuple(
        math.fsum(grouped[city]) / len(grouped[city]) for city in sorted(grouped)
    )
    loss = soft_worst_city(city_excess, tau=0.02)
    loss += 1e-3 * float(np.dot(values, values))
    return float(loss)


def _residual_metrics_from_episodes(
    episodes: Sequence[EpisodeResult],
) -> ResidualMetrics:
    rows = tuple(
        SeriesMetrics(
            series=episode.series,
            city=_city(episode.series),
            csd=(None if episode.worst_csd is None else float(episode.worst_csd)),
            welfare=float(episode.welfare),
            exclusion=float(episode.exclusion),
        )
        for episode in episodes
    )
    return ResidualMetrics(rows, summarize_city_metrics(episodes), ())


def _append_failure(failures: list[str], condition: str) -> None:
    if condition not in failures:
        failures.append(condition)


def score_residual_candidate(
    weights: Sequence[float],
    endpoint: Sequence[float],
    amplitude: float,
    candidate_metrics: ResidualMetrics,
    anchor_metrics: ResidualMetrics,
    mes_metrics: Mapping[str, CityMetrics],
    source: str,
) -> ResidualCandidate:
    """Apply dual-reference city safety and score one training residual."""

    failures = list(candidate_metrics.failed_safety)
    for failure in anchor_metrics.failed_safety:
        _append_failure(failures, f"anchor reference: {failure}")
    candidate_cities = set(candidate_metrics.cities)
    anchor_cities = set(anchor_metrics.cities)
    mes_cities = set(mes_metrics)
    if not candidate_cities:
        _append_failure(failures, "no scoreable candidate cities")
    if candidate_cities != anchor_cities:
        _append_failure(failures, "candidate cities differ from anchor cities")
    if candidate_cities != mes_cities:
        _append_failure(failures, "candidate cities differ from MES cities")

    if candidate_cities == anchor_cities == mes_cities:
        for city in sorted(candidate_cities):
            candidate = candidate_metrics.cities[city]
            anchor = anchor_metrics.cities[city]
            mes = mes_metrics[city]
            references_finite = True
            for owner, metrics in (
                ("candidate", candidate),
                ("anchor", anchor),
                ("MES", mes),
            ):
                for label, value in (
                    ("csd", metrics.city_csd),
                    ("welfare", metrics.city_welfare),
                    ("exclusion", metrics.city_exclusion),
                ):
                    if not math.isfinite(float(value)):
                        _append_failure(
                            failures, f"{owner} {city}: non-finite {label}"
                        )
                        if owner != "candidate":
                            references_finite = False
            if not references_finite:
                continue
            for label, value in (
                ("candidate csd", candidate.city_csd),
                ("candidate welfare", candidate.city_welfare),
                ("candidate exclusion", candidate.city_exclusion),
            ):
                if not math.isfinite(float(value)):
                    _append_failure(failures, f"{city}: non-finite {label}")
            if candidate.city_welfare < 0.99 * anchor.city_welfare:
                _append_failure(failures, f"{city}: below 99% anchor welfare")
            if candidate.city_welfare < 0.99 * mes.city_welfare:
                _append_failure(failures, f"{city}: below 99% MES welfare")
            if candidate.city_exclusion > anchor.city_exclusion:
                _append_failure(failures, f"{city}: above anchor exclusion")
            if candidate.city_exclusion > mes.city_exclusion:
                _append_failure(failures, f"{city}: above MES exclusion")

    safe = not failures
    scored_metrics = ResidualMetrics(
        candidate_metrics.series,
        candidate_metrics.cities,
        tuple(failures),
    )
    try:
        raw_loss = residual_training_loss(scored_metrics, anchor_metrics, weights)
    except ValueError as exc:
        _append_failure(failures, str(exc))
        safe = False
        scored_metrics = ResidualMetrics(
            candidate_metrics.series,
            candidate_metrics.cities,
            tuple(failures),
        )
        raw_loss = float("inf")
    if not math.isfinite(raw_loss):
        _append_failure(failures, "non-finite residual training loss")
        safe = False
        scored_metrics = ResidualMetrics(
            candidate_metrics.series,
            candidate_metrics.cities,
            tuple(failures),
        )
    loss = raw_loss if safe and math.isfinite(raw_loss) else float("inf")
    return ResidualCandidate(
        weights=weights,
        endpoint=endpoint,
        amplitude=amplitude,
        loss=loss,
        metrics=scored_metrics,
        safe=safe,
        source=source,
    )


def evaluate_ray(
    endpoint: Sequence[float],
    evaluator: Callable[[Sequence[float]], ResidualCandidate],
) -> tuple[ResidualCandidate, ...]:
    """Expand one endpoint to the exact approved amplitudes in fixed order."""

    endpoint_values = tuple(float(value) for value in endpoint)
    if len(endpoint_values) != len(FEATURE_NAMES) or not all(
        math.isfinite(value) for value in endpoint_values
    ):
        raise ValueError(f"endpoint must be finite with shape ({len(FEATURE_NAMES)},)")
    rows: list[ResidualCandidate] = []
    for amplitude in RAY_AMPLITUDES:
        if amplitude == 0.0:
            weights = tuple(0.0 for _ in endpoint_values)
            canonical_endpoint = tuple(0.0 for _ in endpoint_values)
        else:
            canonical_endpoint = tuple(
                0.0 if value == 0.0 else value for value in endpoint_values
            )
            weights = tuple(
                0.0 if amplitude * value == 0.0 else amplitude * value
                for value in canonical_endpoint
            )
        rows.append(
            replace(
                evaluator(weights),
                weights=weights,
                endpoint=canonical_endpoint,
                amplitude=amplitude,
            )
        )
    return tuple(rows)


def _residual_norm(candidate: ResidualCandidate) -> float:
    return math.fsum(value * value for value in candidate.weights)


def _finite_safe_residuals(
    candidates: Sequence[ResidualCandidate],
) -> list[ResidualCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate.safe and math.isfinite(candidate.loss)
    ]


def select_ray_candidate(
    candidates: Sequence[ResidualCandidate],
) -> ResidualCandidate:
    """Choose a feasible ray point by loss, preferring smaller amplitude on ties."""

    safe = _finite_safe_residuals(candidates)
    if not safe:
        raise ValueError("ray contains no explicitly safe residual candidate")
    return min(
        safe,
        key=lambda candidate: (
            candidate.loss,
            candidate.amplitude,
            _residual_norm(candidate),
            residual_candidate_sha256(candidate),
        ),
    )


def select_residual_candidate(
    candidates: Sequence[ResidualCandidate],
    replacement_delta: float = 0.002,
) -> ResidualCandidate:
    """Apply the exact actuation threshold against an explicit safe zero."""

    safe = _finite_safe_residuals(candidates)
    zeros = [candidate for candidate in safe if _is_canonical_zero(candidate)]
    if not zeros:
        raise ValueError("residual archive has no explicitly safe exact zero")
    zero = min(zeros, key=residual_candidate_sha256)
    nonzero = [
        candidate for candidate in safe if _has_nonzero_policy_weights(candidate)
    ]
    if not nonzero:
        return zero
    best = min(
        nonzero,
        key=lambda candidate: (
            candidate.loss,
            candidate.amplitude,
            _residual_norm(candidate),
            residual_candidate_sha256(candidate),
        ),
    )
    return best if best.loss <= zero.loss - replacement_delta else zero


def _city(series: str) -> str:
    parts = series.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else series


def summarize_city_metrics(
    episodes: Sequence[EpisodeResult],
) -> dict[str, CityMetrics]:
    """Reduce scoreable series into one CSD, welfare, and exclusion per city."""

    grouped: dict[str, list[EpisodeResult]] = defaultdict(list)
    for episode in episodes:
        if episode.worst_csd is None:
            continue
        values = (episode.worst_csd, episode.welfare, episode.exclusion)
        if not all(math.isfinite(float(value)) for value in values):
            continue
        grouped[_city(episode.series)].append(episode)

    result: dict[str, CityMetrics] = {}
    for city, rows in sorted(grouped.items()):
        result[city] = CityMetrics(
            city_csd=math.fsum(float(row.worst_csd) for row in rows) / len(rows),
            city_welfare=math.fsum(float(row.welfare) for row in rows),
            city_exclusion=math.fsum(float(row.exclusion) for row in rows) / len(rows),
        )
    return result


def rollout_static_training(
    anchor: AnchorSpec,
    data: Sequence[SeriesData],
    env_cfg: EnvConfig | None = None,
) -> list[EpisodeResult]:
    """Roll out an anchor using only each series' fitting years and instances."""

    env_cfg = env_cfg or EnvConfig()
    selector = endowment_selector(anchor_policy(anchor), env_cfg)
    return [
        rollout_selector(
            row.ref,
            selector,
            score_years=row.train_years,
            cfg=env_cfg,
            instances=row.train_only,
        )
        for row in data
        if row.train_years
    ]


def score_static_candidate(
    anchor: AnchorSpec,
    data: Sequence[SeriesData],
    cfg: AnchorSearchConfig,
    mes_metrics: Mapping[str, CityMetrics] | None,
) -> AnchorCandidate:
    """Score one anchor once with the canonical city and safety evaluator."""

    episodes = rollout_static_training(anchor, data)
    metrics = summarize_city_metrics(episodes)
    has_invalid_episode = any(
        episode.worst_csd is None
        or not all(
            math.isfinite(float(value))
            for value in (
                episode.worst_csd,
                episode.welfare,
                episode.exclusion,
            )
        )
        for episode in episodes
    )
    has_unscored_training_row = any(row.train_years for row in data) and not episodes

    baseline = metrics if mes_metrics is None and anchor.family == "mes" else mes_metrics
    complete_cities = (
        baseline is not None
        and bool(metrics)
        and set(metrics) == set(baseline)
    )
    safe = bool(
        complete_cities
        and not has_invalid_episode
        and not has_unscored_training_row
    )
    if safe and baseline is not None:
        for city, city_metrics in metrics.items():
            mes = baseline[city]
            if (
                city_metrics.city_welfare < cfg.welfare_floor * mes.city_welfare
                or city_metrics.city_exclusion
                > mes.city_exclusion + cfg.exclusion_delta
            ):
                safe = False
                break

    objective = (
        soft_worst_city(
            [metrics[city].city_csd for city in sorted(metrics)], cfg.tau
        )
        if safe
        else float("inf")
    )
    return AnchorCandidate(
        anchor=anchor,
        objective=float(objective),
        metrics=metrics,
        safe=safe,
        source="canonical-score",
        seed=None,
    )


def _anchor_payload(anchor: AnchorSpec) -> dict[str, object]:
    return {
        "alpha": anchor.alpha,
        "family": anchor.family,
        "free_logits": list(anchor.free_logits),
        "reference_cell": anchor.reference_cell,
    }


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_number(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _series_metrics_payload(row: SeriesMetrics) -> dict[str, object]:
    return {
        "series": row.series,
        "city": row.city,
        "csd": _json_number(row.csd),
        "welfare": _json_number(row.welfare),
        "exclusion": _json_number(row.exclusion),
    }


def _residual_metrics_payload(metrics: ResidualMetrics) -> dict[str, object]:
    return {
        "series": [_series_metrics_payload(row) for row in metrics.series],
        "cities": {
            city: {
                "city_csd": _json_number(metrics.cities[city].city_csd),
                "city_welfare": _json_number(metrics.cities[city].city_welfare),
                "city_exclusion": _json_number(metrics.cities[city].city_exclusion),
            }
            for city in sorted(metrics.cities)
        },
        "failed_safety": list(metrics.failed_safety),
    }


def _residual_candidate_payload(
    candidate: ResidualCandidate,
    *,
    include_source: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "weights": list(candidate.weights),
        "endpoint": list(candidate.endpoint),
        "amplitude": candidate.amplitude,
        "loss": _json_number(candidate.loss),
        "metrics": _residual_metrics_payload(candidate.metrics),
        "safe": candidate.safe,
    }
    if include_source:
        payload["source"] = candidate.source
    return payload


def residual_candidate_sha256(candidate: ResidualCandidate) -> str:
    """Hash the complete policy/evaluation payload, excluding provenance only."""

    return _canonical_sha256(
        _residual_candidate_payload(candidate, include_source=False)
    )


def anchor_payload_sha256(anchor: AnchorSpec) -> str:
    """Hash the complete canonical anchor payload."""

    return _canonical_sha256(_anchor_payload(anchor))


def select_static_anchor(
    candidates: Sequence[AnchorCandidate], tie_tolerance: float = 1e-4
) -> AnchorCandidate:
    """Select safe candidates by objective, family size, norm, then payload hash."""

    safe = [
        candidate
        for candidate in candidates
        if candidate.safe and math.isfinite(candidate.objective)
    ]
    if not safe:
        raise ValueError("no explicitly safe static anchor candidate")
    best_objective = min(candidate.objective for candidate in safe)
    tied = [
        candidate
        for candidate in safe
        if candidate.objective <= best_objective + tie_tolerance
    ]
    return min(
        tied,
        key=lambda candidate: (
            len(candidate.anchor.free_logits),
            math.fsum(value * value for value in candidate.anchor.free_logits),
            anchor_payload_sha256(candidate.anchor),
        ),
    )


def senior_anchor_grid() -> tuple[float, ...]:
    """Expose the canonical existing 101-point senior search grid unchanged."""

    return alpha_grid()


def age_initializer(candidates: Sequence[AnchorCandidate]) -> AnchorSpec:
    """Embed the best safe senior logit exactly in the 60+ age cell."""

    senior = select_static_anchor(
        [candidate for candidate in candidates if candidate.anchor.family == "senior"]
    )
    return AnchorSpec(
        "age", (0.0, 0.0, senior.anchor.free_logits[0]), AGE_REFERENCE
    )


def age_sex_initializer(candidates: Sequence[AnchorCandidate]) -> AnchorSpec:
    """Copy the best safe age logits into both sex cells, fixing the reference."""

    age = select_static_anchor(
        [candidate for candidate in candidates if candidate.anchor.family == "age"]
    )
    age_25_39, age_40_59, age_60_plus = age.anchor.free_logits
    return AnchorSpec(
        "age_sex",
        (
            0.0,
            age_25_39,
            age_25_39,
            age_40_59,
            age_40_59,
            age_60_plus,
            age_60_plus,
        ),
        AGE_SEX_REFERENCE,
    )


def _ordered_map(
    function: Callable[[np.ndarray], AnchorCandidate],
    rows: Sequence[np.ndarray],
    workers: int,
) -> list[AnchorCandidate]:
    if workers == 1:
        return [function(row) for row in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, rows))


def _score_with_provenance(
    anchor: AnchorSpec,
    data: Sequence[SeriesData],
    cfg: AnchorSearchConfig,
    mes_metrics: Mapping[str, CityMetrics],
    source: str,
    seed: int | None,
) -> AnchorCandidate:
    return replace(
        score_static_candidate(anchor, data, cfg, mes_metrics),
        source=source,
        seed=seed,
    )


def _archive_cma_family(
    family: str,
    initializer: AnchorSpec,
    data: Sequence[SeriesData],
    cfg: AnchorSearchConfig,
    mes_metrics: Mapping[str, CityMetrics],
    workers: int,
) -> list[AnchorCandidate]:
    archived: list[AnchorCandidate] = []
    reference = AGE_REFERENCE if family == "age" else AGE_SEX_REFERENCE

    for seed in cfg.seeds:
        def make_anchor(values: Sequence[float]) -> AnchorSpec:
            return AnchorSpec(family, tuple(float(value) for value in values), reference)

        def evaluate(values: np.ndarray) -> AnchorCandidate:
            return score_static_candidate(make_anchor(values), data, cfg, mes_metrics)

        archived.append(
            _score_with_provenance(
                initializer,
                data,
                cfg,
                mes_metrics,
                f"{family}-seed-{seed}-initializer",
                seed,
            )
        )

        def evaluate_batch(population: np.ndarray) -> list[float]:
            scored = _ordered_map(evaluate, list(population), workers)
            return [candidate.objective for candidate in scored]

        result = minimize_batch(
            evaluate_batch,
            np.asarray(initializer.free_logits, dtype=float),
            CMAESConfig(
                sigma0=cfg.sigma0,
                popsize=cfg.popsize,
                generations=cfg.generations,
                seed=seed,
                bound=cfg.bound,
            ),
        )
        for record in result.history:
            generation = int(record["generation"])
            archived.append(
                _score_with_provenance(
                    make_anchor(record["best_x"]),
                    data,
                    cfg,
                    mes_metrics,
                    f"{family}-seed-{seed}-generation-{generation}",
                    seed,
                )
            )
        archived.append(
            _score_with_provenance(
                make_anchor(result.best_x),
                data,
                cfg,
                mes_metrics,
                f"{family}-seed-{seed}-final",
                seed,
            )
        )
    return archived


def _deduplicate_candidates(
    candidates: Sequence[AnchorCandidate],
) -> tuple[AnchorCandidate, ...]:
    unique: dict[str, AnchorCandidate] = {}
    for candidate in candidates:
        unique.setdefault(anchor_payload_sha256(candidate.anchor), candidate)
    return tuple(unique.values())


def fit_static_anchor(
    data: Sequence[SeriesData],
    cfg: AnchorSearchConfig,
    workers: int,
) -> AnchorFit:
    """Evaluate MES, search all nested families, and select the strongest safe anchor."""

    if workers < 1:
        raise ValueError("workers must be positive")
    mes_anchor = AnchorSpec("mes", (), None)
    mes = replace(
        score_static_candidate(mes_anchor, data, cfg, None),
        source="mes-fallback",
        seed=None,
    )
    if not mes.safe:
        raise ValueError("MES did not produce a scoreable safe training fallback")

    candidates: list[AnchorCandidate] = [mes]
    mes_metrics = mes.metrics
    for alpha in senior_anchor_grid():
        anchor = AnchorSpec("senior", (math.log1p(alpha),), AGE_REFERENCE)
        candidates.append(
            _score_with_provenance(
                anchor,
                data,
                cfg,
                mes_metrics,
                f"senior-grid-alpha-{alpha:.17g}",
                None,
            )
        )

    age_initial = age_initializer(candidates)
    candidates.extend(
        _archive_cma_family(
            "age", age_initial, data, cfg, mes_metrics, workers
        )
    )
    age_sex_initial = age_sex_initializer(candidates)
    candidates.extend(
        _archive_cma_family(
            "age_sex", age_sex_initial, data, cfg, mes_metrics, workers
        )
    )

    deduplicated = _deduplicate_candidates(candidates)
    selected = select_static_anchor(deduplicated, cfg.tie_tolerance)
    return AnchorFit(selected, deduplicated, mes_metrics, cfg)


def _training_only_data(data: Sequence[SeriesData]) -> tuple[SeriesData, ...]:
    """Reject held-out inputs and return path-free training-only worker payloads."""

    sanitized: list[SeriesData] = []
    for row in data:
        if tuple(row.test_years):
            raise ValueError(f"held-out years are forbidden in fit input: {row.ref.key}")
        train_years = tuple(row.train_years)
        train_keys = set(row.train_only)
        if train_keys != set(train_years):
            raise ValueError(f"training instances do not match training years: {row.ref.key}")
        if set(row.all_years) != train_keys:
            raise ValueError(f"held-out instances are forbidden in fit input: {row.ref.key}")
        if hasattr(row.ref, "years") and not set(train_years).issubset(
            set(row.ref.years)
        ):
            raise ValueError(f"training years are absent from series metadata: {row.ref.key}")
        training_instances = dict(row.train_only)
        sanitized.append(
            SeriesData(
                ref=SeriesRef(row.ref.key, train_years, ()),
                train_years=train_years,
                test_years=(),
                train_only=training_instances,
                all_years=dict(training_instances),
            )
        )
    if not sanitized:
        raise ValueError("at least one training series is required")
    return tuple(sanitized)


def _residual_reference_metrics(
    data: Sequence[SeriesData],
    anchor: AnchorSpec,
) -> tuple[ResidualMetrics, Mapping[str, CityMetrics]]:
    anchor_metrics = _residual_metrics_from_episodes(
        rollout_static_training(anchor, data)
    )
    mes_metrics = _residual_metrics_from_episodes(
        rollout_static_training(AnchorSpec("mes", (), None), data)
    )
    if anchor_metrics.failed_safety or mes_metrics.failed_safety:
        raise ValueError("anchor or MES has invalid training metrics")
    if not anchor_metrics.cities or set(anchor_metrics.cities) != set(mes_metrics.cities):
        raise ValueError("anchor and MES training cities must be complete and identical")
    return anchor_metrics, mes_metrics.cities


def _rollout_residual_training(
    weights: Sequence[float],
    data: Sequence[SeriesData],
    anchor: AnchorSpec,
    scaler: FeatureScaler,
    env_cfg: EnvConfig | None = None,
) -> list[EpisodeResult]:
    env_cfg = env_cfg or EnvConfig()
    selector = endowment_selector(residual_policy(anchor, scaler, weights), env_cfg)
    return [
        rollout_selector(
            row.ref,
            selector,
            score_years=row.train_years,
            cfg=env_cfg,
            instances=row.train_only,
        )
        for row in data
        if row.train_years
    ]


def _evaluate_residual_with_context(
    endpoint: Sequence[float],
    amplitude: float,
    source: str,
    data: Sequence[SeriesData],
    anchor: AnchorSpec,
    scaler: FeatureScaler,
    anchor_metrics: ResidualMetrics,
    mes_metrics: Mapping[str, CityMetrics],
) -> ResidualCandidate:
    endpoint_values = tuple(float(value) for value in endpoint)
    weights = tuple(float(amplitude) * value for value in endpoint_values)
    if amplitude == 0.0:
        endpoint_values = tuple(0.0 for _ in endpoint_values)
        weights = tuple(0.0 for _ in endpoint_values)
    metrics = _residual_metrics_from_episodes(
        _rollout_residual_training(weights, data, anchor, scaler)
    )
    return score_residual_candidate(
        weights,
        endpoint_values,
        amplitude,
        metrics,
        anchor_metrics,
        mes_metrics,
        source,
    )


_RESIDUAL_WORKER_DATA: tuple[SeriesData, ...] = ()
_RESIDUAL_WORKER_ANCHOR: AnchorSpec | None = None
_RESIDUAL_WORKER_SCALER: FeatureScaler | None = None
_RESIDUAL_WORKER_ANCHOR_METRICS: ResidualMetrics | None = None
_RESIDUAL_WORKER_MES_METRICS: Mapping[str, CityMetrics] | None = None


def _initialize_residual_worker(
    data: tuple[SeriesData, ...],
    anchor: AnchorSpec,
    scaler: FeatureScaler,
) -> None:
    """Initialize a process from training data, selected anchor, and scaler only."""

    global _RESIDUAL_WORKER_DATA
    global _RESIDUAL_WORKER_ANCHOR
    global _RESIDUAL_WORKER_SCALER
    global _RESIDUAL_WORKER_ANCHOR_METRICS
    global _RESIDUAL_WORKER_MES_METRICS
    _RESIDUAL_WORKER_DATA = ()
    _RESIDUAL_WORKER_ANCHOR = None
    _RESIDUAL_WORKER_SCALER = None
    _RESIDUAL_WORKER_ANCHOR_METRICS = None
    _RESIDUAL_WORKER_MES_METRICS = None
    candidate_data = tuple(data)
    candidate_anchor = anchor
    candidate_scaler = scaler
    anchor_metrics, mes_metrics = _residual_reference_metrics(
        candidate_data, candidate_anchor
    )
    _RESIDUAL_WORKER_DATA = candidate_data
    _RESIDUAL_WORKER_ANCHOR = candidate_anchor
    _RESIDUAL_WORKER_SCALER = candidate_scaler
    _RESIDUAL_WORKER_ANCHOR_METRICS = anchor_metrics
    _RESIDUAL_WORKER_MES_METRICS = mes_metrics


def _residual_worker_evaluate(
    task: tuple[tuple[float, ...], float, str],
) -> ResidualCandidate:
    endpoint, amplitude, source = task
    if (
        _RESIDUAL_WORKER_ANCHOR is None
        or _RESIDUAL_WORKER_SCALER is None
        or _RESIDUAL_WORKER_ANCHOR_METRICS is None
        or _RESIDUAL_WORKER_MES_METRICS is None
    ):
        raise RuntimeError("residual worker was not initialized")
    return _evaluate_residual_with_context(
        endpoint,
        amplitude,
        source,
        _RESIDUAL_WORKER_DATA,
        _RESIDUAL_WORKER_ANCHOR,
        _RESIDUAL_WORKER_SCALER,
        _RESIDUAL_WORKER_ANCHOR_METRICS,
        _RESIDUAL_WORKER_MES_METRICS,
    )


def _deduplicate_residual_candidates(
    candidates: Sequence[ResidualCandidate],
) -> tuple[ResidualCandidate, ...]:
    unique: dict[str, ResidualCandidate] = {}
    for candidate in candidates:
        unique.setdefault(residual_candidate_sha256(candidate), candidate)
    return tuple(unique.values())


def _fit_residual_seed_with_config(
    data: Sequence[SeriesData],
    anchor_fit: AnchorFit,
    scaler: FeatureScaler,
    seed: int,
    workers: int,
    config: ResidualSearchConfig,
) -> ResidualSeedFit:
    if (
        config.tau,
        config.welfare_floor,
        config.norm_penalty,
    ) != (
        _PRODUCTION_RESIDUAL_CONFIG.tau,
        _PRODUCTION_RESIDUAL_CONFIG.welfare_floor,
        _PRODUCTION_RESIDUAL_CONFIG.norm_penalty,
    ):
        raise ValueError("configured fit requires frozen residual scoring constants")
    if workers < 1:
        raise ValueError("workers must be positive")
    if seed not in config.seeds:
        raise ValueError(f"seed must be one of {config.seeds!r}")
    training_data = _training_only_data(data)
    selected_anchor = anchor_fit.selected.anchor
    anchor_metrics, mes_metrics = _residual_reference_metrics(
        training_data, selected_anchor
    )
    if dict(anchor_metrics.cities) != dict(anchor_fit.selected.metrics):
        raise ValueError("selected anchor metrics do not match training-only replay")
    if dict(mes_metrics) != dict(anchor_fit.mes_metrics):
        raise ValueError("MES metrics do not match training-only replay")

    def evaluate_local(
        task: tuple[tuple[float, ...], float, str]
    ) -> ResidualCandidate:
        endpoint, amplitude, source = task
        return _evaluate_residual_with_context(
            endpoint,
            amplitude,
            source,
            training_data,
            selected_anchor,
            scaler,
            anchor_metrics,
            mes_metrics,
        )

    zero_endpoint = tuple(0.0 for _ in FEATURE_NAMES)
    archive: list[ResidualCandidate] = [
        evaluate_local((zero_endpoint, 0.0, f"seed-{seed}-zero-initial"))
    ]
    generation = 0

    def run_optimizer(pool: ProcessPoolExecutor | None):
        nonlocal generation

        def evaluate_batch(population: np.ndarray) -> list[float]:
            nonlocal generation
            tasks = [
                (
                    tuple(float(value) for value in endpoint),
                    amplitude,
                    (
                        f"seed-{seed}-generation-{generation}-"
                        f"member-{member}-amplitude-{amplitude:.2f}"
                    ),
                )
                for member, endpoint in enumerate(population)
                for amplitude in RAY_AMPLITUDES
            ]
            if pool is None:
                scored = [evaluate_local(task) for task in tasks]
            else:
                scored = list(pool.map(_residual_worker_evaluate, tasks, chunksize=1))
            archive.extend(scored)
            generation += 1
            return [
                select_ray_candidate(scored[index : index + len(RAY_AMPLITUDES)]).loss
                for index in range(0, len(scored), len(RAY_AMPLITUDES))
            ]

        return minimize_batch(
            evaluate_batch,
            np.zeros(len(FEATURE_NAMES), dtype=float),
            CMAESConfig(
                sigma0=config.sigma0,
                popsize=config.popsize,
                generations=config.generations,
                seed=seed,
                bound=config.bound,
            ),
        )

    if workers == 1:
        result = run_optimizer(None)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_residual_worker,
            initargs=(training_data, selected_anchor, scaler),
        ) as pool:
            result = run_optimizer(pool)

    archive.append(
        evaluate_local((zero_endpoint, 0.0, f"seed-{seed}-zero-final"))
    )
    deduplicated = _deduplicate_residual_candidates(archive)
    selected = select_residual_candidate(
        deduplicated,
        replacement_delta=config.replacement_delta,
    )
    return ResidualSeedFit(seed, selected, deduplicated, tuple(result.history))


def fit_residual_seed(
    data: Sequence[SeriesData],
    anchor: AnchorFit,
    scaler: FeatureScaler,
    seed: int,
    workers: int,
) -> ResidualSeedFit:
    """Fit one production residual seed using training-only inputs."""

    training_data = _training_only_data(data)
    if not isinstance(anchor, AnchorFit):
        raise TypeError("anchor must be an AnchorFit with selected and MES references")
    if not isinstance(scaler, FeatureScaler):
        raise TypeError("scaler must be a training-fitted FeatureScaler")
    return _fit_residual_seed_with_config(
        training_data,
        anchor,
        scaler,
        seed,
        workers,
        residual_search_config(),
    )


def select_primary_seed(seed_fits: Sequence[ResidualSeedFit]) -> int:
    """Select from training loss only, then amplitude, norm, and seed."""

    if not seed_fits:
        raise ValueError("at least one residual seed fit is required")
    seed_ids = tuple(fit.seed for fit in seed_fits)
    if len(seed_ids) != len(set(seed_ids)):
        raise ValueError("duplicate residual seed IDs are forbidden")
    if any(
        not fit.selected.safe or not math.isfinite(fit.selected.loss)
        for fit in seed_fits
    ):
        raise ValueError("every residual seed must have a safe finite selection")
    return min(
        seed_fits,
        key=lambda fit: (
            fit.selected.loss,
            fit.selected.amplitude,
            _residual_norm(fit.selected),
            fit.seed,
        ),
    ).seed


def _fit_residual_fold_with_config(
    data: Sequence[SeriesData],
    anchor_fit: AnchorFit,
    scaler: FeatureScaler,
    workers: int,
    config: ResidualSearchConfig,
) -> FoldFit:
    seed_fits = tuple(
        _fit_residual_seed_with_config(
            data, anchor_fit, scaler, seed, workers, config
        )
        for seed in config.seeds
    )
    return FoldFit(
        anchor_fit,
        scaler,
        seed_fits,
        select_primary_seed(seed_fits),
        config,
    )


def fit_residual_fold(
    data: Sequence[SeriesData],
    anchor_fit: AnchorFit,
    scaler: FeatureScaler,
    workers: int,
) -> FoldFit:
    """Fit and retain all three fixed production seeds for one training fold."""

    training_data = _training_only_data(data)
    if not isinstance(anchor_fit, AnchorFit):
        raise TypeError("anchor_fit must carry selected and MES references")
    if not isinstance(scaler, FeatureScaler):
        raise TypeError("scaler must be a training-fitted FeatureScaler")
    return _fit_residual_fold_with_config(
        training_data,
        anchor_fit,
        scaler,
        workers,
        residual_search_config(),
    )


def _candidate_payload(candidate: AnchorCandidate) -> dict[str, object]:
    return {
        "anchor": _anchor_payload(candidate.anchor),
        "anchor_sha256": anchor_payload_sha256(candidate.anchor),
        "metrics": {
            city: asdict(candidate.metrics[city]) for city in sorted(candidate.metrics)
        },
        "objective": (
            candidate.objective if math.isfinite(candidate.objective) else None
        ),
        "safe": candidate.safe,
        "seed": candidate.seed,
        "source": candidate.source,
    }


def _fit_payload(fit: AnchorFit) -> dict[str, object]:
    families = ("mes", "senior", "age", "age_sex")
    return {
        "schema_version": 1,
        "training_only": True,
        "config": {**asdict(fit.config), "seeds": list(fit.config.seeds)},
        "mes_fallback": {
            "available": any(row.anchor.family == "mes" for row in fit.candidates),
            "safe": any(
                row.anchor.family == "mes" and row.safe for row in fit.candidates
            ),
        },
        "families_present": [
            family
            for family in families
            if any(row.anchor.family == family for row in fit.candidates)
        ],
        "mes_metrics": {
            city: asdict(fit.mes_metrics[city]) for city in sorted(fit.mes_metrics)
        },
        "selected": _candidate_payload(fit.selected),
        "candidates": [_candidate_payload(row) for row in fit.candidates],
    }


def _ordinary_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _ordinary_payload(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_ordinary_payload(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _residual_seed_payload(fit: ResidualSeedFit) -> dict[str, object]:
    return {
        "seed": fit.seed,
        "selected": _residual_candidate_payload(fit.selected),
        "archive": [_residual_candidate_payload(row) for row in fit.archive],
        "archive_count": len(fit.archive),
        "optimizer_history": [
            _ordinary_payload(record) for record in fit.optimizer_history
        ],
    }


def _residual_fit_payload(
    fit: FoldFit,
    static_artifact_sha256: str,
) -> dict[str, object]:
    primary = next(row for row in fit.seeds if row.seed == fit.primary_seed)
    payload: dict[str, object] = {
        "schema_version": 2,
        "training_only": True,
        "static_artifact_sha256": static_artifact_sha256,
        "config": {**asdict(fit.config), "seeds": list(fit.config.seeds)},
        "anchor": _candidate_payload(fit.anchor_fit.selected),
        "scaler": {
            "feature_names": list(fit.scaler.feature_names),
            "mean": fit.scaler.mean.tolist(),
            "scale": fit.scaler.scale.tolist(),
            "clip": fit.scaler.clip,
            "row_count": fit.scaler.row_count,
            "instance_count": fit.scaler.instance_count,
            "sha256": fit.scaler.sha256,
        },
        "seeds": [_residual_seed_payload(row) for row in fit.seeds],
        "primary_seed": fit.primary_seed,
        "primary": _residual_candidate_payload(primary.selected),
        "synthetic_actuated": _has_nonzero_policy_weights(primary.selected),
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    return payload


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_payload_equal(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    """Compare retained numeric payloads without Python's signed-zero aliasing."""

    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _write_canonical_json(path: Path, payload: Mapping[str, object]) -> str:
    encoded = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _immutable_canonical_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    approved_root: Path | None = None,
) -> str:
    """Create an immutable canonical artifact or authenticate an exact rerun."""

    encoded = _canonical_json_bytes(payload)
    _write_immutable_bytes(path, encoded, approved_root=approved_root)
    return hashlib.sha256(encoded).hexdigest()


def development_cluster_key(series: str) -> str:
    """Remove one exact terminal Gdynia pool label, case-insensitively."""

    if not isinstance(series, str):
        raise TypeError("development series identifier must be a string")
    folded = series.casefold()
    suffixes = (
        "__large_projects",
        "__small_projects",
        " | large",
        " | small",
        "| large",
        "| small",
        "__large",
        "__small",
    )
    for suffix in suffixes:
        if folded.endswith(suffix):
            return series[: -len(suffix)]
    return series


def _development_result_payload(row: DevelopmentSeriesResult) -> dict[str, object]:
    return {
        "fold": row.fold,
        "seed": row.seed,
        "series": row.series,
        "city": row.city,
        "cluster": row.cluster,
        "residual": dict(row.residual),
        "anchor": dict(row.anchor),
        "mes": dict(row.mes),
        "signatures": _ordinary_payload(row.signatures),
    }


def _coerce_development_result(row: object) -> DevelopmentSeriesResult:
    if isinstance(row, DevelopmentSeriesResult):
        return row
    if not isinstance(row, Mapping):
        raise RuntimeError("development evidence rows must be result records")
    try:
        return DevelopmentSeriesResult(
            fold=row["fold"],
            seed=row["seed"],
            series=row["series"],
            city=row["city"],
            cluster=row["cluster"],
            residual=row["residual"],
            anchor=row["anchor"],
            mes=row["mes"],
            signatures=row["signatures"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid development evidence row: {exc}") from exc


def _cluster_city_values(
    rows: Sequence[DevelopmentSeriesResult],
) -> tuple[dict[str, tuple[float, ...]], int]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        delta = float(row.residual["csd"]) - float(row.anchor["csd"])
        if not math.isfinite(delta):
            raise ValueError("development CSD differences must be finite")
        grouped[row.city][row.cluster].append(delta)
    if not grouped:
        raise ValueError("at least one development result is required")
    city_clusters = {
        city: tuple(
            math.fsum(grouped[city][cluster]) / len(grouped[city][cluster])
            for cluster in sorted(grouped[city])
        )
        for city in sorted(grouped)
    }
    return city_clusters, sum(len(values) for values in city_clusters.values())


def cluster_city_bootstrap(
    rows: Sequence[DevelopmentSeriesResult | Mapping[str, object]],
    draws: int = 20000,
    seed: int = 20260818,
) -> dict[str, float]:
    """Cluster first, bootstrap inside cities, then weight city means equally."""

    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("bootstrap draws must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("bootstrap seed must be an integer")
    records = tuple(_coerce_development_result(row) for row in rows)
    city_clusters, cluster_count = _cluster_city_values(records)
    city_means = {
        city: math.fsum(values) / len(values)
        for city, values in city_clusters.items()
    }
    observed = math.fsum(city_means.values()) / len(city_means)
    generator = np.random.default_rng(seed)
    samples = np.zeros(draws, dtype=float)
    for values in city_clusters.values():
        array = np.asarray(values, dtype=float)
        indices = generator.integers(0, len(array), size=(draws, len(array)))
        samples += array[indices].mean(axis=1) / len(city_clusters)
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "mean": float(observed),
        "lower": float(lower),
        "upper": float(upper),
        "draws": draws,
        "seed": seed,
        "series_count": len(records),
        "cluster_count": cluster_count,
        "city_count": len(city_clusters),
    }


def _city_development_means(
    rows: Sequence[DevelopmentSeriesResult],
) -> dict[str, float]:
    city_clusters, _ = _cluster_city_values(rows)
    return {
        city: math.fsum(values) / len(values)
        for city, values in city_clusters.items()
    }


@dataclass(frozen=True)
class DevelopmentGateEvidence:
    """Authenticated immutable evidence for the exact eight-fold gate."""

    folds: Sequence[DevelopmentFold]
    fit_records: Sequence[Mapping[str, object]]
    series_records: Sequence[Mapping[str, object]]
    primary_seeds: Mapping[str, int] = field(init=False)
    expected_scored_series: Mapping[str, tuple[str, ...]] = field(init=False)
    expected_safe_static: Mapping[str, tuple[tuple[str, str], ...]] = field(
        init=False
    )
    safe_static_results: Mapping[str, object] = field(init=False)
    city_results: tuple[DevelopmentSeriesResult, ...] = field(init=False)
    district_results: tuple[DevelopmentSeriesResult, ...] = field(init=False)

    def __post_init__(self) -> None:
        folds = tuple(self.folds)
        fit_records = tuple(copy.deepcopy(tuple(self.fit_records)))
        series_records = tuple(copy.deepcopy(tuple(self.series_records)))
        expected_names = (*_CITY_DEVELOPMENT_FOLDS, *_DISTRICT_DEVELOPMENT_FOLDS)
        expected_kinds = ("city",) * 3 + ("district",) * 5
        if (
            len(folds) != len(expected_names)
            or len(fit_records) != len(expected_names)
            or len(series_records) != len(expected_names)
            or any(not isinstance(fold, DevelopmentFold) for fold in folds)
            or tuple(fold.name for fold in folds) != expected_names
            or tuple(fold.kind for fold in folds) != expected_kinds
            or len({fold.name for fold in folds}) != len(expected_names)
        ):
            raise RuntimeError("trusted development fold inventory differs")

        primary_seeds = {}
        expected_scored = {}
        expected_static = {}
        static_results = {}
        city_rows = []
        district_rows = []
        for fold, fit_record, series_record in zip(
            folds, fit_records, series_records
        ):
            try:
                result, safe_static = _authenticate_development_artifact_pair(
                    fold,
                    fit_record,
                    series_record,
                    anchor_search_config(),
                    residual_search_config(),
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"trusted development evidence differs for {fold.name}: {exc}"
                ) from exc
            primary_seeds[fold.name] = result["primary_seed"]
            expected_scored[fold.name] = tuple(row.ref.key for row in fold.test)
            expected_static[fold.name] = tuple(
                (row["identity"], row["family"]) for row in safe_static
            )
            static_results[fold.name] = result["safe_static"]
            destination = city_rows if fold.kind == "city" else district_rows
            try:
                destination.extend(
                    _coerce_development_result(row)
                    for row in result["series_results"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"trusted development result rows differ for {fold.name}: {exc}"
                ) from exc

        object.__setattr__(self, "folds", folds)
        object.__setattr__(
            self,
            "fit_records",
            tuple(_ImmutableJSONDict(record) for record in fit_records),
        )
        object.__setattr__(
            self,
            "series_records",
            tuple(_ImmutableJSONDict(record) for record in series_records),
        )
        object.__setattr__(self, "primary_seeds", _ImmutableJSONDict(primary_seeds))
        object.__setattr__(
            self, "expected_scored_series", _ImmutableJSONDict(expected_scored)
        )
        object.__setattr__(
            self, "expected_safe_static", _ImmutableJSONDict(expected_static)
        )
        object.__setattr__(
            self, "safe_static_results", _ImmutableJSONDict(static_results)
        )
        object.__setattr__(self, "city_results", tuple(city_rows))
        object.__setattr__(self, "district_results", tuple(district_rows))


def _validate_development_inventory(
    city_rows: Sequence[DevelopmentSeriesResult],
    district_rows: Sequence[DevelopmentSeriesResult],
    primary_seeds: Mapping[str, int],
    expected_scored_series: Mapping[str, tuple[str, ...]],
    expected_safe_static: Mapping[str, tuple[tuple[str, str], ...]],
    safe_static_results: Mapping[str, object],
) -> None:
    expected_folds = (*_CITY_DEVELOPMENT_FOLDS, *_DISTRICT_DEVELOPMENT_FOLDS)
    if set(expected_scored_series) != set(expected_folds):
        raise RuntimeError("development evidence has incomplete expected scored inventory")
    if set(expected_safe_static) != set(expected_folds):
        raise RuntimeError("development evidence has incomplete safe-static inventory")
    if set(safe_static_results) != set(expected_folds):
        raise RuntimeError("development evidence has incomplete safe-static results")
    if any(seed not in _DEVELOPMENT_SEEDS for seed in primary_seeds.values()):
        raise RuntimeError("development evidence has an invalid primary seed")
    expected_keys = {
        (fold, seed, series)
        for fold, series_ids in expected_scored_series.items()
        for seed in _DEVELOPMENT_SEEDS
        for series in series_ids
    }
    actual_keys = [
        (row.fold, row.seed, row.series) for row in (*city_rows, *district_rows)
    ]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise RuntimeError(
            "development evidence differs from the expected fold-series-seed inventory"
        )
    if any(row.fold not in _CITY_DEVELOPMENT_FOLDS for row in city_rows):
        raise RuntimeError("development evidence city rows contain an unexpected fold")
    if any(row.fold not in _DISTRICT_DEVELOPMENT_FOLDS for row in district_rows):
        raise RuntimeError("development evidence district rows contain an unexpected fold")
    for fold in _CITY_DEVELOPMENT_FOLDS:
        expected_city = fold.removeprefix("city_out_").replace("_", "/", 1)
        if {row.city for row in city_rows if row.fold == fold} != {expected_city}:
            raise RuntimeError("development evidence fold and city identities disagree")
    expected_primary_folds = set(_CITY_DEVELOPMENT_FOLDS) | set(
        _DISTRICT_DEVELOPMENT_FOLDS
    )
    if set(primary_seeds) != expected_primary_folds:
        raise RuntimeError("development evidence has incomplete primary-seed inventory")
    for row in (*city_rows, *district_rows):
        if any(set(metrics) != set(_DEVELOPMENT_METRICS) for metrics in (
            row.residual,
            row.anchor,
            row.mes,
        )):
            raise RuntimeError("development evidence has incomplete metric families")
    for fold in expected_folds:
        expected_candidates = dict(expected_safe_static[fold])
        supplied = safe_static_results[fold]
        if not isinstance(supplied, Mapping) or set(supplied) != set(
            expected_candidates
        ):
            raise RuntimeError(
                f"development evidence safe-static candidates differ for {fold}"
            )
        for identity, family in expected_candidates.items():
            candidate = supplied[identity]
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "family",
                "series",
            }:
                raise RuntimeError(
                    f"development evidence safe-static payload differs for {fold}"
                )
            if candidate["family"] != family or not isinstance(
                candidate["series"], Mapping
            ):
                raise RuntimeError(
                    f"development evidence safe-static identity differs for {fold}"
                )
            if set(candidate["series"]) != set(expected_scored_series[fold]):
                raise RuntimeError(
                    f"development evidence safe-static series differ for {fold}"
                )
            try:
                for metrics in candidate["series"].values():
                    _metric_snapshot(metrics, "safe_static")
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"development evidence safe-static metrics differ for {fold}: {exc}"
                ) from exc


def evaluate_development_gate(payload: DevelopmentGateEvidence) -> dict[str, object]:
    """Apply the frozen six-condition old-data development decision."""

    if not isinstance(payload, DevelopmentGateEvidence):
        raise RuntimeError(
            "development gate requires trusted DevelopmentGateEvidence"
        )
    city_rows = payload.city_results
    district_rows = payload.district_results
    primary_seeds = payload.primary_seeds
    expected_scored = payload.expected_scored_series
    expected_static = payload.expected_safe_static
    static_results = payload.safe_static_results
    _validate_development_inventory(
        city_rows,
        district_rows,
        primary_seeds,
        expected_scored,
        expected_static,
        static_results,
    )

    primary = tuple(
        row for row in city_rows if row.seed == primary_seeds[row.fold]
    )
    primary_inference = cluster_city_bootstrap(primary)
    held_city_means = _city_development_means(primary)
    primary_improves = (
        primary_inference["mean"] < 0.0 and primary_inference["upper"] < 0.0
    )
    all_cities_negative = all(value < 0.0 for value in held_city_means.values())

    district_primary = tuple(
        row for row in district_rows if row.seed == primary_seeds[row.fold]
    )
    district_means = {}
    for fold in _DISTRICT_DEVELOPMENT_FOLDS:
        fold_rows = tuple(row for row in district_primary if row.fold == fold)
        values, _ = _cluster_city_values(fold_rows)
        flattened = [value for clusters in values.values() for value in clusters]
        district_means[fold] = math.fsum(flattened) / len(flattened)
    negative_districts = sum(value < 0.0 for value in district_means.values())

    city_safety = {}
    for fold in _CITY_DEVELOPMENT_FOLDS:
        rows = tuple(row for row in primary if row.fold == fold)
        welfare = math.fsum(float(row.residual["welfare"]) for row in rows)
        anchor_welfare = math.fsum(float(row.anchor["welfare"]) for row in rows)
        mes_welfare = math.fsum(float(row.mes["welfare"]) for row in rows)
        exclusion = math.fsum(float(row.residual["exclusion"]) for row in rows) / len(rows)
        anchor_exclusion = math.fsum(float(row.anchor["exclusion"]) for row in rows) / len(rows)
        mes_exclusion = math.fsum(float(row.mes["exclusion"]) for row in rows) / len(rows)
        anchor_welfare_threshold = 0.99 * anchor_welfare
        mes_welfare_threshold = 0.99 * mes_welfare
        comparisons = {
            "welfare_vs_anchor": welfare >= anchor_welfare_threshold,
            "welfare_vs_mes": welfare >= mes_welfare_threshold,
            "exclusion_vs_anchor": exclusion <= anchor_exclusion,
            "exclusion_vs_mes": exclusion <= mes_exclusion,
        }
        city_safety[fold] = {
            "city": rows[0].city,
            "residual_welfare": welfare,
            "anchor_welfare": anchor_welfare,
            "mes_welfare": mes_welfare,
            "anchor_welfare_threshold": anchor_welfare_threshold,
            "mes_welfare_threshold": mes_welfare_threshold,
            "anchor_welfare_ratio": (
                welfare / anchor_welfare if anchor_welfare != 0.0 else None
            ),
            "mes_welfare_ratio": (
                welfare / mes_welfare if mes_welfare != 0.0 else None
            ),
            "residual_exclusion": exclusion,
            "anchor_exclusion": anchor_exclusion,
            "mes_exclusion": mes_exclusion,
            "anchor_exclusion_threshold": anchor_exclusion,
            "mes_exclusion_threshold": mes_exclusion,
            "comparisons": comparisons,
            "failed_comparisons": [
                name for name, passed in comparisons.items() if not passed
            ],
        }

    seed_means = {
        str(seed): cluster_city_bootstrap(
            tuple(row for row in city_rows if row.seed == seed)
        )["mean"]
        for seed in _DEVELOPMENT_SEEDS
    }
    primary_by_series = {}
    for row in primary:
        primary_by_series.setdefault((row.fold, row.series), row)
    actuated = [
        row for row in primary_by_series.values() if row.signatures["actuated"]
    ]
    actuation_rate = len(actuated) / len(primary_by_series)
    actuated_cities = sorted({row.city for row in actuated})

    conditions = {
        "primary_improves_anchor": primary_improves,
        "all_held_cities_negative": all_cities_negative,
        "district_fold_majority": negative_districts >= 4,
        "dual_reference_safety": all(
            all(record["comparisons"].values())
            for record in city_safety.values()
        ),
        "all_seeds_negative": all(value < 0.0 for value in seed_means.values()),
        "actuation_coverage": actuation_rate >= 0.10 and len(actuated_cities) >= 2,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return {
        "classification": "pass" if not failed else "fail",
        "conditions": conditions,
        "failed_conditions": failed,
        "thresholds": {
            "bootstrap_draws": 20000,
            "bootstrap_seed": 20260818,
            "bootstrap_upper_max": 0.0,
            "district_negative_min": 4,
            "welfare_floor": 0.99,
            "actuation_rate_min": 0.10,
            "actuated_city_min": 2,
        },
        "observed": {
            "primary_inference": primary_inference,
            "held_city_means": held_city_means,
            "district_fold_means": district_means,
            "negative_district_folds": negative_districts,
            "city_safety": city_safety,
            "safety_failures": [
                fold
                for fold, record in city_safety.items()
                if record["failed_comparisons"]
            ],
            "seed_equal_city_means": seed_means,
            "actuated_series": len(actuated),
            "held_series": len(primary_by_series),
            "actuation_rate": actuation_rate,
            "actuated_cities": actuated_cities,
        },
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path, "hashed artifact")).hexdigest()


def _canonical_old_input_path(path: Path) -> str:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError("old-development input is outside the repository root") from exc
    return _canonical_persisted_path(
        relative,
        approved_prefixes=("data/pb/", "data/pb_multicity/"),
    )


def _normalized_fold_views(split, index: Mapping[str, SeriesRef]) -> tuple[
    tuple[SeriesData, ...], tuple[SeriesData, ...], dict[str, str]
]:
    train_rows = []
    test_rows = []
    hashes = {}
    for role, entries in (("train", split.train), ("test", split.test)):
        for key, years in entries:
            if key not in index:
                raise RuntimeError(f"development split contains unexpected series {key!r}")
            ref = index[key]
            requested = tuple(years)
            load_years = requested if role == "train" else tuple(ref.years)
            instances = load_series(ref, years=load_years)
            missing = sorted(set(load_years) - set(instances))
            if missing:
                raise RuntimeError(f"development series {key!r} is incomplete: {missing}")
            for year, path in zip(ref.years, ref.paths):
                if year in load_years:
                    hashes[_canonical_old_input_path(path)] = _file_sha256(path)
            if role == "train":
                paths_by_year = dict(zip(ref.years, ref.paths))
                row_ref = SeriesRef(
                    ref.key,
                    requested,
                    tuple(
                        paths_by_year[year]
                        for year in requested
                        if year in paths_by_year
                    ),
                )
            else:
                row_ref = ref
            row = SeriesData(
                ref=row_ref,
                train_years=requested if role == "train" else (),
                test_years=requested if role == "test" else (),
                train_only=dict(instances) if role == "train" else {},
                all_years=dict(instances),
            )
            (train_rows if role == "train" else test_rows).append(row)
    return tuple(train_rows), tuple(test_rows), dict(sorted(hashes.items()))


def _load_city_development_folds() -> tuple[DevelopmentFold, ...]:
    from iclr_multicity_protocol import (
        load_canonical_index_from_manifest,
        load_frozen_split,
    )

    manifest = MULTICITY_RESULT_ROOT / "corpus_manifest.json"
    index = load_canonical_index_from_manifest(manifest, MULTICITY_DATA_DIR)
    if len(index) != 75:
        raise RuntimeError("city development corpus must contain exactly 75 series")
    manifest_sha = _file_sha256(manifest)
    folds = []
    for name in _CITY_DEVELOPMENT_FOLDS:
        split_path = MULTICITY_RESULT_ROOT / "splits" / f"{name}.json"
        split = load_frozen_split(split_path, index)
        train, test, hashes = _normalized_fold_views(split, index)
        if set(row.ref.key for row in (*train, *test)) != set(index):
            raise RuntimeError("city development split has incomplete series inventory")
        folds.append(
            DevelopmentFold(
                name,
                "city",
                train,
                test,
                manifest_sha,
                {
                    "corpus_manifest_sha256": manifest_sha,
                    "split_sha256": _file_sha256(split_path),
                    "input_blob_sha256": hashes,
                },
            )
        )
    return tuple(folds)


def _load_district_development_folds() -> tuple[DevelopmentFold, ...]:
    index = build_series_index(CorpusConfig(data_dir=DISTRICT_DATA_DIR))
    if len(index) != 19:
        raise RuntimeError("district development corpus must contain exactly 19 series")
    source_rows = {
        key: {
            "years": list(ref.years),
            "files": [
                {
                    "path": _canonical_old_input_path(path),
                    "sha256": _file_sha256(path),
                }
                for path in ref.paths
            ],
        }
        for key, ref in sorted(index.items())
    }
    source_sha = _canonical_sha256(source_rows)
    loaded = []
    for fold_index, name in enumerate(_DISTRICT_DEVELOPMENT_FOLDS):
        split_path = DISTRICT_SPLIT_ROOT / f"{name}.json"
        split = load_split(name, DISTRICT_SPLIT_ROOT)
        expected = leave_district_out(index, fold_index, 5)
        if (
            split.name != expected.name
            or split.train != expected.train
            or split.test != expected.test
        ):
            raise RuntimeError(
                f"district split {name!r} differs from its canonical partition"
            )
        loaded.append((name, split_path, split))

    test_counts = {key: 0 for key in index}
    train_counts = {key: 0 for key in index}
    for _, _, split in loaded:
        for key, _ in split.test:
            test_counts[key] += 1
        for key, _ in split.train:
            train_counts[key] += 1
    if set(test_counts.values()) != {1} or set(train_counts.values()) != {4}:
        raise RuntimeError("district split membership counts are not one-test/four-train")

    folds = []
    for name, split_path, split in loaded:
        train, test, hashes = _normalized_fold_views(split, index)
        if set(row.ref.key for row in (*train, *test)) != set(index):
            raise RuntimeError("district development split has incomplete series inventory")
        folds.append(
            DevelopmentFold(
                name,
                "district",
                train,
                test,
                source_sha,
                {
                    "corpus_manifest_sha256": source_sha,
                    "split_sha256": _file_sha256(split_path),
                    "input_blob_sha256": hashes,
                },
            )
        )
    return tuple(folds)


def _load_development_folds_unchecked() -> Sequence[DevelopmentFold]:
    """Load and authenticate the exact three-city plus five-district inventory."""

    folds = (*_load_city_development_folds(), *_load_district_development_folds())
    expected = (*_CITY_DEVELOPMENT_FOLDS, *_DISTRICT_DEVELOPMENT_FOLDS)
    observed = tuple(fold.name for fold in folds)
    if observed != expected or len(observed) != len(set(observed)):
        raise RuntimeError(
            f"development fold inventory differs: expected {expected!r}, found {observed!r}"
        )
    if tuple(fold.kind for fold in folds) != ("city",) * 3 + ("district",) * 5:
        raise RuntimeError("development fold inventory has unexpected fold families")
    return folds


def load_development_folds() -> Sequence[DevelopmentFold]:
    """Guard and load the exact prepared production fold inventory."""

    workers = _APPROVED_WORKERS
    profile = _validate_production_runtime(_measure_production_runtime(workers), workers)
    config, _, prepared = _authenticate_preparation(profile, historical=False)
    folds = _folds_for_execution(prepared)
    if [_prepared_fold_binding(fold) for fold in folds] != config["fold_bindings"]:
        raise RuntimeError("loaded development folds differ from preparation")
    return folds


def _fit_development_training(
    rows: Sequence[SeriesData], workers: int, *, smoke: bool = False
) -> FoldFit:
    training = _training_only_data(rows)
    anchor_cfg = smoke_anchor_search_config() if smoke else anchor_search_config()
    anchor_fit = fit_static_anchor(training, anchor_cfg, workers)
    instances = tuple(
        row.train_only[year]
        for row in training
        for year in sorted(row.train_only)
    )
    scaler = fit_context_scaler(instances)
    if smoke:
        return _fit_residual_fold_with_config(
            training,
            anchor_fit,
            scaler,
            workers,
            smoke_residual_search_config(),
        )
    return fit_residual_fold(training, anchor_fit, scaler, workers)


def _episode_development_metrics(episode: EpisodeResult) -> dict[str, float]:
    values = (episode.worst_csd, episode.welfare, episode.exclusion)
    if episode.worst_csd is None or not all(
        math.isfinite(float(value)) for value in values if value is not None
    ):
        raise RuntimeError(f"unscoreable held-out development series {episode.series!r}")
    return {
        "csd": float(episode.worst_csd),
        "welfare": float(episode.welfare),
        "exclusion": float(episode.exclusion),
    }


def _score_development_policy(
    row: SeriesData, policy
) -> dict[str, float]:
    selector = endowment_selector(policy, EnvConfig())
    episode = rollout_selector(
        row.ref,
        selector,
        score_years=row.test_years,
        cfg=EnvConfig(),
        instances=row.all_years,
    )
    return _episode_development_metrics(episode)


def _fit_development_payload(fit: FoldFit | Mapping[str, object]) -> dict[str, object]:
    if isinstance(fit, FoldFit):
        static_payload = _fit_payload(fit.anchor_fit)
        return {
            "primary_seed": fit.primary_seed,
            "static": static_payload,
            "residual": _residual_fit_payload(
                fit, _canonical_sha256(static_payload)
            ),
        }
    return _ordinary_payload(fit)


def _evaluate_development_scored(
    fold: DevelopmentFold,
    fit: FoldFit,
    workers: int,
) -> dict[str, object]:
    del workers
    if not isinstance(fit, FoldFit):
        raise TypeError("development evaluation requires a validated FoldFit")
    if tuple(seed.seed for seed in fit.seeds) != _DEVELOPMENT_SEEDS:
        raise RuntimeError("development fit must retain seeds 1, 2, and 42")

    from iclr_residual_actuation import scan_fold_actuation

    scored_rows = tuple(
        _materialize_development_series(row) for row in fold.test
    )

    scored_instances = tuple(
        row.all_years[year]
        for row in scored_rows
        for year in sorted(row.test_years)
    )
    actuation = scan_fold_actuation(scored_instances, fit)
    actuation_by_instance = {record.instance: record for record in actuation}
    results = []
    safe_static = {}
    anchor = fit.anchor_fit.selected.anchor
    mes = AnchorSpec("mes", (), None)
    anchor_by_series = {
        row.ref.key: _score_development_policy(row, anchor_policy(anchor))
        for row in scored_rows
    }
    mes_by_series = {
        row.ref.key: _score_development_policy(row, anchor_policy(mes))
        for row in scored_rows
    }
    for candidate in fit.anchor_fit.candidates:
        if not candidate.safe:
            continue
        candidate_key = anchor_payload_sha256(candidate.anchor)
        safe_static[candidate_key] = {
            "family": candidate.anchor.family,
            "series": {
                row.ref.key: _score_development_policy(
                    row, anchor_policy(candidate.anchor)
                )
                for row in scored_rows
            },
        }
    for seed_fit in fit.seeds:
        weights = seed_fit.selected.weights
        for row in scored_rows:
            residual = _score_development_policy(
                row, residual_policy(anchor, fit.scaler, weights)
            )
            records = [
                actuation_by_instance.get(row.all_years[year].path)
                for year in row.test_years
            ]
            actuated = seed_fit.seed == fit.primary_seed and any(
                record is not None and record.actuated for record in records
            )
            result = DevelopmentSeriesResult(
                fold=fold.name,
                seed=seed_fit.seed,
                series=row.ref.key,
                city=row.ref.city,
                cluster=development_cluster_key(row.ref.key),
                residual=residual,
                anchor=anchor_by_series[row.ref.key],
                mes=mes_by_series[row.ref.key],
                signatures={
                    "residual": "actuated" if actuated else "anchor-equivalent",
                    "anchor": "anchor-equivalent",
                    "actuated": actuated,
                },
            )
            results.append(_development_result_payload(result))
    return {
        "fold": fold.name,
        "kind": fold.kind,
        "primary_seed": fit.primary_seed,
        "fit_training_series": [row.ref.key for row in fold.train],
        "scored_series": [row.ref.key for row in fold.test],
        "series_results": results,
        "safe_static": safe_static,
        "actuation": [asdict(record) for record in actuation],
    }


def _with_payload_sha(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["payload_sha256"] = _canonical_sha256(result)
    return result


def _frozen_instance_payload(instance: _FrozenPBInstance) -> dict[str, object]:
    return {
        "path": instance.path,
        "meta": dict(instance.meta),
        "projects": {
            project: asdict(instance.projects[project])
            for project in sorted(instance.projects)
        },
        "votes": [asdict(vote) for vote in instance.votes],
    }


def _development_view_payload(
    rows: Sequence[_FrozenSeriesData],
) -> list[dict[str, object]]:
    return [
        {
            "series": row.ref.key,
            "ref_years": list(row.ref.years),
            "train_years": list(row.train_years),
            "scored_years": list(row.test_years),
            "warmup_years": [
                year for year in sorted(row.all_years) if year not in row.test_years
            ],
            "train_only_instances": {
                str(year): _canonical_sha256(
                    _frozen_instance_payload(row.train_only[year])
                )
                for year in sorted(row.train_only)
            },
            "instances": {
                str(year): _canonical_sha256(
                    _frozen_instance_payload(row.all_years[year])
                )
                for year in sorted(row.all_years)
            },
        }
        for row in rows
    ]


def _fold_input_payload(fold: DevelopmentFold) -> dict[str, object]:
    return {
        "schema_version": 1,
        "fold": fold.name,
        "kind": fold.kind,
        "source_manifest_sha256": fold.source_manifest_sha256,
        "provenance": _ordinary_payload(fold.provenance),
        "training_view": _development_view_payload(fold.train),
        "scored_view": _development_view_payload(fold.test),
    }


def _fold_input_sha256(fold: DevelopmentFold) -> str:
    return _canonical_sha256(_fold_input_payload(fold))


def _read_authenticated_development_artifact(path: Path) -> dict[str, object]:
    try:
        encoded = _read_regular_bytes(path, "immutable development artifact")
        payload = json.loads(encoded.decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("artifact payload must be a mapping")
        embedded = payload["payload_sha256"]
        unsigned = dict(payload)
        unsigned.pop("payload_sha256")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"divergent immutable development artifact: {path}") from exc
    if embedded != _canonical_sha256(unsigned):
        raise RuntimeError(f"divergent immutable development artifact: {path}")
    if encoded != _canonical_json_bytes(payload):
        raise RuntimeError(f"divergent immutable development artifact: {path}")
    return payload


def _expected_safe_static_from_fit(
    fit: FoldFit | Mapping[str, object], result: Mapping[str, object]
) -> list[dict[str, str]]:
    if isinstance(fit, FoldFit):
        return [
            {
                "identity": anchor_payload_sha256(candidate.anchor),
                "family": candidate.anchor.family,
            }
            for candidate in fit.anchor_fit.candidates
            if candidate.safe
        ]
    safe_static = result.get("safe_static", {})
    if not isinstance(safe_static, Mapping):
        raise RuntimeError("development fold safe-static results must be a mapping")
    return [
        {"identity": identity, "family": safe_static[identity]["family"]}
        for identity in sorted(safe_static)
    ]


def _authenticate_development_actuation(
    payload: object,
    fold: DevelopmentFold,
    rows: Sequence[DevelopmentSeriesResult],
    primary_seed: int,
) -> dict[str, bool]:
    from iclr_residual_actuation import ActuationRecord

    keys = {
        "instance",
        "actuated",
        "first_grid_t",
        "refined_low",
        "refined_high",
        "csd_direction",
        "anchor_csd",
        "changed_csd",
    }
    if not isinstance(payload, list):
        raise RuntimeError("development actuation scan must be a list")
    parsed = []
    for raw in payload:
        if not isinstance(raw, Mapping) or set(raw) != keys:
            raise RuntimeError("development actuation record schema differs")
        if any(
            isinstance(raw[name], Real)
            and not isinstance(raw[name], bool)
            and float(raw[name]) == 0.0
            and math.copysign(1.0, float(raw[name])) < 0.0
            for name in (
                "first_grid_t",
                "refined_low",
                "refined_high",
                "anchor_csd",
                "changed_csd",
            )
            if raw[name] is not None
        ):
            raise RuntimeError("development actuation record contains signed zero")
        try:
            record = ActuationRecord(**dict(raw))
            canonical = asdict(record)
            exact = _canonical_payload_equal(canonical, dict(raw))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"development actuation record differs: {exc}"
            ) from exc
        if not exact:
            raise RuntimeError("development actuation record payload differs")
        parsed.append(record)

    expected = sorted(
        (row.all_years[year].path, row.ref.key)
        for row in fold.test
        for year in row.test_years
    )
    expected_paths = [path for path, _ in expected]
    if len(expected_paths) != len(set(expected_paths)):
        raise RuntimeError("development scored instance paths must be unique")
    if [record.instance for record in parsed] != expected_paths:
        raise RuntimeError("development actuation scan inventory differs")

    actuated_by_series = {row.ref.key: False for row in fold.test}
    for record, (_, series) in zip(parsed, expected):
        actuated_by_series[series] = (
            actuated_by_series[series] or record.actuated
        )
    for row in rows:
        expected_actuated = (
            row.seed == primary_seed and actuated_by_series[row.series]
        )
        if row.signatures["actuated"] is not expected_actuated:
            raise RuntimeError(
                "development series actuation signature differs from scan"
            )
    return actuated_by_series


def _authenticate_development_scored_relationships(
    rows: Sequence[DevelopmentSeriesResult],
    safe_static: Mapping[str, object],
    fit: FoldFit,
    actuated_by_series: Mapping[str, bool],
) -> None:
    """Bind retained scored values to exact policies and authenticated scan state."""

    if not isinstance(fit, FoldFit):
        raise RuntimeError("development scored relationships require authenticated fit")
    selected_by_seed = {seed.seed: seed.selected for seed in fit.seeds}
    primary_selected = selected_by_seed[fit.primary_seed]
    if _is_canonical_zero(primary_selected) and any(
        actuated_by_series.values()
    ):
        raise RuntimeError(
            "development canonical-zero primary cannot retain actuation"
        )
    rows_by_series: dict[str, list[DevelopmentSeriesResult]] = {}
    for row in rows:
        rows_by_series.setdefault(row.series, []).append(row)

    common_anchor: dict[str, Mapping[str, object]] = {}
    common_mes: dict[str, Mapping[str, object]] = {}
    for series, series_rows in rows_by_series.items():
        anchor = series_rows[0].anchor
        mes = series_rows[0].mes
        if any(
            not _canonical_payload_equal(dict(row.anchor), dict(anchor))
            for row in series_rows[1:]
        ):
            raise RuntimeError(
                f"development scored anchor metrics differ across seeds: {series}"
            )
        if any(
            not _canonical_payload_equal(dict(row.mes), dict(mes))
            for row in series_rows[1:]
        ):
            raise RuntimeError(
                f"development scored MES metrics differ across seeds: {series}"
            )
        common_anchor[series] = anchor
        common_mes[series] = mes

        policy_groups: dict[bytes, list[DevelopmentSeriesResult]] = {}
        for row in series_rows:
            selected = selected_by_seed[row.seed]
            if _is_canonical_zero(selected) and not _canonical_payload_equal(
                dict(row.residual), dict(row.anchor)
            ):
                raise RuntimeError(
                    f"development scored canonical-zero residual differs from anchor: {series}"
                )
            if (
                row.seed == fit.primary_seed
                and not actuated_by_series[series]
                and not _canonical_payload_equal(
                    dict(row.residual), dict(row.anchor)
                )
            ):
                raise RuntimeError(
                    f"development scored primary inert residual differs from anchor: {series}"
                )
            policy_key = _canonical_json_bytes(
                {"weights": list(selected.weights)}
            )
            policy_groups.setdefault(policy_key, []).append(row)
        for policy_rows in policy_groups.values():
            expected = policy_rows[0].residual
            if any(
                not _canonical_payload_equal(dict(row.residual), dict(expected))
                for row in policy_rows[1:]
            ):
                raise RuntimeError(
                    f"development scored equal policy weights have unequal residual metrics: {series}"
                )

    safe_candidates = tuple(
        candidate for candidate in fit.anchor_fit.candidates if candidate.safe
    )
    candidate_by_identity = {
        anchor_payload_sha256(candidate.anchor): candidate
        for candidate in safe_candidates
    }
    if set(candidate_by_identity) != set(safe_static):
        raise RuntimeError("development scored safe-static fit inventory differs")
    selected_identity = anchor_payload_sha256(fit.anchor_fit.selected.anchor)
    mes_identities = [
        identity
        for identity, candidate in candidate_by_identity.items()
        if candidate.anchor.family == "mes"
    ]
    if len(mes_identities) != 1:
        raise RuntimeError("development scored safe-static MES identity differs")
    mes_identity = mes_identities[0]
    for series in rows_by_series:
        selected_metrics = safe_static[selected_identity]["series"][series]
        if not _canonical_payload_equal(
            dict(selected_metrics), dict(common_anchor[series])
        ):
            raise RuntimeError(
                f"development scored selected static anchor metrics differ: {series}"
            )
        mes_metrics = safe_static[mes_identity]["series"][series]
        if not _canonical_payload_equal(
            dict(mes_metrics), dict(common_mes[series])
        ):
            raise RuntimeError(
                f"development scored MES static metrics differ: {series}"
            )

    static_policy_groups: dict[tuple[float, ...], list[str]] = {}
    for identity, candidate in candidate_by_identity.items():
        static_policy_groups.setdefault(
            _static_policy_signature(candidate.anchor), []
        ).append(identity)
    for identities in static_policy_groups.values():
        for series in rows_by_series:
            expected = safe_static[identities[0]]["series"][series]
            if any(
                not _canonical_payload_equal(
                    dict(safe_static[identity]["series"][series]),
                    dict(expected),
                )
                for identity in identities[1:]
            ):
                raise RuntimeError(
                    f"development scored safe-static policy twins differ: {series}"
                )


def _validate_development_fold_result(
    result: object,
    fold: DevelopmentFold,
    expected_safe_static: Sequence[Mapping[str, str]],
    fit: FoldFit,
) -> dict[str, object]:
    if not isinstance(result, Mapping):
        raise RuntimeError("development fold result must be a mapping")
    required = {
        "fold",
        "kind",
        "primary_seed",
        "fit_training_series",
        "scored_series",
        "series_results",
        "safe_static",
        "actuation",
    }
    if set(result) != required:
        raise RuntimeError("development fold result schema differs")
    training = [row.ref.key for row in fold.train]
    scored = [row.ref.key for row in fold.test]
    if (
        result["fold"] != fold.name
        or result["kind"] != fold.kind
        or result["fit_training_series"] != training
        or result["scored_series"] != scored
        or isinstance(result["primary_seed"], bool)
        or not isinstance(result["primary_seed"], int)
        or result["primary_seed"] not in _DEVELOPMENT_SEEDS
    ):
        raise RuntimeError("development fold result identity differs")
    retained_rows = result["series_results"]
    row_keys = {
        "fold",
        "seed",
        "series",
        "city",
        "cluster",
        "residual",
        "anchor",
        "mes",
        "signatures",
    }
    if (
        not isinstance(retained_rows, list)
        or any(
            not isinstance(row, Mapping) or set(row) != row_keys
            for row in retained_rows
        )
    ):
        raise RuntimeError("development fold result row schema differs")
    try:
        rows = tuple(_coerce_development_result(row) for row in retained_rows)
    except (TypeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"development fold result rows differ: {exc}") from exc
    if any(
        not _canonical_payload_equal(
            _development_result_payload(row), dict(retained)
        )
        for row, retained in zip(rows, retained_rows)
    ):
        raise RuntimeError("development fold result row payload differs")
    expected_keys = [
        (fold.name, seed, series)
        for seed in _DEVELOPMENT_SEEDS
        for series in scored
    ]
    keys = [(row.fold, row.seed, row.series) for row in rows]
    if keys != expected_keys:
        raise RuntimeError("development fold result order or inventory differs")
    actuated_by_series = _authenticate_development_actuation(
        result["actuation"], fold, rows, result["primary_seed"]
    )
    expected_candidates = {
        row["identity"]: row["family"] for row in expected_safe_static
    }
    supplied = result["safe_static"]
    if not isinstance(supplied, Mapping) or set(supplied) != set(
        expected_candidates
    ):
        raise RuntimeError("development fold safe-static inventory differs")
    for identity, family in expected_candidates.items():
        candidate = supplied[identity]
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"family", "series"}
            or candidate["family"] != family
            or not isinstance(candidate["series"], Mapping)
            or set(candidate["series"]) != set(scored)
        ):
            raise RuntimeError("development fold safe-static series differ")
        try:
            for metrics in candidate["series"].values():
                snapshot = _metric_snapshot(metrics, "safe_static")
                if not _canonical_payload_equal(dict(snapshot), dict(metrics)):
                    raise ValueError(
                        "safe-static metric payload is not canonical"
                    )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"development fold safe-static metrics differ: {exc}"
            ) from exc
    _authenticate_development_scored_relationships(
        rows, supplied, fit, actuated_by_series
    )
    return _ordinary_payload(result)


def _authenticated_static_anchor(payload: object) -> AnchorSpec:
    keys = {"alpha", "family", "free_logits", "reference_cell"}
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development static anchor schema differs")
    family = payload["family"]
    logits = payload["free_logits"]
    reference = payload["reference_cell"]
    alpha = payload["alpha"]
    if (
        not isinstance(family, str)
        or family not in {"mes", "senior", "age", "age_sex"}
        or not isinstance(logits, list)
        or any(type(value) is not float or not math.isfinite(value) for value in logits)
        or (reference is not None and not isinstance(reference, str))
        or (
            alpha is not None
            and (type(alpha) is not float or not math.isfinite(alpha))
        )
    ):
        raise RuntimeError("development static anchor payload differs")
    try:
        anchor = AnchorSpec(family, logits, reference)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("development static anchor payload differs") from exc
    if not _canonical_payload_equal(_anchor_payload(anchor), dict(payload)):
        raise RuntimeError("development static anchor payload differs")
    return anchor


def _authenticated_static_city_metrics(
    payload: object, name: str
) -> dict[str, dict[str, float]]:
    metric_keys = {"city_csd", "city_welfare", "city_exclusion"}
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"development {name} metrics schema differs")
    result = {}
    for city, metrics in payload.items():
        if (
            not isinstance(city, str)
            or not city
            or not isinstance(metrics, Mapping)
            or set(metrics) != metric_keys
        ):
            raise RuntimeError(f"development {name} metrics schema differs")
        normalized = {}
        for key in metric_keys:
            value = metrics[key]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError(f"development {name} metrics payload differs")
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError(f"development {name} metrics payload differs")
            normalized[key] = number
        result[city] = normalized
    return result


def _authenticated_static_candidate(payload: object) -> AnchorCandidate:
    keys = {
        "anchor",
        "anchor_sha256",
        "metrics",
        "objective",
        "safe",
        "seed",
        "source",
    }
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development static candidate schema differs")
    anchor = _authenticated_static_anchor(payload["anchor"])
    try:
        identity = _sha256_identity(
            payload["anchor_sha256"], "development static anchor"
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("development static candidate identity differs") from exc
    if identity != anchor_payload_sha256(anchor):
        raise RuntimeError("development static candidate anchor identity differs")
    metrics_payload = _authenticated_static_city_metrics(
        payload["metrics"], "static candidate"
    )
    metrics = {
        city: CityMetrics(**values) for city, values in metrics_payload.items()
    }
    if type(payload["safe"]) is not bool:
        raise RuntimeError("development static candidate safety differs")
    safe = payload["safe"]
    objective = payload["objective"]
    if objective is None:
        if safe:
            raise RuntimeError("development static candidate objective differs")
        objective = float("inf")
    else:
        if isinstance(objective, bool) or not isinstance(objective, Real):
            raise RuntimeError("development static candidate objective differs")
        objective = float(objective)
        if not math.isfinite(objective) or not safe:
            raise RuntimeError("development static candidate objective differs")
    seed = payload["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise RuntimeError("development static candidate seed differs")
    if not isinstance(payload["source"], str) or not payload["source"]:
        raise RuntimeError("development static candidate source differs")
    candidate = AnchorCandidate(
        anchor=anchor,
        objective=objective,
        metrics=metrics,
        safe=safe,
        source=payload["source"],
        seed=seed,
    )
    if not _canonical_payload_equal(_candidate_payload(candidate), dict(payload)):
        raise RuntimeError("development static candidate payload differs")
    return candidate


def _anchor_search_config_payload(config: AnchorSearchConfig) -> dict[str, object]:
    return {**asdict(config), "seeds": list(config.seeds)}


def _authenticate_static_search_config(
    payload: object, expected: AnchorSearchConfig
) -> AnchorSearchConfig:
    if not isinstance(expected, AnchorSearchConfig):
        raise TypeError("expected static config must be an AnchorSearchConfig")
    keys = set(asdict(anchor_search_config()))
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development static config schema differs")
    seeds = payload["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise RuntimeError("development static config seeds differ")
    for name in ("generations", "popsize"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"development static config {name} differs")
    normalized: dict[str, object] = {
        "seeds": list(seeds),
        "generations": payload["generations"],
        "popsize": payload["popsize"],
    }
    for name in (
        "sigma0",
        "bound",
        "tau",
        "welfare_floor",
        "exclusion_delta",
        "tie_tolerance",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(f"development static config {name} differs")
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"development static config {name} differs")
        if name in {"sigma0", "bound", "tau", "tie_tolerance"} and number <= 0.0:
            raise RuntimeError(f"development static config {name} differs")
        normalized[name] = number
    if not _canonical_payload_equal(
        normalized, _anchor_search_config_payload(expected)
    ):
        raise RuntimeError("development static config differs from expected context")
    return expected


def _static_policy_signature(anchor: AnchorSpec) -> tuple[float, ...]:
    """Expand one anchor to its exact canonical age-by-sex log policy."""

    logits = anchor_logit_map(anchor)
    values = [0.0]  # frozen zero fallback for missing demographic metadata
    for cell in AGE_SEX_CELLS:
        if anchor.family == "mes":
            value = 0.0
        elif anchor.family in {"senior", "age"}:
            value = logits.get(cell.split("|", 1)[0], 0.0)
        else:
            value = logits.get(cell, 0.0)
        values.append(0.0 if value == 0.0 else float(value))
    return tuple(values)


def _authenticate_static_provenance(
    candidates: Sequence[AnchorCandidate], config: AnchorSearchConfig
) -> None:
    def evaluation_payload(candidate: AnchorCandidate) -> dict[str, object]:
        payload = _candidate_payload(candidate)
        return {
            "metrics": payload["metrics"],
            "objective": payload["objective"],
            "safe": candidate.safe,
        }

    families = tuple(candidate.anchor.family for candidate in candidates)
    family_order = {family: index for index, family in enumerate(
        ("mes", "senior", "age", "age_sex")
    )}
    if (
        set(families) != set(family_order)
        or tuple(family_order[family] for family in families)
        != tuple(sorted(family_order[family] for family in families))
    ):
        raise RuntimeError("development static family archive differs")

    mes = [candidate for candidate in candidates if candidate.anchor.family == "mes"]
    if (
        len(mes) != 1
        or mes[0].source != "mes-fallback"
        or mes[0].seed is not None
        or not mes[0].safe
        or not math.isfinite(mes[0].objective)
    ):
        raise RuntimeError("development static MES provenance differs")

    senior = [
        candidate for candidate in candidates if candidate.anchor.family == "senior"
    ]
    expected_senior = tuple(
        (
            _anchor_payload(
                AnchorSpec("senior", (math.log1p(alpha),), AGE_REFERENCE)
            ),
            f"senior-grid-alpha-{alpha:.17g}",
        )
        for alpha in senior_anchor_grid()
    )
    if len(senior) != len(expected_senior) or any(
        candidate.seed is not None
        or candidate.source != expected_source
        or not _canonical_payload_equal(
            _anchor_payload(candidate.anchor), expected_anchor
        )
        for candidate, (expected_anchor, expected_source) in zip(
            senior, expected_senior
        )
    ):
        raise RuntimeError("development static senior grid provenance differs")

    alpha_zero = senior[0]
    if not _canonical_payload_equal(
        evaluation_payload(mes[0]), evaluation_payload(alpha_zero)
    ):
        raise RuntimeError(
            "development static MES and alpha-zero equivalent evaluations differ"
        )

    pattern = re.compile(
        r"^(age|age_sex)-seed-([0-9]+)-(initializer|final|generation-([0-9]+))$"
    )
    family_rows: dict[str, list[AnchorCandidate]] = {}
    for family in ("age", "age_sex"):
        rows = [candidate for candidate in candidates if candidate.anchor.family == family]
        if not rows:
            raise RuntimeError(f"development static {family} archive is empty")
        family_rows[family] = rows
        expected_initializer = f"{family}-seed-{config.seeds[0]}-initializer"
        possible_sources = tuple(
            source
            for seed in config.seeds
            for source in (
                f"{family}-seed-{seed}-initializer",
                *(
                    f"{family}-seed-{seed}-generation-{generation}"
                    for generation in range(config.generations)
                ),
                f"{family}-seed-{seed}-final",
            )
        )
        source_rank = {source: index for index, source in enumerate(possible_sources)}
        observed_sources = tuple(candidate.source for candidate in rows)
        if (
            observed_sources[0] != expected_initializer
            or len(observed_sources) != len(set(observed_sources))
            or any(source not in source_rank for source in observed_sources)
            or tuple(source_rank[source] for source in observed_sources)
            != tuple(sorted(source_rank[source] for source in observed_sources))
            or any(
                source.endswith("-initializer") and source != expected_initializer
                for source in observed_sources
            )
        ):
            raise RuntimeError(
                f"development static {family} source lineage differs"
            )
        for candidate in rows:
            match = pattern.fullmatch(candidate.source)
            if match is None or match.group(1) != family:
                raise RuntimeError(
                    f"development static {family} source provenance differs"
                )
            source_seed = int(match.group(2))
            if (
                candidate.seed is None
                or source_seed != candidate.seed
                or candidate.seed not in config.seeds
                or match.group(2) != str(candidate.seed)
            ):
                raise RuntimeError(
                    f"development static {family} seed provenance differs"
                )
            generation = match.group(4)
            if match.group(3) == "final":
                raise RuntimeError(
                    f"development static {family} final source is unreachable"
                )
            if generation is not None and not (
                generation == str(int(generation))
                and 0 <= int(generation) < config.generations
            ):
                raise RuntimeError(
                    f"development static {family} generation provenance differs"
                )
            if generation is not None and any(
                value < -config.bound or value > config.bound
                for value in candidate.anchor.free_logits
            ):
                raise RuntimeError(
                    f"development static {family} CMA bound differs"
                )

    try:
        expected_age = age_initializer(senior)
        expected_age_sex = age_sex_initializer(family_rows["age"])
    except ValueError as exc:
        raise RuntimeError("development static initializer lineage differs") from exc
    if not _canonical_payload_equal(
        _anchor_payload(family_rows["age"][0].anchor),
        _anchor_payload(expected_age),
    ):
        raise RuntimeError("development static age initializer anchor differs")
    if not _canonical_payload_equal(
        _anchor_payload(family_rows["age_sex"][0].anchor),
        _anchor_payload(expected_age_sex),
    ):
        raise RuntimeError("development static age-sex initializer anchor differs")
    senior_parent = select_static_anchor(senior, config.tie_tolerance)
    if not _canonical_payload_equal(
        evaluation_payload(senior_parent),
        evaluation_payload(family_rows["age"][0]),
    ):
        raise RuntimeError(
            "development static senior/age initializer evaluations differ"
        )
    age_parent = select_static_anchor(
        family_rows["age"], config.tie_tolerance
    )
    if not _canonical_payload_equal(
        evaluation_payload(age_parent),
        evaluation_payload(family_rows["age_sex"][0]),
    ):
        raise RuntimeError(
            "development static age/age-sex initializer evaluations differ"
        )

    policy_evaluations: dict[tuple[float, ...], dict[str, object]] = {}
    for candidate in candidates:
        signature = _static_policy_signature(candidate.anchor)
        evaluation = evaluation_payload(candidate)
        prior = policy_evaluations.setdefault(signature, evaluation)
        if not _canonical_payload_equal(prior, evaluation):
            raise RuntimeError(
                "development static policy-twin evaluations differ"
            )


def _authenticated_static_fit(
    payload: object,
    expected_config: AnchorSearchConfig,
    training_cities: Sequence[str],
) -> tuple[AnchorFit, list[dict[str, str]]]:
    keys = {
        "schema_version",
        "training_only",
        "config",
        "mes_fallback",
        "families_present",
        "mes_metrics",
        "selected",
        "candidates",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != keys
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != 1
        or payload["training_only"] is not True
    ):
        raise RuntimeError("development static fit schema differs")
    config = _authenticate_static_search_config(payload["config"], expected_config)
    candidates_payload = payload["candidates"]
    if not isinstance(candidates_payload, list) or not candidates_payload:
        raise RuntimeError("development static candidate archive differs")
    candidates = [
        _authenticated_static_candidate(candidate)
        for candidate in candidates_payload
    ]
    expected_cities = tuple(sorted(training_cities))
    if not expected_cities or any(
        tuple(sorted(candidate.metrics)) != expected_cities
        for candidate in candidates
    ):
        raise RuntimeError("development static candidate training cities differ")
    identities = [anchor_payload_sha256(candidate.anchor) for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise RuntimeError("development static candidate identities must be unique")
    selected = _authenticated_static_candidate(payload["selected"])
    selected_payload = _candidate_payload(selected)
    candidate_payloads = [_candidate_payload(candidate) for candidate in candidates]
    if sum(
        _canonical_payload_equal(selected_payload, candidate)
        for candidate in candidate_payloads
    ) != 1:
        raise RuntimeError("development static selected candidate differs")

    family_order = ("mes", "senior", "age", "age_sex")
    families = [
        family
        for family in family_order
        if any(candidate.anchor.family == family for candidate in candidates)
    ]
    if payload["families_present"] != families:
        raise RuntimeError("development static family inventory differs")
    if families != list(family_order):
        raise RuntimeError("development static family inventory is incomplete")
    fallback = payload["mes_fallback"]
    expected_fallback = {
        "available": any(
            candidate.anchor.family == "mes" for candidate in candidates
        ),
        "safe": any(
            candidate.anchor.family == "mes" and candidate.safe
            for candidate in candidates
        ),
    }
    if (
        not isinstance(fallback, Mapping)
        or set(fallback) != {"available", "safe"}
        or type(fallback["available"]) is not bool
        or type(fallback["safe"]) is not bool
        or dict(fallback) != expected_fallback
    ):
        raise RuntimeError("development static MES fallback differs")
    mes_metrics_payload = _authenticated_static_city_metrics(
        payload["mes_metrics"], "static MES"
    )
    mes_metrics = {
        city: CityMetrics(**values) for city, values in mes_metrics_payload.items()
    }
    if tuple(sorted(mes_metrics)) != expected_cities:
        raise RuntimeError("development static MES training cities differ")
    mes_candidates = [
        candidate
        for candidate in candidates
        if candidate.anchor.family == "mes"
    ]
    if len(mes_candidates) != 1 or not _canonical_payload_equal(
        {
            city: asdict(mes_metrics[city]) for city in sorted(mes_metrics)
        },
        _candidate_payload(mes_candidates[0])["metrics"],
    ):
        raise RuntimeError("development static MES metrics differ")

    recomputed = []
    for candidate in candidates:
        complete = bool(candidate.metrics) and set(candidate.metrics) == set(mes_metrics)
        safe = bool(complete)
        if safe:
            for city, metrics in candidate.metrics.items():
                mes = mes_metrics[city]
                if (
                    metrics.city_welfare
                    < config.welfare_floor * mes.city_welfare
                    or metrics.city_exclusion
                    > mes.city_exclusion + config.exclusion_delta
                ):
                    safe = False
                    break
        if safe != candidate.safe:
            raise RuntimeError("development static candidate safety differs")
        objective = (
            soft_worst_city(
                [candidate.metrics[city].city_csd for city in sorted(candidate.metrics)],
                config.tau,
            )
            if safe
            else float("inf")
        )
        authenticated = AnchorCandidate(
            anchor=candidate.anchor,
            objective=objective,
            metrics=candidate.metrics,
            safe=safe,
            source=candidate.source,
            seed=candidate.seed,
        )
        if not _canonical_payload_equal(
            _candidate_payload(authenticated), _candidate_payload(candidate)
        ):
            raise RuntimeError("development static candidate objective differs")
        recomputed.append(authenticated)

    _authenticate_static_provenance(recomputed, config)
    winner = select_static_anchor(recomputed, config.tie_tolerance)
    if not _canonical_payload_equal(_candidate_payload(winner), selected_payload):
        raise RuntimeError("development static selected winner differs")
    fit = AnchorFit(winner, tuple(recomputed), mes_metrics, config)
    return fit, [
        {
            "identity": anchor_payload_sha256(candidate.anchor),
            "family": candidate.anchor.family,
        }
        for candidate in recomputed
        if candidate.safe
    ]


def _residual_search_config_payload(
    config: ResidualSearchConfig,
) -> dict[str, object]:
    return {**asdict(config), "seeds": list(config.seeds)}


def _authenticate_residual_search_config(
    payload: object, expected: ResidualSearchConfig
) -> ResidualSearchConfig:
    keys = set(asdict(residual_search_config()))
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development residual config schema differs")
    seeds = payload["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise RuntimeError("development residual config seeds differ")
    normalized: dict[str, object] = {"seeds": list(seeds)}
    for name in ("generations", "popsize"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"development residual config {name} differs")
        normalized[name] = value
    for name in (
        "sigma0",
        "bound",
        "tau",
        "welfare_floor",
        "replacement_delta",
        "norm_penalty",
    ):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise RuntimeError(f"development residual config {name} differs")
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError(f"development residual config {name} differs")
        normalized[name] = number
    if not _canonical_payload_equal(
        normalized, _residual_search_config_payload(expected)
    ):
        raise RuntimeError("development residual config differs from expected context")
    return expected


def _authenticated_residual_metrics(payload: object) -> ResidualMetrics:
    keys = {"series", "cities", "failed_safety"}
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development residual metrics schema differs")
    failed = payload["failed_safety"]
    if (
        not isinstance(failed, list)
        or any(not isinstance(value, str) or not value for value in failed)
    ):
        raise RuntimeError("development residual failed-safety evidence differs")
    series_payload = payload["series"]
    if not isinstance(series_payload, list) or not series_payload:
        raise RuntimeError("development residual series metrics differ")
    rows = []
    for row in series_payload:
        if not isinstance(row, Mapping) or set(row) != {
            "series", "city", "csd", "welfare", "exclusion"
        }:
            raise RuntimeError("development residual series metrics schema differs")
        series = row["series"]
        city = row["city"]
        if (
            not isinstance(series, str)
            or not series
            or not isinstance(city, str)
            or not city
        ):
            raise RuntimeError("development residual series identity differs")

        def raw_number(name: str) -> float | None:
            value = row[name]
            if value is None:
                if name == "csd":
                    if f"{series}: non-finite CSD" in failed:
                        return float("nan")
                    return None
                return float("nan")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError("development residual series metric differs")
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("development residual series metric differs")
            return number

        rows.append(
            SeriesMetrics(
                series,
                city,
                raw_number("csd"),
                raw_number("welfare"),
                raw_number("exclusion"),
            )
        )
    cities_payload = payload["cities"]
    if not isinstance(cities_payload, Mapping):
        raise RuntimeError("development residual city metrics differ")
    cities = {}
    for city, metrics in cities_payload.items():
        if (
            not isinstance(city, str)
            or not city
            or not isinstance(metrics, Mapping)
            or set(metrics) != {"city_csd", "city_welfare", "city_exclusion"}
        ):
            raise RuntimeError("development residual city metrics schema differs")
        values = []
        for name in ("city_csd", "city_welfare", "city_exclusion"):
            value = metrics[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise RuntimeError("development residual city metric differs")
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("development residual city metric differs")
            values.append(number)
        cities[city] = CityMetrics(*values)
    try:
        result = ResidualMetrics(tuple(rows), cities, tuple(failed))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("development residual metrics differ") from exc
    if not _canonical_payload_equal(_residual_metrics_payload(result), dict(payload)):
        raise RuntimeError("development residual metrics payload differs")
    return result


def _authenticated_residual_candidate(payload: object) -> ResidualCandidate:
    keys = {"weights", "endpoint", "amplitude", "loss", "metrics", "safe", "source"}
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development residual candidate schema differs")

    def vector(name: str) -> tuple[float, ...]:
        values = payload[name]
        if (
            not isinstance(values, list)
            or len(values) != len(FEATURE_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise RuntimeError(f"development residual candidate {name} differs")
        return tuple(float(value) for value in values)

    amplitude = payload["amplitude"]
    if (
        isinstance(amplitude, bool)
        or not isinstance(amplitude, Real)
        or not math.isfinite(float(amplitude))
    ):
        raise RuntimeError("development residual candidate amplitude differs")
    loss_value = payload["loss"]
    if loss_value is None:
        loss = float("inf")
    elif (
        isinstance(loss_value, bool)
        or not isinstance(loss_value, Real)
        or not math.isfinite(float(loss_value))
    ):
        raise RuntimeError("development residual candidate loss differs")
    else:
        loss = float(loss_value)
    if type(payload["safe"]) is not bool:
        raise RuntimeError("development residual candidate safety differs")
    if not isinstance(payload["source"], str) or not payload["source"]:
        raise RuntimeError("development residual candidate source differs")
    try:
        candidate = ResidualCandidate(
            vector("weights"),
            vector("endpoint"),
            float(amplitude),
            loss,
            _authenticated_residual_metrics(payload["metrics"]),
            payload["safe"],
            payload["source"],
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("development residual candidate differs") from exc
    if not _canonical_payload_equal(
        _residual_candidate_payload(candidate), dict(payload)
    ):
        raise RuntimeError("development residual candidate payload differs")
    return candidate


def _authenticate_development_scaler(payload: object) -> FeatureScaler:
    keys = {
        "feature_names", "mean", "scale", "clip", "row_count",
        "instance_count", "sha256"
    }
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development residual scaler schema differs")
    names = payload["feature_names"]
    mean = payload["mean"]
    scale = payload["scale"]
    if names != list(FEATURE_NAMES):
        raise RuntimeError("development residual scaler feature names differ")
    for label, values in (("mean", mean), ("scale", scale)):
        if (
            not isinstance(values, list)
            or len(values) != len(FEATURE_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise RuntimeError(f"development residual scaler {label} differs")
    clip = payload["clip"]
    if (
        isinstance(clip, bool)
        or not isinstance(clip, Real)
        or not math.isfinite(float(clip))
        or float(clip) <= 0.0
    ):
        raise RuntimeError("development residual scaler clip differs")
    for name in ("row_count", "instance_count"):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"development residual scaler {name} differs")
    try:
        scaler = FeatureScaler(
            names, mean, scale, float(clip), payload["row_count"], payload["instance_count"]
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("development residual scaler differs") from exc
    normalized = {
        "feature_names": list(scaler.feature_names),
        "mean": scaler.mean.tolist(),
        "scale": scaler.scale.tolist(),
        "clip": scaler.clip,
        "row_count": scaler.row_count,
        "instance_count": scaler.instance_count,
        "sha256": scaler.sha256,
    }
    if not _canonical_payload_equal(normalized, dict(payload)):
        raise RuntimeError("development residual scaler identity differs")
    return scaler


def _authenticated_residual_fit(
    payload: object,
    static_payload: Mapping[str, object],
    anchor_fit: AnchorFit,
    expected_config: ResidualSearchConfig,
    training_series: Sequence[str],
    training_cities: Sequence[str],
) -> FoldFit:
    keys = {
        "schema_version", "training_only", "static_artifact_sha256", "config",
        "anchor", "scaler", "seeds", "primary_seed", "primary",
        "synthetic_actuated", "payload_sha256"
    }
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise RuntimeError("development residual fit schema differs")
    unsigned = dict(payload)
    embedded_sha = unsigned.pop("payload_sha256")
    if embedded_sha != _canonical_sha256(unsigned):
        raise RuntimeError("development residual fit digest differs")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != 2
        or payload["training_only"] is not True
    ):
        raise RuntimeError("development residual fit training schema differs")
    if payload["static_artifact_sha256"] != _canonical_sha256(static_payload):
        raise RuntimeError("development residual static link differs")
    config = _authenticate_residual_search_config(payload["config"], expected_config)
    if not _canonical_payload_equal(
        dict(payload["anchor"]), _candidate_payload(anchor_fit.selected)
    ):
        raise RuntimeError("development residual anchor differs from static selection")
    scaler = _authenticate_development_scaler(payload["scaler"])
    seed_payloads = payload["seeds"]
    if not isinstance(seed_payloads, list) or len(seed_payloads) != len(config.seeds):
        raise RuntimeError("development residual seed archive differs")
    if [row.get("seed") if isinstance(row, Mapping) else None for row in seed_payloads] != list(config.seeds):
        raise RuntimeError("development residual seed order differs")

    parsed_rows = []
    for expected_seed, row in zip(config.seeds, seed_payloads):
        if not isinstance(row, Mapping) or set(row) != {
            "seed", "selected", "archive", "archive_count", "optimizer_history"
        }:
            raise RuntimeError("development residual seed schema differs")
        if (
            isinstance(row["seed"], bool)
            or not isinstance(row["seed"], int)
            or row["seed"] != expected_seed
        ):
            raise RuntimeError("development residual seed differs")
        archive_payload = row["archive"]
        if (
            not isinstance(archive_payload, list)
            or not archive_payload
            or isinstance(row["archive_count"], bool)
            or not isinstance(row["archive_count"], int)
            or row["archive_count"] != len(archive_payload)
        ):
            raise RuntimeError("development residual archive count differs")
        archive = tuple(_authenticated_residual_candidate(item) for item in archive_payload)
        selected = _authenticated_residual_candidate(row["selected"])
        if sum(
            _canonical_payload_equal(dict(row["selected"]), dict(item))
            for item in archive_payload
        ) != 1:
            raise RuntimeError("development residual selected membership differs")
        history = row["optimizer_history"]
        if not isinstance(history, list) or any(not isinstance(item, Mapping) for item in history):
            raise RuntimeError("development residual optimizer history differs")
        try:
            frozen_history = tuple(_ImmutableRecord(item) for item in history)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("development residual optimizer history differs") from exc
        if not _canonical_payload_equal(
            {"history": [_ordinary_payload(item) for item in frozen_history]},
            {"history": history},
        ):
            raise RuntimeError("development residual optimizer history payload differs")
        parsed_rows.append((expected_seed, selected, archive, frozen_history))

    expected_series = tuple(training_series)
    expected_cities = tuple(sorted(training_cities))
    zero_metrics_payload = None
    anchor_metrics = None
    for _, _, archive, _ in parsed_rows:
        zeros = [candidate for candidate in archive if _is_canonical_zero(candidate)]
        if len(zeros) != 1 or not zeros[0].safe or not math.isfinite(zeros[0].loss):
            raise RuntimeError("development residual canonical zero differs")
        current = _residual_metrics_payload(zeros[0].metrics)
        if zero_metrics_payload is None:
            zero_metrics_payload = current
            anchor_metrics = zeros[0].metrics
        elif not _canonical_payload_equal(current, zero_metrics_payload):
            raise RuntimeError("development residual zero references differ across seeds")
    assert anchor_metrics is not None
    if not _canonical_payload_equal(
        {
            city: asdict(anchor_metrics.cities[city])
            for city in sorted(anchor_metrics.cities)
        },
        {
            city: asdict(anchor_fit.selected.metrics[city])
            for city in sorted(anchor_fit.selected.metrics)
        },
    ):
        raise RuntimeError("development residual zero differs from static anchor metrics")

    authenticated_seed_fits = []
    for seed, selected, archive, history in parsed_rows:
        recomputed_archive = []
        for candidate in archive:
            if (
                tuple(row.series for row in candidate.metrics.series) != expected_series
                or tuple(sorted(candidate.metrics.cities)) != expected_cities
            ):
                raise RuntimeError("development residual training inventory differs")
            base_metrics = ResidualMetrics(
                candidate.metrics.series, candidate.metrics.cities, ()
            )
            recomputed = score_residual_candidate(
                candidate.weights,
                candidate.endpoint,
                candidate.amplitude,
                base_metrics,
                anchor_metrics,
                anchor_fit.mes_metrics,
                candidate.source,
            )
            if not _canonical_payload_equal(
                _residual_candidate_payload(recomputed),
                _residual_candidate_payload(candidate),
            ):
                raise RuntimeError("development residual candidate scoring differs")
            recomputed_archive.append(recomputed)
        winner = select_residual_candidate(recomputed_archive, config.replacement_delta)
        if not _canonical_payload_equal(
            _residual_candidate_payload(winner),
            _residual_candidate_payload(selected),
        ):
            raise RuntimeError("development residual selected winner differs")
        try:
            seed_fit = ResidualSeedFit(seed, selected, tuple(recomputed_archive), history)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("development residual seed fit differs") from exc
        authenticated_seed_fits.append(seed_fit)

    primary_seed = select_primary_seed(authenticated_seed_fits)
    if (
        isinstance(payload["primary_seed"], bool)
        or not isinstance(payload["primary_seed"], int)
        or payload["primary_seed"] != primary_seed
    ):
        raise RuntimeError("development residual primary seed differs")
    primary = next(row for row in authenticated_seed_fits if row.seed == primary_seed)
    if not _canonical_payload_equal(
        dict(payload["primary"]), _residual_candidate_payload(primary.selected)
    ):
        raise RuntimeError("development residual primary payload differs")
    if (
        type(payload["synthetic_actuated"]) is not bool
        or payload["synthetic_actuated"]
        != _has_nonzero_policy_weights(primary.selected)
    ):
        raise RuntimeError("development residual actuation payload differs")
    try:
        fit = FoldFit(
            anchor_fit, scaler, tuple(authenticated_seed_fits), primary_seed, config
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("development residual fold fit differs") from exc
    rebuilt = _residual_fit_payload(fit, _canonical_sha256(static_payload))
    if not _canonical_payload_equal(rebuilt, dict(payload)):
        raise RuntimeError("development residual canonical payload differs")
    return fit


def _validate_development_fit_payload(
    payload: object,
    primary_seed: int,
    expected_safe_static: Sequence[Mapping[str, str]],
    expected_anchor_config: AnchorSearchConfig,
    expected_residual_config: ResidualSearchConfig | None = None,
    training_series: Sequence[str] = (),
    training_cities: Sequence[str] = (),
    *,
    allow_lightweight: bool = False,
) -> FoldFit | None:
    if (
        isinstance(primary_seed, bool)
        or not isinstance(primary_seed, int)
        or not isinstance(payload, Mapping)
        or payload.get("primary_seed") != primary_seed
    ):
        raise RuntimeError("development fit primary seed differs")
    if "static" not in payload and "residual" not in payload:
        if (
            not allow_lightweight
            or set(payload) != {"primary_seed"}
            or expected_safe_static
            or training_series
            or training_cities
        ):
            raise RuntimeError("development fit payload schema differs")
        return None
    if set(payload) != {"primary_seed", "static", "residual"}:
        raise RuntimeError("development fit payload schema differs")
    static = payload["static"]
    if expected_residual_config is None:
        expected_residual_config = residual_search_config()
    anchor_fit, derived = _authenticated_static_fit(
        static, expected_anchor_config, training_cities
    )
    if derived != list(expected_safe_static):
        raise RuntimeError("development fit safe-static inventory differs")
    fit = _authenticated_residual_fit(
        payload["residual"],
        static,
        anchor_fit,
        expected_residual_config,
        training_series,
        training_cities,
    )
    if fit.primary_seed != primary_seed:
        raise RuntimeError("development fit primary seed differs")
    return fit


def _authenticate_development_artifact_pair(
    fold: DevelopmentFold,
    fit_payload: Mapping[str, object],
    series_payload: Mapping[str, object],
    expected_anchor_config: AnchorSearchConfig,
    expected_residual_config: ResidualSearchConfig | None = None,
) -> tuple[dict[str, object], tuple[dict[str, str], ...]]:
    fit_keys = {
        "schema_version",
        "fold",
        "kind",
        "source_manifest_sha256",
        "provenance",
        "fold_input",
        "fold_input_sha256",
        "primary_seed",
        "expected_safe_static",
        "fit",
        "payload_sha256",
    }
    series_keys = {
        "schema_version",
        "fold",
        "kind",
        "source_manifest_sha256",
        "provenance",
        "fold_input",
        "fold_input_sha256",
        "primary_seed",
        "expected_safe_static",
        "fit_payload_sha256",
        "result",
        "payload_sha256",
    }
    if (
        not isinstance(fit_payload, Mapping)
        or not isinstance(series_payload, Mapping)
        or set(fit_payload) != fit_keys
        or set(series_payload) != series_keys
    ):
        raise RuntimeError("divergent development artifact schema")
    for record in (fit_payload, series_payload):
        unsigned = dict(record)
        embedded = unsigned.pop("payload_sha256")
        if embedded != _canonical_sha256(unsigned):
            raise RuntimeError("divergent development artifact payload identity")
    fold_input = _fold_input_payload(fold)
    fold_input_sha = _canonical_sha256(fold_input)
    expected_provenance = _ordinary_payload(fold.provenance)
    for record in (fit_payload, series_payload):
        retained_fold_input = record.get("fold_input")
        retained_provenance = record.get("provenance")
        if (
            type(record.get("schema_version")) is not int
            or record.get("schema_version") != 2
            or record.get("fold") != fold.name
            or record.get("kind") != fold.kind
            or record.get("source_manifest_sha256") != fold.source_manifest_sha256
            or not isinstance(retained_provenance, Mapping)
            or not _canonical_payload_equal(
                dict(retained_provenance), dict(expected_provenance)
            )
            or not isinstance(retained_fold_input, Mapping)
            or record.get("fold_input_sha256")
            != _canonical_sha256(retained_fold_input)
            or not _canonical_payload_equal(
                dict(retained_fold_input), fold_input
            )
            or record.get("fold_input_sha256") != fold_input_sha
        ):
            raise RuntimeError(
                f"divergent immutable development artifact: {fold.name}"
            )
    if series_payload.get("fit_payload_sha256") != fit_payload.get(
        "payload_sha256"
    ):
        raise RuntimeError(
            f"divergent immutable development artifact link: {fold.name}"
        )
    if not _canonical_payload_equal(
        {
            "primary_seed": fit_payload["primary_seed"],
            "expected_safe_static": fit_payload["expected_safe_static"],
        },
        {
            "primary_seed": series_payload["primary_seed"],
            "expected_safe_static": series_payload["expected_safe_static"],
        },
    ):
        raise RuntimeError("divergent development artifact fit/result metadata")
    raw_static = fit_payload["expected_safe_static"]
    if not isinstance(raw_static, (list, tuple)):
        raise RuntimeError("development fit safe-static inventory differs")
    try:
        safe_static = tuple(
            {"identity": row["identity"], "family": row["family"]}
            for row in raw_static
            if isinstance(row, Mapping) and set(row) == {"identity", "family"}
        )
    except (KeyError, TypeError) as exc:
        raise RuntimeError("development fit safe-static inventory differs") from exc
    if len(safe_static) != len(raw_static) or any(
        not isinstance(row["identity"], str)
        or not row["identity"]
        or not isinstance(row["family"], str)
        or not row["family"]
        for row in safe_static
    ):
        raise RuntimeError("development fit safe-static inventory differs")
    training_series = tuple(row.ref.key for row in fold.train)
    training_cities = tuple(
        sorted({"/".join(series.split("/")[:2]) for series in training_series})
    )
    authenticated_fit = _validate_development_fit_payload(
        fit_payload["fit"],
        fit_payload["primary_seed"],
        safe_static,
        expected_anchor_config,
        residual_search_config()
        if expected_residual_config is None
        else expected_residual_config,
        training_series,
        training_cities,
    )
    if authenticated_fit is None:
        raise RuntimeError("development artifact pair requires a complete fit")
    result = _validate_development_fold_result(
        series_payload["result"], fold, safe_static, authenticated_fit
    )
    if result["primary_seed"] != fit_payload["primary_seed"]:
        raise RuntimeError("divergent development artifact primary seed")
    return result, safe_static


def _fit_and_evaluate_development_fold(
    fold: DevelopmentFold,
    workers: int,
    *,
    root: Path,
    smoke: bool,
    expected_anchor_config: AnchorSearchConfig,
    expected_residual_config: ResidualSearchConfig,
    allow_lightweight: bool = False,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    fit_path = root / "folds" / fold.name / "fit.json"
    series_path = root / "folds" / fold.name / "per_series.json"
    fold_input = _fold_input_payload(fold)
    fold_input_sha = _canonical_sha256(fold_input)
    if fit_path.exists() != series_path.exists():
        raise RuntimeError(f"partial development artifact pair for {fold.name}")
    if fit_path.exists():
        fit_payload = _read_authenticated_development_artifact(fit_path)
        series_payload = _read_authenticated_development_artifact(series_path)
        result, _ = _authenticate_development_artifact_pair(
            fold,
            fit_payload,
            series_payload,
            expected_anchor_config,
            expected_residual_config,
        )
        return result

    training_keys = {row.ref.key for row in fold.train}
    scored_keys = {row.ref.key for row in fold.test}
    if training_keys & scored_keys:
        raise RuntimeError("outer scored series entered development fitting view")
    private_training = tuple(
        _materialize_development_series(row) for row in fold.train
    )
    fit = (
        _fit_development_training(private_training, workers, smoke=True)
        if smoke
        else _fit_development_training(private_training, workers)
    )
    result = _evaluate_development_scored(fold, fit, workers)
    expected_safe_static = _expected_safe_static_from_fit(fit, result)
    primary_seed = fit.primary_seed
    serialized_fit = _fit_development_payload(fit)
    authenticated_fit = _validate_development_fit_payload(
        serialized_fit,
        primary_seed,
        expected_safe_static,
        expected_anchor_config,
        expected_residual_config,
        tuple(row.ref.key for row in fold.train),
        tuple(
            sorted(
                {"/".join(row.ref.key.split("/")[:2]) for row in fold.train}
            )
        ),
        allow_lightweight=allow_lightweight,
    )
    if authenticated_fit is None:
        raise RuntimeError("development writer requires a complete fit")
    result = _validate_development_fold_result(
        result, fold, expected_safe_static, authenticated_fit
    )
    fit_record = _with_payload_sha(
        {
            "schema_version": 2,
            "fold": fold.name,
            "kind": fold.kind,
            "source_manifest_sha256": fold.source_manifest_sha256,
            "provenance": _ordinary_payload(fold.provenance),
            "fold_input": fold_input,
            "fold_input_sha256": fold_input_sha,
            "primary_seed": primary_seed,
            "expected_safe_static": expected_safe_static,
            "fit": serialized_fit,
        }
    )
    series_record = _with_payload_sha(
        {
            "schema_version": 2,
            "fold": fold.name,
            "kind": fold.kind,
            "source_manifest_sha256": fold.source_manifest_sha256,
            "provenance": _ordinary_payload(fold.provenance),
            "fold_input": fold_input,
            "fold_input_sha256": fold_input_sha,
            "primary_seed": primary_seed,
            "expected_safe_static": expected_safe_static,
            "fit_payload_sha256": fit_record["payload_sha256"],
            "result": result,
        }
    )
    _immutable_canonical_json(fit_path, fit_record, approved_root=root)
    _immutable_canonical_json(series_path, series_record, approved_root=root)
    return result


def fit_and_evaluate_development_fold(
    fold: DevelopmentFold, workers: int
) -> dict[str, object]:
    """Guard, authenticate, and execute one prepared outer fold."""

    workers = _require_production_workers(workers)
    if not isinstance(fold, DevelopmentFold):
        raise TypeError("fold must be a DevelopmentFold")
    profile = _validate_production_runtime(_measure_production_runtime(workers), workers)
    config, run_log, prepared = _authenticate_preparation(
        profile, historical=False
    )
    prepared_folds = _folds_for_execution(prepared)
    matching = [
        (index, item)
        for index, item in enumerate(prepared_folds)
        if item.name == fold.name
    ]
    if (
        len(matching) != 1
        or _prepared_fold_binding(matching[0][1]) != _prepared_fold_binding(fold)
        or _fold_input_sha256(matching[0][1]) != _fold_input_sha256(fold)
        or _prepared_fold_binding(fold) not in config["fold_bindings"]
    ):
        raise RuntimeError("supplied fold differs from its prepared identity")
    orphan = _preflight_execution_tree(prepared_folds, run_log)
    completed = run_log.get("completed_folds")
    if (
        run_log.get("state") != "running"
        or orphan is not None
        or type(completed) is not list
        or matching[0][0] != len(completed)
    ):
        raise RuntimeError(
            "public development fold is not the immediate running prefix"
        )
    return _fit_and_evaluate_development_fold(
        fold,
        workers,
        root=DEVELOPMENT_ROOT,
        smoke=False,
        expected_anchor_config=anchor_search_config(),
        expected_residual_config=residual_search_config(),
    )


def _load_development_gate_records(
    folds: Sequence[DevelopmentFold],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    fit_records = []
    series_records = []
    for fold in folds:
        root = DEVELOPMENT_ROOT / "folds" / fold.name
        fit_records.append(
            _read_authenticated_development_artifact(root / "fit.json")
        )
        series_records.append(
            _read_authenticated_development_artifact(root / "per_series.json")
        )
    return tuple(fit_records), tuple(series_records)


_APPROVED_WORKERS = 8
_APPROVED_OLD_PABULIB_COMMIT = "2f4321fec84069f50abf35fb5e90852013d17070"
_PRODUCTION_CITY_SERIES = 75
_PRODUCTION_DISTRICT_SERIES = 19
_PEAK_MEMORY_INTERPRETATION = (
    "maximum observed per-process RSS, not concurrent aggregate memory"
)
_LATER_RUNTIME_ADDITIONS = {
    "src/iclr_residual_protocol.py",
    "tests/test_iclr_residual_protocol.py",
    "src/iclr_residual_evaluate.py",
    "tests/test_iclr_residual_evaluate.py",
}
_CORE_RUNTIME_SOURCES = (
    "src/iclr_residual_policy.py",
    "src/iclr_residual_train.py",
    "src/iclr_residual_actuation.py",
    "src/iclr_residual_protocol.py",
    "src/iclr_residual_evaluate.py",
)
_PER_SERIES_HEADER = (
    "fold", "kind", "seed", "series", "city", "cluster",
    "residual_csd", "residual_welfare", "residual_exclusion",
    "anchor_csd", "anchor_welfare", "anchor_exclusion",
    "mes_csd", "mes_welfare", "mes_exclusion",
    "delta_csd_vs_anchor", "delta_welfare_vs_anchor",
    "delta_exclusion_vs_anchor", "residual_signature", "anchor_signature",
    "actuated",
)
_ACTUATION_HEADER = (
    "fold", "kind", "seed", "series", "instance", "is_primary_seed",
    "actuated", "first_grid_t", "refined_low", "refined_high",
    "csd_direction", "anchor_csd", "changed_csd",
)


def _require_production_workers(workers: object) -> int:
    if type(workers) is not int or workers != _APPROVED_WORKERS:
        raise ValueError("production workers must be the exact integer 8")
    return workers


def _runtime_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _runtime_monotonic() -> float:
    return time.monotonic()


def _trusted_apple_tool(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError("trusted Apple hardware probe is unavailable")
    return path


def _parse_system_profiler_payload(payload: object) -> dict[str, str]:
    try:
        rows = payload["SPHardwareDataType"]  # type: ignore[index]
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise TypeError
        row = rows[0]
        model = row["machine_name"]
        chip = row["chip_type"]
        memory = row["physical_memory"]
        if any(not isinstance(value, str) or not value.strip() for value in (model, chip, memory)):
            raise TypeError
    except (KeyError, TypeError, IndexError) as exc:
        raise RuntimeError("system profiler hardware payload is incomplete") from exc
    return {
        "model_name": model.strip(),
        "chip_name": chip.strip(),
        "physical_memory_display": memory.strip(),
    }


def _run_apple_probe(command: Sequence[str]) -> str:
    try:
        executable = _trusted_apple_tool(command[0])
        completed = subprocess.run(
            [str(executable), *command[1:]],
            shell=False,
            env={"LC_ALL": "C", "LANG": "C"},
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError("Apple hardware probe failed") from exc
    if completed.returncode != 0:
        raise RuntimeError("Apple hardware probe failed")
    return completed.stdout


def _measure_production_runtime(workers: int) -> dict[str, object]:
    _require_production_workers(workers)
    try:
        profiler = json.loads(
            _run_apple_probe(("/usr/sbin/system_profiler", "SPHardwareDataType", "-json"))
        )
        hardware = _parse_system_profiler_payload(profiler)
        os_version = _run_apple_probe(("/usr/bin/sw_vers", "-productVersion")).strip()
        os_build = _run_apple_probe(("/usr/bin/sw_vers", "-buildVersion")).strip()
        memory_text = _run_apple_probe(("/usr/sbin/sysctl", "-n", "hw.memsize")).strip()
        memory = int(memory_text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("hardware measurement failed") from exc
    if not os_version or not os_build or memory < 0:
        raise RuntimeError("hardware measurement failed")
    return {
        "os_name": "macOS",
        "os_version": os_version,
        "os_build": os_build,
        "architecture": platform.machine(),
        "model_name": hardware["model_name"],
        "chip_name": hardware["chip_name"],
        "physical_memory_bytes": memory,
        "physical_memory_display": "28 GiB" if memory == 28 * 1024**3 else hardware["physical_memory_display"],
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "execution_device": "cpu",
        "workers": workers,
        "peak_memory_method": "resource.getrusage(RUSAGE_SELF,RUSAGE_CHILDREN)",
        "peak_memory_units": "bytes",
        "peak_memory_interpretation": _PEAK_MEMORY_INTERPRETATION,
    }


def _validate_production_runtime(
    profile: object, workers: object
) -> dict[str, object]:
    _require_production_workers(workers)
    if not isinstance(profile, Mapping):
        raise RuntimeError("measured production runtime is invalid")
    required = {
        "os_name", "os_version", "os_build", "architecture", "model_name",
        "chip_name", "physical_memory_bytes", "physical_memory_display",
        "python_implementation", "python_version", "numpy_version",
        "execution_device", "workers", "peak_memory_method", "peak_memory_units",
        "peak_memory_interpretation",
    }
    if set(profile) != required:
        raise RuntimeError("measured production runtime identity is incomplete")
    expected = {
        "os_name": "macOS",
        "architecture": "arm64",
        "model_name": "Mac mini",
        "chip_name": "Apple M5",
        "physical_memory_bytes": 30064771072,
        "execution_device": "cpu",
        "workers": 8,
    }
    if any(profile.get(name) != value for name, value in expected.items()):
        raise RuntimeError("hardware/runtime does not match the approved Mac mini")
    if any(
        not isinstance(profile[name], str) or not profile[name]
        for name in required - {"physical_memory_bytes", "workers"}
    ):
        raise RuntimeError("measured production runtime identity is invalid")
    return _ordinary_payload(profile)


def _runtime_peak_memory_sample(
    previous: Mapping[str, object] | None = None,
) -> dict[str, int]:
    values = {
        "parent_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "completed_children_max_rss_bytes": resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss,
    }
    for name, value in values.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} memory RSS must be a nonnegative integer")
        if previous is not None:
            old = previous.get(name)
            if type(old) is not int or old < 0:
                raise ValueError(f"previous {name} memory RSS is invalid")
            values[name] = max(value, old)
    values["reported_peak_max_rss_bytes"] = max(values.values())
    if previous is not None:
        old_peak = previous.get("reported_peak_max_rss_bytes")
        if type(old_peak) is not int or old_peak < 0:
            raise ValueError("previous reported peak memory RSS is invalid")
        values["reported_peak_max_rss_bytes"] = max(
            values["reported_peak_max_rss_bytes"], old_peak
        )
    return values


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_security_anchor(
    path: Path, *, approved_root: Path | None = None
) -> tuple[Path, tuple[str, ...]]:
    """Bind a path lexically to the repository or one approved output root."""

    candidate = _lexical_absolute(path)
    if approved_root is not None:
        approved = _lexical_absolute(approved_root)
        try:
            candidate.relative_to(approved)
        except ValueError as exc:
            raise RuntimeError("path is outside its explicitly approved root") from exc
        anchor = approved
        while True:
            try:
                anchor.lstat()
                break
            except FileNotFoundError:
                if anchor == anchor.parent:
                    raise RuntimeError("approved output root has no existing ancestor")
                anchor = anchor.parent
        return anchor, candidate.relative_to(anchor).parts
    repository = _lexical_absolute(ROOT)
    try:
        relative = candidate.relative_to(repository)
        return repository, relative.parts
    except ValueError:
        pass
    for read_root in (
        MULTICITY_RESULT_ROOT,
        MULTICITY_DATA_DIR,
        DISTRICT_SPLIT_ROOT,
        DISTRICT_DATA_DIR,
    ):
        approved = _lexical_absolute(read_root)
        try:
            relative = candidate.relative_to(approved)
        except ValueError:
            continue
        return approved, relative.parts
    for output_root in (DEVELOPMENT_ROOT, DEVELOPMENT_ANALYSIS_ROOT):
        approved = _lexical_absolute(output_root)
        try:
            candidate.relative_to(approved)
        except ValueError:
            continue
        anchor = approved.parent
        while True:
            try:
                anchor.lstat()
                break
            except FileNotFoundError:
                if anchor == anchor.parent:
                    raise RuntimeError("approved output root has no existing ancestor")
                anchor = anchor.parent
            except OSError as exc:
                raise RuntimeError("approved output root cannot be authenticated") from exc
        return anchor, candidate.relative_to(anchor).parts
    raise RuntimeError("path is outside an approved sealed root")


def _secure_anchor(
    path: Path, *, approved_root: Path | None = None
) -> tuple[Path, tuple[str, ...]]:
    anchor, parts = _path_security_anchor(path, approved_root=approved_root)
    if any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError("path is not lexically contained in its approved root")
    try:
        anchor_stat = anchor.lstat()
    except OSError as exc:
        raise RuntimeError("approved path root is missing") from exc
    if stat.S_ISLNK(anchor_stat.st_mode) or not stat.S_ISDIR(anchor_stat.st_mode):
        raise RuntimeError("approved path root or ancestor is symlinked")
    try:
        resolved_anchor = anchor.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("approved path root cannot be authenticated") from exc
    return resolved_anchor, parts


@contextmanager
def _secure_parent_directory(
    path: Path, *, create: bool, approved_root: Path | None = None
):
    """Open a validated parent by descriptors without following links."""

    anchor, parts = _secure_anchor(path, approved_root=approved_root)
    if not parts:
        raise RuntimeError("artifact path cannot name its approved root")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = os.open(anchor, directory_flags)
        descriptors.append(descriptor)
        for component in parts[:-1]:
            try:
                item = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise RuntimeError("artifact ancestor is missing") from None
                try:
                    os.mkdir(component, mode=0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                item = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError("artifact ancestor cannot be authenticated") from exc
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                raise RuntimeError("artifact ancestor is symlinked or not a directory")
            try:
                child = os.open(component, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise RuntimeError("artifact ancestor is symlinked or unstable") from exc
            descriptors.append(child)
            descriptor = child
        yield descriptor, parts[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _descriptor_leaf_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError("artifact leaf cannot be authenticated") from exc


def _secure_path_mode(path: Path) -> int | None:
    """Return an lstat mode, None for a missing path, and reject link ancestors."""

    anchor, parts = _secure_anchor(path)
    if not parts:
        return anchor.lstat().st_mode
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        descriptor = os.open(anchor, directory_flags)
        descriptors.append(descriptor)
        for index, component in enumerate(parts):
            try:
                item = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise RuntimeError("path component cannot be authenticated") from exc
            final = index == len(parts) - 1
            if final:
                return item.st_mode
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                raise RuntimeError("path ancestor is symlinked or not a directory")
            child = os.open(component, directory_flags, dir_fd=descriptor)
            descriptors.append(child)
            descriptor = child
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    raise AssertionError("unreachable path-mode state")


def _read_regular_bytes(
    path: Path, label: str, *, approved_root: Path | None = None
) -> bytes:
    with _secure_parent_directory(
        path, create=False, approved_root=approved_root
    ) as (parent_fd, name):
        item = _descriptor_leaf_stat(parent_fd, name)
        if item is None:
            raise RuntimeError(f"{label} is missing")
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
            raise RuntimeError(f"{label} must be a regular nonsymlink file")
        try:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
        except OSError as exc:
            raise RuntimeError(f"{label} cannot be opened without following links") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
            ) != (item.st_dev, item.st_ino):
                raise RuntimeError(f"{label} changed during authentication")
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)


def _write_all(descriptor: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short artifact write")
        view = view[written:]


def _atomic_write(path: Path, encoded: bytes) -> None:
    with _secure_parent_directory(path, create=True) as (parent_fd, name):
        current = _descriptor_leaf_stat(parent_fd, name)
        if current is not None and (
            stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
        ):
            raise RuntimeError("output artifact is symlinked or not regular")
        temporary = f".{name}.{secrets.token_hex(16)}.tmp"
        descriptor = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=parent_fd,
            )
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            current = _descriptor_leaf_stat(parent_fd, name)
            if current is not None and (
                stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
            ):
                raise RuntimeError("output artifact is symlinked or not regular")
            os.replace(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise


def _atomic_canonical_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(path, _canonical_json_bytes(payload))


def _require_regular_nonsymlink(path: Path, label: str) -> Path:
    with _secure_parent_directory(path, create=False) as (parent_fd, name):
        item = _descriptor_leaf_stat(parent_fd, name)
    if item is None:
        raise RuntimeError(f"{label} is missing")
    if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
        raise RuntimeError(f"{label} must be a regular nonsymlink file")
    return path


def _secure_walk(root: Path, label: str) -> tuple[tuple[Path, int], ...]:
    """Enumerate a closed tree without following links or hiding link dirs."""

    anchor, parts = _secure_anchor(root)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    entries: list[tuple[Path, int]] = []

    def visit(descriptor: int, current: Path) -> None:
        try:
            children = sorted(os.scandir(descriptor), key=lambda row: row.name)
        except OSError as exc:
            raise RuntimeError(f"{label} inventory cannot be enumerated") from exc
        for child in children:
            try:
                item = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(f"{label} inventory is unstable") from exc
            path = current / child.name
            if stat.S_ISLNK(item.st_mode):
                raise RuntimeError(f"{label} inventory contains a symlink")
            entries.append((path, item.st_mode))
            if stat.S_ISDIR(item.st_mode):
                try:
                    nested = os.open(
                        child.name, directory_flags, dir_fd=descriptor
                    )
                except OSError as exc:
                    raise RuntimeError(
                        f"{label} inventory directory is symlinked or unstable"
                    ) from exc
                try:
                    visit(nested, path)
                finally:
                    os.close(nested)

    try:
        descriptor = os.open(anchor, directory_flags)
        descriptors.append(descriptor)
        current = anchor
        for component in parts:
            try:
                item = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(f"{label} root is missing") from exc
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                raise RuntimeError(f"{label} root or ancestor is symlinked")
            child_fd = os.open(component, directory_flags, dir_fd=descriptor)
            descriptors.append(child_fd)
            descriptor = child_fd
            current = current / component
        visit(descriptor, _lexical_absolute(root))
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return tuple(entries)


def _matching_regular_files(
    root: Path, pattern: str, *, recursive: bool, label: str
) -> tuple[Path, ...]:
    entries = _secure_walk(root, label)
    matches = []
    for path, mode in entries:
        relative = path.relative_to(root)
        if not recursive and len(relative.parts) != 1:
            continue
        if fnmatchcase(path.name, pattern):
            if not stat.S_ISREG(mode):
                raise RuntimeError(f"{label} inventory entry is not regular")
            matches.append(path)
    repository = _lexical_absolute(ROOT)
    return tuple(
        sorted(
            matches,
            key=lambda path: _lexical_absolute(path).relative_to(repository).as_posix(),
        )
    )


def _collect_runtime_source_inventory() -> dict[str, dict[str, object]]:
    paths = (
        *_matching_regular_files(
            ROOT / "src", "*.py", recursive=True, label="runtime source"
        ),
        _require_regular_nonsymlink(
            ROOT / "pyproject.toml", "runtime pyproject source"
        ),
        _require_regular_nonsymlink(ROOT / "uv.lock", "runtime lock source"),
    )
    result = {}
    for path in sorted(paths, key=lambda value: value.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        result[relative] = {"state": "present", "sha256": _file_sha256(path)}
    return result


def _collect_core_source_slots() -> dict[str, dict[str, object]]:
    result = {}
    for relative in _CORE_RUNTIME_SOURCES:
        path = ROOT / relative
        with _secure_parent_directory(path, create=False) as (parent_fd, name):
            item = _descriptor_leaf_stat(parent_fd, name)
        if item is None:
            result[relative] = {"state": "not_created", "sha256": None}
        elif stat.S_ISREG(item.st_mode):
            result[relative] = {"state": "present", "sha256": _file_sha256(path)}
        else:
            raise RuntimeError(f"core source is a symlink or not regular: {relative}")
    return result


def _collect_residual_test_inventory() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): _file_sha256(path)
        for path in _matching_regular_files(
            ROOT / "tests",
            "test_iclr_residual_*.py",
            recursive=False,
            label="residual test",
        )
    }


def _approved_old_pabulib_commit() -> str:
    from iclr_multicity_protocol import PABULIB_COMMIT

    if PABULIB_COMMIT != _APPROVED_OLD_PABULIB_COMMIT:
        raise RuntimeError("bound Pabulib commit differs from the approved commit")
    return PABULIB_COMMIT


def _read_preparation_metadata(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(_read_regular_bytes(path, label).decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is invalid") from exc
    if type(payload) is not dict:
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def _sha256_text(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} SHA-256 identity is invalid")
    return value


def _canonical_series_identifier(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise RuntimeError(f"{label} series identifier is not canonical")
    return value


def _canonical_relative_authority_path(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise RuntimeError(f"{label} path is not canonical")
    return value


def _manifest_old_input_rows(
    manifest: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    required = {
        "schema_version", "source_commit", "source_dir", "destination",
        "n_series", "n_elections", "n_files", "files",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != required
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
    ):
        raise RuntimeError("approved city corpus manifest schema differs")
    rows = manifest.get("files")
    if (
        type(rows) is not list
        or not rows
        or manifest.get("source_commit") != _approved_old_pabulib_commit()
        or type(manifest.get("source_dir")) is not str
        or not manifest.get("source_dir")
        or type(manifest.get("destination")) is not str
        or not manifest.get("destination")
        or any(
            type(manifest.get(name)) is not int or manifest[name] < 0
            for name in ("n_series", "n_elections", "n_files")
        )
        or manifest.get("n_files") != len(rows)
        or manifest.get("n_elections") != len(rows)
    ):
        raise RuntimeError("approved city corpus manifest schema differs")
    normalized = []
    seen_names = set()
    seen_sources = set()
    seen_pairs = set()
    for raw in rows:
        if type(raw) is not dict or set(raw) != {
            "name", "source_relative", "series", "year", "bytes", "sha256"
        }:
            raise RuntimeError("approved city corpus manifest row differs")
        name = raw.get("name")
        source_relative = _canonical_relative_authority_path(
            raw.get("source_relative"), "approved city source"
        )
        series = _canonical_series_identifier(
            raw.get("series"), "approved city input"
        )
        year = raw.get("year")
        size = raw.get("bytes")
        digest = _sha256_text(raw.get("sha256"), "approved city input")
        if (
            type(name) is not str
            or not name
            or Path(name).name != name
            or type(year) is not int
            or type(size) is not int
            or size < 0
            or name in seen_names
            or source_relative in seen_sources
            or (series, year) in seen_pairs
        ):
            raise RuntimeError("approved city corpus manifest row differs")
        seen_names.add(name)
        seen_sources.add(source_relative)
        seen_pairs.add((series, year))
        normalized.append(
            {
                "name": name,
                "source_relative": source_relative,
                "series": series,
                "year": year,
                "bytes": size,
                "sha256": digest,
            }
        )
    if [row["name"] for row in normalized] != sorted(seen_names):
        raise RuntimeError("approved city corpus manifest row order differs")
    if manifest.get("n_series") != len({row["series"] for row in normalized}):
        raise RuntimeError("approved city corpus manifest count differs")
    return tuple(normalized)


def _split_year_mapping(
    value: object, label: str
) -> dict[str, tuple[int, ...]]:
    if type(value) is not dict or not value:
        raise RuntimeError(f"{label} split inventory differs")
    if list(value) != sorted(value):
        raise RuntimeError(f"{label} split inventory order differs")
    result = {}
    for raw_series, raw_years in value.items():
        series = _canonical_series_identifier(raw_series, label)
        if (
            type(raw_years) is not list
            or not raw_years
            or any(type(year) is not int for year in raw_years)
            or raw_years != sorted(set(raw_years))
        ):
            raise RuntimeError(f"{label} split inventory differs")
        result[series] = tuple(raw_years)
    return result


def _split_series_years(
    payload: Mapping[str, object], *, city: bool, expected_name: str
) -> tuple[
    set[tuple[str, int]], set[str]
]:
    pairs: set[tuple[str, int]] = set()
    scored_series: set[str] = set()
    if city:
        required = {
            "schema_version", "name", "note", "train_through",
            "first_test_year", "fit", "warmup", "score", "counts",
        }
        if (
            type(payload) is not dict
            or set(payload) != required
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
            or payload.get("name") != expected_name
            or type(payload.get("name")) is not str
            or type(payload.get("note")) is not str
            or type(payload.get("train_through")) is not int
            or payload.get("train_through") != 2022
            or type(payload.get("first_test_year")) is not int
            or payload.get("first_test_year") != 2023
        ):
            raise RuntimeError("approved city split schema differs")
        fit = _split_year_mapping(payload.get("fit"), "approved city fit")
        warmup = _split_year_mapping(payload.get("warmup"), "approved city warmup")
        score = _split_year_mapping(payload.get("score"), "approved city score")
        if (
            set(fit) & set(score)
            or set(fit) & set(warmup)
            or set(warmup) != set(score)
            or any(year > 2022 for years in fit.values() for year in years)
            or any(year > 2022 for years in warmup.values() for year in years)
            or any(year < 2023 for years in score.values() for year in years)
        ):
            raise RuntimeError("approved city split coverage differs")
        counts = payload.get("counts")
        expected_counts = {
            "fit_series": len(fit),
            "fit_elections": sum(map(len, fit.values())),
            "warmup_series": len(warmup),
            "warmup_elections": sum(map(len, warmup.values())),
            "score_series": len(score),
            "score_elections": sum(map(len, score.values())),
        }
        if (
            type(counts) is not dict
            or set(counts) != set(expected_counts)
            or any(type(value) is not int for value in counts.values())
            or counts != expected_counts
        ):
            raise RuntimeError("approved city split counts differ")
        for field, rows in (("fit", fit), ("warmup", warmup), ("score", score)):
            for series, years in rows.items():
                if field == "score":
                    scored_series.add(series)
                for year in years:
                    if (series, year) in pairs:
                        raise RuntimeError("approved city split inventory differs")
                    pairs.add((series, year))
        return pairs, scored_series
    inventories = _district_split_inventory(payload, expected_name)
    for retained in inventories.values():
        for series, years in retained.items():
            pairs.update((series, year) for year in years)
    return pairs, scored_series


def _district_split_inventory(
    payload: Mapping[str, object], expected_name: str
) -> dict[str, dict[str, tuple[int, ...]]]:
    if (
        type(payload) is not dict
        or set(payload) != {"name", "note", "train", "test"}
        or type(payload.get("name")) is not str
        or payload.get("name") != expected_name
        or type(payload.get("note")) is not str
    ):
        raise RuntimeError("approved district split schema differs")
    inventories: dict[str, dict[str, tuple[int, ...]]] = {}
    all_series: set[str] = set()
    for field in ("train", "test"):
        rows = payload.get(field)
        if type(rows) is not list:
            raise RuntimeError("approved district split schema differs")
        retained: dict[str, tuple[int, ...]] = {}
        for raw in rows:
            if type(raw) is not list or len(raw) != 2:
                raise RuntimeError("approved district split inventory differs")
            series = _canonical_series_identifier(
                raw[0], "approved district split"
            )
            years = raw[1]
            if (
                type(years) is not list
                or not years
                or any(type(year) is not int for year in years)
                or years != sorted(set(years))
                or series in all_series
            ):
                raise RuntimeError("approved district split inventory differs")
            all_series.add(series)
            retained[series] = tuple(years)
        if list(retained) != sorted(retained):
            raise RuntimeError("approved district split inventory order differs")
        inventories[field] = retained
    if not all_series:
        raise RuntimeError("approved district split coverage differs")
    return inventories


def _validate_district_partition(
    payloads: Mapping[str, Mapping[str, object]],
    city_rows: Sequence[Mapping[str, object]],
    *,
    expected_city_series: int,
    expected_district_series: int,
) -> tuple[
    dict[str, dict[str, dict[str, tuple[int, ...]]]],
    set[tuple[str, int]],
]:
    if (
        type(expected_city_series) is not int
        or type(expected_district_series) is not int
        or expected_city_series <= 0
        or expected_district_series <= 0
        or expected_district_series >= expected_city_series
    ):
        raise ValueError("development series cardinalities are invalid")
    if type(payloads) is not dict or tuple(payloads) != _DISTRICT_DEVELOPMENT_FOLDS:
        raise RuntimeError("approved district fold name or order differs")
    parsed = {
        name: _district_split_inventory(payloads[name], name)
        for name in _DISTRICT_DEVELOPMENT_FOLDS
    }
    combined = []
    for name in _DISTRICT_DEVELOPMENT_FOLDS:
        inventories = parsed[name]
        combined.append({**inventories["train"], **inventories["test"]})
    universe = combined[0]
    if any(rows != universe for rows in combined[1:]):
        raise RuntimeError("approved district split universe differs across folds")
    district_pairs = {
        (series, year)
        for series, years in universe.items()
        for year in years
    }
    city_by_pair = {(row["series"], row["year"]): row for row in city_rows}
    if not district_pairs <= set(city_by_pair):
        raise RuntimeError(
            "district split references inputs absent from corpus authority"
        )
    city_series = {series for series, _ in city_by_pair}
    district_series = set(universe)
    if (
        len(city_series) != expected_city_series
        or len(district_series) != expected_district_series
        or not district_series < city_series
    ):
        raise RuntimeError("approved development series cardinality or subset differs")

    test_counts = {series: 0 for series in universe}
    train_counts = {series: 0 for series in universe}
    tested_pairs: set[tuple[str, int]] = set()
    for fold_index, name in enumerate(_DISTRICT_DEVELOPMENT_FOLDS):
        train = parsed[name]["train"]
        test = parsed[name]["test"]
        expected_test = {
            series
            for index, series in enumerate(sorted(universe))
            if index % 5 == fold_index
        }
        if set(test) != expected_test or set(train) != district_series - expected_test:
            raise RuntimeError("approved district canonical partition differs")
        for series in train:
            train_counts[series] += 1
        for series, years in test.items():
            test_counts[series] += 1
            pairs = {(series, year) for year in years}
            if tested_pairs & pairs:
                raise RuntimeError("approved district test membership is duplicated")
            tested_pairs.update(pairs)
    if (
        tested_pairs != district_pairs
        or set(test_counts.values()) != {1}
        or set(train_counts.values()) != {4}
    ):
        raise RuntimeError("approved district membership counts differ")
    return parsed, district_pairs


def _closed_old_input_inventory(
    root: Path, expected: Mapping[str, str], *, label: str
) -> dict[str, str]:
    expected_files = set(expected)
    expected_directories = {
        parent.as_posix()
        for relative in expected_files
        for parent in Path(relative).parents
        if parent.as_posix() != "."
    }
    observed_files = set()
    observed_directories = set()
    for path, mode in _secure_walk(root, f"{label} input"):
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(mode):
            observed_directories.add(relative)
        elif stat.S_ISREG(mode):
            observed_files.add(relative)
        else:
            raise RuntimeError(f"{label} input inventory contains a special entry")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise RuntimeError(f"{label} input inventory differs from its closed allowlist")
    result = {}
    for relative, expected_sha in sorted(expected.items()):
        path = _require_regular_nonsymlink(
            root / relative, f"approved {label} input"
        )
        observed_sha = _file_sha256(path)
        if observed_sha != expected_sha:
            raise RuntimeError(f"approved {label} input hash differs")
        result[_canonical_old_input_path(path)] = observed_sha
    return result


def _collect_development_preparation_inputs_for_counts(
    *, expected_city_series: int, expected_district_series: int
):
    """Validate synthetic metadata with explicit fixture cardinalities."""

    city_manifest = MULTICITY_RESULT_ROOT / "corpus_manifest.json"
    city_splits = {
        name: MULTICITY_RESULT_ROOT / "splits" / f"{name}.json"
        for name in _CITY_DEVELOPMENT_FOLDS
    }
    district_splits = {
        name: DISTRICT_SPLIT_ROOT / f"{name}.json"
        for name in _DISTRICT_DEVELOPMENT_FOLDS
    }
    manifest_payload = _read_preparation_metadata(
        city_manifest, "approved city corpus manifest"
    )
    manifest_rows = _manifest_old_input_rows(manifest_payload)
    city_payloads = {
        name: _read_preparation_metadata(path, f"approved city split {name}")
        for name, path in city_splits.items()
    }
    district_payloads = {
        name: _read_preparation_metadata(path, f"approved district split {name}")
        for name, path in district_splits.items()
    }
    by_pair = {(row["series"], row["year"]): row for row in manifest_rows}
    city_expected = {row["name"]: row["sha256"] for row in manifest_rows}
    city_split_rows = {
        name: _split_series_years(
            city_payloads[name], city=True, expected_name=name
        )
        for name in _CITY_DEVELOPMENT_FOLDS
    }
    district_inventories, district_pairs = _validate_district_partition(
        district_payloads,
        manifest_rows,
        expected_city_series=expected_city_series,
        expected_district_series=expected_district_series,
    )
    district_expected = {
        by_pair[pair]["source_relative"]: by_pair[pair]["sha256"]
        for pair in sorted(district_pairs)
    }
    if len(district_expected) != len(district_pairs):
        raise RuntimeError("district input authority contains filename collisions")
    city_hashes = _closed_old_input_inventory(
        MULTICITY_DATA_DIR, city_expected, label="city"
    )
    district_hashes = _closed_old_input_inventory(
        DISTRICT_DATA_DIR, district_expected, label="district"
    )
    city_manifest_sha = _file_sha256(city_manifest)
    district_source_rows: dict[str, dict[str, object]] = {}
    for series, year in sorted(district_pairs):
        row = by_pair[(series, year)]
        path = DISTRICT_DATA_DIR / row["source_relative"]
        retained = district_source_rows.setdefault(series, {"years": [], "files": []})
        retained["years"].append(year)  # type: ignore[union-attr]
        retained["files"].append(  # type: ignore[union-attr]
            {"path": _canonical_old_input_path(path), "sha256": row["sha256"]}
        )
    district_source_sha = _canonical_sha256(district_source_rows)
    descriptors = []
    for name in _CITY_DEVELOPMENT_FOLDS:
        pairs, scored_series = city_split_rows[name]
        required_pairs = pairs | {
            pair for pair in by_pair if pair[0] in scored_series
        }
        if not required_pairs <= set(by_pair):
            raise RuntimeError("city split references inputs absent from corpus authority")
        descriptors.append(
            {
                "name": name,
                "kind": "city",
                "source_manifest_sha256": city_manifest_sha,
                "split_sha256": _file_sha256(city_splits[name]),
                "input_blob_sha256": {
                    _canonical_old_input_path(MULTICITY_DATA_DIR / by_pair[pair]["name"]):
                    by_pair[pair]["sha256"]
                    for pair in sorted(required_pairs)
                },
            }
        )
    for name in _DISTRICT_DEVELOPMENT_FOLDS:
        inventories = district_inventories[name]
        pairs = {
            (series, year)
            for rows in inventories.values()
            for series, years in rows.items()
            for year in years
        }
        descriptors.append(
            {
                "name": name,
                "kind": "district",
                "source_manifest_sha256": district_source_sha,
                "split_sha256": _file_sha256(district_splits[name]),
                "input_blob_sha256": {
                    _canonical_old_input_path(
                        DISTRICT_DATA_DIR / by_pair[pair]["source_relative"]
                    ): by_pair[pair]["sha256"]
                    for pair in sorted(pairs)
                },
            }
        )
    descriptors = tuple(descriptors)
    provenance = {
        "old_pabulib_commit": _approved_old_pabulib_commit(),
        "city_corpus_manifest_sha256": city_manifest_sha,
        "city_split_sha256": {
            name: _file_sha256(path) for name, path in city_splits.items()
        },
        "district_split_sha256": {
            name: _file_sha256(path) for name, path in district_splits.items()
        },
        "old_input_sha256": dict(sorted({**city_hashes, **district_hashes}.items())),
        "fold_binding_sha256": {
            row["name"]: _canonical_sha256(row) for row in descriptors
        },
    }
    return descriptors, provenance


def _collect_development_preparation_inputs():
    """Hash the fixed 75-city/19-district production metadata authority."""

    return _collect_development_preparation_inputs_for_counts(
        expected_city_series=_PRODUCTION_CITY_SERIES,
        expected_district_series=_PRODUCTION_DISTRICT_SERIES,
    )


def _optimizer_runtime_payload() -> dict[str, object]:
    return {
        "anchor": _anchor_search_config_payload(anchor_search_config()),
        "residual": _residual_search_config_payload(residual_search_config()),
    }


def _inference_runtime_payload() -> dict[str, object]:
    return {
        "bootstrap_draws": 20000,
        "bootstrap_seed": 20260818,
        "bootstrap_upper_max": 0.0,
        "district_negative_min": 4,
        "welfare_floor": 0.99,
        "actuation_rate_min": 0.10,
        "actuated_city_min": 2,
    }


def _prepared_fold_binding(value: object) -> dict[str, object]:
    if isinstance(value, DevelopmentFold):
        provenance = value.provenance
        split_sha = provenance.get("split_sha256")
        input_hashes = provenance.get("input_blob_sha256", {})
        if not isinstance(input_hashes, Mapping):
            raise RuntimeError("loaded fold input binding is invalid")
        return {
            "name": value.name,
            "kind": value.kind,
            "source_manifest_sha256": value.source_manifest_sha256,
            "split_sha256": _sha256_text(split_sha, "loaded fold split"),
            "input_blob_sha256": dict(sorted(input_hashes.items())),
        }
    if (
        isinstance(value, Mapping)
        and set(value)
        == {
            "name",
            "kind",
            "source_manifest_sha256",
            "split_sha256",
            "input_blob_sha256",
        }
        and isinstance(value["name"], str)
        and value["kind"] in {"city", "district"}
        and isinstance(value["input_blob_sha256"], Mapping)
    ):
        source = _sha256_text(value["source_manifest_sha256"], "prepared source")
        split = _sha256_text(value["split_sha256"], "prepared split")
        inputs = {}
        for path, digest in value["input_blob_sha256"].items():
            if not isinstance(path, str):
                raise RuntimeError("prepared fold input path is invalid")
            inputs[_canonical_persisted_path(
                path, approved_prefixes=("data/pb/", "data/pb_multicity/", "synthetic-input/")
            )] = _sha256_text(digest, "prepared fold input")
        return {
            "name": value["name"],
            "kind": value["kind"],
            "source_manifest_sha256": source,
            "split_sha256": split,
            "input_blob_sha256": dict(sorted(inputs.items())),
        }
    raise RuntimeError("prepared fold binding is invalid")


def _canonical_instance_identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise RuntimeError(f"{label} instance path is not canonical")
    for scheme in ("synthetic://", "smoke-training://", "smoke-development://"):
        if value.startswith(scheme):
            tail = value[len(scheme):]
            if (
                not tail
                or tail.startswith("/")
                or any(part in {"", ".", ".."} for part in tail.split("/"))
            ):
                raise RuntimeError(f"{label} instance path is not canonical")
            return value
    if Path(value).is_absolute():
        raise RuntimeError(f"{label} instance path must be repository-relative")
    try:
        canonical = _canonical_persisted_path(
            value,
            approved_prefixes=("data/pb/", "data/pb_multicity/"),
        )
    except ValueError as exc:
        raise RuntimeError(f"{label} instance path is not canonical") from exc
    if canonical != value:
        raise RuntimeError(f"{label} instance path is not canonical")
    return canonical


def _validate_fold_view(
    value: object,
    *,
    expected_name: str | None = None,
    expected_kind: str | None = None,
) -> dict[str, object]:
    required = {
        "name", "kind", "fold_input_sha256", "training_series",
        "scored_series", "scored_instances",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RuntimeError("loaded fold view schema differs")
    name = value.get("name")
    kind = value.get("kind")
    if (
        not isinstance(name, str)
        or not name
        or kind not in {"city", "district"}
        or (expected_name is not None and name != expected_name)
        or (expected_kind is not None and kind != expected_kind)
    ):
        raise RuntimeError("loaded fold view identity differs")
    digest = _sha256_text(value.get("fold_input_sha256"), "loaded fold input")
    retained_series: dict[str, list[str]] = {}
    for field in ("training_series", "scored_series"):
        raw = value.get(field)
        if not isinstance(raw, list):
            raise RuntimeError("loaded fold view series inventory differs")
        rows = [
            _canonical_series_identifier(item, "loaded fold view") for item in raw
        ]
        if len(rows) != len(set(rows)) or rows != sorted(rows):
            raise RuntimeError("loaded fold view series order or uniqueness differs")
        retained_series[field] = rows
    training = retained_series["training_series"]
    scored = retained_series["scored_series"]
    if not training or not scored or set(training) & set(scored):
        raise RuntimeError("loaded fold view training/scored inventory differs")
    raw_instances = value.get("scored_instances")
    if not isinstance(raw_instances, list) or not raw_instances:
        raise RuntimeError("loaded fold view instance inventory differs")
    scored_index = {series: index for index, series in enumerate(scored)}
    instances = []
    paths = set()
    positions = []
    for raw in raw_instances:
        if not isinstance(raw, Mapping) or set(raw) != {"series", "instance"}:
            raise RuntimeError("loaded fold view instance row schema differs")
        series = _canonical_series_identifier(
            raw.get("series"), "loaded fold view instance"
        )
        if series not in scored_index:
            raise RuntimeError("loaded fold view instance series is misbound")
        instance = _canonical_instance_identity(
            raw.get("instance"), "loaded fold view"
        )
        if instance in paths:
            raise RuntimeError("loaded fold view instance inventory is duplicated")
        paths.add(instance)
        positions.append(scored_index[series])
        instances.append({"series": series, "instance": instance})
    if positions != sorted(positions) or set(row["series"] for row in instances) != set(scored):
        raise RuntimeError("loaded fold view instance order or coverage differs")
    return {
        "name": name,
        "kind": kind,
        "fold_input_sha256": digest,
        "training_series": training,
        "scored_series": scored,
        "scored_instances": instances,
    }


def _loaded_fold_view(value: DevelopmentFold) -> dict[str, object]:
    if not isinstance(value, DevelopmentFold):
        raise RuntimeError("loaded semantic fold view requires a DevelopmentFold")
    return _validate_fold_view({
        "name": value.name,
        "kind": value.kind,
        "fold_input_sha256": _fold_input_sha256(value),
        "training_series": [row.ref.key for row in value.train],
        "scored_series": [row.ref.key for row in value.test],
        "scored_instances": [
            {"series": row.ref.key, "instance": row.all_years[year].path}
            for row in value.test
            for year in row.test_years
        ],
    })


def _build_development_config(
    runtime_profile: Mapping[str, object],
    folds: Sequence[object],
    inputs: Mapping[str, object],
) -> dict[str, object]:
    bindings = [_prepared_fold_binding(fold) for fold in folds]
    expected_names = (*_CITY_DEVELOPMENT_FOLDS, *_DISTRICT_DEVELOPMENT_FOLDS)
    expected_kinds = ("city",) * 3 + ("district",) * 5
    if (
        tuple(row["name"] for row in bindings) != expected_names
        or tuple(row["kind"] for row in bindings) != expected_kinds
    ):
        raise RuntimeError("prepared development fold inventory differs")
    normalized_inputs = _ordinary_payload(inputs)
    if normalized_inputs.get("old_pabulib_commit") != _approved_old_pabulib_commit():
        raise RuntimeError("prepared Pabulib commit differs")
    return _with_payload_sha(
        {
            "schema_version": 1,
            "runtime": _ordinary_payload(runtime_profile),
            "fold_inventory": [
                {"name": row["name"], "kind": row["kind"]} for row in bindings
            ],
            "fold_bindings": bindings,
            "inputs": normalized_inputs,
            "optimizer": _optimizer_runtime_payload(),
            "inference": _inference_runtime_payload(),
            "core_sources": _collect_core_source_slots(),
            "runtime_source_inventory": _collect_runtime_source_inventory(),
            "residual_tests": _collect_residual_test_inventory(),
        }
    )


def _new_run_log(config: Mapping[str, object], runtime_profile: Mapping[str, object]):
    memory = _runtime_peak_memory_sample()
    start_utc = _runtime_utc_now()
    event = {"event": "prepared", "state": "prepared", "elapsed_seconds": 0.0}
    return _with_payload_sha(
        {
            "schema_version": 1,
            "state": "prepared",
            "config_sha256": config["payload_sha256"],
            "workers": 8,
            "start_utc": start_utc,
            "end_utc": None,
            "monotonic_start": _runtime_monotonic(),
            "elapsed_seconds": 0.0,
            **memory,
            "peak_memory_interpretation": runtime_profile[
                "peak_memory_interpretation"
            ],
            "completed_folds": [],
            "fold_views": [],
            "events": [event],
        }
    )


def _read_task6_json(path: Path, label: str) -> dict[str, object]:
    try:
        encoded = _read_regular_bytes(path, label)
        payload = json.loads(encoded.decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError
        unsigned = dict(payload)
        embedded = unsigned.pop("payload_sha256")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"{label} is divergent") from exc
    if (
        embedded != _canonical_sha256(unsigned)
        or encoded != _canonical_json_bytes(payload)
    ):
        raise RuntimeError(f"{label} is noncanonical or divergent")
    return payload


def _runtime_inventory_matches(
    frozen: Mapping[str, object], current: Mapping[str, object], *, historical: bool
) -> bool:
    if not historical:
        return _canonical_payload_equal(dict(frozen), dict(current))
    frozen_paths = set(frozen)
    current_paths = set(current)
    additions = current_paths - frozen_paths
    if not additions <= _LATER_RUNTIME_ADDITIONS:
        return False
    return all(
        path in current
        and _canonical_payload_equal(
            {"value": frozen[path]}, {"value": current[path]}
        )
        for path in frozen_paths
    )


def _residual_test_inventory_matches(
    frozen: Mapping[str, object], current: Mapping[str, object], *, historical: bool
) -> bool:
    if not historical:
        return _canonical_payload_equal(dict(frozen), dict(current))
    additions = set(current) - set(frozen)
    if not additions <= _LATER_RUNTIME_ADDITIONS:
        return False
    return all(current.get(path) == digest for path, digest in frozen.items())


def _current_preparation_material(
    profile: Mapping[str, object], *, collect_inputs: bool
) -> tuple[dict[str, object], tuple[object, ...]]:
    if collect_inputs:
        folds, inputs = _collect_development_preparation_inputs()
        folds = tuple(folds)
        config = _build_development_config(profile, folds, inputs)
        return config, folds
    return {}, ()


def _run_log_nonnegative_float(value: object, label: str) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value < 0.0
        or (value == 0.0 and math.copysign(1.0, value) < 0.0)
    ):
        raise RuntimeError(f"run-log {label} schema differs")
    return value


def _run_log_config_runtime(config: object) -> dict[str, object]:
    if type(config) is not dict:
        raise RuntimeError("run-log configuration schema differs")
    embedded = _sha256_text(
        config.get("payload_sha256"), "run-log configuration"
    )
    unsigned = dict(config)
    unsigned.pop("payload_sha256")
    if embedded != _canonical_sha256(unsigned):
        raise RuntimeError("run-log configuration identity differs")
    profile = config.get("runtime")
    if type(profile) is not dict:
        raise RuntimeError("run-log configuration runtime schema differs")
    if type(profile.get("workers")) is not int or profile["workers"] != _APPROVED_WORKERS:
        raise RuntimeError("run-log configuration workers differ")
    if (
        type(profile.get("peak_memory_interpretation")) is not str
        or profile["peak_memory_interpretation"] != _PEAK_MEMORY_INTERPRETATION
    ):
        raise RuntimeError("run-log configuration peak-memory interpretation differs")
    return profile


def _require_exact_run_log_fold_view_types(value: object) -> None:
    required = {
        "name", "kind", "fold_input_sha256", "training_series",
        "scored_series", "scored_instances",
    }
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != required
    ):
        raise RuntimeError("run-log loaded fold view schema differs")
    if any(type(value[name]) is not str for name in ("name", "kind", "fold_input_sha256")):
        raise RuntimeError("run-log loaded fold view scalar schema differs")
    for field in ("training_series", "scored_series"):
        rows = value[field]
        if type(rows) is not list or any(type(row) is not str for row in rows):
            raise RuntimeError("run-log loaded fold view series schema differs")
    instances = value["scored_instances"]
    if type(instances) is not list:
        raise RuntimeError("run-log loaded fold view instance schema differs")
    for row in instances:
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"series", "instance"}
            or type(row["series"]) is not str
            or type(row["instance"]) is not str
        ):
            raise RuntimeError("run-log loaded fold view instance schema differs")


def _authenticate_run_log(
    payload: Mapping[str, object], config: Mapping[str, object], *, finalized: bool | None
) -> dict[str, object]:
    top_keys = {
        "schema_version", "state", "config_sha256", "workers", "start_utc",
        "end_utc", "monotonic_start", "elapsed_seconds",
        "parent_max_rss_bytes", "completed_children_max_rss_bytes",
        "reported_peak_max_rss_bytes", "peak_memory_interpretation",
        "completed_folds", "fold_views", "events", "payload_sha256",
    }
    if (
        type(payload) is not dict
        or any(type(key) is not str for key in payload)
        or set(payload) != top_keys
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise RuntimeError("run-log top-level schema differs")
    config_runtime = _run_log_config_runtime(config)
    payload_sha = _sha256_text(payload.get("payload_sha256"), "run-log payload")
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    if payload_sha != _canonical_sha256(unsigned):
        raise RuntimeError("run-log payload identity differs")
    config_sha = _sha256_text(
        payload.get("config_sha256"), "run-log configuration"
    )
    if config_sha != config.get("payload_sha256"):
        raise RuntimeError("run-log configuration identity differs")
    if (
        type(payload.get("workers")) is not int
        or payload["workers"] != _APPROVED_WORKERS
        or payload["workers"] != config_runtime["workers"]
        or type(payload.get("events")) is not list
        or type(payload.get("completed_folds")) is not list
        or type(payload.get("fold_views")) is not list
        or type(payload.get("state")) is not str
        or type(payload.get("start_utc")) is not str
        or not payload.get("start_utc")
        or type(payload.get("peak_memory_interpretation")) is not str
        or payload["peak_memory_interpretation"] != _PEAK_MEMORY_INTERPRETATION
        or payload["peak_memory_interpretation"]
        != config_runtime["peak_memory_interpretation"]
    ):
        raise RuntimeError("run-log schema differs")
    _run_log_nonnegative_float(payload.get("monotonic_start"), "monotonic clock")
    top_elapsed = _run_log_nonnegative_float(
        payload.get("elapsed_seconds"), "elapsed clock"
    )
    memory_names = (
        "parent_max_rss_bytes",
        "completed_children_max_rss_bytes",
        "reported_peak_max_rss_bytes",
    )
    if any(type(payload.get(name)) is not int or payload[name] < 0 for name in memory_names):
        raise RuntimeError("run-log memory schema differs")
    if payload["reported_peak_max_rss_bytes"] != max(
        payload["parent_max_rss_bytes"], payload["completed_children_max_rss_bytes"]
    ):
        raise RuntimeError("run-log peak memory differs")
    fold_inventory = config.get("fold_inventory")
    if type(fold_inventory) is not list or len(fold_inventory) != 8:
        raise RuntimeError("run-log fold inventory differs")
    if any(
        type(row) is not dict
        or any(type(key) is not str for key in row)
        or set(row) != {"name", "kind"}
        or type(row.get("name")) is not str
        or type(row.get("kind")) is not str
        for row in fold_inventory
    ):
        raise RuntimeError("run-log fold inventory differs")
    expected_names = [row["name"] for row in fold_inventory]
    expected_kinds = [row["kind"] for row in fold_inventory]
    if (
        tuple(expected_names)
        != (*_CITY_DEVELOPMENT_FOLDS, *_DISTRICT_DEVELOPMENT_FOLDS)
        or tuple(expected_kinds) != ("city",) * 3 + ("district",) * 5
    ):
        raise RuntimeError("run-log fold inventory differs")
    events = payload["events"]
    elapsed_values = []
    for event in events:
        if (
            type(event) is not dict
            or any(type(key) is not str for key in event)
            or type(event.get("event")) is not str
            or type(event.get("state")) is not str
        ):
            raise RuntimeError("run-log event schema differs")
        elapsed_values.append(
            _run_log_nonnegative_float(
                event.get("elapsed_seconds"), "elapsed-time history"
            )
        )
    if (
        not events
        or set(events[0]) != {"event", "state", "elapsed_seconds"}
        or events[0]["event"] != "prepared"
        or events[0]["state"] != "prepared"
        or events[0]["elapsed_seconds"] != 0.0
    ):
        raise RuntimeError("run-log prepared event differs")
    if elapsed_values != sorted(elapsed_values) or top_elapsed != elapsed_values[-1]:
        raise RuntimeError("run-log elapsed-time history differs")

    cursor = 1
    completed_events = []
    fold_views = payload["fold_views"]
    if cursor < len(events):
        event = events[cursor]
        if set(event) != {"event", "state", "elapsed_seconds"} or event.get("event") != "execution_started" or event.get("state") != "running":
            raise RuntimeError("run-log execution-start event differs")
        cursor += 1
        if len(fold_views) != 8:
            raise RuntimeError("run-log loaded fold views differ")
        for index, view in enumerate(fold_views):
            _require_exact_run_log_fold_view_types(view)
            normalized = _validate_fold_view(
                view,
                expected_name=expected_names[index],
                expected_kind=expected_kinds[index],
            )
            if not _canonical_payload_equal(dict(view), normalized):
                raise RuntimeError("run-log loaded fold view inventory differs")
    elif fold_views:
        raise RuntimeError("prepared run-log cannot retain loaded fold views")

    while (
        cursor < len(events)
        and isinstance(events[cursor], Mapping)
        and events[cursor].get("event") == "fold_completed"
    ):
        event = events[cursor]
        expected_index = len(completed_events)
        if set(event) != {
            "event", "state", "elapsed_seconds", "fold", "fold_kind",
            "artifact_pair_sha256",
        } or event.get("state") != "running" or expected_index >= 8 or type(event.get("fold")) is not str or type(event.get("fold_kind")) is not str or event.get("fold") != expected_names[expected_index] or event.get("fold_kind") != expected_kinds[expected_index]:
            raise RuntimeError("run-log completed-fold event differs")
        pair = event.get("artifact_pair_sha256")
        if (
            type(pair) is not dict
            or any(type(key) is not str for key in pair)
            or set(pair) != {"fit.json", "per_series.json"}
        ):
            raise RuntimeError("run-log artifact pair schema differs")
        for digest in pair.values():
            _sha256_text(digest, "run-log artifact pair")
        root = DEVELOPMENT_ROOT / "folds" / str(event["fold"])
        for name, digest in pair.items():
            path = root / name
            try:
                observed_digest = _file_sha256(path)
            except RuntimeError as exc:
                raise RuntimeError("run-log artifact pair hash differs") from exc
            if observed_digest != digest:
                raise RuntimeError("run-log artifact pair hash differs")
        completed_events.append(event)
        cursor += 1
    completed_names = [event["fold"] for event in completed_events]
    if (
        any(type(name) is not str for name in payload["completed_folds"])
        or payload["completed_folds"] != completed_names
        or completed_names != expected_names[: len(completed_names)]
    ):
        raise RuntimeError("run-log completed-fold prefix differs")

    terminal = "running" if len(events) > 1 else "prepared"
    aggregate_sha = None
    if (
        cursor < len(events)
        and isinstance(events[cursor], Mapping)
        and events[cursor].get("event") == "folds_complete"
    ):
        event = events[cursor]
        if set(event) != {"event", "state", "elapsed_seconds"} or event.get("state") != "folds_complete" or len(completed_names) != 8:
            raise RuntimeError("run-log folds-complete event differs")
        terminal = "folds_complete"
        cursor += 1
    if (
        cursor < len(events)
        and isinstance(events[cursor], Mapping)
        and events[cursor].get("event") == "aggregated"
    ):
        event = events[cursor]
        if set(event) != {"event", "state", "elapsed_seconds", "aggregate_sha256"} or event.get("state") != "aggregated" or terminal != "folds_complete":
            raise RuntimeError("run-log aggregated event differs")
        aggregate_sha = event.get("aggregate_sha256")
        expected_aggregate_names = {
            "per_series.csv", "summary.json", "evidence_decision.json", "actuation.csv"
        }
        if (
            type(aggregate_sha) is not dict
            or any(type(key) is not str for key in aggregate_sha)
            or set(aggregate_sha) != expected_aggregate_names
        ):
            raise RuntimeError("run-log aggregate inventory differs")
        aggregate_paths = {
            "per_series.csv": DEVELOPMENT_ROOT / "per_series.csv",
            "summary.json": DEVELOPMENT_ROOT / "summary.json",
            "evidence_decision.json": DEVELOPMENT_ROOT / "evidence_decision.json",
            "actuation.csv": DEVELOPMENT_ANALYSIS_ROOT / "actuation.csv",
        }
        for name, digest in aggregate_sha.items():
            _sha256_text(digest, "run-log aggregate")
            path = aggregate_paths[name]
            try:
                observed_digest = _file_sha256(path)
            except RuntimeError as exc:
                raise RuntimeError("run-log aggregate hash differs") from exc
            if observed_digest != digest:
                raise RuntimeError("run-log aggregate hash differs")
        terminal = "aggregated"
        cursor += 1
    if (
        cursor < len(events)
        and isinstance(events[cursor], Mapping)
        and events[cursor].get("event") == "finalized"
    ):
        event = events[cursor]
        if set(event) != {"event", "state", "elapsed_seconds"} or event.get("state") != "finalized" or terminal != "aggregated":
            raise RuntimeError("run-log finalized event differs")
        terminal = "finalized"
        cursor += 1
    if cursor != len(events) or payload.get("state") != terminal:
        raise RuntimeError("run-log legal event prefix differs")
    end_utc = payload.get("end_utc")
    if terminal in {"aggregated", "finalized"}:
        if type(end_utc) is not str or not end_utc:
            raise RuntimeError("aggregated run-log end time differs")
    elif end_utc is not None:
        raise RuntimeError("unfinished run-log has an end time")
    if finalized is True and terminal != "finalized":
        raise RuntimeError("run-log is not finalized")
    if finalized is False and terminal == "finalized":
        raise RuntimeError("run-log is already finalized")
    return dict(payload)


def _authenticate_preparation(
    profile: Mapping[str, object], *, historical: bool = False
) -> tuple[dict[str, object], dict[str, object], tuple[object, ...]]:
    config_path = DEVELOPMENT_ROOT / "config.json"
    log_path = DEVELOPMENT_ROOT / "run_log.json"
    if _secure_path_mode(config_path) is None or _secure_path_mode(log_path) is None:
        raise RuntimeError("authenticated development preparation config is missing")
    config = _read_task6_json(config_path, "development preparation config")
    run_log = _read_task6_json(log_path, "development preparation run log")
    if not _canonical_payload_equal(
        {"runtime": config.get("runtime")}, {"runtime": _ordinary_payload(profile)}
    ):
        raise RuntimeError("prepared runtime identity differs")
    _authenticate_run_log(run_log, config, finalized=None)
    if historical:
        current_sources = _collect_runtime_source_inventory()
        current_tests = _collect_residual_test_inventory()
        if not _runtime_inventory_matches(
            config.get("runtime_source_inventory", {}), current_sources, historical=True
        ) or not _residual_test_inventory_matches(
            config.get("residual_tests", {}), current_tests, historical=True
        ):
            raise RuntimeError("historical runtime source inventory differs")
        return config, run_log, ()
    expected, folds = _current_preparation_material(profile, collect_inputs=True)
    if not _canonical_payload_equal(config, expected):
        raise RuntimeError("development preparation source/input/config identity differs")
    return config, run_log, folds


def _prepare_development(workers: object) -> dict[str, object]:
    workers = _require_production_workers(workers)
    profile = _validate_production_runtime(_measure_production_runtime(workers), workers)
    config, _ = _current_preparation_material(profile, collect_inputs=True)
    config_path = DEVELOPMENT_ROOT / "config.json"
    log_path = DEVELOPMENT_ROOT / "run_log.json"
    root_mode = _secure_path_mode(DEVELOPMENT_ROOT)
    if root_mode is not None:
        if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
            raise RuntimeError("development preparation root is invalid or symlinked")
        inventory = _secure_walk(DEVELOPMENT_ROOT, "development preparation")
        if any(not stat.S_ISREG(mode) for _, mode in inventory):
            raise RuntimeError("development preparation has unexpected directories")
        entries = sorted(
            path.relative_to(DEVELOPMENT_ROOT).as_posix()
            for path, _ in inventory
        )
        if entries != ["config.json", "run_log.json"]:
            raise RuntimeError("development preparation has unexpected files")
        existing = _read_task6_json(config_path, "development preparation config")
        run_log = _read_task6_json(log_path, "development preparation run log")
        if not _canonical_payload_equal(existing, config):
            raise RuntimeError("divergent development preparation")
        _authenticate_run_log(run_log, existing, finalized=False)
        if run_log.get("state") != "prepared":
            raise RuntimeError("development preparation is no longer a header")
        return existing
    run_log = _new_run_log(config, profile)
    _atomic_canonical_json(config_path, config)
    _atomic_canonical_json(log_path, run_log)
    return config


def _advance_run_log(
    run_log: Mapping[str, object],
    state: str,
    *,
    fold: DevelopmentFold | None = None,
    pair_sha256: Mapping[str, str] | None = None,
    fold_views: Sequence[Mapping[str, object]] | None = None,
    aggregate_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    unsigned = copy.deepcopy(dict(run_log))
    unsigned.pop("payload_sha256", None)
    previous_memory = {
        name: unsigned[name]
        for name in (
            "parent_max_rss_bytes",
            "completed_children_max_rss_bytes",
            "reported_peak_max_rss_bytes",
        )
    }
    if state == "finalized":
        elapsed = float(_runtime_monotonic()) - float(unsigned["monotonic_start"])
        if not math.isfinite(elapsed) or elapsed < float(unsigned["elapsed_seconds"]):
            raise RuntimeError("run-log monotonic clock moved backwards")
        memory = _runtime_peak_memory_sample(previous_memory)
    else:
        memory = _runtime_peak_memory_sample(previous_memory)
        elapsed = float(_runtime_monotonic()) - float(unsigned["monotonic_start"])
        if not math.isfinite(elapsed) or elapsed < float(unsigned["elapsed_seconds"]):
            raise RuntimeError("run-log monotonic clock moved backwards")
    event: dict[str, object] = {"state": state, "elapsed_seconds": elapsed}
    if fold is not None:
        if pair_sha256 is None or set(pair_sha256) != {"fit.json", "per_series.json"}:
            raise RuntimeError("run-log fold event lacks an artifact pair identity")
        event.update(
            {
                "event": "fold_completed",
                "fold": fold.name,
                "fold_kind": fold.kind,
                "artifact_pair_sha256": dict(pair_sha256),
            }
        )
        unsigned["completed_folds"] = [
            *unsigned["completed_folds"], fold.name
        ]
    elif state == "running":
        if fold_views is None or unsigned["fold_views"]:
            raise RuntimeError("run-log execution start lacks loaded fold views")
        event["event"] = "execution_started"
        unsigned["fold_views"] = [_ordinary_payload(row) for row in fold_views]
    elif state == "folds_complete":
        event["event"] = "folds_complete"
    elif state == "aggregated":
        if aggregate_sha256 is None:
            raise RuntimeError("run-log aggregate event lacks artifact identities")
        event.update(
            {"event": "aggregated", "aggregate_sha256": dict(aggregate_sha256)}
        )
    elif state == "finalized":
        event["event"] = "finalized"
    else:
        raise RuntimeError("unsupported run-log transition")
    unsigned["state"] = state
    unsigned["elapsed_seconds"] = elapsed
    unsigned.update(memory)
    unsigned["events"] = [*unsigned["events"], event]
    if state == "aggregated":
        unsigned["end_utc"] = _runtime_utc_now()
    elif state == "finalized":
        if not unsigned.get("end_utc"):
            raise RuntimeError("finalized run log lacks its aggregated end time")
        unsigned["end_utc"] = _runtime_utc_now()
    return _with_payload_sha(unsigned)


def _runtime_summary_projection(run_log: Mapping[str, object]) -> dict[str, object]:
    required = {
        "start_utc", "end_utc", "elapsed_seconds", "parent_max_rss_bytes",
        "completed_children_max_rss_bytes", "reported_peak_max_rss_bytes",
        "peak_memory_interpretation", "workers", "config_sha256",
    }
    if run_log.get("state") != "finalized" or any(name not in run_log for name in required):
        raise RuntimeError("finalized runtime summary is incomplete")
    return {name: run_log[name] for name in sorted(required)}


def _format_runtime_number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("runtime numeric value must be finite and non-boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("runtime numeric value must be finite")
    if number == 0.0:
        if math.copysign(1.0, number) < 0.0:
            raise ValueError("runtime numeric value cannot use signed zero")
        return "0.0"
    return repr(number)


def _runtime_csv_cell(value: object) -> str:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if isinstance(value, Real):
        return _format_runtime_number(value)
    if isinstance(value, str):
        return value
    raise TypeError("runtime CSV values must be ordinary JSON scalars")


def _encode_runtime_csv(
    header: Sequence[str], rows: Sequence[Sequence[object]]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(tuple(header))
    for row in rows:
        if len(row) != len(header):
            raise ValueError("runtime CSV row width differs from its header")
        writer.writerow(tuple(_runtime_csv_cell(value) for value in row))
    return output.getvalue().encode("utf-8")


def _build_development_evidence(
    folds: Sequence[DevelopmentFold],
    fit_records: Sequence[Mapping[str, object]],
    series_records: Sequence[Mapping[str, object]],
) -> DevelopmentGateEvidence:
    return DevelopmentGateEvidence(tuple(folds), tuple(fit_records), tuple(series_records))


def _fold_runtime_name(fold: object) -> str:
    return fold.name if isinstance(fold, DevelopmentFold) else fold["name"]  # type: ignore[index]


def _fold_runtime_kind(fold: object) -> str:
    return fold.kind if isinstance(fold, DevelopmentFold) else fold["kind"]  # type: ignore[index]


def _fold_scored_order(fold: object) -> list[str]:
    if isinstance(fold, DevelopmentFold):
        return [row.ref.key for row in fold.test]
    return list(fold["scored_series"])  # type: ignore[index]


def _fold_instance_series(fold: object) -> dict[str, str]:
    if isinstance(fold, DevelopmentFold):
        return {
            instance.path: row.ref.key
            for row in fold.test
            for year, instance in row.all_years.items()
            if year in row.test_years
        }
    return {
        row["instance"]: row["series"] for row in fold["scored_instances"]  # type: ignore[index]
    }


def _sealed_development_evidence(
    fold_views: Sequence[Mapping[str, object]],
    fit_records: Sequence[Mapping[str, object]],
    series_records: Sequence[Mapping[str, object]],
) -> DevelopmentGateEvidence:
    from iclr_residual_actuation import ActuationRecord

    if len(fold_views) != 8 or len(fit_records) != 8 or len(series_records) != 8:
        raise RuntimeError("sealed development fold inventory differs")
    primary_seeds = {}
    expected_scored = {}
    expected_static = {}
    static_results = {}
    city_rows = []
    district_rows = []
    normalized_views = []
    for view, fit_record, series_record in zip(fold_views, fit_records, series_records):
        normalized_view = _validate_fold_view(view)
        if not _canonical_payload_equal(dict(view), normalized_view):
            raise RuntimeError("sealed development fold view differs")
        name = normalized_view["name"]
        kind = normalized_view["kind"]
        for record in (fit_record, series_record):
            if (
                record.get("fold") != name
                or record.get("kind") != kind
                or record.get("fold_input_sha256") != normalized_view["fold_input_sha256"]
            ):
                raise RuntimeError("sealed development artifact identity differs")
        if series_record.get("fit_payload_sha256") != fit_record.get("payload_sha256"):
            raise RuntimeError("sealed development artifact pair link differs")
        result = series_record.get("result")
        if not isinstance(result, Mapping) or result.get("fold") != name or result.get("kind") != kind:
            raise RuntimeError("sealed development result identity differs")
        fold_input = fit_record.get("fold_input")
        series_fold_input = series_record.get("fold_input")
        if (
            not isinstance(fold_input, Mapping)
            or not isinstance(series_fold_input, Mapping)
            or fit_record.get("fold_input_sha256") != _canonical_sha256(fold_input)
            or not _canonical_payload_equal(
                dict(fold_input), dict(series_fold_input)
            )
        ):
            raise RuntimeError("sealed development fold-input identity differs")
        training_view = fold_input.get("training_view")
        scored_view = fold_input.get("scored_view")
        scans = result.get("actuation")
        if (
            not isinstance(training_view, list)
            or not isinstance(scored_view, list)
            or not isinstance(scans, list)
        ):
            raise RuntimeError("sealed development semantic fold inventory differs")
        try:
            training = [
                _canonical_series_identifier(row["series"], "sealed training view")
                for row in training_view
                if isinstance(row, Mapping)
            ]
            scored = [
                _canonical_series_identifier(row["series"], "sealed scored view")
                for row in scored_view
                if isinstance(row, Mapping)
            ]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("sealed development semantic fold inventory differs") from exc
        if len(training) != len(training_view) or len(scored) != len(scored_view):
            raise RuntimeError("sealed development semantic fold inventory differs")
        expected_instance_series = []
        for row, series in zip(scored_view, scored):
            years = row.get("scored_years")
            if (
                not isinstance(years, list)
                or any(type(year) is not int for year in years)
                or years != sorted(set(years))
            ):
                raise RuntimeError("sealed development scored-year inventory differs")
            expected_instance_series.extend([series] * len(years))
        if len(scans) != len(expected_instance_series):
            raise RuntimeError("sealed development actuation inventory differs")
        expected_view = _validate_fold_view(
            {
                "name": name,
                "kind": kind,
                "fold_input_sha256": _canonical_sha256(fold_input),
                "training_series": training,
                "scored_series": scored,
                "scored_instances": [
                    {
                        "series": series,
                        "instance": raw.get("instance")
                        if isinstance(raw, Mapping)
                        else None,
                    }
                    for series, raw in zip(expected_instance_series, scans)
                ],
            }
        )
        if not _canonical_payload_equal(normalized_view, expected_view):
            raise RuntimeError("sealed development fold view differs from fold input")
        if result.get("scored_series") != scored or result.get("fit_training_series") != training:
            raise RuntimeError("sealed development result inventory differs")
        retained = result.get("series_results")
        if not isinstance(retained, list):
            raise RuntimeError("sealed development result rows differ")
        try:
            rows = tuple(_coerce_development_result(row) for row in retained)
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeError("sealed development result rows differ") from exc
        expected_keys = [
            (name, seed, series) for seed in _DEVELOPMENT_SEEDS for series in scored
        ]
        if [(row.fold, row.seed, row.series) for row in rows] != expected_keys:
            raise RuntimeError("sealed development result row order differs")
        expected_instances = [row["instance"] for row in normalized_view["scored_instances"]]
        if not isinstance(scans, list) or [row.get("instance") for row in scans] != expected_instances:
            raise RuntimeError("sealed development actuation inventory differs")
        try:
            for raw in scans:
                record = ActuationRecord(**dict(raw))
                if not _canonical_payload_equal(asdict(record), dict(raw)):
                    raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("sealed development actuation payload differs") from exc
        primary = result.get("primary_seed")
        if type(primary) is not int or primary not in _DEVELOPMENT_SEEDS:
            raise RuntimeError("sealed development primary seed differs")
        raw_static = fit_record.get("expected_safe_static")
        supplied_static = result.get("safe_static")
        if not isinstance(raw_static, list) or not isinstance(supplied_static, Mapping):
            raise RuntimeError("sealed development safe-static payload differs")
        static_inventory = tuple(
            (row["identity"], row["family"])
            for row in raw_static
            if isinstance(row, Mapping) and set(row) == {"identity", "family"}
        )
        if len(static_inventory) != len(raw_static) or set(supplied_static) != {
            identity for identity, _ in static_inventory
        }:
            raise RuntimeError("sealed development safe-static inventory differs")
        safe_inventory_payload = [
            {"identity": identity, "family": family}
            for identity, family in static_inventory
        ]
        training_cities = tuple(
            sorted({"/".join(series.split("/")[:2]) for series in training})
        )
        authenticated_fit = _validate_development_fit_payload(
            fit_record.get("fit"),
            primary,
            safe_inventory_payload,
            anchor_search_config(),
            residual_search_config(),
            tuple(training),
            training_cities,
        )
        if authenticated_fit is None:
            raise RuntimeError("sealed development fit is incomplete")
        instance_series = {
            row["instance"]: row["series"] for row in normalized_view["scored_instances"]
        }
        actuated_by_series = {series: False for series in scored}
        for raw in scans:
            series = instance_series[raw["instance"]]
            actuated_by_series[series] = actuated_by_series[series] or raw["actuated"]
        _authenticate_development_scored_relationships(
            rows, supplied_static, authenticated_fit, actuated_by_series
        )
        primary_seeds[name] = primary
        expected_scored[name] = tuple(scored)
        expected_static[name] = static_inventory
        static_results[name] = supplied_static
        (city_rows if kind == "city" else district_rows).extend(rows)
        normalized_views.append(_ordinary_payload(normalized_view))
    _validate_development_inventory(
        city_rows, district_rows, primary_seeds, expected_scored,
        expected_static, static_results,
    )
    evidence = object.__new__(DevelopmentGateEvidence)
    object.__setattr__(evidence, "folds", tuple(normalized_views))
    object.__setattr__(evidence, "fit_records", tuple(_ImmutableJSONDict(row) for row in fit_records))
    object.__setattr__(evidence, "series_records", tuple(_ImmutableJSONDict(row) for row in series_records))
    object.__setattr__(evidence, "primary_seeds", _ImmutableJSONDict(primary_seeds))
    object.__setattr__(evidence, "expected_scored_series", _ImmutableJSONDict(expected_scored))
    object.__setattr__(evidence, "expected_safe_static", _ImmutableJSONDict(expected_static))
    object.__setattr__(evidence, "safe_static_results", _ImmutableJSONDict(static_results))
    object.__setattr__(evidence, "city_results", tuple(city_rows))
    object.__setattr__(evidence, "district_results", tuple(district_rows))
    return evidence


def _scientific_aggregate_rows(evidence: DevelopmentGateEvidence):
    rows_by_key = {
        (row.fold, row.seed, row.series): row
        for row in (*evidence.city_results, *evidence.district_results)
    }
    per_rows = []
    actuation_rows = []
    for fold in evidence.folds:
        fold_name = _fold_runtime_name(fold)
        fold_kind = _fold_runtime_kind(fold)
        scored_order = _fold_scored_order(fold)
        for seed in _DEVELOPMENT_SEEDS:
            for series in scored_order:
                row = rows_by_key[(fold_name, seed, series)]
                per_rows.append(
                    (
                        fold_name, fold_kind, seed, series, row.city, row.cluster,
                        row.residual["csd"], row.residual["welfare"], row.residual["exclusion"],
                        row.anchor["csd"], row.anchor["welfare"], row.anchor["exclusion"],
                        row.mes["csd"], row.mes["welfare"], row.mes["exclusion"],
                        float(row.residual["csd"]) - float(row.anchor["csd"]),
                        float(row.residual["welfare"]) - float(row.anchor["welfare"]),
                        float(row.residual["exclusion"]) - float(row.anchor["exclusion"]),
                        row.signatures["residual"], row.signatures["anchor"],
                        row.signatures["actuated"],
                    )
                )
        series_record = evidence.series_records[
            tuple(_fold_runtime_name(item) for item in evidence.folds).index(fold_name)
        ]
        result = series_record["result"]
        path_to_series = _fold_instance_series(fold)
        for record in result["actuation"]:
            instance = record["instance"]
            actuation_rows.append(
                (
                    fold_name, fold_kind, result["primary_seed"],
                    path_to_series[instance], instance, True, record["actuated"],
                    record["first_grid_t"], record["refined_low"],
                    record["refined_high"], record["csd_direction"],
                    record["anchor_csd"], record["changed_csd"],
                )
            )
    return tuple(per_rows), tuple(actuation_rows)


def _development_report_facts(
    evidence: DevelopmentGateEvidence,
) -> dict[str, object]:
    return {
        "folds": [
            {
                "fold": _fold_runtime_name(fold),
                "anchor_family": fit_record["fit"]["static"]["selected"]["anchor"][
                    "family"
                ],
                "primary_seed": fit_record["primary_seed"],
            }
            for fold, fit_record in zip(evidence.folds, evidence.fit_records)
        ]
    }


def _report_bytes(
    decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    artifact_hashes: Mapping[str, str],
) -> bytes:
    condition_order = (
        "primary_improves_anchor", "all_held_cities_negative",
        "district_fold_majority", "dual_reference_safety",
        "all_seeds_negative", "actuation_coverage",
    )
    threshold_order = (
        "bootstrap_draws", "bootstrap_seed", "bootstrap_upper_max",
        "district_negative_min", "welfare_floor", "actuation_rate_min",
        "actuated_city_min",
    )
    conditions = decision.get("conditions")
    thresholds = decision.get("thresholds")
    observed = decision.get("observed")
    facts = decision.get("report_facts")
    if (
        not isinstance(conditions, Mapping)
        or set(conditions) != set(condition_order)
        or not isinstance(thresholds, Mapping)
        or set(thresholds) != set(threshold_order)
        or not isinstance(observed, Mapping)
        or not isinstance(facts, Mapping)
        or not isinstance(facts.get("folds"), list)
    ):
        raise RuntimeError("development report factual projection is incomplete")
    derived_failed_conditions = [
        name for name in condition_order if conditions[name] is False
    ]
    failed_conditions = decision.get(
        "failed_conditions", derived_failed_conditions
    )
    if failed_conditions != derived_failed_conditions:
        raise RuntimeError("development report failed-condition projection differs")
    lines = [
        "# Development gate execution report", "",
        f"Classification: `{decision['classification']}`.", "",
        "## Six frozen conditions", "",
    ]
    for name in condition_order:
        value = conditions[name]
        if type(value) is not bool:
            raise RuntimeError("development report condition is not boolean")
        lines.append(
            f"- `{name}`: `{str(value).lower()}`; "
            f"failed condition: `{str(not value).lower()}`"
        )

    primary = observed.get("primary_inference")
    city_safety = observed.get("city_safety")
    cities = observed.get("actuated_cities")
    if (
        not isinstance(primary, Mapping)
        or not isinstance(city_safety, Mapping)
        or not isinstance(cities, list)
        or any(not isinstance(city, str) for city in cities)
    ):
        raise RuntimeError("development report observed evidence differs")
    threshold_observed = {
        "bootstrap_draws": primary.get("draws"),
        "bootstrap_seed": primary.get("seed"),
        "bootstrap_upper_max": primary.get("upper"),
        "district_negative_min": observed.get("negative_district_folds"),
        "welfare_floor": "authenticated city ratios listed below",
        "actuation_rate_min": observed.get("actuation_rate"),
        "actuated_city_min": len(cities),
    }
    if any(value is None for value in threshold_observed.values()):
        raise RuntimeError("development report threshold observation is incomplete")
    lines.extend(["", "## Frozen thresholds and observed values", ""])
    for name in threshold_order:
        lines.append(
            f"- `{name}`: threshold `{thresholds[name]}`; "
            f"observed `{threshold_observed[name]}`"
        )

    lines.extend(["", "## Primary inference", ""])
    for field in ("mean", "lower", "series_count", "cluster_count", "city_count"):
        if field in primary:
            lines.append(f"- `{field}`: `{primary[field]}`")
    held_city_means = observed.get("held_city_means", {})
    district_fold_means = observed.get("district_fold_means", {})
    if not isinstance(held_city_means, Mapping) or not isinstance(
        district_fold_means, Mapping
    ):
        raise RuntimeError("development report held-out means differ")
    lines.extend(["", "## Held-city means", ""])
    for city, value in sorted(held_city_means.items()):
        lines.append(f"- `{city}`: `{value}`")
    lines.extend(["", "## District-fold means", ""])
    for fold, value in sorted(district_fold_means.items()):
        lines.append(f"- `{fold}`: `{value}`")

    lines.extend(["", "## Fold selections", ""])
    for row in facts["folds"]:
        if not isinstance(row, Mapping) or set(row) != {
            "fold", "anchor_family", "primary_seed"
        }:
            raise RuntimeError("development report fold selection differs")
        lines.append(
            f"- `{row['fold']}`: anchor `{row['anchor_family']}`, "
            f"primary seed `{row['primary_seed']}`"
        )

    seed_means = observed.get("seed_equal_city_means")
    if not isinstance(seed_means, Mapping):
        raise RuntimeError("development report seed directions are incomplete")
    lines.extend(["", "## Per-seed directions", ""])
    for seed in (1, 2, 42):
        value = seed_means.get(str(seed))
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise RuntimeError("development report seed direction differs")
        direction = "negative" if float(value) < 0.0 else "nonnegative"
        lines.append(f"- Seed `{seed}`: `{direction}` (mean `{value}`)")

    lines.extend(
        [
            "", "## Actuation coverage", "",
            f"- Actuated series: `{observed.get('actuated_series')}`",
            f"- Held series: `{observed.get('held_series')}`",
            "- Actuated cities: " + ", ".join(f"`{city}`" for city in cities),
            "", "## Dual-reference safety", "",
        ]
    )
    safety_fields = (
        "residual_welfare", "anchor_welfare", "mes_welfare",
        "anchor_welfare_threshold", "mes_welfare_threshold",
        "anchor_welfare_ratio", "mes_welfare_ratio",
        "residual_exclusion", "anchor_exclusion", "mes_exclusion",
        "anchor_exclusion_threshold", "mes_exclusion_threshold",
    )
    comparison_order = (
        "welfare_vs_anchor", "welfare_vs_mes",
        "exclusion_vs_anchor", "exclusion_vs_mes",
    )
    derived_safety_failures = []
    for fold in sorted(city_safety):
        row = city_safety[fold]
        if not isinstance(row, Mapping) or not isinstance(row.get("city"), str):
            raise RuntimeError("development report safety row differs")
        comparisons = row.get("comparisons")
        if not isinstance(comparisons, Mapping) or set(comparisons) != set(
            comparison_order
        ):
            raise RuntimeError("development report safety comparisons differ")
        failed = []
        for comparison in comparison_order:
            value = comparisons[comparison]
            if type(value) is not bool:
                raise RuntimeError(
                    "development report safety comparison is not boolean"
                )
            if not value:
                failed.append(comparison)
        if row.get("failed_comparisons", failed) != failed:
            raise RuntimeError("development report failed safety comparisons differ")
        if failed:
            derived_safety_failures.append(fold)
        lines.append(
            f"- Fold `{fold}`, city `{row['city']}`; "
            f"safety failure: `{str(bool(failed)).lower()}`"
        )
        for field in safety_fields:
            if field not in row:
                raise RuntimeError("development report safety row is incomplete")
            lines.append(f"  - `{field}`: `{row[field]}`")
        for comparison in comparison_order:
            value = comparisons[comparison]
            lines.append(
                f"  - `{comparison}`: `{str(value).lower()}`; "
                f"failed comparison: `{str(not value).lower()}`"
            )
    if observed.get("safety_failures", derived_safety_failures) != derived_safety_failures:
        raise RuntimeError("development report safety-failure projection differs")

    lines.extend(["", "## Runtime", ""])
    for field in (
        "workers", "start_utc", "end_utc", "elapsed_seconds",
        "parent_max_rss_bytes", "completed_children_max_rss_bytes",
        "reported_peak_max_rss_bytes", "peak_memory_interpretation",
        "config_sha256",
    ):
        if field in runtime_summary:
            lines.append(f"- `{field}`: `{runtime_summary[field]}`")
    lines.extend(["", "## Covered artifact hashes", ""])
    for path, digest in sorted(artifact_hashes.items()):
        lines.append(f"- `{path}`: `{digest}`")
    lines.extend(
        [
            "",
            "This report records authenticated old-data development evidence only; it adds no causal or external-validation interpretation.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_development_aggregate_bytes(
    folds: Sequence[DevelopmentFold],
    fit_records: Sequence[Mapping[str, object]],
    series_records: Sequence[Mapping[str, object]],
    run_log: Mapping[str, object],
) -> dict[str, bytes]:
    evidence = _build_development_evidence(folds, fit_records, series_records)
    return _build_development_aggregate_from_evidence(evidence, run_log)


def _build_development_aggregate_from_evidence(
    evidence: DevelopmentGateEvidence, run_log: Mapping[str, object]
) -> dict[str, bytes]:
    decision = evaluate_development_gate(evidence)
    per_rows, actuation_rows = _scientific_aggregate_rows(evidence)
    per_series = _encode_runtime_csv(_PER_SERIES_HEADER, per_rows)
    actuation = _encode_runtime_csv(_ACTUATION_HEADER, actuation_rows)
    summary = _with_payload_sha(
        {
            "schema_version": 1,
            "counts": {
                "folds": len(evidence.folds),
                "series_rows": len(per_rows),
                "actuation_rows": len(actuation_rows),
                "manifest_entries": 23,
            },
            "decision": decision,
            "primary_seeds": dict(evidence.primary_seeds),
            "selected_anchor_families": {
                _fold_runtime_name(fold): fit_record["fit"]["static"]["selected"][
                    "anchor"
                ]["family"]
                for fold, fit_record in zip(evidence.folds, evidence.fit_records)
            },
        }
    )
    decision_bytes = _canonical_json_bytes(decision)
    summary_bytes = _canonical_json_bytes(summary)
    scientific_hashes = {
        "results/iclr_residual_upgrade/development/per_series.csv": hashlib.sha256(per_series).hexdigest(),
        "results/iclr_residual_upgrade/development/summary.json": hashlib.sha256(summary_bytes).hexdigest(),
        "results/iclr_residual_upgrade/development/evidence_decision.json": hashlib.sha256(decision_bytes).hexdigest(),
        "analysis-output/iclr_residual_upgrade/development/actuation.csv": hashlib.sha256(actuation).hexdigest(),
    }
    if run_log.get("state") == "finalized" and all(
        name in run_log
        for name in (
            "start_utc", "end_utc", "parent_max_rss_bytes",
            "completed_children_max_rss_bytes", "reported_peak_max_rss_bytes",
            "peak_memory_interpretation", "workers", "config_sha256",
        )
    ):
        runtime_summary = _runtime_summary_projection(run_log)
    else:
        runtime_summary = _ordinary_payload(run_log)
    report_decision = {
        **decision,
        "report_facts": _development_report_facts(evidence),
    }
    report = _report_bytes(report_decision, runtime_summary, scientific_hashes)
    return {
        "per_series.csv": per_series,
        "summary.json": summary_bytes,
        "evidence_decision.json": decision_bytes,
        "actuation.csv": actuation,
        "report.md": report,
    }


def _logical_development_paths(folds: Sequence[object]) -> dict[str, Path]:
    result_prefix = "results/iclr_residual_upgrade/development"
    analysis_prefix = "analysis-output/iclr_residual_upgrade/development"
    paths = {
        f"{result_prefix}/config.json": DEVELOPMENT_ROOT / "config.json",
        f"{result_prefix}/run_log.json": DEVELOPMENT_ROOT / "run_log.json",
        f"{result_prefix}/per_series.csv": DEVELOPMENT_ROOT / "per_series.csv",
        f"{result_prefix}/summary.json": DEVELOPMENT_ROOT / "summary.json",
        f"{result_prefix}/evidence_decision.json": DEVELOPMENT_ROOT / "evidence_decision.json",
        f"{analysis_prefix}/actuation.csv": DEVELOPMENT_ANALYSIS_ROOT / "actuation.csv",
        f"{analysis_prefix}/report.md": DEVELOPMENT_ANALYSIS_ROOT / "report.md",
    }
    for fold in folds:
        name = _fold_runtime_name(fold)
        paths[f"{result_prefix}/folds/{name}/fit.json"] = (
            DEVELOPMENT_ROOT / "folds" / name / "fit.json"
        )
        paths[f"{result_prefix}/folds/{name}/per_series.json"] = (
            DEVELOPMENT_ROOT / "folds" / name / "per_series.json"
        )
    return dict(sorted(paths.items()))


def _manifest_bytes(paths: Mapping[str, Path]) -> bytes:
    lines = []
    for logical, path in sorted(paths.items()):
        _canonical_persisted_path(
            logical,
            approved_prefixes=(
                "results/iclr_residual_upgrade/development/",
                "analysis-output/iclr_residual_upgrade/development/",
            ),
        )
        try:
            digest = _file_sha256(path)
        except RuntimeError as exc:
            raise RuntimeError(
                f"manifest artifact is missing or symlinked: {logical}"
            ) from exc
        lines.append(f"{digest}  {logical}\n")
    if len(lines) != 23:
        raise RuntimeError("development manifest must contain exactly 23 artifacts")
    return "".join(lines).encode("utf-8")


def _expected_output_directories(folds: Sequence[object]) -> set[Path]:
    return {
        DEVELOPMENT_ROOT,
        DEVELOPMENT_ROOT / "folds",
        *(DEVELOPMENT_ROOT / "folds" / _fold_runtime_name(fold) for fold in folds),
        DEVELOPMENT_ANALYSIS_ROOT,
    }


def _authenticate_development_manifest() -> dict[str, str]:
    run_log = _read_task6_json(
        DEVELOPMENT_ROOT / "run_log.json", "development preparation run log"
    )
    folds = run_log.get("fold_views")
    if not isinstance(folds, list):
        raise RuntimeError("development manifest fold inventory differs")
    expected_paths = _logical_development_paths(folds)
    manifest_path = DEVELOPMENT_ROOT / "manifest.sha256"
    expected_files = set(expected_paths.values()) | {manifest_path}
    observed_files = set()
    observed_directories = set()
    for root in (DEVELOPMENT_ROOT, DEVELOPMENT_ANALYSIS_ROOT):
        mode = _secure_path_mode(root)
        if mode is None or stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise RuntimeError("development manifest root is missing or symlinked")
        observed_directories.add(root)
        for path, entry_mode in _secure_walk(root, "development manifest"):
            if stat.S_ISDIR(entry_mode):
                observed_directories.add(path)
            elif stat.S_ISREG(entry_mode):
                observed_files.add(path)
            else:
                raise RuntimeError("development manifest inventory contains a special file")
    if observed_files != expected_files or observed_directories != _expected_output_directories(folds):
        raise RuntimeError("development manifest file/directory inventory differs")
    try:
        encoded = _read_regular_bytes(manifest_path, "development manifest")
        text = encoded.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError("development manifest encoding differs") from exc
    expected = _manifest_bytes(expected_paths)
    if encoded != expected:
        raise RuntimeError("development manifest bytes or hashes differ")
    result = {}
    for line in text.splitlines():
        if re.fullmatch(r"[0-9a-f]{64}  [^\\\n]+", line) is None:
            raise RuntimeError("development manifest line differs")
        digest, logical = line.split("  ", 1)
        if logical in result:
            raise RuntimeError("development manifest contains duplicate paths")
        result[logical] = digest
    return result


def _write_immutable_bytes(
    path: Path, encoded: bytes, *, approved_root: Path | None = None
) -> None:
    with _secure_parent_directory(
        path, create=True, approved_root=approved_root
    ) as (parent_fd, name):
        item = _descriptor_leaf_stat(parent_fd, name)
        if item is not None:
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISREG(item.st_mode):
                raise RuntimeError(
                    "immutable output artifact is symlinked or not regular: "
                    f"{path.name}"
                )
            if _read_regular_bytes(
                path,
                "immutable development artifact",
                approved_root=approved_root,
            ) != encoded:
                raise RuntimeError(
                    f"divergent immutable development artifact: {path.name}"
                )
            return
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o644,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            if _read_regular_bytes(
                path,
                "immutable development artifact",
                approved_root=approved_root,
            ) != encoded:
                raise RuntimeError(
                    f"divergent immutable development artifact: {path.name}"
                )
            return
        except OSError as exc:
            raise RuntimeError(
                f"immutable development artifact cannot be created: {path.name}"
            ) from exc
        try:
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            try:
                os.unlink(name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise
        else:
            os.close(descriptor)
            os.fsync(parent_fd)


def _folds_for_execution(prepared: Sequence[object]) -> tuple[DevelopmentFold, ...]:
    if prepared and all(isinstance(fold, DevelopmentFold) for fold in prepared):
        folds = tuple(prepared)
    else:
        folds = tuple(_load_development_folds_unchecked())
    expected = (*_CITY_DEVELOPMENT_FOLDS, *_DISTRICT_DEVELOPMENT_FOLDS)
    if tuple(fold.name for fold in folds) != expected:
        raise RuntimeError("loaded development fold inventory differs from preparation")
    return folds


def _pair_records(
    fold: DevelopmentFold,
) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    root = DEVELOPMENT_ROOT / "folds" / fold.name
    fit_path = root / "fit.json"
    series_path = root / "per_series.json"
    fit_mode = _secure_path_mode(fit_path)
    series_mode = _secure_path_mode(series_path)
    if (fit_mode is None) != (series_mode is None):
        raise RuntimeError(f"partial development artifact pair for {fold.name}")
    if fit_mode is None or series_mode is None:
        raise RuntimeError(f"development artifact pair is missing for {fold.name}")
    if not stat.S_ISREG(fit_mode) or not stat.S_ISREG(series_mode):
        raise RuntimeError(f"development artifact pair is symlinked for {fold.name}")
    fit_record = _read_authenticated_development_artifact(fit_path)
    series_record = _read_authenticated_development_artifact(series_path)
    _authenticate_development_artifact_pair(
        fold, fit_record, series_record, anchor_search_config(), residual_search_config()
    )
    pair = {
        "fit.json": _file_sha256(fit_path),
        "per_series.json": _file_sha256(series_path),
    }
    return fit_record, series_record, pair


def _output_tree_inventory(root: Path, label: str) -> tuple[bool, set[str], set[str]]:
    mode = _secure_path_mode(root)
    if mode is None:
        return False, set(), set()
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"{label} output root is symlinked or not a directory")
    files = set()
    directories = set()
    for path, entry_mode in _secure_walk(root, f"{label} output"):
        relative = path.relative_to(root).as_posix()
        if stat.S_ISREG(entry_mode):
            files.add(relative)
        elif stat.S_ISDIR(entry_mode):
            directories.add(relative)
        else:
            raise RuntimeError(f"{label} output inventory contains a special entry")
    return True, files, directories


def _preflight_execution_tree(
    folds: Sequence[DevelopmentFold], run_log: Mapping[str, object]
) -> int | None:
    """Authenticate the complete legal crash prefix before any action."""

    state = run_log.get("state")
    if state not in {"prepared", "running", "folds_complete", "aggregated", "finalized"}:
        raise RuntimeError("development output state is unsupported")
    result_exists, result_files, result_directories = _output_tree_inventory(
        DEVELOPMENT_ROOT, "development"
    )
    analysis_exists, analysis_files, analysis_directories = _output_tree_inventory(
        DEVELOPMENT_ANALYSIS_ROOT, "development analysis"
    )
    if not result_exists or analysis_directories:
        raise RuntimeError("development output inventory differs")
    base_files = {"config.json", "run_log.json"}
    pair_names = {
        f"folds/{fold.name}/{filename}"
        for fold in folds
        for filename in ("fit.json", "per_series.json")
    }
    result_aggregates = {
        "per_series.csv", "summary.json", "evidence_decision.json"
    }
    known_result = base_files | pair_names | result_aggregates | {"manifest.sha256"}
    known_analysis = {"actuation.csv", "report.md"}
    if result_files - known_result or analysis_files - known_analysis:
        raise RuntimeError("development output inventory contains an extra artifact")
    if not base_files <= result_files:
        raise RuntimeError("development output preparation files are missing")

    present_pairs = []
    for index, fold in enumerate(folds):
        fit_name = f"folds/{fold.name}/fit.json"
        series_name = f"folds/{fold.name}/per_series.json"
        fit_present = fit_name in result_files
        series_present = series_name in result_files
        if fit_present != series_present:
            raise RuntimeError("partial development artifact pair in output prefix")
        if fit_present:
            present_pairs.append(index)
    expected_pair_directories = (
        {"folds"}
        | {f"folds/{folds[index].name}" for index in present_pairs}
        if present_pairs
        else set()
    )
    if result_directories != expected_pair_directories:
        raise RuntimeError("development fold directory inventory differs")

    completed = run_log.get("completed_folds")
    if not isinstance(completed, list):
        raise RuntimeError("development completed-fold prefix differs")
    completed_count = len(completed)
    expected_completed = [fold.name for fold in folds[:completed_count]]
    if completed != expected_completed:
        raise RuntimeError("development completed-fold prefix differs")
    orphan = None
    if state == "prepared":
        if present_pairs:
            raise RuntimeError("prepared development pair prefix is impossible")
    elif state == "running":
        expected = list(range(completed_count))
        if present_pairs == [*expected, completed_count] and completed_count < len(folds):
            orphan = completed_count
        elif present_pairs != expected:
            raise RuntimeError("running development orphan-pair prefix is impossible")
    elif present_pairs != list(range(len(folds))):
        raise RuntimeError("durable development state lacks the complete pair prefix")
    for index in present_pairs:
        _pair_records(folds[index])

    aggregate_order = (
        ("result", "per_series.csv"),
        ("result", "summary.json"),
        ("result", "evidence_decision.json"),
        ("analysis", "actuation.csv"),
    )
    present_aggregates = [
        index
        for index, (root_name, name) in enumerate(aggregate_order)
        if name in (result_files if root_name == "result" else analysis_files)
    ]
    report_present = "report.md" in analysis_files
    manifest_present = "manifest.sha256" in result_files
    if manifest_present:
        raise RuntimeError("unsealed development state contains a manifest")
    if state in {"prepared", "running"}:
        if present_aggregates or report_present or analysis_exists:
            raise RuntimeError("unfinished development state has aggregate output")
    elif state == "folds_complete":
        if present_aggregates != list(range(len(present_aggregates))) or report_present:
            raise RuntimeError("development aggregate prefix order differs")
    elif state == "aggregated":
        if present_aggregates != list(range(4)) or report_present:
            raise RuntimeError(
                "aggregated development state has an impossible report prefix"
            )
    elif state == "finalized":
        if present_aggregates != list(range(4)):
            raise RuntimeError("finalized development state lacks exact aggregates")
    return orphan


def _validate_existing_aggregate_prefix(
    expected: Mapping[str, bytes], state: str
) -> None:
    for name, path in _aggregate_output_paths().items():
        mode = _secure_path_mode(path)
        if mode is None:
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"development aggregate is symlinked: {name}")
        if _read_regular_bytes(path, f"development aggregate {name}") != expected[name]:
            raise RuntimeError(f"development aggregate is divergent: {name}")


def _aggregate_output_paths() -> dict[str, Path]:
    return {
        "per_series.csv": DEVELOPMENT_ROOT / "per_series.csv",
        "summary.json": DEVELOPMENT_ROOT / "summary.json",
        "evidence_decision.json": DEVELOPMENT_ROOT / "evidence_decision.json",
        "actuation.csv": DEVELOPMENT_ANALYSIS_ROOT / "actuation.csv",
    }


def _report_evidence_hashes(folds: Sequence[object]) -> dict[str, str]:
    excluded = {"analysis-output/iclr_residual_upgrade/development/report.md"}
    result = {}
    for logical, path in _logical_development_paths(folds).items():
        if logical in excluded:
            continue
        mode = _secure_path_mode(path)
        if mode is None or not stat.S_ISREG(mode):
            raise RuntimeError("report evidence artifact is missing or not regular")
        result[logical] = _file_sha256(path)
    expected = set(_logical_development_paths(folds)) - excluded
    if set(result) != expected:
        raise RuntimeError("report evidence artifact inventory differs")
    return result


def _run_guarded_development(workers: object) -> dict[str, object]:
    workers = _require_production_workers(workers)
    profile = _validate_production_runtime(_measure_production_runtime(workers), workers)
    config, run_log, prepared = _authenticate_preparation(profile, historical=False)
    manifest_mode = _secure_path_mode(DEVELOPMENT_ROOT / "manifest.sha256")
    if manifest_mode is not None:
        if not stat.S_ISREG(manifest_mode):
            raise RuntimeError("development manifest is symlinked or not regular")
        _verify_development_outputs()
        return _read_sealed_decision_noatime()
    folds = _folds_for_execution(prepared)
    if [_prepared_fold_binding(fold) for fold in folds] != config["fold_bindings"]:
        raise RuntimeError("loaded development fold binding differs from preparation")
    fold_views = [_loaded_fold_view(fold) for fold in folds]
    if run_log["state"] != "prepared" and run_log["fold_views"] != fold_views:
        raise RuntimeError("loaded semantic fold views differ from the run log")
    orphan = _preflight_execution_tree(folds, run_log)
    log_path = DEVELOPMENT_ROOT / "run_log.json"
    if run_log["state"] == "prepared":
        candidate = _advance_run_log(run_log, "running", fold_views=fold_views)
        _authenticate_run_log(candidate, config, finalized=False)
        _atomic_canonical_json(log_path, candidate)
        run_log = candidate

    if orphan is not None:
        orphan_fold = folds[orphan]
        _, _, pair = _pair_records(orphan_fold)
        candidate = _advance_run_log(
            run_log, "running", fold=orphan_fold, pair_sha256=pair
        )
        _authenticate_run_log(candidate, config, finalized=False)
        _atomic_canonical_json(log_path, candidate)
        run_log = candidate

    completed = list(run_log["completed_folds"])
    fit_records = []
    series_records = []
    for index, fold in enumerate(folds):
        root = DEVELOPMENT_ROOT / "folds" / fold.name
        existing = index < len(completed)
        if index < len(completed):
            if completed[index] != fold.name or not existing:
                raise RuntimeError("development completed-fold prefix differs")
            fit_record, series_record, pair = _pair_records(fold)
            status = "resumed"
        elif run_log["state"] == "running":
            _authenticate_run_log(run_log, config, finalized=False)
            if existing:
                fit_record, series_record, pair = _pair_records(fold)
                status = "resumed"
            else:
                fit_and_evaluate_development_fold(fold, workers)
                fit_record, series_record, pair = _pair_records(fold)
                status = "completed"
            candidate = _advance_run_log(
                run_log, "running", fold=fold, pair_sha256=pair
            )
            _authenticate_run_log(candidate, config, finalized=False)
            _atomic_canonical_json(log_path, candidate)
            run_log = candidate
            completed.append(fold.name)
        else:
            if not existing:
                raise RuntimeError("durable run-log state lacks a completed fold pair")
            fit_record, series_record, pair = _pair_records(fold)
            status = "resumed"
        fit_records.append(fit_record)
        series_records.append(series_record)
        selected = fit_record["fit"]["static"]["selected"]
        logical = f"results/iclr_residual_upgrade/development/folds/{fold.name}/per_series.json"
        print(
            "DEVELOPMENT_FOLD "
            f"fold={fold.name} status={status} elapsed={run_log['elapsed_seconds']} "
            f"peak_memory={run_log['reported_peak_max_rss_bytes']} "
            f"anchor={selected['anchor']['family']} seed={fit_record['primary_seed']} "
            f"artifact={logical}"
        )
    if run_log["state"] == "running":
        candidate = _advance_run_log(run_log, "folds_complete")
        _authenticate_run_log(candidate, config, finalized=False)
        _atomic_canonical_json(log_path, candidate)
        run_log = candidate

    evidence = _build_development_evidence(folds, fit_records, series_records)
    decision = evaluate_development_gate(evidence)
    aggregate = _build_development_aggregate_from_evidence(evidence, run_log)
    _validate_existing_aggregate_prefix(aggregate, str(run_log["state"]))
    output_paths = _aggregate_output_paths()
    for name, path in output_paths.items():
        _authenticate_run_log(
            run_log, config, finalized=run_log["state"] == "finalized"
        )
        _write_immutable_bytes(path, aggregate[name])
    aggregate_hashes = {
        name: hashlib.sha256(aggregate[name]).hexdigest() for name in output_paths
    }
    if run_log["state"] == "folds_complete":
        candidate = _advance_run_log(
            run_log, "aggregated", aggregate_sha256=aggregate_hashes
        )
        _authenticate_run_log(candidate, config, finalized=False)
        _atomic_canonical_json(log_path, candidate)
        run_log = candidate
    elif run_log["state"] not in {"aggregated", "finalized"}:
        raise RuntimeError("development aggregate run-log prefix differs")

    report_decision = {
        **decision,
        "report_facts": _development_report_facts(evidence),
    }
    if run_log["state"] == "aggregated":
        candidate = _advance_run_log(run_log, "finalized")
        _authenticate_run_log(run_log, config, finalized=False)
        _authenticate_run_log(candidate, config, finalized=True)
        _atomic_canonical_json(log_path, candidate)
        run_log = candidate
    elif run_log["state"] != "finalized":
        raise RuntimeError("development finalization run-log prefix differs")

    report = _report_bytes(
        report_decision,
        _runtime_summary_projection(run_log),
        _report_evidence_hashes(folds),
    )
    _write_immutable_bytes(DEVELOPMENT_ANALYSIS_ROOT / "report.md", report)

    _authenticate_run_log(run_log, config, finalized=True)
    manifest = _manifest_bytes(_logical_development_paths(fold_views))
    _write_immutable_bytes(DEVELOPMENT_ROOT / "manifest.sha256", manifest)
    _authenticate_run_log(run_log, config, finalized=True)
    _authenticate_development_manifest()
    return decision


def run_development(workers: int) -> dict[str, object]:
    """Compatibility API for the guarded, prepared Task 6 execution path."""

    return _run_guarded_development(workers)


def _authenticate_final_run_log(
    run_log: Mapping[str, object], config: Mapping[str, object]
) -> None:
    _authenticate_run_log(run_log, config, finalized=True)


def _load_sealed_records(
    fold_views: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    fit_records = []
    series_records = []
    for view in fold_views:
        name = view["name"]
        root = DEVELOPMENT_ROOT / "folds" / name
        fit_records.append(
            _read_authenticated_development_artifact(root / "fit.json")
        )
        series_records.append(
            _read_authenticated_development_artifact(root / "per_series.json")
        )
    return tuple(fit_records), tuple(series_records)


@contextmanager
def _noatime_sealed_output_snapshot():
    """Verify cloned APFS trees so source content and metadata remain untouched."""

    if sys.platform != "darwin":
        raise RuntimeError("read-only verification requires supported no-atime semantics")
    for root in (DEVELOPMENT_ROOT, DEVELOPMENT_ANALYSIS_ROOT):
        mode = _secure_path_mode(root)
        if mode is None or stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise RuntimeError("sealed development root is missing or symlinked")
    try:
        library = ctypes.CDLL(None, use_errno=True)
        clonefile = library.clonefile
        clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        clonefile.restype = ctypes.c_int
    except Exception as exc:
        raise RuntimeError(
            "read-only verification cannot establish no-atime semantics"
        ) from exc
    original_result = DEVELOPMENT_ROOT
    original_analysis = DEVELOPMENT_ANALYSIS_ROOT
    with tempfile.TemporaryDirectory(prefix="task6-verify-clone-") as temporary:
        temporary_root = Path(temporary).resolve(strict=True)
        cloned_result = temporary_root / "result"
        cloned_analysis = temporary_root / "analysis"
        for source, destination in (
            (original_result, cloned_result),
            (original_analysis, cloned_analysis),
        ):
            try:
                resolved_source = source.resolve(strict=True)
                result = clonefile(
                    os.fsencode(resolved_source), os.fsencode(destination), 0x0008
                )
            except Exception as exc:
                raise RuntimeError(
                    "read-only verification cannot establish no-atime semantics"
                ) from exc
            if result != 0:
                error = ctypes.get_errno()
                raise RuntimeError(
                    "read-only verification cannot establish no-atime semantics"
                ) from OSError(error, os.strerror(error))
        globals()["DEVELOPMENT_ROOT"] = cloned_result
        globals()["DEVELOPMENT_ANALYSIS_ROOT"] = cloned_analysis
        try:
            yield
        finally:
            globals()["DEVELOPMENT_ROOT"] = original_result
            globals()["DEVELOPMENT_ANALYSIS_ROOT"] = original_analysis


def _verify_development_outputs_in_snapshot() -> tuple[str, str]:
    workers = _APPROVED_WORKERS
    profile = _validate_production_runtime(_measure_production_runtime(workers), workers)
    config, run_log, _ = _authenticate_preparation(profile, historical=True)
    _authenticate_final_run_log(run_log, config)
    _authenticate_development_manifest()
    fold_views = run_log.get("fold_views")
    if not isinstance(fold_views, list):
        raise RuntimeError("sealed development fold views differ")
    fit_records, series_records = _load_sealed_records(fold_views)
    evidence = _sealed_development_evidence(fold_views, fit_records, series_records)
    aggregate = _build_development_aggregate_from_evidence(evidence, run_log)
    expected = {
        "per_series.csv": DEVELOPMENT_ROOT / "per_series.csv",
        "summary.json": DEVELOPMENT_ROOT / "summary.json",
        "evidence_decision.json": DEVELOPMENT_ROOT / "evidence_decision.json",
        "actuation.csv": DEVELOPMENT_ANALYSIS_ROOT / "actuation.csv",
    }
    for name, path in expected.items():
        if _read_regular_bytes(path, f"sealed development aggregate {name}") != aggregate[name]:
            raise RuntimeError(f"sealed development aggregate differs: {name}")
    decision = json.loads(aggregate["evidence_decision.json"].decode("utf-8"))
    report_decision = {
        **decision,
        "report_facts": _development_report_facts(evidence),
    }
    report = _report_bytes(
        report_decision,
        _runtime_summary_projection(run_log),
        _report_evidence_hashes(fold_views),
    )
    if _read_regular_bytes(
        DEVELOPMENT_ANALYSIS_ROOT / "report.md", "sealed development report"
    ) != report:
        raise RuntimeError("sealed development report differs")
    summary_sha = hashlib.sha256(aggregate["summary.json"]).hexdigest()
    decision_sha = hashlib.sha256(aggregate["evidence_decision.json"]).hexdigest()
    return summary_sha, decision_sha


def _verify_development_outputs() -> tuple[str, str]:
    with _noatime_sealed_output_snapshot():
        return _verify_development_outputs_in_snapshot()


def _read_sealed_decision_noatime() -> dict[str, object]:
    with _noatime_sealed_output_snapshot():
        path = DEVELOPMENT_ROOT / "evidence_decision.json"
        try:
            encoded = _read_regular_bytes(path, "sealed development decision")
            payload = json.loads(encoded.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError("sealed development decision differs") from exc
        if not isinstance(payload, dict) or encoded != _canonical_json_bytes(payload):
            raise RuntimeError("sealed development decision differs")
        return payload


def _smoke_training_data() -> list[SeriesData]:
    projects = {
        "shared": Project("shared", 1.0, None),
        "young": Project("young", 1.0, None),
        "senior": Project("senior", 1.0, None),
    }
    votes = [
        Vote("v1", ("shared", "young"), age=20, sex="F"),
        Vote("v2", ("shared", "young"), age=35, sex="M"),
        Vote("v3", ("shared", "senior"), age=50, sex="F"),
        Vote("v4", ("shared", "senior"), age=70, sex="M"),
    ]
    instance = PBInstance(
        path="smoke-training://Poland/Synthetic/one/2022",
        meta={"budget": "2", "instance": "synthetic_2022"},
        projects=projects,
        votes=votes,
    )
    ref = SeriesRef(
        key="Poland/Synthetic/one",
        years=(2022,),
        paths=(Path("smoke-training.pb"),),
    )
    return [
        SeriesData(
            ref=ref,
            train_years=(2022,),
            test_years=(),
            train_only={2022: instance},
            all_years={2022: instance},
        )
    ]


def _smoke_development_series(
    key: str, *, scored: bool
) -> SeriesData:
    base = _smoke_training_data()[0].train_only[2022]

    def instance(year: int) -> PBInstance:
        return PBInstance(
            path=f"smoke-development://{key}/{year}",
            meta={"budget": "2", "instance": f"synthetic_{year}"},
            projects=dict(base.projects),
            votes=list(base.votes),
        )

    years = (2022, 2023) if scored else (2022,)
    instances = {year: instance(year) for year in years}
    return SeriesData(
        ref=SeriesRef(key, years, ()),
        train_years=() if scored else (2022,),
        test_years=(2023,) if scored else (),
        train_only={} if scored else {2022: instances[2022]},
        all_years=instances,
    )


def _smoke_development_folds() -> tuple[DevelopmentFold, DevelopmentFold]:
    return (
        DevelopmentFold(
            "smoke_city",
            "city",
            (_smoke_development_series("Poland/SmokeTrain/city", scored=False),),
            (_smoke_development_series("Poland/SmokeCity/held", scored=True),),
            "0" * 64,
            {"synthetic": True},
        ),
        DevelopmentFold(
            "smoke_district",
            "district",
            (_smoke_development_series("Poland/SmokeDistrict/train", scored=False),),
            (_smoke_development_series("Poland/SmokeDistrict/held", scored=True),),
            "1" * 64,
            {"synthetic": True},
        ),
    )


def _execute_smoke_development_fold(
    fold: DevelopmentFold, workers: int
) -> dict[str, object]:
    private_training = tuple(
        _materialize_development_series(row) for row in fold.train
    )
    fit = _fit_development_training(private_training, workers, smoke=True)
    result = _evaluate_development_scored(fold, fit, workers)
    expected_safe_static = _expected_safe_static_from_fit(fit, result)
    _validate_development_fit_payload(
        _fit_development_payload(fit),
        fit.primary_seed,
        expected_safe_static,
        smoke_anchor_search_config(),
        smoke_residual_search_config(),
        tuple(row.ref.key for row in fold.train),
        tuple(
            sorted(
                {"/".join(row.ref.key.split("/")[:2]) for row in fold.train}
            )
        ),
    )
    primary = next(row for row in fit.seeds if row.seed == fit.primary_seed).selected
    return {
        "fold": fold.name,
        "kind": fold.kind,
        "primary_seed": fit.primary_seed,
        "amplitude": primary.amplitude,
        "weights": list(primary.weights),
        "loss": primary.loss,
        "series_count": len(result["series_results"]),
        "actuated": any(row["signatures"]["actuated"] for row in result["series_results"]),
    }


def _run_smoke_development(workers: int) -> int:
    if workers < 1:
        raise ValueError("workers must be positive")
    folds = _smoke_development_folds()
    summaries = [_execute_smoke_development_fold(fold, workers) for fold in folds]
    payload = {
        "schema_version": 1,
        "classification": "smoke_only",
        "synthetic_only": True,
        "fold_kinds": [fold.kind for fold in folds],
        "anchor_config": {
            **asdict(smoke_anchor_search_config()),
            "seeds": list(smoke_anchor_search_config().seeds),
        },
        "residual_config": {
            **asdict(smoke_residual_search_config()),
            "seeds": list(smoke_residual_search_config().seeds),
        },
        "folds": summaries,
    }
    digest = _write_canonical_json(SMOKE_DEVELOPMENT_ARTIFACT, payload)
    print(
        "SMOKE_DEVELOPMENT_CONFIG "
        + json.dumps(
            {
                "workers": workers,
                "anchor": payload["anchor_config"],
                "residual": payload["residual_config"],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    print("SMOKE_DEVELOPMENT_CLASSIFICATION smoke_only")
    print(f"SMOKE_DEVELOPMENT_ARTIFACT {SMOKE_DEVELOPMENT_ARTIFACT}")
    print(f"SMOKE_DEVELOPMENT_SHA256 {digest}")
    return 0


def _run_smoke_anchor(workers: int) -> int:
    config = smoke_anchor_search_config()
    print(
        "SMOKE_CONFIG "
        + json.dumps(
            {**asdict(config), "seeds": list(config.seeds)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    fit = fit_static_anchor(_smoke_training_data(), config, workers)
    payload = _fit_payload(fit)
    digest = _write_canonical_json(SMOKE_ARTIFACT, payload)
    mes_available = payload["mes_fallback"]["available"]
    mes_safe = payload["mes_fallback"]["safe"]
    print(
        "MES_SAFE_FALLBACK "
        f"available={str(mes_available).lower()} safe={str(mes_safe).lower()}"
    )
    print(f"SMOKE_SELECTED_FAMILY {fit.selected.anchor.family}")
    print(f"SMOKE_ARTIFACT {SMOKE_ARTIFACT}")
    print(f"SMOKE_SHA256 {digest}")
    return 0


def _run_smoke_residual(workers: int) -> int:
    config = smoke_residual_search_config()
    print(
        "SMOKE_RESIDUAL_CONFIG "
        + json.dumps(
            {**asdict(config), "seeds": list(config.seeds)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    data = _smoke_training_data()
    anchor_fit = fit_static_anchor(
        data,
        smoke_anchor_search_config(),
        workers=1,
    )
    expected_static = _canonical_json_bytes(_fit_payload(anchor_fit))
    if not SMOKE_ARTIFACT.is_file():
        raise FileNotFoundError(f"static smoke artifact is missing: {SMOKE_ARTIFACT}")
    static_bytes = SMOKE_ARTIFACT.read_bytes()
    if static_bytes != expected_static:
        raise ValueError("static smoke artifact does not authenticate fitted anchor")
    static_digest = hashlib.sha256(static_bytes).hexdigest()
    scaler = fit_context_scaler(
        tuple(
            row.train_only[year]
            for row in data
            for year in sorted(row.train_only)
        )
    )
    fit = _fit_residual_fold_with_config(
        data,
        anchor_fit,
        scaler,
        workers,
        config,
    )
    payload = _residual_fit_payload(fit, static_digest)
    digest = _write_canonical_json(SMOKE_RESIDUAL_ARTIFACT, payload)
    primary_fit = next(row for row in fit.seeds if row.seed == fit.primary_seed)
    selected = primary_fit.selected
    print(f"SMOKE_RESIDUAL_PRIMARY_SEED {fit.primary_seed}")
    print(f"SMOKE_RESIDUAL_AMPLITUDE {selected.amplitude:.17g}")
    print(
        "SMOKE_RESIDUAL_WEIGHTS "
        + json.dumps(list(selected.weights), separators=(",", ":"), allow_nan=False)
    )
    print(f"SMOKE_RESIDUAL_LOSS {selected.loss:.17g}")
    print(f"SMOKE_RESIDUAL_PAYLOAD_SHA256 {payload['payload_sha256']}")
    print(
        "SMOKE_RESIDUAL_ACTUATED "
        + ("yes" if selected.amplitude > 0.0 else "no")
    )
    print(f"SMOKE_STATIC_ARTIFACT {SMOKE_ARTIFACT}")
    print(f"SMOKE_STATIC_SHA256 {static_digest}")
    print(f"SMOKE_RESIDUAL_ARTIFACT {SMOKE_RESIDUAL_ARTIFACT}")
    print(f"SMOKE_RESIDUAL_SHA256 {digest}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser(
        "smoke-anchor", help="run the tiny synthetic training-only anchor smoke"
    )
    smoke.add_argument("--workers", type=int, default=1)
    residual = commands.add_parser(
        "smoke-residual", help="run the synthetic training-only residual smoke"
    )
    residual.add_argument("--workers", type=int, default=1)
    development_smoke = commands.add_parser(
        "smoke-development",
        help="run the synthetic two-fold development smoke",
    )
    development_smoke.add_argument("--workers", type=int, default=1)
    prepare = commands.add_parser(
        "prepare-development",
        help="seal the measured Task 6 runtime and old-data input identities",
    )
    prepare.add_argument("--workers", type=int, required=True)
    development = commands.add_parser(
        "development",
        help="run the guarded, prepared eight-fold old-data development gate",
    )
    development.add_argument("--workers", type=int, required=True)
    aggregate = commands.add_parser(
        "aggregate-development",
        help="read-only verification of the sealed Task 6 outputs",
    )
    aggregate.add_argument("--verify-only", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "smoke-anchor":
        return _run_smoke_anchor(args.workers)
    if args.command == "smoke-residual":
        return _run_smoke_residual(args.workers)
    if args.command == "smoke-development":
        return _run_smoke_development(args.workers)
    try:
        if args.command == "prepare-development":
            _prepare_development(args.workers)
            print("DEVELOPMENT_PREPARED")
            return 0
        if args.command == "development":
            decision = run_development(args.workers)
            print(
                "DEVELOPMENT_CLASSIFICATION "
                + str(decision["classification"])
            )
            return 0
        if args.command == "aggregate-development":
            summary_sha, decision_sha = _verify_development_outputs()
            print(f"SUMMARY_SHA256 {summary_sha}")
            print(f"DECISION_SHA256 {decision_sha}")
            return 0
    except Exception:
        print("DEVELOPMENT_COMMAND_REJECTED", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
