"""Frozen corpus and split protocol for the three-city ICLR experiment.

This module is intentionally separate from the released Warsaw-only corpus.
It selects one non-overlapping local electorate per longitudinal series in
Warsaw, Gdynia, and Łódź, then constructs the prespecified 2022/2023 temporal
boundary and leave-one-city-out fitting views.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Dict, Mapping, Sequence, Tuple

from iclr_corpus import (
    CorpusConfig,
    SeriesRef,
    Split,
    build_series_index,
    load_series,
)
from iclr_outcome import PROJECT_FEATURES
from iclr_policy import FEATURE_NAMES

TRAIN_THROUGH = 2022
FIRST_TEST_YEAR = 2023
CANONICAL_CITIES: Tuple[str, ...] = (
    "Poland/Warszawa",
    "Poland/Gdynia",
    "Poland/Łódź",
)
PABULIB_COMMIT = "2f4321fec84069f50abf35fb5e90852013d17070"
LOCK_PROFILE = "multicity-priority-v1"
FIT_SOURCE_FILES: Tuple[str, ...] = (
    "iclr_multicity_train.py",
    "iclr_multicity_protocol.py",
    "iclr_priority_mes.py",
    "iclr_outcome.py",
    "iclr_policy.py",
    "iclr_env.py",
    "iclr_cmaes.py",
    "rules.py",
)
LOCKED_SOURCE_FILES: Tuple[str, ...] = (
    "iclr_multicity_protocol.py",
    "iclr_priority_mes.py",
    "iclr_multicity_train.py",
    "iclr_multicity_evaluate.py",
    "iclr_corpus.py",
    "iclr_env.py",
    "iclr_outcome.py",
    "iclr_policy.py",
    "iclr_cmaes.py",
    "iclr_train.py",
    "iclr_stats.py",
    "rules.py",
    "cohorts.py",
    "parse_pb.py",
    "run_experiments.py",
)
LOCKED_TEST_FILES: Tuple[str, ...] = (
    "test_iclr_multicity_protocol.py",
    "test_iclr_priority_mes.py",
    "test_iclr_multicity_train.py",
    "test_iclr_multicity_evaluate.py",
)
AMENDMENTS_RELATIVE_PATH = Path("results") / "iclr_lock_amendments.json"
MULTICITY_MANIFEST_RELATIVE_PATH = (
    Path("results") / "iclr_multicity" / "corpus_manifest.json"
).as_posix()
_LOCATION_REDACTION_KIND = "json-top-level-location-redaction-v1"
_LOCATION_FIELDS = ("destination", "source_dir")


@dataclass(frozen=True)
class MultiCityProtocol:
    """Hard corpus assertions from the approved pre-analysis design."""

    train_through: int = TRAIN_THROUGH
    first_test_year: int = FIRST_TEST_YEAR
    expected_series: int = 75
    expected_elections: int = 397
    expected_train_elections: int = 174
    expected_test_elections: int = 223


def _series_budget(ref: SeriesRef) -> float:
    """Total recorded budget, used only to choose one Gdynia ballot track."""

    return float(sum(instance.budget for instance in load_series(ref).values()))


def _gdynia_electorate_key(key: str) -> str:
    """Remove a Gdynia small/large suffix while preserving the place key."""

    head, separator, tail = key.rpartition("/")
    if not separator:
        return key
    return f"{head}/{tail.rsplit(' | ', 1)[0]}"


def _has_temporal_views(ref: SeriesRef) -> bool:
    return (
        any(year <= TRAIN_THROUGH for year in ref.years)
        and any(year >= FIRST_TEST_YEAR for year in ref.years)
    )


def _from_year(ref: SeriesRef, first_year: int) -> SeriesRef:
    """Return the same series restricted to its frozen lower year boundary."""

    kept = [
        (year, path)
        for year, path in zip(ref.years, ref.paths)
        if year >= first_year
    ]
    return SeriesRef(
        key=ref.key,
        years=tuple(year for year, _ in kept),
        paths=tuple(path for _, path in kept),
    )


def select_canonical_series(
    index: Mapping[str, SeriesRef],
) -> Tuple[SeriesRef, ...]:
    """Select non-overlapping local series in the three prespecified cities."""

    direct = []
    gdynia: Dict[str, list[SeriesRef]] = defaultdict(list)
    for _, ref in sorted(index.items()):
        if ref.city not in CANONICAL_CITIES or not _has_temporal_views(ref):
            continue
        tail = ref.key.rsplit("/", 1)[-1]
        if tail == "CITYWIDE":
            continue
        if ref.city == "Poland/Warszawa":
            if ref.key == "Poland/Warszawa/subunit Wawer":
                continue
            # Reuse the released Warsaw observation window exactly. Wawer is
            # the one district whose frozen primary series begins before 2019.
            direct.append(_from_year(ref, 2016 if tail == "Wawer" else 2019))
        elif ref.city == "Poland/Gdynia":
            if tail.casefold() == "green budget":
                continue
            gdynia[_gdynia_electorate_key(ref.key)].append(ref)
        elif ref.city == "Poland/Łódź":
            direct.append(ref)

    for refs in gdynia.values():
        # Track identity is present in the series key. Prefer the canonical
        # "large" ballot without reading any election budget, especially no
        # post-2022 budget, during fitting.
        direct.append(
            max(
                refs,
                key=lambda ref: (
                    ref.key.rsplit("/", 1)[-1].casefold().endswith(" | large"),
                    ref.key,
                ),
            )
        )
    return tuple(sorted(direct, key=lambda ref: ref.key))


def validate_canonical_counts(
    refs: Sequence[SeriesRef], protocol: MultiCityProtocol | None = None
) -> None:
    """Raise when any frozen corpus count differs from the approved design."""

    protocol = protocol or MultiCityProtocol()
    observed_series = len(refs)
    observed_elections = sum(len(ref.years) for ref in refs)
    observed_train = sum(
        sum(year <= protocol.train_through for year in ref.years) for ref in refs
    )
    observed_test = sum(
        sum(year >= protocol.first_test_year for year in ref.years) for ref in refs
    )
    checks = (
        ("series", observed_series, protocol.expected_series),
        ("elections", observed_elections, protocol.expected_elections),
        ("training elections", observed_train, protocol.expected_train_elections),
        ("held-out elections", observed_test, protocol.expected_test_elections),
    )
    for label, observed, expected in checks:
        if observed != expected:
            raise ValueError(f"expected {expected} {label}, found {observed}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def multicity_manifest_retained_sha256(payload: Mapping[str, object]) -> str:
    """Hash every manifest field except the two machine-location fields."""

    retained = {
        key: value for key, value in payload.items() if key not in _LOCATION_FIELDS
    }
    return _canonical_json_sha256(retained)


def _manifest_semantic_constraint(
    record: Mapping[str, object],
) -> Mapping[str, object]:
    constraint = record.get("semantic_constraint")
    expected_keys = {
        "kind",
        "excluded_top_level_fields",
        "retained_payload_sha256",
        "current_values",
    }
    if not isinstance(constraint, dict) or set(constraint) != expected_keys:
        raise RuntimeError("multicity manifest semantic constraint is malformed")
    current_values = constraint.get("current_values")
    if (
        constraint.get("kind") != _LOCATION_REDACTION_KIND
        or constraint.get("excluded_top_level_fields") != list(_LOCATION_FIELDS)
        or not _is_sha256(constraint.get("retained_payload_sha256"))
        or not isinstance(current_values, dict)
        or set(current_values) != set(_LOCATION_FIELDS)
        or not all(isinstance(value, str) for value in current_values.values())
    ):
        raise RuntimeError("multicity manifest semantic constraint is malformed")
    return constraint


def _load_protocol_amendments(
    root: Path,
) -> Dict[Tuple[str, str, str], Mapping[str, object]]:
    """Load the strict repository-local amendment inventory, if present."""

    path = Path(root) / AMENDMENTS_RELATIVE_PATH
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed amendment inventory: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"malformed amendment inventory: {path}")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise RuntimeError(f"malformed amendment inventory: {path}")
    entries = payload.get("amendments")
    if not isinstance(entries, list):
        raise RuntimeError(f"malformed amendment inventory: {path}")

    indexed: Dict[Tuple[str, str, str], Mapping[str, object]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"malformed amendment record {position}: {path}")
        relative = entry.get("path")
        locked = entry.get("locked_sha256")
        amended = entry.get("amended_sha256")
        affects = entry.get("affects_recorded_decision")
        if not isinstance(relative, str):
            raise RuntimeError(f"malformed amendment record {position}: {path}")
        normalized = PurePosixPath(relative)
        if (
            normalized.is_absolute()
            or ".." in normalized.parts
            or normalized.as_posix() != relative
            or not _is_sha256(locked)
            or not _is_sha256(amended)
            or not isinstance(affects, bool)
        ):
            raise RuntimeError(f"malformed amendment record {position}: {path}")
        key = (relative, locked, amended)
        if key in indexed:
            raise RuntimeError(f"duplicate amendment record {position}: {path}")
        if relative == MULTICITY_MANIFEST_RELATIVE_PATH:
            _manifest_semantic_constraint(entry)
        indexed[key] = entry
    return indexed


def _verify_manifest_semantic_constraint(
    path: Path,
    record: Mapping[str, object],
) -> None:
    """Verify that a release manifest changed only machine-location fields."""

    constraint = _manifest_semantic_constraint(record)
    current_values = constraint.get("current_values")
    if not isinstance(current_values, dict):  # guarded by the schema check above
        raise RuntimeError("multicity manifest semantic constraint is malformed")
    try:
        current = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "multicity manifest semantic constraint cannot be verified"
        ) from exc
    if not isinstance(current, dict):
        raise RuntimeError("multicity manifest semantic constraint cannot be verified")
    if any(current.get(field) != current_values[field] for field in _LOCATION_FIELDS):
        raise RuntimeError(
            "multicity manifest semantic constraint current values differ"
        )
    if (
        multicity_manifest_retained_sha256(current)
        != constraint["retained_payload_sha256"]
    ):
        raise RuntimeError("multicity manifest semantic constraint digest differs")


def write_json_artifact(path: Path, payload: Mapping[str, object]) -> str:
    """Write deterministic UTF-8 JSON and return its SHA-256 digest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return _sha256(path)


def write_immutable_text_artifact(path: Path, content: str) -> str:
    """Write text once; permit only byte-identical reruns."""

    path = Path(path)
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        return _sha256(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return _sha256(path)


def write_immutable_json_artifact(
    path: Path, payload: Mapping[str, object]
) -> str:
    """Write canonical JSON once; permit only byte-identical reruns."""

    content = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return write_immutable_text_artifact(path, content)


def load_canonical_index_from_manifest(
    manifest_path: Path,
    data_dir: Path,
    *,
    enforce_expected_counts: bool = True,
) -> Dict[str, SeriesRef]:
    """Verify every frozen file and reconstruct the index without parsing data."""

    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files", [])
    if len(rows) != int(manifest.get("n_files", -1)):
        raise RuntimeError("corpus manifest file count is internally inconsistent")
    if enforce_expected_counts and manifest.get("source_commit") != PABULIB_COMMIT:
        raise RuntimeError(
            f"unexpected Pabulib commit {manifest.get('source_commit')!r}"
        )
    listed_names = [str(row["name"]) for row in rows]
    if len(listed_names) != len(set(listed_names)):
        raise RuntimeError("corpus manifest contains duplicate file names")
    actual_names = sorted(path.name for path in data_dir.glob("*.pb"))
    if sorted(listed_names) != actual_names:
        raise RuntimeError("staged corpus files differ from the corpus manifest")

    by_series: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for row in rows:
        name = str(row["name"])
        path = data_dir / name
        if path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"byte-size mismatch for corpus file {name}")
        observed_hash = _sha256(path)
        if observed_hash != row["sha256"]:
            raise RuntimeError(
                f"hash mismatch for corpus file {name}: "
                f"expected {row['sha256']}, found {observed_hash}"
            )
        series = str(row["series"])
        year = int(row["year"])
        if year in by_series[series]:
            raise RuntimeError(f"duplicate corpus election {series} {year}")
        by_series[series][year] = path
    index = {
        series: SeriesRef(
            key=series,
            years=tuple(sorted(years)),
            paths=tuple(years[year] for year in sorted(years)),
        )
        for series, years in sorted(by_series.items())
    }
    if enforce_expected_counts:
        validate_canonical_counts(tuple(index.values()))
    if int(manifest.get("n_series", -1)) != len(index):
        raise RuntimeError("corpus manifest series count is internally inconsistent")
    if int(manifest.get("n_elections", -1)) != sum(
        len(ref.years) for ref in index.values()
    ):
        raise RuntimeError("corpus manifest election count is internally inconsistent")
    return index


def stage_canonical_corpus(
    source_dir: Path,
    destination: Path,
    *,
    source_commit: str = PABULIB_COMMIT,
    enforce_expected_counts: bool = True,
) -> Dict[str, object]:
    """Copy the canonical corpus and return a hash-complete manifest.

    Existing byte-identical files are retained. A name collision with different
    bytes is a hard failure so an old or partial corpus cannot be overwritten
    silently.
    """

    source_dir = Path(source_dir).resolve()
    destination = Path(destination).resolve()
    index = build_series_index(CorpusConfig(data_dir=source_dir))
    refs = select_canonical_series(index)
    if enforce_expected_counts:
        validate_canonical_counts(refs)
    destination.mkdir(parents=True, exist_ok=True)

    selected_by_name: Dict[str, Tuple[Path, str, int]] = {}
    for ref in refs:
        for year, source in zip(ref.years, ref.paths):
            source = Path(source).resolve()
            prior = selected_by_name.get(source.name)
            if prior is not None and _sha256(prior[0]) != _sha256(source):
                raise RuntimeError(
                    f"source filename collision with different hashes: {source.name}"
                )
            selected_by_name[source.name] = (source, ref.key, year)

    expected_names = set(selected_by_name)
    unexpected = sorted(
        path.name for path in destination.glob("*.pb") if path.name not in expected_names
    )
    if unexpected:
        raise RuntimeError(
            "destination contains non-canonical PB files: " + ", ".join(unexpected[:5])
        )

    rows = []
    for name, (source, series, year) in sorted(selected_by_name.items()):
        target = destination / name
        source_hash = _sha256(source)
        if target.exists():
            target_hash = _sha256(target)
            if target_hash != source_hash:
                raise RuntimeError(f"hash conflict for existing staged file: {target}")
        else:
            shutil.copy2(source, target)
        try:
            source_relative = str(source.relative_to(source_dir))
        except ValueError:
            source_relative = source.name
        rows.append(
            {
                "name": name,
                "source_relative": source_relative,
                "series": series,
                "year": year,
                "bytes": target.stat().st_size,
                "sha256": source_hash,
            }
        )

    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "source_dir": str(source_dir),
        "destination": str(destination),
        "n_series": len(refs),
        "n_elections": sum(len(ref.years) for ref in refs),
        "n_files": len(rows),
        "files": rows,
    }


def build_multicity_splits(index: Mapping[str, SeriesRef]) -> Dict[str, Split]:
    """Construct the temporal split and three leave-one-city-out fit views."""

    train = tuple(
        (key, tuple(year for year in ref.years if year <= TRAIN_THROUGH))
        for key, ref in sorted(index.items())
    )
    test = tuple(
        (key, tuple(year for year in ref.years if year >= FIRST_TEST_YEAR))
        for key, ref in sorted(index.items())
    )
    splits: Dict[str, Split] = {
        "temporal_2022": Split(
            name="temporal_2022",
            train=train,
            test=test,
            note="fit <=2022 in all cities; score >=2023 after same-series warm-up",
        )
    }
    for held_out_city in CANONICAL_CITIES:
        name = f"city_out_{held_out_city.replace('/', '_')}"
        fit = tuple(
            (key, tuple(year for year in ref.years if year <= TRAIN_THROUGH))
            for key, ref in sorted(index.items())
            if ref.city != held_out_city
        )
        score = tuple(
            (key, tuple(year for year in ref.years if year >= FIRST_TEST_YEAR))
            for key, ref in sorted(index.items())
            if ref.city == held_out_city
        )
        splits[name] = Split(
            name=name,
            train=fit,
            test=score,
            note=(
                f"fit other cities <=2022; warm {held_out_city} on its own <=2022; "
                f"score {held_out_city} >=2023"
            ),
        )
    return splits


def validate_split_integrity(
    splits: Mapping[str, Split], index: Mapping[str, SeriesRef]
) -> Dict[str, object]:
    """Validate non-overlap, membership, and warm-up availability."""

    overlap_count = 0
    n_fit = 0
    n_score = 0
    for name, split in splits.items():
        fit_ids = {(key, year) for key, years in split.train for year in years}
        score_ids = {(key, year) for key, years in split.test for year in years}
        overlap_count += len(fit_ids & score_ids)
        n_fit += len(fit_ids)
        n_score += len(score_ids)
        for key, year in fit_ids | score_ids:
            if key not in index or year not in index[key].years:
                raise ValueError(f"{name}: unknown election {key} {year}")
        for key, years in split.test:
            if not years:
                raise ValueError(f"{name}: empty scored suffix for {key}")
            if not any(year <= TRAIN_THROUGH for year in index[key].years):
                raise ValueError(f"{name}: no warm-up prefix for {key}")
    if overlap_count:
        raise ValueError(f"train/test election overlap: {overlap_count}")
    return {
        "n_splits": len(splits),
        "fit_election_entries": n_fit,
        "score_election_entries": n_score,
        "train_test_overlap": overlap_count,
    }


def split_payload(split: Split, index: Mapping[str, SeriesRef]) -> Dict[str, object]:
    """Serialize fit, held-city warm-up, and scored views without ambiguity."""

    fit = {key: list(years) for key, years in split.train}
    score = {key: list(years) for key, years in split.test}
    warmup = {
        key: [year for year in index[key].years if year <= TRAIN_THROUGH]
        for key, _ in split.test
    }
    return {
        "schema_version": 1,
        "name": split.name,
        "note": split.note,
        "train_through": TRAIN_THROUGH,
        "first_test_year": FIRST_TEST_YEAR,
        "fit": fit,
        "warmup": warmup,
        "score": score,
        "counts": {
            "fit_series": len(fit),
            "fit_elections": sum(len(years) for years in fit.values()),
            "warmup_series": len(warmup),
            "warmup_elections": sum(len(years) for years in warmup.values()),
            "score_series": len(score),
            "score_elections": sum(len(years) for years in score.values()),
        },
    }


def load_frozen_split(path: Path, index: Mapping[str, SeriesRef]) -> Split:
    """Deserialize one split and require byte-content semantics to match protocol."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = str(payload.get("name", ""))
    generated = build_multicity_splits(index)
    if name not in generated:
        raise RuntimeError(f"unknown frozen split name {name!r}")
    expected_payload = split_payload(generated[name], index)
    if payload != expected_payload:
        raise RuntimeError(f"frozen split differs from generated protocol: {name}")
    return generated[name]


def freeze_multicity_splits(
    index: Mapping[str, SeriesRef], destination: Path
) -> Dict[str, str]:
    """Write all four prespecified splits and return their content hashes."""

    splits = build_multicity_splits(index)
    validate_split_integrity(splits, index)
    destination = Path(destination)
    hashes = {}
    for name, split in sorted(splits.items()):
        hashes[name] = write_json_artifact(
            destination / f"{name}.json", split_payload(split, index)
        )
    return hashes


def locked_fit_inventory() -> list[Dict[str, object]]:
    """Canonical 24-run fit matrix shared by lock, trainer, and evaluator."""

    inventory: list[Dict[str, object]] = []
    for arm in ("priority", "endowment", "outcome"):
        for seed in (1, 2, 42):
            inventory.append(
                {"split": "temporal_2022", "arm": arm, "seed": seed}
            )
    for city in ("Poland_Warszawa", "Poland_Gdynia", "Poland_Łódź"):
        split = f"city_out_{city}"
        for seed in (1, 2, 42):
            inventory.append({"split": split, "arm": "priority", "seed": seed})
        for arm in ("endowment", "outcome"):
            inventory.append({"split": split, "arm": arm, "seed": 42})
    return inventory


def evidence_gate_config() -> Dict[str, object]:
    """Single source of truth for gold, silver, and falsification thresholds."""

    return {
        "gold": {
            "exact_mes_containment": True,
            "negative_temporal_difference_each_city": True,
            "city_macro_difference_max": -0.010,
            "negative_city_out_difference_each_city": True,
            "welfare_ratio_min_each_city": 0.98,
            "exclusion_increase_max_each_city": 0.005,
            "negative_direction_all_seeds_each_city": True,
            "city_macro_seed_spread_max": 0.005,
            "priority_actuation_greater_than_endowment": True,
        },
        "silver": {
            "city_macro_difference_min_magnitude": 0.005,
            "city_macro_difference_max_magnitude": 0.010,
            "maximum_null_city_out_folds": 1,
        },
        "falsification": {
            "city_macro_gain_below": 0.005,
            "city_specific_only": True,
            "welfare_or_exclusion_failure": True,
            "seed_direction_reversal": True,
        },
    }


def protocol_config() -> Dict[str, object]:
    """Single source of truth for all non-result experimental semantics."""

    return {
        "cities": list(CANONICAL_CITIES),
        "train_through": TRAIN_THROUGH,
        "first_test_year": FIRST_TEST_YEAR,
        "seeds": [1, 2, 42],
        "primary_seed": 42,
        "generations": 30,
        "coefficient_bound": 10.0,
        "sigma0": 0.4,
        "execution_mode": "serial",
        "completion": "approval_count",
        "objective": {
            "fairness": "equal-weight mean of city means of series worst-group CSD",
            "welfare_penalty": 2.0,
            "welfare_floor_ratio": 1.0,
            "welfare_aggregation": "mean of city-specific hinges",
        },
        "arms": {
            "priority": {
                "feature_names": list(PROJECT_FEATURES),
                "initial": "MES (all zeros)",
            },
            "endowment": {
                "feature_names": list(FEATURE_NAMES),
                "initial": "RES(1.0)",
            },
            "outcome": {
                "feature_names": list(PROJECT_FEATURES),
                "initial": "greedy approvals per cost",
            },
        },
        "bootstrap": {"draws": 10000, "seed": 20260812},
        "sign_flip": {"draws": 200000, "seed": 20260812},
    }


def build_protocol_lock(
    repo_root: Path, tracked_files: Mapping[str, Path]
) -> Dict[str, object]:
    """Build the hash-addressed pre-analysis lock without opening score views."""

    repo_root = Path(repo_root).resolve()
    tracked = {}
    for label, path in sorted(tracked_files.items()):
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"tracked protocol file not found: {resolved}")
        try:
            relative = resolved.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"tracked file lies outside repository root: {resolved}") from exc
        tracked[label] = {
            "path": str(relative),
            "sha256": _sha256(resolved),
        }
    return {
        "schema_version": 1,
        "tracked_files": tracked,
        "fit_inventory": locked_fit_inventory(),
        "protocol": protocol_config(),
        "evidence_gates": evidence_gate_config(),
    }


def verify_protocol_lock(
    lock_path: Path, repo_root: Path | None = None
) -> Dict[str, object]:
    """Load a lock and permit only exact, non-decision amendment triples."""

    lock_path = Path(lock_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else lock_path.parent
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    tracked = payload.get("tracked_files")
    if not isinstance(tracked, dict):
        raise RuntimeError(
            f"protocol lock tracked-file inventory is malformed: {lock_path}"
        )
    amendments = _load_protocol_amendments(root)
    for label, row in tracked.items():
        if (
            not isinstance(label, str)
            or not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not _is_sha256(row.get("sha256"))
        ):
            raise RuntimeError(f"malformed tracked file row for {label!r}")
        relative = row["path"]
        raw_relative = PurePosixPath(relative)
        if (
            raw_relative.is_absolute()
            or ".." in raw_relative.parts
            or raw_relative.as_posix() != relative
        ):
            raise RuntimeError(
                f"malformed tracked file path for {label}: {relative!r}"
            )
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"tracked file escapes repository for {label}: {path}"
            ) from exc
        if not path.is_file():
            raise RuntimeError(f"tracked file missing for {label}: {path}")
        observed = _sha256(path)
        if observed != row["sha256"]:
            record = amendments.get((relative, row["sha256"], observed))
            if record is None or record.get("affects_recorded_decision") is not False:
                raise RuntimeError(
                    f"hash mismatch for tracked file {label}: "
                    f"expected {row['sha256']}, found {observed}"
                )
            if relative == MULTICITY_MANIFEST_RELATIVE_PATH:
                _verify_manifest_semantic_constraint(path, record)
    return payload


def validate_mandatory_lock_profile(
    payload: Mapping[str, object],
    repo_root: Path,
    expected_files: Mapping[str, Path],
) -> None:
    """Require the exact centrally defined label/path set, with no omissions."""

    if payload.get("lock_profile") != LOCK_PROFILE:
        raise RuntimeError(
            f"invalid protocol lock profile {payload.get('lock_profile')!r}"
        )
    if payload.get("schema_version") != 1:
        raise RuntimeError("protocol lock schema semantics differ from canonical version")
    if payload.get("protocol") != protocol_config():
        raise RuntimeError("protocol semantics differ from canonical definition")
    if payload.get("evidence_gates") != evidence_gate_config():
        raise RuntimeError("evidence-gate semantics differ from canonical definition")
    tracked = payload.get("tracked_files", {})
    if set(tracked) != set(expected_files):
        missing = sorted(set(expected_files) - set(tracked))
        extra = sorted(set(tracked) - set(expected_files))
        raise RuntimeError(
            f"mandatory tracked-file set differs: missing={missing}, extra={extra}"
        )
    root = Path(repo_root).resolve()
    for label, expected_path in expected_files.items():
        relative = str(Path(expected_path).resolve().relative_to(root))
        if tracked[label].get("path") != relative:
            raise RuntimeError(
                f"mandatory tracked-file path differs for {label}: "
                f"{tracked[label].get('path')!r} != {relative!r}"
            )
    if payload.get("fit_inventory") != locked_fit_inventory():
        raise RuntimeError("protocol lock fit inventory is not canonical")


def mandatory_protocol_files(
    repo_root: Path, result_root: Path, data_dir: Path
) -> Dict[str, Path]:
    """Centrally enumerate every file that a valid multi-city lock must bind."""

    repo_root = Path(repo_root).resolve()
    result_root = Path(result_root).resolve()
    data_dir = Path(data_dir).resolve()
    manifest_path = result_root / "corpus_manifest.json"
    index = load_canonical_index_from_manifest(manifest_path, data_dir)
    files: Dict[str, Path] = {
        "artifact/corpus_manifest": manifest_path,
        "artifact/structural_gates": result_root / "structural_gates.json",
        "environment/pyproject": repo_root / "pyproject.toml",
        "environment/uv_lock": repo_root / "uv.lock",
    }
    for name in (
        "temporal_2022",
        "city_out_Poland_Warszawa",
        "city_out_Poland_Gdynia",
        "city_out_Poland_Łódź",
    ):
        files[f"split/{name}"] = result_root / "splits" / f"{name}.json"
    for name in LOCKED_SOURCE_FILES:
        files[f"source/{name}"] = repo_root / "src" / name
    for name in LOCKED_TEST_FILES:
        files[f"test/{name}"] = repo_root / "tests" / name
    for ref in index.values():
        for path in ref.paths:
            files[f"corpus/{path.name}"] = path
    for label, path in files.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"mandatory protocol file missing for {label}: {path}")
    verify_structural_gates(files["artifact/structural_gates"])
    return files


def build_multicity_protocol_lock(
    repo_root: Path, result_root: Path, data_dir: Path
) -> Dict[str, object]:
    """Build the only lock profile accepted by trainer and evaluator."""

    files = mandatory_protocol_files(repo_root, result_root, data_dir)
    payload = build_protocol_lock(repo_root, files)
    payload["lock_profile"] = LOCK_PROFILE
    payload["mandatory_file_count"] = len(files)
    return payload


def verify_multicity_protocol_lock(
    lock_path: Path, repo_root: Path, result_root: Path, data_dir: Path
) -> Dict[str, object]:
    """Verify hashes plus the exact mandatory multi-city lock profile."""

    payload = verify_protocol_lock(lock_path, repo_root)
    expected = mandatory_protocol_files(repo_root, result_root, data_dir)
    validate_mandatory_lock_profile(payload, repo_root, expected)
    if payload.get("mandatory_file_count") != len(expected):
        raise RuntimeError("protocol lock mandatory-file count differs")
    return payload


def verify_structural_gates(path: Path) -> Dict[str, object]:
    """Require the complete corpus-wide pre-evaluation structural report."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "status": "pass",
        "n_series": 75,
        "n_elections": 397,
        "identity_checks": 794,
        "determinism_checks": 794,
        "budget_checks": 794,
        "completion_modes": [False, True],
        "zero_weights": [0.0] * 5,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"structural gate field {field!r} is {payload.get(field)!r}, "
                f"expected {value!r}"
            )
    payment_rounds = payload.get("payment_rounds_checked")
    if not isinstance(payment_rounds, int) or payment_rounds <= 0:
        raise RuntimeError("structural gate field 'payment_rounds_checked' must be positive")
    return payload


def open_heldout_once(
    lock_path: Path,
    receipt_path: Path,
    required_fit_inventory: Sequence[Tuple[Mapping[str, object], Path]],
    structural_gate_path: Path,
    repo_root: Path | None = None,
) -> Dict[str, object]:
    """Create the one-way receipt after lock and fit inventory verification."""

    lock_path = Path(lock_path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else lock_path.parent
    lock_payload = verify_protocol_lock(lock_path, root)
    verify_structural_gates(structural_gate_path)

    def canonical(rows):
        return sorted(
            (str(row["split"]), str(row["arm"]), int(row["seed"]))
            for row in rows
        )

    locked_specs = canonical(lock_payload.get("fit_inventory", []))
    supplied_specs = canonical(spec for spec, _ in required_fit_inventory)
    if locked_specs != supplied_specs:
        raise RuntimeError(
            f"supplied fit inventory differs from protocol lock: "
            f"locked={locked_specs}, supplied={supplied_specs}"
        )
    fit_hashes: Dict[str, str] = {}
    resolved_inventory = sorted(
        ((spec, Path(path).resolve()) for spec, path in required_fit_inventory),
        key=lambda item: str(item[1]),
    )
    if len({path for _, path in resolved_inventory}) != len(resolved_inventory):
        raise RuntimeError("required fit paths are not one-to-one")
    for spec, path in resolved_inventory:
        expected_tail = Path(
            "fits",
            str(spec["split"]),
            str(spec["arm"]),
            f"seed-{int(spec['seed'])}.json",
        )
        if tuple(path.parts[-4:]) != tuple(expected_tail.parts):
            raise RuntimeError(
                f"fit path does not match specification {spec}: {path}"
            )
        if not path.is_file():
            raise RuntimeError(f"missing required fit: {path}")
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        fit_hashes[label] = _sha256(path)
    expected = {
        "schema_version": 1,
        "lock_sha256": _sha256(lock_path),
        "fit_sha256": fit_hashes,
        "structural_gates_sha256": _sha256(Path(structural_gate_path)),
    }
    receipt_path = Path(receipt_path)
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise RuntimeError("held-out receipt exists for a different lock or fit inventory")
        return existing
    receipt = {
        **expected,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_artifact(receipt_path, receipt)
    return receipt
