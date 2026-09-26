"""Protocol and immutable staging CLI for the primary-Warsaw v2 replay.

This module can stage the frozen inputs, run the structural gate, and install or
verify the base lock. Held-out raw files remain closed until an authenticated
evaluation-replay receipt is supplied to the post-receipt APIs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from iclr_approval_semantics_v2 import (
    SEMANTICS_PROFILE,
    canonical_instance_sha256,
    canonicalize_instance,
    load_series_authenticated_v2,
)
from iclr_corpus import SeriesRef, Split
from iclr_env import RolloutState
from iclr_multicity_protocol import PABULIB_COMMIT, multicity_manifest_retained_sha256
from iclr_multicity_protocol_v2 import (
    STRUCTURAL_SOURCE_FILES_V2,
    preflight_immutable_bytes_artifact_v2,
    read_regular_bytes_artifact_v2,
    write_immutable_bytes_artifact_v2,
)
from iclr_outcome import N_PROJECT_FEATURES
from iclr_priority_mes import priority_mes_outcome
from parse_pb import parse_pb_file
from rules import mes_with_endowments


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT_V2 = ROOT / "results" / "iclr_primary_warsaw_v2"
LOCK_PROFILE_V2 = "primary-warsaw-corrected-post-hoc-v2"
EVALUATION_REPLAY_RECEIPT_NAME_V2 = "evaluation_replay_started.json"
TRAINING_ACCESS_SCOPE_V2 = "training_only"
EVALUATION_ACCESS_SCOPE_V2 = "evaluation_authorized"
PREFIT_STRUCTURAL_GATE_SCOPE_V2 = (
    "pre-fit metadata and approval-semantics authentication only"
)
POSTRECEIPT_STRUCTURAL_GATE_SCOPE_V2 = "post-receipt full-corpus replay"
EVALUATION_REPLAY_DISCLOSURE_V2 = (
    "Warsaw held-out outcomes were already known before this corrected "
    "post-hoc replay; this receipt is not a fresh holdout or preregistration."
)

PARENT_FROZEN_LIST_SHA256 = (
    "5fae438226ede5b3e365c5ee5d161231e0db26adab7e7d470cce3a6e491f3d05"
)
EXPECTED_MULTICITY_MANIFEST_RETAINED_SHA256 = (
    "2c2d727c42a78c57cf2de890d0816bf5267bb31b0066e070164866eb5cdf427a"
)
EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256 = (
    "5e38dd79b07b6b7dc233f3fd14e6fadc1840b3e2e7825d289b0935ddf179a4fa"
)
EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256 = (
    "368dc78da2f998b53c07358a781777effd24fae4a813333b4dd7f8d54b342bd6"
)

SPLIT_SHA256_V2 = {
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

STATIC_AGE_GRID_VALUES = tuple(index / 20.0 for index in range(61))

PRIMARY_STRUCTURAL_SOURCE_FILES_V2 = (
    *STRUCTURAL_SOURCE_FILES_V2,
    "iclr_primary_warsaw_protocol_v2.py",
)

TRAINING_GRID_SPEC_V2 = {
    "artifact_label": "artifact/static_senior_alpha_grid",
    "relative_path": "training_grid/static_senior_alpha.json",
    "split": "temporal_2022",
    "training_only": True,
    "alphas": list(STATIC_AGE_GRID_VALUES),
    "selection_metric": "training worst-cohort CSD",
    "tie_break": "smaller alpha",
    "created_after_lock": True,
}

_DIRECT_LODZ_ANCHORS = {
    "Poland_Lodz_2023.pb": (
        "Poland/Łódź/CITYWIDE",
        2023,
        "c8a56f93aff01bc00de63b67b41e4499e34dbe167c0389e25870daa7a2a555f7",
    ),
    "Poland_Lodz_2024.pb": (
        "Poland/Łódź/CITYWIDE",
        2024,
        "17a17f11ca3f0f7a91175d89fb49cc009a9538e874954b528060bd22611c4be6",
    ),
    "Poland_Lodz_2025.pb": (
        "Poland/Łódź/CITYWIDE",
        2025,
        "5b64223e423298572e144f761d78a766caa2c9a87a76fc0af87edae8829b8641",
    ),
}

_EXPECTED_COUNTS = {
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

_EXPECTED_WOLA_ANOMALY = {
    "series": "Poland/Warszawa/Wola",
    "year": 2021,
    "source_id": "Poland_Warszawa_2022_Wola.pb",
    "vote_index_zero_based": 2976,
    "voter_id": "37051",
    "original_projects": ["147", "1040", "1040"],
    "canonical_projects": ["147", "1040"],
    "duplicate_projects": ["1040"],
}

_EXPECTED_WOLA_FILE_ROW = {
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

_INVENTORY_FIELDS = {
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


@dataclass(frozen=True)
class PrimaryProtocolLockSnapshotV2:
    """One protocol lock parsed from the exact authenticated byte image."""

    path: Path
    content: bytes
    sha256: str
    payload: dict[str, object]


@dataclass(frozen=True)
class JsonArtifactSnapshotV2:
    """One strict JSON artifact parsed from one authenticated byte image."""

    label: str
    path: Path
    content: bytes
    sha256: str
    payload: dict[str, object]


@dataclass(frozen=True)
class RawArtifactIdentityV2:
    """Authenticated identity of one raw election file."""

    series: str
    year: int
    name: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class PrimaryInputsSnapshotV2:
    """Authenticated primary manifest, receipt, and training split views."""

    result_root: Path
    data_dir: Path
    protocol_lock_path: Path
    protocol_lock_content: bytes
    protocol_lock_sha256: str
    manifest: JsonArtifactSnapshotV2
    semantics_receipt: JsonArtifactSnapshotV2
    split_artifacts: dict[str, JsonArtifactSnapshotV2]
    raw_artifacts: dict[tuple[str, int, str], RawArtifactIdentityV2]
    index: dict[str, SeriesRef]
    splits: dict[str, Split]
    access_scope: str = TRAINING_ACCESS_SCOPE_V2
    evaluation_replay_receipt: JsonArtifactSnapshotV2 | None = None


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def exact_json_equal_v2(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's bool/int/float aliases."""

    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(
            exact_json_equal_v2(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            exact_json_equal_v2(observed, wanted)
            for observed, wanted in zip(actual, expected)
        )
    return actual == expected


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise RuntimeError(f"JSON number must be finite: {value}")


def _require_finite_json(value: object, *, label: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{label} must contain only finite JSON numbers")
        return
    if type(value) is list:
        for child in value:
            _require_finite_json(child, label=label)
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise RuntimeError(f"{label} contains a non-string JSON key")
            _require_finite_json(child, label=label)
        return
    raise RuntimeError(f"{label} contains a non-JSON value")


def load_json_object_bytes_strict_v2(
    content: bytes,
    *,
    label: str,
) -> dict[str, object]:
    """Parse one finite JSON object while rejecting duplicate keys."""

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except RuntimeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"malformed {label}") from exc
    if type(payload) is not dict:
        raise RuntimeError(f"{label} must be a JSON object")
    _require_finite_json(payload, label=label)
    return payload


def load_json_object_strict_v2(path: Path, *, label: str) -> dict[str, object]:
    """Read and strictly parse one regular, no-follow JSON artifact."""

    content = read_regular_bytes_artifact_v2(Path(path), label=label)
    assert content is not None
    return load_json_object_bytes_strict_v2(content, label=label)


def _target_label(value: float) -> str:
    return format(value, ".12g")


def _fit_row(
    *,
    fit_id: str,
    family: str,
    split: str,
    arm: str,
    seed: int,
    generations: int,
    bound: float,
    init: str,
    soft_welfare_target: float = 0.0,
    welfare_penalty: float = 0.0,
    masked_feature_names: Sequence[str] = (),
    support_floor_kappa: float | None = None,
    grid_dependency: str | None = None,
    wola_2021_in_training: bool = True,
) -> dict[str, object]:
    return {
        "fit_id": fit_id,
        "family": family,
        "split": split,
        "arm": arm,
        "seed": seed,
        "generations": generations,
        "popsize": None,
        "sigma0": 0.4,
        "bound": bound,
        "init": init,
        "soft_welfare_target": soft_welfare_target,
        "welfare_penalty": welfare_penalty,
        "masked_feature_names": list(masked_feature_names),
        "support_floor_kappa": support_floor_kappa,
        "grid_dependency": grid_dependency,
        "wola_2021_in_training": wola_2021_in_training,
        "relative_path": f"fits/{fit_id}.json",
    }


def _build_locked_fit_inventory_v2() -> list[dict[str, object]]:
    """Construct the frozen matrix without recursively invoking validation."""

    rows: list[dict[str, object]] = []
    for seed in (1, 2, 3, 42):
        for target in (0.0, 0.85, 0.95, 1.0, 1.02, 1.05):
            label = _target_label(target)
            rows.append(
                _fit_row(
                    fit_id=(
                        "outcome_frontier/temporal_2022/"
                        f"target-{label}/seed-{seed}"
                    ),
                    family="outcome_frontier",
                    split="temporal_2022",
                    arm="outcome",
                    seed=seed,
                    generations=25,
                    bound=10.0,
                    init="res",
                    soft_welfare_target=target,
                    welfare_penalty=0.0 if target == 0.0 else 2.0,
                )
            )
    for target in (0.0, 0.99, 1.0):
        label = _target_label(target)
        rows.append(
            _fit_row(
                fit_id=(
                    "endowment_frontier/temporal_2022/"
                    f"target-{label}/seed-42"
                ),
                family="endowment_frontier",
                split="temporal_2022",
                arm="endowment",
                seed=42,
                generations=25,
                bound=10.0,
                init="res",
                soft_welfare_target=target,
                welfare_penalty=0.0 if target == 0.0 else 2.0,
            )
        )
    for seed, generations in ((1, 25), (2, 25), (42, 30)):
        rows.append(
            _fit_row(
                fit_id=f"temporal_endowment/temporal_2022/seed-{seed}",
                family="temporal_endowment",
                split="temporal_2022",
                arm="endowment",
                seed=seed,
                generations=generations,
                bound=10.0,
                init="res",
            )
        )
    for fold in range(5):
        split = f"district_out_f{fold}of5"
        for bound in (10.0, 40.0):
            bound_label = _target_label(bound)
            rows.append(
                _fit_row(
                    fit_id=(
                        f"district_endowment/{split}/bound-{bound_label}/seed-42"
                    ),
                    family="district_endowment",
                    split=split,
                    arm="endowment",
                    seed=42,
                    generations=25,
                    bound=bound,
                    init="res",
                    wola_2021_in_training=fold != 4,
                )
            )
    for feature_name in (
        "approval_share",
        "approvals_per_cost",
        "cost_share",
        "deficit_weighted",
        "cohort_concentration",
    ):
        rows.append(
            _fit_row(
                fit_id=(
                    "outcome_loo/temporal_2022/"
                    f"mask-{feature_name}/seed-42"
                ),
                family="outcome_loo",
                split="temporal_2022",
                arm="outcome",
                seed=42,
                generations=25,
                bound=10.0,
                init="res",
                soft_welfare_target=1.0,
                welfare_penalty=2.0,
                masked_feature_names=(feature_name,),
            )
        )
    for kappa in (1.0, 2.0):
        label = _target_label(kappa)
        rows.append(
            _fit_row(
                fit_id=(
                    "static_support_floor/temporal_2022/"
                    f"kappa-{label}/seed-42"
                ),
                family="static_support_floor",
                split="temporal_2022",
                arm="outcome",
                seed=42,
                generations=25,
                bound=10.0,
                init="res",
                support_floor_kappa=kappa,
            )
        )
    rows.append(
        _fit_row(
            fit_id="history_free/temporal_2022/seed-42",
            family="history_free",
            split="temporal_2022",
            arm="endowment",
            seed=42,
            generations=30,
            bound=10.0,
            init="zeros",
            masked_feature_names=("deficit_per_capita", "deficit_normalized"),
        )
    )
    rows.append(
        _fit_row(
            fit_id="static_age_lookup/temporal_2022/seed-42",
            family="static_age_lookup",
            split="temporal_2022",
            arm="static_age_lookup",
            seed=42,
            generations=30,
            bound=3.0,
            init="training_grid",
            grid_dependency="training_grid/static_senior_alpha.json",
        )
    )
    rows.sort(key=lambda row: str(row["fit_id"]))
    return rows


def locked_fit_inventory_v2() -> list[dict[str, object]]:
    """Return the exact validated 49-coordinate primary-Warsaw matrix."""

    rows = _build_locked_fit_inventory_v2()
    validate_locked_fit_inventory_v2(rows)
    return rows


def validate_locked_fit_inventory_v2(
    inventory: object,
) -> list[dict[str, object]]:
    """Require exact JSON types and exact equality with the frozen matrix."""

    if type(inventory) is not list or len(inventory) != 49:
        raise RuntimeError("fit inventory differs from the 49-coordinate contract")
    for row in inventory:
        if type(row) is not dict or set(row) != _INVENTORY_FIELDS:
            raise RuntimeError("fit inventory row schema differs")
        type_contract = {
            "fit_id": str,
            "family": str,
            "split": str,
            "arm": str,
            "seed": int,
            "generations": int,
            "sigma0": float,
            "bound": float,
            "init": str,
            "soft_welfare_target": float,
            "welfare_penalty": float,
            "masked_feature_names": list,
            "wola_2021_in_training": bool,
            "relative_path": str,
        }
        if any(type(row[field]) is not expected for field, expected in type_contract.items()):
            raise RuntimeError("fit inventory contains a JSON type alias")
        if row["popsize"] is not None:
            raise RuntimeError("fit inventory population size differs")
        kappa = row["support_floor_kappa"]
        if kappa is not None and type(kappa) is not float:
            raise RuntimeError("fit inventory support-floor type differs")
        dependency = row["grid_dependency"]
        if dependency is not None and type(dependency) is not str:
            raise RuntimeError("fit inventory grid dependency type differs")
        if row["relative_path"] != f"fits/{row['fit_id']}.json":
            raise RuntimeError("fit inventory relative path differs")
        fit_id = Path(str(row["fit_id"]))
        if fit_id.is_absolute() or ".." in fit_id.parts or len(fit_id.parts) < 2:
            raise RuntimeError("fit inventory contains a non-canonical fit id")

    fit_ids = [str(row["fit_id"]) for row in inventory]
    if fit_ids != sorted(fit_ids) or len(set(fit_ids)) != len(fit_ids):
        raise RuntimeError("fit inventory order or identities differ")

    expected = _build_locked_fit_inventory_v2()
    if not exact_json_equal_v2(inventory, expected):
        raise RuntimeError("fit inventory differs from the locked matrix")
    return inventory


def canonical_fit_path_v2(
    result_root: Path,
    spec: Mapping[str, object],
) -> Path:
    """Return the one canonical result path for a locked fit coordinate."""

    validate_locked_fit_inventory_v2(locked_fit_inventory_v2())
    if not any(exact_json_equal_v2(row, spec) for row in locked_fit_inventory_v2()):
        raise RuntimeError("fit inventory coordinate is not locked")
    relative = Path(str(spec["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("fit inventory path is not canonical")
    return Path(result_root).absolute() / relative


def canonical_grid_path_v2(result_root: Path) -> Path:
    """Return the sole deterministic training-grid path."""

    return Path(result_root).absolute() / "training_grid" / "static_senior_alpha.json"


def canonical_split_path_v2(result_root: Path, name: str) -> Path:
    """Return one of the seven exact primary split artifact paths."""

    if type(name) is not str or name not in SPLIT_SHA256_V2:
        raise RuntimeError(f"unknown primary-v2 split: {name!r}")
    return Path(result_root).absolute() / "splits" / f"{name}.json"


def _read_regular(path: Path, *, label: str) -> bytes:
    content = read_regular_bytes_artifact_v2(Path(path), label=label)
    assert content is not None
    return content


def _parse_frozen_list(content: bytes) -> list[tuple[str, str, int]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("primary frozen list is not UTF-8") from exc
    rows: list[tuple[str, str, int]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise RuntimeError(f"malformed primary frozen-list line {line_number}")
        name, series, raw_year = fields
        try:
            year = int(raw_year)
        except ValueError as exc:
            raise RuntimeError(
                f"malformed primary frozen-list year on line {line_number}"
            ) from exc
        rows.append((name, series, year))
    identities = [(name, series, year) for name, series, year in rows]
    if len(rows) != 132 or len(set(identities)) != len(rows):
        raise RuntimeError("primary frozen-list inventory differs")
    return rows


def primary_manifest_retained_sha256_v2(payload: Mapping[str, object]) -> str:
    """Hash a primary manifest while excluding machine-location fields."""

    retained = {
        key: value
        for key, value in payload.items()
        if key not in {"source_dir", "destination"}
    }
    return _canonical_json_sha256(retained)


def build_primary_manifest_v2(
    multicity_manifest_path: Path,
    frozen_list_path: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Authenticate the 129-row overlap and three direct Łódź anchors."""

    frozen_bytes = _read_regular(frozen_list_path, label="primary frozen list")
    if hashlib.sha256(frozen_bytes).hexdigest() != PARENT_FROZEN_LIST_SHA256:
        raise RuntimeError("primary frozen-list digest differs")
    frozen_rows = _parse_frozen_list(frozen_bytes)

    parent = load_json_object_strict_v2(
        multicity_manifest_path,
        label="authenticated multicity-v2 manifest",
    )
    if (
        parent.get("source_commit") != PABULIB_COMMIT
        or multicity_manifest_retained_sha256(parent)
        != EXPECTED_MULTICITY_MANIFEST_RETAINED_SHA256
    ):
        raise RuntimeError("authenticated multicity-v2 manifest differs")
    parent_rows = parent.get("files")
    if type(parent_rows) is not list:
        raise RuntimeError("authenticated multicity-v2 manifest rows are malformed")
    by_name: dict[str, Mapping[str, object]] = {}
    expected_row_fields = {
        "name",
        "source_relative",
        "series",
        "year",
        "bytes",
        "sha256",
    }
    for row in parent_rows:
        if type(row) is not dict or set(row) != expected_row_fields:
            raise RuntimeError("authenticated multicity-v2 manifest row is malformed")
        name = row.get("name")
        if type(name) is not str or name in by_name:
            raise RuntimeError("authenticated multicity-v2 manifest names differ")
        by_name[name] = row

    selected: list[dict[str, object]] = []
    overlap = 0
    data = Path(data_dir).absolute()
    for name, series, year in frozen_rows:
        raw_path = data / name
        raw = _read_regular(raw_path, label=f"primary raw election {name}")
        observed_sha256 = hashlib.sha256(raw).hexdigest()
        parent_row = by_name.get(name)
        if parent_row is not None:
            overlap += 1
            if (
                parent_row.get("series") != series
                or type(parent_row.get("year")) is not int
                or parent_row.get("year") != year
                or type(parent_row.get("bytes")) is not int
                or parent_row.get("bytes") != len(raw)
                or parent_row.get("sha256") != observed_sha256
            ):
                raise RuntimeError(f"multicity overlap identity differs for {name}")
            selected.append(dict(parent_row))
            continue

        direct = _DIRECT_LODZ_ANCHORS.get(name)
        if direct is None or direct != (series, year, observed_sha256):
            raise RuntimeError(f"unauthenticated direct primary anchor: {name}")
        selected.append(
            {
                "name": name,
                "source_relative": name,
                "series": series,
                "year": year,
                "bytes": len(raw),
                "sha256": observed_sha256,
            }
        )

    if overlap != 129 or {row["name"] for row in selected if row["name"] in _DIRECT_LODZ_ANCHORS} != set(_DIRECT_LODZ_ANCHORS):
        raise RuntimeError("primary overlap/direct-anchor coverage differs")
    selected.sort(key=lambda row: str(row["name"]))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_commit": PABULIB_COMMIT,
        "source_dir": "data/pb",
        "destination": "data/pb",
        "n_series": len({str(row["series"]) for row in selected}),
        "n_elections": len(selected),
        "n_files": len(selected),
        "files": selected,
    }
    if (
        manifest["n_series"] != 19
        or manifest["n_elections"] != 132
        or primary_manifest_retained_sha256_v2(manifest)
        != EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256
    ):
        raise RuntimeError("primary manifest retained identity differs")
    return manifest


def validate_primary_split_sources_v2(split_dir: Path) -> dict[str, str]:
    """Authenticate the exact seven protected split snapshots."""

    observed: dict[str, str] = {}
    for name, expected in SPLIT_SHA256_V2.items():
        path = Path(split_dir) / f"{name}.json"
        content = _read_regular(path, label=f"primary split {name}")
        load_json_object_bytes_strict_v2(content, label=f"primary split {name}")
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise RuntimeError(f"primary split digest differs for {name}")
        observed[name] = digest
    return observed


def _canonical_json_bytes_v2(payload: Mapping[str, object]) -> bytes:
    _require_finite_json(payload, label="primary-v2 artifact")
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def stage_protocol_inputs_v2(
    *,
    multicity_manifest_path: Path,
    frozen_list_path: Path,
    split_source_dir: Path,
    data_dir: Path,
    result_root: Path,
) -> dict[str, object]:
    """Create the primary manifest, receipt, and seven inherited splits once."""

    split_hashes = validate_primary_split_sources_v2(split_source_dir)
    if split_hashes != SPLIT_SHA256_V2:
        raise RuntimeError("primary split identity set differs")
    manifest = build_primary_manifest_v2(
        multicity_manifest_path,
        frozen_list_path,
        data_dir,
    )
    retained = primary_manifest_retained_sha256_v2(manifest)
    if retained != EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256:
        raise RuntimeError("primary manifest retained digest differs")
    receipt = build_primary_semantics_receipt_v2(manifest, data_dir)
    validate_primary_semantics_receipt_v2(receipt)

    root = Path(result_root).absolute()
    manifest_bytes = _canonical_json_bytes_v2(manifest)
    receipt_bytes = _canonical_json_bytes_v2(receipt)
    artifacts: list[tuple[Path, bytes, str]] = [
        (root / "corpus_manifest.json", manifest_bytes, "primary-v2 corpus manifest"),
        (
            root / "approval_semantics_receipt.json",
            receipt_bytes,
            "primary-v2 approval semantics receipt",
        ),
    ]
    for name in sorted(SPLIT_SHA256_V2):
        content = _read_regular(
            Path(split_source_dir) / f"{name}.json",
            label=f"primary split source {name}",
        )
        if hashlib.sha256(content).hexdigest() != split_hashes[name]:
            raise RuntimeError(f"primary split changed while staging: {name}")
        artifacts.append(
            (root / "splits" / f"{name}.json", content, f"primary-v2 split {name}")
        )

    for path, content, label in artifacts:
        preflight_immutable_bytes_artifact_v2(path, content, label=label)
    installed = {
        str(path): write_immutable_bytes_artifact_v2(path, content, label=label)
        for path, content, label in artifacts
    }
    if any(installed[str(path)] != hashlib.sha256(content).hexdigest() for path, content, _ in artifacts):
        raise RuntimeError("primary-v2 staged artifact digest differs after install")
    return {
        "corpus_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "approval_semantics_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "primary_manifest_retained_sha256": retained,
        "corpus_semantic_sha256": receipt["corpus_semantic_sha256"],
        "split_sha256": dict(split_hashes),
    }


def _locked_artifact_snapshot_v2(
    path: Path,
    label: str,
    lock_payload: Mapping[str, object] | None,
) -> JsonArtifactSnapshotV2:
    content = read_regular_bytes_artifact_v2(Path(path), label=label)
    assert content is not None
    digest = hashlib.sha256(content).hexdigest()
    if lock_payload is not None:
        tracked = lock_payload.get("tracked_files")
        row = tracked.get(label) if isinstance(tracked, Mapping) else None
        if not isinstance(row, Mapping) or not _is_sha256(row.get("sha256")):
            raise RuntimeError(f"primary-v2 lock is missing {label}")
        if row["sha256"] != digest:
            raise RuntimeError(f"primary-v2 locked digest differs for {label}")
    payload = load_json_object_bytes_strict_v2(content, label=label)
    return JsonArtifactSnapshotV2(
        label=label,
        path=Path(path).absolute(),
        content=content,
        sha256=digest,
        payload=payload,
    )


def _primary_index_metadata_from_snapshot_v2(
    manifest: JsonArtifactSnapshotV2,
    data_dir: Path,
    *,
    enforce_expected_counts: bool,
) -> tuple[
    dict[str, SeriesRef],
    dict[tuple[str, int, str], RawArtifactIdentityV2],
]:
    payload = manifest.payload
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("source_commit") != PABULIB_COMMIT
    ):
        raise RuntimeError("primary-v2 manifest schema or source differs")
    rows = payload.get("files")
    if (
        type(rows) is not list
        or type(payload.get("n_files")) is not int
        or payload.get("n_files") != len(rows)
        or type(payload.get("n_elections")) is not int
        or payload.get("n_elections") != len(rows)
    ):
        raise RuntimeError("primary-v2 manifest counts differ")
    if enforce_expected_counts and (
        len(rows) != 132
        or payload.get("n_series") != 19
        or primary_manifest_retained_sha256_v2(payload)
        != EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256
    ):
        raise RuntimeError("primary-v2 manifest retained identity differs")

    by_series: dict[str, dict[int, Path]] = {}
    raw_metadata: dict[tuple[str, int, str], RawArtifactIdentityV2] = {}
    names: set[str] = set()
    identities: set[tuple[str, int, str]] = set()
    root = Path(data_dir).absolute()
    expected_fields = {
        "name",
        "source_relative",
        "series",
        "year",
        "bytes",
        "sha256",
    }
    for row in rows:
        if type(row) is not dict or set(row) != expected_fields:
            raise RuntimeError("primary-v2 manifest file row differs")
        name = row.get("name")
        series = row.get("series")
        year = row.get("year")
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            type(name) is not str
            or type(row.get("source_relative")) is not str
            or type(series) is not str
            or type(year) is not int
            or type(size) is not int
            or not _is_sha256(digest)
            or Path(name).name != name
        ):
            raise RuntimeError("primary-v2 manifest file-row types differ")
        identity = (series, year, name)
        if name in names or identity in identities:
            raise RuntimeError("primary-v2 manifest contains a duplicate identity")
        names.add(name)
        identities.add(identity)
        path = root / name
        raw_metadata[identity] = RawArtifactIdentityV2(
            series=series,
            year=year,
            name=name,
            path=path,
            size=size,
            sha256=digest,
        )
        years = by_series.setdefault(series, {})
        if year in years:
            raise RuntimeError(f"duplicate primary-v2 series year: {series} {year}")
        years[year] = path
    index = {
        series: SeriesRef(
            key=series,
            years=tuple(sorted(years)),
            paths=tuple(years[year] for year in sorted(years)),
        )
        for series, years in sorted(by_series.items())
    }
    if type(payload.get("n_series")) is not int or payload.get("n_series") != len(index):
        raise RuntimeError("primary-v2 manifest series count differs")
    return index, raw_metadata


def _primary_index_from_snapshot_v2(
    manifest: JsonArtifactSnapshotV2,
    data_dir: Path,
    *,
    enforce_expected_counts: bool,
) -> tuple[
    dict[str, SeriesRef],
    dict[tuple[str, int, str], RawArtifactIdentityV2],
]:
    """Compatibility wrapper returning manifest metadata without raw reads."""

    return _primary_index_metadata_from_snapshot_v2(
        manifest,
        data_dir,
        enforce_expected_counts=enforce_expected_counts,
    )


def _authenticate_raw_artifacts_v2(
    raw_metadata: Mapping[tuple[str, int, str], RawArtifactIdentityV2],
    authorized_identities: set[tuple[str, int, str]],
) -> dict[tuple[str, int, str], RawArtifactIdentityV2]:
    """Read and authenticate exactly the raw identities authorized for this use."""

    if not authorized_identities <= set(raw_metadata):
        raise RuntimeError("primary-v2 authorized raw identity is absent from manifest")
    authenticated: dict[tuple[str, int, str], RawArtifactIdentityV2] = {}
    for identity in sorted(authorized_identities):
        artifact = raw_metadata[identity]
        content = read_regular_bytes_artifact_v2(
            artifact.path,
            label=f"primary-v2 authorized raw election {artifact.name}",
        )
        assert content is not None
        if (
            len(content) != artifact.size
            or hashlib.sha256(content).hexdigest() != artifact.sha256
        ):
            raise RuntimeError(
                f"primary-v2 authorized raw election identity differs: {artifact.name}"
            )
        authenticated[identity] = artifact
    return authenticated


def _validate_manifest_receipt_identity_v2(
    index: Mapping[str, SeriesRef],
    raw_artifacts: Mapping[tuple[str, int, str], RawArtifactIdentityV2],
    receipt: Mapping[str, object],
) -> None:
    if receipt.get("semantics_profile") != SEMANTICS_PROFILE:
        raise RuntimeError("primary-v2 semantics receipt profile differs")
    rows = receipt.get("files")
    if type(rows) is not list:
        raise RuntimeError("primary-v2 semantics receipt rows are malformed")
    receipt_identities: set[tuple[str, int, str]] = set()
    raw_by_identity: dict[tuple[str, int, str], str] = {}
    for row in rows:
        if type(row) is not dict:
            raise RuntimeError("primary-v2 semantics receipt row is malformed")
        series, year, name = row.get("series"), row.get("year"), row.get("name")
        raw_digest = row.get("raw_file_sha256")
        semantic_digest = row.get("v2_semantic_sha256")
        if (
            type(series) is not str
            or type(year) is not int
            or type(name) is not str
            or not _is_sha256(raw_digest)
            or not _is_sha256(semantic_digest)
        ):
            raise RuntimeError("primary-v2 semantics receipt identity is malformed")
        identity = (series, year, name)
        if identity in receipt_identities:
            raise RuntimeError("primary-v2 semantics receipt identity is duplicated")
        receipt_identities.add(identity)
        raw_by_identity[identity] = raw_digest
    manifest_identities = {
        (series, year, path.name)
        for series, ref in index.items()
        for year, path in zip(ref.years, ref.paths)
    }
    if receipt_identities != manifest_identities:
        raise RuntimeError("primary-v2 manifest and receipt identities differ")
    if set(raw_artifacts) != manifest_identities:
        raise RuntimeError("primary-v2 manifest raw-artifact identities differ")
    for identity, artifact in raw_artifacts.items():
        if artifact.sha256 != raw_by_identity[identity]:
            raise RuntimeError("primary-v2 manifest and receipt raw digests differ")


def _split_views_from_snapshot_v2(
    artifact: JsonArtifactSnapshotV2,
    index: Mapping[str, SeriesRef],
    *,
    expected_name: str,
) -> tuple[
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    str,
]:
    payload = artifact.payload
    if set(payload) not in ({"name", "note", "train", "test"}, {"name", "train", "test"}):
        raise RuntimeError(f"primary-v2 inherited split schema differs: {expected_name}")
    if payload.get("name") != expected_name or type(payload.get("train")) is not list or type(payload.get("test")) is not list:
        raise RuntimeError(f"primary-v2 inherited split identity differs: {expected_name}")

    def parse_rows(raw_rows: object, view: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
        if type(raw_rows) is not list:
            raise RuntimeError(f"primary-v2 split {view} rows are malformed")
        parsed: list[tuple[str, tuple[int, ...]]] = []
        identities: set[tuple[str, int]] = set()
        series_seen: set[str] = set()
        for raw in raw_rows:
            if type(raw) is not list or len(raw) != 2 or type(raw[0]) is not str or type(raw[1]) is not list or any(type(year) is not int for year in raw[1]):
                raise RuntimeError(f"primary-v2 split {view} row is malformed")
            series = raw[0]
            years = tuple(raw[1])
            if series in series_seen or series not in index or any(year not in index[series].years for year in years):
                raise RuntimeError(f"primary-v2 split {view} identity differs")
            series_seen.add(series)
            for year in years:
                if (series, year) in identities:
                    raise RuntimeError(f"primary-v2 split {view} election is duplicated")
                identities.add((series, year))
            parsed.append((series, years))
        return tuple(parsed)

    train = parse_rows(payload["train"], "train")
    test = parse_rows(payload["test"], "test")
    if {(key, year) for key, years in train for year in years} & {
        (key, year) for key, years in test for year in years
    }:
        raise RuntimeError("primary-v2 split train/test overlap detected")
    return train, test, str(payload.get("note", ""))


def _training_split_from_snapshot_v2(
    artifact: JsonArtifactSnapshotV2,
    index: Mapping[str, SeriesRef],
    *,
    expected_name: str,
) -> Split:
    """Return only the training projection of one authenticated split."""

    train, _, note = _split_views_from_snapshot_v2(
        artifact,
        index,
        expected_name=expected_name,
    )
    return Split(name=expected_name, train=train, test=(), note=note)


def _authorized_raw_identities_v2(
    raw_metadata: Mapping[tuple[str, int, str], RawArtifactIdentityV2],
    split_artifacts: Mapping[str, JsonArtifactSnapshotV2],
    index: Mapping[str, SeriesRef],
    *,
    include_test: bool,
) -> set[tuple[str, int, str]]:
    """Resolve the exact raw identity union for selected split views."""

    by_pair: dict[tuple[str, int], tuple[str, int, str]] = {}
    for identity in raw_metadata:
        pair = identity[:2]
        if pair in by_pair:
            raise RuntimeError("primary-v2 manifest repeats a series-year identity")
        by_pair[pair] = identity
    pairs: set[tuple[str, int]] = set()
    for name, artifact in split_artifacts.items():
        train, test, _ = _split_views_from_snapshot_v2(
            artifact,
            index,
            expected_name=name,
        )
        views = (train, test) if include_test else (train,)
        for view in views:
            pairs.update(
                (series, year)
                for series, years in view
                for year in years
            )
    if not pairs <= set(by_pair):
        raise RuntimeError("primary-v2 split authorizes an unknown raw identity")
    return {by_pair[pair] for pair in pairs}


def validate_protocol_lock_snapshot_identity_v2(
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
    result_root: Path,
) -> Mapping[str, object]:
    """Validate the byte identity and result-root binding of a verified lock."""

    if type(lock_snapshot) is not PrimaryProtocolLockSnapshotV2:
        raise RuntimeError("primary-v2 input loader requires a verified lock snapshot")
    expected_path = Path(result_root).absolute() / "protocol_lock.json"
    if lock_snapshot.path.absolute() != expected_path:
        raise RuntimeError("primary-v2 verified lock path is not canonical")
    if (
        type(lock_snapshot.content) is not bytes
        or not _is_sha256(lock_snapshot.sha256)
        or hashlib.sha256(lock_snapshot.content).hexdigest() != lock_snapshot.sha256
    ):
        raise RuntimeError("primary-v2 verified lock byte identity differs")
    current = read_regular_bytes_artifact_v2(
        lock_snapshot.path,
        label="primary-v2 verified protocol lock",
    )
    if current != lock_snapshot.content:
        raise RuntimeError("primary-v2 verified lock path differs from its snapshot")
    parsed = load_json_object_bytes_strict_v2(
        lock_snapshot.content,
        label="primary-v2 verified protocol lock snapshot",
    )
    if not exact_json_equal_v2(parsed, lock_snapshot.payload):
        raise RuntimeError("primary-v2 verified lock payload differs from its bytes")
    required_fields = {
        "schema_version",
        "lock_profile",
        "mandatory_file_count",
        "tracked_files",
        "fit_inventory",
        "training_grid_spec",
        "protocol",
        "corpus_semantic_sha256",
    }
    if (
        set(parsed) != required_fields
        or type(parsed.get("schema_version")) is not int
        or parsed.get("schema_version") != 2
        or parsed.get("lock_profile") != LOCK_PROFILE_V2
        or type(parsed.get("corpus_semantic_sha256")) is not str
        or parsed.get("corpus_semantic_sha256")
        != EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
    ):
        raise RuntimeError("primary-v2 verified lock schema differs")
    validate_locked_fit_inventory_v2(parsed.get("fit_inventory"))
    if not exact_json_equal_v2(parsed.get("training_grid_spec"), TRAINING_GRID_SPEC_V2):
        raise RuntimeError("primary-v2 verified lock training-grid contract differs")
    if not exact_json_equal_v2(parsed.get("protocol"), protocol_config_v2()):
        raise RuntimeError("primary-v2 verified lock protocol differs")
    tracked = parsed.get("tracked_files")
    if type(tracked) is not dict or type(parsed.get("mandatory_file_count")) is not int:
        raise RuntimeError("primary-v2 verified lock tracked-file closure differs")
    if parsed["mandatory_file_count"] != len(tracked):
        raise RuntimeError("primary-v2 verified lock tracked-file count differs")
    return parsed


def _tracked_digest_from_lock_v2(
    lock_payload: Mapping[str, object],
    label: str,
) -> str:
    tracked = lock_payload.get("tracked_files")
    row = tracked.get(label) if isinstance(tracked, Mapping) else None
    digest = row.get("sha256") if isinstance(row, Mapping) else None
    if not _is_sha256(digest):
        raise RuntimeError(f"primary-v2 lock lacks tracked digest {label}")
    return digest


def _authenticate_evaluation_matrix_v2(
    receipt_payload: Mapping[str, object],
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
    result_root: Path,
    data_dir: Path,
) -> Mapping[str, object]:
    """Authenticate the canonical grid and all 49 fits without raw-data access."""

    # Imported lazily because the trainer imports this protocol module. This
    # verifier is read-only; its base-lock verification skips raw file bytes.
    from iclr_primary_warsaw_train_v2 import verify_complete_matrix_v2

    result = Path(result_root).absolute()
    matrix = verify_complete_matrix_v2(
        repo_root=result.parents[1],
        result_root=result,
        data_dir=Path(data_dir).absolute(),
    )
    expected = {
        "schema_version": 2,
        "status": "complete",
        "protocol_lock_sha256": lock_snapshot.sha256,
        "grid_sha256": receipt_payload.get("training_grid_sha256"),
        "fit_count": 49,
        "fit_sha256": receipt_payload.get("fit_sha256"),
    }
    if not exact_json_equal_v2(matrix, expected):
        raise RuntimeError(
            "primary-v2 evaluation receipt differs from the authenticated "
            "complete matrix"
        )
    return matrix


def _validate_evaluation_replay_receipt_snapshot_v2(
    replay_receipt_snapshot: object,
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
    result_root: Path,
    data_dir: Path,
) -> JsonArtifactSnapshotV2:
    """Authenticate the on-disk capability before any held-out raw access."""

    lock_payload = validate_protocol_lock_snapshot_identity_v2(
        lock_snapshot,
        Path(result_root).absolute(),
    )
    path_value = getattr(replay_receipt_snapshot, "path", None)
    content = getattr(replay_receipt_snapshot, "content", None)
    digest = getattr(replay_receipt_snapshot, "sha256", None)
    supplied_payload = getattr(replay_receipt_snapshot, "payload", None)
    expected_path = (
        Path(result_root).absolute() / EVALUATION_REPLAY_RECEIPT_NAME_V2
    )
    if not isinstance(path_value, Path) or path_value.absolute() != expected_path:
        raise RuntimeError("primary-v2 evaluation receipt path is not canonical")
    if type(content) is not bytes or not _is_sha256(digest):
        raise RuntimeError("primary-v2 evaluation receipt byte identity differs")
    if hashlib.sha256(content).hexdigest() != digest:
        raise RuntimeError("primary-v2 evaluation receipt digest differs")
    observed = read_regular_bytes_artifact_v2(
        expected_path,
        label="primary-v2 evaluation replay receipt",
    )
    if observed != content:
        raise RuntimeError("primary-v2 evaluation receipt changed before raw access")
    payload = load_json_object_bytes_strict_v2(
        content,
        label="primary-v2 evaluation replay receipt",
    )
    if not exact_json_equal_v2(payload, supplied_payload):
        raise RuntimeError("primary-v2 evaluation receipt payload differs from bytes")
    fields = {
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
    fit_sha256 = payload.get("fit_sha256")
    expected_fit_ids = sorted(
        str(row["fit_id"]) for row in locked_fit_inventory_v2()
    )
    if (
        set(payload) != fields
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("event") != "evaluation replay started"
        or payload.get("classification") != "corrected post-hoc replay"
        or type(payload.get("heldout_outcomes_already_known")) is not bool
        or payload.get("heldout_outcomes_already_known") is not True
        or type(payload.get("fresh_holdout")) is not bool
        or payload.get("fresh_holdout") is not False
        or type(payload.get("preregistered")) is not bool
        or payload.get("preregistered") is not False
        or payload.get("disclosure") != EVALUATION_REPLAY_DISCLOSURE_V2
        or payload.get("protocol_lock_sha256") != lock_snapshot.sha256
        or payload.get("semantics_profile") != SEMANTICS_PROFILE
        or payload.get("semantics_receipt_sha256")
        != _tracked_digest_from_lock_v2(
            lock_payload, "artifact/approval_semantics_receipt"
        )
        or payload.get("corpus_semantic_sha256")
        != lock_payload.get("corpus_semantic_sha256")
        or payload.get("structural_gates_sha256")
        != _tracked_digest_from_lock_v2(lock_payload, "artifact/structural_gates")
        or not _is_sha256(payload.get("training_grid_sha256"))
        or type(payload.get("fit_count")) is not int
        or payload.get("fit_count") != 49
        or type(fit_sha256) is not dict
        or list(fit_sha256) != expected_fit_ids
        or any(not _is_sha256(value) for value in fit_sha256.values())
    ):
        raise RuntimeError("primary-v2 evaluation receipt authorization differs")
    _authenticate_evaluation_matrix_v2(
        payload,
        lock_snapshot,
        Path(result_root),
        Path(data_dir),
    )
    return JsonArtifactSnapshotV2(
        label="artifact/evaluation_replay_started",
        path=expected_path,
        content=content,
        sha256=digest,
        payload=payload,
    )


def _validate_json_snapshot_v2(
    artifact: JsonArtifactSnapshotV2,
    *,
    label: str,
    path: Path,
    tracked: Mapping[str, object],
) -> None:
    if type(artifact) is not JsonArtifactSnapshotV2:
        raise RuntimeError(f"primary-v2 input snapshot is malformed: {label}")
    row = tracked.get(label)
    expected_digest = row.get("sha256") if isinstance(row, Mapping) else None
    if (
        artifact.label != label
        or artifact.path != Path(path).absolute()
        or type(artifact.content) is not bytes
        or hashlib.sha256(artifact.content).hexdigest() != artifact.sha256
        or artifact.sha256 != expected_digest
    ):
        raise RuntimeError(f"primary-v2 input snapshot identity differs: {label}")
    parsed = load_json_object_bytes_strict_v2(
        artifact.content,
        label=f"primary-v2 input snapshot {label}",
    )
    if not exact_json_equal_v2(parsed, artifact.payload):
        raise RuntimeError(f"primary-v2 input snapshot payload differs: {label}")


def validate_protocol_inputs_snapshot_v2(
    snapshot: PrimaryInputsSnapshotV2,
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
) -> None:
    """Cross-bind every in-memory input view to one verified base-lock image."""

    if type(snapshot) is not PrimaryInputsSnapshotV2:
        raise RuntimeError("primary-v2 computation requires an authenticated input snapshot")
    result_root = snapshot.result_root
    if (
        not isinstance(result_root, Path)
        or result_root != result_root.absolute()
        or not isinstance(snapshot.data_dir, Path)
        or snapshot.data_dir != snapshot.data_dir.absolute()
    ):
        raise RuntimeError("primary-v2 input snapshot roots are not canonical")
    lock_payload = validate_protocol_lock_snapshot_identity_v2(
        lock_snapshot,
        result_root,
    )
    if (
        snapshot.protocol_lock_path.absolute() != lock_snapshot.path.absolute()
        or snapshot.protocol_lock_content != lock_snapshot.content
        or snapshot.protocol_lock_sha256 != lock_snapshot.sha256
    ):
        raise RuntimeError("primary-v2 input snapshot belongs to a different lock")
    tracked = lock_payload["tracked_files"]
    assert isinstance(tracked, Mapping)
    _validate_json_snapshot_v2(
        snapshot.manifest,
        label="artifact/corpus_manifest",
        path=result_root / "corpus_manifest.json",
        tracked=tracked,
    )
    _validate_json_snapshot_v2(
        snapshot.semantics_receipt,
        label="artifact/approval_semantics_receipt",
        path=result_root / "approval_semantics_receipt.json",
        tracked=tracked,
    )
    if (
        snapshot.semantics_receipt.payload.get("corpus_semantic_sha256")
        != lock_payload["corpus_semantic_sha256"]
    ):
        raise RuntimeError("primary-v2 input snapshot corpus semantic digest differs")

    expected_index, raw_metadata = _primary_index_metadata_from_snapshot_v2(
        snapshot.manifest,
        snapshot.data_dir,
        enforce_expected_counts=False,
    )
    if snapshot.index != expected_index:
        raise RuntimeError("primary-v2 input snapshot index differs from manifest")
    _validate_manifest_receipt_identity_v2(
        snapshot.index,
        raw_metadata,
        snapshot.semantics_receipt.payload,
    )

    if set(snapshot.split_artifacts) != set(snapshot.splits) or not snapshot.splits:
        raise RuntimeError("primary-v2 input snapshot split inventory differs")
    for name, artifact in snapshot.split_artifacts.items():
        _validate_json_snapshot_v2(
            artifact,
            label=f"split/{name}",
            path=result_root / "splits" / f"{name}.json",
            tracked=tracked,
        )
        expected_split = _training_split_from_snapshot_v2(
            artifact,
            snapshot.index,
            expected_name=name,
        )
        if snapshot.splits.get(name) != expected_split:
            raise RuntimeError(f"primary-v2 input snapshot split view differs: {name}")

    if snapshot.access_scope == TRAINING_ACCESS_SCOPE_V2:
        if snapshot.evaluation_replay_receipt is not None:
            raise RuntimeError("primary-v2 training snapshot retains an evaluation receipt")
        include_test = False
    elif snapshot.access_scope == EVALUATION_ACCESS_SCOPE_V2:
        if type(snapshot.evaluation_replay_receipt) is not JsonArtifactSnapshotV2:
            raise RuntimeError("primary-v2 evaluation snapshot lacks its receipt")
        authorized_receipt = _validate_evaluation_replay_receipt_snapshot_v2(
            snapshot.evaluation_replay_receipt,
            lock_snapshot,
            result_root,
            snapshot.data_dir,
        )
        if authorized_receipt != snapshot.evaluation_replay_receipt:
            raise RuntimeError("primary-v2 evaluation receipt snapshot differs")
        include_test = True
    else:
        raise RuntimeError("primary-v2 input snapshot access scope differs")

    authorized = _authorized_raw_identities_v2(
        raw_metadata,
        snapshot.split_artifacts,
        snapshot.index,
        include_test=include_test,
    )
    if set(snapshot.raw_artifacts) != authorized:
        raise RuntimeError("primary-v2 input snapshot raw authorization differs")
    receipt_rows = snapshot.semantics_receipt.payload.get("files")
    receipt_raw = {
        (row["series"], row["year"], row["name"]): row["raw_file_sha256"]
        for row in receipt_rows
        if type(row) is dict
    } if type(receipt_rows) is list else {}
    for identity, artifact in snapshot.raw_artifacts.items():
        expected = raw_metadata.get(identity)
        if (
            type(artifact) is not RawArtifactIdentityV2
            or expected is None
            or artifact != expected
            or receipt_raw.get(identity) != artifact.sha256
        ):
            raise RuntimeError("primary-v2 input snapshot raw identity differs")


def _load_protocol_inputs_snapshot_scoped_v2(
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
    result_root: Path,
    data_dir: Path,
    split_names: Sequence[str],
    *,
    enforce_expected_counts: bool,
    access_scope: str,
    evaluation_replay_receipt: JsonArtifactSnapshotV2 | None,
) -> PrimaryInputsSnapshotV2:
    """Load one already-authorized raw scope from locked metadata."""

    names = tuple(split_names)
    if (
        not names
        or len(names) != len(set(names))
        or any(name not in SPLIT_SHA256_V2 for name in names)
    ):
        raise RuntimeError("primary-v2 input snapshot split names differ")
    root = Path(result_root).absolute()
    lock_payload = validate_protocol_lock_snapshot_identity_v2(lock_snapshot, root)
    manifest = _locked_artifact_snapshot_v2(
        root / "corpus_manifest.json",
        "artifact/corpus_manifest",
        lock_payload,
    )
    receipt = _locked_artifact_snapshot_v2(
        root / "approval_semantics_receipt.json",
        "artifact/approval_semantics_receipt",
        lock_payload,
    )
    split_artifacts = {
        name: _locked_artifact_snapshot_v2(
            root / "splits" / f"{name}.json",
            f"split/{name}",
            lock_payload,
        )
        for name in names
    }
    index, raw_metadata = _primary_index_metadata_from_snapshot_v2(
        manifest,
        data_dir,
        enforce_expected_counts=enforce_expected_counts,
    )
    if enforce_expected_counts:
        validate_primary_semantics_receipt_v2(receipt.payload)
    _validate_manifest_receipt_identity_v2(index, raw_metadata, receipt.payload)
    splits = {
        name: _training_split_from_snapshot_v2(
            split_artifacts[name],
            index,
            expected_name=name,
        )
        for name in names
    }
    if enforce_expected_counts:
        for name, artifact in split_artifacts.items():
            if artifact.sha256 != SPLIT_SHA256_V2[name]:
                raise RuntimeError(f"primary-v2 inherited split digest differs: {name}")
    if access_scope == TRAINING_ACCESS_SCOPE_V2:
        if evaluation_replay_receipt is not None:
            raise RuntimeError("primary-v2 training raw scope cannot retain a receipt")
        include_test = False
    elif access_scope == EVALUATION_ACCESS_SCOPE_V2:
        if type(evaluation_replay_receipt) is not JsonArtifactSnapshotV2:
            raise RuntimeError("primary-v2 evaluation raw scope lacks authorization")
        include_test = True
    else:
        raise RuntimeError("primary-v2 raw access scope differs")
    authorized = _authorized_raw_identities_v2(
        raw_metadata,
        split_artifacts,
        index,
        include_test=include_test,
    )
    raw_artifacts = _authenticate_raw_artifacts_v2(raw_metadata, authorized)
    snapshot = PrimaryInputsSnapshotV2(
        result_root=root,
        data_dir=Path(data_dir).absolute(),
        protocol_lock_path=lock_snapshot.path.absolute(),
        protocol_lock_content=lock_snapshot.content,
        protocol_lock_sha256=lock_snapshot.sha256,
        manifest=manifest,
        semantics_receipt=receipt,
        split_artifacts=split_artifacts,
        raw_artifacts=raw_artifacts,
        index=index,
        splits=splits,
        access_scope=access_scope,
        evaluation_replay_receipt=evaluation_replay_receipt,
    )
    validate_protocol_inputs_snapshot_v2(snapshot, lock_snapshot)
    return snapshot


def load_protocol_inputs_snapshot_v2(
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
    result_root: Path,
    data_dir: Path,
    split_names: Sequence[str],
    *,
    enforce_expected_counts: bool = True,
) -> PrimaryInputsSnapshotV2:
    """Authenticate and retain only raw identities named by ``split.train``."""

    return _load_protocol_inputs_snapshot_scoped_v2(
        lock_snapshot,
        result_root,
        data_dir,
        split_names,
        enforce_expected_counts=enforce_expected_counts,
        access_scope=TRAINING_ACCESS_SCOPE_V2,
        evaluation_replay_receipt=None,
    )


def load_evaluation_inputs_snapshot_v2(
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
    result_root: Path,
    data_dir: Path,
    split_names: Sequence[str],
    *,
    replay_receipt_snapshot: object,
    enforce_expected_counts: bool = True,
) -> PrimaryInputsSnapshotV2:
    """Authorize full split views only after authenticating the replay receipt."""

    root = Path(result_root).absolute()
    authorized_receipt = _validate_evaluation_replay_receipt_snapshot_v2(
        replay_receipt_snapshot,
        lock_snapshot,
        root,
        Path(data_dir).absolute(),
    )
    return _load_protocol_inputs_snapshot_scoped_v2(
        lock_snapshot,
        root,
        data_dir,
        split_names,
        enforce_expected_counts=enforce_expected_counts,
        access_scope=EVALUATION_ACCESS_SCOPE_V2,
        evaluation_replay_receipt=authorized_receipt,
    )


def assert_protocol_inputs_unchanged_v2(
    snapshot: PrimaryInputsSnapshotV2,
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
) -> None:
    """Reject lock, artifact, or raw-training drift after computation."""

    validate_protocol_inputs_snapshot_v2(snapshot, lock_snapshot)
    current_lock = read_regular_bytes_artifact_v2(
        snapshot.protocol_lock_path,
        label="primary-v2 protocol lock",
    )
    if current_lock != snapshot.protocol_lock_content:
        raise RuntimeError("primary-v2 protocol lock changed after input snapshot")
    for artifact in (
        snapshot.manifest,
        snapshot.semantics_receipt,
        *snapshot.split_artifacts.values(),
    ):
        current = read_regular_bytes_artifact_v2(
            artifact.path,
            label=artifact.label,
        )
        if current != artifact.content:
            raise RuntimeError(f"primary-v2 input changed: {artifact.label}")
    for identity in sorted(snapshot.raw_artifacts):
        artifact = snapshot.raw_artifacts[identity]
        current = read_regular_bytes_artifact_v2(
            artifact.path,
            label=f"primary-v2 authorized raw input {artifact.name}",
        )
        assert current is not None
        if (
            len(current) != artifact.size
            or hashlib.sha256(current).hexdigest() != artifact.sha256
        ):
            raise RuntimeError(
                f"primary-v2 authorized raw input changed: {artifact.name}"
            )


def build_primary_semantics_receipt_v2(
    manifest: Mapping[str, object],
    data_dir: Path,
) -> dict[str, object]:
    """Authenticate and summarize set canonicalization for all 132 files."""

    if primary_manifest_retained_sha256_v2(manifest) != EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256:
        raise RuntimeError("primary semantics manifest binding differs")
    rows = manifest.get("files")
    if type(rows) is not list or len(rows) != 132:
        raise RuntimeError("primary semantics manifest rows differ")

    files: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []
    affected_files: set[str] = set()
    ballots = 0
    raw_tokens = 0
    canonical_tokens = 0
    empty_ballots = 0
    unknown_project_tokens = 0
    identities: set[tuple[str, int, str]] = set()
    data = Path(data_dir).absolute()
    for row in rows:
        if type(row) is not dict:
            raise RuntimeError("primary semantics manifest row is malformed")
        name = row.get("name")
        series = row.get("series")
        year = row.get("year")
        expected_sha256 = row.get("sha256")
        if (
            type(name) is not str
            or type(series) is not str
            or type(year) is not int
            or not _is_sha256(expected_sha256)
        ):
            raise RuntimeError("primary semantics manifest identity is malformed")
        identity = (series, year, name)
        if identity in identities:
            raise RuntimeError("primary semantics manifest contains a duplicate identity")
        identities.add(identity)

        path = data / name
        before = _read_regular(path, label=f"primary semantics source {name}")
        if hashlib.sha256(before).hexdigest() != expected_sha256:
            raise RuntimeError(f"primary raw digest differs before parse: {name}")
        raw_instance = parse_pb_file(path)
        normalized = canonicalize_instance(raw_instance, source_id=name)
        after = _read_regular(path, label=f"primary semantics source {name}")
        if after != before:
            raise RuntimeError(f"primary raw election changed during parse: {name}")

        ballots += len(raw_instance.votes)
        raw_tokens += sum(len(vote.projects) for vote in raw_instance.votes)
        canonical_tokens += sum(
            len(vote.projects) for vote in normalized.instance.votes
        )
        empty_ballots += sum(not vote.projects for vote in normalized.instance.votes)
        known_projects = set(raw_instance.projects)
        unknown_project_tokens += sum(
            project not in known_projects
            for vote in raw_instance.votes
            for project in vote.projects
        )
        raw_semantic_sha256 = canonical_instance_sha256(raw_instance)
        v2_semantic_sha256 = canonical_instance_sha256(normalized.instance)
        changed = raw_semantic_sha256 != v2_semantic_sha256
        if changed:
            affected_files.add(name)
        files.append(
            {
                "series": series,
                "year": year,
                "name": name,
                "raw_file_sha256": expected_sha256,
                "raw_semantic_sha256": raw_semantic_sha256,
                "v2_semantic_sha256": v2_semantic_sha256,
                "changed": changed,
            }
        )
        for anomaly in normalized.anomalies:
            anomalies.append(
                {
                    "series": series,
                    "year": year,
                    **{
                        key: list(value) if type(value) is tuple else value
                        for key, value in asdict(anomaly).items()
                    },
                }
            )

    files.sort(key=lambda row: (str(row["series"]), int(row["year"]), str(row["name"])))
    anomalies.sort(
        key=lambda row: (
            str(row["series"]),
            int(row["year"]),
            str(row["source_id"]),
            int(row["vote_index_zero_based"]),
        )
    )
    counts = {
        "series": len({str(row["series"]) for row in files}),
        "files": len(files),
        "ballots": ballots,
        "raw_approval_tokens": raw_tokens,
        "canonical_approval_tokens": canonical_tokens,
        "affected_files": len(affected_files),
        "affected_ballots": len(anomalies),
        "removed_tokens": raw_tokens - canonical_tokens,
        "empty_ballots": empty_ballots,
        "unknown_project_tokens": unknown_project_tokens,
    }
    receipt: dict[str, object] = {
        "schema_version": 2,
        "semantics_profile": SEMANTICS_PROFILE,
        "parent_manifest_retained_sha256": primary_manifest_retained_sha256_v2(manifest),
        "source_commit": manifest.get("source_commit"),
        "counts": counts,
        "files": files,
        "anomalies": anomalies,
        "corpus_semantic_sha256": _canonical_json_sha256(
            {"semantics_profile": SEMANTICS_PROFILE, "files": files}
        ),
    }
    return validate_primary_semantics_receipt_v2(receipt)


def validate_primary_semantics_receipt_v2(
    receipt: dict[str, object],
) -> dict[str, object]:
    """Require the exact primary corpus semantic transition and JSON types."""

    expected_fields = {
        "schema_version",
        "semantics_profile",
        "parent_manifest_retained_sha256",
        "source_commit",
        "counts",
        "files",
        "anomalies",
        "corpus_semantic_sha256",
    }
    if type(receipt) is not dict or set(receipt) != expected_fields:
        raise RuntimeError("primary semantics receipt schema differs")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 2
        or receipt.get("semantics_profile") != SEMANTICS_PROFILE
        or receipt.get("source_commit") != PABULIB_COMMIT
        or receipt.get("parent_manifest_retained_sha256")
        != EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256
    ):
        raise RuntimeError("primary semantics receipt identity differs")
    counts = receipt.get("counts")
    if (
        type(counts) is not dict
        or set(counts) != set(_EXPECTED_COUNTS)
        or any(type(value) is not int for value in counts.values())
        or counts != _EXPECTED_COUNTS
    ):
        raise RuntimeError("primary semantics receipt counts differ")
    if not exact_json_equal_v2(receipt.get("anomalies"), [_EXPECTED_WOLA_ANOMALY]):
        raise RuntimeError("primary semantics receipt anomaly inventory differs")

    files = receipt.get("files")
    file_fields = {
        "series",
        "year",
        "name",
        "raw_file_sha256",
        "raw_semantic_sha256",
        "v2_semantic_sha256",
        "changed",
    }
    if type(files) is not list or len(files) != 132:
        raise RuntimeError("primary semantics receipt file inventory differs")
    identities: list[tuple[str, int, str]] = []
    changed_rows: list[dict[str, object]] = []
    for row in files:
        if type(row) is not dict or set(row) != file_fields:
            raise RuntimeError("primary semantics receipt file row differs")
        if (
            type(row.get("series")) is not str
            or type(row.get("year")) is not int
            or type(row.get("name")) is not str
            or type(row.get("changed")) is not bool
            or any(
                not _is_sha256(row.get(field))
                for field in (
                    "raw_file_sha256",
                    "raw_semantic_sha256",
                    "v2_semantic_sha256",
                )
            )
        ):
            raise RuntimeError("primary semantics receipt file row types differ")
        identities.append((row["series"], row["year"], row["name"]))
        expected_changed = row["raw_semantic_sha256"] != row["v2_semantic_sha256"]
        if row["changed"] != expected_changed:
            raise RuntimeError("primary semantics receipt change flag differs")
        if row["changed"]:
            changed_rows.append(row)
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise RuntimeError("primary semantics receipt identities are not canonical")
    if not exact_json_equal_v2(changed_rows, [_EXPECTED_WOLA_FILE_ROW]):
        raise RuntimeError("primary semantics receipt changed-file inventory differs")
    expected_digest = _canonical_json_sha256(
        {"semantics_profile": SEMANTICS_PROFILE, "files": files}
    )
    if (
        receipt.get("corpus_semantic_sha256") != expected_digest
        or expected_digest != EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
    ):
        raise RuntimeError("primary semantics receipt corpus digest differs")
    return receipt


def primary_structural_source_sha256_v2(
    repo_root: Path = ROOT,
) -> dict[str, str]:
    """Hash the complete source closure used by the primary structural gate."""

    root = Path(repo_root).absolute()
    hashes: dict[str, str] = {}
    for name in PRIMARY_STRUCTURAL_SOURCE_FILES_V2:
        path = root / "src" / name
        content = read_regular_bytes_artifact_v2(
            path,
            label=f"primary-v2 structural source {name}",
        )
        assert content is not None
        hashes[f"src/{name}"] = hashlib.sha256(content).hexdigest()
    if (
        not hashes
        or any(
            type(name) is not str
            or not name.startswith("src/")
            or not _is_sha256(digest)
            for name, digest in hashes.items()
        )
    ):
        raise RuntimeError("primary-v2 structural source closure is malformed")
    return dict(sorted(hashes.items()))


def _verify_primary_priority_mes_corpus_v2(
    refs: Sequence[SeriesRef],
    *,
    semantics_receipt: Mapping[str, object],
    semantics_receipt_sha256: str,
    corpus_semantic_sha256: str,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    """Run the receipt-authorized full-corpus solver replay."""

    if not _is_sha256(semantics_receipt_sha256):
        raise RuntimeError("primary-v2 semantics receipt digest is malformed")
    if not _is_sha256(corpus_semantic_sha256):
        raise RuntimeError("primary-v2 corpus semantic digest is malformed")
    if not isinstance(refs, Sequence) or any(
        type(ref) is not SeriesRef for ref in refs
    ):
        raise RuntimeError("primary-v2 structural reference inventory is malformed")

    source_sha256 = primary_structural_source_sha256_v2(repo_root)
    zero = np.zeros(N_PROJECT_FEATURES, dtype=float)
    n_elections = 0
    approval_ballots_checked = 0
    duplicate_tokens_remaining = 0
    winner_identity_checks = 0
    determinism_checks = 0
    budget_checks = 0
    unique_payer_round_checks = 0
    payment_rounds_checked = 0

    for ref in refs:
        instances = load_series_authenticated_v2(ref, semantics_receipt)
        if tuple(instances) != ref.years:
            raise RuntimeError(
                f"primary-v2 structural years differ for {ref.key}"
            )
        for year in ref.years:
            instance = instances[year]
            n_elections += 1
            approval_ballots_checked += len(instance.votes)
            duplicate_tokens_remaining += sum(
                len(vote.projects) - len(set(vote.projects))
                for vote in instance.votes
            )
            for completion in (False, True):
                expected = mes_with_endowments(instance, completion=completion)
                first = priority_mes_outcome(
                    instance,
                    RolloutState(),
                    zero,
                    completion=completion,
                )
                second = priority_mes_outcome(
                    instance,
                    RolloutState(),
                    zero,
                    completion=completion,
                )
                winner_identity_checks += 1
                if set(first.winners) != expected:
                    raise RuntimeError(
                        "primary-v2 MES containment failed for "
                        f"{ref.key} {year} completion={completion}"
                    )
                determinism_checks += 1
                if first != second:
                    raise RuntimeError(
                        "primary-v2 determinism failed for "
                        f"{ref.key} {year} completion={completion}"
                    )
                spent = sum(
                    instance.projects[project_id].cost
                    for project_id in first.winners
                )
                budget_checks += 1
                if spent > instance.budget + 1e-6:
                    raise RuntimeError(
                        f"primary-v2 budget feasibility failed for {ref.key} "
                        f"{year}: {spent}"
                    )
                for round_ in first.rounds:
                    unique_payer_round_checks += 1
                    if len(round_.supporter_ids) != len(set(round_.supporter_ids)):
                        raise RuntimeError(
                            "primary-v2 duplicate payer in "
                            f"{ref.key} {year} project {round_.project_id}"
                        )
                payment_rounds_checked += len(first.rounds)

    if duplicate_tokens_remaining:
        raise RuntimeError(
            "primary-v2 ingestion retained "
            f"{duplicate_tokens_remaining} duplicate approvals"
        )
    if unique_payer_round_checks != payment_rounds_checked:
        raise RuntimeError("primary-v2 structural payment-round accounting differs")
    return {
        "schema_version": 2,
        "status": "pass",
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_receipt_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "source_sha256": source_sha256,
        "n_series": len(refs),
        "n_elections": n_elections,
        "approval_ballots_checked": approval_ballots_checked,
        "duplicate_tokens_remaining": duplicate_tokens_remaining,
        "winner_identity_checks": winner_identity_checks,
        "determinism_checks": determinism_checks,
        "budget_checks": budget_checks,
        "unique_payer_round_checks": unique_payer_round_checks,
        "payment_rounds_checked": payment_rounds_checked,
        "completion_modes": [False, True],
        "zero_weights": zero.tolist(),
    }


def validate_structural_gates_v2(
    path: Path,
    *,
    semantics_receipt_sha256: str,
    corpus_semantic_sha256: str,
    repo_root: Path = ROOT,
) -> dict[str, object]:
    """Authenticate the exact pre-fit metadata/semantics structural gate."""

    if not _is_sha256(semantics_receipt_sha256):
        raise RuntimeError("primary-v2 semantics receipt digest is malformed")
    if (
        not _is_sha256(corpus_semantic_sha256)
        or corpus_semantic_sha256 != EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
    ):
        raise RuntimeError("primary-v2 corpus semantic digest differs")
    payload = load_json_object_strict_v2(
        Path(path),
        label="primary-v2 structural gate",
    )
    fields = {
        "schema_version",
        "status",
        "gate_scope",
        "semantics_profile",
        "semantics_receipt_sha256",
        "corpus_semantic_sha256",
        "source_sha256",
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
    }
    if set(payload) != fields:
        raise RuntimeError("primary-v2 structural gate fields differ")
    integer_fields = (
        "n_series",
        "n_elections",
        "manifest_file_bindings",
        "approval_ballots_authenticated",
        "duplicate_tokens_remaining",
        "split_artifacts_authenticated",
        "raw_election_files_opened",
        "election_parser_calls",
        "outcome_solver_calls",
    )
    if any(type(payload.get(field)) is not int for field in integer_fields):
        raise RuntimeError("primary-v2 structural gate integer field differs")
    fixed = {
        "schema_version": 2,
        "status": "pass",
        "gate_scope": PREFIT_STRUCTURAL_GATE_SCOPE_V2,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_receipt_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "source_sha256": primary_structural_source_sha256_v2(repo_root),
        "n_series": 19,
        "n_elections": 132,
        "manifest_file_bindings": 132,
        "approval_ballots_authenticated": 821_572,
        "duplicate_tokens_remaining": 0,
        "split_artifacts_authenticated": len(SPLIT_SHA256_V2),
        "raw_election_files_opened": 0,
        "election_parser_calls": 0,
        "outcome_solver_calls": 0,
        "heldout_outcomes_scored": False,
    }
    for field, value in fixed.items():
        if not exact_json_equal_v2(payload.get(field), value):
            raise RuntimeError(
                f"primary-v2 structural gate field {field!r} differs"
            )
    if type(payload.get("heldout_outcomes_scored")) is not bool:
        raise RuntimeError("primary-v2 structural held-out flag type differs")
    return payload


def build_primary_structural_report_v2(
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Authenticate pre-fit metadata without opening or solving elections."""

    root = Path(repo_root).absolute()
    result = Path(result_root).absolute()
    data = Path(data_dir).absolute()
    manifest = _locked_artifact_snapshot_v2(
        result / "corpus_manifest.json",
        "artifact/corpus_manifest",
        None,
    )
    receipt = _locked_artifact_snapshot_v2(
        result / "approval_semantics_receipt.json",
        "artifact/approval_semantics_receipt",
        None,
    )
    validate_primary_semantics_receipt_v2(receipt.payload)
    index, raw_metadata = _primary_index_metadata_from_snapshot_v2(
        manifest,
        data,
        enforce_expected_counts=True,
    )
    _validate_manifest_receipt_identity_v2(
        index,
        raw_metadata,
        receipt.payload,
    )
    if receipt.payload.get("corpus_semantic_sha256") != (
        EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
    ):
        raise RuntimeError("primary-v2 structural semantic commitment differs")
    split_artifacts = {
        name: _locked_artifact_snapshot_v2(
            result / "splits" / f"{name}.json",
            f"split/{name}",
            None,
        )
        for name in sorted(SPLIT_SHA256_V2)
    }
    for name, artifact in split_artifacts.items():
        if artifact.sha256 != SPLIT_SHA256_V2[name]:
            raise RuntimeError(f"primary-v2 inherited split digest differs: {name}")
        _split_views_from_snapshot_v2(
            artifact,
            index,
            expected_name=name,
        )
    counts = receipt.payload.get("counts")
    if type(counts) is not dict:
        raise RuntimeError("primary-v2 structural receipt counts differ")
    source_sha256 = primary_structural_source_sha256_v2(root)
    return {
        "schema_version": 2,
        "status": "pass",
        "gate_scope": PREFIT_STRUCTURAL_GATE_SCOPE_V2,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": receipt.sha256,
        "corpus_semantic_sha256": EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256,
        "source_sha256": source_sha256,
        "n_series": len(index),
        "n_elections": len(raw_metadata),
        "manifest_file_bindings": len(raw_metadata),
        "approval_ballots_authenticated": counts.get("ballots"),
        "duplicate_tokens_remaining": 0,
        "split_artifacts_authenticated": len(split_artifacts),
        "raw_election_files_opened": 0,
        "election_parser_calls": 0,
        "outcome_solver_calls": 0,
        "heldout_outcomes_scored": False,
    }


def build_postreceipt_structural_replay_v2(
    input_snapshot: PrimaryInputsSnapshotV2,
    lock_snapshot: PrimaryProtocolLockSnapshotV2,
) -> dict[str, object]:
    """Run solver checks only from a receipt-authorized full-corpus snapshot."""

    validate_protocol_inputs_snapshot_v2(input_snapshot, lock_snapshot)
    if (
        input_snapshot.access_scope != EVALUATION_ACCESS_SCOPE_V2
        or type(input_snapshot.evaluation_replay_receipt)
        is not JsonArtifactSnapshotV2
    ):
        raise RuntimeError(
            "primary-v2 post-receipt structural replay lacks evaluation authorization"
        )
    _, raw_metadata = _primary_index_metadata_from_snapshot_v2(
        input_snapshot.manifest,
        input_snapshot.data_dir,
        enforce_expected_counts=True,
    )
    if set(input_snapshot.raw_artifacts) != set(raw_metadata):
        raise RuntimeError(
            "primary-v2 post-receipt structural replay requires the full corpus"
        )
    assert_protocol_inputs_unchanged_v2(input_snapshot, lock_snapshot)
    repo_root = input_snapshot.result_root.parents[1]
    report = _verify_primary_priority_mes_corpus_v2(
        tuple(input_snapshot.index[key] for key in sorted(input_snapshot.index)),
        semantics_receipt=input_snapshot.semantics_receipt.payload,
        semantics_receipt_sha256=input_snapshot.semantics_receipt.sha256,
        corpus_semantic_sha256=str(
            input_snapshot.semantics_receipt.payload.get(
                "corpus_semantic_sha256", ""
            )
        ),
        repo_root=repo_root,
    )
    assert_protocol_inputs_unchanged_v2(input_snapshot, lock_snapshot)
    return {
        **report,
        "gate_scope": POSTRECEIPT_STRUCTURAL_GATE_SCOPE_V2,
        "evaluation_replay_receipt_sha256": (
            input_snapshot.evaluation_replay_receipt.sha256
        ),
    }


def write_primary_structural_gates_v2(
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Double-build, preflight, atomically install, and revalidate the gate."""

    path = Path(result_root).absolute() / "structural_gates.json"
    first = build_primary_structural_report_v2(repo_root, result_root, data_dir)
    content = _canonical_json_bytes_v2(first)
    preflight_immutable_bytes_artifact_v2(
        path,
        content,
        label="primary-v2 structural gate",
    )
    second = build_primary_structural_report_v2(repo_root, result_root, data_dir)
    second_content = _canonical_json_bytes_v2(second)
    if second_content != content:
        raise RuntimeError("primary-v2 structural gate is not deterministic")
    preflight_immutable_bytes_artifact_v2(
        path,
        content,
        label="primary-v2 structural gate",
    )
    digest = write_immutable_bytes_artifact_v2(
        path,
        content,
        label="primary-v2 structural gate",
    )
    validate_structural_gates_v2(
        path,
        semantics_receipt_sha256=str(first.get("semantics_receipt_sha256", "")),
        corpus_semantic_sha256=str(first.get("corpus_semantic_sha256", "")),
        repo_root=repo_root,
    )
    installed = read_regular_bytes_artifact_v2(
        path,
        label="primary-v2 structural gate",
    )
    if installed != content or digest != hashlib.sha256(content).hexdigest():
        raise RuntimeError("primary-v2 structural gate changed after installation")
    return {
        "status": "pass",
        "path": str(path),
        "sha256": digest,
        "gate_scope": first.get("gate_scope"),
    }


def protocol_config_v2() -> dict[str, object]:
    """Return the result-independent primary replay contract."""

    return {
        "classification": "corrected post-hoc replay",
        "heldout_outcomes_already_known": True,
        "semantics_profile": SEMANTICS_PROFILE,
        "fit_count": 49,
        "optimizer": "CMA-ES",
        "execution_mode": "serial within each fit",
        "completion": "approval_count",
        "training_grid": dict(TRAINING_GRID_SPEC_V2),
        "evaluation_boundary": (
            "evaluation starts only after all 49 fits and the training grid "
            "authenticate; no fresh-holdout or preregistration claim"
        ),
    }


def _tracked_file_rows_v2(
    repo_root: Path,
    mandatory_files: Mapping[str, Path],
) -> dict[str, dict[str, str]]:
    root = Path(repo_root).absolute()
    if not mandatory_files:
        raise RuntimeError("primary-v2 mandatory file closure is empty")
    tracked: dict[str, dict[str, str]] = {}
    for label, source in sorted(mandatory_files.items()):
        if type(label) is not str or not label or label.startswith("/"):
            raise RuntimeError("primary-v2 mandatory file label is malformed")
        path = Path(source).absolute()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"primary-v2 mandatory file escapes repository: {path}"
            ) from exc
        if ".." in relative.parts:
            raise RuntimeError("primary-v2 mandatory file path traverses parents")
        content = read_regular_bytes_artifact_v2(
            path,
            label=f"primary-v2 mandatory file {label}",
        )
        assert content is not None
        tracked[label] = {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    return tracked


def _locked_corpus_semantic_sha256_v2(
    mandatory_files: Mapping[str, Path],
    tracked_files: Mapping[str, object],
) -> str:
    """Read the tracked receipt and extract its exact corpus commitment."""

    label = "artifact/approval_semantics_receipt"
    path = mandatory_files.get(label)
    row = tracked_files.get(label)
    tracked_digest = row.get("sha256") if isinstance(row, Mapping) else None
    if path is None or not _is_sha256(tracked_digest):
        raise RuntimeError("primary-v2 lock is missing the semantics receipt")
    content = read_regular_bytes_artifact_v2(
        Path(path),
        label="primary-v2 lock semantics receipt",
    )
    assert content is not None
    if hashlib.sha256(content).hexdigest() != tracked_digest:
        raise RuntimeError("primary-v2 semantics receipt changed during lock build")
    receipt = load_json_object_bytes_strict_v2(
        content,
        label="primary-v2 lock semantics receipt",
    )
    semantic = receipt.get("corpus_semantic_sha256")
    if (
        type(semantic) is not str
        or semantic != EXPECTED_PRIMARY_CORPUS_SEMANTIC_SHA256
    ):
        raise RuntimeError("primary-v2 lock corpus semantic digest differs")
    return semantic


def build_protocol_lock_payload_v2(
    repo_root: Path,
    mandatory_files: Mapping[str, Path],
) -> dict[str, object]:
    """Build the immutable base lock without hashing the future grid output."""

    tracked = _tracked_file_rows_v2(repo_root, mandatory_files)
    corpus_semantic_sha256 = _locked_corpus_semantic_sha256_v2(
        mandatory_files,
        tracked,
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "lock_profile": LOCK_PROFILE_V2,
        "mandatory_file_count": len(tracked),
        "tracked_files": tracked,
        "fit_inventory": locked_fit_inventory_v2(),
        "training_grid_spec": dict(TRAINING_GRID_SPEC_V2),
        "protocol": protocol_config_v2(),
        "corpus_semantic_sha256": corpus_semantic_sha256,
    }
    validate_protocol_lock_payload_v2(payload, repo_root, mandatory_files)
    return payload


def validate_protocol_lock_payload_v2(
    payload: Mapping[str, object],
    repo_root: Path,
    mandatory_files: Mapping[str, Path],
) -> Mapping[str, object]:
    """Rehash the complete base-lock closure and reject any drift."""

    fields = {
        "schema_version",
        "lock_profile",
        "mandatory_file_count",
        "tracked_files",
        "fit_inventory",
        "training_grid_spec",
        "protocol",
        "corpus_semantic_sha256",
    }
    if type(payload) is not dict or set(payload) != fields:
        raise RuntimeError("primary-v2 protocol lock schema differs")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("lock_profile") != LOCK_PROFILE_V2
        or type(payload.get("mandatory_file_count")) is not int
        or payload.get("mandatory_file_count") != len(mandatory_files)
    ):
        raise RuntimeError("primary-v2 protocol lock identity or count differs")
    inventory = payload.get("fit_inventory")
    validate_locked_fit_inventory_v2(inventory)
    if not exact_json_equal_v2(payload.get("training_grid_spec"), TRAINING_GRID_SPEC_V2):
        raise RuntimeError("primary-v2 training grid protocol differs")
    if not exact_json_equal_v2(payload.get("protocol"), protocol_config_v2()):
        raise RuntimeError("primary-v2 protocol configuration differs")

    tracked = payload.get("tracked_files")
    if type(tracked) is not dict or set(tracked) != set(mandatory_files):
        raise RuntimeError("primary-v2 tracked-file closure differs")
    current = _tracked_file_rows_v2(repo_root, mandatory_files)
    if not exact_json_equal_v2(tracked, current):
        raise RuntimeError("primary-v2 tracked-file digest drift detected")
    current_semantic = _locked_corpus_semantic_sha256_v2(
        mandatory_files,
        current,
    )
    if (
        type(payload.get("corpus_semantic_sha256")) is not str
        or payload.get("corpus_semantic_sha256") != current_semantic
    ):
        raise RuntimeError("primary-v2 lock corpus semantic commitment differs")
    _require_finite_json(payload, label="primary-v2 protocol lock")
    return payload


def validate_protocol_lock_payload_metadata_v2(
    payload: Mapping[str, object],
    repo_root: Path,
    mandatory_files: Mapping[str, Path],
) -> Mapping[str, object]:
    """Verify the lock closure without opening any tracked raw election file."""

    fields = {
        "schema_version",
        "lock_profile",
        "mandatory_file_count",
        "tracked_files",
        "fit_inventory",
        "training_grid_spec",
        "protocol",
        "corpus_semantic_sha256",
    }
    if type(payload) is not dict or set(payload) != fields:
        raise RuntimeError("primary-v2 protocol lock schema differs")
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("lock_profile") != LOCK_PROFILE_V2
        or type(payload.get("mandatory_file_count")) is not int
        or payload.get("mandatory_file_count") != len(mandatory_files)
    ):
        raise RuntimeError("primary-v2 protocol lock identity or count differs")
    validate_locked_fit_inventory_v2(payload.get("fit_inventory"))
    if not exact_json_equal_v2(payload.get("training_grid_spec"), TRAINING_GRID_SPEC_V2):
        raise RuntimeError("primary-v2 training grid protocol differs")
    if not exact_json_equal_v2(payload.get("protocol"), protocol_config_v2()):
        raise RuntimeError("primary-v2 protocol configuration differs")

    tracked = payload.get("tracked_files")
    if type(tracked) is not dict or set(tracked) != set(mandatory_files):
        raise RuntimeError("primary-v2 tracked-file closure differs")
    root = Path(repo_root).absolute()
    authenticated_content: dict[str, bytes] = {}
    for label, supplied_path in sorted(mandatory_files.items()):
        row = tracked.get(label)
        path = Path(supplied_path).absolute()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"primary-v2 mandatory file escapes repository: {path}"
            ) from exc
        if (
            type(row) is not dict
            or set(row) != {"path", "sha256"}
            or row.get("path") != relative
            or not _is_sha256(row.get("sha256"))
        ):
            raise RuntimeError(f"primary-v2 tracked-file row differs: {label}")
        if label.startswith("data/"):
            continue
        content = read_regular_bytes_artifact_v2(
            path,
            label=f"primary-v2 mandatory metadata/source file {label}",
        )
        assert content is not None
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise RuntimeError("primary-v2 tracked-file digest drift detected")
        authenticated_content[label] = content

    manifest_content = authenticated_content.get("artifact/corpus_manifest")
    manifest_path = mandatory_files.get("artifact/corpus_manifest")
    if manifest_content is None or manifest_path is None:
        raise RuntimeError("primary-v2 lock is missing the corpus manifest")
    manifest_payload = load_json_object_bytes_strict_v2(
        manifest_content,
        label="primary-v2 metadata-only lock corpus manifest",
    )
    manifest_snapshot = JsonArtifactSnapshotV2(
        label="artifact/corpus_manifest",
        path=Path(manifest_path).absolute(),
        content=manifest_content,
        sha256=hashlib.sha256(manifest_content).hexdigest(),
        payload=manifest_payload,
    )
    _, raw_metadata = _primary_index_metadata_from_snapshot_v2(
        manifest_snapshot,
        Path(next(
            (
                path.parent
                for label, path in mandatory_files.items()
                if label.startswith("data/")
            ),
            root / "data" / "pb",
        )),
        enforce_expected_counts=False,
    )
    expected_data = {
        f"data/{artifact.name}": artifact
        for artifact in raw_metadata.values()
    }
    observed_data_labels = {
        label for label in mandatory_files if label.startswith("data/")
    }
    if observed_data_labels != set(expected_data):
        raise RuntimeError("primary-v2 raw-data lock inventory differs from manifest")
    for label, artifact in expected_data.items():
        row = tracked[label]
        if (
            Path(mandatory_files[label]).absolute() != artifact.path
            or row["sha256"] != artifact.sha256
        ):
            raise RuntimeError(
                "primary-v2 raw-data lock identity differs from manifest metadata"
            )

    current_semantic = _locked_corpus_semantic_sha256_v2(
        mandatory_files,
        tracked,
    )
    if (
        type(payload.get("corpus_semantic_sha256")) is not str
        or payload.get("corpus_semantic_sha256") != current_semantic
    ):
        raise RuntimeError("primary-v2 lock corpus semantic commitment differs")
    _require_finite_json(payload, label="primary-v2 protocol lock")
    return payload


_PRIMARY_SOURCE_FILES_V2 = (
    "cohorts.py",
    "iclr_approval_semantics_v2.py",
    "iclr_cmaes.py",
    "iclr_corpus.py",
    "iclr_env.py",
    "iclr_multicity_protocol.py",
    "iclr_multicity_protocol_v2.py",
    "iclr_outcome.py",
    "iclr_policy.py",
    "iclr_primary_warsaw_delta_audit_v2.py",
    "iclr_primary_warsaw_evaluate_v2.py",
    "iclr_primary_warsaw_protocol_v2.py",
    "iclr_primary_warsaw_train_v2.py",
    "iclr_priority_mes.py",
    "iclr_stats.py",
    "iclr_train.py",
    "parse_pb.py",
    "rules.py",
    "run_experiments.py",
)
_PRIMARY_TEST_FILES_V2 = (
    "test_iclr_primary_warsaw_delta_audit_v2.py",
    "test_iclr_primary_warsaw_evaluate_v2.py",
    "test_iclr_primary_warsaw_protocol_v2.py",
    "test_iclr_primary_warsaw_train_v2.py",
)


def mandatory_protocol_files_v2(
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, Path]:
    """Resolve the complete source, test, input, and raw-data lock closure."""

    root = Path(repo_root).absolute()
    result = Path(result_root).absolute()
    data = Path(data_dir).absolute()
    files: dict[str, Path] = {
        "artifact/corpus_manifest": result / "corpus_manifest.json",
        "artifact/approval_semantics_receipt": result
        / "approval_semantics_receipt.json",
        "artifact/structural_gates": result / "structural_gates.json",
        "parent/frozen_instances": root / "results" / "frozen_instances.txt",
        "parent/multicity_v2_manifest": root
        / "results"
        / "iclr_multicity_v2"
        / "corpus_manifest.json",
        "parent/multicity_v2_semantics_receipt": root
        / "results"
        / "iclr_multicity_v2"
        / "approval_semantics_receipt.json",
        "parent/multicity_v2_protocol_lock": root
        / "results"
        / "iclr_multicity_v2"
        / "protocol_lock.json",
        "environment/pyproject": root / "pyproject.toml",
        "environment/uv_lock": root / "uv.lock",
    }
    files.update(
        {
            f"split/{name}": result / "splits" / f"{name}.json"
            for name in SPLIT_SHA256_V2
        }
    )
    files.update(
        {f"source/{name}": root / "src" / name for name in _PRIMARY_SOURCE_FILES_V2}
    )
    files.update(
        {f"test/{name}": root / "tests" / name for name in _PRIMARY_TEST_FILES_V2}
    )

    manifest = load_json_object_strict_v2(
        result / "corpus_manifest.json",
        label="primary-v2 corpus manifest",
    )
    if primary_manifest_retained_sha256_v2(manifest) != EXPECTED_PRIMARY_MANIFEST_RETAINED_SHA256:
        raise RuntimeError("primary-v2 mandatory manifest digest differs")
    rows = manifest.get("files")
    if type(rows) is not list or len(rows) != 132:
        raise RuntimeError("primary-v2 mandatory raw-data inventory differs")
    for row in rows:
        if type(row) is not dict or type(row.get("name")) is not str:
            raise RuntimeError("primary-v2 mandatory raw-data row is malformed")
        name = row["name"]
        files[f"data/{name}"] = data / name
    receipt_path = result / "approval_semantics_receipt.json"
    receipt_content = read_regular_bytes_artifact_v2(
        receipt_path,
        label="primary-v2 approval semantics receipt",
    )
    assert receipt_content is not None
    receipt = load_json_object_bytes_strict_v2(
        receipt_content,
        label="primary-v2 approval semantics receipt",
    )
    validate_primary_semantics_receipt_v2(receipt)
    if receipt.get("parent_manifest_retained_sha256") != (
        primary_manifest_retained_sha256_v2(manifest)
    ):
        raise RuntimeError("primary-v2 manifest and semantics receipt differ")
    validate_structural_gates_v2(
        result / "structural_gates.json",
        semantics_receipt_sha256=hashlib.sha256(receipt_content).hexdigest(),
        corpus_semantic_sha256=str(receipt.get("corpus_semantic_sha256")),
        repo_root=root,
    )
    return files


def verify_primary_warsaw_protocol_lock_snapshot_v2(
    lock_path: Path,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> PrimaryProtocolLockSnapshotV2:
    """Verify locked metadata/source closure without opening raw elections."""

    path = Path(lock_path)
    expected = Path(result_root) / "protocol_lock.json"
    if path.absolute() != expected.absolute():
        raise RuntimeError("primary-v2 lock path is not canonical")
    content = read_regular_bytes_artifact_v2(path, label="primary-v2 protocol lock")
    assert content is not None
    payload = load_json_object_bytes_strict_v2(
        content,
        label="primary-v2 protocol lock",
    )
    mandatory = mandatory_protocol_files_v2(repo_root, result_root, data_dir)
    validate_protocol_lock_payload_metadata_v2(payload, repo_root, mandatory)
    if read_regular_bytes_artifact_v2(
        path,
        label="primary-v2 protocol lock",
    ) != content:
        raise RuntimeError("primary-v2 protocol lock changed during verification")
    return PrimaryProtocolLockSnapshotV2(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
    )


def verify_primary_warsaw_full_protocol_lock_snapshot_v2(
    lock_path: Path,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> PrimaryProtocolLockSnapshotV2:
    """Rehash every locked file, including all raw elections."""

    path = Path(lock_path)
    expected = Path(result_root) / "protocol_lock.json"
    if path.absolute() != expected.absolute():
        raise RuntimeError("primary-v2 lock path is not canonical")
    content = read_regular_bytes_artifact_v2(path, label="primary-v2 protocol lock")
    assert content is not None
    payload = load_json_object_bytes_strict_v2(
        content,
        label="primary-v2 protocol lock",
    )
    mandatory = mandatory_protocol_files_v2(repo_root, result_root, data_dir)
    validate_protocol_lock_payload_v2(payload, repo_root, mandatory)
    if read_regular_bytes_artifact_v2(
        path,
        label="primary-v2 protocol lock",
    ) != content:
        raise RuntimeError("primary-v2 protocol lock changed during verification")
    return PrimaryProtocolLockSnapshotV2(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        payload=payload,
    )


def write_primary_protocol_lock_v2(
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> dict[str, object]:
    """Double-build and immutably install the complete primary base lock."""

    root = Path(repo_root).absolute()
    result = Path(result_root).absolute()
    data = Path(data_dir).absolute()
    path = result / "protocol_lock.json"
    first_files = mandatory_protocol_files_v2(root, result, data)
    first = build_protocol_lock_payload_v2(root, first_files)
    content = _canonical_json_bytes_v2(first)
    preflight_immutable_bytes_artifact_v2(
        path,
        content,
        label="primary-v2 protocol lock",
    )
    second_files = mandatory_protocol_files_v2(root, result, data)
    second = build_protocol_lock_payload_v2(root, second_files)
    second_content = _canonical_json_bytes_v2(second)
    if second_content != content:
        raise RuntimeError("primary-v2 protocol closure changed during lock build")
    preflight_immutable_bytes_artifact_v2(
        path,
        content,
        label="primary-v2 protocol lock",
    )
    digest = write_immutable_bytes_artifact_v2(
        path,
        content,
        label="primary-v2 protocol lock",
    )
    snapshot = verify_primary_warsaw_full_protocol_lock_snapshot_v2(
        path,
        root,
        result,
        data,
    )
    if (
        snapshot.content != content
        or snapshot.sha256 != digest
        or not exact_json_equal_v2(snapshot.payload, first)
    ):
        raise RuntimeError("primary-v2 protocol lock changed after installation")
    return {
        "status": "locked",
        "path": str(path),
        "sha256": digest,
        "mandatory_file_count": len(first_files),
        "fit_count": len(first["fit_inventory"]),
    }


def _add_path_arguments_v2(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT_V2)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "pb")


def main(argv: Sequence[str] | None = None) -> None:
    """Run one ordered primary-v2 staging or lock action."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="stage immutable primary inputs")
    _add_path_arguments_v2(stage)
    stage.add_argument(
        "--multicity-manifest",
        type=Path,
        default=ROOT / "results" / "iclr_multicity_v2" / "corpus_manifest.json",
    )
    stage.add_argument(
        "--frozen-list",
        type=Path,
        default=ROOT / "results" / "frozen_instances.txt",
    )
    stage.add_argument(
        "--split-source-dir",
        type=Path,
        default=ROOT / "results" / "iclr_splits",
    )

    structural = subparsers.add_parser(
        "structural-gate",
        help="run and immutably install the primary structural gate",
    )
    _add_path_arguments_v2(structural)
    lock = subparsers.add_parser("lock", help="create the immutable base lock")
    _add_path_arguments_v2(lock)
    verify = subparsers.add_parser(
        "verify-lock",
        help="rehash and verify the complete immutable base lock",
    )
    _add_path_arguments_v2(verify)
    args = parser.parse_args(argv)

    common = {
        "repo_root": args.repo_root,
        "result_root": args.result_root,
        "data_dir": args.data_dir,
    }
    if args.command == "stage":
        result = stage_protocol_inputs_v2(
            multicity_manifest_path=args.multicity_manifest,
            frozen_list_path=args.frozen_list,
            split_source_dir=args.split_source_dir,
            data_dir=args.data_dir,
            result_root=args.result_root,
        )
        summary = {"command": "stage", "status": "staged", **result}
    elif args.command == "structural-gate":
        result = write_primary_structural_gates_v2(**common)
        summary = {"command": "structural-gate", **result}
    elif args.command == "lock":
        result = write_primary_protocol_lock_v2(**common)
        summary = {"command": "lock", **result}
    else:
        snapshot = verify_primary_warsaw_full_protocol_lock_snapshot_v2(
            Path(args.result_root) / "protocol_lock.json",
            args.repo_root,
            args.result_root,
            args.data_dir,
        )
        summary = {
            "command": "verify-lock",
            "status": "verified",
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
            "mandatory_file_count": snapshot.payload["mandatory_file_count"],
            "fit_count": len(snapshot.payload["fit_inventory"]),
        }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
