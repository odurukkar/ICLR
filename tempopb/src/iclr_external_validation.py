"""One-way external confirmation on untouched Krakow and Katowice series."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Mapping, Sequence

import numpy as np

from iclr_corpus import SeriesRef, Split
from iclr_env import EnvConfig, EpisodeResult, endowment_selector, uniform_policy
from iclr_multicity_evaluate import (
    EXACT_SIGN_FLIP_LIMIT,
    TIE_TOLERANCE,
    _city,
    _evaluated_payload,
    _fit_selector,
    _paired_bootstrap_interval,
    _sha256,
    _write_per_series_csv,
    count_actuated_series,
    evaluate_policy_on_split,
    holm_adjust,
    evaluate_gold_gate,
    required_fit_inventory,
    verify_fit_inventory,
    verify_lock_fit_inventory,
)
from iclr_multicity_protocol import (
    PABULIB_COMMIT,
    verify_multicity_protocol_lock,
    verify_structural_gates,
    write_immutable_json_artifact,
)
from parse_pb import parse_pb_file
from iclr_stats import (
    exact_paired_sign_flip_pvalue,
    monte_carlo_paired_sign_flip_pvalue,
)


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results" / "iclr_external_validation"
DATA_DIR = ROOT / "data" / "pb_external_validation"
ANCHOR_NAME = "published_lock_anchor.json"
ORIGINAL_RESULT_ROOT = ROOT / "results" / "iclr_multicity"
PRIMARY_SEED = 42
SEEDS = (1, 2, 42)
TRAIN_THROUGH = 2022
FIRST_SCORE_YEAR = 2023
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260812
SIGN_FLIP_DRAWS = 200_000
SIGN_FLIP_SEED = 20260812
PABULIB_REMOTE = "https://github.com/pabulib/pabulib_files.git"

KATOWICE_SERIES = (
    "Bogucice",
    "Dab",
    "Dabrowka_Mala",
    "Giszowiec",
    "Kostuchna",
    "Koszutka",
    "Murcki",
    "Osiedle_Tysiaclecia",
    "Osiedle_Witosa",
    "Podlesie",
    "Srodmiescie",
    "Zaleze",
    "Zarzecze",
    "Zawodzie",
)
KRAKOW_SERIES = (
    "Bienczyce",
    "Biezanow-Prokocim",
    "Bronowice",
    "Czyzyny",
    "Debniki",
    "Grzegorzki",
    "Krowodrza",
    "Lagiewniki-Borek_Falecki",
    "Mistrzejowice",
    "Nowa_Huta",
    "Podgorze",
    "Podgorze_Duchackie",
    "Pradnik_Bialy",
    "Pradnik_Czerwony",
    "Stare_Miasto",
    "Swoszowice",
    "Wzgorza_Krzeslawickie",
    "Zwierzyniec",
)
CITY_CONFIG = {
    "Katowice": {"years": tuple(range(2020, 2026)), "tokens": KATOWICE_SERIES},
    "Krakow": {"years": tuple(range(2018, 2026)), "tokens": KRAKOW_SERIES},
}
EXPECTED_SERIES = 32
EXPECTED_ELECTIONS = 228
EXPECTED_WARMUP = 132
EXPECTED_SCORE = 96
EXPECTED_CITY_SERIES = {"Poland/Katowice": 14, "Poland/Krakow": 18}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectedFile:
    path: Path
    city: str
    token: str
    year: int


def evidence_gate_config() -> Dict[str, object]:
    """Return the canonical external-confirmation decision thresholds."""

    return {
        "gold": {
            "city_macro_difference_max": -0.005,
            "ci_high_max": 0.0,
            "holm_p_value_max": 0.05,
            "welfare_ratio_min": 0.98,
            "exclusion_increase_max": 0.005,
            "city_macro_seed_spread_max": 0.005,
            "minimum_endowment_actuated_series": 8,
            "endowment_actuation_greater_than_priority": True,
        },
        "silver": {"maximum_primary_inference_failures": 1},
    }


def external_protocol_config() -> Dict[str, object]:
    """Return all result-independent semantics bound into the lock."""

    return {
        "study": "fresh-city-endowment-confirmation-v1",
        "source_commit": PABULIB_COMMIT,
        "cities": {
            city: {
                "years": list(row["years"]),
                "series_tokens": list(row["tokens"]),
            }
            for city, row in CITY_CONFIG.items()
        },
        "expected_counts": {
            "series": EXPECTED_SERIES,
            "elections": EXPECTED_ELECTIONS,
            "warmup_elections": EXPECTED_WARMUP,
            "score_elections": EXPECTED_SCORE,
        },
        "train_through": TRAIN_THROUGH,
        "first_score_year": FIRST_SCORE_YEAR,
        "canonical_paths": {
            "data_dir": str(DATA_DIR.relative_to(ROOT)),
            "result_root": str(RESULT_ROOT.relative_to(ROOT)),
        },
        "policies": {
            "primary": "fixed pooled-temporal learned endowment",
            "negative_control": "fixed pooled-temporal learned priority",
            "baseline": "ordinary MES",
            "seeds": list(SEEDS),
            "primary_seed": PRIMARY_SEED,
        },
        "statistics": {
            "unit": "series",
            "city_aggregation": "equal-weight city macro",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "sign_flip_draws": SIGN_FLIP_DRAWS,
            "sign_flip_seed": SIGN_FLIP_SEED,
            "multiplicity": "Holm across two cities at primary seed",
        },
        "evidence_gates": evidence_gate_config(),
    }


def select_external_files(source_dir: Path) -> list[SelectedFile]:
    """Select the exact corpus from filenames without parsing ballot content."""

    source_dir = Path(source_dir)
    selected: list[SelectedFile] = []
    missing: list[str] = []
    for city, row in CITY_CONFIG.items():
        for year in row["years"]:
            for token in row["tokens"]:
                path = source_dir / f"Poland_{city}_{year}_{token}.pb"
                if not path.is_file():
                    missing.append(path.name)
                    continue
                selected.append(SelectedFile(path, city, token, int(year)))
    if len(selected) != EXPECTED_ELECTIONS:
        raise RuntimeError(
            f"expected {EXPECTED_ELECTIONS} filename-selected files, "
            f"found {len(selected)}; missing={missing[:5]}"
        )
    warmup = sum(row.year <= TRAIN_THROUGH for row in selected)
    score = sum(row.year >= FIRST_SCORE_YEAR for row in selected)
    if warmup != EXPECTED_WARMUP or score != EXPECTED_SCORE:
        raise RuntimeError(
            f"filename split count differs: warmup={warmup}, score={score}"
        )
    return sorted(selected, key=lambda row: row.path.name)


def require_canonical_paths(data_dir: Path, result_root: Path) -> None:
    """Disallow alternate production lineages after an outcome is observed."""

    if Path(data_dir).resolve() != DATA_DIR.resolve() or Path(result_root).resolve() != RESULT_ROOT.resolve():
        raise RuntimeError(
            "external confirmation requires canonical external paths: "
            f"data={DATA_DIR}, results={RESULT_ROOT}"
        )


def _git_blob_sha1(path: Path) -> str:
    content = Path(path).read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def verify_source_checkout(
    source_dir: Path, selected: Sequence[SelectedFile]
) -> Dict[str, str]:
    """Authenticate HEAD, official origin, and every selected Git blob."""

    source_dir = Path(source_dir).resolve()
    checkout = source_dir.parent
    if source_dir.name != "pb_files" or not (checkout / ".git").exists():
        raise RuntimeError("external source must be the pinned Pabulib Git checkout")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    if git("rev-parse", "HEAD") != PABULIB_COMMIT:
        raise RuntimeError("Pabulib checkout HEAD differs from the pinned commit")
    origin = git("remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    expected_origin = PABULIB_REMOTE.removesuffix("/").removesuffix(".git")
    if origin != expected_origin:
        raise RuntimeError(f"Pabulib checkout origin differs: {origin!r}")
    tree_output = git("ls-tree", "-r", PABULIB_COMMIT, "--", "pb_files")
    tree = {}
    for line in tree_output.splitlines():
        metadata, relative = line.split("\t", 1)
        tree[relative] = metadata.split()[2]
    blobs = {}
    for row in selected:
        relative = f"pb_files/{row.path.name}"
        expected_blob = tree.get(relative)
        observed_blob = _git_blob_sha1(row.path)
        if expected_blob is None or observed_blob != expected_blob:
            raise RuntimeError(f"selected Pabulib blob differs from commit: {row.path.name}")
        blobs[row.path.name] = observed_blob
    return blobs


def _manifest_payload(
    selected: Sequence[SelectedFile], data_dir: Path, source_blobs: Mapping[str, str]
) -> Dict[str, object]:
    rows = []
    for selected_file in selected:
        path = Path(data_dir) / selected_file.path.name
        rows.append(
            {
                "name": path.name,
                "city_token": selected_file.city,
                "series_token": selected_file.token,
                "year": selected_file.year,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "git_blob_sha1": source_blobs[path.name],
            }
        )
    return {
        "schema_version": 1,
        "source_commit": PABULIB_COMMIT,
        "selection": "explicit persistent filename tokens; no PB parsing",
        "n_series": EXPECTED_SERIES,
        "n_elections": EXPECTED_ELECTIONS,
        "n_warmup": EXPECTED_WARMUP,
        "n_score": EXPECTED_SCORE,
        "files": rows,
    }


def stage_external_corpus(
    selected: Sequence[SelectedFile],
    data_dir: Path,
    manifest_path: Path,
    source_blobs: Mapping[str, str],
) -> Dict[str, object]:
    """Copy and hash opaque PB files without invoking the PB parser."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = {row.path.name for row in selected}
    unexpected = sorted(path.name for path in data_dir.glob("*.pb") if path.name not in expected)
    if unexpected:
        raise RuntimeError(f"external data directory has unexpected files: {unexpected[:5]}")
    for row in selected:
        target = data_dir / row.path.name
        if target.is_file():
            if _sha256(target) != _sha256(row.path):
                raise RuntimeError(f"staged file differs from source: {target.name}")
        else:
            shutil.copy2(row.path, target)
    if set(source_blobs) != expected:
        raise RuntimeError("authenticated source-blob inventory differs")
    payload = _manifest_payload(selected, data_dir, source_blobs)
    write_immutable_json_artifact(manifest_path, payload)
    return payload


def original_fit_paths() -> Dict[str, Path]:
    paths = {}
    for arm in ("endowment", "priority"):
        for seed in SEEDS:
            paths[f"fit/{arm}/seed-{seed}"] = (
                ORIGINAL_RESULT_ROOT
                / "fits"
                / "temporal_2022"
                / arm
                / f"seed-{seed}.json"
            )
    return paths


def verify_original_study() -> None:
    """Require the first study, all fits, and its falsification receipt."""

    lock_path = ORIGINAL_RESULT_ROOT / "protocol_lock.json"
    payload = verify_multicity_protocol_lock(
        lock_path, ROOT, ORIGINAL_RESULT_ROOT, ROOT / "data" / "pb_multicity"
    )
    verify_lock_fit_inventory(payload)
    verify_structural_gates(ORIGINAL_RESULT_ROOT / "structural_gates.json")
    inventory = required_fit_inventory(ORIGINAL_RESULT_ROOT)
    verify_fit_inventory(inventory, payload, _sha256(lock_path))
    receipt_path = ORIGINAL_RESULT_ROOT / "heldout_opened.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_fit_hashes = {
        str(Path(path).resolve().relative_to(ROOT)): _sha256(path)
        for _, path in sorted(inventory, key=lambda item: str(item[1]))
    }
    expected_receipt = {
        "schema_version": 1,
        "lock_sha256": _sha256(lock_path),
        "fit_sha256": expected_fit_hashes,
        "structural_gates_sha256": _sha256(
            ORIGINAL_RESULT_ROOT / "structural_gates.json"
        ),
    }
    if {key: receipt.get(key) for key in expected_receipt} != expected_receipt:
        raise RuntimeError("original held-out receipt does not match lock, fits, and gates")
    evidence = json.loads(
        (ORIGINAL_RESULT_ROOT / "evaluation" / "evidence_payload.json").read_text(
            encoding="utf-8"
        )
    )
    decision = json.loads(
        (ORIGINAL_RESULT_ROOT / "evidence_decision.json").read_text(encoding="utf-8")
    )
    recomputed = evaluate_gold_gate(evidence, payload["evidence_gates"])
    if decision != recomputed or decision.get("classification") != "falsified":
        raise RuntimeError("original falsification does not recompute from locked evidence")


def external_tracked_files(
    data_dir: Path = DATA_DIR, result_root: Path = RESULT_ROOT
) -> Dict[str, Path]:
    """Enumerate the exact files that the external lock must bind."""

    files: Dict[str, Path] = {
        "artifact/corpus_manifest": Path(result_root) / "corpus_manifest.json",
        "environment/pyproject": ROOT / "pyproject.toml",
        "environment/uv_lock": ROOT / "uv.lock",
        "original/protocol_lock": ORIGINAL_RESULT_ROOT / "protocol_lock.json",
        "original/heldout_receipt": ORIGINAL_RESULT_ROOT / "heldout_opened.json",
        "original/evidence_decision": ORIGINAL_RESULT_ROOT / "evidence_decision.json",
        "original/evidence_payload": ORIGINAL_RESULT_ROOT / "evaluation" / "evidence_payload.json",
        "original/structural_gates": ORIGINAL_RESULT_ROOT / "structural_gates.json",
        "test/external_validation": ROOT / "tests" / "test_iclr_external_validation.py",
    }
    files.update(original_fit_paths())
    for path in sorted((ROOT / "src").glob("*.py")):
        files[f"source/{path.name}"] = path
    for path in sorted(Path(data_dir).glob("*.pb")):
        files[f"corpus/{path.name}"] = path
    missing = [label for label, path in files.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"external lock files missing: {missing[:5]}")
    if sum(label.startswith("corpus/") for label in files) != EXPECTED_ELECTIONS:
        raise RuntimeError("external lock corpus count differs")
    return files


def write_external_lock(
    lock_path: Path, repo_root: Path, tracked_files: Mapping[str, Path]
) -> str:
    """Write a canonical, immutable hash lock for this confirmation."""

    repo_root = Path(repo_root).resolve()
    tracked = {}
    for label, path in sorted(tracked_files.items()):
        resolved = Path(path).resolve()
        tracked[label] = {
            "path": str(resolved.relative_to(repo_root)),
            "sha256": _sha256(resolved),
        }
    payload = {
        "schema_version": 1,
        "protocol": external_protocol_config(),
        "tracked_files": tracked,
    }
    return write_immutable_json_artifact(Path(lock_path), payload)


def verify_external_lock(
    lock_path: Path, repo_root: Path, expected_files: Mapping[str, Path]
) -> Dict[str, object]:
    """Verify canonical semantics, the exact label/path set, and all hashes."""

    repo_root = Path(repo_root).resolve()
    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("external lock schema differs")
    if payload.get("protocol") != external_protocol_config():
        raise RuntimeError("external protocol differs from canonical semantics")
    tracked = payload.get("tracked_files", {})
    if set(tracked) != set(expected_files):
        raise RuntimeError("external tracked-file set differs")
    for label, expected in expected_files.items():
        resolved = Path(expected).resolve()
        relative = str(resolved.relative_to(repo_root))
        if tracked[label].get("path") != relative:
            raise RuntimeError(f"external tracked path differs for {label}")
        observed = _sha256(resolved)
        if tracked[label].get("sha256") != observed:
            raise RuntimeError(f"hash mismatch for tracked file {label}")
    return payload


def write_published_anchor(
    anchor_path: Path, lock_sha256: str, confirmation: str
) -> str:
    """Record the exact lock hash confirmed outside the prepare invocation."""

    expected_confirmation = f"confirm {lock_sha256}"
    if confirmation != expected_confirmation:
        raise RuntimeError("published lock confirmation text differs")
    return write_immutable_json_artifact(
        Path(anchor_path),
        {
            "schema_version": 1,
            "study": "fresh-city-endowment-confirmation-v1",
            "lock_sha256": lock_sha256,
            "confirmation": confirmation,
        },
    )


def verify_published_anchor(anchor_path: Path, lock_path: Path) -> Dict[str, object]:
    """Require the separately confirmed lock hash before evaluation."""

    payload = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    actual = _sha256(Path(lock_path))
    expected = {
        "schema_version": 1,
        "study": "fresh-city-endowment-confirmation-v1",
        "lock_sha256": actual,
        "confirmation": f"confirm {actual}",
    }
    if payload != expected:
        raise RuntimeError("published lock anchor does not match the protocol lock")
    return payload


def assert_evaluation_unopened(result_root: Path) -> None:
    """Reject any second opening of the canonical external study."""

    result_root = Path(result_root)
    guarded = (
        result_root / "heldout_opened.json",
        result_root / "evaluation" / "per_series.json",
        result_root / "evaluation" / "per_series.csv",
        result_root / "evaluation" / "summary.json",
        result_root / "evaluation" / "evidence_payload.json",
        result_root / "evidence_decision.json",
    )
    if any(path.exists() for path in guarded):
        raise RuntimeError("external evaluation is one-way and has already been opened")


def _condition(passed: bool, observed, threshold) -> Dict[str, object]:
    return {"passed": bool(passed), "observed": observed, "threshold": threshold}


def _inference_config() -> Dict[str, int]:
    return {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sign_flip_draws": SIGN_FLIP_DRAWS,
        "sign_flip_seed": SIGN_FLIP_SEED,
    }


def _validate_evidence_family(evidence: Mapping[str, object]) -> None:
    expected_seeds = {str(seed) for seed in SEEDS}
    seeds = evidence.get("seeds")
    if (
        str(evidence.get("primary_seed")) != str(PRIMARY_SEED)
        or not isinstance(seeds, Mapping)
        or set(seeds) != expected_seeds
        or evidence.get("inference") != _inference_config()
    ):
        raise RuntimeError("external evidence family is incomplete or noncanonical")
    for seed in expected_seeds:
        payload = seeds[seed]
        cities = payload.get("cities") if isinstance(payload, Mapping) else None
        if not isinstance(cities, Mapping) or set(cities) != set(EXPECTED_CITY_SERIES):
            raise RuntimeError("external evidence family is missing a city")
        macro = float(payload.get("city_macro_difference"))
        if not math.isfinite(macro):
            raise RuntimeError("external evidence family has a non-finite macro")
        for city, expected_count in EXPECTED_CITY_SERIES.items():
            row = cities[city]
            if int(row.get("n_series", -1)) != expected_count:
                raise RuntimeError("external evidence family has the wrong series count")
            required = (
                "difference",
                "ci_high",
                "welfare_ratio",
                "exclusion_difference",
            )
            if any(not math.isfinite(float(row.get(field, float("nan")))) for field in required):
                raise RuntimeError("external evidence family has non-finite metrics")
    primary_cities = seeds[str(PRIMARY_SEED)]["cities"]
    if any(
        not math.isfinite(float(primary_cities[city].get("holm_p_value", float("nan"))))
        for city in EXPECTED_CITY_SERIES
    ):
        raise RuntimeError("external evidence family lacks the two-city Holm family")
    for field in ("endowment_actuated_series", "priority_actuated_series"):
        value = int(evidence.get(field, -1))
        if not 0 <= value <= EXPECTED_SERIES:
            raise RuntimeError("external evidence family has invalid actuation counts")


def evaluate_external_gate(
    evidence: Mapping[str, object], gates: Mapping[str, object] | None = None
) -> Dict[str, object]:
    """Apply the preregistered gold/silver decision without discretion."""

    canonical = evidence_gate_config()
    gates = canonical if gates is None else gates
    if gates != canonical:
        raise RuntimeError("external evidence gates differ from canonical thresholds")
    _validate_evidence_family(evidence)
    gold = gates["gold"]
    seeds = evidence["seeds"]
    primary = seeds[str(evidence["primary_seed"])]
    differences = {
        seed: {city: float(row["difference"]) for city, row in payload["cities"].items()}
        for seed, payload in seeds.items()
    }
    macros = {seed: float(payload["city_macro_difference"]) for seed, payload in seeds.items()}
    welfare = {
        seed: {city: float(row["welfare_ratio"]) for city, row in payload["cities"].items()}
        for seed, payload in seeds.items()
    }
    exclusion = {
        seed: {
            city: float(row["exclusion_difference"])
            for city, row in payload["cities"].items()
        }
        for seed, payload in seeds.items()
    }
    primary_ci = {city: float(row["ci_high"]) for city, row in primary["cities"].items()}
    primary_holm = {
        city: float(row["holm_p_value"]) for city, row in primary["cities"].items()
    }
    macro_spread = max(macros.values()) - min(macros.values())
    endowment_actuation = int(evidence["endowment_actuated_series"])
    priority_actuation = int(evidence["priority_actuated_series"])

    conditions = {
        "negative_each_city_all_seeds": _condition(
            all(value < 0 for row in differences.values() for value in row.values()),
            differences,
            "difference < 0 for both cities and every seed",
        ),
        "macro_at_most_minus_0_005_all_seeds": _condition(
            all(value <= float(gold["city_macro_difference_max"]) for value in macros.values()),
            macros,
            "city macro <= -0.005 for every seed",
        ),
        "primary_ci_excludes_zero_each_city": _condition(
            all(value < float(gold["ci_high_max"]) for value in primary_ci.values()),
            primary_ci,
            "95% interval upper endpoint < 0 in both cities",
        ),
        "primary_holm_p_each_city": _condition(
            all(value <= float(gold["holm_p_value_max"]) for value in primary_holm.values()),
            primary_holm,
            "Holm-adjusted p <= 0.05 in both cities",
        ),
        "welfare_ratio_all_cities_seeds": _condition(
            all(value >= float(gold["welfare_ratio_min"]) for row in welfare.values() for value in row.values()),
            welfare,
            "welfare ratio >= 0.98 for both cities and every seed",
        ),
        "exclusion_increase_all_cities_seeds": _condition(
            all(value <= float(gold["exclusion_increase_max"]) for row in exclusion.values() for value in row.values()),
            exclusion,
            "exclusion increase <= 0.005 for both cities and every seed",
        ),
        "macro_seed_spread": _condition(
            macro_spread <= float(gold["city_macro_seed_spread_max"]),
            macro_spread,
            "city-macro seed spread <= 0.005",
        ),
        "endowment_actuation": _condition(
            endowment_actuation >= int(gold["minimum_endowment_actuated_series"])
            and endowment_actuation > priority_actuation,
            {"endowment": endowment_actuation, "priority": priority_actuation},
            "endowment >= 8 series and endowment > priority",
        ),
    }
    gold_pass = all(row["passed"] for row in conditions.values())
    inference_failures = sum(
        not (
            primary_ci[city] < float(gold["ci_high_max"])
            and primary_holm[city] <= float(gold["holm_p_value_max"])
        )
        for city in primary_ci
    )
    silver_pass = (
        conditions["negative_each_city_all_seeds"]["passed"]
        and conditions["macro_at_most_minus_0_005_all_seeds"]["passed"]
        and conditions["welfare_ratio_all_cities_seeds"]["passed"]
        and conditions["exclusion_increase_all_cities_seeds"]["passed"]
        and conditions["macro_seed_spread"]["passed"]
        and inference_failures
        <= int(gates["silver"]["maximum_primary_inference_failures"])
    )
    classification = "gold" if gold_pass else "silver" if silver_pass else "falsified"
    return {
        "classification": classification,
        "gold_pass": gold_pass,
        "primary_seed": str(evidence["primary_seed"]),
        "conditions": conditions,
    }


def load_external_corpus(
    manifest_path: Path, data_dir: Path, receipt_path: Path
) -> tuple[Dict[str, SeriesRef], Dict[str, Dict[int, object]]]:
    """Parse the locked corpus only after the held-out receipt exists."""

    if not Path(receipt_path).is_file():
        raise RuntimeError("external held-out receipt must exist before PB parsing")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_series: dict[str, dict[int, Path]] = defaultdict(dict)
    instances: dict[str, dict[int, object]] = defaultdict(dict)
    token_to_key: dict[tuple[str, str], str] = {}
    for row in manifest["files"]:
        path = Path(data_dir) / row["name"]
        if path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            raise RuntimeError(f"locked external file differs: {path.name}")
        instance = parse_pb_file(path)
        if instance.vote_type != "approval":
            raise RuntimeError(f"external file is not approval voting: {path.name}")
        if int(instance.year) != int(row["year"]):
            raise RuntimeError(f"external year differs from filename: {path.name}")
        key = instance.series_key()
        expected_city = f"Poland/{row['city_token']}"
        if not key.startswith(expected_city + "/"):
            raise RuntimeError(f"external city differs for {path.name}: {key}")
        token_id = (row["city_token"], row["series_token"])
        prior = token_to_key.setdefault(token_id, key)
        if prior != key:
            raise RuntimeError(f"filename token maps to multiple series: {token_id}")
        if int(instance.year) in by_series[key]:
            raise RuntimeError(f"duplicate external election: {key} {instance.year}")
        by_series[key][int(instance.year)] = path
        instances[key][int(instance.year)] = instance

    index = {
        key: SeriesRef(
            key=key,
            years=tuple(sorted(years)),
            paths=tuple(years[year] for year in sorted(years)),
        )
        for key, years in sorted(by_series.items())
    }
    if len(index) != EXPECTED_SERIES:
        raise RuntimeError(f"expected {EXPECTED_SERIES} parsed series, found {len(index)}")
    if sum(len(ref.years) for ref in index.values()) != EXPECTED_ELECTIONS:
        raise RuntimeError("parsed external election count differs")
    expected_years = {
        "Poland/Katowice": tuple(CITY_CONFIG["Katowice"]["years"]),
        "Poland/Krakow": tuple(CITY_CONFIG["Krakow"]["years"]),
    }
    for ref in index.values():
        city = ref.city
        if ref.years != expected_years.get(city):
            raise RuntimeError(f"external series years differ for {ref.key}: {ref.years}")
    return index, {key: dict(years) for key, years in instances.items()}


def external_split(index: Mapping[str, SeriesRef]) -> Split:
    train = []
    test = []
    for key, ref in sorted(index.items()):
        warmup = tuple(year for year in ref.years if year <= TRAIN_THROUGH)
        score = tuple(year for year in ref.years if year >= FIRST_SCORE_YEAR)
        train.append((key, warmup))
        test.append((key, score))
    if sum(len(years) for _, years in train) != EXPECTED_WARMUP:
        raise RuntimeError("external warm-up count differs")
    if sum(len(years) for _, years in test) != EXPECTED_SCORE:
        raise RuntimeError("external score count differs")
    return Split(
        name="external_transfer_2023",
        train=tuple(train),
        test=tuple(test),
        note="fixed policies; warm-up through 2022; score 2023--2025",
    )


def _paired_pvalue_external(differences: np.ndarray) -> tuple[float, str]:
    if differences.size <= EXACT_SIGN_FLIP_LIMIT:
        return exact_paired_sign_flip_pvalue(differences), "exact_sign_flip"
    return (
        monte_carlo_paired_sign_flip_pvalue(
            differences, draws=SIGN_FLIP_DRAWS, seed=SIGN_FLIP_SEED
        ),
        "monte_carlo_sign_flip",
    )


def paired_external_summary(
    learned: Sequence[EpisodeResult], baseline: Sequence[EpisodeResult]
) -> Dict[str, Dict[str, float | int | str]]:
    """Compute the locked paired summary with every inference argument explicit."""

    learned_by_series = {episode.series: episode for episode in learned}
    baseline_by_series = {episode.series: episode for episode in baseline}
    if set(learned_by_series) != set(baseline_by_series) or len(learned_by_series) != EXPECTED_SERIES:
        raise RuntimeError("external paired-series identities differ")
    grouped: dict[str, list[tuple[EpisodeResult, EpisodeResult]]] = defaultdict(list)
    for series in sorted(learned_by_series):
        left = learned_by_series[series]
        right = baseline_by_series[series]
        if left.worst_csd is None or right.worst_csd is None:
            raise RuntimeError(f"external series is not scoreable: {series}")
        grouped[_city(series)].append((left, right))
    if set(grouped) != set(EXPECTED_CITY_SERIES):
        raise RuntimeError("external paired summary has the wrong cities")

    rows: Dict[str, Dict[str, float | int | str]] = {}
    for city, pairs in sorted(grouped.items()):
        if len(pairs) != EXPECTED_CITY_SERIES[city]:
            raise RuntimeError(f"external paired summary count differs for {city}")
        differences = np.asarray(
            [left.worst_csd - right.worst_csd for left, right in pairs], dtype=float
        )
        low, high = _paired_bootstrap_interval(
            differences, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
        )
        pvalue, kind = _paired_pvalue_external(differences)
        learned_welfare = sum(left.welfare for left, _ in pairs)
        baseline_welfare = sum(right.welfare for _, right in pairs)
        rows[city] = {
            "n_series": len(pairs),
            "learned_csd": float(np.mean([left.worst_csd for left, _ in pairs])),
            "baseline_csd": float(np.mean([right.worst_csd for _, right in pairs])),
            "difference": float(differences.mean()),
            "ci_low": low,
            "ci_high": high,
            "wins": int(np.count_nonzero(differences < -TIE_TOLERANCE)),
            "ties": int(np.count_nonzero(np.abs(differences) <= TIE_TOLERANCE)),
            "losses": int(np.count_nonzero(differences > TIE_TOLERANCE)),
            "welfare_ratio": float(learned_welfare / baseline_welfare),
            "exclusion_difference": float(
                np.mean([left.exclusion - right.exclusion for left, right in pairs])
            ),
            "p_value": float(pvalue),
            "p_value_kind": kind,
        }
    city_rows = list(rows.values())
    rows["CITY_MACRO"] = {
        "n_cities": len(city_rows),
        "n_series": sum(int(row["n_series"]) for row in city_rows),
        "learned_csd": float(np.mean([float(row["learned_csd"]) for row in city_rows])),
        "baseline_csd": float(np.mean([float(row["baseline_csd"]) for row in city_rows])),
        "difference": float(np.mean([float(row["difference"]) for row in city_rows])),
        "welfare_ratio": float(np.mean([float(row["welfare_ratio"]) for row in city_rows])),
        "exclusion_difference": float(
            np.mean([float(row["exclusion_difference"]) for row in city_rows])
        ),
        "wins": sum(int(row["wins"]) for row in city_rows),
        "ties": sum(int(row["ties"]) for row in city_rows),
        "losses": sum(int(row["losses"]) for row in city_rows),
    }
    return rows


def prepare(source_dir: Path, data_dir: Path, result_root: Path) -> Dict[str, object]:
    """Stage opaque files and write the lock without parsing PB content."""

    result_root = Path(result_root)
    require_canonical_paths(data_dir, result_root)
    assert_evaluation_unopened(result_root)
    if (result_root / ANCHOR_NAME).exists():
        raise RuntimeError("published lock anchor already exists")
    verify_original_study()
    selected = select_external_files(source_dir)
    source_blobs = verify_source_checkout(source_dir, selected)
    manifest = stage_external_corpus(
        selected,
        data_dir,
        result_root / "corpus_manifest.json",
        source_blobs,
    )
    tracked = external_tracked_files(data_dir, result_root)
    lock_path = result_root / "protocol_lock.json"
    lock_hash = write_external_lock(lock_path, ROOT, tracked)
    verify_external_lock(lock_path, ROOT, tracked)
    return {
        "status": "locked",
        "lock_sha256": lock_hash,
        "tracked_file_count": len(tracked),
        "n_series": manifest["n_series"],
        "n_elections": manifest["n_elections"],
        "n_warmup": manifest["n_warmup"],
        "n_score": manifest["n_score"],
        "pb_parser_called": False,
    }


def publish_anchor(
    result_root: Path, lock_sha256: str, confirmation: str
) -> Dict[str, object]:
    """Create the canonical anchor only after the user confirms the published hash."""

    require_canonical_paths(DATA_DIR, result_root)
    assert_evaluation_unopened(result_root)
    lock_path = Path(result_root) / "protocol_lock.json"
    if _sha256(lock_path) != lock_sha256:
        raise RuntimeError("confirmed lock hash differs from the canonical protocol lock")
    anchor_path = Path(result_root) / ANCHOR_NAME
    anchor_sha256 = write_published_anchor(anchor_path, lock_sha256, confirmation)
    verify_published_anchor(anchor_path, lock_path)
    return {
        "status": "anchored",
        "lock_sha256": lock_sha256,
        "anchor_sha256": anchor_sha256,
    }


def evaluate(data_dir: Path, result_root: Path) -> Dict[str, object]:
    """Open the locked corpus once, evaluate fixed policies, and decide."""

    result_root = Path(result_root)
    require_canonical_paths(data_dir, result_root)
    lock_path = result_root / "protocol_lock.json"
    receipt_path = result_root / "heldout_opened.json"
    assert_evaluation_unopened(result_root)
    verify_original_study()
    tracked = external_tracked_files(data_dir, result_root)
    verify_external_lock(lock_path, ROOT, tracked)
    verify_published_anchor(result_root / ANCHOR_NAME, lock_path)
    receipt = {
        "schema_version": 1,
        "protocol_lock_sha256": _sha256(lock_path),
        "corpus_manifest_sha256": _sha256(result_root / "corpus_manifest.json"),
        "fixed_fit_sha256": {
            label: _sha256(path) for label, path in sorted(original_fit_paths().items())
        },
    }
    write_immutable_json_artifact(receipt_path, receipt)

    index, instances = load_external_corpus(
        result_root / "corpus_manifest.json", data_dir, receipt_path
    )
    split = external_split(index)
    cfg = EnvConfig()
    policies = {"mes": endowment_selector(uniform_policy, cfg)}
    for arm in ("endowment", "priority"):
        for seed in SEEDS:
            path = original_fit_paths()[f"fit/{arm}/seed-{seed}"]
            policies[f"{arm}/seed-{seed}"] = _fit_selector(path, cfg)
    evaluated = {}
    for name, selector in sorted(policies.items()):
        logger.info("evaluating %s on untouched external cities", name)
        evaluated[name] = evaluate_policy_on_split(
            split, index, instances, selector, cfg
        )
    baseline = [row.episode for row in evaluated["mes"]]
    expected_identities = set(index)
    for name, rows in evaluated.items():
        identities = [row.episode.series for row in rows]
        if len(identities) != EXPECTED_SERIES or set(identities) != expected_identities:
            raise RuntimeError(f"external evaluated-series identities differ for {name}")
    summaries = {
        name: paired_external_summary([row.episode for row in rows], baseline)
        for name, rows in evaluated.items()
        if name != "mes"
    }
    primary = summaries[f"endowment/seed-{PRIMARY_SEED}"]
    raw_pvalues = {
        city: float(primary[city]["p_value"]) for city in EXPECTED_CITY_SERIES
    }
    if set(raw_pvalues) != set(EXPECTED_CITY_SERIES):
        raise RuntimeError("external Holm family must contain exactly two cities")
    adjusted = holm_adjust(raw_pvalues)
    for city, value in adjusted.items():
        primary[city]["holm_p_value"] = value
    evidence = {
        "primary_seed": str(PRIMARY_SEED),
        "seeds": {
            str(seed): {
                "cities": {
                    city: dict(row)
                    for city, row in summaries[f"endowment/seed-{seed}"].items()
                    if city != "CITY_MACRO"
                },
                "city_macro_difference": float(
                    summaries[f"endowment/seed-{seed}"]["CITY_MACRO"]["difference"]
                ),
            }
            for seed in SEEDS
        },
        "endowment_actuated_series": count_actuated_series(
            evaluated[f"endowment/seed-{PRIMARY_SEED}"], evaluated["mes"]
        ),
        "priority_actuated_series": count_actuated_series(
            evaluated[f"priority/seed-{PRIMARY_SEED}"], evaluated["mes"]
        ),
        "inference": _inference_config(),
    }
    decision = evaluate_external_gate(evidence)
    evaluation_dir = result_root / "evaluation"
    write_immutable_json_artifact(
        evaluation_dir / "per_series.json",
        {name: [_evaluated_payload(row) for row in rows] for name, rows in sorted(evaluated.items())},
    )
    _write_per_series_csv(
        evaluation_dir / "per_series.csv", {split.name: evaluated}
    )
    write_immutable_json_artifact(evaluation_dir / "summary.json", summaries)
    write_immutable_json_artifact(evaluation_dir / "evidence_payload.json", evidence)
    write_immutable_json_artifact(result_root / "evidence_decision.json", decision)
    return decision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    anchor_parser = subparsers.add_parser("anchor")
    anchor_parser.add_argument("--lock-sha256", required=True)
    anchor_parser.add_argument("--confirmation", required=True)
    for command_parser in (prepare_parser, evaluate_parser, anchor_parser):
        command_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
        command_parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "prepare":
        result = prepare(args.source_dir, args.data_dir, args.result_root)
    elif args.command == "anchor":
        require_canonical_paths(args.data_dir, args.result_root)
        result = publish_anchor(
            args.result_root, args.lock_sha256, args.confirmation
        )
    else:
        result = evaluate(args.data_dir, args.result_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
