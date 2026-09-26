"""Technical amendment for Unicode city identity in support-set confirmation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

import iclr_external_support_set as v2
import iclr_external_validation as v1
from parse_pb import PBInstance


CITY_UNIT_ALIASES = {
    "Katowice": "Katowice",
    "Krakow": "Kraków",
}

ROOT = v2.ROOT
DATA_DIR = v2.DATA_DIR
V2_RESULT_ROOT = v2.RESULT_ROOT
RESULT_RELATIVE = Path("results") / "iclr_external_support_set_amendment"
RESULT_ROOT = ROOT / RESULT_RELATIVE
STUDY = "fresh-city-support-set-confirmation-v3-unicode-amendment"
ANCHOR_NAME = "published_lock_anchor.json"
V2_LOCK_SHA256 = "1464a34e227402b81fd852291ad5008cb12ffb0a9b77e4567510fb3af8300bf5"
V2_ABORT_SHA256 = "18f40453b1562c46df5f191412c6c32b2da531b3d2db7c6f117460217f5ea98b"
V2_ANCHOR_SHA256 = "86aad58495ca0f5f8a698459d0cd59b379287fbcc9531db6fee832ea8a48e4b1"
V2_RECEIPT_SHA256 = "864cfe025340c12104eda614519b1e6cc33e0d6f42c5359ca06f12f991fe85a8"
logger = logging.getLogger(__name__)


def amendment_protocol_config() -> dict:
    """Return v2 semantics plus the sole Unicode city-identity amendment."""

    config = copy.deepcopy(v2.external_protocol_config())
    v1_attempt = config.pop("prior_attempt")
    config["study"] = STUDY
    config["canonical_paths"]["result_root"] = str(RESULT_RELATIVE)
    config["prior_attempts"] = [
        v1_attempt,
        {
            "study": v2.STUDY,
            "classification": "technical_invalid",
            "metrics_computed": False,
            "protocol_lock_sha256": V2_LOCK_SHA256,
            "abort_record_sha256": V2_ABORT_SHA256,
        },
    ]
    config["amendment"] = {
        "kind": "unicode_city_identity_alias",
        "source_metadata_units": dict(CITY_UNIT_ALIASES),
        "analysis_city_tokens": {city: city for city in CITY_UNIT_ALIASES},
        "applied_before_series_key": True,
    }
    return config


def require_canonical_paths(data_dir: Path, result_root: Path) -> None:
    if Path(data_dir).resolve() != DATA_DIR.resolve() or Path(result_root).resolve() != RESULT_ROOT.resolve():
        raise RuntimeError(
            "Unicode amendment requires canonical paths: "
            f"data={DATA_DIR}, results={RESULT_ROOT}"
        )


def assert_evaluation_unopened(result_root: Path) -> None:
    v1.assert_evaluation_unopened(result_root)
    if (Path(result_root) / "evaluation" / "demographic_coverage.json").exists():
        raise RuntimeError("v3 amendment evaluation is one-way and has already been opened")


def _load_v2_lock_snapshot(result_root: Path) -> tuple[dict, dict[str, bytes], bytes]:
    lock_path = Path(result_root) / "protocol_lock.json"
    lock_bytes = lock_path.read_bytes()
    payload = json.loads(lock_bytes.decode("utf-8"))
    if payload.get("schema_version") != 2 or payload.get("protocol") != v2.external_protocol_config():
        raise RuntimeError("v2 protocol lock differs")
    tracked = payload.get("tracked_files")
    if not isinstance(tracked, dict) or not tracked:
        raise RuntimeError("v2 tracked-file inventory differs")
    locked_bytes: dict[str, bytes] = {}
    for label, row in tracked.items():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RuntimeError(f"v2 tracked row differs for {label}")
        path = ROOT / row["path"]
        if not path.is_file():
            raise RuntimeError(f"v2 tracked file is missing for {label}")
        content = path.read_bytes()
        if v2._sha256_bytes(content) != row["sha256"]:
            raise RuntimeError(f"v2 tracked file differs for {label}")
        locked_bytes[label] = content
    v2._verify_v1_against_locked_snapshot(payload, locked_bytes)
    return payload, locked_bytes, lock_bytes


def verify_v2_abort(result_root: Path = V2_RESULT_ROOT) -> dict:
    """Verify the complete v2 lineage and the exact pre-metric technical abort."""

    result_root = Path(result_root)
    required = {
        "protocol_lock": result_root / "protocol_lock.json",
        "published_anchor": result_root / ANCHOR_NAME,
        "heldout_receipt": result_root / "heldout_opened.json",
        "abort_record": result_root / "abort_record.json",
        "corpus_manifest": result_root / "corpus_manifest.json",
    }
    missing = [label for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v2 abort lineage is missing: {missing}")
    guarded = (
        result_root / "evaluation" / "demographic_coverage.json",
        result_root / "evaluation" / "per_series.json",
        result_root / "evaluation" / "per_series.csv",
        result_root / "evaluation" / "summary.json",
        result_root / "evaluation" / "evidence_payload.json",
        result_root / "evidence_decision.json",
    )
    if any(path.exists() for path in guarded):
        raise RuntimeError("v2 evaluation output exists despite the recorded abort")

    _, locked_bytes, lock_bytes = _load_v2_lock_snapshot(result_root)
    lock_sha = v2._sha256_bytes(lock_bytes)
    anchor_bytes = required["published_anchor"].read_bytes()
    receipt_bytes = required["heldout_receipt"].read_bytes()
    manifest_bytes = required["corpus_manifest"].read_bytes()
    if manifest_bytes != locked_bytes["artifact/corpus_manifest"]:
        raise RuntimeError("v2 corpus manifest differs from its locked bytes")
    expected_anchor = {
        "schema_version": 1,
        "study": v2.STUDY,
        "lock_sha256": lock_sha,
        "confirmation": f"confirm {lock_sha}",
    }
    if json.loads(anchor_bytes.decode("utf-8")) != expected_anchor:
        raise RuntimeError("v2 published anchor differs from the confirmed lock")
    expected_receipt = {
        "schema_version": 2,
        "protocol_lock_sha256": lock_sha,
        "corpus_manifest_sha256": v2._sha256_bytes(manifest_bytes),
        "fixed_fit_sha256": {
            label: v2._sha256_bytes(locked_bytes[label])
            for label in sorted(v1.original_fit_paths())
        },
        "ballot_projection": v2.external_protocol_config()["ballot_projection"],
    }
    if json.loads(receipt_bytes.decode("utf-8")) != expected_receipt:
        raise RuntimeError("v2 held-out receipt differs from the confirmed lock")
    expected_abort = {
        "schema_version": 1,
        "study": v2.STUDY,
        "classification": "technical_invalid",
        "protocol_lock_sha256": lock_sha,
        "published_anchor_sha256": v2._sha256_bytes(anchor_bytes),
        "heldout_receipt_sha256": v2._sha256_bytes(receipt_bytes),
        "error_type": "RuntimeError",
        "error": "external city differs for Poland_Krakow_2018_Bienczyce.pb: Poland/Kraków/Bieńczyce",
        "failed_stage": "series_identity_validation_before_demographic_coverage",
        "coverage_computed": False,
        "policies_constructed": False,
        "metrics_computed": False,
        "observed_metadata_units": {
            "Katowice": {"files": 84, "unit": "Katowice"},
            "Krakow": {"files": 144, "unit": "Kraków"},
        },
    }
    abort = json.loads(required["abort_record"].read_text(encoding="utf-8"))
    if abort != expected_abort:
        raise RuntimeError("v2 abort record is contradictory or incomplete")
    return abort


def amendment_tracked_files() -> Dict[str, Path]:
    """Return the exact v2 snapshot plus the new amendment implementation."""

    v2_lock = json.loads(
        (V2_RESULT_ROOT / "protocol_lock.json").read_text(encoding="utf-8")
    )
    files = {
        f"v2_snapshot/{label}": ROOT / row["path"]
        for label, row in v2_lock["tracked_files"].items()
    }
    files.update(
        {
            "v2/protocol_lock": V2_RESULT_ROOT / "protocol_lock.json",
            "v2/published_anchor": V2_RESULT_ROOT / ANCHOR_NAME,
            "v2/heldout_receipt": V2_RESULT_ROOT / "heldout_opened.json",
            "v2/abort_record": V2_RESULT_ROOT / "abort_record.json",
            "v2/corpus_manifest": V2_RESULT_ROOT / "corpus_manifest.json",
            "source/amendment": ROOT / "src" / "iclr_external_support_set_amendment.py",
            "test/amendment": ROOT / "tests" / "test_iclr_external_support_set_amendment.py",
        }
    )
    missing = [label for label, path in files.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"v3 amendment lock files missing: {missing[:5]}")
    return files


def write_amendment_lock(
    lock_path: Path, repo_root: Path, tracked_files: Mapping[str, Path]
) -> str:
    repo_root = Path(repo_root).resolve()
    tracked = {}
    for label, path in sorted(tracked_files.items()):
        resolved = Path(path).resolve()
        content = resolved.read_bytes()
        tracked[label] = {
            "path": str(resolved.relative_to(repo_root)),
            "sha256": v2._sha256_bytes(content),
        }
    payload = {
        "schema_version": 3,
        "protocol": amendment_protocol_config(),
        "tracked_files": tracked,
    }
    return v2.write_immutable_json_artifact(Path(lock_path), payload)


def _verify_retained_v2_snapshot(
    v3_payload: Mapping[str, object], locked_bytes: Mapping[str, bytes]
) -> None:
    expected_hashes = {
        "v2/protocol_lock": V2_LOCK_SHA256,
        "v2/published_anchor": V2_ANCHOR_SHA256,
        "v2/heldout_receipt": V2_RECEIPT_SHA256,
        "v2/abort_record": V2_ABORT_SHA256,
    }
    for label, expected_hash in expected_hashes.items():
        content = locked_bytes.get(label)
        if content is None or v2._sha256_bytes(content) != expected_hash:
            raise RuntimeError(f"retained v2 lineage differs for {label}")

    v2_lock = json.loads(locked_bytes["v2/protocol_lock"].decode("utf-8"))
    if v2_lock.get("schema_version") != 2 or v2_lock.get("protocol") != v2.external_protocol_config():
        raise RuntimeError("retained v2 protocol semantics differ")
    v3_by_path = {
        row["path"]: label for label, row in v3_payload["tracked_files"].items()
    }
    retained_v2: dict[str, bytes] = {}
    for label, row in v2_lock["tracked_files"].items():
        v3_label = v3_by_path.get(row["path"])
        if v3_label is None:
            raise RuntimeError(f"v3 snapshot omits v2 tracked file {label}")
        content = locked_bytes[v3_label]
        if v2._sha256_bytes(content) != row["sha256"]:
            raise RuntimeError(f"v3 snapshot differs from v2 for {label}")
        retained_v2[label] = content
    v2._verify_v1_against_locked_snapshot(v2_lock, retained_v2)


def load_amendment_lock(
    lock_path: Path, repo_root: Path, expected_files: Mapping[str, Path]
) -> tuple[dict, dict[str, bytes], bytes]:
    """Verify and retain the exact v3 bytes that evaluation may consume."""

    repo_root = Path(repo_root).resolve()
    lock_bytes = Path(lock_path).read_bytes()
    payload = json.loads(lock_bytes.decode("utf-8"))
    if payload.get("schema_version") != 3:
        raise RuntimeError("v3 amendment lock schema differs")
    if payload.get("protocol") != amendment_protocol_config():
        raise RuntimeError("v3 amendment protocol differs from canonical semantics")
    tracked = payload.get("tracked_files")
    if not isinstance(tracked, dict) or set(tracked) != set(expected_files):
        raise RuntimeError("v3 amendment tracked-file set differs")
    locked_bytes: dict[str, bytes] = {}
    for label, expected in expected_files.items():
        resolved = Path(expected).resolve()
        relative = str(resolved.relative_to(repo_root))
        row = tracked.get(label)
        if not isinstance(row, dict) or row.get("path") != relative:
            raise RuntimeError(f"v3 amendment tracked path differs for {label}")
        content = resolved.read_bytes()
        if row.get("sha256") != v2._sha256_bytes(content):
            raise RuntimeError(f"hash mismatch for v3 tracked file {label}")
        locked_bytes[label] = content
    _verify_retained_v2_snapshot(payload, locked_bytes)
    verify_v2_abort()
    return payload, locked_bytes, lock_bytes


def write_published_anchor(
    anchor_path: Path, lock_sha256: str, confirmation: str
) -> str:
    if confirmation != f"confirm {lock_sha256}":
        raise RuntimeError("published lock confirmation text differs")
    return v2.write_immutable_json_artifact(
        Path(anchor_path),
        {
            "schema_version": 1,
            "study": STUDY,
            "lock_sha256": lock_sha256,
            "confirmation": confirmation,
        },
    )


def _verify_anchor_for_digest(anchor_path: Path, lock_sha256: str) -> dict:
    payload = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "study": STUDY,
        "lock_sha256": lock_sha256,
        "confirmation": f"confirm {lock_sha256}",
    }
    if payload != expected:
        raise RuntimeError("published v3 lock anchor does not match the protocol lock")
    return payload


def verify_published_anchor(anchor_path: Path, lock_path: Path) -> dict:
    return _verify_anchor_for_digest(
        anchor_path, v2._sha256_bytes(Path(lock_path).read_bytes())
    )


def canonical_series_key(instance: PBInstance, city_token: str) -> str:
    """Validate source metadata and return the locked ASCII analysis identity."""

    expected_unit = CITY_UNIT_ALIASES.get(city_token)
    if instance.country != "Poland" or expected_unit is None or instance.unit != expected_unit:
        raise RuntimeError(
            f"external city differs for {instance.path}: {instance.series_key()}"
        )
    return f"Poland/{city_token}/{instance.subunit or 'CITYWIDE'}"


def load_external_corpus(
    manifest: Mapping[str, object],
    data_dir: Path,
    receipt_path: Path,
    locked_bytes: Mapping[str, bytes],
) -> tuple[Dict[str, v1.SeriesRef], Dict[str, Dict[int, object]]]:
    """Load exact v2 bytes while applying only the locked city-label alias."""

    if not Path(receipt_path).is_file():
        raise RuntimeError("external held-out receipt must exist before PB parsing")
    by_series: dict[str, dict[int, Path]] = defaultdict(dict)
    instances: dict[str, dict[int, object]] = defaultdict(dict)
    token_to_key: dict[tuple[str, str], str] = {}
    for row in manifest["files"]:
        path = Path(data_dir) / row["name"]
        content = locked_bytes[f"corpus/{row['name']}"]
        if len(content) != int(row["bytes"]) or v2._sha256_bytes(content) != row["sha256"]:
            raise RuntimeError(f"locked external bytes differ: {path.name}")
        expected_type = v2.EXPECTED_VOTE_TYPES[row["city_token"]]
        if row.get("vote_type") != expected_type:
            raise RuntimeError(f"locked support-set vote type differs: {path.name}")
        actual_type = v2.vote_type_from_bytes(content, path.name)
        actual_columns = list(v2.vote_columns_from_bytes(content, path.name))
        if actual_type != row.get("vote_type") or actual_columns != row.get("vote_columns"):
            raise RuntimeError(f"locked support-set schema differs: {path.name}")
        expected_demo = row["city_token"] == "Katowice" or int(row["year"]) >= v1.FIRST_SCORE_YEAR
        actual_demo = {"age", "sex"}.issubset(actual_columns)
        if bool(row.get("has_age_sex")) != expected_demo or actual_demo != expected_demo:
            raise RuntimeError(f"locked demographic schema differs: {path.name}")
        instance = v2.parse_support_set_bytes(content, path, expected_type)
        if int(instance.year) != int(row["year"]):
            raise RuntimeError(f"external year differs from filename: {path.name}")
        key = canonical_series_key(instance, row["city_token"])
        token_id = (row["city_token"], row["series_token"])
        prior = token_to_key.setdefault(token_id, key)
        if prior != key:
            raise RuntimeError(f"filename token maps to multiple series: {token_id}")
        if int(instance.year) in by_series[key]:
            raise RuntimeError(f"duplicate external election: {key} {instance.year}")
        by_series[key][int(instance.year)] = path
        instances[key][int(instance.year)] = instance

    index = {
        key: v1.SeriesRef(
            key=key,
            years=tuple(sorted(years)),
            paths=tuple(years[year] for year in sorted(years)),
        )
        for key, years in sorted(by_series.items())
    }
    if len(index) != v1.EXPECTED_SERIES:
        raise RuntimeError(f"expected {v1.EXPECTED_SERIES} parsed series, found {len(index)}")
    if sum(len(ref.years) for ref in index.values()) != v1.EXPECTED_ELECTIONS:
        raise RuntimeError("parsed external election count differs")
    expected_years = {
        f"Poland/{city}": tuple(row["years"]) for city, row in v1.CITY_CONFIG.items()
    }
    for ref in index.values():
        if ref.years != expected_years.get(ref.city):
            raise RuntimeError(f"external series years differ for {ref.key}: {ref.years}")
    return index, {key: dict(years) for key, years in instances.items()}


def prepare(
    data_dir: Path = DATA_DIR,
    result_root: Path = RESULT_ROOT,
) -> dict:
    """Write only the v3 lock; parse no ballot and expose no held-out metric."""

    result_root = Path(result_root)
    require_canonical_paths(data_dir, result_root)
    assert_evaluation_unopened(result_root)
    if (result_root / ANCHOR_NAME).exists():
        raise RuntimeError("published v3 lock anchor already exists")
    verify_v2_abort()
    tracked = amendment_tracked_files()
    lock_path = result_root / "protocol_lock.json"
    lock_sha = write_amendment_lock(lock_path, ROOT, tracked)
    load_amendment_lock(lock_path, ROOT, tracked)
    return {
        "status": "locked",
        "lock_sha256": lock_sha,
        "tracked_file_count": len(tracked),
        "n_series": v1.EXPECTED_SERIES,
        "n_elections": v1.EXPECTED_ELECTIONS,
        "n_warmup": v1.EXPECTED_WARMUP,
        "n_score": v1.EXPECTED_SCORE,
        "amendment": "unicode_city_identity_alias",
        "pb_parser_called": False,
    }


def publish_anchor(
    result_root: Path, lock_sha256: str, confirmation: str
) -> dict:
    require_canonical_paths(DATA_DIR, result_root)
    assert_evaluation_unopened(result_root)
    verify_v2_abort()
    lock_path = Path(result_root) / "protocol_lock.json"
    actual = v2._sha256_bytes(lock_path.read_bytes())
    if actual != lock_sha256:
        raise RuntimeError("confirmed lock hash differs from the canonical v3 lock")
    anchor_path = Path(result_root) / ANCHOR_NAME
    anchor_sha = write_published_anchor(anchor_path, lock_sha256, confirmation)
    _verify_anchor_for_digest(anchor_path, actual)
    return {
        "status": "anchored",
        "lock_sha256": lock_sha256,
        "anchor_sha256": anchor_sha,
    }


def evaluate(data_dir: Path = DATA_DIR, result_root: Path = RESULT_ROOT) -> dict:
    """Consume exact v3-locked bytes, evaluate fixed policies, and decide once."""

    result_root = Path(result_root)
    require_canonical_paths(data_dir, result_root)
    receipt_path = result_root / "heldout_opened.json"
    assert_evaluation_unopened(result_root)
    verify_v2_abort()
    tracked = amendment_tracked_files()
    _, locked_bytes, lock_bytes = load_amendment_lock(
        result_root / "protocol_lock.json", ROOT, tracked
    )
    lock_sha = v2._sha256_bytes(lock_bytes)
    _verify_anchor_for_digest(result_root / ANCHOR_NAME, lock_sha)
    manifest_bytes = locked_bytes["v2/corpus_manifest"]
    receipt = {
        "schema_version": 3,
        "protocol_lock_sha256": lock_sha,
        "v2_protocol_lock_sha256": V2_LOCK_SHA256,
        "v2_abort_record_sha256": V2_ABORT_SHA256,
        "corpus_manifest_sha256": v2._sha256_bytes(manifest_bytes),
        "fixed_fit_sha256": {
            label: v2._sha256_bytes(locked_bytes[f"v2_snapshot/{label}"])
            for label in sorted(v1.original_fit_paths())
        },
        "ballot_projection": amendment_protocol_config()["ballot_projection"],
        "amendment": amendment_protocol_config()["amendment"],
    }
    v2.write_immutable_json_artifact(receipt_path, receipt)

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    v2_snapshot = {
        label.removeprefix("v2_snapshot/"): content
        for label, content in locked_bytes.items()
        if label.startswith("v2_snapshot/")
    }
    index, instances = load_external_corpus(
        manifest, data_dir, receipt_path, v2_snapshot
    )
    split = v1.external_split(index)
    coverage = v2.demographic_coverage_report(instances, dict(split.test))
    v2.write_immutable_json_artifact(
        result_root / "evaluation" / "demographic_coverage.json", coverage
    )
    v2.enforce_demographic_coverage(coverage)
    cfg = v1.EnvConfig()
    policies = {"mes": v1.endowment_selector(v1.uniform_policy, cfg)}
    for arm in ("endowment", "priority"):
        for seed in v1.SEEDS:
            label = f"fit/{arm}/seed-{seed}"
            payload = json.loads(v2_snapshot[label].decode("utf-8"))
            policies[f"{arm}/seed-{seed}"] = v2._fit_selector_from_payload(
                payload, arm, seed, cfg
            )
    evaluated = {}
    for name, selector in sorted(policies.items()):
        logger.info("evaluating %s under the Unicode city-label amendment", name)
        evaluated[name] = v1.evaluate_policy_on_split(
            split, index, instances, selector, cfg
        )
    baseline = [row.episode for row in evaluated["mes"]]
    expected_identities = set(index)
    for name, rows in evaluated.items():
        identities = [row.episode.series for row in rows]
        if len(identities) != v1.EXPECTED_SERIES or set(identities) != expected_identities:
            raise RuntimeError(f"external evaluated-series identities differ for {name}")
    summaries = {
        name: v1.paired_external_summary([row.episode for row in rows], baseline)
        for name, rows in evaluated.items()
        if name != "mes"
    }
    primary = summaries[f"endowment/seed-{v1.PRIMARY_SEED}"]
    raw_pvalues = {
        city: float(primary[city]["p_value"]) for city in v1.EXPECTED_CITY_SERIES
    }
    if set(raw_pvalues) != set(v1.EXPECTED_CITY_SERIES):
        raise RuntimeError("external Holm family must contain exactly two cities")
    adjusted = v1.holm_adjust(raw_pvalues)
    for city, value in adjusted.items():
        primary[city]["holm_p_value"] = value
    evidence = {
        "primary_seed": str(v1.PRIMARY_SEED),
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
            for seed in v1.SEEDS
        },
        "endowment_actuated_series": v1.count_actuated_series(
            evaluated[f"endowment/seed-{v1.PRIMARY_SEED}"], evaluated["mes"]
        ),
        "priority_actuated_series": v1.count_actuated_series(
            evaluated[f"priority/seed-{v1.PRIMARY_SEED}"], evaluated["mes"]
        ),
        "inference": v1._inference_config(),
    }
    decision = v1.evaluate_external_gate(evidence)
    evaluation_dir = result_root / "evaluation"
    v2.write_immutable_json_artifact(
        evaluation_dir / "per_series.json",
        {
            name: [v1._evaluated_payload(row) for row in rows]
            for name, rows in sorted(evaluated.items())
        },
    )
    v1._write_per_series_csv(
        evaluation_dir / "per_series.csv", {split.name: evaluated}
    )
    v2.write_immutable_json_artifact(evaluation_dir / "summary.json", summaries)
    v2.write_immutable_json_artifact(evaluation_dir / "evidence_payload.json", evidence)
    v2.write_immutable_json_artifact(result_root / "evidence_decision.json", decision)
    return decision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    anchor_parser = subparsers.add_parser("anchor")
    anchor_parser.add_argument("--lock-sha256", required=True)
    anchor_parser.add_argument("--confirmation", required=True)
    subparsers.add_parser("evaluate")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "prepare":
        result = prepare()
    elif args.command == "anchor":
        result = publish_anchor(RESULT_ROOT, args.lock_sha256, args.confirmation)
    else:
        result = evaluate()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
