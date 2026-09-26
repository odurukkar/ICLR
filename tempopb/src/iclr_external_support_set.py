"""One-way v2 confirmation under a fixed support-set ballot projection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import subprocess
from typing import Dict, Mapping, Sequence

import numpy as np

import iclr_external_validation as v1
from iclr_multicity_protocol import write_immutable_json_artifact
from iclr_multicity_train import build_multicity_arm
from cohorts import cohort_of
from parse_pb import PBInstance, Project, Vote


ROOT = v1.ROOT
DATA_DIR = v1.DATA_DIR
RESULT_ROOT = ROOT / "results" / "iclr_external_support_set"
V1_RESULT_ROOT = ROOT / "results" / "iclr_external_validation"
ANCHOR_NAME = v1.ANCHOR_NAME
EXPECTED_VOTE_TYPES = {"Katowice": "cumulative", "Krakow": "ordinal"}
STUDY = "fresh-city-support-set-confirmation-v2"
MIN_JOINT_DEMOGRAPHIC_COVERAGE = 0.5
logger = logging.getLogger(__name__)

_V1_PROTOCOL_CONFIG = v1.external_protocol_config
_V1_TRACKED_FILES = v1.external_tracked_files


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def read_locked_bytes(path: Path, expected_sha256: str) -> bytes:
    """Open once, verify that exact byte string, and return it for consumption."""

    content = Path(path).read_bytes()
    if _sha256_bytes(content) != expected_sha256:
        raise RuntimeError(f"hash mismatch for locked bytes: {Path(path).name}")
    return content


def read_locked_json(path: Path, expected_sha256: str) -> Dict[str, object]:
    content = read_locked_bytes(path, expected_sha256)
    return json.loads(content.decode("utf-8"))


def vote_type_from_bytes(content: bytes, name: str = "<bytes>") -> str:
    vote_type = None
    found_boundary = False
    for raw in content.splitlines():
        line = raw.rstrip(b"\r\n")
        if line == b"PROJECTS":
            found_boundary = True
            break
        if line.startswith(b"vote_type;"):
            vote_type = line.split(b";", 1)[1].decode("utf-8").strip()
    if not found_boundary:
        raise RuntimeError(f"PROJECTS boundary missing after META: {name}")
    if vote_type is None:
        raise RuntimeError(f"vote_type missing from META: {name}")
    return vote_type


def read_vote_type_from_meta(path: Path) -> str:
    """Read only the META prefix and require the PROJECTS boundary."""

    return vote_type_from_bytes(Path(path).read_bytes(), Path(path).name)


def vote_columns_from_bytes(content: bytes, name: str = "<bytes>") -> tuple[str, ...]:
    """Read the VOTES header from bytes and ignore all ballot rows."""

    lines = iter(content.splitlines())
    for raw in lines:
        if raw.rstrip(b"\r\n") == b"VOTES":
            break
    else:
        raise RuntimeError(f"VOTES boundary missing: {name}")
    for raw in lines:
        line = raw.rstrip(b"\r\n")
        if line:
            return tuple(value.strip() for value in line.decode("utf-8").split(";"))
    raise RuntimeError(f"VOTES header missing: {name}")


def read_vote_columns(path: Path) -> tuple[str, ...]:
    """Read the VOTES header and stop before the first ballot row."""

    return vote_columns_from_bytes(Path(path).read_bytes(), Path(path).name)


def support_set_metadata_from_bytes(
    selected: Sequence[v1.SelectedFile], source_bytes: Mapping[str, bytes]
) -> Dict[str, Dict[str, object]]:
    """Derive every schema claim from the authenticated byte snapshot."""

    file_meta: Dict[str, Dict[str, object]] = {}
    for row in selected:
        content = source_bytes[row.path.name]
        observed = vote_type_from_bytes(content, row.path.name)
        expected = EXPECTED_VOTE_TYPES[row.city]
        if observed != expected:
            raise RuntimeError(
                f"external vote type differs for {row.path.name}: "
                f"expected {expected}, observed {observed}"
            )
        columns = vote_columns_from_bytes(content, row.path.name)
        required = {"voter_id", "vote"}
        if observed == "cumulative":
            required.add("points")
        if not required.issubset(columns):
            raise RuntimeError(f"external VOTES header differs for {row.path.name}")
        has_age_sex = {"age", "sex"}.issubset(columns)
        expected_demographics = row.city == "Katowice" or row.year >= v1.FIRST_SCORE_YEAR
        if has_age_sex != expected_demographics:
            raise RuntimeError(f"external demographic schema differs for {row.path.name}")
        file_meta[row.path.name] = {
            "vote_type": observed,
            "vote_columns": list(columns),
            "has_age_sex": has_age_sex,
        }
    return file_meta


def _validate_full_schema_counts(
    selected: Sequence[v1.SelectedFile], file_meta: Mapping[str, Mapping[str, object]]
) -> None:
    if len(selected) != v1.EXPECTED_ELECTIONS:
        raise RuntimeError("external schema selection count differs")
    if sum(bool(row["has_age_sex"]) for row in file_meta.values()) != 138:
        raise RuntimeError("external demographic-file count differs")
    score_names = {
        row.path.name for row in selected if row.year >= v1.FIRST_SCORE_YEAR
    }
    if sum(bool(file_meta[name]["has_age_sex"]) for name in score_names) != 96:
        raise RuntimeError("not every scored election has age and sex")


def select_support_set_files(
    source_dir: Path,
) -> tuple[list[v1.SelectedFile], Dict[str, Dict[str, object]]]:
    """Apply v1 filenames and screen result-independent META/header fields."""

    selected = v1.select_external_files(source_dir)
    source_bytes = {row.path.name: row.path.read_bytes() for row in selected}
    file_meta = support_set_metadata_from_bytes(selected, source_bytes)
    _validate_full_schema_counts(selected, file_meta)
    return selected, file_meta


def external_protocol_config() -> Dict[str, object]:
    """Return v1 semantics with only the explicit projection and lineage changed."""

    config = copy.deepcopy(_V1_PROTOCOL_CONFIG())
    config["study"] = STUDY
    config["canonical_paths"]["result_root"] = str(RESULT_ROOT.relative_to(ROOT))
    config["ballot_projection"] = {
        "Katowice": "cumulative listed-project support; ignore point magnitude",
        "Krakow": "ordinal listed-project support; ignore list order",
    }
    config["demographic_schema"] = {
        "all_scored_elections_have_age_and_sex": True,
        "demographic_warmup_elections": 42,
        "minimum_joint_valid_age_sex_coverage_per_scored_election": (
            MIN_JOINT_DEMOGRAPHIC_COVERAGE
        ),
        "nondemographic_warmup_elections": 90,
    }
    config["prior_attempt"] = {
        "study": "fresh-city-endowment-confirmation-v1",
        "classification": "protocol_invalid",
        "metrics_computed": False,
    }
    return config


def require_canonical_paths(data_dir: Path, result_root: Path) -> None:
    if Path(data_dir).resolve() != DATA_DIR.resolve() or Path(result_root).resolve() != RESULT_ROOT.resolve():
        raise RuntimeError(
            "support-set confirmation requires canonical paths: "
            f"data={DATA_DIR}, results={RESULT_ROOT}"
        )


def assert_evaluation_unopened(result_root: Path) -> None:
    v1.assert_evaluation_unopened(result_root)
    if (Path(result_root) / "evaluation" / "demographic_coverage.json").exists():
        raise RuntimeError("external evaluation is one-way and has already been opened")


def _git_blob_sha1_bytes(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def authenticated_source_bytes(
    source_dir: Path, selected: Sequence[v1.SelectedFile]
) -> tuple[Dict[str, str], Dict[str, bytes]]:
    """Authenticate each selected Git blob from the same bytes later staged."""

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

    if git("rev-parse", "HEAD") != v1.PABULIB_COMMIT:
        raise RuntimeError("Pabulib checkout HEAD differs from the pinned commit")
    origin = git("remote", "get-url", "origin").removesuffix("/").removesuffix(".git")
    expected_origin = v1.PABULIB_REMOTE.removesuffix("/").removesuffix(".git")
    if origin != expected_origin:
        raise RuntimeError(f"Pabulib checkout origin differs: {origin!r}")
    tree_output = git("ls-tree", "-r", v1.PABULIB_COMMIT, "--", "pb_files")
    tree = {}
    for line in tree_output.splitlines():
        metadata, relative = line.split("\t", 1)
        tree[relative] = metadata.split()[2]

    blobs: Dict[str, str] = {}
    content_by_name: Dict[str, bytes] = {}
    for row in selected:
        content = row.path.read_bytes()
        observed_blob = _git_blob_sha1_bytes(content)
        expected_blob = tree.get(f"pb_files/{row.path.name}")
        if expected_blob is None or observed_blob != expected_blob:
            raise RuntimeError(f"selected Pabulib blob differs from commit: {row.path.name}")
        blobs[row.path.name] = observed_blob
        content_by_name[row.path.name] = content
    return blobs, content_by_name


def _manifest_payload(
    selected: Sequence[v1.SelectedFile],
    source_blobs: Mapping[str, str],
    source_bytes: Mapping[str, bytes],
    file_meta: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    rows = []
    for selected_file in selected:
        name = selected_file.path.name
        content = source_bytes[name]
        rows.append(
            {
                "name": name,
                "city_token": selected_file.city,
                "series_token": selected_file.token,
                "year": selected_file.year,
                "vote_type": file_meta[name]["vote_type"],
                "vote_columns": file_meta[name]["vote_columns"],
                "has_age_sex": file_meta[name]["has_age_sex"],
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
                "git_blob_sha1": source_blobs[name],
            }
        )
    return {
        "schema_version": 2,
        "source_commit": v1.PABULIB_COMMIT,
        "selection": "v1 filenames plus META/VOTES-header city schema screen",
        "projection": external_protocol_config()["ballot_projection"],
        "n_series": v1.EXPECTED_SERIES,
        "n_elections": v1.EXPECTED_ELECTIONS,
        "n_warmup": v1.EXPECTED_WARMUP,
        "n_score": v1.EXPECTED_SCORE,
        "files": rows,
    }


def stage_support_set_corpus(
    selected: Sequence[v1.SelectedFile],
    data_dir: Path,
    manifest_path: Path,
    source_blobs: Mapping[str, str],
    source_bytes: Mapping[str, bytes],
    file_meta: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """Stage the exact authenticated bytes without parsing ballots."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    expected = {row.path.name for row in selected}
    unexpected = sorted(path.name for path in data_dir.glob("*.pb") if path.name not in expected)
    if unexpected:
        raise RuntimeError(f"external data directory has unexpected files: {unexpected[:5]}")
    if set(source_blobs) != expected or set(source_bytes) != expected or set(file_meta) != expected:
        raise RuntimeError("authenticated source or schema inventory differs")
    for row in selected:
        target = data_dir / row.path.name
        content = source_bytes[row.path.name]
        if target.is_file():
            if target.read_bytes() != content:
                raise RuntimeError(f"staged file differs from source: {target.name}")
        else:
            with target.open("xb") as stream:
                stream.write(content)
    payload = _manifest_payload(selected, source_blobs, source_bytes, file_meta)
    write_immutable_json_artifact(manifest_path, payload)
    return payload


def v1_lineage_files() -> Dict[str, Path]:
    return {
        "v1/protocol_lock": V1_RESULT_ROOT / "protocol_lock.json",
        "v1/published_anchor": V1_RESULT_ROOT / "published_lock_anchor.json",
        "v1/heldout_receipt": V1_RESULT_ROOT / "heldout_opened.json",
        "v1/abort_record": V1_RESULT_ROOT / "abort_record.json",
        "v1/corpus_manifest": V1_RESULT_ROOT / "corpus_manifest.json",
    }


def _assert_v1_has_no_evidence_outputs() -> None:
    guarded = (
        V1_RESULT_ROOT / "evidence_decision.json",
        V1_RESULT_ROOT / "evaluation" / "per_series.json",
        V1_RESULT_ROOT / "evaluation" / "per_series.csv",
        V1_RESULT_ROOT / "evaluation" / "summary.json",
        V1_RESULT_ROOT / "evaluation" / "evidence_payload.json",
    )
    if any(path.exists() for path in guarded):
        raise RuntimeError("v1 evidence output exists despite the recorded abort")


def verify_v1_protocol_lock_bytes(lock_bytes: bytes) -> Dict[str, object]:
    payload = json.loads(lock_bytes.decode("utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("v1 protocol lock schema differs")
    if payload.get("protocol") != _V1_PROTOCOL_CONFIG():
        raise RuntimeError("v1 protocol semantics differ")
    tracked = payload.get("tracked_files")
    if not isinstance(tracked, dict) or not tracked:
        raise RuntimeError("v1 protocol tracked-file inventory differs")
    for label, row in tracked.items():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RuntimeError(f"v1 protocol tracked row differs for {label}")
    return payload


def _expected_v1_receipt(
    lock_payload: Mapping[str, object], lock_sha: str
) -> Dict[str, object]:
    tracked = lock_payload["tracked_files"]
    fit_labels = sorted(v1.original_fit_paths())
    return {
        "schema_version": 1,
        "protocol_lock_sha256": lock_sha,
        "corpus_manifest_sha256": tracked["artifact/corpus_manifest"]["sha256"],
        "fixed_fit_sha256": {
            label: tracked[label]["sha256"] for label in fit_labels
        },
    }


def _verify_v1_abort_payloads(
    lock_bytes: bytes, anchor_bytes: bytes, receipt_bytes: bytes, abort_bytes: bytes
) -> Dict[str, object]:
    lock_payload = verify_v1_protocol_lock_bytes(lock_bytes)
    lock_sha = _sha256_bytes(lock_bytes)
    anchor = json.loads(anchor_bytes.decode("utf-8"))
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    abort = json.loads(abort_bytes.decode("utf-8"))
    expected_anchor = {
        "schema_version": 1,
        "study": "fresh-city-endowment-confirmation-v1",
        "lock_sha256": lock_sha,
        "confirmation": f"confirm {lock_sha}",
    }
    expected_abort = {
        "schema_version": 1,
        "classification": "protocol_invalid",
        "error": "external file is not approval voting: Poland_Katowice_2020_Bogucice.pb",
        "metrics_computed": False,
        "observed_vote_types": {
            "Poland/Katowice": {"files": 84, "vote_type": "cumulative"},
            "Poland/Krakow": {"files": 144, "vote_type": "ordinal"},
        },
        "protocol_lock_sha256": lock_sha,
    }
    if anchor != expected_anchor:
        raise RuntimeError("v1 published anchor differs from the failed lock")
    if receipt != _expected_v1_receipt(lock_payload, lock_sha):
        raise RuntimeError("v1 held-out receipt differs from the failed lock")
    if abort != expected_abort:
        raise RuntimeError("v1 abort record is contradictory or incomplete")
    return lock_payload


def verify_v1_abort() -> None:
    files = v1_lineage_files()
    missing = [label for label, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v1 abort lineage is missing: {missing}")
    _assert_v1_has_no_evidence_outputs()
    lock_bytes = files["v1/protocol_lock"].read_bytes()
    lock_payload = _verify_v1_abort_payloads(
        lock_bytes,
        files["v1/published_anchor"].read_bytes(),
        files["v1/heldout_receipt"].read_bytes(),
        files["v1/abort_record"].read_bytes(),
    )
    for label, row in lock_payload["tracked_files"].items():
        path = ROOT / row["path"]
        if not path.is_file() or _sha256_bytes(path.read_bytes()) != row["sha256"]:
            raise RuntimeError(f"v1 protocol tracked file differs for {label}")


def external_tracked_files(
    data_dir: Path = DATA_DIR, result_root: Path = RESULT_ROOT
) -> Dict[str, Path]:
    files = _V1_TRACKED_FILES(data_dir, result_root)
    files["test/external_support_set"] = ROOT / "tests" / "test_iclr_external_support_set.py"
    files.update(v1_lineage_files())
    missing = [label for label, path in files.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"support-set lock files missing: {missing[:5]}")
    return files


def write_external_lock(
    lock_path: Path, repo_root: Path, tracked_files: Mapping[str, Path]
) -> str:
    repo_root = Path(repo_root).resolve()
    tracked = {}
    for label, path in sorted(tracked_files.items()):
        resolved = Path(path).resolve()
        content = resolved.read_bytes()
        tracked[label] = {
            "path": str(resolved.relative_to(repo_root)),
            "sha256": _sha256_bytes(content),
        }
    payload = {
        "schema_version": 2,
        "protocol": external_protocol_config(),
        "tracked_files": tracked,
    }
    return write_immutable_json_artifact(Path(lock_path), payload)


def _verify_v1_against_locked_snapshot(
    v2_payload: Mapping[str, object], locked_bytes: Mapping[str, bytes]
) -> None:
    required = set(v1_lineage_files())
    if not required.issubset(locked_bytes):
        missing = sorted(required - set(locked_bytes))
        raise RuntimeError(f"v1 lineage labels are missing: {missing}")
    v1_lock = _verify_v1_abort_payloads(
        locked_bytes["v1/protocol_lock"],
        locked_bytes["v1/published_anchor"],
        locked_bytes["v1/heldout_receipt"],
        locked_bytes["v1/abort_record"],
    )
    by_path = {
        row["path"]: label
        for label, row in v2_payload["tracked_files"].items()
    }
    for label, row in v1_lock["tracked_files"].items():
        v2_label = by_path.get(row["path"])
        if v2_label is None or _sha256_bytes(locked_bytes[v2_label]) != row["sha256"]:
            raise RuntimeError(f"v1 protocol snapshot differs for {label}")
    _assert_v1_has_no_evidence_outputs()


def load_external_lock(
    lock_path: Path, repo_root: Path, expected_files: Mapping[str, Path]
) -> tuple[Dict[str, object], Dict[str, bytes], bytes]:
    """Verify and retain the exact bytes that evaluation may consume."""

    repo_root = Path(repo_root).resolve()
    lock_bytes = Path(lock_path).read_bytes()
    payload = json.loads(lock_bytes.decode("utf-8"))
    if payload.get("schema_version") != 2:
        raise RuntimeError("support-set lock schema differs")
    if payload.get("protocol") != external_protocol_config():
        raise RuntimeError("support-set protocol differs from canonical semantics")
    tracked = payload.get("tracked_files", {})
    if set(tracked) != set(expected_files):
        raise RuntimeError("support-set tracked-file set differs")
    locked_bytes: Dict[str, bytes] = {}
    for label, expected in expected_files.items():
        resolved = Path(expected).resolve()
        relative = str(resolved.relative_to(repo_root))
        if tracked[label].get("path") != relative:
            raise RuntimeError(f"support-set tracked path differs for {label}")
        content = resolved.read_bytes()
        if tracked[label].get("sha256") != _sha256_bytes(content):
            raise RuntimeError(f"hash mismatch for tracked file {label}")
        locked_bytes[label] = content
    _verify_v1_against_locked_snapshot(payload, locked_bytes)
    return payload, locked_bytes, lock_bytes


def verify_external_lock(
    lock_path: Path, repo_root: Path, expected_files: Mapping[str, Path]
) -> Dict[str, object]:
    payload, _, _ = load_external_lock(lock_path, repo_root, expected_files)
    return payload


def write_published_anchor(
    anchor_path: Path, lock_sha256: str, confirmation: str
) -> str:
    if confirmation != f"confirm {lock_sha256}":
        raise RuntimeError("published lock confirmation text differs")
    return write_immutable_json_artifact(
        Path(anchor_path),
        {
            "schema_version": 1,
            "study": STUDY,
            "lock_sha256": lock_sha256,
            "confirmation": confirmation,
        },
    )


def _verify_anchor_for_digest(anchor_path: Path, lock_sha256: str) -> Dict[str, object]:
    payload = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "study": STUDY,
        "lock_sha256": lock_sha256,
        "confirmation": f"confirm {lock_sha256}",
    }
    if payload != expected:
        raise RuntimeError("published v2 lock anchor does not match the protocol lock")
    return payload


def verify_published_anchor(anchor_path: Path, lock_path: Path) -> Dict[str, object]:
    return _verify_anchor_for_digest(anchor_path, _sha256_bytes(Path(lock_path).read_bytes()))


def _parse_int(value: str):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_support_set_bytes(
    content: bytes, source_path: Path, expected_vote_type: str
) -> PBInstance:
    """Parse the verified bytes while projecting each ballot to unique support."""

    instance = PBInstance(path=str(source_path))
    section = None
    header: list[str] = []
    vote_rows = 0
    text = content.decode("utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        upper = line.strip().upper()
        if upper in ("META", "PROJECTS", "VOTES"):
            section = upper
            header = []
            continue
        fields = line.split(";")
        if section == "META":
            if len(fields) >= 2 and fields[0] != "key":
                instance.meta[fields[0].strip()] = fields[1].strip()
        elif section == "PROJECTS":
            if not header:
                header = [field.strip() for field in fields]
                continue
            row = dict(zip(header, fields))
            pid = row.get("project_id", "").strip()
            if not pid:
                continue
            try:
                cost = float(row.get("cost", "0").replace(",", "."))
            except ValueError:
                cost = 0.0
            instance.projects[pid] = Project(
                pid=pid,
                cost=cost,
                selected=_parse_int(row.get("selected", "")),
                name=row.get("name", ""),
                category=row.get("category", ""),
                target=row.get("target", ""),
                neighborhood=row.get("neighborhood", ""),
            )
        elif section == "VOTES":
            if not header:
                header = [field.strip() for field in fields]
                required = {"voter_id", "vote"}
                if expected_vote_type == "cumulative":
                    required.add("points")
                if not required.issubset(header):
                    raise RuntimeError(f"support-set columns differ: {Path(source_path).name}")
                continue
            row = dict(zip(header, fields))
            projects = tuple(value for value in row.get("vote", "").split(",") if value)
            if not projects:
                raise RuntimeError(f"support-set ballot is empty: {Path(source_path).name}")
            if len(projects) != len(set(projects)):
                raise RuntimeError(f"duplicate project in support-set ballot: {Path(source_path).name}")
            unknown = set(projects) - set(instance.projects)
            if unknown:
                raise RuntimeError(f"unknown project in support-set ballot: {Path(source_path).name}")
            if expected_vote_type == "cumulative":
                points = tuple(value for value in row.get("points", "").split(",") if value)
                if len(points) != len(projects):
                    raise RuntimeError(f"support-set vector length differs: {Path(source_path).name}")
                try:
                    numeric = [float(value) for value in points]
                except ValueError as exc:
                    raise RuntimeError(f"cumulative points are nonnumeric: {Path(source_path).name}") from exc
                if any(not math.isfinite(value) or value <= 0 for value in numeric):
                    raise RuntimeError(f"cumulative listed points must be positive: {Path(source_path).name}")
            instance.votes.append(
                Vote(
                    vid=row.get("voter_id", "").strip(),
                    projects=projects,
                    age=_parse_int(row.get("age", "")),
                    sex=row.get("sex", "").strip().upper(),
                    neighborhood=row.get("neighborhood", "").strip(),
                )
            )
            vote_rows += 1
    if instance.vote_type != expected_vote_type:
        raise RuntimeError(f"parsed support-set vote type differs: {Path(source_path).name}")
    if vote_rows == 0:
        raise RuntimeError(f"support-set ballot section is empty: {Path(source_path).name}")
    return instance


def validate_support_set_rows(path: Path, vote_type: str) -> int:
    instance = parse_support_set_bytes(Path(path).read_bytes(), path, vote_type)
    return len(instance.votes)


def load_external_corpus(
    manifest: Mapping[str, object],
    data_dir: Path,
    receipt_path: Path,
    locked_bytes: Mapping[str, bytes],
) -> tuple[Dict[str, v1.SeriesRef], Dict[str, Dict[int, object]]]:
    if not Path(receipt_path).is_file():
        raise RuntimeError("external held-out receipt must exist before PB parsing")
    by_series: dict[str, dict[int, Path]] = defaultdict(dict)
    instances: dict[str, dict[int, object]] = defaultdict(dict)
    token_to_key: dict[tuple[str, str], str] = {}
    for row in manifest["files"]:
        path = Path(data_dir) / row["name"]
        content = locked_bytes[f"corpus/{row['name']}"]
        if len(content) != int(row["bytes"]) or _sha256_bytes(content) != row["sha256"]:
            raise RuntimeError(f"locked external bytes differ: {path.name}")
        expected_type = EXPECTED_VOTE_TYPES[row["city_token"]]
        if row.get("vote_type") != expected_type:
            raise RuntimeError(f"locked support-set vote type differs: {path.name}")
        actual_type = vote_type_from_bytes(content, path.name)
        actual_columns = list(vote_columns_from_bytes(content, path.name))
        if actual_type != row.get("vote_type") or actual_columns != row.get("vote_columns"):
            raise RuntimeError(f"locked support-set schema differs: {path.name}")
        expected_demo = row["city_token"] == "Katowice" or int(row["year"]) >= v1.FIRST_SCORE_YEAR
        actual_demo = {"age", "sex"}.issubset(actual_columns)
        if bool(row.get("has_age_sex")) != expected_demo or actual_demo != expected_demo:
            raise RuntimeError(f"locked demographic schema differs: {path.name}")
        instance = parse_support_set_bytes(content, path, expected_type)
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


def joint_age_sex_coverage(instance: PBInstance) -> float:
    if not instance.votes:
        return 0.0
    valid = sum(cohort_of(vote, "age_sex") is not None for vote in instance.votes)
    return valid / len(instance.votes)


def demographic_coverage_report(
    instances: Mapping[str, Mapping[int, PBInstance]],
    score_years: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    rows = []
    for series, years in sorted(score_years.items()):
        for year in sorted(years):
            instance = instances[series][int(year)]
            valid = sum(
                cohort_of(vote, "age_sex") is not None for vote in instance.votes
            )
            total = len(instance.votes)
            rows.append(
                {
                    "series": series,
                    "year": int(year),
                    "n_votes": total,
                    "n_joint_valid_age_sex": valid,
                    "coverage": valid / total if total else 0.0,
                }
            )
    return {
        "schema_version": 1,
        "threshold": MIN_JOINT_DEMOGRAPHIC_COVERAGE,
        "n_scored_elections": len(rows),
        "minimum_observed": min((row["coverage"] for row in rows), default=0.0),
        "files": rows,
    }


def enforce_demographic_coverage(report: Mapping[str, object]) -> None:
    rows = report.get("files", [])
    if int(report.get("n_scored_elections", -1)) != v1.EXPECTED_SCORE:
        # Unit tests may exercise a reduced, explicit family.
        if not rows:
            raise RuntimeError("demographic coverage family is empty")
    failed = [
        row for row in rows
        if float(row["coverage"]) < MIN_JOINT_DEMOGRAPHIC_COVERAGE
    ]
    if failed:
        raise RuntimeError(
            "scored-election demographic coverage is below the locked threshold: "
            f"{failed[0]['series']} {failed[0]['year']}={failed[0]['coverage']:.4f}"
        )


def require_demographic_coverage(
    instances: Mapping[str, Mapping[int, PBInstance]],
    score_years: Mapping[str, Sequence[int]],
) -> Dict[str, object]:
    report = demographic_coverage_report(instances, score_years)
    enforce_demographic_coverage(report)
    return report


def _fit_selector_from_payload(
    payload: Mapping[str, object], expected_arm: str, expected_seed: int, cfg
):
    config = payload.get("config", {})
    if config.get("arm") != expected_arm or int(config.get("seed", -1)) != expected_seed:
        raise RuntimeError("locked fit identity differs")
    arm = build_multicity_arm(expected_arm)
    weights = np.asarray(payload.get("result", {}).get("best_weights", []), dtype=float)
    if weights.shape != arm.initial.shape or not np.all(np.isfinite(weights)):
        raise RuntimeError("locked fit weights differ")
    return arm.selector(weights, cfg)


def prepare(source_dir: Path, data_dir: Path, result_root: Path) -> Dict[str, object]:
    result_root = Path(result_root)
    require_canonical_paths(data_dir, result_root)
    assert_evaluation_unopened(result_root)
    if (result_root / ANCHOR_NAME).exists():
        raise RuntimeError("published lock anchor already exists")
    v1.verify_original_study()
    verify_v1_abort()
    selected = v1.select_external_files(source_dir)
    source_blobs, source_bytes = authenticated_source_bytes(source_dir, selected)
    file_meta = support_set_metadata_from_bytes(selected, source_bytes)
    _validate_full_schema_counts(selected, file_meta)
    manifest = stage_support_set_corpus(
        selected,
        data_dir,
        result_root / "corpus_manifest.json",
        source_blobs,
        source_bytes,
        file_meta,
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
        "meta_vote_types": {"cumulative": 84, "ordinal": 144},
        "demographic_schema_files": 138,
    }


def publish_anchor(
    result_root: Path, lock_sha256: str, confirmation: str
) -> Dict[str, object]:
    require_canonical_paths(DATA_DIR, result_root)
    assert_evaluation_unopened(result_root)
    verify_v1_abort()
    lock_path = Path(result_root) / "protocol_lock.json"
    actual = _sha256_bytes(lock_path.read_bytes())
    if actual != lock_sha256:
        raise RuntimeError("confirmed lock hash differs from the canonical v2 lock")
    anchor_path = Path(result_root) / ANCHOR_NAME
    anchor_sha256 = write_published_anchor(anchor_path, lock_sha256, confirmation)
    _verify_anchor_for_digest(anchor_path, actual)
    return {
        "status": "anchored",
        "lock_sha256": lock_sha256,
        "anchor_sha256": anchor_sha256,
    }


def evaluate(data_dir: Path, result_root: Path) -> Dict[str, object]:
    """Consume exact locked bytes, evaluate fixed policies, and decide once."""

    result_root = Path(result_root)
    require_canonical_paths(data_dir, result_root)
    receipt_path = result_root / "heldout_opened.json"
    assert_evaluation_unopened(result_root)
    v1.verify_original_study()
    verify_v1_abort()
    tracked = external_tracked_files(data_dir, result_root)
    lock_path = result_root / "protocol_lock.json"
    _, locked_bytes, lock_bytes = load_external_lock(lock_path, ROOT, tracked)
    lock_sha = _sha256_bytes(lock_bytes)
    _verify_anchor_for_digest(result_root / ANCHOR_NAME, lock_sha)
    _assert_v1_has_no_evidence_outputs()
    _verify_v1_abort_payloads(
        locked_bytes["v1/protocol_lock"],
        locked_bytes["v1/published_anchor"],
        locked_bytes["v1/heldout_receipt"],
        locked_bytes["v1/abort_record"],
    )
    manifest = json.loads(locked_bytes["artifact/corpus_manifest"].decode("utf-8"))
    receipt = {
        "schema_version": 2,
        "protocol_lock_sha256": lock_sha,
        "corpus_manifest_sha256": _sha256_bytes(
            locked_bytes["artifact/corpus_manifest"]
        ),
        "fixed_fit_sha256": {
            label: _sha256_bytes(locked_bytes[label])
            for label in sorted(v1.original_fit_paths())
        },
        "ballot_projection": external_protocol_config()["ballot_projection"],
    }
    write_immutable_json_artifact(receipt_path, receipt)

    index, instances = load_external_corpus(
        manifest, data_dir, receipt_path, locked_bytes
    )
    split = v1.external_split(index)
    coverage = demographic_coverage_report(instances, dict(split.test))
    write_immutable_json_artifact(
        result_root / "evaluation" / "demographic_coverage.json", coverage
    )
    enforce_demographic_coverage(coverage)
    cfg = v1.EnvConfig()
    policies = {"mes": v1.endowment_selector(v1.uniform_policy, cfg)}
    for arm in ("endowment", "priority"):
        for seed in v1.SEEDS:
            label = f"fit/{arm}/seed-{seed}"
            payload = json.loads(locked_bytes[label].decode("utf-8"))
            policies[f"{arm}/seed-{seed}"] = _fit_selector_from_payload(
                payload, arm, seed, cfg
            )
    evaluated = {}
    for name, selector in sorted(policies.items()):
        logger.info("evaluating %s on support-set projected external cities", name)
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
    write_immutable_json_artifact(
        evaluation_dir / "per_series.json",
        {
            name: [v1._evaluated_payload(row) for row in rows]
            for name, rows in sorted(evaluated.items())
        },
    )
    v1._write_per_series_csv(
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
    anchor_parser = subparsers.add_parser("anchor")
    anchor_parser.add_argument("--lock-sha256", required=True)
    anchor_parser.add_argument("--confirmation", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    for command_parser in (prepare_parser, anchor_parser, evaluate_parser):
        command_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
        command_parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    require_canonical_paths(args.data_dir, args.result_root)
    if args.command == "prepare":
        result = prepare(args.source_dir, args.data_dir, args.result_root)
    elif args.command == "anchor":
        result = publish_anchor(args.result_root, args.lock_sha256, args.confirmation)
    else:
        result = evaluate(args.data_dir, args.result_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
