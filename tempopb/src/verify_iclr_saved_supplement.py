"""Read-only verification of the September 25 saved-results supplement.

Checks stored evidence and manuscript reconstruction, never experimental replay.
The top-level manifest is an integrity inventory; trust its published ZIP digest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

import iclr_manuscript_reconstruct_v2 as reconstruction
from analyze_external_support_set_results import load_and_verify_bundle

WORKSPACE = Path(__file__).resolve().parents[2]
MULTICITY_AUDIT_SHA = "44ff3d85536438f22712004aaef4a9f5db6fd10de32aa520cf4c668cc99e7b31"
FINAL_PDF_SHA = "c63358b5ee78245c05bca4aafebbc55c079ef019b24e5ec68ade96e5ed6e44e7"
CONTEXTUAL_ROWS_SHA = "f9d7aaa03c492e73e72fbe6017f56bee58758c69f0e1ec25200d7454662c2d59"


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("invalid relative path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != relative:
        raise ValueError(f"unsafe relative path: {relative}")
    path = root
    for part in parsed.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink forbidden: {relative}")
    return path


def read_json(path: Path) -> dict:
    return reconstruction._json(reconstruction._read(path), str(path))


def check_hash(root: Path, relative: str, expected: str) -> bytes:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("invalid expected digest")
    raw = reconstruction._read(safe_path(root, relative))
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"digest mismatch: {relative}")
    return raw


def verify_inventory(root: Path) -> int:
    manifest = read_json(safe_path(root, "MANIFEST.json"))
    if manifest.get("schema") != "iclr-saved-results-supplement-v1":
        raise ValueError("wrong supplement schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or "MANIFEST.json" in files:
        raise ValueError("invalid supplement inventory")
    for name, row in files.items():
        if name.endswith(".pb") or name.startswith("tempopb/.venv/"):
            raise ValueError("raw ballot file in saved-results bundle")
        raw = check_hash(root, name, row["sha256"])
        if type(row["bytes"]) is not int or len(raw) != row["bytes"]:
            raise ValueError(f"size mismatch: {name}")
    actual = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        if parent == root / "tempopb" and ".venv" in dirs:
            dirs.remove(".venv")  # uv-created runtime, never a packaged artifact.
        if any((parent / name).is_symlink() for name in dirs + names):
            raise ValueError("symlink in extracted supplement")
        for name in names:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if "__pycache__" in path.parts and path.suffix == ".pyc":
                continue  # Python's local import cache is not evidence.
            actual.add(relative)
    if actual != set(files) | {"MANIFEST.json"}:
        raise ValueError("unexpected or missing extracted files")
    return len(files)


def verify_locks_and_omissions(root: Path) -> int:
    repo = root / "tempopb"
    expected_raw: dict[str, str] = {}
    for lineage, expected_count in (("iclr_primary_warsaw_v2", 132), ("iclr_multicity_v2", 397)):
        lock = read_json(repo / "results" / lineage / "protocol_lock.json")
        raw_count = 0
        for row in lock["tracked_files"].values():
            relative = row["path"]
            path = safe_path(repo, relative)
            if relative.endswith(".pb"):
                if not relative.startswith(("data/pb/", "data/pb_multicity/")):
                    raise ValueError("unexpected raw corpus location")
                if path.exists():
                    raise ValueError("raw ballots must be excluded from this bundle")
                expected_raw["tempopb/" + relative] = row["sha256"]
                raw_count += 1
            else:
                check_hash(repo, relative, row["sha256"])
        if raw_count != expected_count:
            raise ValueError("locked raw inventory cardinality differs")
    omitted = read_json(root / "EXCLUDED_RAW_DATA.json")
    if omitted.get("schema") != "omitted-public-ballot-files-v1":
        raise ValueError("wrong raw-data omission schema")
    recorded = {name: row["sha256"] for name, row in omitted["files"].items()}
    if recorded != expected_raw:
        raise ValueError("omission inventory differs from corrected locks")
    return len(expected_raw)


def verify_multicity(repo: Path) -> None:
    base = repo / "results/iclr_multicity_v2"
    check_hash(base, "v1_v2_delta_audit.json", MULTICITY_AUDIT_SHA)
    audit = read_json(base / "v1_v2_delta_audit.json")
    auth = audit["authenticated_inputs"]
    if audit["status"] != "pass":
        raise ValueError("saved multicity audit did not pass")
    for version in ("v1", "v2"):
        folder = base if version == "v2" else repo / "results/iclr_multicity"
        check_hash(folder, "protocol_lock.json", auth[version + "_protocol_lock_sha256"])
        check_hash(folder, "heldout_opened.json", auth[version + "_heldout_receipt_sha256"])
        for name, digest in auth[version + "_output_sha256"].items():
            check_hash(folder, name, digest)
    check_hash(base, "approval_semantics_receipt.json", auth["v2_semantics_receipt_sha256"])
    check_hash(base, audit["artifacts"]["delta_rows_csv"], audit["artifacts"]["delta_rows_csv_sha256"])
    fits = read_json(base / "heldout_opened.json")["fit_sha256"]
    if len(fits) != 24:
        raise ValueError("wrong multicity fit inventory")
    for name, digest in fits.items():
        check_hash(repo, name, digest)


def verify(root: Path = WORKSPACE) -> dict:
    root = root.resolve()
    file_count = verify_inventory(root)
    omitted_count = verify_locks_and_omissions(root)
    repo = root / "tempopb"
    captured = reconstruction.capture_inputs(repo, allow_missing_raw_data=True)
    payload = reconstruction.reconstruct(captured)
    saved = repo / "results/iclr_primary_warsaw_v2/manuscript_reconstruction"
    for name, raw in reconstruction.render_outputs(payload).items():
        if (saved / name).read_bytes() != raw:
            raise ValueError(f"saved reconstruction differs: {name}")
    expected = {name: r["value"] for name, r in payload["records"].items() if r["value"] is not None}
    final = reconstruction.audit.parse_numbers_tex((root / "manuscript/tex/numbers.tex").read_bytes())
    if final != expected or len(final) != 592:
        raise ValueError("final manuscript macro definitions differ")
    check_hash(root, "manuscript/paper.pdf", FINAL_PDF_SHA)
    contextual = captured["contextual"]
    check_hash(repo, "src/iclr_contextual_lookup_correction.py", contextual["analysis_script_sha256"])
    check_hash(repo, "src/iclr_stats.py", contextual["statistical_helper_sha256"])
    check_hash(repo, "analysis-output/contextual-age-lookup-20260920/per_series.csv", CONTEXTUAL_ROWS_SHA)
    verify_multicity(repo)
    load_and_verify_bundle(repo / "results/iclr_external_support_set_amendment")
    external = repo / "analysis-output/external-support-set"
    for name, digest in read_json(external / "manifest.json")["output_sha256"].items():
        check_hash(external, name, digest)
    reconstruction.recheck_inputs(captured)
    # Recheck the package inventory after all reads to detect concurrent drift.
    verify_inventory(root)
    return {"status": "pass", "scope": "saved results and manuscript values only; no experimental replay",
            "included_files_checked": file_count, "raw_ballot_files_omitted": omitted_count,
            "primary_fits": captured["matrix"]["fit_count"], "multicity_fits": 24,
            "macro_classifications": payload["macro_count"], "final_macro_definitions": len(final),
            "raw_ballot_bytes_rechecked": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
