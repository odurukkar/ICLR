"""Collect saved evidence without experiments; write one new ZIP only with --write."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT = "output/artifact/ICLR2027_Coverage_Is_Not_Fairness_REPRODUCIBILITY_2026-09-25_r3.zip"
PRIMARY = "tempopb/results/iclr_primary_warsaw_v2"
MULTICITY = "tempopb/results/iclr_multicity_v2"
FINAL_SOURCE = "output/artifact/ICLR2027_Coverage_Is_Not_Fairness_SOURCE_2026-09-25_r3.zip"
FINAL_PDF = "output/pdf/ICLR2027_Coverage_Is_Not_Fairness_FINAL_2026-09-25_r3.pdf"
FINAL_SOURCE_SHA = "67dd5fa182897b122530b7764a78fbb74b16092288170e77591fe1ac2bf23792"
FINAL_PDF_SHA = "c63358b5ee78245c05bca4aafebbc55c079ef019b24e5ec68ade96e5ed6e44e7"
MC_AUDIT_SHA = "44ff3d85536438f22712004aaef4a9f5db6fd10de32aa520cf4c668cc99e7b31"
RECON_MANIFEST_SHA = "76b3b93a4d0b994150ebe853aa278d78aefa05857c17428f40c1473d129bf049"
EXTERNAL_MANIFEST_SHA = "d9fff666a907b8c398477bd0fbae791737919d89892714020659677c8649bc73"
CONTEXTUAL_CSV_SHA = "f9d7aaa03c492e73e72fbe6017f56bee58758c69f0e1ec25200d7454662c2d59"
HISTORICAL = "protected historical reproducibility ZIP"
SYNTHETIC_PATH_FIXTURES = {
    "tests/test_iclr_artifact_manifest.py": "/" + "Users/reviewer/secret/project/data/pb_multicity",
    "tests/test_iclr_multicity_protocol.py": "/" + "Users/author/project/data/pb_multicity",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def safe_name(name: str, *, allow_raw: bool = False) -> str:
    path = PurePosixPath(name)
    if (not name or not path.parts or path.is_absolute() or path.as_posix() != name or ":" in name
            or "\\" in name or any(ord(char) < 32 for char in name)
            or any(part in (".", "..") for part in path.parts)):
        raise ValueError(f"unsafe archive path: {name!r}")
    if not allow_raw and path.suffix.lower() == ".pb":
        raise ValueError(f"raw ballot file is excluded: {name}")
    return name


def check_content(raw: bytes, name: str) -> None:
    # Assemble the pattern so the included checker does not match its own text.
    pattern = rb"/" + rb"(?:Users|home)/[A-Za-z0-9_.-]+(?:/|\b)"
    windows = rb"[A-Za-z]:[\\/]" + rb"Users[\\/][A-Za-z0-9_. -]+[\\/]"
    fixture = SYNTHETIC_PATH_FIXTURES.get(name.removeprefix("tempopb/"))
    for match in re.finditer(pattern + b"|" + windows, raw):
        if fixture is None or not raw.startswith(fixture.encode(), match.start()):
            raise ValueError(f"user-identifying absolute path in {name}; no redaction performed")


def zip_members(raw: bytes) -> dict[str, bytes]:
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for item in archive.infolist():
            name = safe_name(item.filename.rstrip("/") if item.is_dir() else item.filename)
            mode = item.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                raise ValueError(f"nonregular ZIP member: {name}")
            if item.is_dir():
                continue
            if name in result:
                raise ValueError(f"duplicate ZIP member: {name}")
            content = archive.read(item)
            check_content(content, name)
            result[name] = content
    return result


class Inventory:
    def __init__(self, workspace: Path):
        if workspace.is_symlink():
            raise ValueError("workspace cannot be a symlink")
        self.workspace = workspace.resolve()
        self.files: dict[str, bytes] = {}
        self.origins: dict[str, str] = {}
        self.replacements: list[dict] = []

    def path(self, name: str) -> Path:
        path = self.workspace
        for part in PurePosixPath(safe_name(name)).parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"symlink input or parent: {name}")
        return path

    def read(self, name: str, expected: str | None = None) -> bytes:
        path = self.path(name)
        if not path.is_file():
            raise ValueError(f"required regular file missing: {name}")
        raw = path.read_bytes()
        if expected is not None and (not re.fullmatch(r"[0-9a-f]{64}", expected) or sha(raw) != expected):
            raise ValueError(f"digest mismatch: {name}")
        return raw

    def add(self, name: str, raw: bytes, origin: str, *, replace_historical: bool = False) -> None:
        safe_name(name)
        check_content(raw, name)
        if name in self.files and self.files[name] != raw:
            if not (replace_historical and self.origins[name] == HISTORICAL
                    and name.startswith(("tempopb/results/", "tempopb/analysis-output/"))):
                raise ValueError(f"conflicting duplicate input: {name}")
            self.replacements.append({"path": name, "historical_sha256": sha(self.files[name]),
                                      "current_sha256": sha(raw), "authenticated_by": origin})
        self.files[name], self.origins[name] = raw, origin

    def file(self, name: str, expected: str | None = None, *, destination: str | None = None,
             origin: str | None = None, replace_historical: bool = False) -> bytes:
        raw = self.read(name, expected)
        self.add(destination or name, raw, origin or name, replace_historical=replace_historical)
        return raw


def constants(raw: bytes, names: tuple[str, ...]) -> dict:
    """Read only literal pin declarations; never import an experimental module."""
    found = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in names:
                value = node.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Path":
                    value = value.args[0]
                found[key] = ast.literal_eval(value)
    if set(found) != set(names):
        raise ValueError("required literal pins missing")
    return found


def import_closure(inventory: Inventory) -> None:
    """Include local Python dependencies, including imports inside functions/tests."""
    pending = [name for name in inventory.files if name.endswith(".py")]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse(inventory.files[name], filename=name)):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                for directory in ("tempopb/src", "tempopb/tests"):
                    relative = directory + "/" + module.replace(".", "/") + ".py"
                    if inventory.path(relative).is_file() and relative not in seen:
                        inventory.file(relative)
                        pending.append(relative)


def collect(workspace: Path = WORKSPACE) -> dict[str, bytes]:
    inventory = Inventory(workspace)
    audit_names = tuple(f"PROTECTED_{kind}_{suffix}" for kind in ("PDF", "REPRO", "SOURCE")
                        for suffix in ("RELATIVE", "SHA256")) + ("PROTECTED_NUMBERS_SHA256",)
    pins = constants(inventory.read("tempopb/src/iclr_primary_warsaw_delta_audit_v2.py"), audit_names)
    reconstruction_pins = constants(inventory.read("tempopb/src/iclr_manuscript_reconstruct_v2.py"),
                                    ("COMPAT_SHA", "WORKING_NUMBERS_SHA", "CONTEXTUAL_SHA"))
    for kind in ("PDF", "REPRO", "SOURCE"):
        inventory.file(pins[f"PROTECTED_{kind}_RELATIVE"], pins[f"PROTECTED_{kind}_SHA256"])
    old = zip_members(inventory.files[pins["PROTECTED_REPRO_RELATIVE"]])
    original_source = zip_members(inventory.files[pins["PROTECTED_SOURCE_RELATIVE"]])
    if sha(original_source["numbers.tex"]) != pins["PROTECTED_NUMBERS_SHA256"]:
        raise ValueError("historical numbers digest mismatch")
    for name, raw in old.items():
        if name.startswith(("results/", "analysis-output/")) or name in ("DATA_MANIFEST.tsv", "PABULIB_SOURCE.md"):
            inventory.add("tempopb/" + name, raw, HISTORICAL)
    compat = json.loads(inventory.file(PRIMARY + "/v1_v2_delta_compat_audit.json", reconstruction_pins["COMPAT_SHA"]))
    bindings = compat["compatibility_amendment"]["sealed_file_sha256"]
    if len(bindings) != 288:
        raise ValueError("compatibility sealed inventory count differs")
    for binding, digest in bindings.items():
        scope, relative = binding.split("/", 1)
        safe_name(relative, allow_raw=True)
        if scope not in ("repo", "release"):
            raise ValueError("invalid sealed binding scope")
        if relative.endswith(".pb"):
            continue
        if binding == "release/iclr_paper/tex/numbers.tex":
            if digest != pins["PROTECTED_NUMBERS_SHA256"]:
                raise ValueError("historical number binding differs")
            continue
        inventory.file(("tempopb/" if scope == "repo" else "") + relative, digest,
                       origin="compatibility sealed binding " + binding, replace_historical=True)
    inventory.file("iclr_paper/tex/numbers.tex", reconstruction_pins["WORKING_NUMBERS_SHA"])
    inventory.file(PRIMARY + "/" + safe_name(compat["delta_csv"]["relative_path"]), compat["delta_csv"]["sha256"])
    raw_files = {}
    for family, count in ((PRIMARY, 132), (MULTICITY, 397)):
        lock_path = family + "/protocol_lock.json"
        lock = json.loads(inventory.files[lock_path])
        omitted = 0
        for record in lock["tracked_files"].values():
            relative, digest = safe_name(record["path"], allow_raw=True), record["sha256"]
            if relative.endswith(".pb"):
                name = "tempopb/" + relative
                if name in raw_files and raw_files[name]["sha256"] != digest:
                    raise ValueError("conflicting omitted raw-file digest")
                raw_files.setdefault(name, {"sha256": digest, "recorded_in": []})["recorded_in"].append(lock_path)
                omitted += 1
            else:
                inventory.file("tempopb/" + relative, digest, origin=lock_path, replace_historical=True)
        if omitted != count:
            raise ValueError(f"omitted raw-file count differs: {family}")
    mc = json.loads(inventory.file(MULTICITY + "/v1_v2_delta_audit.json", MC_AUDIT_SHA))
    if len(mc["fit_audit"]["fits"]) != 24:
        raise ValueError("multicity fit count differs")
    for fit in mc["fit_audit"]["fits"]:
        inventory.file(MULTICITY + "/fits/" + safe_name(fit["coordinate"]), fit["v2_sha256"])
    for name, digest in mc["authenticated_inputs"]["v2_output_sha256"].items():
        inventory.file(MULTICITY + "/" + safe_name(name), digest)
    inventory.file(MULTICITY + "/heldout_opened.json", mc["authenticated_inputs"]["v2_heldout_receipt_sha256"])
    inventory.file(MULTICITY + "/" + safe_name(mc["artifacts"]["delta_rows_csv"]), mc["artifacts"]["delta_rows_csv_sha256"])
    for directory, digest in ((PRIMARY + "/manuscript_reconstruction", RECON_MANIFEST_SHA),
                              ("tempopb/analysis-output/external-support-set", EXTERNAL_MANIFEST_SHA)):
        manifest = json.loads(inventory.file(directory + "/manifest.json", digest,
                                             replace_historical=True))
        for name, expected in manifest["output_sha256"].items():
            inventory.file(directory + "/" + safe_name(name), expected,
                           origin=directory + "/manifest.json", replace_historical=True)
    contextual = "tempopb/analysis-output/contextual-age-lookup-20260920/"
    inventory.file(contextual + "summary.json", reconstruction_pins["CONTEXTUAL_SHA"])
    inventory.file(contextual + "per_series.csv", CONTEXTUAL_CSV_SHA)
    final_source = inventory.read(FINAL_SOURCE, FINAL_SOURCE_SHA)
    for name, raw in zip_members(final_source).items():
        inventory.add("manuscript/" + name, raw, "final manuscript source ZIP")
    inventory.file(FINAL_PDF, FINAL_PDF_SHA, destination="manuscript/paper.pdf")
    for module in ("verify_iclr_saved_supplement", "build_iclr_saved_supplement", "iclr_manuscript_reconstruct_v2",
                   "iclr_manuscript_core_v2", "iclr_manuscript_tables_v2", "iclr_manuscript_aux_v2",
                   "iclr_contextual_lookup_correction", "analyze_external_support_set_results", "gen_iclr_numbers"):
        inventory.file(f"tempopb/src/{module}.py")
    for test in ("iclr_manuscript_reconstruct_v2", "iclr_manuscript_core_v2", "iclr_manuscript_tables_v2",
                 "iclr_saved_supplement", "iclr_saved_verification"):
        inventory.file(f"tempopb/tests/test_{test}.py")
    import_closure(inventory)
    inventory.file("iclr_paper/strong_accept/submission_20260925_r3/SUPPLEMENT_README.md", destination="README.md")
    inventory.add("EXCLUDED_RAW_DATA.json", json_bytes({"schema": "omitted-public-ballot-files-v1", "files": raw_files,
                  "disclosure": "Raw .pb ballot files are omitted. Their recorded paths and hashes are preserved; omitted file bytes were not rechecked in data-free mode. The included public anomaly receipt contains one pseudonymous ballot record."}), "generated omission inventory")
    manifest = {"schema": "iclr-saved-results-supplement-v1",
                "files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(inventory.files.items())},
                "historical_replacements": inventory.replacements,
                "historical_numbers": {"archive": pins["PROTECTED_SOURCE_RELATIVE"], "member": "numbers.tex",
                                       "sha256": pins["PROTECTED_NUMBERS_SHA256"]},
                "synthetic_path_fixture_disclosure": {
                    "test_members": sorted(SYNTHETIC_PATH_FIXTURES),
                    "description": "Two immutable tests contain artificial home-directory paths with the labels reviewer and author. These exact fixture paths are permitted only in their corresponding test files; no record was redacted."},
                "final_source_zip_sha256": FINAL_SOURCE_SHA, "final_pdf_sha256": FINAL_PDF_SHA}
    inventory.add("MANIFEST.json", json_bytes(manifest), "generated package manifest")
    return dict(sorted(inventory.files.items()))


def zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            safe_name(name)
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.create_system, item.external_attr, item.compress_type = 3, 0o100644 << 16, zipfile.ZIP_DEFLATED
            archive.writestr(item, raw, compresslevel=9)
    return output.getvalue()


def build(workspace: Path = WORKSPACE, *, write: bool = False) -> dict:
    files = collect(workspace)
    report = {"files": len(files), "uncompressed_bytes": sum(map(len, files.values())),
              "output": OUTPUT, "written": False}
    if write:
        path = Inventory(workspace).path(OUTPUT)
        raw = zip_bytes(files)
        with path.open("xb") as handle:
            handle.write(raw)
        report.update(written=True, zip_bytes=len(raw), sha256=sha(raw))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="exclusively create the fixed new archive after review")
    print(json.dumps(build(write=parser.parse_args().write), indent=2))
