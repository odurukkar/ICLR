"""Append-only manuscript reconstruction from authenticated, already saved results.

Never invokes a trainer, evaluator, allocator, or the legacy paper generator.
The candidate is not automatically installed into the paper. Missing timing
evidence remains explicit and prevents this bundle from being a final release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

import iclr_primary_warsaw_delta_audit_v2 as audit
from iclr_manuscript_aux_v2 import format_aux_macros
from iclr_manuscript_tables_v2 import format_table_macros
from iclr_manuscript_core_v2 import format_core_macros

ROOT = Path(__file__).resolve().parent.parent
COMPAT_SHA = "f380ef80a1da1fecd8bb2b44dd4a46e09414827fc6e26524a927a4a6740ceeb8"
WORKING_NUMBERS_SHA = "331c9588361195aae77cc8fc772c58898e7e1cf38f8d70059ca21a25734a2d9f"
CONTEXTUAL_SHA = "2b0a3a7b59cbd2d6952a6dca341b657c7b5269747a3cdfc7785a87ad15a6242c"
RUNTIME_NAMES = {"EndowRuntimeLoMin", "EndowRuntimeHiMin", "OutcomeRuntimeLoMin", "OutcomeRuntimeHiMin"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes, label: str) -> dict:
    return audit._load_json_bytes(raw, label=label)


def _read(path: Path) -> bytes:
    return audit._read_regular(path, label=str(path))


def _zip_member(raw: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = audit._safe_zip_members(z, label="protected archive")
        return z.read(members[name])


def _binding_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or "/" not in name:
        raise ValueError("invalid sealed binding name")
    scope, relative = name.split("/", 1)
    audit._safe_member_name(relative, label="sealed binding")
    if scope not in ("repo", "release") or "\\" in relative:
        raise ValueError("invalid sealed binding scope or path")
    return (root if scope == "repo" else root.parent) / relative


def _raw_relative(name: str) -> str:
    """Accept only the canonical, flat raw corpus directory."""
    if not isinstance(name, str) or not name.startswith("repo/"):
        raise ValueError("invalid raw binding path")
    relative = name[len("repo/"):]
    path = PurePosixPath(relative)
    if (path.parts[:2] != ("data", "pb") or len(path.parts) != 3
            or path.as_posix() != relative or path.suffix != ".pb" or "\\" in relative
            or any(ord(character) < 32 for character in relative)):
        raise ValueError("invalid raw binding path")
    return relative


def _authenticated_raw_bindings(bindings: dict, lock: dict, manifest: dict,
                                manifest_sha256: str) -> dict[str, str]:
    """Cross-check already hash-authenticated metadata; never infer missing inputs."""
    manifest_relative = (audit.V2_RESULT_RELATIVE / "corpus_manifest.json").as_posix()
    tracked = lock.get("tracked_files")
    if (not isinstance(tracked, dict)
            or tracked.get("artifact/corpus_manifest") != {
                "path": manifest_relative, "sha256": manifest_sha256}
            or bindings.get("repo/" + manifest_relative) != manifest_sha256):
        raise ValueError("raw corpus manifest lock identity differs")
    rows = manifest.get("files")
    if (not isinstance(rows, list) or len(rows) != 132
            or any(type(manifest.get(k)) is not int or manifest[k] != 132
                   for k in ("n_files", "n_elections"))
            or manifest.get("source_dir") != "data/pb" or manifest.get("destination") != "data/pb"):
        raise ValueError("raw corpus inventory or canonical paths differ")
    expected = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("invalid raw corpus row")
        name = "repo/data/pb/" + row["name"]
        _raw_relative(name)
        digest = row.get("sha256")
        if (row.get("source_relative") != row["name"] or name in expected
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("raw corpus duplicate, path, or digest differs")
        expected[name] = digest
    sealed_raw = {name: digest for name, digest in bindings.items() if name.startswith("repo/data/")}
    locked_raw = {}
    for label, row in tracked.items():
        if label.startswith("data/"):
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise ValueError("invalid raw lock row")
            path = row["path"]
            if not isinstance(path, str):
                raise ValueError("invalid raw lock path")
            name = "repo/" + path
            _raw_relative(name)
            if label != "data/" + PurePosixPath(path).name or name in locked_raw:
                raise ValueError("raw lock identity differs")
            locked_raw[name] = row["sha256"]
    if expected != sealed_raw or expected != locked_raw:
        raise ValueError("raw corpus, sealed bindings, and primary lock differ")
    return expected


def _raw_path_missing(root: Path, name: str) -> bool:
    """Only absence is optional: reject symlinks and non-directory ancestors."""
    target = root / _raw_relative(name)
    current = Path(target.anchor)
    for index, part in enumerate(target.parts[1:], 1):
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return True
        if stat.S_ISLNK(mode):
            raise ValueError(f"raw input path is a symlink: {current}")
        is_last = index == len(target.parts) - 1
        if not (stat.S_ISREG(mode) if is_last else stat.S_ISDIR(mode)):
            raise ValueError(f"raw input path has invalid file type: {current}")
    return False


def _capture_sealed_binding(root: Path, name: str, digest: str, snapshots: dict,
                            allowed_missing_raw: dict[str, str]) -> dict | None:
    path = _binding_path(root, name)
    if name in allowed_missing_raw:
        if digest != allowed_missing_raw[name]:
            raise ValueError("raw omission digest differs from authenticated metadata")
        if _raw_path_missing(root, name):
            return {"path": name, "sha256": digest}
    content = _read(path)
    if _sha(content) != digest:
        raise ValueError(f"input digest differs: {path}")
    snapshots[path] = digest
    return None


def capture_inputs(root: Path = ROOT, *, allow_missing_raw_data: bool = False) -> dict:
    """Authenticate original snapshots and revised evidence without altering either."""
    if type(allow_missing_raw_data) is not bool:
        raise ValueError("allow_missing_raw_data must be an explicit boolean")
    root = root.resolve(); snapshots: dict[Path, str] = {}

    def capture(path: Path, expected: str | None = None) -> bytes:
        content = _read(path); actual = _sha(content)
        if expected is not None and actual != expected:
            raise ValueError(f"input digest differs: {path}")
        snapshots[path] = actual
        return content

    source_zip = capture(root.parent / audit.PROTECTED_SOURCE_RELATIVE, audit.PROTECTED_SOURCE_SHA256)
    original_bytes = _zip_member(source_zip, audit.PROTECTED_NUMBERS_MEMBER)
    if _sha(original_bytes) != audit.PROTECTED_NUMBERS_SHA256:
        raise ValueError("protected archived numbers differ")
    original = audit.parse_numbers_tex(original_bytes)
    working_path = root.parent / "iclr_paper/tex/numbers.tex"
    working = audit.parse_numbers_tex(capture(working_path, WORKING_NUMBERS_SHA))
    v2root = root / audit.V2_RESULT_RELATIVE
    compat = _json(capture(v2root / "v1_v2_delta_compat_audit.json", COMPAT_SHA), "compatibility audit")
    bindings = compat["compatibility_amendment"]["sealed_file_sha256"]
    if not isinstance(bindings, dict) or len(bindings) != 288:
        raise ValueError("sealed compatibility inventory changed")
    for name, digest in bindings.items():
        _binding_path(root, name)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid sealed binding digest")
    allowed_missing_raw = {}
    if allow_missing_raw_data:
        lock_path = v2root / "protocol_lock.json"
        lock_bytes = capture(lock_path, bindings["repo/" + lock_path.relative_to(root).as_posix()])
        if _sha(lock_bytes) != compat["authenticated_inputs"]["v2_protocol_lock_sha256"]:
            raise ValueError("sealed primary lock identity differs")
        manifest_path = v2root / "corpus_manifest.json"
        manifest_bytes = capture(manifest_path, bindings["repo/" + manifest_path.relative_to(root).as_posix()])
        allowed_missing_raw = _authenticated_raw_bindings(
            bindings, _json(lock_bytes, "primary lock"),
            _json(manifest_bytes, "primary corpus manifest"), _sha(manifest_bytes))
    omitted_raw_data = []
    for name, digest in bindings.items():
        if name == "release/iclr_paper/tex/numbers.tex":
            if digest != audit.PROTECTED_NUMBERS_SHA256:
                raise ValueError("archived manuscript digest changed")
            continue  # Authenticated from original source ZIP above, not waived.
        omitted = _capture_sealed_binding(root, name, digest, snapshots, allowed_missing_raw)
        if omitted is not None:
            omitted_raw_data.append(omitted)
    matrix, _fits = audit.authenticate_complete_v2_matrix(root)
    evaluation, rows, consumers = audit.authenticate_v2_evaluation_commit(root, matrix)
    summary, classified = audit.classify_protected_manuscript_values(
        original, consumers["manuscript_macros"], consumers["provenance"]["macro_coordinates"])
    if compat["manuscript_audit"] != {**summary, "rows": classified}:
        raise ValueError("sealed macro classification differs from current authenticated evidence")
    expected_auth = {"v2_evaluation":evaluation,"v2_fit_count":matrix["fit_count"],
                     "v2_fit_sha256":matrix["fit_sha256"],"v2_protocol_lock_sha256":matrix["protocol_lock_sha256"],
                     "v2_training_grid_sha256":matrix["grid_sha256"]}
    if compat["authenticated_inputs"] != expected_auth:
        raise ValueError("sealed fit/evaluation identities differ")
    contextual_path = root / "analysis-output/contextual-age-lookup-20260920/summary.json"
    contextual = _json(capture(contextual_path, CONTEXTUAL_SHA), "contextual contrast")
    csv_path = root / contextual["source_csv"]
    if csv_path != v2root / "evaluation/per_series.csv":
        raise ValueError("unexpected contextual source path")
    capture(csv_path, contextual["source_csv_sha256"])
    if any(matrix["fit_sha256"].get(k) != v for k,v in contextual["fit_sha256"].items()):
        raise ValueError("contextual fit provenance differs")
    split_path = v2root / "splits/temporal_2022.json"
    split = _json(capture(split_path, bindings["repo/"+str(split_path.relative_to(root))]), "v2 split")
    optimizer_path = root / "src/iclr_cmaes.py"
    optimizer = capture(optimizer_path, bindings["repo/src/iclr_cmaes.py"]).decode("utf-8")
    repro = capture(root.parent / audit.PROTECTED_REPRO_RELATIVE, audit.PROTECTED_REPRO_SHA256)
    degeneracy_bytes = _zip_member(repro, "results/iclr_degeneracy.csv")
    artifacts = {
        "split":{"path":str(split_path.relative_to(root)),"sha256":_sha(_read(split_path))},
        "optimizer":{"path":"src/iclr_cmaes.py","sha256":_sha(optimizer.encode())},
        "per_series":{"path":str(audit.V2_RESULT_RELATIVE / "evaluation/per_series.json"),"sha256":evaluation["per_series_sha256"]},
        "contextual":{"path":str(contextual_path.relative_to(root)),"sha256":CONTEXTUAL_SHA},
        "archived_numbers":{"archive":str(audit.PROTECTED_SOURCE_RELATIVE),"archive_sha256":audit.PROTECTED_SOURCE_SHA256,
                            "member":audit.PROTECTED_NUMBERS_MEMBER,"sha256":audit.PROTECTED_NUMBERS_SHA256},
        "archived_degeneracy":{"archive":str(audit.PROTECTED_REPRO_RELATIVE),"archive_sha256":audit.PROTECTED_REPRO_SHA256,
                               "member":"results/iclr_degeneracy.csv","sha256":_sha(degeneracy_bytes)},
    }
    return {"root":root,"snapshots":snapshots,"original":original,"working":working,
            "classification":classified,"consumers":consumers,"per_series":rows,
            "split":split,"optimizer":optimizer,"contextual":contextual,
            "degeneracy":list(csv.DictReader(io.StringIO(degeneracy_bytes.decode()))),
            "artifacts":artifacts,"evaluation":evaluation,"matrix":matrix,
            "allow_missing_raw_data":allow_missing_raw_data,
            "omitted_raw_data":sorted(omitted_raw_data, key=lambda row: row["path"])}


def validate_sources(records: dict, captured: dict) -> None:
    """Reject dangling provenance; attach authenticated enclosing file hashes."""
    consumers = captured["consumers"]["consumers"]
    for name,r in records.items():
        if not re.fullmatch("[A-Za-z]+",name) or not r.get("transformation") or not r.get("sources"):
            raise ValueError(f"invalid macro/provenance record: {name}")
        if r["value"] is not None and (not isinstance(r["value"],str) or any(t in r["value"].lower() for t in ("nan","infinity"))):
            raise ValueError(f"invalid macro value: {name}")
        for s in r["sources"]:
            if not isinstance(s.get("coordinates"),dict) or not s.get("fields"):
                raise ValueError(f"missing coordinates/fields: {name}")
            if "consumer" in s:
                table=consumers[s["consumer"]]["values"][s["table"]]
                rows=table if isinstance(table,list) else [table]
                chosen=[row for row in rows if isinstance(row,dict) and all(row.get(k)==v for k,v in s["coordinates"].items())]
                if not chosen or any(field not in row for row in chosen for field in s["fields"]):
                    raise ValueError(f"dangling consumer provenance: {name}: {s}")
                s["artifact_sha256"]=captured["evaluation"]["paper_consumers_sha256"]
            else:
                artifact=s["artifact"]
                if artifact not in captured["artifacts"]:
                    raise ValueError(f"unknown artifact provenance: {name}")
                s["artifact_identity"]=captured["artifacts"][artifact]


def _legacy_degeneracy(rows: list[dict]) -> dict[str,str]:
    by_n={int(r["voters"]):r for r in rows}
    if len(by_n)!=len(rows) or set(by_n)!={80,200,400,800}:
        raise ValueError("archived synthetic inventory differs")
    largest=by_n[800]
    result={"DegGroups":largest["groups"],"DegSupporters":largest["supporters"],"DegVoters":largest["voters"],
            "DegCSD":f"{float(largest['worst_abs_csd']):.4f}","DegExcl":f"{100*float(largest['exclusion']):.1f}\\%"}
    for n,tag in [(80,"Eighty"),(200,"TwoHundred"),(400,"FourHundred"),(800,"EightHundred")]:
        r=by_n[n]
        result.update({"Deg"+tag+"Voters":str(n),"Deg"+tag+"CSD":f"{float(r['worst_abs_csd']):.4f}",
                       "Deg"+tag+"Excl":f"{100*float(r['exclusion']):.1f}\\%"})
    return result


def reconstruct(captured: dict) -> dict:
    consumers=captured["consumers"]["consumers"]; records={}
    groups=[format_core_macros(consumers),format_table_macros(consumers),
            format_aux_macros(consumers,captured["per_series"],captured["split"],captured["optimizer"],captured["contextual"])]
    for group in groups:
        if set(records)&set(group):
            raise ValueError(f"overlapping macro ownership: {set(records)&set(group)}")
        records.update(group)
    for r in records.values():
        r["classification"]="reconstructed_primary_v2"
    for name,r in list(records.items()):
        if name.endswith(("Diff","Delta")):
            try: number=float(r["value"])
            except ValueError: continue
            if not math.isfinite(number): raise ValueError("nonfinite formatted delta")
            twin=name+"Abs"
            if twin in records: raise ValueError("duplicate absolute-delta macro")
            records[twin]={**r,"value":f"{abs(number):.4f}","sources":[dict(s) for s in r["sources"]],
                           "transformation":f"absolute value of already-rounded {name}; four decimals", "derived_from_macro":name}
    for row in captured["classification"]:
        name=row["name"]
        if row["classification"] == "unaffected_non_primary" and name not in records:
            records[name]={"value":row["v1_value"],"classification":"unaffected_non_primary",
                           "sources":[{"artifact":"archived_numbers","coordinates":{"name":name},"fields":[name]}],
                           "transformation":"carry forward authenticated original external/non-primary value; not reconstructed primary-v2 evidence"}
    for name,value in _legacy_degeneracy(captured["degeneracy"]).items():
        if value!=captured["original"][name]: raise ValueError("archived synthetic value differs")
        records[name]={"value":value,"classification":"archived_synthetic_carry_forward",
                       "sources":[{"artifact":"archived_degeneracy","coordinates":{},"fields":["voters","groups","supporters","worst_abs_csd","exclusion"]}],
                       "transformation":"format separately authenticated archived synthetic illustration; not primary-v2 replay"}
    for name in sorted(RUNTIME_NAMES):
        records[name]={"value":None,"classification":"unavailable_v2_runtime",
                       "sources":[{"artifact":"archived_numbers","coordinates":{"name":name},"fields":[name]}],
                       "transformation":"old runtime is not a corrected-v2 elapsed measurement; candidate definition intentionally omitted"}
    expected=set(captured["working"])
    if set(records)!=expected:
        raise ValueError(f"macro coverage mismatch: missing={sorted(expected-set(records))}; extra={sorted(set(records)-expected)}")
    for name,value in captured["consumers"]["manuscript_macros"].items():
        if records[name]["value"] != value:
            raise ValueError(f"direct-map disagreement: {name}: {records[name]['value']} != {value}")
    for name,r in records.items():
        r["previous_value"]=captured["working"][name]
        r["changed"]=r["value"] != r["previous_value"]
    validate_sources(records,captured)
    counts={status:sum(r["classification"]==status for r in records.values()) for status in sorted({r["classification"] for r in records.values()})}
    return {"schema_version":1,"classification":"corrected post-hoc replay; manuscript candidate, not release clearance",
            "counts":counts,"macro_count":len(records),"direct_map_exact":len(captured["consumers"]["manuscript_macros"]),
            "input_identity":{"matrix":captured["matrix"],"evaluation":captured["evaluation"],"compatibility_audit_sha256":COMPAT_SHA,
                              "working_numbers_sha256":WORKING_NUMBERS_SHA,"artifacts":captured["artifacts"]},
            "records":dict(sorted(records.items()))}


def recheck_inputs(captured: dict) -> None:
    for path,digest in captured["snapshots"].items():
        if _sha(_read(path))!=digest: raise ValueError(f"input changed during reconstruction: {path}")
    for row in captured.get("omitted_raw_data", []):
        if not _raw_path_missing(captured["root"], row["path"]):
            raise ValueError(f"omitted raw input appeared during reconstruction: {row['path']}")


def render_outputs(payload: dict) -> dict[str,bytes]:
    lines=["% CANDIDATE: authenticated primary-v2 reconstruction; NOT installed in the manuscript.",
           "% Four runtime definitions are deliberately omitted; resolve their manuscript uses before compiling."]
    for name,r in payload["records"].items():
        if r["value"] is None: lines.append(f"% UNAVAILABLE corrected-v2 runtime: {name}")
        else: lines.append(f"\\newcommand{{\\{name}}}{{{r['value']}}}")
    report=["# Corrected manuscript reconstruction candidate","",f"Total classified macros: {payload['macro_count']}; exact direct-map checks: {payload['direct_map_exact']}.",
            "","## Classification", ""]+[f"- {k}: {v}" for k,v in payload["counts"].items()]
    report += ["","## Changes versus working manuscript","", "| Macro | Current | Candidate |","|---|---|---|"]
    report += [f"| {n} | {r['previous_value']} | {r['value'] if r['value'] is not None else 'UNAVAILABLE'} |" for n,r in payload["records"].items() if r["changed"]]
    report += ["","No fits, allocations, bootstrap resampling, or exact hypothesis tests were rerun. This candidate does not certify tables, figures, non-primary correction lineages, or supplementary archives. Four runtime claims need removal or explicitly historical treatment before promotion."]
    return {"macro_records.json":(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode(),
            "numbers_candidate.tex":("\n".join(lines)+"\n").encode(),"coverage.md":("\n".join(report)+"\n").encode()}


def write_outputs(captured: dict, payload: dict) -> Path:
    """New fixed output directory only; no override flag and no manuscript writes."""
    if captured.get("allow_missing_raw_data") or captured.get("omitted_raw_data"):
        raise ValueError("saved-results-only capture is read-only")
    destination=captured["root"]/audit.V2_RESULT_RELATIVE/"manuscript_reconstruction"
    output=render_outputs(payload)
    recheck_inputs(captured)
    destination.mkdir(exist_ok=False)
    for name,raw in output.items():
        with (destination/name).open("xb") as stream: stream.write(raw)
    recheck_inputs(captured)
    manifest={"output_sha256":{name:_sha(raw) for name,raw in output.items()},"status":"candidate; unresolved manuscript promotion"}
    with (destination/"manifest.json").open("x") as stream:
        json.dump(manifest,stream,indent=2);stream.write("\n")
    return destination


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--write",action="store_true",help="create a new candidate directory after checks; never overwrite")
    mode.add_argument("--saved-results-only",action="store_true",
                      help="read-only verification; allow only authenticated missing raw ballot files")
    args=parser.parse_args();captured=capture_inputs(allow_missing_raw_data=args.saved_results_only)
    payload=reconstruct(captured);recheck_inputs(captured)
    print(json.dumps({"macro_count":payload["macro_count"],"counts":payload["counts"],
                      "direct_map_exact":payload["direct_map_exact"],
                      "omitted_raw_data":captured["omitted_raw_data"]}))
    if args.write: print(write_outputs(captured,payload))


if __name__ == "__main__":
    main()
