"""Corrected post-hoc three-path MES parity, with append-only gate reports.

The trace is a canonicalized outcome-equivalent MES witness, not a claim of
literal branch-iteration equivalence. Empirical parity cannot prove floating
runtime certificates or authorize cache reuse.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import resource
import secrets
import sys
import time
from typing import Callable, Generator, Mapping, Sequence

import iclr_approval_semantics_v2 as semantics_v2
import iclr_multicity_evaluate_v2 as evaluate_v2
import iclr_multicity_protocol_v2 as protocol_v2
from iclr_trace_parity import compare_three_paths
from cohorts import group_outcome
from iclr_env import RolloutState
from iclr_policy import linear_policy
from parse_pb import PBInstance, Project, Vote


ROOT = Path(__file__).resolve().parent.parent
RESULT_RELATIVE = Path("results/iclr_multicity_v2")
PARITY_RESULT_RELATIVE = Path("results/iclr_trace_parity_v2")
PRIMARY_DELTA_RELATIVE = Path("results/iclr_primary_warsaw_v2/v1_v2_delta_compat_audit.json")
PRIMARY_DELTA_SHA256 = "f380ef80a1da1fecd8bb2b44dd4a46e09414827fc6e26524a927a4a6740ceeb8"
MULTICITY_DELTA_SHA256 = "44ff3d85536438f22712004aaef4a9f5db6fd10de32aa520cf4c668cc99e7b31"
STAGE_COUNTS = {
    "warsaw-sentinels": (10, 6, 4),
    "representative-temporal-scored": (6, 6, 0),
    "temporal-scored": (446, 446, 0),
    "native": (794, 446, 348),
    "full": (3176, 1784, 1392),
}
LONG_STAGES = ("temporal-scored", "native", "full")
SENTINELS = (
    "Poland_Warszawa_2020_Bielany.pb", "Poland_Warszawa_2020_Targowek.pb",
    "Poland_Warszawa_2026_Bemowo.pb", "Poland_Warszawa_2026_Bielany.pb",
    "Poland_Warszawa_2026_Targowek.pb",
)
REPRESENTATIVES = (
    "Poland_Gdynia_2023_Kamienna_Gora__large.pb", "Poland_Lodz_2024_Rokicie.pb",
    "Poland_Warszawa_2024_Mokotow.pb",
)
FIT_COUNTS = {
    "city_out_Poland_Gdynia/endowment/seed-42": (124, 61, 63),
    "city_out_Poland_Warszawa/endowment/seed-42": (129, 54, 75),
    "city_out_Poland_Łódź/endowment/seed-42": (144, 108, 36),
    "temporal_2022/endowment/seed-1": (397, 223, 174),
    "temporal_2022/endowment/seed-2": (397, 223, 174),
    "temporal_2022/endowment/seed-42": (397, 223, 174),
}
SOURCE_MODULES = (
    "cohorts", "iclr_corpus", "iclr_env", "iclr_outcome", "iclr_policy",
    "iclr_priority_mes", "iclr_residual_actuation", "iclr_residual_policy",
    "parse_pb", "rules", "run_experiments", "iclr_trace_parity",
    "iclr_approval_semantics_v2", "iclr_multicity_evaluate_v2",
    "iclr_multicity_protocol_v2", "iclr_trace_parity_v2",
)


def _read(path: Path) -> bytes:
    content = protocol_v2.read_regular_bytes_artifact_v2(path, label="parity sealed input")
    if content is None:
        raise RuntimeError(f"parity sealed input is missing: {path}")
    return content


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(content: bytes) -> dict[str, object]:
    def reject(value):
        raise RuntimeError(f"non-finite JSON constant: {value}")
    value = json.loads(content, object_pairs_hook=_unique_object, parse_constant=reject)
    if type(value) is not dict:
        raise RuntimeError("parity input must be a JSON object")
    _json_bytes(value)  # Also rejects exponent-overflow floats.
    return value


_IMPORTED_SOURCE_SHA256 = tuple(
    (name, _sha(_read(ROOT / "src" / f"{name}.py"))) for name in SOURCE_MODULES
)
_RUN_AUTHORITY = secrets.token_bytes(32)


def _authentication_tag(kind: str, fingerprint: str) -> str:
    return hmac.new(_RUN_AUTHORITY, f"{kind}/{fingerprint}".encode(), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Election:
    series: str
    year: int
    raw_path: Path
    raw_sha256: str
    semantic_sha256: str
    phase: str
    instance_path: str
    meta: tuple[tuple[str, str], ...]
    projects: tuple[tuple[str, Project], ...]
    votes: tuple[Vote, ...]

    def materialize(self) -> PBInstance:
        return PBInstance(path=self.instance_path, meta=dict(self.meta), projects=dict(self.projects), votes=self.votes)


@dataclass(frozen=True)
class EndowmentFit:
    context: str
    split: str
    seed: int
    sha256: str
    weights: tuple[float, ...]


@dataclass(frozen=True)
class AuthenticatedContext:
    repo_root: Path
    elections: tuple[Election, ...]
    fits: tuple[EndowmentFit, ...]
    deployments: tuple[tuple[str, tuple[tuple[str, tuple[int, ...]], ...]], ...]
    sealed_files: tuple[tuple[str, str], ...]
    provenance_json: bytes
    context_sha256: str
    factory_tag: str = ""
    execution_report_sha256: str = ""
    execution_tag: str = ""


def freeze_election(series: str, year: int, path: Path, raw_sha256: str, instance: PBInstance, phase: str) -> Election:
    protocol_v2._require_sha256(raw_sha256, "raw election")
    if phase not in {"scored", "warmup"} or type(year) is not int:
        raise RuntimeError("parity election phase/year differs")
    if not instance.path.endswith("#" + semantics_v2.SEMANTICS_PROFILE):
        raise RuntimeError("parity election is missing corrected semantics provenance")
    if any(len(vote.projects) != len(set(vote.projects)) for vote in instance.votes):
        raise RuntimeError("corrected parity election retained duplicate approvals")
    return Election(
        series, year, Path(path), raw_sha256,
        semantics_v2.canonical_instance_sha256(instance), phase, instance.path,
        tuple(sorted(instance.meta.items())), tuple(instance.projects.items()), tuple(instance.votes),
    )


def validate_opening_receipt(receipt: Mapping[str, object], *, lock_sha256: str, fit_sha256: Mapping[str, str], semantics_sha256: str, corpus_semantic_sha256: str, structural_sha256: str) -> None:
    expected = {
        "schema_version": 2, "lock_sha256": lock_sha256, "fit_sha256": dict(fit_sha256),
        "semantics_profile": semantics_v2.SEMANTICS_PROFILE,
        "semantics_receipt_sha256": semantics_sha256,
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "structural_gates_sha256": structural_sha256,
    }
    if set(receipt) != set(expected) | {"opened_at_utc"} or len(fit_sha256) != 24:
        raise RuntimeError("corrected opening receipt schema/inventory differs")
    if any(not protocol_v2.exact_json_equal_v2(receipt.get(key), value) for key, value in expected.items()):
        raise RuntimeError("corrected opening receipt lineage binding differs")
    try:
        timestamp = datetime.fromisoformat(receipt["opened_at_utc"])
    except (ValueError, TypeError) as exc:
        raise RuntimeError("corrected opening receipt timestamp differs") from exc
    if timestamp.utcoffset() != timedelta(0):
        raise RuntimeError("corrected opening receipt timestamp must be UTC")


def _binding_path(root: Path, label: str) -> Path:
    scope, relative = label.split("/", 1)
    if scope not in {"repo", "release"} or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("parity sealed path scope differs")
    return (root if scope == "repo" else root.parent) / relative


def _assert_files(root: Path, bindings: tuple[tuple[str, str], ...]) -> None:
    if len(dict(bindings)) != len(bindings):
        raise RuntimeError("parity sealed file bindings contain duplicates")
    for name, digest in bindings:
        protocol_v2._require_sha256(digest, "parity sealed input")
        if _sha(_read(_binding_path(root, name))) != digest:
            raise RuntimeError(f"parity sealed input changed: {name}")


def _fingerprint(context: AuthenticatedContext) -> str:
    payload = {
        "provenance": _json(context.provenance_json), "sealed_files": context.sealed_files,
        "deployments": context.deployments,
        "fits": [(fit.context, fit.split, fit.seed, fit.sha256, [weight.hex() for weight in fit.weights]) for fit in context.fits],
        "elections": [(row.series, row.year, str(row.raw_path), row.instance_path, row.raw_sha256, row.semantic_sha256, row.phase) for row in context.elections],
    }
    return _sha(_json_bytes(payload))


def assert_context_unchanged(context: AuthenticatedContext) -> None:
    if _fingerprint(context) != context.context_sha256:
        raise RuntimeError("parity context fingerprint changed")
    _assert_files(context.repo_root, context.sealed_files)
    for row in context.elections:
        if semantics_v2.canonical_instance_sha256(row.materialize()) != row.semantic_sha256:
            raise RuntimeError(f"parity materialized semantics changed: {row.series}/{row.year}")


def _seal_lineages(root: Path, bindings: dict[str, str]) -> None:
    primary_content = _read(root / PRIMARY_DELTA_RELATIVE)
    if _sha(primary_content) != PRIMARY_DELTA_SHA256:
        raise RuntimeError("sealed primary compatibility audit digest differs")
    primary = _json(primary_content)
    if primary.get("status") != "pass":
        raise RuntimeError("primary compatibility audit is not sealed pass")
    bindings.update(primary["compatibility_amendment"]["sealed_file_sha256"])
    bindings["repo/" + PRIMARY_DELTA_RELATIVE.as_posix()] = PRIMARY_DELTA_SHA256
    bindings["repo/" + (PRIMARY_DELTA_RELATIVE.parent / primary["delta_csv"]["relative_path"]).as_posix()] = primary["delta_csv"]["sha256"]
    multicity_path = RESULT_RELATIVE / "v1_v2_delta_audit.json"
    content = _read(root / multicity_path)
    if _sha(content) != MULTICITY_DELTA_SHA256:
        raise RuntimeError("sealed multicity delta audit digest differs")
    multicity = _json(content)
    if multicity.get("status") != "pass":
        raise RuntimeError("multicity delta audit is not sealed pass")
    bindings["repo/" + multicity_path.as_posix()] = MULTICITY_DELTA_SHA256
    for name, digest in multicity["authenticated_inputs"]["v2_output_sha256"].items():
        bindings["repo/" + (RESULT_RELATIVE / name).as_posix()] = digest
    bindings["repo/" + (RESULT_RELATIVE / multicity["artifacts"]["delta_rows_csv"]).as_posix()] = multicity["artifacts"]["delta_rows_csv_sha256"]


def load_authenticated_context(repo_root: Path = ROOT) -> AuthenticatedContext:
    """Read-only v2 factory; never fit, open held-out data, or replay evaluation."""
    root = Path(repo_root).resolve()
    result, data = root / RESULT_RELATIVE, root / "data/pb_multicity"
    lock = protocol_v2.verify_multicity_protocol_lock_snapshot_v2(result / "protocol_lock.json", root, result, data)
    evaluate_v2.verify_lock_fit_inventory_v2(lock.payload)
    semantics_bytes = _read(result / "approval_semantics_receipt.json")
    semantics = _json(semantics_bytes)
    semantics_v2.validate_multicity_semantics_receipt(semantics)
    corpus_sha = semantics["corpus_semantic_sha256"]
    fits = evaluate_v2.verify_fit_inventory_v2(
        evaluate_v2.required_fit_inventory_v2(result), lock.payload, lock.sha256,
        corpus_semantic_sha256=corpus_sha, result_root=result, repo_root=root,
    )
    all_fits = {fit.label: fit.sha256 for fit in fits}
    structural_bytes = _read(result / "structural_gates.json")
    protocol_v2.validate_structural_gates_v2(result / "structural_gates.json", semantics_receipt_sha256=_sha(semantics_bytes), corpus_semantic_sha256=corpus_sha, repo_root=root)
    opening_bytes = _read(result / "heldout_opened.json")
    validate_opening_receipt(_json(opening_bytes), lock_sha256=lock.sha256, fit_sha256=all_fits, semantics_sha256=_sha(semantics_bytes), corpus_semantic_sha256=corpus_sha, structural_sha256=_sha(structural_bytes))
    inputs = protocol_v2.load_protocol_inputs_snapshot_v2(lock.payload, result, data, protocol_v2.SPLIT_NAMES)
    semantics_v2.validate_manifest_receipt_identities_v2(inputs.index, semantics)
    instances = evaluate_v2.load_evaluation_instances_v2(inputs.index, semantics)
    file_rows = {(row["series"], row["year"], row["name"]): row for row in semantics["files"]}
    temporal = {(series, year) for series, years in inputs.splits["temporal_2022"].test for year in years}
    elections = []
    bindings = {}

    def bind(path: Path, digest: str) -> None:
        name = "repo/" + path.relative_to(root).as_posix()
        if name in bindings and bindings[name] != digest:
            raise RuntimeError(f"conflicting parity input digest: {name}")
        bindings[name] = digest

    for series, ref in sorted(inputs.index.items()):
        for year, path in zip(ref.years, ref.paths):
            row = file_rows[(series, year, path.name)]
            election = freeze_election(series, year, path, row["raw_file_sha256"], instances[series][year], "scored" if (series, year) in temporal else "warmup")
            if election.semantic_sha256 != row["v2_semantic_sha256"]:
                raise RuntimeError("corrected parity semantic receipt binding differs")
            elections.append(election)
            bind(path, election.raw_sha256)
    bind(lock.path, lock.sha256)
    for record in lock.payload["tracked_files"].values():
        bind(root / record["path"], record["sha256"])
    for path, content in ((result / "approval_semantics_receipt.json", semantics_bytes), (result / "heldout_opened.json", opening_bytes), (result / "structural_gates.json", structural_bytes)):
        bind(path, _sha(content))
    selected = []
    for fit in fits:
        bind(fit.path, fit.sha256)
        if fit.arm == "endowment":
            weights = tuple(float(value) for value in fit.payload["result"]["best_weights"])
            if len(weights) != 6 or not all(math.isfinite(value) for value in weights):
                raise RuntimeError("corrected endowment fit weights differ")
            selected.append(EndowmentFit(f"{fit.split}/endowment/seed-{fit.seed}", fit.split, fit.seed, fit.sha256, weights))
    for name, digest in _IMPORTED_SOURCE_SHA256:
        module = sys.modules[__name__ if name == "iclr_trace_parity_v2" else name]
        if Path(module.__file__).resolve() != (root / "src" / f"{name}.py").resolve():
            raise RuntimeError(f"parity imported source origin differs: {name}")
        bind(root / "src" / f"{name}.py", digest)
    test_path = root / "tests/test_iclr_trace_parity_v2.py"
    bind(test_path, _sha(_read(test_path)))
    _seal_lineages(root, bindings)
    provenance = {
        "classification": "corrected post-hoc replay", "heldout_outcomes_already_known": True,
        "fresh_holdout": False, "preregistered": False,
        "semantics_profile": semantics_v2.SEMANTICS_PROFILE,
        "protocol_lock_sha256": lock.sha256, "heldout_receipt_sha256": _sha(opening_bytes),
        "semantics_receipt_sha256": _sha(semantics_bytes), "corpus_semantic_sha256": corpus_sha,
        "structural_gates_sha256": _sha(structural_bytes), "manifest_sha256": inputs.manifest.sha256,
        "split_sha256": {name: artifact.sha256 for name, artifact in inputs.split_artifacts.items()},
        "fit_inventory_count": len(fits), "all_fit_sha256": all_fits,
        "source_sha256": dict(_IMPORTED_SOURCE_SHA256),
        "primary_delta_sha256": PRIMARY_DELTA_SHA256, "multicity_delta_sha256": MULTICITY_DELTA_SHA256,
        "python_version": sys.version.split()[0],
    }
    context = AuthenticatedContext(root, tuple(elections), tuple(sorted(selected, key=lambda fit: fit.context)), tuple((name, split.test) for name, split in sorted(inputs.splits.items())), tuple(sorted(bindings.items())), _json_bytes(provenance), "")
    context = replace(context, context_sha256=_fingerprint(context))
    if len(elections) != 397 or len(inputs.index) != 75 or set(fit.context for fit in selected) != set(FIT_COUNTS):
        raise RuntimeError("corrected parity corpus/fit identity coverage differs")
    for stage, expected in STAGE_COUNTS.items():
        if stage_counts(context, stage) != expected:
            raise RuntimeError(f"corrected parity stage counts differ: {stage}")
    assert_context_unchanged(context)
    return replace(context, factory_tag=_authentication_tag("factory", context.context_sha256))


def selected_long_stages(value: str) -> tuple[str, ...]:
    if value == "":
        return ()
    if value == "all":
        return LONG_STAGES
    if value not in LONG_STAGES:
        raise ValueError(f"unknown corrected parity stage: {value}")
    return (value,)


def select_elections(context: AuthenticatedContext, stage: str) -> tuple[Election, ...]:
    if stage not in STAGE_COUNTS or stage == "full":
        raise ValueError(f"unknown uniform parity stage: {stage}")
    if stage in {"warsaw-sentinels", "representative-temporal-scored"}:
        names = SENTINELS if stage == "warsaw-sentinels" else REPRESENTATIVES
        by_name = {row.raw_path.name: row for row in context.elections}
        if len(by_name) != len(context.elections) or any(name not in by_name for name in names):
            raise RuntimeError("corrected parity staged election identity differs")
        return tuple(by_name[name] for name in names)
    return tuple(row for row in context.elections if stage == "native" or row.phase == "scored")


def stage_counts(context: AuthenticatedContext, stage: str) -> tuple[int, int, int]:
    if stage != "full":
        rows = select_elections(context, stage)
        return (2 * len(rows), 2 * sum(row.phase == "scored" for row in rows), 2 * sum(row.phase == "warmup" for row in rows))
    deployments = dict(context.deployments)
    count = scored = warmup = 0
    for fit in context.fits:
        observed = [0, 0, 0]
        for series, years in deployments[fit.split]:
            elections = [row for row in context.elections if row.series == series]
            if not set(years) <= {row.year for row in elections}:
                raise RuntimeError("corrected fitted deployment year differs")
            observed[0] += len(elections)
            observed[1] += len(years)
            observed[2] += len(elections) - len(years)
        if tuple(observed) != FIT_COUNTS.get(fit.context):
            raise RuntimeError(f"corrected fitted deployment count differs: {fit.context}")
        count += observed[0]
        scored += observed[1]
        warmup += observed[2]
    return (count * 2, scored * 2, warmup * 2)


@dataclass(frozen=True)
class Case:
    election: Election
    instance: PBInstance
    endowments: tuple[float, ...]
    completion: bool
    context: str
    phase: str


class RolloutFailure(RuntimeError):
    """Carry the actual policy-preparation or state-update election identity."""

    def __init__(self, election: Election, completion: bool, context: str, phase: str, operation: str, error: Exception):
        super().__init__(str(error))
        self.election = election
        self.completion = completion
        self.context = context
        self.phase = phase
        self.operation = operation
        self.error = error


def _uniform_cases(elections: tuple[Election, ...], stage: str) -> Generator[Case, tuple[str, ...], None]:
    # Every invocation begins at case 1, with completion disabled.
    for completion in (False, True):
        for election in elections:
            instance = election.materialize()
            n = len(instance.votes)
            endowments = (instance.budget / n,) * n if n else ()
            yield Case(election, instance, endowments, completion, f"uniform/{stage}", election.phase)


def _fitted_cases(context: AuthenticatedContext) -> Generator[Case, tuple[str, ...], None]:
    deployments = dict(context.deployments)
    by_series = {}
    for row in context.elections:
        by_series.setdefault(row.series, []).append(row)
    for completion in (False, True):
        for fit in context.fits:
            policy = linear_policy(fit.weights, scheme="age_sex")
            for series, scored_years in deployments[fit.split]:
                state = RolloutState()
                for election in sorted(by_series[series], key=lambda row: row.year):
                    phase = "scored" if election.year in scored_years else "warmup"
                    try:
                        instance = election.materialize()
                        endowments = tuple(float(value) for value in policy(instance, state))
                    except Exception as error:
                        raise RolloutFailure(election, completion, fit.context, phase, "policy_preparation", error) from error
                    agreed_winners = yield Case(election, instance, endowments, completion, fit.context, phase)
                    if agreed_winners is None:
                        raise RuntimeError("fitted rollout requires an agreed three-path outcome")
                    try:
                        state.update(group_outcome(instance, set(agreed_winners), "age_sex"))
                    except Exception as error:
                        raise RolloutFailure(election, completion, fit.context, phase, "state_update", error) from error


def _safe_error(error: Exception) -> dict[str, object]:
    message = str(error)
    return {"exception_type": type(error).__name__, "message": message[:1000], "message_truncated": len(message) > 1000}


def _winner_detail(winners: tuple[str, ...]) -> dict[str, object]:
    return {"count": len(winners), "sha256": _sha(_json_bytes(list(winners))), "first_32": list(winners[:32]), "truncated": len(winners) > 32}


def _case_identity(case: Case | RolloutFailure, index: int) -> dict[str, object]:
    return {
        "case_index": index, "context": case.context, "series": case.election.series,
        "year": case.election.year, "raw_filename": case.election.raw_path.name,
        "raw_sha256": case.election.raw_sha256, "semantic_sha256": case.election.semantic_sha256,
        "completion": case.completion, "phase": case.phase,
    }


def _run_cases(cases: Generator[Case, tuple[str, ...], None], *, stage: str, expected_counts: tuple[int, int, int], progress: Callable[[dict[str, object]], None]) -> tuple[dict[str, object], dict[str, object]]:
    """Run the kernels until completion or first failure; never issue a gate."""
    started = time.perf_counter()
    attempted = compared = exact = scored = warmup = 0
    modes: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    failure = None
    status = "pass"
    case = None
    agreed = None
    while True:
        try:
            case = next(cases) if attempted == 0 else cases.send(agreed)
        except StopIteration:
            break
        except RolloutFailure as error:
            if error.operation == "policy_preparation":
                attempted += 1
            failure = {"kind": error.operation + "_error", "three_path_exact": None, **_safe_error(error.error), **_case_identity(error, attempted)}
            status = "error"
            break
        except Exception as error:
            failure = {"kind": "rollout_error", "three_path_exact": None, "next_case_index": attempted + 1, "election_identity": "unavailable", **_safe_error(error)}
            status = "error"
            break
        attempted += 1
        identity = _case_identity(case, attempted)
        progress({"event": "case-start", "stage": stage, "expected_cases": expected_counts[0], **identity})
        try:
            observed = compare_three_paths(case.instance, case.endowments, completion=case.completion, instance_sha256=case.election.raw_sha256)
        except Exception as error:
            failure = {"kind": "kernel_error", "three_path_exact": None, **identity, **_safe_error(error)}
            status = "error"
            break
        compared += 1
        scored += case.phase == "scored"
        warmup += case.phase == "warmup"
        modes[str(case.completion).lower()] += 1
        contexts[case.context] += 1
        if not observed.exact:
            failure = {
                "kind": "winner_mismatch", "three_path_exact": False, **identity,
                "reference": _winner_detail(observed.reference),
                "canonicalized_witness": _winner_detail(observed.trace),
                "production": _winner_detail(observed.production),
            }
            status = "mismatch"
            break
        exact += 1
        agreed = observed.reference
        elapsed = time.perf_counter() - started
        eta = elapsed / compared * max(0, expected_counts[0] - compared)
        progress({"event": "case-complete", "stage": stage, **identity, "cases_compared": compared, "cases_exact": exact, "expected_cases": expected_counts[0], "elapsed_seconds": elapsed, "eta_seconds": eta})
    if failure is None and (compared, scored, warmup) != expected_counts:
        status = "error"
        failure = {"kind": "count_mismatch", "three_path_exact": None, "expected": list(expected_counts), "observed": [compared, scored, warmup]}
    expected_modes = {"false": expected_counts[0] // 2, "true": expected_counts[0] // 2}
    if failure is None and dict(modes) != expected_modes:
        status = "error"
        failure = {"kind": "completion_count_mismatch", "three_path_exact": None, "expected": expected_modes, "observed": dict(modes)}
    stats = {
        "status": status, "cases_attempted": attempted, "cases_compared": compared,
        "cases_exact": exact, "scored_cases": scored, "warmup_cases": warmup,
        "completion_counts": dict(sorted(modes.items())), "context_counts": dict(sorted(contexts.items())),
        "expected_counts": list(expected_counts), "failure": failure,
    }
    if failure is not None:
        progress({"event": "stage-failed", "stage": stage, **stats})
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    timing = {"elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}
    return stats, timing


def _report_claims(stage: str) -> dict[str, object]:
    return {
        "schema_version": 2, "stage": stage, "classification": "corrected post-hoc replay",
        "semantics_profile": semantics_v2.SEMANTICS_PROFILE,
        "witness_interpretation": "canonicalized outcome-equivalent MES witness",
        "literal_branch_equivalence_claimed": False,
        "float_runtime_proved": False, "cache_safe": False,
        "fresh_holdout": False, "preregistered": False,
        "heldout_outcomes_already_known": True,
    }


def failure_report(stage: str, error: Exception) -> dict[str, object]:
    return {
        **_report_claims(stage), "status": "error", "gate_eligible": False,
        "gate_passed": False, "three_path_exact": None,
        "cases_attempted": 0, "cases_compared": 0, "cases_exact": 0,
        "failure": {"kind": "authentication_or_execution_error", "three_path_exact": None, **_safe_error(error)},
    }


def run_authenticated_stage(stage: str, repo_root: Path = ROOT, progress: Callable[[dict[str, object]], None] | None = None) -> tuple[dict[str, object], dict[str, object], AuthenticatedContext | None]:
    """Authenticate afresh, run one complete stage, and recheck its sealed inputs."""
    if stage not in STAGE_COUNTS:
        raise ValueError(f"unknown corrected parity stage: {stage}")
    emit = progress if progress is not None else lambda row: None
    started = time.perf_counter()
    emit({"event": "authentication-start", "stage": stage})
    try:
        context = load_authenticated_context(repo_root)
    except Exception as error:
        return failure_report(stage, error), {"elapsed_seconds": time.perf_counter() - started}, None
    emit({"event": "authentication-complete", "stage": stage, "context_sha256": context.context_sha256, "expected_cases": STAGE_COUNTS[stage][0]})
    cases = _fitted_cases(context) if stage == "full" else _uniform_cases(select_elections(context, stage), stage)
    stats, timing = _run_cases(cases, stage=stage, expected_counts=STAGE_COUNTS[stage], progress=emit)
    report = {
        **_report_claims(stage), **stats, "gate_eligible": True,
        "gate_passed": stats["status"] == "pass",
        "three_path_exact": True if stats["status"] == "pass" else (False if stats["status"] == "mismatch" else None),
        "context_sha256": context.context_sha256, "provenance": _json(context.provenance_json),
        "sealed_file_sha256": dict(context.sealed_files),
        "election_bindings": [{"series": row.series, "year": row.year, "raw_filename": row.raw_path.name, "raw_sha256": row.raw_sha256, "v2_semantic_sha256": row.semantic_sha256} for row in context.elections],
        "selected_fits": [{"context": fit.context, "sha256": fit.sha256, "weights_hex": [value.hex() for value in fit.weights]} for fit in context.fits],
    }
    try:
        assert_context_unchanged(context)
    except Exception as error:
        report.update(status="error", gate_eligible=False, gate_passed=False, three_path_exact=None)
        report["failure"] = {"kind": "sealed_input_changed", "three_path_exact": None, **_safe_error(error), "prior_failure": stats["failure"]}
        context = None  # This report is explicitly an unauthenticated failure.
    timing["solver_elapsed_seconds"] = timing.pop("elapsed_seconds")
    timing["elapsed_seconds"] = time.perf_counter() - started
    if context is not None:
        report_sha = _sha(_json_bytes(report))
        context = replace(context, execution_report_sha256=report_sha, execution_tag=_authentication_tag("execution", context.context_sha256 + "/" + report_sha))
    return report, timing, context


_HELD_WRITERS: set[Path] = set()


@contextmanager
def report_writer(output: Path):
    parent = protocol_v2._ensure_nonsymlink_directory_v2(output.parent, "parity report directory")
    if parent in _HELD_WRITERS:
        raise RuntimeError("parity report single writer is already running")
    descriptor = os.open(parent, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("parity report single writer is already running") from error
        _HELD_WRITERS.add(parent)
        try:
            yield
        finally:
            _HELD_WRITERS.remove(parent)
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _timing_path(output: Path) -> Path:
    return output.with_name(output.stem + ".timing.json")


def _require_absent_outputs(output: Path) -> None:
    if output.suffix != ".json":
        raise RuntimeError("parity report output must have a .json suffix")
    for path in (output, _timing_path(output)):
        protocol_v2._require_nonsymlink_components_v2(path, "parity output", allow_missing=True)
        if path.exists():
            raise RuntimeError(f"immutable parity output already exists: {path}")


def _validate_execution_report(report: Mapping[str, object], context: AuthenticatedContext | None) -> None:
    """A report can pass only with the exact bytes sealed by a real stage run."""
    if any(report.get(key) is not False for key in ("float_runtime_proved", "cache_safe", "literal_branch_equivalence_claimed")):
        raise RuntimeError("parity report cannot authorize certificates, caches, or literal branch equivalence")
    if context is None:
        if report.get("gate_passed") is not False or report.get("gate_eligible") is not False or report.get("status") != "error" or report.get("three_path_exact") is not None:
            raise RuntimeError("an unauthenticated report cannot claim a parity gate verdict")
        return
    expected_factory = _authentication_tag("factory", context.context_sha256)
    if not hmac.compare_digest(context.factory_tag, expected_factory):
        raise RuntimeError("parity report context was not authenticated by the factory")
    report_sha = _sha(_json_bytes(report))
    expected_execution = _authentication_tag("execution", context.context_sha256 + "/" + report_sha)
    if report_sha != context.execution_report_sha256 or not hmac.compare_digest(context.execution_tag, expected_execution):
        raise RuntimeError("parity report differs from authenticated execution bytes")
    if report.get("context_sha256") != context.context_sha256 or not protocol_v2.exact_json_equal_v2(report.get("provenance"), _json(context.provenance_json)) or not protocol_v2.exact_json_equal_v2(report.get("sealed_file_sha256"), dict(context.sealed_files)):
        raise RuntimeError("parity report provenance differs from its execution context")
    if report.get("gate_passed") is True:
        stage = report.get("stage")
        if stage not in STAGE_COUNTS:
            raise RuntimeError("passing parity gate names an unknown stage")
        total, scored, warmup = STAGE_COUNTS[stage]
        expected = {
            **_report_claims(stage), "status": "pass", "gate_eligible": True,
            "gate_passed": True, "three_path_exact": True, "failure": None,
            "cases_attempted": total, "cases_compared": total, "cases_exact": total,
            "scored_cases": scored, "warmup_cases": warmup,
            "expected_counts": [total, scored, warmup],
            "completion_counts": {"false": total // 2, "true": total // 2},
            "context_counts": ({fit: 2 * counts[0] for fit, counts in FIT_COUNTS.items()} if stage == "full" else {f"uniform/{stage}": total}),
        }
        if any(not protocol_v2.exact_json_equal_v2(report.get(key), value) for key, value in expected.items()):
            raise RuntimeError("passing parity gate execution/count evidence is inconsistent")
    elif report.get("status") not in {"error", "mismatch"}:
        raise RuntimeError("nonpassing parity gate status is inconsistent")


def publish_report(output: Path, report: Mapping[str, object], timing: Mapping[str, object], context: AuthenticatedContext | None = None, *, _writer_held: bool = False) -> dict[str, object]:
    """Install timing first and the deterministic gate report last; never overwrite."""
    output = Path(output).absolute()
    if not _writer_held:
        with report_writer(output):
            return publish_report(output, report, timing, context, _writer_held=True)
    _require_absent_outputs(output)
    _validate_execution_report(report, context)
    report_bytes, timing_bytes = _json_bytes(report), _json_bytes(timing)
    timing_path = _timing_path(output)
    protocol_v2.preflight_immutable_bytes_artifact_v2(timing_path, timing_bytes, label="parity timing")
    protocol_v2.preflight_immutable_bytes_artifact_v2(output, report_bytes, label="parity gate report")
    if context is not None:
        assert_context_unchanged(context)
    summary = {"status": report["status"], "gate_passed": report["gate_passed"], "report_sha256": _sha(report_bytes), "report": str(output), "timing": str(timing_path)}
    protocol_v2.write_immutable_bytes_artifact_v2(timing_path, timing_bytes, label="parity timing")
    protocol_v2.write_immutable_bytes_artifact_v2(output, report_bytes, label="parity gate report")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_COUNTS), required=True)
    parser.add_argument("--output", type=Path, help="new gate report JSON path; timing uses a sibling .timing.json")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else ROOT / PARITY_RESULT_RELATIVE / args.stage / "gate_report.json"
    def progress(row):
        print(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    with report_writer(output):
        _require_absent_outputs(output)
        report, timing, context = run_authenticated_stage(args.stage, ROOT, progress)
        summary = publish_report(output, report, timing, context, _writer_held=True)
    progress({"event": "report-written", **summary})
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
