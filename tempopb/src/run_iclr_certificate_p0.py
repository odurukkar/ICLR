"""Bounded, append-only P0 certificate feasibility runner.

Run only after independent code review and explicit operational authorization.
Prior sealed campaigns are authenticated, never reexecuted. This program emits
no final report unless current synthetic and frozen-direction real gates pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, is_dataclass
from decimal import Decimal, DecimalException, localcontext
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import iclr_actuation_certificate as cert
from iclr_env import RolloutState
from iclr_residual_policy import AnchorSpec, FEATURE_NAMES, FeatureScaler, fit_context_scaler
from parse_pb import PBInstance, Project, Vote, parse_pb_file


_SENTINEL = '{"anchor":{"family":"age_sex","free_logits_hex":["-0x1.6666666666666p-1","-0x1.0000000000000p-1","-0x1.3333333333333p-2","-0x1.999999999999ap-4","0x1.999999999999ap-4","0x1.3333333333333p-2","0x1.0000000000000p-1"],"reference_cell":"age<25|F"},"backoff_exponent_range":[0,60],"base_weights_hex":["0x1.7ae147ae147aep-2","-0x1.d70a3d70a3d71p-3","0x1.a3d70a3d70a3dp-2","-0x1.851eb851eb852p-3"],"completion_modes":[false,true],"direction":[1,-1,2,-2],"input_sha256":"823c6b7489bd52d37d996ad6421ed8bb8eb1334575509eff50bdaaae1a31fbe3","parameter_box":[[-10,10],[-10,10],[-10,10],[-10,10]],"precision_schedule":[80,160,320,640],"scaler_sha256":"78033a6272654b43c25a8fa7cf448700c02cb111e5c02b073561dbda81b872ff"}'
_SENTINEL_SHA256 = "e83cb4222c0a6d676fb8bfa7a99930b8264adfad7a25c843a3698ffc2994dec3"
_PRIOR_REPORTS = (
    ("warsaw-sentinels", "cad8980ffc15484e7fd8ba58946f8ec1a202a2f030d59db933a5423c7f69d69e", 10),
    ("representative-temporal-scored", "01d86a4e86c3dd2432c9dbddd4e2132622e749faa8285dbc11bcfbe13ea9a196", 6),
    ("temporal-scored", "96eb59dc903f83b970cfcf3b2b3fa2d531c734992565de7bdb0717ec8843ab75", 446),
    ("native", "673261fde76521a1ae757acc067e92bfda4b4a69acaafbdd2de17f61dad1edbd", 794),
    ("full", "c7735e916f00d68a7550f1e0a08dd8b4d26a4f606ad7f5c4f5de04f08919567f", 3176),
)
_TEST_FILES = ("test_iclr_actuation_certificate.py", "test_iclr_trace_parity.py",
               "test_iclr_residual_actuation.py", "test_iclr_priority_mes.py",
               "test_iclr_residual_policy.py", "test_iclr_tie_breaking.py")
_GATES = ("prior", "integration_tests", "synthetic_campaign", "wesola")
_INTEGRATION_EXCLUSIONS = {
    "tests/test_iclr_actuation_certificate.py::test_wesola_frozen_nonuniform_symbolic_prerequisite[False]": "completed Task5 prerequisite; new directional audit is separate",
    "tests/test_iclr_actuation_certificate.py::test_wesola_frozen_nonuniform_symbolic_prerequisite[True]": "completed Task5 prerequisite; new directional audit is separate",
    "tests/test_iclr_trace_parity.py::test_five_repaired_warsaw_sentinels_have_zero_mismatches_in_both_modes": "legacy scientific campaign superseded by sealed corrected-v2 warsaw-sentinels gate",
    "tests/test_iclr_trace_parity.py::test_low_median_high_temporal_scored_files_have_zero_mismatches": "legacy scientific campaign superseded by sealed corrected-v2 representative gate",
    "tests/test_iclr_trace_parity.py::test_all_223_temporal_scored_elections_have_zero_mismatches": "opt-in legacy scientific campaign superseded by sealed corrected-v2 temporal-scored gate",
    "tests/test_iclr_trace_parity.py::test_all_397_native_elections_have_zero_uniform_mismatches": "opt-in legacy scientific campaign superseded by sealed corrected-v2 native gate",
    "tests/test_iclr_trace_parity.py::test_six_frozen_fits_have_zero_split_correct_mismatches": "opt-in legacy scientific campaign superseded by sealed corrected-v2 full gate",
}


def _verified_bytes(path: Path, expected: str) -> bytes:
    data = cert._read_regular_nofollow(path, "P0 evidence")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"evidence digest mismatch: {path}")
    return data


def wesola_config():
    if hashlib.sha256(_SENTINEL.encode()).hexdigest() != _SENTINEL_SHA256:
        raise ValueError("frozen Wesoła sentinel digest mismatch")
    return json.loads(_SENTINEL)


def load_prior_gates(root: Path):
    reports = []
    for stage, digest, count in _PRIOR_REPORTS:
        path = root / "results/iclr_trace_parity_v2" / stage / "gate_report.json"
        saved = json.loads(_verified_bytes(path, digest))
        if (saved["status"] != "pass" or saved["gate_passed"] is not True
                or saved["three_path_exact"] is not True or saved["failure"] is not None
                or saved["cases_compared"] != count or saved["cases_exact"] != count
                or saved["float_runtime_proved"] is not False or saved["cache_safe"] is not False):
            raise ValueError(f"prior gate is incomplete: {stage}")
        for module, source_digest in saved["provenance"]["source_sha256"].items():
            _verified_bytes(root / "src" / f"{module}.py", source_digest)
        reports.append({"stage": stage, "sha256": digest, "cases": count, "path": str(path)})
    logs = root.parent / "iclr_paper/strong_accept"
    stdout_sha = "b99c28a182483eb1098a9d5790ca3fe90a1fc9215c75ec3ef333b6c5fd07cc09"
    stderr_sha = "a511ff49857c90f95597e5c37fae4a2008e3d3c1506040de6dcdb671ee8a4fa5"
    stdout = _verified_bytes(logs / "wesola_certificate_20260911.stdout.log", stdout_sha).decode()
    _verified_bytes(logs / "wesola_certificate_20260911.stderr.log", stderr_sha)
    for mode in (False, True):
        if f"test_wesola_frozen_nonuniform_symbolic_prerequisite[{mode}] PASSED" not in stdout:
            raise ValueError("prior Task5 mode evidence missing")
    return {"parity_reports": reports, "prior_three_path_cases": sum(r["cases"] for r in reports),
            "task5_wesola_modes": [False, True], "task5_stdout_sha256": stdout_sha,
            "task5_stderr_sha256": stderr_sha, "reexecuted_prior_gates": False}


def dyadic_targets(config):
    base = tuple(Fraction.from_float(float.fromhex(v)) for v in config["base_weights_hex"])
    direction = tuple(Fraction(v) for v in config["direction"])
    result = []
    for exponent in range(config["backoff_exponent_range"][0], config["backoff_exponent_range"][1] + 1):
        requested = tuple(w + d / 2**exponent for w, d in zip(base, direction))
        weights = tuple(float(v) for v in requested)
        actual = tuple(Fraction.from_float(v) for v in weights)
        result.append({"exponent": exponent, "requested": requested, "weights": weights, "actual": actual,
                       "weights_hex": tuple(v.hex() for v in weights),
                       "actual_frozen_ray_parameter": cert._exact_ray_parameter(
                           base, tuple(w + d for w, d in zip(base, direction)), actual)})
    return result


def _synthetic_trace(completion, *, both=False, cost=5, weights=(0., 0., 0., 0.)):
    projects = {"a": Project("a", cost, None)}
    if both:
        projects.update({"b": Project("b", cost, None), "z": Project("z", 0, None)})
    inst = PBInstance("fixture://p0-runner", {"budget": "12"}, projects,
                      [Vote("0", ("a", "z") if both else ("a",), age=30, sex="F"),
                       Vote("1", ("b",) if both else (), age=70, sex="M")])
    scaler = FeatureScaler(FEATURE_NAMES, (0., 0., .125 if both else 0., 0.),
                           (1., 1., .375 if both else 1., 1.), 3., 2, 1)
    bound = cert.compile_bound_trace(inst, AnchorSpec("senior", (0.,), "age<25"), scaler,
                                     weights, RolloutState(), completion=completion)
    trace = cert.compile_signed_trace(bound)
    if type(trace) is not cert.CompiledTrace:
        raise ValueError(f"synthetic base did not compile: {trace.reason}")
    return trace


def _audit_counts(audit):
    return {"three_path_cases": audit.new_three_path_cases,
            "three_kernel_mismatches": sum(p.three_kernel_match is False for p in audit.probes),
            "covered_interior_violations": sum(p.covered_interior and
                (p.three_kernel_match is False or p.projection_matches is False) for p in audit.probes),
            "covered_endpoint_violations": sum(p.covered_by_real_proof and not p.covered_interior and
                (p.three_kernel_match is False or p.projection_matches is False) for p in audit.probes),
            "audit_only_projection_changes": sum(not p.covered_by_real_proof and p.projection_matches is False
                                                  for p in audit.probes)}


def _float_audit_summary(audit):
    """Separate observed outcomes from complete evidence for this runner's probes."""
    # audit_fixed_direction requests the three default dyadic interiors plus
    # the endpoint. Status alone cannot establish that all four were verified.
    expected = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1))
    counts = _audit_counts(audit)
    complete = (audit.base.status == "enclosed" and audit.mismatch_count is not None
                and tuple(p.requested_parameter for p in audit.probes) == expected
                and all(p.trace_validated is True and type(p.three_kernel_match) is bool
                        and type(p.projection_matches) is bool for p in audit.probes))
    outcomes = {key: value for key, value in counts.items() if key != "three_path_cases"}
    observed = {"probe_records": len(audit.probes), "three_path_cases": audit.new_three_path_cases,
                "trace_validated_probes": sum(p.trace_validated is True for p in audit.probes),
                "known_three_kernel_results": sum(type(p.three_kernel_match) is bool for p in audit.probes),
                "known_projection_results": sum(type(p.projection_matches) is bool for p in audit.probes),
                "covered_interior_points": audit.covered_interior_count,
                **{f"known_{key}": value for key, value in outcomes.items()}}
    return {"status": audit.status, "reason": audit.reason, "route_failure": audit.route_failure,
            "mismatch_count": audit.mismatch_count, "expected_probe_count": len(expected),
            "sample_evidence_complete": complete,
            "sample_parity_verified": complete and audit.status == "sampled_parity" and audit.mismatch_count == 0,
            "observed_sample_counts": observed,
            "observed_count_scope": "known observations only; unknown/unattempted outcomes are not zero",
            "three_path_cases": audit.new_three_path_cases,
            "covered_interiors": audit.covered_interior_count if complete else None,
            **{key: value if complete else None for key, value in outcomes.items()}}


def run_synthetic_campaign():
    cases, counts = [], Counter()
    with localcontext() as context:
        context.prec = 100
        x = (Decimal(13) / 11).ln() / Decimal(2).ln()
        root = ((1 + x) / (1 - x)).ln() / 2
    boundary = float(root)
    for completion in (False, True):
        single, double = _synthetic_trace(completion), _synthetic_trace(completion, both=True)
        charge = _synthetic_trace(completion, cost=6, weights=(0., .1, 0., 0.))
        counts["three_path_cases"] += 3  # Three base bindings, each ran all kernels.
        requests = [("same_point", single, (0.,) * 4, ((-2, 2),) * 4, "uncertified"),
                    ("zero_score_direction", single, (.1, 0., 0., 0.), ((-2, 2),) * 4, "proved_real")]
        for both, trace in ((False, single), (True, double)):
            for side, value in (("inside", math.nextafter(boundary, -math.inf)),
                                ("representative", boundary), ("outside", math.nextafter(boundary, math.inf))):
                target = (0., 0., value, 0.) if both else (0., -value, 0., 0.)
                requests.append((f"conservative_{both}_{side}", trace, target, ((-2, 2),) * 4,
                                 "proved_real" if Decimal.from_float(value) < root else "uncertified"))
                counts["conservative_boundary_probes"] += 1
        for dim in range(4):
            for face in (-.125, .125):
                for side, value in (("lower", math.nextafter(face, -math.inf)), ("face", face),
                                    ("upper", math.nextafter(face, math.inf))):
                    target = [0.] * 4
                    target[dim] = value
                    requests.append((f"box_{dim}_{face}_{side}", single, tuple(target), ((-.125, .125),) * 4,
                                     "proved_real" if -.125 <= value <= .125 else "uncertified"))
                    counts["box_boundary_probes"] += 1
        for value in (math.nextafter(0., -math.inf), 0., math.nextafter(0., math.inf)):
            requests.append((f"true_charge_{value.hex()}", charge, (0., value, 0., 0.), ((-2, 2),) * 4, "uncertified"))
            counts["true_charge_boundary_probes"] += 1
        for name, trace, target, box, expected in requests:
            audit = cert.audit_float_segment(trace, target, box, include_endpoint=True)
            if audit.real_proof.status != expected or audit.status != "sampled_parity" or audit.route_failure:
                raise ValueError(f"synthetic gate failed: {completion}:{name}:{audit.reason}")
            counts.update(_audit_counts(audit))
            cases.append({"id": name, "completion": completion, "proof_status": audit.real_proof.status,
                          "audit": audit})
    return {"gate_passed": True, "completion_modes": [False, True], "cases": cases,
            **{key: counts[key] for key in ("three_path_cases", "three_kernel_mismatches", "covered_interior_violations", "covered_endpoint_violations",
                "audit_only_projection_changes", "conservative_boundary_probes", "box_boundary_probes", "true_charge_boundary_probes")}}


def _integration_pytest_args(root, mode, junit):
    # Ignore project/ancestor configuration, conftest hooks and all implicit
    # selectors. The exact seven historical nodes are the only deselections.
    args = ["-q", "-c", os.devnull, f"--rootdir={root}", "--noconftest", "-p", "no:cacheprovider",
            "-o", "addopts=", "-k", "", "-m", "", *(f"tests/{name}" for name in _TEST_FILES)]
    if mode == "collect":
        return [*args, "--collect-only"]
    if mode != "execute":
        raise ValueError("unknown integration worker mode")
    return [*args, *(f"--deselect={node}" for node in _INTEGRATION_EXCLUSIONS), f"--junitxml={junit}"]


class _IntegrationRecorder:
    """Observe actual pytest identities and all three execution phases."""

    def __init__(self):
        self.inventory, self.reports = [], []

    def pytest_collection_finish(self, session):
        self.inventory = [item.nodeid for item in session.items]
        for item in session.items:
            item.user_properties.append(("p0_nodeid", item.nodeid))

    def pytest_runtest_logreport(self, report):
        self.reports.append({"nodeid": report.nodeid, "when": report.when,
                             "outcome": report.outcome, "xfail": getattr(report, "wasxfail", None)})


def _integration_worker(root, mode, output, junit):
    import pytest
    recorder = _IntegrationRecorder()
    exit_code = pytest.main(_integration_pytest_args(root, mode, junit), plugins=[recorder])
    Path(output).write_text(json.dumps({"inventory": recorder.inventory, "reports": recorder.reports},
                                      sort_keys=True, separators=(",", ":")))
    return int(exit_code)


def _required_integration_inventory(collected):
    modules = {f"tests/{name}" for name in _TEST_FILES}
    if (not collected or any(type(node) is not str for node in collected)
            or len(set(collected)) != len(collected)
            or any(node.split("::", 1)[0] not in modules for node in collected)):
        raise ValueError("integration collection has missing, duplicate or unexpected identities")
    if not set(_INTEGRATION_EXCLUSIONS).issubset(collected):
        raise ValueError("integration collection is missing an approved historical exclusion")
    required = tuple(node for node in collected if node not in _INTEGRATION_EXCLUSIONS)
    if {node.split("::", 1)[0] for node in required} != modules:
        raise ValueError("integration collection is missing a required module")
    return required


def _validate_integration_execution(required, execution, junit):
    if not required or len(set(required)) != len(required):
        raise ValueError("integration required inventory is empty or duplicated")
    if Counter(execution["inventory"]) != Counter(required):
        raise ValueError("integration executed inventory differs from independently collected requirements")
    reports = execution["reports"]
    phases = Counter((node, phase) for node in required for phase in ("setup", "call", "teardown"))
    if (Counter((r["nodeid"], r["when"]) for r in reports) != phases
            or any(r["outcome"] != "passed" or r["xfail"] is not None for r in reports)):
        raise ValueError("integration has failed, skipped, duplicate or unverified required phases")
    actual = []
    for case in ET.parse(junit).getroot().iter("testcase"):
        identities = [p.attrib.get("value") for p in case.findall("./properties/property")
                      if p.attrib.get("name") == "p0_nodeid"]
        if len(identities) != 1 or any(child.tag != "properties" for child in case):
            raise ValueError("integration JUnit has missing identity or unsuccessful evidence")
        actual.append(identities[0])
    if Counter(actual) != Counter(required):
        raise ValueError("integration JUnit inventory is partial, duplicate or unexpected")


def _integration_source_directory():
    return str(Path(__file__).resolve().parent)


def _run_current_integration_tests(root, diagnostic=None):
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    for key in ("ICLR_TRACE_PARITY_STAGE", "TEMPOPB_RUN_WESOLA_CERTIFICATE", "PYTHONOPTIMIZE", "PYTHONINSPECT"):
        environment.pop(key, None)
    environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONPATH=_integration_source_directory())
    selection = {key: environment.get(key) for key in
                 ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH",
                  "ICLR_TRACE_PARITY_STAGE", "TEMPOPB_RUN_WESOLA_CERTIFICATE", "PYTHONOPTIMIZE", "PYTHONINSPECT")}
    test_hashes = {name: hashlib.sha256(cert._read_regular_nofollow(root / "tests" / name, "integration test")).hexdigest()
                   for name in _TEST_FILES}
    entrypoint = "import sys; from run_iclr_certificate_p0 import _integration_worker; raise SystemExit(_integration_worker(*sys.argv[1:]))"
    # Canonicalize only the OS-owned temp base (macOS /var aliases /private/var),
    # before creating our private directory; never resolve diagnostic targets.
    with tempfile.TemporaryDirectory(prefix="iclr-p0-integration-", dir=Path(tempfile.gettempdir()).resolve()) as directory:
        junit = Path(directory) / "integration.xml"
        phases = {}
        for mode in ("collect", "execute"):
            output = Path(directory) / f"{mode}.json"
            command = [sys.executable, "-c", entrypoint, str(root), mode, str(output), str(junit)]
            stdout_path, stderr_path = Path(directory) / f"{mode}-stdout.txt", Path(directory) / f"{mode}-stderr.txt"
            details = {"command": command, "pytest_args": _integration_pytest_args(root, mode, junit),
                       "selection_environment": selection, "selection_environment_sha256": cert._canonical_sha256(selection),
                       "test_sha256": test_hashes, "exclusions": dict(_INTEGRATION_EXCLUSIONS)}
            token = diagnostic.begin(f"integration_{mode}") if diagnostic else None
            completed = None
            try:
                with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
                    completed = subprocess.run(command, cwd=root, env=environment, stdout=stdout_file,
                                               stderr=stderr_file, text=True, check=False)
                stdout, stderr = stdout_path.read_text(), stderr_path.read_text()
                if completed.returncode != 0 or not output.exists():
                    raise ValueError(f"integration {mode} failed: {stdout[-2000:]} {stderr[-2000:]}")
                observed = json.loads(output.read_text())
                if mode == "collect":
                    required = _required_integration_inventory(observed["inventory"])
                else:
                    _validate_integration_execution(required, observed, junit)
                phases[mode] = {"command": command, "pytest_args": _integration_pytest_args(root, mode, junit),
                                "command_sha256": cert._canonical_sha256(command),
                                "pytest_args_sha256": cert._canonical_sha256(_integration_pytest_args(root, mode, junit)),
                                "evidence_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                                "inventory": observed["inventory"]}
            finally:
                if diagnostic:
                    diagnostic.retain(mode, {f"{mode}.json": output, "stdout.txt": stdout_path,
                                             "stderr.txt": stderr_path, "junit.xml": junit},
                                      returncode=None if completed is None else completed.returncode, **details)
            if diagnostic:
                diagnostic.end(token, {"inventory_count": len(observed["inventory"]),
                                        "inventory_sha256": cert._canonical_sha256(observed["inventory"]),
                                        "required_count": len(required), "validated": True})
        if any(hashlib.sha256(cert._read_regular_nofollow(root / "tests" / name, "integration test")).hexdigest() != digest
               for name, digest in test_hashes.items()):
            raise ValueError("integration test source changed during execution")
        selected_hash = cert._canonical_sha256(sorted(required))
        executed_hash = cert._canonical_sha256(sorted(phases["execute"]["inventory"]))
        return {"passed": len(required), "inventory": list(required), "excluded": dict(_INTEGRATION_EXCLUSIONS),
                "phases": phases, "selection_environment": selection,
                "selection_environment_sha256": cert._canonical_sha256(selection), "test_sha256": test_hashes,
                "collected_inventory_sha256": cert._canonical_sha256(sorted(phases["collect"]["inventory"])),
                "selected_inventory_sha256": selected_hash, "executed_inventory_sha256": executed_hash,
                "exclusions_sha256": cert._canonical_sha256(_INTEGRATION_EXCLUSIONS),
                "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest()}


def screen_sufficient_bound(trace, target):
    """Reject only a proved failure of the existing sufficient bound; never certify.

    Sealed base margin upper bounds and a lower envelope movement bound can
    exclude distant dyadic points without re-enclosing every base guard. An
    inconclusive screen always falls through to the full reviewed factory.
    """
    _check_screen_trace(trace)
    return _screen_at_target(trace, target, _rank_screen_guards(trace))


def _check_screen_trace(trace):
    """Run every original check; return identities, never a reusable authorization."""
    if type(trace) is not cert.CompiledTrace:
        raise TypeError("screen requires a concrete CompiledTrace")
    cert._require_bound(trace.bound)
    cert._require_compiled_decision_types(trace)
    digest = cert._compiled_payload_digest(trace)
    if digest != trace.payload_digest:
        raise ValueError("screen compiled payload mismatch")
    cert.CompiledTrace.validate_coverage(trace)
    return trace.bound.binding.context_digest, digest


def _rank_screen_guards(trace):
    """Exact target-independent selection; retain only bounded selected metadata."""
    candidates = []
    for index, event in enumerate(trace.events):
        decision = event.decision
        if decision is None or decision.kind is not cert.DecisionKind.VARIABLE:
            continue
        factor = decision.guard.spec.budget * (max(decision.guard.h) - min(decision.guard.h))
        if factor > 0:
            candidates.append((Fraction(decision.margin.upper) / factor, index, factor))
    if not candidates:
        return None
    _, index, factor = min(candidates, key=lambda item: item[0])
    event = trace.events[index]
    return index, event.event_id, factor, event.decision.margin.upper


def _screen_at_target(trace, target, selection):
    """Private arithmetic only; callers must freshly authenticate before entry."""
    if selection is None:
        return {"excluded": False, "reason": "no_variable_movement"}
    _, event_id, factor, margin_upper = selection
    surface = trace.surface
    oracle = cert.BaseSignOracle(surface.spec, surface.anchor_logits, surface.feature_rows, surface.base_weights)
    weights = tuple(cert._exact_fraction(cert._float_from_bits(v)) for v in cert._freeze_weights(target))
    try:
        changes, _, _, _ = cert._segment_envelope(oracle, weights, 80)
        arithmetic = cert._BaseArithmetic(80)
        envelope_lower = arithmetic.down.subtract(max(Decimal(0), *(d.lower for d in changes)),
                                                   min(Decimal(0), *(d.upper for d in changes)))
        ratio = arithmetic.tanh(arithmetic.div(cert._DecimalInterval(envelope_lower, envelope_lower),
                                               arithmetic.lift(4)))
        movement_lower = arithmetic.mul(arithmetic.lift(factor), ratio).lower
    except (ArithmeticError, DecimalException) as exc:
        return {"excluded": False, "reason": f"screen_arithmetic_failure:{type(exc).__name__}"}
    return {"excluded": margin_upper <= movement_lower, "precision": 80,
            "limiting_guard": event_id, "margin_upper": margin_upper,
            "movement_lower": movement_lower, "envelope_lower": envelope_lower,
            "endpoint_changes": changes, "reason": "sufficient_bound_test_only"}


def _screen_evaluator(trace):
    """Own one audit's ranking, not its authentication or target results.

    The closure strongly retains the trace. Its first successful authentication
    fixes the context/seal pair; every later call repeats all checks before
    reading selection metadata. No caller supplies or receives memo state.
    As with the stateless screen, concurrent mutation within a call is outside
    the existing model; this adds no cross-call unchecked interval.
    """
    identity, selection, closed = None, None, False

    def evaluate(target):
        nonlocal identity, selection, closed
        if closed:
            raise ValueError("screen evaluator is closed after failure")
        try:
            checked = _check_screen_trace(trace)
            if identity is None:
                selection = _rank_screen_guards(trace)
                identity = checked
            elif checked != identity:
                raise ValueError("screen invocation context or compiled identity changed")
            return _screen_at_target(trace, target, selection)
        except BaseException:
            identity, selection, closed = None, None, True
            raise

    return evaluate


def audit_fixed_direction(trace, config, diagnostic=None):
    """First proved endpoint; the gate additionally requires a usable interior."""
    trials, audit = [], None
    evaluate_screen = _screen_evaluator(trace)
    for point in dyadic_targets(config):
        print(f"Directional audit completion={trace.bound.binding.completion}: dyadic exponent={point['exponent']}", flush=True)
        metadata = {"completion": trace.bound.binding.completion, "exponent": point["exponent"],
                    "requested_target": point["requested"], "actual_target": point["actual"],
                    "target_bits": point["weights_hex"], "frozen_ray_parameter": point["actual_frozen_ray_parameter"]}
        token = diagnostic.begin("trial", **metadata) if diagnostic else None
        screen = _observed(diagnostic, "screen", lambda: evaluate_screen(point["weights"]),
                           lambda value: {key: value.get(key) for key in
                               ("excluded", "reason", "precision", "limiting_guard", "margin_upper", "movement_lower", "envelope_lower")},
                           **metadata)
        proof = None if screen["excluded"] else _observed(
            diagnostic, "segment_proof", lambda: cert.certify_real_segment(trace, point["weights"], config["parameter_box"]),
            _proof_summary, **metadata)
        trials.append({**point, "screen": screen, "proof": proof})
        if diagnostic:
            diagnostic.end(token, {"screen_excluded": screen["excluded"],
                                   "proof_status": None if proof is None else proof.status,
                                   "proof_not_executed_reason": "screen_excluded" if proof is None else None})
        ray_parameter = point["actual_frozen_ray_parameter"]
        if proof is not None and proof.status == "proved_real" and ray_parameter is not None and ray_parameter > 0:
            audit = _observed(diagnostic, "float_audit", lambda: cert.audit_float_segment(
                trace, point["weights"], config["parameter_box"], requested_target=point["requested"], include_endpoint=True),
                _float_audit_summary, **metadata)
            if audit.route_failure:
                raise ValueError("covered-interior/endpoint violation: stop certificate route")
            break
    return {"completion": trace.bound.binding.completion, "trials": trials, "audit": audit,
            "unattempted_exponents": list(range(len(trials), config["backoff_exponent_range"][1] + 1)),
            "passed": audit is not None and audit.status == "sampled_parity"
                       and audit.covered_interior_count > 0 and audit.mismatch_count == 0}


def run_wesola_campaign(root, diagnostic=None):
    config = wesola_config()
    path = root / "data/pb/Poland_Warszawa_2020_Wesola.pb"
    _observed(diagnostic, "real_input_validation", lambda: _verified_bytes(path, config["input_sha256"]),
              lambda data: {"actual_input_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
              input_path=str(path), expected_input_sha256=config["input_sha256"])
    inst = parse_pb_file(path)
    scaler = fit_context_scaler([inst])
    if scaler.sha256 != config["scaler_sha256"] or (len(inst.votes), len(inst.projects)) != (1505, 21):
        raise ValueError("frozen Wesoła input/scaler dimensions changed")
    anchor = AnchorSpec(config["anchor"]["family"], tuple(float.fromhex(v) for v in config["anchor"]["free_logits_hex"]),
                        config["anchor"]["reference_cell"])
    base_weights = tuple(float.fromhex(v) for v in config["base_weights_hex"])
    modes = []
    for completion in config["completion_modes"]:
        print(f"Wesoła completion={completion}: compiling current base for directional audit", flush=True)
        bound = _observed(diagnostic, "real_bound_compile", lambda: cert.compile_bound_trace(
            inst, anchor, scaler, base_weights, RolloutState(), completion=completion, input_paths={"wesola": path}),
            lambda value: {"result_type": type(value).__name__, "context_digest": value.context_digest
                           if type(value) is cert.UncertifiedTrace else value.binding.context_digest,
                           "reason": getattr(value, "reason", None)}, completion=completion)
        if type(bound) is not cert.BoundTrace:
            raise ValueError(f"Wesoła current base unverified: {bound.reason}")
        trace = _observed(diagnostic, "real_symbolic_compile", lambda: cert.compile_signed_trace(bound),
                          lambda value: {"result_type": type(value).__name__, "reason": getattr(value, "reason", None),
                                         "events": len(value.events) if type(value) is cert.CompiledTrace else None,
                                         "compiled_payload_digest": getattr(value, "payload_digest", None)}, completion=completion)
        if type(trace) is not cert.CompiledTrace:
            raise ValueError(f"Wesoła current exact base unverified: {trace.reason}")
        modes.append(audit_fixed_direction(trace, config, diagnostic=diagnostic))
    return {"gate_passed": len(modes) == 2 and all(m["passed"] for m in modes),
            "config": config, "config_sha256": _SENTINEL_SHA256, "modes": modes,
            "interpretation": "first successful fixed-direction dyadic backoff; later exponents not attempted"}


def _json_value(value):
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_final_report(path: Path, report):
    if any(report.get("gates", {}).get(name) is not True for name in _GATES):
        raise ValueError("all actual deterministic gates must pass before report creation")
    if report.get("float_runtime_proved") is not False or report.get("cache_safe") is not False:
        raise ValueError("report cannot authorize runtime proof or cache use")
    payload = (json.dumps(_json_value(report), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    absolute = Path(os.path.abspath(path))
    directory = _open_directory(absolute.parent)
    try:
        _publish_at(directory, absolute.name, payload)
    finally:
        os.close(directory)


def _open_directory(path):
    absolute = Path(os.path.abspath(path))
    directory = os.open(absolute.anchor, cert._DIRECTORY_FLAGS)
    try:
        for component in absolute.parts[1:]:
            try:
                os.mkdir(component, dir_fd=directory)
                os.fsync(directory)
            except FileExistsError:
                pass
            child = os.open(component, cert._DIRECTORY_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
    except BaseException:
        os.close(directory)
        raise
    return directory


def _publish_at(directory, name, payload):
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("artifact requires one local filename")
    # Atomic exclusive link publishes only fully written bytes and never
    # overwrites an earlier report, including a symlink at the destination.
    temporary = f".p0-report-{secrets.token_hex(16)}"
    created = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
    finally:
        if created:
            os.unlink(temporary, dir_fd=directory)


class _DiagnosticRun:
    """Append-only observations, never authority to skip gates or reuse proofs."""

    def __init__(self, base):
        self.run_id = "run-" + secrets.token_hex(16)
        self.path = Path(os.path.abspath(base)) / self.run_id
        parent = _open_directory(base)
        try:
            os.mkdir(self.run_id, mode=0o700, dir_fd=parent)  # No collision retry.
            os.fsync(parent)
            self.directory = os.open(self.run_id, cert._DIRECTORY_FLAGS, dir_fd=parent)
        finally:
            os.close(parent)
        self.sequence, self.previous, self.artifact_sequence = 0, None, 0
        self.manifest_sha256 = None
        self.stack, self.current, self.last_completed = [], None, None
        self.started = self._snapshot()
        print(f"Diagnostic-only evidence: {self.path}", flush=True)
        try:
            self.record("run_created", summary=None)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _snapshot():
        return (time.monotonic(), resource.getrusage(resource.RUSAGE_SELF),
                resource.getrusage(resource.RUSAGE_CHILDREN))

    def resources(self, component_start=None):
        wall, usage, children = self._snapshot()
        begin, old, old_children = self.started
        return {"wall_seconds": wall - begin, "self_user_seconds": usage.ru_utime - old.ru_utime,
                "self_system_seconds": usage.ru_stime - old.ru_stime,
                "children_user_seconds": children.ru_utime - old_children.ru_utime,
                "children_system_seconds": children.ru_stime - old_children.ru_stime,
                "self_max_rss": usage.ru_maxrss, "children_max_rss": children.ru_maxrss,
                "rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
                "component_wall_seconds": None if component_start is None else wall - component_start[0],
                "component_self_cpu_seconds": None if component_start is None else
                    usage.ru_utime + usage.ru_stime - component_start[1].ru_utime - component_start[1].ru_stime,
                "component_children_cpu_seconds": None if component_start is None else
                    children.ru_utime + children.ru_stime - component_start[2].ru_utime - component_start[2].ru_stime}

    def record(self, event, **data):
        record = {"schema": "iclr-certificate-p0-diagnostic-v1", "run_id": self.run_id,
                  "sequence": self.sequence + 1, "previous_sha256": self.previous,
                  "manifest_sha256": self.manifest_sha256, "event": event,
                  "stage": None, "completion": None, "summary": None,
                  "resources": self.resources(), **data,
                  "diagnostic_only": True, "authorizes_gate_reuse": False, "p0_result": None,
                  "float_runtime_proved": False, "cache_safe": False}
        payload = (json.dumps(_json_value(record), sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode()
        if len(payload) > 256 * 1024:
            raise ValueError("diagnostic summary exceeds bounded record size")
        _publish_at(self.directory, f"record-{self.sequence + 1:06d}.json", payload)
        self.sequence += 1
        self.previous = hashlib.sha256(payload).hexdigest()

    def identify(self, identities, config):
        self.record("manifest", identities=identities, config=config,
                    config_sha256=cert._canonical_sha256(config), expected_input_sha256=config["input_sha256"],
                    summary=None)
        self.manifest_sha256 = self.previous

    def begin(self, stage, **metadata):
        token = {"component": {"stage": stage, **metadata}, "started": self._snapshot()}
        self.stack.append(token)
        self.current = token["component"]
        self.record("stage_start", **self.current, summary=None)
        return token

    def end(self, token, summary):
        if not self.stack or self.stack[-1] is not token:
            raise ValueError("diagnostic stage nesting mismatch")
        self.record("stage_end", **token["component"], summary=summary,
                    resources=self.resources(token["started"]))
        self.last_completed = {**token["component"], "record_sha256": self.previous}
        self.stack.pop()
        self.current = self.stack[-1]["component"] if self.stack else None

    def call(self, stage, operation, summarize, **metadata):
        token = self.begin(stage, **metadata)
        value = operation()
        self.end(token, summarize(value))
        return value

    def retain(self, mode, files, **details):
        artifacts = {}
        for label, path in files.items():
            if not path.exists():
                artifacts[label] = None
                continue
            payload = cert._read_regular_nofollow(path, "integration diagnostic evidence")
            self.artifact_sequence += 1
            name = f"artifact-{self.artifact_sequence:06d}-{mode}-{label}"
            _publish_at(self.directory, name, payload)
            artifacts[label] = {"name": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        self.record("integration_artifacts", mode=mode, artifacts=artifacts, validation="not_asserted",
                    summary=None, **details)

    def fail(self, error):
        category = ("timeout" if isinstance(error, TimeoutError) else
                    "evidence_or_io" if isinstance(error, OSError) else "execution_or_validation")
        self.record("terminal_error", error_category=category, error_type=type(error).__name__,
                    error=str(error)[:1000], current_component=self.current,
                    last_completed=self.last_completed, summary=None)

    def close(self):
        if self.directory is not None:
            os.close(self.directory)
            self.directory = None


def _observed(diagnostic, stage, operation, summarize, **metadata):
    return operation() if diagnostic is None else diagnostic.call(stage, operation, summarize, **metadata)


def _proof_summary(proof):
    return {"status": proof.status, "reason": proof.reason, "guard_count": len(proof.guards),
            "limiting_guard": proof.limiting_guard, "precisions": proof.precisions,
            "envelope_upper": proof.envelope_upper, "ratio_upper": proof.ratio_upper}


def _identities(root):
    sources = dict(cert._snapshot_source_roles())
    sources["runner"] = hashlib.sha256(cert._read_regular_nofollow(Path(__file__).resolve(), "runner")).hexdigest()
    tests = {name: hashlib.sha256(cert._read_regular_nofollow(root / "tests" / name, "test")).hexdigest()
             for name in _TEST_FILES}
    return {"sources": sources, "tests": tests, "runtime": cert._runtime_payload(cert._runtime_snapshot())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, required=True,
                        help="reviewed one-shot wall-time cap; expiry writes no final report")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 3600:
        parser.error("timeout must be between1 and3600 seconds")
    root = Path(__file__).resolve().parents[1]
    output = root / "analysis-output/iclr_certificate_p0/report.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    started, usage = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    children_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    def expire(signum, frame):
        raise TimeoutError("bounded P0 run expired; no final report")
    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(args.timeout_seconds)
    diagnostic = None
    try:
        diagnostic = _DiagnosticRun(output.parent / "diagnostics")
        identities = diagnostic.call("identities", lambda: _identities(root),
                                     lambda value: {"sha256": cert._canonical_sha256(value)})
        diagnostic.identify(identities, wesola_config())
        prior = diagnostic.call("prior_authentication", lambda: load_prior_gates(root), lambda value: value)
        tests = diagnostic.call("integration", lambda: _run_current_integration_tests(root, diagnostic=diagnostic),
                                lambda value: {key: value.get(key) for key in
                                    ("passed", "selected_inventory_sha256", "executed_inventory_sha256", "junit_sha256")})
        synthetic = diagnostic.call("synthetic", run_synthetic_campaign,
                                    lambda value: {"gate_passed": value.get("gate_passed"),
                                                   "cases": len(value["cases"]) if "cases" in value else None,
                                                   **{key: value.get(key) for key in
                                                      ("three_path_cases", "three_kernel_mismatches", "covered_interior_violations")}})
        real = diagnostic.call("real", lambda: run_wesola_campaign(root, diagnostic=diagnostic),
                               lambda value: {"gate_passed": value.get("gate_passed"),
                                              "completed_modes": len(value["modes"]) if "modes" in value else None})
        if not real["gate_passed"]:
            raise ValueError("Wesoła usable covered interiors unverified; no final report")
        if diagnostic.call("identity_recheck", lambda: _identities(root),
                           lambda value: {"matches": value == identities}) != identities:
            raise ValueError("source/test/runtime identity changed during run")
        preparation = diagnostic.begin("final_preparation")
        finished = resource.getrusage(resource.RUSAGE_SELF)
        children_finished = resource.getrusage(resource.RUSAGE_CHILDREN)
        audits = [mode["audit"] for mode in real["modes"]]
        totals = Counter()
        for audit in audits:
            totals.update(_audit_counts(audit))
        trials = [trial for mode in real["modes"] for trial in mode["trials"]]
        report = {"schema": "iclr-certificate-p0-v1", "status": "passed",
                  "gates": {name: True for name in _GATES}, "identities": identities,
                  "prior_gates": prior, "integration_tests": tests, "synthetic": synthetic, "wesola": real,
                  "count_scope": "explicit campaign three-kernel groups only; historical gates separate; pytest internal solver calls uninstrumented",
                  "pytest_internal_solver_calls": None,
                  "wesola_counts": {**dict(totals), "current_base_three_path_cases": 2,
                                    "covered_interiors": sum(a.covered_interior_count for a in audits)},
                  "directional_counts": {"positive_proved_segments": sum(t["proof"] is not None and t["proof"].status == "proved_real" for t in trials),
                                         "abstentions": sum(t["proof"] is not None and t["proof"].status == "uncertified" for t in trials),
                                         "screened_sufficient_bound_failures": sum(t["screen"]["excluded"] for t in trials),
                                         "positive_frozen_ray_segments": sum(t["proof"] is not None and t["proof"].status == "proved_real"
                                             and t["actual_frozen_ray_parameter"] is not None and t["actual_frozen_ray_parameter"] > 0 for t in trials),
                                         "exact_zero_radius": None,
                                         "zero_radius_note": "abstention does not prove a zero maximal radius"},
                  "active_guards": dict(Counter(t["proof"].limiting_guard[1] for t in trials
                                                if t["proof"] is not None and t["proof"].limiting_guard is not None)),
                  "resource": {"wall_seconds": time.monotonic() - started, "timeout_seconds": args.timeout_seconds,
                               "self_user_seconds": finished.ru_utime - usage.ru_utime,
                               "self_system_seconds": finished.ru_stime - usage.ru_stime,
                               "children_user_seconds": children_finished.ru_utime - children_usage.ru_utime,
                               "children_system_seconds": children_finished.ru_stime - children_usage.ru_stime,
                               "self_max_rss_native_units": finished.ru_maxrss,
                               "children_max_rss_native_units": children_finished.ru_maxrss,
                               "rss_unit": "bytes" if sys.platform == "darwin" else "KiB", "platform": sys.platform},
                  "claim": "real-arithmetic certificate with benchmark-audited floating-point outcome parity",
                  "scope": "frozen Wesoła feasibility sentinel and enumerated synthetic fixtures; directional segments only",
                  "source_hashes_are_execution_proof": False,
                  "float_runtime_proved": False, "cache_safe": False}
        # Commit all required diagnostics before publishing the final success.
        # No mandatory diagnostic write after report.json can turn success into
        # an evidence failure while leaving a newly published success behind.
        diagnostic.end(preparation, {"gates": report["gates"]})
        diagnostic.begin("final_publication")
        diagnostic.record("ready_for_final_publication", summary={"gates": report["gates"]},
                          final_report_published=False, final_report_path=str(output))
        report["diagnostic_evidence"] = {"run_id": diagnostic.run_id, "path": str(diagnostic.path),
                                         "last_record_sha256": diagnostic.previous, "records": diagnostic.sequence}
        write_final_report(output, report)
        print(f"All gates passed; append-only report: {output}", flush=True)
        return 0
    except BaseException as error:
        if diagnostic is not None:
            try:
                diagnostic.fail(error)
            except BaseException as receipt_error:
                print(f"Terminal diagnostic publication failed: {type(receipt_error).__name__}: {receipt_error}", file=sys.stderr, flush=True)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if diagnostic is not None:
            diagnostic.close()


if __name__ == "__main__":
    raise SystemExit(main())
