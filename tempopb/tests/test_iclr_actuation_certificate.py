"""Focused contracts for immutable, process-scoped trace bindings."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal, InvalidOperation, Overflow, Underflow, localcontext
from fractions import Fraction
import math
import os
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

import iclr_actuation_certificate as certificate
from iclr_env import RolloutState
from iclr_policy import instance_features
from iclr_priority_mes import fast_mes_with_endowments
from iclr_residual_policy import AnchorSpec, FEATURE_NAMES, FeatureScaler
from parse_pb import PBInstance, Project, Vote


# These legal sites are handwritten independently of the emitted trace.
_CONTROL_INVENTORY = {
    "initial_support": ("static",), "initial_cost": ("static",),
    "affordability": ("variable", "structural"),
    "payer_order": ("variable", "structural"),
    "breakpoint": ("variable", "structural"),
    "rho_order": ("variable", "structural"),
    "candidate_cost": ("static",), "candidate_id": ("static",),
    "charge": ("variable", "structural"),
    "completion_count": ("static",), "completion_cost": ("static",),
    "completion_id": ("static",), "completion_positive": ("static",),
    "completion_affordability": ("static",),
    "initial_active": (None,), "active_before": (None,),
    "candidate_visit": (None,), "removal": (None,),
    "incumbent": (None,), "chosen": (None,), "active_after": (None,),
    "termination": (None,), "completion_mode": (None,),
    "completion_visit": (None,), "completion_skip": (None,),
    "completion_remaining": (None,), "completion_decision": (None,),
    "final_outcome": (None,),
}


def test_coverage_literal_registry_and_one_use_obligations():
    assert hasattr(certificate, "CONTROL_SITES"), "independent registry is missing"
    assert {site.name: tuple(None if k is None else k.value for k in site.kinds)
            for site in certificate.CONTROL_SITES} == _CONTROL_INVENTORY
    ledger = certificate._CoverageLedger()
    obligation = ledger.open("initial_active", ("initial",))
    event = ledger.emit(obligation, "snapshot", ("a",))
    with pytest.raises(ValueError, match="consumed"):
        ledger.emit(obligation, "snapshot", ("a",))
    ledger.validate((event,))
    for damaged in ((), (event, event), (replace(event, event_id=(99, "initial_active", ("initial",))),)):
        with pytest.raises(ValueError, match="coverage"):
            ledger.validate(damaged)
    ledger.open("final_outcome", ("final",))
    with pytest.raises(ValueError, match="coverage"):
        ledger.validate((event,))
    with pytest.raises(ValueError, match="unknown"):
        ledger.open("unregistered_branch", ())


def test_coverage_rejects_wrong_site_outcome_and_foreign_obligation():
    assert hasattr(certificate, "CONTROL_SITES"), "independent registry is missing"
    first, second = certificate._CoverageLedger(), certificate._CoverageLedger()
    obligation = first.open("termination", (0,))
    with pytest.raises(ValueError, match="foreign"):
        second.emit(obligation, "empty_active", ())
    with pytest.raises(ValueError, match="outcome"):
        first.emit(obligation, "invented", ())


def test_compile_surface_is_bound_and_uses_exact_per_voter_class_shares():
    assert hasattr(certificate, "ResidualClassSurface"), "authenticated surface is missing"
    inst = _instance()
    inst.votes.append(Vote("v3", ("q",), age=30, sex="F"))
    bound = _compile(inst, anchor=_anchor(0.5), weights=(0.37, -0.23, 0.41, -0.19))
    surface = certificate.ResidualClassSurface.from_bound(bound)
    assert surface.spec.class_order == ("__missing_demographic__", "age25-39|F", "age60+|M")
    assert surface.spec.multiplicities == (1, 2, 1)
    assert surface.voter_classes == (1, 2, 0, 1)
    assert surface.anchor_logits == (Fraction(0), Fraction(0), Fraction(1, 2))
    assert surface.feature_rows[0] == (0, 0, 0, 0)
    assert surface.initial_balances[0].h == (0, Fraction(1, 2), 0)
    assert surface.initial_balances[0] == surface.initial_balances[3]
    assert surface.initial_balances[2].h == (1, 0, 0)
    assert surface.context_digest == bound.binding.context_digest
    assert surface.base_weights[0] == Fraction.from_float(0.37)
    assert _compile(inst, state=_state(year=5)).binding.context_digest != surface.context_digest
    with pytest.raises(TypeError):
        certificate.ResidualClassSurface()
    class FakeBound(certificate.BoundTrace):
        def require(self, expected):
            pytest.fail("subclass dispatch must never run")
    fake = object.__new__(FakeBound)
    with pytest.raises(TypeError, match="concrete"):
        certificate.ResidualClassSurface.from_bound(fake)


def test_compile_surface_rejects_expired_process_and_context(monkeypatch):
    assert hasattr(certificate, "ResidualClassSurface"), "authenticated surface is missing"
    bound = _compile()
    monkeypatch.setattr(certificate, "_PROCESS_NONCE", "expired")
    with pytest.raises(ValueError, match="process"):
        certificate.ResidualClassSurface.from_bound(bound)


def _symbolic_fixture(costs, approvals, *, budget=12, completion=True, demographics=None, anchor=None):
    inst = PBInstance("fixture://symbolic", {"budget": str(budget)},
                      {pid: Project(pid, cost, None) for pid, cost in costs.items()},
                      [Vote(str(i), tuple(pids), age=(demographics[i][0] if demographics else 30),
                            sex=(demographics[i][1] if demographics else "F"))
                       for i, pids in enumerate(approvals)])
    return _compile(inst, completion=completion, anchor=anchor)


def _segment_trace(*, both=False, cost=5, weights=(0., 0., 0., 0.)):
    inst = PBInstance("fixture://segment", {"budget": "12"},
                      {"a": Project("a", cost, None),
                       **({"b": Project("b", cost, None), "z": Project("z", 0, None)} if both else {})},
                      [Vote("0", ("a", "z") if both else ("a",), age=30, sex="F"),
                       Vote("1", ("b",) if both else (), age=70, sex="M")])
    scaler = FeatureScaler(FEATURE_NAMES, (0., 0., .125, 0.), (1., 1., .375, 1.),
                           clip=3., row_count=2, instance_count=1) if both else _scaler()
    result = certificate.compile_signed_trace(_compile(inst, weights=weights, scaler=scaler, completion=False))
    assert type(result) is certificate.CompiledTrace, getattr(result, "reason", "")
    return result


def _segment(trace, target, box=((-2., 2.),) * 4):
    assert hasattr(certificate, "certify_real_segment"), "directional proof is missing"
    return certificate.certify_real_segment(trace, target, box)


def test_segment_strict_interior_records_every_variable_guard_and_scope():
    trace = _segment_trace()
    proof = _segment(trace, (0., -.1, 0., 0.))
    assert proof.status == "proved_real" and proof.reason is None
    assert proof.context_digest == trace.bound.binding.context_digest
    assert not proof.float_runtime_proved and not proof.cache_safe
    assert proof.target_weights == (0, Fraction.from_float(-.1), 0, 0)
    variable_ids = tuple(e.event_id for e in trace.events
                         if e.decision and e.decision.kind == certificate.DecisionKind.VARIABLE)
    assert tuple(g.event_id for g in proof.guards) == variable_ids
    assert all(g.margin.lower > g.movement_upper >= 0 and g.slack_lower > 0
               and g.precision == 80 for g in proof.guards)
    assert proof.limiting_guard == next(e.event_id for e in trace.events if e.site == "charge")
    assert proof.precisions == (80,)
    assert proof.score_change_signs == (-1, 1)
    assert proof.endpoint_changes[0].upper <= 0 <= proof.endpoint_changes[1].lower
    assert Decimal(".13") < proof.envelope_upper < Decimal(".14")
    with pytest.raises(TypeError):
        certificate.RealSegmentProof(status="proved_real", cache_safe=True)
    with pytest.raises(FrozenInstanceError):
        proof.cache_safe = True


def test_segment_same_point_abstains_but_exact_zero_score_direction_is_provable():
    trace = _segment_trace()
    for target in ((0., 0., 0., 0.), (-0., 0., 0., 0.)):
        result = _segment(trace, target)
        assert result.status == "uncertified" and result.reason == "same_point"
    result = _segment(trace, (.1, 0., 0., 0.))
    assert result.status == "proved_real"
    assert result.score_change_signs == (0, 0)
    assert result.envelope_upper == 0 and result.ratio_upper == 0
    assert all(g.movement_upper == 0 for g in result.guards)


def _certificate_boundary():
    # Independent closed expression: 12*tanh(ln(2)*tanh(t)/2) = 1.
    with localcontext() as context:
        context.prec = 100
        x = ((Decimal(13) / 11).ln()) / Decimal(2).ln()
        return ((1 + x) / (1 - x)).ln() / 2


@pytest.mark.parametrize("both", (False, True))
def test_segment_first_conservative_boundary_and_nextafter_neighbors(both):
    trace = _segment_trace(both=both)
    # The selected coordinates give exact scores (-t,+t) in both fixtures.
    coordinate = 2 if both else 1
    assert tuple(row[coordinate] for row in trace.surface.feature_rows) == ((-1, 1) if both else (1, -1))
    root = _certificate_boundary()
    boundary = float(root)
    inside, outside = math.nextafter(boundary, -math.inf), math.nextafter(boundary, math.inf)
    assert Decimal.from_float(inside) < root < Decimal.from_float(outside)
    for value in (inside, boundary, outside):
        target = (0., 0., value, 0.) if both else (0., -value, 0., 0.)
        proof = _segment(trace, target)
        assert proof.status == ("proved_real" if Decimal.from_float(value) < root else "uncertified")
        assert proof.limiting_guard[1] == "charge"
        if both:
            charges = [g for g in proof.guards if g.event_id[1] == "charge"]
            assert len(charges) == 2
            assert all(g.margin.lower <= 1 <= g.margin.upper for g in charges)
            assert charges[0].movement_upper == charges[1].movement_upper
    # A sufficient-bound failure does not assert that the true branch changed.
    with localcontext() as context:
        context.prec = 80
        x = (Decimal(7) / 5).ln() / (2 * Decimal(2).ln())
        true_boundary = ((1 + x) / (1 - x)).ln() / 2
    assert root < true_boundary


@pytest.mark.parametrize("dimension", range(4))
@pytest.mark.parametrize("face", (-.125, .125))
def test_segment_closed_box_faces_and_both_nextafter_sides(dimension, face):
    trace = _segment_trace()
    box = ((-.125, .125),) * 4
    for value in (math.nextafter(face, -math.inf), face, math.nextafter(face, math.inf)):
        target = [0.] * 4
        target[dimension] = value
        result = _segment(trace, target, box)
        expected = "proved_real" if -.125 <= value <= .125 else "uncertified"
        assert result.status == expected
        if expected == "uncertified":
            assert result.reason == "target_outside_box"


@pytest.mark.parametrize("target,box", [
    ((0., 0., 0.), ((-1, 1),) * 4),
    ((0., math.nan, 0., 0.), ((-1, 1),) * 4),
    ((0., math.inf, 0., 0.), ((-1, 1),) * 4),
    ((0., True, 0., 0.), ((-1, 1),) * 4),
    ((0., .1, 0., 0.), ((-1, 1),) * 3),
    ((0., .1, 0., 0.), ((1, -1),) * 4),
    ((0., .1, 0., 0.), ((-1, math.inf),) * 4),
])
def test_segment_rejects_malformed_problem_inputs(target, box):
    trace = _segment_trace()
    assert hasattr(certificate, "certify_real_segment"), "directional proof is missing"
    with pytest.raises((TypeError, ValueError)):
        certificate.certify_real_segment(trace, target, box)


def test_segment_rejects_base_outside_box_and_nonstructural_zero():
    trace = _segment_trace()
    result = _segment(trace, (.1, 0., 0., 0.), ((.05, .2), (-1, 1), (-1, 1), (-1, 1)))
    assert result.status == "uncertified" and result.reason == "base_outside_box"
    zero = certificate.compile_signed_trace(_symbolic_fixture(
        {"p": 5}, [("p",), ("p",)], demographics=((30, "F"), (70, "M"))))
    assert type(zero) is certificate.UncertifiedTrace
    with pytest.raises(TypeError, match="concrete"):
        _segment(zero, (.1, 0., 0., 0.))


@pytest.mark.parametrize("damage", ("coverage", "decision", "surface", "binding", "process"))
def test_segment_rejects_tampered_compiled_context(monkeypatch, damage):
    trace = _segment_trace()
    if damage == "coverage":
        object.__setattr__(trace, "events", trace.events[1:])
    elif damage == "decision":
        event = next(e for e in trace.events if e.site == "charge")
        object.__setattr__(event.decision, "kind", certificate.DecisionKind.STRUCTURAL)
    elif damage == "surface":
        object.__setattr__(trace.surface, "feature_rows", ((0, 0, 0, 0),) * 2)
    elif damage == "binding":
        object.__setattr__(trace.bound.binding, "context_digest", "forged")
    else:
        monkeypatch.setattr(certificate, "_PROCESS_NONCE", "expired")
    assert hasattr(certificate, "certify_real_segment"), "directional proof is missing"
    with pytest.raises((TypeError, ValueError)):
        certificate.certify_real_segment(trace, (0., -.1, 0., 0.), ((-2, 2),) * 4)


def test_segment_rejects_compiled_subclass_before_dispatch():
    class Forged(certificate.CompiledTrace):
        def validate_coverage(self, *args):
            pytest.fail("untrusted override dispatched")
    with pytest.raises(TypeError, match="concrete"):
        _segment(object.__new__(Forged), (0., -.1, 0., 0.))


def test_segment_rejects_kind_tuple_masquerade_without_digest_replacement():
    trace = _segment_trace()
    assert _segment(trace, (0., -.3, 0., 0.)).status == "uncertified"
    decisions = [e.decision for e in trace.events
                 if e.decision and e.decision.kind is certificate.DecisionKind.VARIABLE]
    assert len(decisions) == 3
    for decision in decisions:
        object.__setattr__(decision, "kind", ("decision_kind", "variable"))
    # Preserve the stored seal exactly as in the independent review's exploit.
    with pytest.raises(TypeError, match="decision kind"):
        _segment(trace, (0., -.3, 0., 0.))


@pytest.mark.parametrize("value,masquerade", (
    (certificate.DecisionKind.VARIABLE, ("decision_kind", "variable")),
    (Fraction(1, 2), ("fraction", 1, 2)),
    (Decimal("0.5"), ("decimal", "0.5")),
    (certificate._DecimalInterval(Decimal(0), Decimal(1)),
     ("_DecimalInterval", (("lower", ("decimal", "0")), ("upper", ("decimal", "1"))))),
))
def test_segment_payload_digest_separates_typed_records_from_ordinary_tuples(value, masquerade):
    trace = _segment_trace()
    event = trace.events[0]
    object.__setattr__(event, "payload", (value,))
    typed_digest = certificate._compiled_payload_digest(trace)
    object.__setattr__(event, "payload", (masquerade,))
    assert certificate._compiled_payload_digest(trace) != typed_digest


@pytest.mark.parametrize("field,value", (
    ("kind", "variable"), ("kind", Fraction(1)),
    ("guard", ("AffineForm", ())), ("orientation", True),
    ("margin", ("_DecimalInterval", ())), ("base_result", ("_BaseSignResult", ())),
))
def test_segment_validates_expected_decision_field_types_before_coverage(field, value):
    trace = _segment_trace()
    decision = next(e.decision for e in trace.events if e.site == "charge")
    object.__setattr__(decision, field, value)
    # An unhashable obligation must not be consumed before the typed decision
    # rejection; this distinguishes validation order without replacing code.
    object.__setattr__(trace, "obligations", ([],))
    with pytest.raises(TypeError, match="decision"):
        _segment(trace, (0., -.1, 0., 0.))


def test_segment_all_static_trace_can_still_prove_nonzero_motion():
    trace = _signed(_symbolic_fixture({"z": 0}, [("z",)]))
    assert all(e.decision is None or e.decision.kind is certificate.DecisionKind.STATIC
               for e in trace.events)
    proof = _segment(trace, (.1, 0., 0., 0.))
    assert proof.status == "proved_real" and proof.guards == ()


def test_segment_target_tanh_underflow_abstains():
    result = _segment(_segment_trace(), (0., 1e7, 0., 0.), ((-1e8, 1e8),) * 4)
    assert result.status == "uncertified" and "arithmetic_failure" in result.reason
    assert result.precisions == (80,)


@pytest.mark.parametrize("failure", (Overflow, Underflow, InvalidOperation))
def test_segment_numerical_failure_never_becomes_a_proof(monkeypatch, failure):
    trace = _segment_trace()
    def fail(*args):
        raise failure
    monkeypatch.setattr(certificate._BaseArithmetic, "exp", fail)
    proof = _segment(trace, (0., -.1, 0., 0.))
    assert proof.status == "uncertified" and "arithmetic_failure" in proof.reason
    assert not proof.float_runtime_proved and not proof.cache_safe


def test_segment_strict_evidence_exact_equality_and_float_neighbors():
    assert hasattr(certificate, "_segment_guard_evidence"), "strict guard bound is missing"
    spec = certificate.ClassSpec(("a", "b"), (1, 1), 8)
    oracle = certificate.BaseSignOracle(spec, (0, 0), ((0,), (0,)), (0,))
    decision = certificate._compile_affine_decision(certificate.AffineForm(spec, (1, 0), -3), oracle)
    event = certificate.TraceEvent((0, "charge", ()), "charge", "positive", (), decision)
    # Exact margin = 1, B*osc(h)=8: equality is exactly ratio=1/8.
    for ratio in (math.nextafter(.125, 0.), .125, math.nextafter(.125, math.inf)):
        evidence = certificate._segment_guard_evidence(event, oracle, Decimal.from_float(ratio), 80)
        assert evidence.margin.lower <= 1 <= evidence.margin.upper
        assert (evidence.slack_lower > 0) == (ratio < .125)
        if ratio == .125:
            assert evidence.movement_upper == 1


def test_segment_envelope_exact_signs_and_shared_range():
    assert hasattr(certificate, "_segment_envelope"), "directional envelope is missing"
    spec = certificate.ClassSpec(("a", "b", "c"), (1, 1, 1), 12)
    # Exact dot cancellation retains a direction binary64 arithmetic would lose.
    oracle = certificate.BaseSignOracle(spec, (0, 0, 0),
                                       ((1, 1, -1, 0), (-1, -1, 1, 0), (0, 0, 0, 0)),
                                       (0, 0, 0, 0))
    target = (Fraction(2**53), Fraction(1), Fraction(2**53), Fraction(0))
    changes, signs, envelope, ratio = certificate._segment_envelope(oracle, target, 80)
    assert signs == (1, -1, 0)
    assert changes[0].lower > 0 and changes[1].upper < 0
    assert changes[2].lower == changes[2].upper == 0
    assert Decimal("1.05") < envelope < Decimal("1.06")
    assert Decimal(".25") < ratio < Decimal(".26")


def test_segment_structural_identity_and_constant_guards_allow_motion():
    trace = _signed(_symbolic_fixture({"a": 4, "b": 4}, [("a", "b"), ("a", "b")]))
    assert any(e.decision and e.decision.kind == certificate.DecisionKind.STRUCTURAL
               for e in trace.events)
    proof = _segment(trace, (.1, .1, .1, .1))
    assert proof.status == "proved_real"
    assert all(g.movement_upper == 0 for g in proof.guards)


def test_segment_precision_exhaustion_and_ambient_decimal_independence():
    trace = _segment_trace()
    outside = _segment(trace, (0., -.3, 0., 0.))
    assert outside.status == "uncertified" and outside.reason == "strict_margin_not_proved"
    assert outside.precisions == (80, 160, 320, 640)
    expected = _segment(trace, (0., -.1, 0., 0.))
    with localcontext() as context:
        context.prec, context.Emax = 2, 2
        assert _segment(trace, (0., -.1, 0., 0.)) == expected


def test_segment_exact_charge_boundary_zero_and_both_nextafter_sides():
    # At target score zero the exact class total is 6 and charge margin is 0.
    # A positive base score admits a compiled strict trace. The conservative
    # segment bound fails before this true branch boundary, on all three probes.
    trace = _segment_trace(cost=6, weights=(0., .1, 0., 0.))
    surface = trace.surface
    guard = certificate.AffineForm(surface.spec, (1, 0), -6)
    for value, side in ((math.nextafter(0., -math.inf), -1), (0., None),
                        (math.nextafter(0., math.inf), 1)):
        target = (0., value, 0., 0.)
        proof = _segment(trace, target)
        assert proof.status == "uncertified"
        oracle = certificate.BaseSignOracle(surface.spec, surface.anchor_logits,
                                            surface.feature_rows, target)
        exact_sign = oracle.evaluate(guard)
        assert exact_sign.side == side
        assert exact_sign.precisions == (80, 160, 320, 640)
        if side is None:
            assert exact_sign.reason == "precision_exhausted"


def test_segment_negative_base_guard_uses_compiler_orientation():
    trace = _segment_trace(cost=7)
    decision = next(e.decision for e in trace.events if e.site == "affordability")
    assert decision.orientation == -1
    proof = _segment(trace, (0., -.1, 0., 0.))
    assert proof.status == "proved_real"
    assert all(g.margin.lower > 0 for g in proof.guards)


def _float_audit(trace, target=(0., -.1, 0., 0.), **kwargs):
    assert hasattr(certificate, "audit_float_segment"), "separate floating audit is missing"
    return certificate.audit_float_segment(trace, target, ((-2, 2),) * 4, **kwargs)


@pytest.mark.parametrize("completion", (False, True))
def test_float_audit_encloses_base_and_runs_three_kernel_projected_interiors(completion):
    bound = _symbolic_fixture({"a": 5}, [("a",), ()], completion=completion,
                              demographics=((30, "F"), (70, "M")))
    trace = _signed(bound)
    audit = _float_audit(trace)
    assert audit.status == "sampled_parity" and audit.reason is None
    assert audit.real_proof.status == "proved_real"
    assert audit.completion is completion
    assert audit.base.runtime_sum == 12 and audit.base.mass_defect == 0
    assert audit.base.normalization_recipient == 0
    assert audit.base.normalization_branch == "all_equal"
    assert audit.base.status == "enclosed" and audit.base.l1_error_upper >= 0
    assert all(x.lower <= 6 <= x.upper for x in audit.base.ideal_per_voter)
    assert tuple(p.requested_parameter for p in audit.probes) == (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    assert audit.new_three_path_cases == 3 and audit.mismatch_count == 0
    assert all(p.status == "sampled_parity" and p.three_kernel_match and p.projection_matches
               and p.trace_validated for p in audit.probes)
    assert all(p.covered_interior and p.actual_ray_parameter is not None for p in audit.probes)
    assert not audit.full_trace_identity and not audit.float_runtime_proved and not audit.cache_safe
    assert "clamp" in audit.production_difference
    assert {"payer_order", "breakpoint_order", "candidate_scan_order", "charge_sides"} <= set(audit.omitted_micro_events)
    with pytest.raises(TypeError):
        certificate.FloatSegmentAudit(status="sampled_parity", cache_safe=True)


def test_float_audit_exact_rational_mass_defect_is_not_hidden_by_fsum():
    inst = PBInstance("fixture://mass-defect", {"budget": "1"}, {"z": Project("z", 0, None)},
                      [Vote(str(i), ("z",), age=30, sex="F") for i in range(6)])
    trace = _signed(_compile(inst))
    assert math.fsum(certificate._float_from_bits(x)
                     for x in trace.bound.binding.endowment_bits) == 1
    audit = _float_audit(trace, target=(.1, 0., 0., 0.))
    assert audit.base.mass_defect == Fraction(1, 2**55)
    assert audit.base.runtime_sum == 1 + Fraction(1, 2**55)
    assert audit.base.normalization_recipient == 0
    assert audit.base.normalization_adjustment == Fraction(3, 2**55)
    assert all(Fraction(v.lower) <= Fraction(1, 6) <= Fraction(v.upper) for v in audit.base.ideal_per_voter)
    assert Fraction(audit.base.l1_error_upper) >= sum(abs(x - Fraction(1, 6)) for x in audit.base.runtime_shares)


def test_float_audit_rounded_off_ray_points_are_audit_only():
    trace = _segment_trace()
    audit = _float_audit(trace, target=(.1, -.1, 0., 0.),
                         requested_target=(Fraction.from_float(.1), Fraction.from_float(-.1) + Fraction(1, 2**58), 0, 0))
    assert any(not p.requested_equals_actual for p in audit.probes)
    assert any(p.actual_ray_parameter is None for p in audit.probes)
    assert all(not p.covered_interior for p in audit.probes if p.actual_ray_parameter is None)
    for p in audit.probes:
        assert p.actual_weights == tuple(Fraction.from_float(certificate._float_from_bits(b)) for b in p.actual_weight_bits)


def test_float_audit_adjacent_float_endpoint_has_no_representable_interior():
    trace = _segment_trace(weights=(.1, 0., 0., 0.))
    audit = _float_audit(trace, target=(math.nextafter(.1, math.inf), 0., 0., 0.))
    assert audit.real_proof.status == "proved_real"
    assert audit.covered_interior_count == 0
    assert all(p.actual_weights in (audit.real_proof.base_weights, audit.real_proof.target_weights)
               for p in audit.probes)


def test_float_audit_does_not_promote_an_uncertified_segment():
    audit = _float_audit(_segment_trace(), target=(0., -.3, 0., 0.))
    assert audit.real_proof.status == "uncertified"
    assert audit.covered_interior_count == 0
    assert not audit.float_runtime_proved and not audit.cache_safe


@pytest.mark.parametrize("parameters", ((), (0,), (1,), (Fraction(1, 3),), (Fraction(1, 2), Fraction(1, 2))))
def test_float_audit_rejects_noninterior_nondyadic_or_duplicate_probe_parameters(parameters):
    assert hasattr(certificate, "audit_float_segment"), "separate floating audit is missing"
    with pytest.raises(ValueError, match="probe"):
        _float_audit(_segment_trace(), probe_parameters=parameters)


def test_float_audit_numerical_failure_keeps_missing_l1_and_mismatch_count_unknown(monkeypatch):
    trace = _segment_trace()
    def fail(*args):
        raise Overflow
    monkeypatch.setattr(certificate._BaseArithmetic, "exp", fail)
    audit = _float_audit(trace)
    assert audit.status == "uncertified"
    assert audit.base.l1_error_upper is None
    assert audit.mismatch_count is None and audit.probes == ()


def test_float_audit_detects_three_kernel_mismatch_at_covered_interior(monkeypatch):
    trace = _segment_trace()
    monkeypatch.setattr(certificate.production, "fast_mes_with_endowments", lambda *a, **k: {"wrong"})
    audit = _float_audit(trace)
    assert audit.status == "uncertified" and audit.route_failure
    assert audit.mismatch_count == 1 and audit.new_three_path_cases == 1
    assert audit.probes[0].covered_interior and not audit.probes[0].three_kernel_match


def test_float_audit_nonuniform_residue_and_endpoint_evidence():
    trace = _signed(_symbolic_fixture({"a": 2}, [("a",), ()],
                                     demographics=((30, "F"), (70, "M")), anchor=_anchor(.5)))
    audit = _float_audit(trace, include_endpoint=True)
    assert audit.base.normalization_branch == "log_softmax"
    assert audit.base.normalization_recipient == 1
    assert audit.base.runtime_shares[1] > audit.base.runtime_shares[0]
    assert audit.base_projection == trace.audit
    assert audit.new_three_path_cases == 4
    endpoint = audit.probes[-1]
    assert endpoint.requested_parameter == 1
    assert endpoint.covered_by_real_proof and not endpoint.covered_interior
    assert endpoint.actual_weights == audit.real_proof.target_weights
    assert endpoint.charge_side_observations is not None


def test_float_audit_endpoint_projection_change_is_observed_not_claimed_covered():
    audit = _float_audit(_segment_trace(), target=(0., -1., 0., 0.), include_endpoint=True)
    assert audit.real_proof.status == "uncertified"
    assert audit.status == "sampled_parity" and not audit.route_failure
    assert audit.probes[-1].projection_matches is False
    assert audit.probes[-1].covered_by_real_proof is False


def _p0_runner():
    import importlib.util
    assert importlib.util.find_spec("run_iclr_certificate_p0") is not None, "gated P0 runner is missing"
    import run_iclr_certificate_p0
    return run_iclr_certificate_p0


def test_p0_runner_loads_pinned_prior_gates_without_reexecuting_them():
    runner = _p0_runner()
    evidence = runner.load_prior_gates(Path(__file__).resolve().parents[1])
    assert evidence["prior_three_path_cases"] == 4432
    assert len(evidence["parity_reports"]) == 5
    assert evidence["task5_wesola_modes"] == [False, True]
    assert evidence["reexecuted_prior_gates"] is False


def test_p0_runner_rejects_missing_or_changed_gate_evidence(tmp_path):
    runner = _p0_runner()
    with pytest.raises((OSError, ValueError, RuntimeError)):
        runner.load_prior_gates(tmp_path)
    path = tmp_path / "gate.json"
    path.write_text('{"status":"pass"}')
    with pytest.raises(ValueError, match="digest"):
        runner._verified_bytes(path, "0" * 64)


def test_p0_runner_synthetic_boundary_inventory_executes_both_modes():
    runner = _p0_runner()
    campaign = runner.run_synthetic_campaign()
    assert campaign["gate_passed"]
    assert campaign["completion_modes"] == [False, True]
    assert campaign["box_boundary_probes"] == 48
    assert campaign["conservative_boundary_probes"] == 12
    assert campaign["true_charge_boundary_probes"] == 6
    assert campaign["three_kernel_mismatches"] == 0
    assert campaign["covered_interior_violations"] == 0
    assert campaign["three_path_cases"] > 0
    assert any(case["proof_status"] == "uncertified" for case in campaign["cases"])
    assert any(case["proof_status"] == "proved_real" for case in campaign["cases"])


def test_p0_runner_dyadic_targets_retain_frozen_exact_intent_and_rounding():
    runner = _p0_runner()
    config = runner.wesola_config()
    targets = runner.dyadic_targets(config)
    assert len(targets) == 61
    base = tuple(Fraction.from_float(float.fromhex(x)) for x in config["base_weights_hex"])
    for exponent, point in enumerate(targets):
        assert point["requested"] == tuple(w + Fraction(d, 2**exponent) for w, d in zip(base, (1, -1, 2, -2)))
        assert point["actual"] == tuple(Fraction.from_float(v) for v in point["weights"])
    assert any(p["actual"] != p["requested"] for p in targets)
    assert targets[-1]["actual"] == base


@pytest.mark.parametrize("missing_gate", ("prior", "integration_tests", "synthetic_campaign", "wesola"))
def test_p0_runner_never_writes_a_final_report_with_a_missing_gate(tmp_path, missing_gate):
    runner = _p0_runner()
    gates = {name: True for name in ("prior", "integration_tests", "synthetic_campaign", "wesola")}
    gates[missing_gate] = None
    with pytest.raises(ValueError, match="gate"):
        runner.write_final_report(tmp_path / "report.json", {"gates": gates})
    assert not (tmp_path / "report.json").exists()


def test_p0_runner_report_write_is_append_only_and_preserves_exact_numbers(tmp_path):
    runner = _p0_runner()
    report = {"gates": {name: True for name in ("prior", "integration_tests", "synthetic_campaign", "wesola")},
              "float_runtime_proved": False, "cache_safe": False, "mass": Fraction(1, 2**55),
              "missing": None, "bound": Decimal("1.0000000000000000000000000001")}
    target = tmp_path / "new-analysis" / "report.json"
    runner.write_final_report(target, report)
    original = target.read_bytes()
    import json
    saved = json.loads(original)
    assert saved["mass"] == {"numerator": "1", "denominator": str(2**55)}
    assert saved["missing"] is None and saved["bound"] == str(report["bound"])
    with pytest.raises(FileExistsError):
        runner.write_final_report(target, report)
    assert target.read_bytes() == original


def test_p0_runner_refuses_unbounded_real_launch():
    runner = _p0_runner()
    with pytest.raises(SystemExit):
        runner.main([])


def test_p0_runner_rejects_symlink_ancestor_for_report(tmp_path):
    runner = _p0_runner()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked-analysis"
    link.symlink_to(outside, target_is_directory=True)
    report = {"gates": {name: True for name in ("prior", "integration_tests", "synthetic_campaign", "wesola")},
              "float_runtime_proved": False, "cache_safe": False}
    with pytest.raises((OSError, ValueError)):
        runner.write_final_report(link / "nested" / "report.json", report)
    assert not (outside / "nested" / "report.json").exists()


def test_p0_runner_fixed_direction_backoff_is_testable_without_real_data():
    runner = _p0_runner()
    assert hasattr(runner, "audit_fixed_direction"), "fixed-direction gate is missing"
    config = runner.wesola_config()
    config["base_weights_hex"] = [0.0.hex()] * 4
    result = runner.audit_fixed_direction(_segment_trace(), config)
    assert result["passed"] and result["audit"].covered_interior_count == 3
    assert len(result["trials"]) > 1
    assert result["trials"][-1]["actual_frozen_ray_parameter"] > 0
    assert all(trial["requested"] == tuple(Fraction(d, 2**trial["exponent"]) for d in (1, -1, 2, -2))
               for trial in result["trials"])


def test_p0_runner_rejection_screen_is_one_sided_and_preserves_boundary_sides():
    runner = _p0_runner()
    assert hasattr(runner, "screen_sufficient_bound"), "economical rejection screen is missing"
    trace = _segment_trace()
    root = _certificate_boundary()
    for value in (.1, math.nextafter(float(root), -math.inf),
                  math.nextafter(float(root), math.inf), .3):
        screen = runner.screen_sufficient_bound(trace, (0., -value, 0., 0.))
        assert screen["excluded"] == (Decimal.from_float(value) > root)
        if screen["excluded"]:
            assert screen["margin_upper"] <= screen["movement_lower"]
            assert screen["limiting_guard"][1] == "charge"


def test_p0_runner_screen_skips_full_proofs_only_for_definite_failures(monkeypatch):
    runner = _p0_runner()
    trace = _segment_trace()
    config = runner.wesola_config()
    config["base_weights_hex"] = [0.0.hex()] * 4
    original = certificate.certify_real_segment
    targets = []
    def record(compiled, target, box):
        targets.append(tuple(target))
        return original(compiled, target, box)
    monkeypatch.setattr(certificate, "certify_real_segment", record)
    result = runner.audit_fixed_direction(trace, config)
    assert result["passed"]
    assert len(targets) == 2  # final full proof and its independent audit consumption
    assert targets[0] == targets[1] == result["trials"][-1]["weights"]
    assert all(t["proof"] is None and t["screen"]["excluded"] for t in result["trials"][:-1])


def _ranking_evaluator(runner, trace):
    assert hasattr(runner, "_screen_evaluator"), "invocation-local authenticated ranking reuse is missing"
    return runner._screen_evaluator(trace)


@pytest.mark.parametrize("completion", (False, True))
@pytest.mark.parametrize("both", (False, True))
def test_p0_ranking_full_results_preserve_boundary_sides_and_simultaneous_guards(completion, both):
    runner = _p0_runner()
    trace = runner._synthetic_trace(completion, both=both)
    evaluate = _ranking_evaluator(runner, trace)
    root = _certificate_boundary()
    boundary = float(root)
    values = (0., .1, math.nextafter(boundary, -math.inf), boundary,
              math.nextafter(boundary, math.inf), .3, .125,
              math.nextafter(.125, -math.inf), math.nextafter(.125, math.inf))
    charge_ids = [e.event_id for e in trace.events if e.site == "charge"]
    if both:
        charges = [e for e in trace.events if e.site == "charge"]
        assert len(charges) == 2
        # Two independent projects each charge 5 from initial balance 6.
        assert all(e.decision.margin.lower <= 1 <= e.decision.margin.upper for e in charges)
    for value in values:
        target = (0., 0., value, 0.) if both else (0., -value, 0., 0.)
        result = evaluate(target)
        assert result == runner.screen_sufficient_bound(trace, target)
        assert result["excluded"] is (Decimal.from_float(value) > root)
        # The ideal charge margins coincide, but their outward decimal upper
        # endpoints need not tie exactly; the frozen stateless result decides.
        assert result["limiting_guard"] in charge_ids
        assert Decimal("0.999999999999999") < result["margin_upper"] < Decimal("1.000000000000001")
        if value == 0:
            assert result["movement_lower"] == result["envelope_lower"] == 0


def test_p0_ranking_first_exact_tie_uses_event_order_on_genuine_shared_guards():
    runner = _p0_runner()
    trace = _signed(_symbolic_fixture({"a": 8, "b": 8}, [("a", "b")] * 3 + [()],
                                     demographics=[(30, "F")] * 3 + [(70, "M")]))
    evaluate = _ranking_evaluator(runner, trace)
    result = evaluate((0., 0., 0., 0.))
    assert result == runner.screen_sufficient_bound(trace, (0., 0., 0., 0.))
    chosen = next(e for e in trace.events if e.event_id == result["limiting_guard"])
    identical = [e for e in trace.events if e.decision is chosen.decision]
    assert len(identical) >= 2  # Exact same compiled decision, not near-equal margins.
    assert result["limiting_guard"] == identical[0].event_id
    assert result["excluded"] is False and result["movement_lower"] == 0


def test_p0_ranking_reuses_only_selection_and_repeats_every_genuine_check(monkeypatch):
    runner = _p0_runner()
    trace = runner._synthetic_trace(False)
    evaluate = _ranking_evaluator(runner, trace)
    calls = []
    def observe(owner, name, label):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            calls.append(label)
            return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, wrapped)
    observe(certificate, "_require_bound", "bound")
    observe(certificate, "_require_compiled_decision_types", "types")
    observe(certificate, "_compiled_payload_digest", "seal")
    observe(certificate.CompiledTrace, "validate_coverage", "coverage")
    observe(runner, "_rank_screen_guards", "rank")
    observe(certificate, "_segment_envelope", "envelope")
    check = ["bound", "types", "seal", "coverage", "bound"]
    for value in (.3, .2, .1):
        evaluate((0., -value, 0., 0.))
    assert calls == check + ["rank", "envelope"] + (check + ["envelope"]) * 2
    calls.clear()
    _ranking_evaluator(runner, trace)((0., -.1, 0., 0.))
    assert calls == check + ["rank", "envelope"]
    calls.clear()
    runner.screen_sufficient_bound(trace, (0., -.1, 0., 0.))
    runner.screen_sufficient_bound(trace, (0., -.1, 0., 0.))
    assert calls == (check + ["rank", "envelope"]) * 2


@pytest.mark.parametrize("kind", ("static", "constant_variable"))
def test_p0_ranking_no_candidates_precede_malformed_target_validation(kind, monkeypatch):
    runner = _p0_runner()
    costs = {"z": 0} if kind == "static" else {"a": 4}
    approvals = [("z",)] if kind == "static" else [("a",), ("a",)]
    trace = _signed(_symbolic_fixture(costs, approvals))
    variables = [e.decision for e in trace.events if e.decision and e.decision.kind is certificate.DecisionKind.VARIABLE]
    assert bool(variables) is (kind == "constant_variable")
    assert all(max(d.guard.h) == min(d.guard.h) for d in variables)
    evaluate = _ranking_evaluator(runner, trace)
    ranks = []
    original = runner._rank_screen_guards
    def observe(*args):
        ranks.append(None)
        return original(*args)
    monkeypatch.setattr(runner, "_rank_screen_guards", observe)
    for target in (object(), (float("nan"),), (0.,) * 4):
        assert evaluate(target) == {"excluded": False, "reason": "no_variable_movement"}
    assert len(ranks) == 1
    assert runner.screen_sufficient_bound(trace, object()) == {"excluded": False, "reason": "no_variable_movement"}


def test_p0_ranking_is_independent_of_changed_ambient_decimal_context():
    runner = _p0_runner()
    trace = runner._synthetic_trace(False, both=True)
    targets = [(0., 0., v, 0.) for v in (.1, .3, 0.)]
    expected = [runner.screen_sufficient_bound(trace, target) for target in targets]
    evaluate = _ranking_evaluator(runner, trace)
    for index, (target, want) in enumerate(zip(targets, expected)):
        with localcontext() as context:
            context.prec, context.Emax, context.Emin = 2 + index, 2, -2
            context.rounding = "ROUND_UP" if index % 2 else "ROUND_DOWN"
            assert evaluate(target) == runner.screen_sufficient_bound(trace, target) == want


@pytest.mark.parametrize("damage", ("event", "decision_kind", "coefficients", "margin", "obligations",
                                   "surface", "context", "binding", "pid", "nonce", "resealed"))
def test_p0_ranking_rejects_cross_trial_tampering_before_endpoint_work(monkeypatch, damage):
    runner = _p0_runner()
    trace = runner._synthetic_trace(False)
    other = runner._synthetic_trace(True) if damage == "binding" else None
    evaluate = _ranking_evaluator(runner, trace)
    endpoints = []
    original = certificate._segment_envelope
    def observed(*args, **kwargs):
        endpoints.append(None)
        return original(*args, **kwargs)
    monkeypatch.setattr(certificate, "_segment_envelope", observed)
    assert evaluate((0., -.1, 0., 0.))["excluded"] is False
    event = next(e for e in trace.events if e.site == "charge")
    if damage == "event":
        object.__setattr__(event, "payload", ("changed",))
    elif damage == "decision_kind":
        object.__setattr__(event.decision, "kind", ("decision_kind", "variable"))
    elif damage == "coefficients":
        object.__setattr__(event.decision.guard, "h", (Fraction(2), Fraction(0)))
    elif damage in ("margin", "resealed"):
        object.__setattr__(event.decision, "margin", certificate._DecimalInterval(Decimal(2), Decimal(2)))
        if damage == "resealed":
            object.__setattr__(trace, "payload_digest", certificate._compiled_payload_digest(trace))
    elif damage == "obligations":
        object.__setattr__(trace, "obligations", trace.obligations[:-1])
        object.__setattr__(trace, "payload_digest", certificate._compiled_payload_digest(trace))
    elif damage == "surface":
        object.__setattr__(trace.surface, "feature_rows", ((Fraction(0),) * 4,) * 2)
    elif damage == "context":
        object.__setattr__(trace.bound.binding, "context_digest", "changed")
    elif damage == "binding":
        object.__setattr__(trace, "bound", other.bound)
        object.__setattr__(trace, "payload_digest", certificate._compiled_payload_digest(trace))
    elif damage == "pid":
        object.__setattr__(trace.bound.binding.runtime, "process_id", -1)
    else:
        monkeypatch.setattr(certificate, "_PROCESS_NONCE", "expired")
    reason = {"decision_kind": "concrete DecisionKind", "obligations": "coverage mismatch",
              "context": "binding context digest mismatch", "binding": "identity changed",
              "pid": "process scope mismatch", "nonce": "process scope mismatch",
              "resealed": "identity changed"}.get(damage, "compiled payload mismatch")
    with pytest.raises((TypeError, ValueError), match=reason):
        evaluate((0., -.1, 0., 0.))
    assert len(endpoints) == 1


@pytest.mark.parametrize("kind", ("trace", "event", "decision"))
def test_p0_ranking_rejects_concrete_subclass_substitution(kind):
    runner = _p0_runner()
    trace = runner._synthetic_trace(False)
    if kind == "trace":
        class ForeignTrace(certificate.CompiledTrace):
            pass
        foreign = object.__new__(ForeignTrace)
        with pytest.raises(TypeError, match="concrete CompiledTrace"):
            _ranking_evaluator(runner, foreign)((0., -.1, 0., 0.))
        with pytest.raises(TypeError, match="concrete CompiledTrace"):
            runner.screen_sufficient_bound(foreign, (0., -.1, 0., 0.))
    else:
        evaluate = _ranking_evaluator(runner, trace)
        assert evaluate((0., -.1, 0., 0.))["excluded"] is False
        if kind == "event":
            class ForeignEvent(certificate.TraceEvent):
                pass
            object.__setattr__(trace, "events", (object.__new__(ForeignEvent),) + trace.events[1:])
        else:
            class ForeignDecision(certificate.CompiledDecision):
                pass
            event = next(e for e in trace.events if e.site == "charge")
            object.__setattr__(event, "decision", object.__new__(ForeignDecision))
        with pytest.raises(TypeError, match="concrete"):
            evaluate((0., -.1, 0., 0.))


def test_p0_ranking_failure_discards_evaluator_but_arithmetic_abstention_is_not_cached():
    runner = _p0_runner()
    trace = runner._synthetic_trace(False)
    evaluate = _ranking_evaluator(runner, trace)
    target = (0., 1e7, 0., 0.)  # Genuine Decimal exponential underflow, no injected oracle.
    result = evaluate(target)
    assert result == runner.screen_sufficient_bound(trace, target)
    assert result == {"excluded": False, "reason": "screen_arithmetic_failure:Underflow"}
    assert evaluate((0., -.1, 0., 0.))["reason"] == "sufficient_bound_test_only"
    malformed = (0., 0., 0.)
    with pytest.raises(ValueError) as standalone:
        runner.screen_sufficient_bound(trace, malformed)
    with pytest.raises(ValueError) as cached:
        evaluate(malformed)
    assert str(cached.value) == str(standalone.value)
    with pytest.raises(ValueError, match="closed"):
        evaluate((0., -.1, 0., 0.))
    assert _ranking_evaluator(runner, trace)((0., -.1, 0., 0.))["excluded"] is False


@pytest.mark.parametrize("completion", (False, True))
def test_p0_ranking_audit_owns_one_selection_and_keeps_real_fallthrough(tmp_path, monkeypatch, completion):
    runner = _p0_runner()
    trace = runner._synthetic_trace(completion)
    _ranking_evaluator(runner, trace)  # RED gate before the new private API exists.
    config = runner.wesola_config()
    config["base_weights_hex"] = [0.0.hex()] * 4
    ranks = []
    original = runner._rank_screen_guards
    def observed(*args):
        ranks.append(None)
        return original(*args)
    monkeypatch.setattr(runner, "_rank_screen_guards", observed)
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    try:
        result = runner.audit_fixed_direction(trace, config, diagnostic=run)
        assert result["passed"] and len(ranks) == 1
        assert len(result["trials"]) > 1
        assert [t["exponent"] for t in result["trials"]] == list(range(len(result["trials"])))
        for t in result["trials"]:
            assert t["requested"] == tuple(Fraction(d, 2**t["exponent"]) for d in (1, -1, 2, -2))
            assert t["actual"] == tuple(Fraction.from_float(w) for w in t["weights"])
        assert all(t["screen"]["excluded"] and t["proof"] is None for t in result["trials"][:-1])
        assert result["trials"][-1]["proof"].status == "proved_real"
        assert not result["audit"].float_runtime_proved and not result["audit"].cache_safe
        records = _receipt_records(run.path)
        screens = [r for r in records if r["event"] == "stage_end" and r["stage"] == "screen"]
        assert [r["exponent"] for r in screens] == list(range(len(result["trials"])))
        assert all(r["completion"] is completion for r in screens)
        assert sum(r["event"] == "stage_end" and r["stage"] == "float_audit" for r in records) == 1
        # A separate invocation does not reuse even the identical trace's selection.
        assert runner.audit_fixed_direction(trace, config)["passed"]
        assert len(ranks) == 2
    finally:
        run.close()


def _tiny_integration_root(tmp_path):
    # Six tiny surrogate modules exercise the controller without executing the
    # real integration inventory or any historical/data-dependent test body.
    names = ("test_iclr_actuation_certificate.py", "test_iclr_trace_parity.py",
             "test_iclr_residual_actuation.py", "test_iclr_priority_mes.py",
             "test_iclr_residual_policy.py", "test_iclr_tie_breaking.py")
    (tmp_path / "tests").mkdir()
    for name in names:
        content = "def test_required():\n    assert 2 + 2 == 4\n"
        if name == names[0]:
            content += ("import pytest\n@pytest.mark.parametrize('completion', (False, True))\n"
                        "def test_wesola_frozen_nonuniform_symbolic_prerequisite(completion):\n"
                        "    raise AssertionError('historical gate must never run')\n")
        if name == names[1]:
            for test in ("test_five_repaired_warsaw_sentinels_have_zero_mismatches_in_both_modes",
                         "test_low_median_high_temporal_scored_files_have_zero_mismatches",
                         "test_all_223_temporal_scored_elections_have_zero_mismatches",
                         "test_all_397_native_elections_have_zero_uniform_mismatches",
                         "test_six_frozen_fits_have_zero_split_correct_mismatches"):
                content += f"def {test}():\n    raise AssertionError('historical gate must never run')\n"
        (tmp_path / "tests" / name).write_text(content)
    return names


def test_p0_integration_ignores_ambient_selection_and_binds_six_module_inventory(tmp_path, monkeypatch):
    runner = _p0_runner()
    names = _tiny_integration_root(tmp_path)
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -k impossible\npython_functions = never_*\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--deselect=tests/test_iclr_residual_policy.py::test_required")
    monkeypatch.setenv("PYTEST_PLUGINS", "nonexistent_selection_plugin")
    monkeypatch.setenv("ICLR_TRACE_PARITY_STAGE", "full")
    monkeypatch.setenv("TEMPOPB_RUN_WESOLA_CERTIFICATE", "1")
    assert hasattr(runner, "_run_current_integration_tests"), "six-module inventory gate is missing"
    result = runner._run_current_integration_tests(tmp_path)
    expected = [f"tests/{name}::test_required" for name in names]
    assert result["inventory"] == expected and result["passed"] == 6
    assert len(result["excluded"]) == 7
    assert result["selected_inventory_sha256"] == result["executed_inventory_sha256"]
    assert result["selection_environment"]["PYTEST_ADDOPTS"] is None
    assert result["selection_environment"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert set(result["test_sha256"]) == set(names)


@pytest.mark.parametrize("damage", ("missing", "duplicate", "unexpected", "skipped", "missing_teardown",
                                  "inventory_missing", "inventory_duplicate", "inventory_unexpected",
                                  "failed_phase", "xfail"))
def test_p0_integration_rejects_incomplete_or_changed_executed_identities(tmp_path, damage):
    runner = _p0_runner()
    expected = ("tests/test_one.py::test_a", "tests/test_two.py::test_b[x.y]")
    actual = list(expected)
    if damage == "missing":
        actual.pop()
    elif damage == "duplicate":
        actual[-1] = actual[0]
    elif damage == "unexpected":
        actual[-1] = "tests/test_two.py::test_unexpected"
    import xml.etree.ElementTree as ET
    suite = ET.Element("testsuite")
    for node in actual:
        case = ET.SubElement(suite, "testcase", classname="unused", name="unused")
        props = ET.SubElement(case, "properties")
        ET.SubElement(props, "property", name="p0_nodeid", value=node)
        if damage == "skipped":
            ET.SubElement(case, "skipped")
    path = tmp_path / "integration.xml"
    ET.ElementTree(suite).write(path)
    # XML corruption must be rejected even when independent execution records
    # are complete and successful; inventory corruption is tested separately.
    inventory = list(expected)
    reports = [{"nodeid": node, "when": phase, "outcome": "passed", "xfail": None}
               for node in expected for phase in ("setup", "call", "teardown")]
    if damage == "missing_teardown":
        reports.pop()
    elif damage == "inventory_missing":
        inventory.pop()
    elif damage == "inventory_duplicate":
        inventory[-1] = inventory[0]
    elif damage == "inventory_unexpected":
        inventory[-1] = "tests/test_two.py::test_unexpected"
    elif damage == "failed_phase":
        reports[1]["outcome"] = "failed"
    elif damage == "xfail":
        reports[1]["xfail"] = "unexpected pass cannot satisfy the gate"
    assert hasattr(runner, "_validate_integration_execution"), "exact executed inventory check is missing"
    with pytest.raises(ValueError, match="integration"):
        runner._validate_integration_execution(expected, {"inventory": inventory, "reports": reports}, path)


@pytest.mark.parametrize("damage", ("missing_exclusion", "missing_module", "duplicate"))
def test_p0_integration_rejects_missing_exclusion_or_required_module(damage):
    runner = _p0_runner()
    assert hasattr(runner, "_required_integration_inventory"), "independent required inventory is missing"
    nodes = [f"tests/{name}::test_required" for name in runner._TEST_FILES] + list(runner._INTEGRATION_EXCLUSIONS)
    if damage == "missing_exclusion":
        nodes.pop()
    elif damage == "missing_module":
        nodes.remove("tests/test_iclr_residual_policy.py::test_required")
    else:
        nodes.append(nodes[0])
    with pytest.raises(ValueError, match="integration"):
        runner._required_integration_inventory(nodes)


def _receipt_records(path):
    import hashlib
    import json
    records, previous = [], None
    for file in sorted(path.glob("record-*.json")):
        raw = file.read_bytes()
        record = json.loads(raw)
        assert record["sequence"] == len(records) + 1
        assert record["previous_sha256"] == previous
        assert record["diagnostic_only"] is True
        assert record["float_runtime_proved"] is record["cache_safe"] is False
        previous = hashlib.sha256(raw).hexdigest()
        records.append(record)
    return records


def test_p0_receipts_exclusive_paths_and_hash_chain(tmp_path, monkeypatch):
    runner = _p0_runner()
    monkeypatch.setattr(runner.secrets, "token_hex", lambda n: "b" * (2 * n))
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    try:
        run.identify({"fixture": "synthetic"}, runner.wesola_config())
        result = run.call("synthetic", lambda: 7, lambda value: {"count": value})
        assert result == 7
        records = _receipt_records(run.path)
        assert records[-1]["summary"] == {"count": 7}
        assert records[-1]["resources"]["wall_seconds"] >= 0
        old = (run.path / "record-000001.json").read_bytes()
        with pytest.raises(FileExistsError):
            runner._DiagnosticRun(tmp_path / "diagnostics")
        with pytest.raises(FileExistsError):
            runner._publish_at(run.directory, "record-000001.json", b"replacement")
        assert (run.path / "record-000001.json").read_bytes() == old
    finally:
        run.close()


def test_p0_receipts_reject_symlink_directory_and_artifact(tmp_path):
    runner = _p0_runner()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        runner._DiagnosticRun(linked / "diagnostics")
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    try:
        (run.path / "artifact.json").symlink_to(outside / "target")
        with pytest.raises(FileExistsError):
            runner._publish_at(run.directory, "artifact.json", b"new")
        assert not (outside / "target").exists()
    finally:
        run.close()


@pytest.mark.parametrize("failure", ("after_integration", "synthetic", "trial"))
def test_p0_receipts_timeout_retains_raw_integration_and_current_component(tmp_path, monkeypatch, failure):
    import hashlib
    runner = _p0_runner()
    _tiny_integration_root(tmp_path)
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "src" / "runner.py"))
    monkeypatch.setattr(runner, "_identities", lambda root: {"fixture": "synthetic"})
    monkeypatch.setattr(runner, "load_prior_gates", lambda root: {"prior_three_path_cases": 0})
    original_integration = runner._run_current_integration_tests
    # Keep child imports on the real source directory despite the isolated main root.
    source = str(Path(certificate.__file__).parent)
    monkeypatch.setattr(runner, "_integration_source_directory", lambda: source)
    def integration(root, diagnostic=None):
        result = original_integration(root, diagnostic=diagnostic)
        if failure == "after_integration":
            raise TimeoutError("synthetic interruption after child validation")
        return result
    monkeypatch.setattr(runner, "_run_current_integration_tests", integration)
    def synthetic():
        if failure == "synthetic":
            raise TimeoutError("injected synthetic timeout")
        return {"gate_passed": True, "cases": [], "three_path_cases": 0}
    monkeypatch.setattr(runner, "run_synthetic_campaign", synthetic)
    def trial(root, diagnostic=None):
        token = diagnostic.begin("trial", completion=False, exponent=0)
        diagnostic.call("screen", lambda: (_ for _ in ()).throw(TimeoutError("trial timeout")),
                        lambda value: value, completion=False, exponent=0)
        diagnostic.end(token, {})
    monkeypatch.setattr(runner, "run_wesola_campaign", trial)
    with pytest.raises(TimeoutError):
        runner.main(["--timeout-seconds", "10"])
    area = tmp_path / "analysis-output" / "iclr_certificate_p0"
    paths = list((area / "diagnostics").iterdir())
    assert len(paths) == 1 and not (area / "report.json").exists()
    records = _receipt_records(paths[0])
    terminal = records[-1]
    assert terminal["event"] == "terminal_error" and terminal["error_category"] == "timeout"
    expected = {"after_integration": "integration", "synthetic": "synthetic", "trial": "screen"}[failure]
    assert terminal["current_component"]["stage"] == expected
    assert terminal["summary"] is None
    assert terminal["resources"]["component_wall_seconds"] is None
    artifacts = [artifact for r in records if r["event"] == "integration_artifacts"
                 for artifact in r["artifacts"].values() if artifact is not None]
    assert any(a["name"].endswith("junit.xml") for a in artifacts)
    assert any(a["name"].endswith("execute.json") for a in artifacts)
    for artifact in artifacts:
        assert hashlib.sha256((paths[0] / artifact["name"]).read_bytes()).hexdigest() == artifact["sha256"]


@pytest.mark.parametrize("damage", ("missing", "truncated"))
def test_p0_receipts_invalid_integration_keeps_raw_but_never_completes_gate(tmp_path, monkeypatch, damage):
    from xml.etree.ElementTree import ParseError
    runner = _p0_runner()
    _tiny_integration_root(tmp_path)
    original = runner._validate_integration_execution
    def corrupt(required, execution, junit):
        if damage == "missing":
            junit.unlink()
        else:
            junit.write_bytes(b"<testsuite>")
        original(required, execution, junit)
    monkeypatch.setattr(runner, "_validate_integration_execution", corrupt)
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    try:
        with pytest.raises(FileNotFoundError if damage == "missing" else ParseError):
            run.call("integration", lambda: runner._run_current_integration_tests(tmp_path, diagnostic=run),
                     lambda value: {"passed": value["passed"]})
        records = _receipt_records(run.path)
        retained = next(r for r in records if r["event"] == "integration_artifacts" and r["mode"] == "execute")
        if damage == "missing":
            assert retained["artifacts"]["junit.xml"] is None
        else:
            assert (run.path / retained["artifacts"]["junit.xml"]["name"]).read_bytes() == b"<testsuite>"
        assert retained["artifacts"]["execute.json"] is not None
        assert not any(r["event"] == "stage_end" and r["stage"] == "integration" for r in records)
    finally:
        run.close()


def test_p0_receipts_publication_failure_aborts_and_preserves_old_records(tmp_path, monkeypatch):
    runner = _p0_runner()
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    original = runner._publish_at
    def fail(directory, name, payload):
        if name == "record-000003.json":
            raise OSError("injected publication failure")
        return original(directory, name, payload)
    monkeypatch.setattr(runner, "_publish_at", fail)
    try:
        with pytest.raises(OSError):
            run.call("synthetic", lambda: 1, lambda value: {"count": value})
        assert len(_receipt_records(run.path)) == 2
        assert run.current["stage"] == "synthetic"
        assert run.last_completed is None
        assert not (tmp_path / "report.json").exists()
    finally:
        run.close()


def test_p0_receipts_raw_publication_failure_prevents_validated_gate_end(tmp_path, monkeypatch):
    runner = _p0_runner()
    _tiny_integration_root(tmp_path)
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    original = runner._publish_at
    def fail(directory, name, payload):
        if name.endswith("-execute-execute.json"):
            raise OSError("raw execution evidence cannot be committed")
        return original(directory, name, payload)
    monkeypatch.setattr(runner, "_publish_at", fail)
    try:
        with pytest.raises(OSError, match="raw execution evidence"):
            run.call("integration", lambda: runner._run_current_integration_tests(tmp_path, diagnostic=run),
                     lambda value: {"passed": value["passed"]})
        records = _receipt_records(run.path)
        assert any(r["event"] == "integration_artifacts" and r["mode"] == "collect" for r in records)
        assert not any(r["event"] == "stage_end" and r["stage"] in ("integration_execute", "integration") for r in records)
    finally:
        run.close()


@pytest.mark.parametrize("publication_fails", (False, True))
def test_p0_receipts_synthetic_main_flow_requires_evidence_before_final(tmp_path, monkeypatch, publication_fails):
    import json
    runner = _p0_runner()
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "src" / "runner.py"))
    monkeypatch.setattr(runner, "_identities", lambda root: {"fixture": "synthetic-main-only"})
    monkeypatch.setattr(runner, "load_prior_gates", lambda root: {"fixture": "stub-no-historical-read"})
    monkeypatch.setattr(runner, "_run_current_integration_tests", lambda root, diagnostic=None: {"passed": 1})
    monkeypatch.setattr(runner, "run_synthetic_campaign", lambda: {"gate_passed": True, "cases": []})
    # Real-campaign-shaped return from two genuine tiny synthetic audits only.
    modes = [{"audit": certificate.audit_float_segment(runner._synthetic_trace(mode),
                  (0., -.1, 0., 0.), ((-2, 2),) * 4), "trials": []} for mode in (False, True)]
    monkeypatch.setattr(runner, "run_wesola_campaign", lambda root, diagnostic=None: {"gate_passed": True, "modes": modes})
    original = runner._publish_at
    def publish(directory, name, payload):
        if publication_fails and name.startswith("record-") and json.loads(payload)["event"] == "ready_for_final_publication":
            raise OSError("injected mandatory evidence publication failure")
        return original(directory, name, payload)
    monkeypatch.setattr(runner, "_publish_at", publish)
    if publication_fails:
        with pytest.raises(OSError, match="mandatory evidence"):
            runner.main(["--timeout-seconds", "10"])
    else:
        assert runner.main(["--timeout-seconds", "10"]) == 0
    area = tmp_path / "analysis-output/iclr_certificate_p0"
    records = _receipt_records(next((area / "diagnostics").iterdir()))
    assert (area / "report.json").exists() is not publication_fails
    if publication_fails:
        assert records[-1]["event"] == "terminal_error"
    else:
        report = json.loads((area / "report.json").read_bytes())
        assert all(report["gates"].values())
        assert report["float_runtime_proved"] is report["cache_safe"] is False
        assert records[-1]["event"] == "ready_for_final_publication"
        assert report["diagnostic_evidence"]["records"] == len(records)


def test_p0_receipts_real_trial_observations_keep_exact_targets(tmp_path):
    runner = _p0_runner()
    trace = runner._synthetic_trace(False)
    config = runner.wesola_config()
    config["base_weights_hex"] = [0.0.hex()] * 4
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    try:
        result = runner.audit_fixed_direction(trace, config, diagnostic=run)
        assert result["passed"]
        records = _receipt_records(run.path)
        starts = [r for r in records if r["event"] == "stage_start"]
        assert {r["stage"] for r in starts} >= {"trial", "screen", "segment_proof", "float_audit"}
        trials = [r for r in starts if r["stage"] == "trial"]
        assert len(trials) == len(result["trials"])
        for observed, actual in zip(trials, result["trials"]):
            assert observed["target_bits"] == list(actual["weights_hex"])
            assert observed["requested_target"] == runner._json_value(actual["requested"])
            assert observed["completion"] is False
    finally:
        run.close()


@pytest.mark.parametrize("case", ("no_probe", "partial_unknown", "complete", "short_known", "known_mismatch", "final_mismatch"))
def test_p0_receipts_float_audit_preserves_unknown_aggregate_evidence(tmp_path, monkeypatch, case):
    runner = _p0_runner()
    trace = runner._synthetic_trace(False)
    with monkeypatch.context() as failures:
        if case == "no_probe":
            def overflow(*args):
                raise Overflow
            failures.setattr(certificate._BaseArithmetic, "exp", overflow)
        elif case == "partial_unknown":
            original = certificate.compile_bound_trace
            calls = []
            def fail_second(*args, **kwargs):
                calls.append(None)
                if len(calls) == 2:
                    raise ValueError("synthetic unverified second probe")
                return original(*args, **kwargs)
            failures.setattr(certificate, "compile_bound_trace", fail_second)
        elif case == "known_mismatch":
            failures.setattr(certificate.production, "fast_mes_with_endowments", lambda *args, **kwargs: {"wrong"})
        elif case == "final_mismatch":
            original_projection = certificate._runtime_outcome_projection
            calls = []
            def fail_endpoint(bound_trace):
                calls.append(None)
                result = original_projection(bound_trace)
                return ("synthetic changed endpoint",) if len(calls) == 5 else result
            failures.setattr(certificate, "_runtime_outcome_projection", fail_endpoint)
        kwargs = {"probe_parameters": (Fraction(1, 4),)} if case == "short_known" else {}
        audit = certificate.audit_float_segment(trace, (0., -.1, 0., 0.), ((-2, 2),) * 4,
                                                include_endpoint=True, **kwargs)
    expected_probes = {"no_probe": 0, "partial_unknown": 2, "complete": 4, "short_known": 2, "known_mismatch": 1, "final_mismatch": 4}[case]
    expected_verified = {"no_probe": 0, "partial_unknown": 1, "complete": 4, "short_known": 2, "known_mismatch": 0, "final_mismatch": 4}[case]
    assert len(audit.probes) == expected_probes
    # Feed the genuine failed/partial audit into the real receipt-producing path;
    # no modified proof or successful fabricated audit is used.
    monkeypatch.setattr(certificate, "audit_float_segment", lambda *args, **kwargs: audit)
    config = runner.wesola_config()
    config["base_weights_hex"] = [0.0.hex()] * 4
    run = runner._DiagnosticRun(tmp_path / "diagnostics")
    try:
        if audit.route_failure:
            with pytest.raises(ValueError, match="violation"):
                runner.audit_fixed_direction(trace, config, diagnostic=run)
        else:
            runner.audit_fixed_direction(trace, config, diagnostic=run)
        records = _receipt_records(run.path)
        summary = next(r["summary"] for r in records if r["event"] == "stage_end" and r["stage"] == "float_audit")
        assert summary["mismatch_count"] == audit.mismatch_count
        complete = case in ("complete", "final_mismatch")
        assert summary["sample_evidence_complete"] is complete
        assert summary["sample_parity_verified"] is (case == "complete")
        observed = summary["observed_sample_counts"]
        assert observed["probe_records"] == expected_probes
        assert observed["trace_validated_probes"] == expected_verified
        assert observed["three_path_cases"] == audit.new_three_path_cases
        assert observed["known_three_kernel_mismatches"] == (1 if case == "known_mismatch" else 0)
        for field in ("three_kernel_mismatches", "covered_interior_violations", "covered_endpoint_violations",
                      "audit_only_projection_changes"):
            expected = int(case == "final_mismatch" and field == "covered_endpoint_violations") if complete else None
            assert summary[field] == expected
        if case in ("no_probe", "partial_unknown"):
            assert summary["mismatch_count"] is None
        assert summary["covered_interiors"] == (3 if complete else None)
    finally:
        run.close()


def _legacy_compiled_payload_bytes(trace):
    """Test-only pre-streaming reference; deliberately constructs the old tree."""
    import json
    from dataclasses import fields
    records = (certificate.ResidualClassSurface, certificate.ClassSpec, certificate.AffineForm,
               certificate.TraceEvent, certificate.CompiledDecision, certificate._BaseSignResult,
               certificate._DecimalInterval, certificate.RuntimeAuditProjection)

    def encode(value):
        kind = type(value)
        if value is None:
            return ("none",)
        if kind in (str, int, bool):
            return (kind.__name__, value)
        if kind is Fraction:
            return ("fraction", value.numerator, value.denominator)
        if kind is Decimal:
            return ("decimal", str(value))
        if kind is certificate.DecisionKind:
            return ("decision_kind", value.value)
        if kind is tuple:
            return ("tuple", tuple(encode(item) for item in value))
        if kind in records:
            return ("record", kind.__name__, tuple((f.name, encode(getattr(value, f.name)))
                                                   for f in fields(value)))
        raise TypeError("compiled payload requires concrete proof records")

    payload = (trace.bound.binding.context_digest,
               tuple((f.name, encode(getattr(trace, f.name))) for f in fields(certificate.CompiledTrace)
                     if f.name not in ("bound", "payload_digest")))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


@pytest.mark.parametrize("completion", (False, True))
@pytest.mark.parametrize("payload_kind", ("genuine", "primitives", "shared_records"))
def test_streaming_payload_exact_legacy_bytes_and_digest(completion, payload_kind):
    # Catches lost type tags, escaping, ordering or repeated shared subtrees.
    import hashlib
    trace = _p0_runner()._synthetic_trace(completion)
    if payload_kind == "primitives":
        text = "a" * 1023 + '\\雪\n"\t\b\f\r\x00🙂\u2028\u2029' + "z" * 1024 + '"\\終'
        payload = (None, "", text, True, False, 0, -123, 2**200,
                   Fraction(-7, 13), Decimal("-0.00"), Decimal("1E-999"),
                   Decimal("Infinity"), Decimal("NaN42"), *tuple(certificate.DecisionKind),
                   (), ("record", ("decision_kind", "variable")))
        object.__setattr__(trace.events[0], "payload", payload)
    elif payload_kind == "shared_records":
        shared = (trace.surface.spec, trace.surface.initial_balances[0], trace.audit)
        object.__setattr__(trace.events[0], "payload", (shared, shared, tuple(reversed(shared))))
    expected = _legacy_compiled_payload_bytes(trace)
    chunks = tuple(certificate._compiled_payload_chunks(trace))
    assert chunks and all(type(chunk) is bytes and len(chunk) <= 8192 for chunk in chunks)
    assert b"".join(chunks) == expected
    assert certificate._compiled_payload_digest(trace) == hashlib.sha256(expected).hexdigest()
    if payload_kind == "genuine":
        assert certificate._compiled_payload_digest(trace) == trace.payload_digest


@pytest.mark.parametrize("shape", ("long_string", "shared_graph"))
def test_streaming_payload_hash_avoids_full_size_temporary_buffers(shape):
    # Catches rebuilding the legacy tagged tree/string/UTF8 payload inside digest.
    import hashlib
    import tracemalloc
    trace = _segment_trace()
    if shape == "long_string":
        object.__setattr__(trace.audit, "production_difference", "x" * (1024 * 1024))
    else:
        # Serialization fixture only: duplicated obligation IDs never enter a
        # proof/coverage consumer. Detect expansion of repeated shared records.
        object.__setattr__(trace, "events", trace.events * 100)
        object.__setattr__(trace, "obligations", trace.obligations * 100)
    expected = hashlib.sha256(_legacy_compiled_payload_bytes(trace)).hexdigest()
    assert not tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        actual = certificate._compiled_payload_digest(trace)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == expected
    assert peak < 256 * 1024, f"{shape} required {peak} temporary Python bytes"


def test_streaming_payload_preserves_unicode_encoding_failure():
    trace = _segment_trace()
    object.__setattr__(trace.events[0], "payload", ("x" * 1024 + "\ud800",))
    with pytest.raises(UnicodeEncodeError):
        _legacy_compiled_payload_bytes(trace)
    with pytest.raises(UnicodeEncodeError):
        certificate._compiled_payload_digest(trace)


@pytest.mark.parametrize("kind", ("str", "int", "tuple", "fraction", "decimal", "record", "unsupported"))
def test_streaming_payload_rejects_concrete_subclass_overrides(kind):
    class DerivedString(str):
        pass
    class DerivedInt(int):
        pass
    class DerivedTuple(tuple):
        pass
    class DerivedFraction(Fraction):
        pass
    class DerivedDecimal(Decimal):
        pass
    class DerivedForm(certificate.AffineForm):
        pass
    trace = _segment_trace()
    values = {"str": DerivedString("x"), "int": DerivedInt(1), "tuple": DerivedTuple((1,)),
              "fraction": DerivedFraction(1, 2), "decimal": DerivedDecimal(".5"),
              "record": DerivedForm(trace.surface.spec, (0,) * len(trace.surface.spec.class_order)),
              "unsupported": 1.0}
    object.__setattr__(trace.events[0], "payload", (values[kind],))
    with pytest.raises(TypeError, match="concrete proof records"):
        certificate._compiled_payload_digest(trace)


@pytest.mark.parametrize("mutation", ("shared_spec", "nested_guard"))
def test_streaming_payload_detects_cross_call_nested_shared_record_tampering(mutation):
    trace = _segment_trace()
    original = certificate._compiled_payload_digest(trace)
    assert original == trace.payload_digest
    if mutation == "shared_spec":
        assert trace.surface.initial_balances[0].spec is trace.surface.spec
        object.__setattr__(trace.surface.spec, "budget", trace.surface.spec.budget + 1)
    else:
        guard = next(e.decision.guard for e in trace.events
                     if e.decision is not None and e.decision.kind is certificate.DecisionKind.VARIABLE)
        object.__setattr__(guard, "beta", guard.beta + 1)
    assert certificate._compiled_payload_digest(trace) != original
    with pytest.raises(ValueError, match="payload context mismatch"):
        certificate.certify_real_segment(trace, (0., -.01, 0., 0.), ((-2, 2),) * 4)


def _signed(bound):
    assert hasattr(certificate, "compile_signed_trace"), "symbolic replay is missing"
    result = certificate.compile_signed_trace(bound)
    assert isinstance(result, certificate.CompiledTrace), getattr(result, "reason", "not compiled")
    result.validate_coverage()
    _assert_all_event_mutations_fail(result)
    return result


def test_compile_two_sided_water_fill_preserves_divergent_voter_histories():
    trace = _signed(_symbolic_fixture({"a": 2, "b": 7}, [("a", "b"), ("b",), ("b",)], completion=False))
    assert trace.payment_order == ("a", "b")
    assert tuple(b.evaluate((12,)) for b in trace.final_balances) == (0, Fraction(3, 2), Fraction(3, 2))
    assert trace.histories[0] != trace.histories[1]
    assert trace.histories[1] == trace.histories[2]
    assert {e.outcome for e in trace.events if e.site == "breakpoint"} == {"negative", "positive"}
    assert {e.outcome for e in trace.events if e.site == "charge"} == {"negative", "positive"}
    assert trace.audit.payment_order_matches and trace.audit.final_winners_match
    assert not trace.float_runtime_proved and not trace.cache_safe


@pytest.mark.parametrize("completion", (False, True))
def test_compile_tolerance_sliver_and_empty_active_termination(completion):
    trace = _signed(_symbolic_fixture({"p": 1 + 5e-10}, [("p",)], budget=1, completion=completion))
    assert trace.payment_order == ()
    assert [(e.site, e.outcome) for e in trace.events if e.site in ("removal", "termination")] == [
        ("removal", "no_rho"), ("termination", "no_candidate")]
    empty = _signed(_symbolic_fixture({"z": 0, "n": 2}, [("z",)], budget=1, completion=completion))
    assert [e.outcome for e in empty.events if e.site == "termination"] == ["empty_active"]


@pytest.mark.parametrize("completion", (False, True))
def test_compile_structural_rho_cost_id_ties_and_complete_completion_decisions(completion):
    tied = _signed(_symbolic_fixture({"a": 4, "b": 4, "c": 8}, [("a", "b", "c"), ("c",)], budget=10, completion=completion))
    assert tied.payment_order == ("a",)
    assert any(e.site == "rho_order" and e.decision.kind == certificate.DecisionKind.STRUCTURAL for e in tied.events)
    assert any(e.site == "candidate_cost" for e in tied.events)
    assert any(e.site == "candidate_id" for e in tied.events)
    trace = _signed(_symbolic_fixture({"a": 6, "b": 4, "c": 7, "z": 0, "zz": 0, "n": 1},
                                      [("a", "b", "z", "zz"), ("c", "z", "zz")], budget=10, completion=completion))
    assert trace.payment_order == ("b",)
    assert trace.winners == (("a", "b") if completion else ("b",))
    if completion:
        assert {e.outcome for e in trace.events if e.site == "completion_decision"} == {"fund", "zero_cost", "unaffordable"}
        assert {e.outcome for e in trace.events if e.site == "completion_skip"} == {"winner", "zero_support"}
        assert {e.site for e in trace.events} >= {"completion_count", "completion_cost", "completion_id"}


def test_compile_nonstructural_base_tie_abstains():
    assert hasattr(certificate, "compile_signed_trace"), "symbolic replay is missing"
    bound = _symbolic_fixture({"p": 5}, [("p",), ("p",)], demographics=((30, "F"), (70, "M")))
    result = certificate.compile_signed_trace(bound)
    assert isinstance(result, certificate.UncertifiedTrace)
    assert "payer_order" in result.reason and "precision_exhausted" in result.reason


def test_coverage_every_event_deletion_duplication_or_id_substitution_fails():
    _signed(_symbolic_fixture({"a": 2, "b": 7, "z": 0}, [("a", "b", "z"), ("b",), ("b",)]))


def _assert_all_event_mutations_fail(trace):
    for i, event in enumerate(trace.events):
        for damaged in (trace.events[:i] + trace.events[i + 1:],
                        trace.events[:i] + (event,) + trace.events[i:],
                        trace.events[:i] + (replace(event, event_id=(99999, event.site, ())),) + trace.events[i + 1:]):
            with pytest.raises(ValueError, match="coverage"):
                trace.validate_coverage(damaged)


def test_coverage_registry_checks_kind_orientation_and_literal_outcomes():
    comparison_sites = {"initial_support", "initial_cost", "affordability", "payer_order",
                        "breakpoint", "rho_order", "candidate_cost", "candidate_id", "charge",
                        "completion_count", "completion_cost", "completion_id", "completion_positive",
                        "completion_affordability"}
    assert {s.name: s.outcomes for s in certificate.CONTROL_SITES if s.name not in comparison_sites} == {
        "initial_active": ("snapshot",), "active_before": ("snapshot",), "candidate_visit": ("visit",),
        "removal": ("unaffordable", "no_rho"), "incumbent": ("initialize", "retain", "replace"),
        "chosen": ("payment",), "active_after": ("snapshot",),
        "termination": ("empty_active", "no_candidate"), "completion_mode": ("enabled", "disabled"),
        "completion_visit": ("visit",), "completion_skip": ("winner", "zero_support"),
        "completion_remaining": ("initial", "transition"),
        "completion_decision": ("fund", "zero_cost", "unaffordable"), "final_outcome": ("winners",),
    }
    ledger = certificate._CoverageLedger()
    spec = _class_spec()
    decision = certificate._compile_affine_decision(certificate.AffineForm(spec, (0, 0), 1), _base_oracle(spec))
    obligation = ledger.open("affordability", ())
    with pytest.raises(ValueError, match="orientation"):
        ledger.emit(obligation, "zero", (), decision)


def test_compile_records_each_active_removal_transition():
    trace = _signed(_symbolic_fixture({"a": 20, "b": 2}, [("a", "b")], budget=10))
    removal = next(e for e in trace.events if e.site == "removal")
    index = trace.events.index(removal)
    assert trace.events[index - 1].site == "active_before"
    assert trace.events[index - 1].payload == ("a", "b")
    assert trace.events[index + 1].site == "active_after"
    assert trace.events[index + 1].payload == ("b",)


def test_compile_static_accumulation_disagreement_keeps_full_bound_context():
    bound = _symbolic_fixture({"a": 0.1, "b": 0.2, "c": 0.7}, [("a",), ("b",), ("c",)], budget=1)
    result = certificate.compile_signed_trace(bound)
    assert isinstance(result, certificate.UncertifiedTrace)
    assert "static_runtime_disagreement:completion_affordability" in result.reason
    assert result.context_digest == bound.binding.context_digest


def test_compile_static_candidate_tie_requires_observed_runtime_resolution():
    bound = _symbolic_fixture({"a": 0.15, "b": 0.5, "c": 0.15},
                              [("a", "b"), ("a", "b"), ("a", "b"), ("b",), ("c",), ()], budget=1)
    result = certificate.compile_signed_trace(bound)
    assert isinstance(result, certificate.UncertifiedTrace)
    assert "static_runtime_disagreement:candidate_cost" in result.reason


def test_compile_remaining_branch_sides_and_audit_comparison_inventory():
    trace = _signed(_symbolic_fixture({"a": 6, "b": 2, "c": 8}, [("a", "b", "c")], budget=12))
    assert trace.payment_order == ("b", "a")
    assert {e.outcome for e in trace.events if e.site == "incumbent"} == {"initialize", "replace", "retain"}
    assert {e.outcome for e in trace.events if e.site == "rho_order"} == {"negative", "positive"}
    exhausted = _signed(_symbolic_fixture({"p": 1}, [("p",)], budget=1))
    assert [e.outcome for e in exhausted.events if e.site == "charge"] == ["zero"]
    for demographics, side in [(((30, "F"), (70, "M")), "positive"), (((70, "M"), (30, "F")), "negative")]:
        ordered = _signed(_symbolic_fixture({"p": 2}, [("p",), ("p",)],
                                           demographics=demographics, anchor=_anchor(0.5)))
        assert [e.outcome for e in ordered.events if e.site == "payer_order"] == [side]
    assert trace.audit.compared_events == ("payment_winner_sequence", "removed_project_sets_by_round",
                                           "completion_decisions_in_order", "final_winners", "matching_charge_sides",
                                           "static_comparison_resolutions")
    assert any(row[0] == "candidate_rho_order" for row in trace.audit.omitted_micro_events)


def test_coverage_literal_complete_single_project_stream():
    trace = _signed(_symbolic_fixture({"p": 1}, [("p",)], budget=1))
    assert [(e.site, e.outcome) for e in trace.events] == [
        ("initial_support", "positive"), ("initial_cost", "positive"),
        ("initial_active", "snapshot"), ("active_before", "snapshot"),
        ("candidate_visit", "visit"), ("affordability", "positive"),
        ("breakpoint", "positive"), ("incumbent", "initialize"),
        ("active_before", "snapshot"), ("chosen", "payment"), ("charge", "zero"),
        ("active_after", "snapshot"), ("termination", "empty_active"),
        ("completion_mode", "enabled"), ("completion_remaining", "initial"),
        ("completion_visit", "visit"), ("completion_skip", "winner"),
        ("final_outcome", "winners"),
    ]


# Literal driver sites, not extracted from decisions or an observed trace.
@pytest.mark.parametrize("helper, site", [
    ("static", "initial_support"), ("static", "initial_cost"),
    ("affine", "affordability"), ("affine", "payer_order"),
    ("affine", "breakpoint"), ("affine", "rho_order"),
    ("static", "candidate_cost"), ("static", "candidate_id"),
    ("affine", "charge"), ("static", "completion_count"),
    ("static", "completion_cost"), ("static", "completion_id"),
    ("static", "completion_positive"), ("static", "completion_affordability"),
    ("event", "initial_active"), ("event", "active_before"),
    ("event", "candidate_visit"), ("event", "removal"),
    ("event", "incumbent"), ("event", "chosen"),
    ("event", "active_after"), ("event", "termination"),
    ("event", "completion_mode"), ("event", "completion_visit"),
    ("event", "completion_skip"), ("event", "completion_remaining"),
    ("event", "completion_decision"), ("event", "final_outcome"),
])
def test_coverage_driver_obligations_survive_skipped_helpers(monkeypatch, helper, site):
    if site in ("candidate_cost", "candidate_id"):
        bound = _symbolic_fixture({"a": 4, "b": 4, "c": 8}, [("a", "b", "c"), ("c",)], budget=10)
    else:
        bound = _symbolic_fixture({"a": 2, "b": 7, "n": 20, "u": 1, "z": 0, "zz": 0},
                                  [("a", "b", "z", "zz"), ("b", "n"), ("b",)])
    original = getattr(certificate._SymbolicReplay, helper)
    skipped = []
    finalized = []
    original_validate = certificate._CoverageLedger.validate

    def skip_emission(self, *args, **kwargs):
        # Supports the old site argument and the corrected preopened obligation;
        # the RED test must run against the unfixed implementation as well.
        emitted_site = args[0] if isinstance(args[0], str) else args[0].event_id[1]
        if emitted_site != site:
            return original(self, *args, **kwargs)
        skipped.append(emitted_site)
        if helper == "event":
            return None
        if helper == "affine":
            # Continue the real exact arithmetic branch while dropping only its
            # coverage emission. No caller-supplied sign or floating shortcut.
            decision = certificate._compile_affine_decision(args[-1], self.oracle)
            assert decision.orientation is not None
            return decision.orientation
        left, right = args[-2:]
        return (left > right) - (left < right)

    def observe_finalization(self, events):
        finalized.append(True)
        return original_validate(self, events)

    monkeypatch.setattr(certificate._SymbolicReplay, helper, skip_emission)
    monkeypatch.setattr(certificate._CoverageLedger, "validate", observe_finalization)
    with pytest.raises(ValueError, match="coverage"):
        certificate.compile_signed_trace(bound)
    assert skipped, "fixture must visit the literal target site"
    assert finalized, "control flow must continue to coverage finalization"


def test_coverage_static_domain_sites_exclude_impossible_negative_outcomes():
    assert {s.name: s.outcomes for s in certificate.CONTROL_SITES
            if s.name in ("initial_support", "initial_cost", "completion_positive")} == {
        "initial_support": ("zero", "positive"), "initial_cost": ("zero", "positive"),
        "completion_positive": ("zero", "positive"),
    }


def test_compile_structural_affordability_and_breakpoint_equalities():
    affordable = _signed(_symbolic_fixture({"p": 2e-9}, [("p",)], budget=1e-9, completion=False))
    assert [e.outcome for e in affordable.events if e.site == "affordability"] == ["zero"]
    assert any(e.site == "removal" and e.outcome == "no_rho" for e in affordable.events)
    breakpoint = _signed(_symbolic_fixture({"p": 2e-12}, [("p",)], budget=1e-12, completion=False))
    assert [e.outcome for e in breakpoint.events if e.site == "breakpoint"] == ["zero"]
    assert breakpoint.payment_order == ("p",)


# Frozen before either real-data compiler prerequisite; do not edit to obtain a pass.
_WESOLA_SENTINEL_JSON = '{"anchor":{"family":"age_sex","free_logits_hex":["-0x1.6666666666666p-1","-0x1.0000000000000p-1","-0x1.3333333333333p-2","-0x1.999999999999ap-4","0x1.999999999999ap-4","0x1.3333333333333p-2","0x1.0000000000000p-1"],"reference_cell":"age<25|F"},"backoff_exponent_range":[0,60],"base_weights_hex":["0x1.7ae147ae147aep-2","-0x1.d70a3d70a3d71p-3","0x1.a3d70a3d70a3dp-2","-0x1.851eb851eb852p-3"],"completion_modes":[false,true],"direction":[1,-1,2,-2],"input_sha256":"823c6b7489bd52d37d996ad6421ed8bb8eb1334575509eff50bdaaae1a31fbe3","parameter_box":[[-10,10],[-10,10],[-10,10],[-10,10]],"precision_schedule":[80,160,320,640],"scaler_sha256":"78033a6272654b43c25a8fa7cf448700c02cb111e5c02b073561dbda81b872ff"}'


def _wesola_config():
    import hashlib
    import json
    assert hashlib.sha256(_WESOLA_SENTINEL_JSON.encode()).hexdigest() == "e83cb4222c0a6d676fb8bfa7a99930b8264adfad7a25c843a3698ffc2994dec3"
    config = json.loads(_WESOLA_SENTINEL_JSON)
    assert json.dumps(config, sort_keys=True, separators=(",", ":")) == _WESOLA_SENTINEL_JSON
    return config


def test_wesola_sentinel_config_and_input_are_frozen():
    import hashlib
    config = _wesola_config()
    path = Path(__file__).resolve().parents[1] / "data/pb/Poland_Warszawa_2020_Wesola.pb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["input_sha256"]


@pytest.mark.skipif(os.environ.get("TEMPOPB_RUN_WESOLA_CERTIFICATE") != "1",
                    reason="real-data prerequisite requires an explicit power/cost-approved run")
@pytest.mark.parametrize("completion", (False, True))
def test_wesola_frozen_nonuniform_symbolic_prerequisite(completion):
    from iclr_residual_policy import fit_context_scaler
    from parse_pb import parse_pb_file
    import hashlib
    config = _wesola_config()
    path = Path(__file__).resolve().parents[1] / "data/pb/Poland_Warszawa_2020_Wesola.pb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["input_sha256"]
    inst = parse_pb_file(path)
    assert (len(inst.votes), len(inst.projects)) == (1505, 21)
    scaler = fit_context_scaler([inst])
    assert scaler.sha256 == config["scaler_sha256"]
    anchor = AnchorSpec("age_sex", tuple(float.fromhex(v) for v in config["anchor"]["free_logits_hex"]), "age<25|F")
    bound = _compile(inst, anchor=anchor, scaler=scaler,
                     weights=tuple(float.fromhex(v) for v in config["base_weights_hex"]),
                     state=RolloutState(deficits={}, entitlements={}, year_index=0),
                     completion=completion, input_paths={"wesola": path})
    assert isinstance(bound, certificate.BoundTrace), getattr(bound, "reason", "unbound")
    trace = certificate.compile_signed_trace(bound)
    assert isinstance(trace, certificate.CompiledTrace), getattr(trace, "reason", "not compiled")
    trace.validate_coverage()
    assert len(trace.surface.spec.class_order) == 9
    missing = trace.surface.spec.class_order.index("__missing_demographic__")
    assert trace.surface.spec.multiplicities[missing] == 3
    assert len(set(trace.surface.anchor_logits)) > 1
    assert trace.audit.payment_order_matches and trace.audit.removed_project_sets_match
    assert trace.audit.completion_decisions_match and trace.audit.final_winners_match
    assert not trace.float_runtime_proved and not trace.cache_safe


def _instance(
    *,
    path: str = "fixture://shared-path",
    budget: str = "12",
    project_order: tuple[str, ...] = ("p", "q"),
    vote_order: tuple[int, ...] = (0, 1, 2),
    first_approvals: tuple[str, ...] = ("p", "q"),
) -> PBInstance:
    projects = {
        pid: Project(pid, {"p": 5.0, "q": 7.0}[pid], None, name=pid.upper())
        for pid in project_order
    }
    votes = (
        Vote("v0", first_approvals, age=30, sex="F", neighborhood="north"),
        Vote("v1", ("q",), age=70, sex="M", neighborhood="south"),
        Vote("v2", ("p",), age=None, sex="", neighborhood=""),
    )
    return PBInstance(
        path=path,
        meta={"budget": budget, "country": "X", "instance": "tiny_2025"},
        projects=projects,
        votes=[votes[index] for index in vote_order],
    )


def _anchor(logit: float = 0.0) -> AnchorSpec:
    return AnchorSpec("senior", (logit,), "age<25")


def _scaler(*, mean0: float = 0.0) -> FeatureScaler:
    return FeatureScaler(
        FEATURE_NAMES,
        (mean0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        clip=3.0,
        row_count=3,
        instance_count=1,
    )


def _state(*, deficit: float = 2.0, year: int = 4) -> RolloutState:
    return RolloutState(
        deficits={"age25-39|F": deficit},
        entitlements={"age25-39|F": 10.0},
        year_index=year,
    )


def _compile(
    inst: PBInstance | None = None,
    *,
    anchor: AnchorSpec | None = None,
    scaler: FeatureScaler | None = None,
    weights: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    state: RolloutState | None = None,
    completion: bool = True,
    input_paths: dict[str, Path] | None = None,
):
    return certificate.compile_bound_trace(
        inst or _instance(),
        anchor or _anchor(),
        scaler or _scaler(),
        weights,
        state or _state(),
        completion=completion,
        input_paths=input_paths,
    )


def test_binding_is_immutable_factory_only_and_rejects_a_mismatched_trace() -> None:
    first = _compile()
    second = _compile(_instance(budget="13"))

    assert isinstance(first, certificate.BoundTrace)
    assert first.require(first.binding) is first.trace
    with pytest.raises(ValueError, match="trace context mismatch"):
        first.require(second.binding)
    with pytest.raises(FrozenInstanceError):
        first.binding.context_digest = "0" * 64
    for cls in (
        certificate.NumericRuntime,
        certificate.TraceBinding,
        certificate.BoundTrace,
    ):
        with pytest.raises(TypeError, match="public compiler"):
            cls()


def _require_in_fork(bound: certificate.BoundTrace) -> str:
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            bound.require(bound.binding)
        except Exception as exc:  # The payload is asserted by the parent process.
            payload = f"{type(exc).__name__}:{exc}"
        else:
            payload = "accepted"
        os.write(write_fd, payload.encode("utf-8"))
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    try:
        payload = os.read(read_fd, 4096).decode("utf-8")
    finally:
        os.close(read_fd)
        waited_pid, status = os.waitpid(pid, 0)
    assert waited_pid == pid
    assert os.waitstatus_to_exitcode(status) == 0
    return payload


def test_binding_rejects_cross_process_consumption_after_fork() -> None:
    bound = _compile()

    observed = _require_in_fork(bound)

    assert observed == "ValueError:trace process scope mismatch"


def test_binding_rejects_consumption_when_process_nonce_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = _compile()
    monkeypatch.setattr(certificate, "_PROCESS_NONCE", "f" * 64)

    with pytest.raises(ValueError, match="trace process scope mismatch"):
        bound.require(bound.binding)


@pytest.mark.parametrize(
    "mutation",
    (
        "project-order",
        "vote-order",
        "approval-order",
        "cost",
        "budget",
        "state-value",
        "state-year",
        "anchor",
        "scaler",
        "base-weight",
        "completion",
    ),
)
def test_binding_digest_changes_for_every_solver_significant_mutation(
    mutation: str,
) -> None:
    baseline = _compile()
    kwargs: dict[str, object] = {}
    inst = _instance()
    if mutation == "project-order":
        inst = _instance(project_order=("q", "p"))
    elif mutation == "vote-order":
        inst = _instance(vote_order=(1, 0, 2))
    elif mutation == "approval-order":
        inst = _instance(first_approvals=("q", "p"))
    elif mutation == "cost":
        inst.projects["p"] = Project("p", 5.5, None, name="P")
    elif mutation == "budget":
        inst = _instance(budget="13")
    elif mutation == "state-value":
        kwargs["state"] = _state(deficit=math.nextafter(2.0, math.inf))
    elif mutation == "state-year":
        kwargs["state"] = _state(year=5)
    elif mutation == "anchor":
        kwargs["anchor"] = _anchor(math.nextafter(0.0, math.inf))
    elif mutation == "scaler":
        kwargs["scaler"] = _scaler(mean0=math.nextafter(0.0, math.inf))
    elif mutation == "base-weight":
        kwargs["weights"] = (math.nextafter(0.0, math.inf), 0.0, 0.0, 0.0)
    elif mutation == "completion":
        kwargs["completion"] = False

    changed = _compile(inst, **kwargs)

    assert isinstance(changed, certificate.BoundTrace)
    assert changed.binding.context_digest != baseline.binding.context_digest


def test_binding_instance_identity_excludes_path_but_includes_full_content() -> None:
    first = _compile(_instance(path="fixture://one"))
    second = _compile(_instance(path="fixture://two"))
    changed = _instance(path="fixture://one")
    changed.meta["extra"] = "bound metadata"
    third = _compile(changed)

    assert first.binding.instance_semantics_sha256 == second.binding.instance_semantics_sha256
    assert first.binding.context_digest == second.binding.context_digest
    assert third.binding.instance_semantics_sha256 != first.binding.instance_semantics_sha256
    assert first.binding.instance_path == (
        "certificate://v1/" + first.binding.instance_semantics_sha256
    )


def test_binding_warmed_path_caches_cannot_cross_contaminate_frozen_content() -> None:
    first = PBInstance(
        path="fixture://cache-collision",
        meta={"budget": "5"},
        projects={"p": Project("p", 5.0, None), "q": Project("q", 5.0, None)},
        votes=[Vote("v", ("p",), age=30, sex="F")],
    )
    second = PBInstance(
        path="fixture://cache-collision",
        meta={"budget": "5"},
        projects={"p": Project("p", 5.0, None), "q": Project("q", 5.0, None)},
        votes=[Vote("v", ("q",), age=30, sex="F")],
    )
    instance_features(first, scheme="age_sex")
    fast_mes_with_endowments(first, (5.0,), completion=True)

    first_bound = _compile(first)
    second_bound = _compile(second)

    assert first_bound.trace.winners == ("p",)
    assert second_bound.trace.winners == ("q",)
    assert first_bound.binding.instance_path != second_bound.binding.instance_path


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda inst: inst.meta.__setitem__("budget", "nan"), "budget"),
        (lambda inst: inst.meta.__setitem__("budget", "0"), "budget"),
        (lambda inst: inst.meta.__setitem__("budget", "-1"), "budget"),
        (lambda inst: inst.projects.__setitem__("p", Project("p", -1.0, None)), "cost"),
        (lambda inst: inst.projects.__setitem__("p", Project("p", float("nan"), None)), "cost"),
        (lambda inst: inst.projects.__setitem__("p", Project("p", float("inf"), None)), "cost"),
        (
            lambda inst: inst.votes.__setitem__(
                0, Vote("v0", ("missing",), age=30, sex="F")
            ),
            "unknown project",
        ),
        (
            lambda inst: inst.votes.__setitem__(
                0, Vote("v0", ("p", "p"), age=30, sex="F")
            ),
            "duplicate project approval",
        ),
        (lambda inst: inst.votes.clear(), "at least one voter"),
    ),
)
def test_binding_rejects_malformed_instance_before_solver_execution(
    monkeypatch: pytest.MonkeyPatch, mutate, message: str
) -> None:
    inst = _instance()
    mutate(inst)
    monkeypatch.setattr(
        certificate.actuation,
        "trace_equal_shares",
        lambda *args, **kwargs: pytest.fail("solver ran before validation"),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _compile(inst)


def test_binding_rejects_project_mapping_key_identity_disagreement() -> None:
    inst = _instance()
    inst.projects["p"] = Project("not-p", 5.0, None)

    with pytest.raises(ValueError, match="mapping key.*project ID"):
        _compile(inst)


@pytest.mark.parametrize(
    "partition",
    (
        (("age25-39|F", (0,)), ("__missing_demographic__", (2,))),
        (("age25-39|F", (0, 1)), ("age60+|M", (1, 2))),
        (("age25-39|F", (0,)), ("age60+|M", (1,)), ("empty", ())),
    ),
    ids=("incomplete", "overlap", "zero-count"),
)
def test_binding_rejects_invalid_internally_derived_partition(
    monkeypatch: pytest.MonkeyPatch, partition
) -> None:
    monkeypatch.setattr(
        certificate,
        "_derive_partition_snapshot",
        lambda votes: partition,
    )

    with pytest.raises(ValueError, match="partition"):
        _compile()


def test_normalization_rejects_a_one_ulp_error_without_epsilon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = PBInstance(
        path="fixture://ulp",
        meta={"budget": "1"},
        projects={"p": Project("p", 1.0, None)},
        votes=[Vote("v", ("p",), age=30, sex="F")],
    )
    monkeypatch.setattr(
        certificate.residual,
        "residual_policy",
        lambda *args: lambda frozen_inst, state: [math.nextafter(1.0, 0.0)],
    )

    result = _compile(inst)

    assert isinstance(result, certificate.UncertifiedTrace)
    assert result.certified is False
    assert "math.fsum" in result.reason
    assert result.float_runtime_proved is False
    assert result.cache_safe is False


def test_normalization_uses_fsum_for_the_cancellation_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inst = PBInstance(
        path="fixture://cancellation",
        meta={"budget": "1e16"},
        projects={"p": Project("p", 1.0, None)},
        votes=[
            Vote("small-a", ("p",), age=30, sex="F"),
            Vote("large", ("p",), age=30, sex="F"),
            Vote("small-b", ("p",), age=30, sex="F"),
        ],
    )
    # On CPython 3.12, this ordering still defeats ordinary ``sum`` while
    # ``math.fsum`` exposes the one-ULP excess over the declared budget.
    sentinel = [1e-16, 1.0, 1e16]
    assert sum(sentinel) == inst.budget
    assert math.fsum(sentinel) != inst.budget
    monkeypatch.setattr(
        certificate.residual,
        "residual_policy",
        lambda *args: lambda frozen_inst, state: sentinel,
    )

    result = _compile(inst)

    assert isinstance(result, certificate.UncertifiedTrace)
    assert "math.fsum" in result.reason


def test_normalization_returns_uncertified_for_unrepresentable_subnormal_budget() -> None:
    inst = PBInstance(
        path="fixture://subnormal-budget",
        meta={"budget": "5e-324"},
        projects={"p": Project("p", 5e-324, None)},
        votes=[
            Vote("v0", ("p",), age=30, sex="F"),
            Vote("v1", ("p",), age=40, sex="M"),
            Vote("v2", ("p",), age=70, sex="F"),
        ],
    )

    result = _compile(inst)

    assert isinstance(result, certificate.UncertifiedTrace)
    assert result.certified is False
    assert "not representable" in result.reason
    assert result.endowment_bits == ()
    assert result.float_runtime_proved is False
    assert result.cache_safe is False


def test_normalization_keeps_unrelated_policy_value_errors_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unrelated_policy(*args):
        del args

        def run(inst, state):
            del inst, state
            raise ValueError("unrelated policy failure")

        return run

    monkeypatch.setattr(certificate.residual, "residual_policy", unrelated_policy)

    with pytest.raises(ValueError, match="unrelated policy failure"):
        _compile()


def test_binding_records_exact_bits_partition_trace_runtime_and_false_flags() -> None:
    result = _compile()

    assert isinstance(result, certificate.BoundTrace)
    binding = result.binding
    assert binding.budget_bits == "4028000000000000"
    assert binding.endowment_bits == (
        "4010000000000000",
        "4010000000000000",
        "4010000000000000",
    )
    assert binding.class_partition == (
        ("__missing_demographic__", (2,)),
        ("age25-39|F", (0,)),
        ("age60+|M", (1,)),
    )
    assert binding.trace_payload
    assert binding.residual_policy_state_semantics == "ignored-but-bound"
    assert binding.scope == "process"
    assert binding.float_runtime_proved is False
    assert binding.cache_safe is False
    assert binding.source_hashes_are_execution_proof is False
    assert binding.runtime.process_id > 0
    assert len(dict(binding.source_role_sha256)) == 9
    assert len(binding.project_id_sha256) == 64


def test_binding_snapshots_numpy_base_weights() -> None:
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    result = _compile(weights=weights)

    assert isinstance(result, certificate.BoundTrace)
    assert result.binding.base_weight_bits == ("0000000000000000",) * 4


def test_binding_digest_changes_with_schema_solver_python_numpy_and_endowments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = _compile()

    monkeypatch.setattr(certificate, "BINDING_SCHEMA", "trace-context-test-mutation")
    schema = _compile()
    monkeypatch.setattr(certificate, "BINDING_SCHEMA", baseline.binding.schema)

    copied_solver = tmp_path / "rules.py"
    copied_solver.write_bytes(Path(certificate.rules.__file__).read_bytes() + b"\n# mutation\n")
    source_paths = dict(certificate._SOURCE_ROLE_PATHS)
    source_paths["reference_solver"] = copied_solver
    monkeypatch.setattr(certificate, "_SOURCE_ROLE_PATHS", tuple(source_paths.items()))
    solver = _compile()
    monkeypatch.setattr(certificate, "_SOURCE_ROLE_PATHS", certificate._DEFAULT_SOURCE_ROLE_PATHS)

    monkeypatch.setattr(certificate.sys, "version", certificate.sys.version + "-mutated")
    python = _compile()
    monkeypatch.undo()
    monkeypatch.setattr(certificate.np, "__version__", np.__version__ + "-mutated")
    numpy_runtime = _compile()
    monkeypatch.undo()

    original_policy = certificate.residual.residual_policy
    monkeypatch.setattr(
        certificate.residual,
        "residual_policy",
        lambda *args: lambda frozen_inst, state: [5.0, 4.0, 3.0],
    )
    endowments = _compile()
    monkeypatch.setattr(certificate.residual, "residual_policy", original_policy)

    observed = {
        schema.binding.context_digest,
        solver.binding.context_digest,
        python.binding.context_digest,
        numpy_runtime.binding.context_digest,
        endowments.binding.context_digest,
    }
    assert baseline.binding.context_digest not in observed
    assert len(observed) == 5


def test_binding_snapshots_optional_input_content_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    data = tmp_path / "fit.json"
    data.write_bytes(b'{"weight":1}\n')
    first = _compile(input_paths={"fit": data})
    data.write_bytes(b'{"weight":2}\n')
    second = _compile(input_paths={"fit": data})
    linked = tmp_path / "linked.json"
    linked.symlink_to(data)

    assert first.binding.context_digest != second.binding.context_digest
    assert dict(first.binding.input_file_sha256)["fit"] != dict(
        second.binding.input_file_sha256
    )["fit"]
    with pytest.raises(RuntimeError, match="regular nonsymlink file"):
        _compile(input_paths={"fit": linked})


@pytest.mark.parametrize("invalid", ("anchor", "scaler", "weights", "weight-shape"))
def test_binding_rejects_nonfinite_model_inputs_before_compilation(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    anchor = _anchor()
    scaler = _scaler()
    weights = (0.0, 0.0, 0.0, 0.0)
    if invalid == "anchor":
        object.__setattr__(anchor, "free_logits", (float("nan"),))
    elif invalid == "scaler":
        object.__setattr__(scaler, "mean", np.asarray((float("nan"), 0.0, 0.0, 0.0)))
    elif invalid == "weights":
        weights = (float("nan"), 0.0, 0.0, 0.0)
    else:
        weights = (0.0, 0.0, 0.0)
    monkeypatch.setattr(
        certificate,
        "_snapshot_source_roles",
        lambda: pytest.fail("source snapshot ran before model validation"),
    )

    with pytest.raises(ValueError, match="finite|base weights"):
        _compile(anchor=anchor, scaler=scaler, weights=weights)


def _class_spec(order=("a", "b"), counts=(1, 1), budget=12):
    return certificate.ClassSpec(order, counts, budget)


def test_affine_lifts_binary_values_without_decimal_approximation() -> None:
    spec = _class_spec(budget=0.1)
    form = certificate.AffineForm(spec, (1e-9, 1e-12), 0.1)
    assert spec.budget == Fraction(3602879701896397, 36028797018963968)
    assert form.h == (Fraction.from_float(1e-9), Fraction.from_float(1e-12))
    assert form.beta != Fraction(1, 10)
    assert certificate._exact_fraction(2**80 + 1) == 2**80 + 1
    assert certificate._exact_fraction(Fraction(2, 3)) == Fraction(2, 3)


def test_affine_add_subtract_scale_divide_and_evaluate_are_exact() -> None:
    spec = _class_spec()
    a = certificate.AffineForm(spec, (1, 2), 3)
    b = certificate.AffineForm(spec, (4, -1), Fraction(1, 2))
    assert (a + b).evaluate((5, 7)) == Fraction(71, 2)
    assert (a - b).evaluate((5, 7)) == Fraction(17, 2)
    assert (Fraction(2, 3) * a).evaluate((5, 7)) == Fraction(44, 3)
    assert (a / 2).evaluate((5, 7)) == 11
    assert (-a).evaluate((5, 7)) == -22
    assert (a - a).is_structural
    with pytest.raises(FrozenInstanceError):
        a.beta = 4


@pytest.mark.parametrize("payer_count", (0, -1, True, 2.0, Fraction(2, 1)))
def test_affine_division_requires_an_exact_positive_integer_payer_count(payer_count):
    form = certificate.AffineForm(_class_spec(), (1, 0), 0)
    with pytest.raises((TypeError, ValueError), match="payer count"):
        form / payer_count


@pytest.mark.parametrize(
    "other_spec",
    ((('b', 'a'), (1, 1), 12), (('a',), (1,), 12),
     (('a', 'b'), (1, 1), 13), (('a', 'b'), (2, 1), 12)),
)
def test_affine_rejects_operations_across_distinct_domains(other_spec):
    a = certificate.AffineForm(_class_spec(), (1, 0), 0)
    spec = certificate.ClassSpec(*other_spec)
    b = certificate.AffineForm(spec, (0,) * len(spec.class_order), 0)
    for operation in (lambda: a + b, lambda: a - b):
        with pytest.raises(ValueError, match="domain"):
            operation()


@pytest.mark.parametrize(
    "order,counts,budget,h,beta,canonical_h,canonical_beta,structural",
    [
        (("a", "b"), (1, 1), 12, (2, 2), -24, (0, 0), 0, True),
        (("a", "b"), (1, 1), 12, (1, -1), 0, (2, 0), -12, False),
        (("a", "b"), (1, 1), 12, (2, 2), -23, (0, 0), 1, False),
        (("only",), (3,), 12, (2,), -24, (0,), 0, True),
        (("only",), (3,), 12, (2,), -23, (0,), 1, False),
        (("a", "b", "c"), (1, 1, 1), 12, (1, 1, -2), 0,
         (3, 3, 0), -24, False),
    ],
)
def test_structural_identity_requires_exact_simplex_canonicalization(
    order, counts, budget, h, beta, canonical_h, canonical_beta, structural,
):
    form = certificate.AffineForm(certificate.ClassSpec(order, counts, budget), h, beta)
    canonical = form.canonical()
    assert canonical.h == tuple(Fraction(v) for v in canonical_h)
    assert canonical.beta == canonical_beta
    assert canonical.is_structural is structural
    assert form.is_structural is structural
    assert canonical.canonical() == canonical


def test_structural_zero_direction_or_equal_base_totals_is_not_identity() -> None:
    two = certificate.AffineForm(_class_spec(), (1, -1), 0)
    assert two.evaluate((6, 6)) == 0
    assert not two.is_structural
    three = certificate.AffineForm(_class_spec(("a", "b", "c"), (1, 1, 1)),
                                   (1, 1, -2), 0)
    assert three.evaluate((4, 4, 4)) == three.evaluate((5, 3, 4)) == 0
    assert three.evaluate((5, 4, 3)) == 3
    assert not three.is_structural


@pytest.mark.parametrize(
    "call",
    (lambda: _class_spec(counts=(0, 1)), lambda: _class_spec(counts=(True, 1)),
     lambda: _class_spec(order=("a", "a")), lambda: _class_spec(budget=0),
     lambda: _class_spec(budget=float("inf")),
     lambda: certificate.AffineForm(_class_spec(), (1,), 0),
     lambda: certificate.AffineForm(_class_spec(), (1, float("nan")), 0)),
)
def test_affine_rejects_invalid_domain_or_nonfinite_inputs(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def _base_oracle(spec=None, anchors=(0, 0), features=((0,), (0,)), weights=(0,)):
    return certificate.BaseSignOracle(spec or _class_spec(), anchors, features, weights)


def test_base_sign_decimal_operations_enclose_exact_rational_results() -> None:
    arithmetic = certificate._BaseArithmetic(20)
    third = arithmetic.lift(Fraction(1, 3))
    seventh = arithmetic.lift(Fraction(1, 7))
    operations = (
        (third, Fraction(1, 3)),
        (arithmetic.add(third, seventh), Fraction(10, 21)),
        (arithmetic.sub(third, seventh), Fraction(4, 21)),
        (arithmetic.mul(third, arithmetic.lift(-7)), Fraction(-7, 3)),
        (arithmetic.div(third, seventh), Fraction(7, 3)),
    )
    for interval, exact in operations:
        assert Fraction(interval.lower) <= exact <= Fraction(interval.upper)
    with pytest.raises(ArithmeticError, match="positive denominator"):
        arithmetic.div(third, arithmetic.lift(0))


def test_base_sign_transcendentals_enclose_log_two_and_signed_tanh() -> None:
    arithmetic = certificate._BaseArithmetic(80)
    log_two = arithmetic.log_two()
    known_low = Decimal("0.69314718055994530941723212145817656807550013436025525412068000949339362196969471560")
    known_high = Decimal("0.69314718055994530941723212145817656807550013436025525412068000949339362196969471561")
    assert log_two.lower < known_low < known_high < log_two.upper
    assert not log_two.lower <= Decimal.from_float(math.log(2)) <= log_two.upper
    for value in (-2, 0, 2):
        interval = arithmetic.tanh(arithmetic.lift(value))
        with localcontext() as context:
            context.prec = 120
            exp_twice = Decimal(2 * value).exp()
            reference = (exp_twice - 1) / (exp_twice + 1)
        assert interval.lower <= reference <= interval.upper
        assert -1 <= interval.lower <= interval.upper <= 1


def test_base_sign_freezes_and_exactly_lifts_all_model_inputs() -> None:
    anchors, rows, weights = [0.1, 0], [[1e-9], [1e-12]], [0.2]
    oracle = _base_oracle(anchors=anchors, features=rows, weights=weights)
    anchors[0] = 7
    rows[0][0] = 7
    weights[0] = 7
    assert oracle.anchor_logits[0] == Fraction.from_float(0.1)
    assert oracle.feature_rows == ((Fraction.from_float(1e-9),),
                                   (Fraction.from_float(1e-12),))
    assert oracle.base_weights == (Fraction.from_float(0.2),)
    with pytest.raises(FrozenInstanceError):
        oracle.base_weights = (Fraction(7),)


@pytest.mark.parametrize(
    "anchors,features,weights",
    [((0,), ((0,), (0,)), (0,)), ((0, 0), ((0,),), (0,)),
     ((0, 0), ((0, 1), (0,)), (0,)), ((0, 0), ((0,), (0,)), ()),
     ((float('nan'), 0), ((0,), (0,)), (0,)),
     ((0, 0), ((float('inf'),), (0,)), (0,))],
)
def test_base_sign_rejects_invalid_problem_data(anchors, features, weights):
    with pytest.raises((TypeError, ValueError)):
        _base_oracle(anchors=anchors, features=features, weights=weights)


def test_base_sign_shared_denominator_preserves_signed_weighted_numerator() -> None:
    spec = _class_spec(counts=(1, 3))
    oracle = _base_oracle(spec)
    # u=(1,3), D=4: 12*(5*1 - 1*3)/4 - 1/3 = 17/3.
    form = certificate.AffineForm(spec, (5, -1), Fraction(-1, 3))
    result = oracle.evaluate(form)
    assert result.status == "separated"
    assert result.side == 1
    assert Fraction(result.interval.lower) <= Fraction(17, 3) <= Fraction(result.interval.upper)
    assert result.interval.lower > 0
    opposite = oracle.evaluate(-form)
    assert opposite.side == -1
    assert opposite.interval.upper < 0


def test_base_sign_uses_ideal_residual_surface_with_both_saturation_signs() -> None:
    oracle = _base_oracle(features=((1,), (-1,)), weights=(2,))
    form = certificate.AffineForm(oracle.spec, (1, -1), 0)
    result = oracle.evaluate(form)
    # Independent coarse analytic bound: tanh(2) > .9 makes u0/u1 > 2**1.8.
    assert result.side == 1
    assert Decimal(6) < result.interval.lower < result.interval.upper < Decimal(8)


def test_base_sign_witness_side_can_only_restrict_a_separating_interval() -> None:
    oracle = _base_oracle()
    form = certificate.AffineForm(oracle.spec, (1, 0), 0)
    assert oracle.evaluate(form, proposed_side=1).side == 1
    mismatch = oracle.evaluate(form, proposed_side=-1)
    assert mismatch.status == "uncertified"
    assert mismatch.side is None
    assert mismatch.reason == "witness_side_mismatch"
    with pytest.raises(ValueError, match="proposed side"):
        oracle.evaluate(form, proposed_side=0)


def test_base_sign_refines_tiny_positive_margin_and_exhausts_exact_zero() -> None:
    oracle = _base_oracle()
    near_zero = certificate.AffineForm(oracle.spec, (1, -1), Fraction(1, 10**100))
    refined = oracle.evaluate(near_zero)
    assert refined.side == 1
    assert refined.precisions == (80, 160)
    exact_zero = oracle.evaluate(certificate.AffineForm(oracle.spec, (1, -1), 0))
    assert exact_zero.status == "uncertified"
    assert exact_zero.side is None
    assert exact_zero.precisions == (80, 160, 320, 640)
    assert exact_zero.reason == "precision_exhausted"


@pytest.mark.parametrize("anchor", (10**7, -(10**7)))
def test_base_sign_overflow_or_underflow_abstains_without_float_fallback(anchor):
    oracle = _base_oracle(anchors=(anchor, anchor))
    result = oracle.evaluate(certificate.AffineForm(oracle.spec, (1, 0), 0))
    assert result.status == "uncertified"
    assert result.side is None
    assert result.reason.startswith("arithmetic_failure:")


def test_base_sign_ignores_ambient_decimal_precision_and_rejects_wrong_domain() -> None:
    oracle = _base_oracle()
    form = certificate.AffineForm(oracle.spec, (1, 0), 0)
    baseline = oracle.evaluate(form)
    with localcontext() as context:
        context.prec = 2
        context.Emax = 2
        assert oracle.evaluate(form) == baseline
    with pytest.raises(ValueError, match="domain"):
        oracle.evaluate(certificate.AffineForm(_class_spec(budget=13), (1, 0), 0))


def test_guard_factory_derives_structural_orientation_and_unresolved_variable() -> None:
    oracle = _base_oracle()
    structural = certificate._compile_affine_decision(
        certificate.AffineForm(oracle.spec, (2, 2), -24), oracle)
    assert structural.kind is certificate.DecisionKind.STRUCTURAL
    assert structural.orientation == 0
    assert structural.margin.lower == structural.margin.upper == 0
    negative = certificate.AffineForm(oracle.spec, (-1, 0), 0)
    variable = certificate._compile_affine_decision(negative, oracle)
    assert variable.kind is certificate.DecisionKind.VARIABLE
    assert variable.orientation == -1
    assert variable.guard == -negative
    assert variable.margin.lower > 0
    unresolved = certificate._compile_affine_decision(
        certificate.AffineForm(oracle.spec, (1, -1), 0), oracle)
    assert unresolved.kind is certificate.DecisionKind.VARIABLE
    assert unresolved.orientation is None
    assert unresolved.margin is None
    assert unresolved.base_result.status == "uncertified"


def test_guard_decisions_cannot_be_constructed_or_declared_by_public_callers() -> None:
    with pytest.raises(TypeError, match="public compiler"):
        certificate.CompiledDecision(kind=certificate.DecisionKind.STRUCTURAL,
                                     orientation=1, margin=42)
    oracle = _base_oracle()
    form = certificate.AffineForm(oracle.spec, (1, 0), 0)
    with pytest.raises(TypeError):
        certificate._compile_affine_decision(form, oracle, kind="static", margin=42)
    decision = certificate._compile_affine_decision(form, oracle)
    with pytest.raises(FrozenInstanceError):
        decision.orientation = -1
    assert "CompiledDecision" not in certificate.__all__


@pytest.mark.parametrize("override", ("structural", "canonical"))
def test_guard_rejects_form_subclasses_that_forge_structural_identity(override):
    class LyingStructural(certificate.AffineForm):
        @property
        def is_structural(self):
            return True

    class LyingCanonical(certificate.AffineForm):
        def canonical(self):
            return certificate.AffineForm(self.spec, (0, 0), 0)

    oracle = _base_oracle()
    subclass = LyingStructural if override == "structural" else LyingCanonical
    form = subclass(oracle.spec, (0, 0), 1)
    with pytest.raises(TypeError, match="concrete AffineForm"):
        certificate._compile_affine_decision(form, oracle)


@pytest.mark.parametrize("override", ("evaluate", "enclose", "duck"))
@pytest.mark.parametrize("h", ((1, -1), (-1, 0)))
def test_guard_rejects_oracle_overrides_that_forge_positive_separation(override, h):
    real_oracle = _base_oracle()
    positive = real_oracle.evaluate(certificate.AffineForm(real_oracle.spec, (1, 0), 0))

    class LyingEvaluate(certificate.BaseSignOracle):
        def evaluate(self, *args, **kwargs):
            return positive

    class LyingEnclose(certificate.BaseSignOracle):
        def _enclose(self, *args, **kwargs):
            return positive.interval

    class DuckOracle:
        def _require_domain(self, form):
            pass

        def evaluate(self, *args, **kwargs):
            return positive

    if override == "duck":
        oracle = DuckOracle()
    else:
        subclass = LyingEvaluate if override == "evaluate" else LyingEnclose
        oracle = subclass(real_oracle.spec, (0, 0), ((0,), (0,)), (0,))
    form = certificate.AffineForm(real_oracle.spec, h, 0)
    with pytest.raises(TypeError, match="concrete BaseSignOracle"):
        certificate._compile_affine_decision(form, oracle)


@pytest.mark.parametrize("consumer", ("affine", "oracle"))
def test_guard_rejects_overridable_class_domains(consumer):
    class LyingDomain(certificate.ClassSpec):
        def __eq__(self, other):
            return True

    spec = LyingDomain(("a", "b"), (1, 1), 13)
    with pytest.raises(TypeError, match="concrete ClassSpec"):
        if consumer == "affine":
            certificate.AffineForm(spec, (1, 0), 0)
        else:
            _base_oracle(spec)


def test_guard_normalizes_fraction_overrides_before_exact_structural_checks():
    class LyingZero(Fraction):
        def __eq__(self, other):
            return True

        def __add__(self, other):
            return self

    oracle = _base_oracle()
    form = certificate.AffineForm(oracle.spec, (0, 0), LyingZero(1, 1))
    decision = certificate._compile_affine_decision(form, oracle)
    assert type(form.beta) is Fraction
    assert form.beta == 1
    assert decision.kind is certificate.DecisionKind.VARIABLE
    assert decision.orientation == 1
    assert decision.margin.lower > 0
