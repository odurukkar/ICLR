"""Immutable, process-scoped context bindings for residual MES traces.

The bindings in this module authenticate one observed floating execution. They
are audit provenance, not an exact-real certificate and not evidence that the
bytes currently on disk are the bytecode already loaded by this interpreter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import Counter
from dataclasses import dataclass, fields
from decimal import (
    Context, Decimal, DecimalException, DivisionByZero, InvalidOperation,
    Overflow, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Subnormal, Underflow,
)
from enum import Enum
from fractions import Fraction
import hashlib
from functools import cmp_to_key
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import secrets
import stat
import struct
import sys

import numpy as np

import cohorts
import iclr_priority_mes as production
import iclr_residual_actuation as actuation
import iclr_residual_policy as residual
from iclr_env import RolloutState
from parse_pb import PBInstance, Project, Vote
import rules


__all__ = [
    "NumericRuntime",
    "TraceBinding",
    "BoundTrace",
    "UncertifiedTrace",
    "compile_bound_trace",
    "RealSegmentProof",
    "certify_real_segment",
    "FloatSegmentAudit",
    "audit_float_segment",
]

BINDING_SCHEMA = "trace-context-v1"
_INSTANCE_SCHEMA = "pb-instance-semantics-v1"
_MISSING_DEMOGRAPHIC = "__missing_demographic__"
_PROCESS_NONCE = secrets.token_hex(32)
_SOURCE_DIRECTORY = Path(__file__).resolve().parent
_DEFAULT_SOURCE_ROLE_PATHS = (
    ("reference_solver", _SOURCE_DIRECTORY / "rules.py"),
    ("witness", _SOURCE_DIRECTORY / "iclr_residual_actuation.py"),
    ("production_solver", _SOURCE_DIRECTORY / "iclr_priority_mes.py"),
    ("production_features", _SOURCE_DIRECTORY / "iclr_outcome.py"),
    ("residual_policy", _SOURCE_DIRECTORY / "iclr_residual_policy.py"),
    ("static_features", _SOURCE_DIRECTORY / "iclr_policy.py"),
    ("cohort_assignment", _SOURCE_DIRECTORY / "cohorts.py"),
    ("instance_schema", _SOURCE_DIRECTORY / "parse_pb.py"),
    ("certificate_module", Path(__file__).resolve()),
)
_SOURCE_ROLE_PATHS = _DEFAULT_SOURCE_ROLE_PATHS
_EXPECTED_SOURCE_ROLES = frozenset(role for role, _ in _DEFAULT_SOURCE_ROLE_PATHS)
_POLICY_NORMALIZATION_FAILURES = frozenset(
    {
        "normalized log endowments are not representable",
        "logit produces a non-representable multiplier",
        "positive endowments are not representable for this budget",
        "normalization failed to conserve the budget",
    }
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class _FactoryOnly:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("objects are created only by the public compiler")


@dataclass(frozen=True, init=False)
class NumericRuntime(_FactoryOnly):
    """Primitive runtime identity retained in every process-scoped binding."""

    python_implementation: str
    python_version: str
    python_cache_tag: str
    python_version_info: tuple[str, ...]
    numpy_version: str
    byteorder: str
    float_mant_dig: int
    float_max_exp: int
    float_rounds: int
    process_id: int
    process_nonce: str


@dataclass(frozen=True, init=False)
class TraceBinding(_FactoryOnly):
    """Complete immutable semantic context for one trace payload."""

    schema: str
    scope: str
    runtime: NumericRuntime
    instance_semantics_sha256: str
    instance_path: str
    project_id_sha256: str
    metadata: tuple[tuple[str, str], ...]
    budget_bits: str
    projects: tuple[tuple[object, ...], ...]
    votes: tuple[tuple[object, ...], ...]
    class_partition: tuple[tuple[str, tuple[int, ...]], ...]
    anchor: tuple[object, ...]
    scaler: tuple[object, ...]
    base_weight_bits: tuple[str, ...]
    endowment_bits: tuple[str, ...]
    completion: bool
    rollout_state: tuple[object, ...]
    trace_payload: tuple[object, ...]
    reference_winners: tuple[str, ...]
    production_winners: tuple[str, ...]
    source_role_sha256: tuple[tuple[str, str], ...]
    input_file_sha256: tuple[tuple[str, str], ...]
    residual_policy_state_semantics: str
    source_hashes_are_execution_proof: bool
    float_runtime_proved: bool
    cache_safe: bool
    context_digest: str


@dataclass(frozen=True, init=False)
class BoundTrace(_FactoryOnly):
    """A trace that can be consumed only under its exact compiled binding."""

    binding: TraceBinding
    trace: actuation.EqualSharesTrace

    def require(self, expected: TraceBinding) -> actuation.EqualSharesTrace:
        runtime = self.binding.runtime
        if (
            runtime.process_id != os.getpid()
            or runtime.process_nonce != _PROCESS_NONCE
        ):
            raise ValueError("trace process scope mismatch")
        if self.binding != expected:
            raise ValueError("trace context mismatch")
        return self.trace


@dataclass(frozen=True, init=False)
class UncertifiedTrace(_FactoryOnly):
    """Explicit abstention when a runtime prerequisite cannot be established."""

    certified: bool
    reason: str
    context_digest: str
    instance_semantics_sha256: str
    endowment_bits: tuple[str, ...]
    runtime: NumericRuntime
    source_role_sha256: tuple[tuple[str, str], ...]
    input_file_sha256: tuple[tuple[str, str], ...]
    float_runtime_proved: bool
    cache_safe: bool


def _factory_create(cls, /, **values):
    expected = {item.name for item in fields(cls)}
    if set(values) != expected:
        raise AssertionError(f"internal factory fields differ for {cls.__name__}")
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    return instance


def _float_bits(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number and not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return struct.pack(">d", number).hex()


def _float_from_bits(value: str) -> float:
    return struct.unpack(">d", bytes.fromhex(value))[0]


def _exact_fraction(value: object) -> Fraction:
    """Lift runtime binary values, never their rounded decimal spelling."""
    if isinstance(value, bool):
        raise TypeError("exact inputs must not be boolean")
    if isinstance(value, Fraction):
        # Retain numerical data, never subclass arithmetic/equality overrides.
        return Fraction(int(value.numerator), int(value.denominator))
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("exact inputs must be finite")
        numerator, denominator = value.as_integer_ratio()
        return Fraction(numerator, denominator)
    raise TypeError("exact inputs must be integers, Fractions or binary floats")


@dataclass(frozen=True)
class ClassSpec:
    """Ordered affine domain; public construction supplies problem data only."""

    class_order: tuple[str, ...]
    multiplicities: tuple[int, ...]
    budget: Fraction

    def __post_init__(self) -> None:
        order = tuple(self.class_order)
        counts = tuple(self.multiplicities)
        if not order or any(not isinstance(key, str) or not key for key in order):
            raise ValueError("class order requires nonempty string identifiers")
        if len(set(order)) != len(order):
            raise ValueError("class identifiers must be unique")
        if len(counts) != len(order) or any(
            isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in counts
        ):
            raise ValueError("class multiplicities must be positive integers")
        budget = _exact_fraction(self.budget)
        if budget <= 0:
            raise ValueError("municipal budget must be positive")
        object.__setattr__(self, "class_order", order)
        object.__setattr__(self, "multiplicities", tuple(int(n) for n in counts))
        object.__setattr__(self, "budget", budget)


@dataclass(frozen=True)
class AffineForm:
    """Exact h.y + beta, bound to one ordered budget simplex."""

    spec: ClassSpec
    h: tuple[Fraction, ...]
    beta: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if type(self.spec) is not ClassSpec:
            raise TypeError("affine domain must be a concrete ClassSpec")
        coefficients = tuple(_exact_fraction(value) for value in self.h)
        if len(coefficients) != len(self.spec.class_order):
            raise ValueError("affine coefficient dimension differs from its domain")
        object.__setattr__(self, "h", coefficients)
        object.__setattr__(self, "beta", _exact_fraction(self.beta))

    def _require_domain(self, other: AffineForm) -> None:
        if not isinstance(other, AffineForm) or self.spec != other.spec:
            raise ValueError("affine operations require the same domain")

    def __add__(self, other: AffineForm) -> AffineForm:
        self._require_domain(other)
        return AffineForm(self.spec, tuple(a + b for a, b in zip(self.h, other.h)),
                          self.beta + other.beta)

    def __sub__(self, other: AffineForm) -> AffineForm:
        self._require_domain(other)
        return self + (-other)

    def __neg__(self) -> AffineForm:
        return self * -1

    def __mul__(self, scalar: object) -> AffineForm:
        factor = _exact_fraction(scalar)
        return AffineForm(self.spec, tuple(factor * h for h in self.h), factor * self.beta)

    __rmul__ = __mul__

    def __truediv__(self, payer_count: int) -> AffineForm:
        if isinstance(payer_count, bool) or not isinstance(payer_count, Integral):
            raise TypeError("payer count must be an exact positive integer")
        if payer_count <= 0:
            raise ValueError("payer count must be an exact positive integer")
        return self * Fraction(1, int(payer_count))

    def evaluate(self, class_totals: Sequence[object]) -> Fraction:
        values = tuple(_exact_fraction(value) for value in class_totals)
        if len(values) != len(self.h):
            raise ValueError("class total dimension differs from affine domain")
        return sum((h * value for h, value in zip(self.h, values)), self.beta)

    def canonical(self) -> AffineForm:
        reference = self.h[-1]
        return AffineForm(self.spec, tuple(h - reference for h in self.h),
                          self.beta + reference * self.spec.budget)

    @property
    def is_structural(self) -> bool:
        canonical = self.canonical()
        return canonical.beta == 0 and all(h == 0 for h in canonical.h)


@dataclass(frozen=True)
class _DecimalInterval:
    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        if not self.lower.is_finite() or not self.upper.is_finite():
            raise ArithmeticError("nonfinite interval endpoint")
        if self.lower > self.upper:
            raise ArithmeticError("reversed interval endpoints")

    def negated(self) -> _DecimalInterval:
        # Decimal unary minus uses the ambient context; copy_negate is exact.
        return _DecimalInterval(self.upper.copy_negate(), self.lower.copy_negate())


class _BaseArithmetic:
    """Outward operations independent of the caller's Decimal context."""

    def __init__(self, precision: int) -> None:
        self.nearest = Context(
            prec=precision, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
            traps=[DivisionByZero, InvalidOperation, Overflow, Underflow, Subnormal],
        )
        self.down = self.nearest.copy()
        self.down.rounding = ROUND_FLOOR
        self.up = self.nearest.copy()
        self.up.rounding = ROUND_CEILING

    def lift(self, value: object) -> _DecimalInterval:
        exact = _exact_fraction(value)
        numerator, denominator = Decimal(exact.numerator), Decimal(exact.denominator)
        return _DecimalInterval(self.down.divide(numerator, denominator),
                                self.up.divide(numerator, denominator))

    def add(self, a: _DecimalInterval, b: _DecimalInterval) -> _DecimalInterval:
        return _DecimalInterval(self.down.add(a.lower, b.lower),
                                self.up.add(a.upper, b.upper))

    def sub(self, a: _DecimalInterval, b: _DecimalInterval) -> _DecimalInterval:
        return self.add(a, b.negated())

    def mul(self, a: _DecimalInterval, b: _DecimalInterval) -> _DecimalInterval:
        pairs = tuple((x, y) for x in (a.lower, a.upper) for y in (b.lower, b.upper))
        return _DecimalInterval(min(self.down.multiply(x, y) for x, y in pairs),
                                max(self.up.multiply(x, y) for x, y in pairs))

    def div(self, a: _DecimalInterval, b: _DecimalInterval) -> _DecimalInterval:
        if b.lower <= 0:
            raise ArithmeticError("division requires a positive denominator")
        pairs = tuple((x, y) for x in (a.lower, a.upper) for y in (b.lower, b.upper))
        return _DecimalInterval(min(self.down.divide(x, y) for x, y in pairs),
                                max(self.up.divide(x, y) for x, y in pairs))

    def exp(self, value: _DecimalInterval) -> _DecimalInterval:
        # Decimal exp/ln are correctly rounded nearest; adjacent representable
        # values enclose their exact-real result at each monotone endpoint.
        lower = self.nearest.exp(value.lower)
        upper = self.nearest.exp(value.upper)
        return _DecimalInterval(self.nearest.next_minus(lower),
                                self.nearest.next_plus(upper))

    def log_two(self) -> _DecimalInterval:
        value = self.nearest.ln(Decimal(2))
        return _DecimalInterval(self.nearest.next_minus(value),
                                self.nearest.next_plus(value))

    def _tanh_point(self, value: Decimal) -> _DecimalInterval:
        if value == 0:
            return self.lift(0)
        # exp(-2 |s|) avoids positive exponent overflow for saturated tanh.
        absolute = _DecimalInterval(value.copy_abs(), value.copy_abs())
        exponential = self.exp(self.mul(self.lift(-2), absolute))
        one = self.lift(1)
        positive = self.div(self.sub(one, exponential), self.add(one, exponential))
        positive = _DecimalInterval(max(Decimal(0), positive.lower),
                                    min(Decimal(1), positive.upper))
        return positive.negated() if value < 0 else positive

    def tanh(self, value: _DecimalInterval) -> _DecimalInterval:
        # tanh is increasing, including intervals straddling zero.
        return _DecimalInterval(self._tanh_point(value.lower).lower,
                                self._tanh_point(value.upper).upper)


@dataclass(frozen=True, init=False)
class _BaseSignResult(_FactoryOnly):
    """An affine base sign only; this is never a trace or runtime proof."""

    status: str
    side: int | None
    interval: _DecimalInterval | None
    precisions: tuple[int, ...]
    reason: str | None


@dataclass(frozen=True)
class BaseSignOracle:
    """Enclose the ideal real base model from exact binary problem inputs.

    Inputs are a_c, z_c, w0. Their extraction from an authenticated bound trace
    belongs to later symbolic compilation, not to this algebra-only layer.
    """

    spec: ClassSpec
    anchor_logits: tuple[Fraction, ...]
    feature_rows: tuple[tuple[Fraction, ...], ...]
    base_weights: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        if type(self.spec) is not ClassSpec:
            raise TypeError("base oracle domain must be a concrete ClassSpec")
        anchors = tuple(_exact_fraction(value) for value in self.anchor_logits)
        rows = tuple(tuple(_exact_fraction(value) for value in row)
                     for row in self.feature_rows)
        weights = tuple(_exact_fraction(value) for value in self.base_weights)
        if len(anchors) != len(self.spec.class_order) or len(rows) != len(anchors):
            raise ValueError("base model class dimensions differ from its domain")
        if any(len(row) != len(weights) for row in rows):
            raise ValueError("base model feature and weight dimensions differ")
        object.__setattr__(self, "anchor_logits", anchors)
        object.__setattr__(self, "feature_rows", rows)
        object.__setattr__(self, "base_weights", weights)

    def _require_domain(self, form: AffineForm) -> None:
        if not isinstance(form, AffineForm) or form.spec != self.spec:
            raise ValueError("base oracle and guard require the same domain")

    def _enclose(self, form: AffineForm, precision: int) -> _DecimalInterval:
        arithmetic = _BaseArithmetic(precision)
        log_two = arithmetic.log_two()
        numerator = denominator = arithmetic.lift(0)
        for count, anchor, row, coefficient in zip(
            self.spec.multiplicities, self.anchor_logits, self.feature_rows, form.h,
        ):
            score = sum((z * w for z, w in zip(row, self.base_weights)), Fraction(0))
            residual_logit = arithmetic.mul(log_two, arithmetic.tanh(arithmetic.lift(score)))
            u = arithmetic.mul(arithmetic.lift(count),
                               arithmetic.exp(arithmetic.add(arithmetic.lift(anchor),
                                                              residual_logit)))
            denominator = arithmetic.add(denominator, u)
            numerator = arithmetic.add(numerator, arithmetic.mul(arithmetic.lift(coefficient), u))
        # Aggregate the signed weighted numerator before the ONE shared D.
        # Separate enclosures of y_c = B*u_c/D lose denominator correlation.
        quotient = arithmetic.div(arithmetic.mul(arithmetic.lift(self.spec.budget),
                                                 numerator), denominator)
        return arithmetic.add(quotient, arithmetic.lift(form.beta))

    def evaluate(self, form: AffineForm, *, proposed_side: int | None = None) -> _BaseSignResult:
        self._require_domain(form)
        if proposed_side is not None and (
            isinstance(proposed_side, bool) or not isinstance(proposed_side, Integral)
            or proposed_side not in (-1, 1)
        ):
            raise ValueError("proposed side must be -1 or 1")
        attempted: list[int] = []
        interval = None
        for precision in (80, 160, 320, 640):
            attempted.append(precision)
            try:
                interval = self._enclose(form, precision)
            except (DecimalException, ArithmeticError) as exc:
                return _factory_create(
                    _BaseSignResult, status="uncertified", side=None, interval=None,
                    precisions=tuple(attempted), reason=f"arithmetic_failure:{type(exc).__name__}",
                )
            side = 1 if interval.lower > 0 else -1 if interval.upper < 0 else None
            if side is not None:
                mismatch = proposed_side is not None and side != proposed_side
                return _factory_create(
                    _BaseSignResult, status="uncertified" if mismatch else "separated",
                    side=None if mismatch else side, interval=interval,
                    precisions=tuple(attempted), reason="witness_side_mismatch" if mismatch else None,
                )
        return _factory_create(
            _BaseSignResult, status="uncertified", side=None, interval=interval,
            precisions=tuple(attempted), reason="precision_exhausted",
        )


class DecisionKind(Enum):
    VARIABLE = "variable"
    STRUCTURAL = "structural"
    STATIC = "static"


@dataclass(frozen=True, init=False)
class CompiledDecision(_FactoryOnly):
    """Compiler-derived local decision, with no trace-coverage implication.

    A variable's oriented margin is enclosed only when the base sign separates.
    Strict positivity is a later theorem precondition, not an input invariant.
    An unresolved variable remains present with no orientation or margin proof.
    """

    kind: DecisionKind
    guard: AffineForm
    orientation: int | None
    margin: _DecimalInterval | None
    base_result: _BaseSignResult | None


def _compile_affine_decision(
    form: AffineForm, oracle: BaseSignOracle, *, proposed_side: int | None = None,
) -> CompiledDecision:
    # Validate concrete inputs before calling any potentially overridable method.
    if type(form) is not AffineForm:
        raise TypeError("decision guard must be a concrete AffineForm")
    if type(oracle) is not BaseSignOracle:
        raise TypeError("decision oracle must be a concrete BaseSignOracle")
    oracle._require_domain(form)
    if form.is_structural:
        return _factory_create(
            CompiledDecision, kind=DecisionKind.STRUCTURAL, guard=form.canonical(),
            orientation=0, margin=_DecimalInterval(Decimal(0), Decimal(0)), base_result=None,
        )
    result = oracle.evaluate(form, proposed_side=proposed_side)
    oriented = -form if result.side == -1 else form
    margin = None
    if result.side is not None:
        margin = result.interval.negated() if result.side == -1 else result.interval
    return _factory_create(
        CompiledDecision, kind=DecisionKind.VARIABLE, guard=oriented,
        orientation=result.side, margin=margin, base_result=result,
    )


@dataclass(frozen=True)
class ControlSite:
    name: str
    kinds: tuple[DecisionKind | None, ...]
    outcomes: tuple[str, ...]


# Independent handwritten control-flow contract, never built from a trace or
# the DecisionKind enum. Comparisons record the sign of their original guard.
_STATIC = (DecisionKind.STATIC,)
_AFFINE = (DecisionKind.VARIABLE, DecisionKind.STRUCTURAL)
_SIGNS = ("negative", "zero", "positive")
CONTROL_SITES = (
    ControlSite("initial_support", _STATIC, ("zero", "positive")),
    ControlSite("initial_cost", _STATIC, ("zero", "positive")),
    ControlSite("affordability", _AFFINE, _SIGNS),
    ControlSite("payer_order", _AFFINE, _SIGNS),
    ControlSite("breakpoint", _AFFINE, _SIGNS),
    ControlSite("rho_order", _AFFINE, _SIGNS),
    ControlSite("candidate_cost", _STATIC, _SIGNS),
    ControlSite("candidate_id", _STATIC, _SIGNS),
    ControlSite("charge", _AFFINE, _SIGNS),
    ControlSite("completion_count", _STATIC, _SIGNS),
    ControlSite("completion_cost", _STATIC, _SIGNS),
    ControlSite("completion_id", _STATIC, _SIGNS),
    ControlSite("completion_positive", _STATIC, ("zero", "positive")),
    ControlSite("completion_affordability", _STATIC, _SIGNS),
    ControlSite("initial_active", (None,), ("snapshot",)),
    ControlSite("active_before", (None,), ("snapshot",)),
    ControlSite("candidate_visit", (None,), ("visit",)),
    ControlSite("removal", (None,), ("unaffordable", "no_rho")),
    ControlSite("incumbent", (None,), ("initialize", "retain", "replace")),
    ControlSite("chosen", (None,), ("payment",)),
    ControlSite("active_after", (None,), ("snapshot",)),
    ControlSite("termination", (None,), ("empty_active", "no_candidate")),
    ControlSite("completion_mode", (None,), ("enabled", "disabled")),
    ControlSite("completion_visit", (None,), ("visit",)),
    ControlSite("completion_skip", (None,), ("winner", "zero_support")),
    ControlSite("completion_remaining", (None,), ("initial", "transition")),
    ControlSite("completion_decision", (None,), ("fund", "zero_cost", "unaffordable")),
    ControlSite("final_outcome", (None,), ("winners",)),
)
_SITE_BY_NAME = {site.name: site for site in CONTROL_SITES}


@dataclass(frozen=True, init=False)
class BranchObligation(_FactoryOnly):
    event_id: tuple[object, ...]
    owner: object


@dataclass(frozen=True)
class TraceEvent:
    event_id: tuple[object, ...]
    site: str
    outcome: str
    payload: tuple[object, ...]
    decision: CompiledDecision | None = None


class _CoverageLedger:
    """Driver opens obligations before branching; factory consumes each once."""

    def __init__(self):
        self._owner = object()
        self._opened: dict[tuple[object, ...], BranchObligation] = {}
        self._consumed: dict[tuple[object, ...], TraceEvent] = {}

    def open(self, site: str, location: tuple[object, ...]) -> BranchObligation:
        if site not in _SITE_BY_NAME:
            raise ValueError("unknown control site")
        event_id = (len(self._opened), site, location)
        obligation = _factory_create(BranchObligation, event_id=event_id, owner=self._owner)
        self._opened[event_id] = obligation
        return obligation

    def emit(self, obligation, outcome, payload, decision=None) -> TraceEvent:
        if type(obligation) is not BranchObligation or obligation.owner is not self._owner:
            raise ValueError("foreign branch obligation")
        key = obligation.event_id
        if self._opened.get(key) is not obligation:
            raise ValueError("unknown branch obligation")
        if key in self._consumed:
            raise ValueError("branch obligation already consumed")
        site = _SITE_BY_NAME[key[1]]
        if decision is not None and type(decision) is not CompiledDecision:
            raise TypeError("coverage requires a concrete compiler decision")
        kind = None if decision is None else decision.kind
        if outcome not in site.outcomes or kind not in site.kinds:
            raise ValueError("illegal control-site kind or outcome")
        if decision is not None and (
            decision.orientation not in (-1, 0, 1)
            or outcome != _SIGNS[decision.orientation + 1]
            or (kind is DecisionKind.STRUCTURAL and decision.orientation != 0)
            or (kind is DecisionKind.VARIABLE and decision.orientation == 0)
        ):
            raise ValueError("control-site outcome disagrees with decision orientation")
        event = TraceEvent(key, site.name, outcome, tuple(payload), decision)
        self._consumed[key] = event
        return event

    def validate(self, events: Sequence[TraceEvent]) -> None:
        if (Counter(self._opened.keys()) != Counter(event.event_id for event in events)
                or self._opened.keys() != self._consumed.keys()
                or any(self._consumed.get(event.event_id) != event for event in events)):
            raise ValueError("trace coverage mismatch")


def _require_bound(bound: BoundTrace) -> TraceBinding:
    if type(bound) is not BoundTrace or type(bound.binding) is not TraceBinding:
        raise TypeError("symbolic compiler requires a concrete BoundTrace and TraceBinding")
    binding = bound.binding
    if type(binding.runtime) is not NumericRuntime:
        raise TypeError("symbolic compiler requires a concrete runtime binding")
    trace = BoundTrace.require(bound, binding)
    if type(trace) is not actuation.EqualSharesTrace or _freeze_trace(trace) != binding.trace_payload:
        raise ValueError("trace payload context mismatch")
    payload = {f.name: (_runtime_payload(binding.runtime) if f.name == "runtime"
                       else getattr(binding, f.name))
               for f in fields(binding) if f.name != "context_digest"}
    if _canonical_sha256(payload) != binding.context_digest:
        raise ValueError("binding context digest mismatch")
    return binding


@dataclass(frozen=True, init=False)
class ResidualClassSurface(_FactoryOnly):
    """Exact binary model inputs extracted only from a live context binding."""

    spec: ClassSpec
    voter_classes: tuple[int, ...]
    anchor_logits: tuple[Fraction, ...]
    feature_rows: tuple[tuple[Fraction, ...], ...]
    base_weights: tuple[Fraction, ...]
    initial_balances: tuple[AffineForm, ...]
    context_digest: str

    @classmethod
    def from_bound(cls, bound: BoundTrace) -> ResidualClassSurface:
        if cls is not ResidualClassSurface:
            raise TypeError("surface must be a concrete ResidualClassSurface")
        binding = _require_bound(bound)
        snapshot = (binding.metadata, binding.budget_bits, binding.projects, binding.votes)
        inst = _materialize_instance(snapshot, binding.instance_semantics_sha256)
        family, logits, reference, _ = binding.anchor
        anchor = residual.AnchorSpec(family, tuple(_float_from_bits(v) for v in logits), reference)
        names, means, scales, clip, rows, instances, _ = binding.scaler
        scaler = residual.FeatureScaler(
            names, tuple(_float_from_bits(v) for v in means),
            tuple(_float_from_bits(v) for v in scales), _float_from_bits(clip), rows, instances,
        )
        partition = _validate_partition(binding.class_partition, len(binding.votes))
        if partition != _derive_partition_snapshot(inst.votes):
            raise ValueError("bound class partition disagrees with demographic inputs")
        spec = ClassSpec(tuple(label for label, _ in partition),
                         tuple(len(members) for _, members in partition),
                         _float_from_bits(binding.budget_bits))
        contexts = residual.standardized_context(inst, scaler)
        anchor_map = residual.anchor_logit_map(anchor)
        anchors, features = [], []
        voter_classes = [0] * len(inst.votes)
        for c, (label, members) in enumerate(partition):
            key = label.split("|", 1)[0] if family in ("age", "senior") else label
            anchors.append(_exact_fraction(anchor_map.get(key, 0.0)))
            features.append(tuple(_exact_fraction(v) for v in contexts.get(label, (0.0,) * 4)))
            for voter in members:
                voter_classes[voter] = c
        initial = tuple(AffineForm(spec, tuple(Fraction(1, spec.multiplicities[c])
                                               if j == c else Fraction(0)
                                               for j in range(len(partition))))
                        for c in voter_classes)
        return _factory_create(
            ResidualClassSurface, spec=spec, voter_classes=tuple(voter_classes),
            anchor_logits=tuple(anchors), feature_rows=tuple(features),
            base_weights=tuple(_exact_fraction(_float_from_bits(v)) for v in binding.base_weight_bits),
            initial_balances=initial, context_digest=binding.context_digest,
        )


@dataclass(frozen=True, init=False)
class RuntimeAuditProjection(_FactoryOnly):
    compared_events: tuple[str, ...]
    payment_order_matches: bool
    removed_project_sets_match: bool
    completion_decisions_match: bool
    final_winners_match: bool
    matched_charge_sides: int
    charge_side_matches: tuple[tuple[object, ...], ...]
    omitted_micro_events: tuple[tuple[object, ...], ...]
    production_difference: str
    full_trace_identity: bool


@dataclass(frozen=True, init=False)
class CompiledTrace(_FactoryOnly):
    """Complete exact-real base trace, not a movement or runtime certificate."""

    bound: BoundTrace
    surface: ResidualClassSurface
    events: tuple[TraceEvent, ...]
    obligations: tuple[tuple[object, ...], ...]
    payment_order: tuple[str, ...]
    completion_order: tuple[str, ...]
    winners: tuple[str, ...]
    final_balances: tuple[AffineForm, ...]
    histories: tuple[tuple[tuple[str, AffineForm], ...], ...]
    audit: RuntimeAuditProjection
    float_runtime_proved: bool
    cache_safe: bool
    payload_digest: str

    def validate_coverage(self, events: Sequence[TraceEvent] | None = None) -> None:
        _require_bound(self.bound)
        observed = self.events if events is None else tuple(events)
        if (Counter(self.obligations) != Counter(e.event_id for e in observed)
                or observed != self.events):
            raise ValueError("trace coverage mismatch")


def _compiled_payload_chunks(trace: CompiledTrace):
    """Yield the legacy typed canonical JSON as bounded UTF8 chunks.

    Walk shared records on every occurrence and every call. No expanded tagged
    tree, complete JSON string, complete byte buffer or identity cache is built.
    JSON string escaping is character-local, so slicing before escaping preserves
    exactly ensure_ascii=False bytes, including control characters and Unicode.
    """
    records = (ResidualClassSurface, ClassSpec, AffineForm, TraceEvent,
               CompiledDecision, _BaseSignResult, _DecimalInterval, RuntimeAuditProjection)

    def scalar(value):
        if type(value) is str:
            yield b'"'
            for offset in range(0, len(value), 1024):
                escaped = json.dumps(value[offset:offset + 1024], ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":"), allow_nan=False)
                yield escaped[1:-1].encode("utf-8")
            yield b'"'
        else:
            # Integers/bools (and the raw context value) keep JSON's exact
            # representation and errors. Only scalar text is materialized.
            text = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
            for offset in range(0, len(text), 1024):
                yield text[offset:offset + 1024].encode("utf-8")

    def sequence(values):
        yield b'['
        for index, value in enumerate(values):
            if index:
                yield b','
            yield from encode(value)
        yield b']'

    def named_values(values):
        yield b'['
        for index, (name, value) in enumerate(values):
            if index:
                yield b','
            yield b'['
            yield from scalar(name)
            yield b','
            yield from encode(value)
            yield b']'
        yield b']'

    def encode(value):
        kind = type(value)
        if value is None:
            yield b'["none"]'
        elif kind in (str, int, bool):
            yield b'["' + kind.__name__.encode("ascii") + b'",'
            yield from scalar(value)
            yield b']'
        elif kind is Fraction:
            yield b'["fraction",'
            yield from scalar(value.numerator)
            yield b','
            yield from scalar(value.denominator)
            yield b']'
        elif kind is Decimal:
            yield b'["decimal",'
            yield from scalar(str(value))
            yield b']'
        elif kind is DecisionKind:
            yield b'["decision_kind",'
            yield from scalar(value.value)
            yield b']'
        elif kind is tuple:
            # Ordinary tuples must never impersonate another type's encoding.
            yield b'["tuple",'
            yield from sequence(value)
            yield b']'
        elif kind in records:
            yield b'["record",'
            yield from scalar(kind.__name__)
            yield b','
            yield from named_values((f.name, getattr(value, f.name)) for f in fields(value))
            yield b']'
        else:
            raise TypeError("compiled payload requires concrete proof records")

    yield b'['
    yield from scalar(trace.bound.binding.context_digest)
    yield b','
    yield from named_values((f.name, getattr(trace, f.name)) for f in fields(CompiledTrace)
                           if f.name not in ("bound", "payload_digest"))
    yield b']'


def _compiled_payload_digest(trace: CompiledTrace) -> str:
    """Seal all proof fields with the unchanged digest and a 64 KiB logical buffer.

    Detect altered immutable records, not arbitrary interpreter code execution.
    Concrete types are checked by the emitter before overridable dispatch; the
    caller's existing live context/decision/coverage checks remain separate.
    """
    digest, buffer = hashlib.sha256(), bytearray()
    for chunk in _compiled_payload_chunks(trace):
        if len(buffer) + len(chunk) > 65536:
            digest.update(buffer)
            buffer.clear()
        buffer.extend(chunk)
    digest.update(buffer)
    return digest.hexdigest()


def _require_compiled_decision_types(trace: CompiledTrace) -> None:
    """Validate decision field types before coverage hashing or branch filtering."""
    if type(trace.events) is not tuple:
        raise TypeError("compiled events require a concrete tuple")
    for event in trace.events:
        if type(event) is not TraceEvent:
            raise TypeError("compiled events require concrete TraceEvent records")
        decision = event.decision
        if decision is None:
            continue
        if type(decision) is not CompiledDecision:
            raise TypeError("compiled decision requires a concrete CompiledDecision")
        if type(decision.kind) is not DecisionKind:
            raise TypeError("compiled decision kind requires a concrete DecisionKind")
        if type(decision.guard) is not AffineForm:
            raise TypeError("compiled decision guard requires a concrete AffineForm")
        guard = decision.guard
        if (type(guard.spec) is not ClassSpec or type(guard.h) is not tuple
                or any(type(h) is not Fraction for h in guard.h)
                or type(guard.beta) is not Fraction):
            raise TypeError("compiled decision guard requires concrete exact fields")
        if type(decision.orientation) is not int or decision.orientation not in (-1, 0, 1):
            raise TypeError("compiled decision orientation requires a concrete signed integer")
        if decision.margin is not None and type(decision.margin) is not _DecimalInterval:
            raise TypeError("compiled decision margin requires a concrete interval")
        if decision.base_result is not None and type(decision.base_result) is not _BaseSignResult:
            raise TypeError("compiled decision base_result requires a concrete base result")
        if decision.kind is DecisionKind.VARIABLE:
            if decision.orientation == 0 or decision.margin is None or decision.base_result is None:
                raise ValueError("compiled variable decision lacks strict base evidence")
        elif decision.kind is DecisionKind.STRUCTURAL:
            if decision.orientation != 0 or decision.margin is None or decision.base_result is not None:
                raise ValueError("compiled structural decision has invalid evidence fields")
        elif decision.margin is not None or decision.base_result is not None:
            raise ValueError("compiled static decision has invalid evidence fields")


@dataclass(frozen=True, init=False)
class SegmentGuardEvidence(_FactoryOnly):
    """Outward sufficient-bound evidence for one independently covered guard."""

    event_id: tuple[object, ...]
    margin: _DecimalInterval
    movement_upper: Decimal
    slack_lower: Decimal
    precision: int


@dataclass(frozen=True, init=False)
class RealSegmentProof(_FactoryOnly):
    """An ideal exact-real directional segment only; never permission to reuse.

    The segment is w(t) = base + t*(target-base), 0 <= t <= 1. Failure of
    its conservative sufficient bound says nothing about an actual branch
    change. Same-point hashing and floating runtime audits are separate work.
    """

    status: str
    reason: str | None
    context_digest: str
    compiled_payload_digest: str
    base_weights: tuple[Fraction, ...]
    target_weights: tuple[Fraction, ...]
    parameter_box: tuple[tuple[Fraction, Fraction], ...]
    score_change_signs: tuple[int, ...]
    endpoint_changes: tuple[_DecimalInterval, ...]
    envelope_upper: Decimal | None
    ratio_upper: Decimal | None
    guards: tuple[SegmentGuardEvidence, ...]
    limiting_guard: tuple[object, ...] | None
    precisions: tuple[int, ...]
    float_runtime_proved: bool
    cache_safe: bool


def _segment_envelope(oracle: BaseSignOracle, target: tuple[Fraction, ...], precision: int):
    arithmetic = _BaseArithmetic(precision)
    log_two = arithmetic.log_two()
    changes, signs = [], []
    direction = tuple(w1 - w0 for w0, w1 in zip(oracle.base_weights, target))
    for row in oracle.feature_rows:
        score0 = sum((z * w for z, w in zip(row, oracle.base_weights)), Fraction(0))
        score_change = sum((z * d for z, d in zip(row, direction)), Fraction(0))
        side = _sign(score_change)
        signs.append(side)
        if side == 0:
            # Exact score identity over this ray, not a numerical-zero exemption
            # for a guard: all nonstructural base margins are still checked.
            changes.append(arithmetic.lift(0))
            continue
        delta = arithmetic.mul(log_two, arithmetic.sub(
            arithmetic.tanh(arithmetic.lift(score0 + score_change)),
            arithmetic.tanh(arithmetic.lift(score0))))
        changes.append(_DecimalInterval(max(Decimal(0), delta.lower), delta.upper)
                       if side > 0 else
                       _DecimalInterval(delta.lower, min(Decimal(0), delta.upper)))
    upper = max(Decimal(0), *(delta.upper for delta in changes))
    lower = min(Decimal(0), *(delta.lower for delta in changes))
    envelope = arithmetic.up.subtract(upper, lower)
    ratio = arithmetic.tanh(arithmetic.div(_DecimalInterval(envelope, envelope),
                                          arithmetic.lift(4))).upper
    return tuple(changes), tuple(signs), envelope, ratio


def _segment_guard_evidence(event: TraceEvent, oracle: BaseSignOracle,
                            ratio_upper: Decimal, precision: int) -> SegmentGuardEvidence:
    arithmetic = _BaseArithmetic(precision)
    guard = event.decision.guard
    # Orientation was proved independently by the compiler, never taken from
    # the runtime witness. Re-enclose that oriented exact guard at this precision.
    margin = oracle._enclose(guard, precision)
    oscillation = max(guard.h) - min(guard.h)
    movement = arithmetic.mul(arithmetic.lift(oracle.spec.budget * oscillation),
                              _DecimalInterval(ratio_upper, ratio_upper)).upper
    return _factory_create(SegmentGuardEvidence, event_id=event.event_id, margin=margin,
                           movement_upper=movement,
                           slack_lower=arithmetic.down.subtract(margin.lower, movement),
                           precision=precision)


def certify_real_segment(compiled: CompiledTrace, target_weights: Sequence[float],
                         parameter_box: Sequence[Sequence[float]]) -> RealSegmentProof:
    """Prove a proposed segment from a concrete live compiled trace and 4D box.

    Invalid/tampered inputs raise; an out-of-box endpoint, same point, failed
    numerical enclosure or unproved strict bound returns explicit abstention.
    No caller-supplied guard, orientation, margin or safety flag is accepted.
    """
    if type(compiled) is not CompiledTrace:
        raise TypeError("segment proof requires a concrete CompiledTrace")
    binding = _require_bound(compiled.bound)
    _require_compiled_decision_types(compiled)
    if _compiled_payload_digest(compiled) != compiled.payload_digest:
        raise ValueError("compiled proof payload context mismatch")
    CompiledTrace.validate_coverage(compiled)
    surface = compiled.surface
    if type(surface) is not ResidualClassSurface or surface.context_digest != binding.context_digest:
        raise ValueError("compiled surface context mismatch")
    target = tuple(_exact_fraction(_float_from_bits(v)) for v in _freeze_weights(target_weights))
    try:
        box = tuple(tuple(_exact_fraction(value) for value in pair) for pair in parameter_box)
    except TypeError as exc:
        raise TypeError("parameter box requires four finite endpoint pairs") from exc
    if len(box) != 4 or any(len(pair) != 2 or pair[0] > pair[1] for pair in box):
        raise ValueError("parameter box requires four ordered closed endpoint pairs")
    oracle = BaseSignOracle(surface.spec, surface.anchor_logits,
                            surface.feature_rows, surface.base_weights)
    attempts, evidence = [], ()
    changes, signs, envelope, ratio = (), (), None, None

    def result(status, reason=None):
        limiting = min(evidence, key=lambda g: g.slack_lower).event_id if evidence else None
        return _factory_create(
            RealSegmentProof, status=status, reason=reason, context_digest=binding.context_digest,
            compiled_payload_digest=compiled.payload_digest, base_weights=surface.base_weights,
            target_weights=target, parameter_box=box, score_change_signs=signs,
            endpoint_changes=changes, envelope_upper=envelope, ratio_upper=ratio,
            guards=evidence, limiting_guard=limiting, precisions=tuple(attempts),
            float_runtime_proved=False, cache_safe=False,
        )

    if any(not lo <= w <= hi for w, (lo, hi) in zip(surface.base_weights, box)):
        return result("uncertified", "base_outside_box")
    if any(not lo <= w <= hi for w, (lo, hi) in zip(target, box)):
        return result("uncertified", "target_outside_box")
    if target == surface.base_weights:
        return result("uncertified", "same_point")
    variable = tuple(e for e in compiled.events
                     if e.decision is not None and e.decision.kind is DecisionKind.VARIABLE)
    for precision in (80, 160, 320, 640):
        attempts.append(precision)
        try:
            changes, signs, envelope, ratio = _segment_envelope(oracle, target, precision)
            # Equal forms share one numerical enclosure but retain every event's
            # independently opened obligation and its own evidence record.
            by_guard = {}
            current = []
            for event in variable:
                guard = event.decision.guard
                entry = by_guard.get(guard)
                if entry is None:
                    entry = _segment_guard_evidence(event, oracle, ratio, precision)
                    by_guard[guard] = entry
                current.append(_factory_create(
                    SegmentGuardEvidence, event_id=event.event_id, margin=entry.margin,
                    movement_upper=entry.movement_upper, slack_lower=entry.slack_lower,
                    precision=precision))
            evidence = tuple(current)
        except (DecimalException, ArithmeticError) as exc:
            return result("uncertified", f"arithmetic_failure:{type(exc).__name__}")
        if all(entry.margin.lower > entry.movement_upper for entry in evidence):
            return result("proved_real")
    return result("uncertified", "strict_margin_not_proved")


@dataclass(frozen=True, init=False)
class BaseFloatError(_FactoryOnly):
    status: str
    reason: str | None
    runtime_shares: tuple[Fraction, ...]
    runtime_sum: Fraction
    mass_defect: Fraction
    normalization_recipient: int
    normalization_branch: str
    normalization_adjustment: Fraction
    ideal_per_voter: tuple[_DecimalInterval, ...]
    l1_error_upper: Decimal | None
    precisions: tuple[int, ...]


@dataclass(frozen=True, init=False)
class FloatProbe(_FactoryOnly):
    requested_parameter: Fraction
    requested_weights: tuple[Fraction, ...]
    actual_weights: tuple[Fraction, ...]
    actual_weight_bits: tuple[str, ...]
    requested_equals_actual: bool
    actual_ray_parameter: Fraction | None
    covered_by_real_proof: bool
    covered_interior: bool
    status: str
    reason: str | None
    three_kernel_match: bool | None
    projection_matches: bool | None
    trace_validated: bool
    trace_payload_sha256: str | None
    context_digest: str | None
    winners: tuple[str, ...] | None
    charge_side_observations: tuple[tuple[object, ...], ...] | None


@dataclass(frozen=True, init=False)
class FloatSegmentAudit(_FactoryOnly):
    """Sampled runtime outcome evidence, never a floating-operation proof."""

    status: str
    reason: str | None
    completion: bool
    real_proof: RealSegmentProof
    requested_target: tuple[Fraction, ...]
    base: BaseFloatError
    base_projection: RuntimeAuditProjection
    probes: tuple[FloatProbe, ...]
    new_three_path_cases: int
    covered_interior_count: int
    mismatch_count: int | None
    route_failure: bool
    compared_events: tuple[str, ...]
    omitted_micro_events: tuple[str, ...]
    production_difference: str
    full_trace_identity: bool
    float_runtime_proved: bool
    cache_safe: bool


def _bound_policy_inputs(binding: TraceBinding):
    snapshot = (binding.metadata, binding.budget_bits, binding.projects, binding.votes)
    inst = _materialize_instance(snapshot, binding.instance_semantics_sha256)
    family, logits, reference, _ = binding.anchor
    anchor = residual.AnchorSpec(family, tuple(_float_from_bits(v) for v in logits), reference)
    names, means, scales, clip, rows, instances, _ = binding.scaler
    scaler = residual.FeatureScaler(names, tuple(_float_from_bits(v) for v in means),
                                   tuple(_float_from_bits(v) for v in scales),
                                   _float_from_bits(clip), rows, instances)
    return inst, anchor, scaler, _materialize_state(binding.rollout_state)


def _base_float_error(compiled: CompiledTrace) -> BaseFloatError:
    binding, surface = compiled.bound.binding, compiled.surface
    shares = tuple(_exact_fraction(_float_from_bits(v)) for v in binding.endowment_bits)
    budget = _float_from_bits(binding.budget_bits)
    # Reproduce the frozen runtime normalization to identify its actual residue
    # recipient, including ties introduced by adding the common log scale.
    weights = np.asarray([float(w) for w in surface.base_weights])
    class_logs = [float(a) + residual._residual_log_factor(float(np.dot(weights, [float(z) for z in row])))
                  for a, row in zip(surface.anchor_logits, surface.feature_rows)]
    logs = np.asarray([class_logs[c] for c in surface.voter_classes])
    if np.all(logs == logs[0]):
        recipient, branch = 0, "all_equal"
        provisional = budget / len(shares)
    else:
        log_shares = logs + (math.log(budget) - residual._logsumexp(logs))
        recipient, branch = int(np.argmax(log_shares)), "log_softmax"
        provisional = residual._checked_exp(float(log_shares[recipient]))
    reproduced = residual._final_shares_from_logs(logs, budget)
    if tuple(_float_bits(v, "reproduced runtime share") for v in reproduced) != binding.endowment_bits:
        raise ValueError("runtime normalization reproduction differs from bound shares")
    exact_sum = sum(shares, Fraction(0))
    oracle = BaseSignOracle(surface.spec, surface.anchor_logits, surface.feature_rows, surface.base_weights)
    attempts, intervals, error, reason = [], (), None, None
    for precision in (80, 160, 320, 640):
        attempts.append(precision)
        try:
            arithmetic = _BaseArithmetic(precision)
            by_class = []
            for c, count in enumerate(surface.spec.multiplicities):
                form = AffineForm(surface.spec, tuple(Fraction(1, count) if j == c else 0
                                                     for j in range(len(surface.spec.class_order))))
                by_class.append(oracle._enclose(form, precision))
            intervals = tuple(by_class[c] for c in surface.voter_classes)
            error = Decimal(0)
            for share, ideal in zip(shares, intervals):
                delta = arithmetic.sub(arithmetic.lift(share), ideal)
                error = arithmetic.up.add(error, max(delta.lower.copy_abs(), delta.upper.copy_abs()))
            break
        except (ArithmeticError, DecimalException) as exc:
            reason = f"arithmetic_failure:{type(exc).__name__}"
            intervals, error = (), None
            break
    return _factory_create(
        BaseFloatError, status="enclosed" if error is not None else "uncertified", reason=reason,
        runtime_shares=shares, runtime_sum=exact_sum, mass_defect=exact_sum - surface.spec.budget,
        normalization_recipient=recipient, normalization_branch=branch,
        normalization_adjustment=shares[recipient] - _exact_fraction(provisional),
        ideal_per_voter=intervals, l1_error_upper=error, precisions=tuple(attempts))


def _exact_ray_parameter(base, target, point) -> Fraction | None:
    parameter = None
    for w0, w1, actual in zip(base, target, point):
        if w1 == w0:
            if actual != w0:
                return None
        else:
            value = (actual - w0) / (w1 - w0)
            if parameter is not None and value != parameter:
                return None
            parameter = value
    return parameter


def _runtime_outcome_projection(trace: actuation.EqualSharesTrace):
    removed, current = [], set()
    candidates = {}
    for step in trace.steps:
        if step.phase == "payment-candidate" and step.rho is None:
            current.add(step.project)
        elif step.phase == "payment-charge":
            removed.append(tuple(sorted(current)))
            current = set()
        elif step.phase == "completion-candidate":
            candidates[step.project] = dict(step.comparison_margins)
    if current:
        removed.append(tuple(sorted(current)))
    completion = []
    for step in trace.steps:
        if step.phase != "completion-order":
            continue
        margins = dict(step.comparison_margins)
        if step.project in trace.payment_order:
            outcome = "winner"
        elif margins["approval_count"] == 0:
            outcome = "zero_support"
        else:
            decision = candidates[step.project]
            outcome = ("zero_cost" if decision["positive_cost_result"] == "reject" else
                       "fund" if decision["affordability_result"] == "fund" else "unaffordable")
        completion.append((step.project, outcome))
    return (tuple(trace.payment_order), tuple(removed), tuple(completion), tuple(trace.winners))


def _charge_side_observations(compiled, trace):
    """Diagnostic comparisons only for the base audit's matching charge subset."""
    charges = [step for step in trace.steps if step.phase == "payment-charge"]
    observations = []
    for round_index, pid, voter, ideal_side in compiled.audit.charge_side_matches:
        side = None
        if round_index < len(charges) and charges[round_index].project == pid:
            step = charges[round_index]
            before = dict(step.comparison_margins).get(f"approver:{voter}:budget_before")
            if before is not None:
                side = (before > step.rho) - (before < step.rho)
        observations.append((round_index, pid, voter, ideal_side, side))
    return tuple(observations)


def audit_float_segment(compiled: CompiledTrace, target_weights: Sequence[float],
                        parameter_box: Sequence[Sequence[float]], *,
                        requested_target: Sequence[object] | None = None,
                        probe_parameters: Sequence[object] = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)),
                        include_endpoint: bool = False) -> FloatSegmentAudit:
    """Audit deterministic dyadic probes; rounded points carry exact coverage labels.

    Each invocation covers its bound completion mode. The P0 runner requires
    both modes. Runtime witness validation and three-kernel winner agreement do
    not establish literal trace identity or enclose every floating operation.
    """
    proof = certify_real_segment(compiled, target_weights, parameter_box)
    requested = proof.target_weights if requested_target is None else tuple(_exact_fraction(v) for v in requested_target)
    if len(requested) != 4:
        raise ValueError("requested target must have four finite exact coordinates")
    parameters = tuple(_exact_fraction(v) for v in probe_parameters)
    if (not parameters or len(set(parameters)) != len(parameters)
            or any(not 0 < t < 1 or t.denominator & (t.denominator - 1) for t in parameters)):
        raise ValueError("probe parameters must be unique dyadic strict interiors")
    if type(include_endpoint) is not bool:
        raise TypeError("include_endpoint must be boolean")
    if include_endpoint:
        parameters += (Fraction(1),)
    binding = compiled.bound.binding
    base_error = _base_float_error(compiled)
    probes, cases, mismatch_count, failure = [], 0, None, None
    route_failure = False

    def finish():
        return _factory_create(
            FloatSegmentAudit, status="sampled_parity" if failure is None else "uncertified",
            reason=failure, completion=binding.completion, real_proof=proof, requested_target=requested,
            base=base_error, base_projection=compiled.audit, probes=tuple(probes), new_three_path_cases=cases,
            covered_interior_count=sum(p.covered_interior for p in probes), mismatch_count=mismatch_count,
            route_failure=route_failure,
            compared_events=("three_kernel_winners", "payment_winner_sequence", "removed_project_sets_by_round",
                             "completion_decisions_in_order", "final_winners", "base_matched_charge_sides_diagnostic"),
            omitted_micro_events=("candidate_scan_order", "payer_order", "breakpoint_order", "candidate_rho_order",
                                  "charge_sides", "continuous_prices_charges_balances"),
            production_difference=compiled.audit.production_difference, full_trace_identity=False,
            float_runtime_proved=False, cache_safe=False)

    if base_error.status != "enclosed":
        failure = base_error.reason
        return finish()
    baseline = _runtime_outcome_projection(compiled.bound.trace)
    inst, anchor, scaler, state = _bound_policy_inputs(binding)
    mismatch_count = 0
    for t in parameters:
        exact_point = tuple(a + t * (b - a) for a, b in zip(proof.base_weights, requested))
        bits = _freeze_weights(tuple(float(v) for v in exact_point))
        actual = tuple(_exact_fraction(_float_from_bits(v)) for v in bits)
        ray_t = _exact_ray_parameter(proof.base_weights, proof.target_weights, actual)
        covered = proof.status == "proved_real" and ray_t is not None and 0 <= ray_t <= 1
        interior = covered and 0 < ray_t < 1
        match = projection = None
        validated = False
        payload_digest = context_digest = winners = reason = charge_observations = None
        try:
            bound = compile_bound_trace(inst, anchor, scaler, tuple(_float_from_bits(v) for v in bits),
                                        state, completion=binding.completion)
            if type(bound) is BoundTrace:
                _require_bound(bound)
                cases += 1
                match, validated = True, True
                projection = _runtime_outcome_projection(bound.trace) == baseline
                payload_digest = _canonical_sha256(bound.binding.trace_payload)
                context_digest, winners = bound.binding.context_digest, tuple(bound.trace.winners)
                charge_observations = _charge_side_observations(compiled, bound.trace)
                if not projection:
                    reason = "base_outcome_projection_changed"
            else:
                reason = bound.reason
                if reason == "three-kernel runtime winner parity could not be established":
                    cases += 1
                    match = False
        except (ArithmeticError, DecimalException, ValueError) as exc:
            reason = f"runtime_probe_failure:{type(exc).__name__}:{exc}"
        mismatch = match is False or projection is False
        # Projection changes outside proved interiors are observations, not a
        # theorem violation. Three-kernel disagreements always fail the audit.
        failed = match is not True or (covered and projection is False)
        if mismatch:
            mismatch_count += 1
        elif match is None:
            mismatch_count = None
        probes.append(_factory_create(
            FloatProbe, requested_parameter=t, requested_weights=exact_point, actual_weights=actual,
            actual_weight_bits=bits, requested_equals_actual=actual == exact_point,
            actual_ray_parameter=ray_t, covered_by_real_proof=covered, covered_interior=interior,
            status="uncertified" if failed else "sampled_parity", reason=reason,
            three_kernel_match=match, projection_matches=projection, trace_validated=validated,
            trace_payload_sha256=payload_digest, context_digest=context_digest, winners=winners,
            charge_side_observations=charge_observations))
        if failed:
            failure = reason or "probe_unverified"
            route_failure = covered and mismatch
            break
    return finish()


class _ReplayAbstention(Exception):
    pass


def _sign(value):
    return (value > 0) - (value < 0)


class _SymbolicReplay:
    def __init__(self, bound, surface):
        self.bound, self.surface = bound, surface
        self.spec = surface.spec
        self.oracle = BaseSignOracle(self.spec, surface.anchor_logits,
                                     surface.feature_rows, surface.base_weights)
        self.ledger = _CoverageLedger()
        self.events = []
        self.decisions = {}
        self.costs = {row[1]: _exact_fraction(_float_from_bits(row[3]))
                      for row in bound.binding.projects}
        self.supporters = {pid: [] for pid in self.costs}
        for row in bound.binding.votes:
            for pid in row[2]:
                self.supporters[pid].append(row[0])
        self.balances = list(surface.initial_balances)
        self.histories = [[] for _ in self.balances]
        self.payment, self.completion = [], []
        self.removed, self.completion_decisions = [], []
        self.payer_orders, self.charge_sides = {}, {}
        self.runtime_candidates, self.runtime_charges = {}, {}
        self.runtime_removed = []
        current_removed, round_index = set(), 0
        for step in bound.trace.steps:
            if step.phase == "payment-candidate":
                self.runtime_candidates[(round_index, step.project)] = step
                if step.rho is None:
                    current_removed.add(step.project)
            elif step.phase == "payment-charge":
                self.runtime_charges[round_index] = step
                self.runtime_removed.append(frozenset(current_removed))
                current_removed = set()
                round_index += 1
        if current_removed:
            self.runtime_removed.append(frozenset(current_removed))

    def constant(self, value):
        return AffineForm(self.spec, (0,) * len(self.spec.class_order), value)

    def event(self, obligation: BranchObligation, outcome, payload=()):
        # Consumption only: the control-flow driver has already registered the
        # expectation, even if this helper is accidentally omitted or skipped.
        self.events.append(self.ledger.emit(obligation, outcome, payload))

    def affine(self, obligation: BranchObligation, form):
        _, site, location = obligation.event_id
        # Cache exact same guards only; no numerical or demographic equivalence.
        canonical = form.canonical()
        decision = self.decisions.get(canonical)
        if decision is None:
            decision = _compile_affine_decision(canonical, self.oracle)
            self.decisions[canonical] = decision
        side = decision.orientation
        if side is None:
            raise _ReplayAbstention(f"{site}:{location}:{decision.base_result.reason}")
        self.events.append(self.ledger.emit(obligation, _SIGNS[side + 1], (), decision))
        return side

    def static(self, obligation: BranchObligation, left, right, *, runtime_left=None, runtime_right=None):
        # No public static factory. Only enumerated exact problem-data rules
        # enter here; every one audits its binary64 counterpart as well.
        _, site, location = obligation.event_id
        if site in ("candidate_cost", "candidate_id"):
            round_index, pid, incumbent = location
            step = self.runtime_candidates.get((round_index, pid))
            margins = {} if step is None else dict(step.comparison_margins)
            candidate_key, incumbent_key = margins.get("candidate_key"), margins.get("incumbent_key")
            if (candidate_key is None or incumbent_key is None or incumbent_key[2] != incumbent
                    or candidate_key[0] != incumbent_key[0]
                    or (site == "candidate_id" and candidate_key[1] != incumbent_key[1])):
                raise _ReplayAbstention(f"static_runtime_disagreement:{site}:{location}")
        if site in ("candidate_id", "completion_id"):
            side = (left > right) - (left < right)
            guard = self.constant(side)
            runtime_side = side
        else:
            difference = _exact_fraction(left) - _exact_fraction(right)
            side = _sign(difference)
            guard = self.constant(difference)
            a = float(left) if runtime_left is None else runtime_left
            b = float(right) if runtime_right is None else runtime_right
            runtime_side = (a > b) - (a < b)
        if side != runtime_side:
            raise _ReplayAbstention(f"static_runtime_disagreement:{site}:{location}")
        decision = _factory_create(CompiledDecision, kind=DecisionKind.STATIC,
                                   guard=guard, orientation=side, margin=None, base_result=None)
        self.events.append(self.ledger.emit(obligation, _SIGNS[side + 1], (left, right), decision))
        return side

    def run(self):
        # Each control site registers its required occurrence here, separately
        # from all emission/decision helpers. Never move these opens into them:
        # their omission must leave an outstanding obligation at finalization.
        active = set()
        for pid, cost in self.costs.items():
            obligation = self.ledger.open("initial_support", (pid,))
            if self.static(obligation, len(self.supporters[pid]), 0) > 0:
                obligation = self.ledger.open("initial_cost", (pid,))
                if self.static(obligation, cost, 0) > 0:
                    active.add(pid)
        obligation = self.ledger.open("initial_active", ())
        self.event(obligation, "snapshot", tuple(sorted(active)))
        round_index = 0
        while active:
            obligation = self.ledger.open("active_before", (round_index,))
            self.event(obligation, "snapshot", tuple(sorted(active)))
            best_pid = best_rho = None
            removed = set()
            for pid in sorted(active):
                loc = (round_index, pid)
                obligation = self.ledger.open("candidate_visit", loc)
                self.event(obligation, "visit")
                cost, supporters = self.costs[pid], self.supporters[pid]
                total = sum((self.balances[v] for v in supporters), self.constant(0))
                obligation = self.ledger.open("affordability", loc)
                if self.affine(obligation, total - self.constant(cost - _exact_fraction(1e-9))) < 0:
                    obligation = self.ledger.open("active_before", (*loc, "remove"))
                    self.event(obligation, "snapshot", tuple(sorted(active)))
                    obligation = self.ledger.open("removal", loc)
                    self.event(obligation, "unaffordable")
                    active.remove(pid)
                    obligation = self.ledger.open("active_after", (*loc, "remove"))
                    self.event(obligation, "snapshot", tuple(sorted(active)))
                    removed.add(pid)
                    continue

                def compare_voters(left, right):
                    obligation = self.ledger.open("payer_order", (*loc, left, right))
                    side = self.affine(obligation,
                                       self.balances[left] - self.balances[right])
                    return side or ((left > right) - (left < right))

                payers = sorted(supporters, key=cmp_to_key(compare_voters))
                self.payer_orders[loc] = tuple(payers)
                paid, rho = self.constant(0), None
                for position, voter in enumerate(payers):
                    candidate = (self.constant(cost) - paid) / (len(payers) - position)
                    obligation = self.ledger.open("breakpoint", (*loc, position, voter))
                    if self.affine(obligation,
                                   self.balances[voter] + self.constant(_exact_fraction(1e-12)) - candidate) >= 0:
                        rho = candidate
                        break
                    paid = paid + self.balances[voter]
                if rho is None:
                    obligation = self.ledger.open("active_before", (*loc, "remove"))
                    self.event(obligation, "snapshot", tuple(sorted(active)))
                    obligation = self.ledger.open("removal", loc)
                    self.event(obligation, "no_rho")
                    active.remove(pid)
                    obligation = self.ledger.open("active_after", (*loc, "remove"))
                    self.event(obligation, "snapshot", tuple(sorted(active)))
                    removed.add(pid)
                    continue
                if best_pid is None:
                    obligation = self.ledger.open("incumbent", loc)
                    self.event(obligation, "initialize", (pid,))
                    best_pid, best_rho = pid, rho
                    continue
                obligation = self.ledger.open("rho_order", (*loc, best_pid))
                side = self.affine(obligation, rho - best_rho)
                if side == 0:
                    obligation = self.ledger.open("candidate_cost", (*loc, best_pid))
                    side = self.static(obligation, cost, self.costs[best_pid])
                    if side == 0:
                        obligation = self.ledger.open("candidate_id", (*loc, best_pid))
                        side = self.static(obligation, pid, best_pid)
                obligation = self.ledger.open("incumbent", loc)
                self.event(obligation, "replace" if side < 0 else "retain", (best_pid, pid))
                if side < 0:
                    best_pid, best_rho = pid, rho
            self.removed.append(frozenset(removed))
            if best_pid is None:
                obligation = self.ledger.open("active_after", (round_index,))
                self.event(obligation, "snapshot", tuple(sorted(active)))
                obligation = self.ledger.open("termination", (round_index,))
                self.event(obligation, "no_candidate")
                break
            obligation = self.ledger.open("active_before", (round_index, best_pid, "choose"))
            self.event(obligation, "snapshot", tuple(sorted(active)))
            obligation = self.ledger.open("chosen", (round_index, best_pid))
            self.event(obligation, "payment", (best_rho,))
            for voter in self.supporters[best_pid]:
                obligation = self.ledger.open("charge", (round_index, best_pid, voter))
                side = self.affine(obligation,
                                   self.balances[voter] - best_rho)
                charge = self.balances[voter] if side <= 0 else best_rho
                self.charge_sides[(round_index, best_pid, voter)] = side
                self.balances[voter] = self.balances[voter] - charge
                self.histories[voter].append((best_pid, charge))
            self.payment.append(best_pid)
            active.remove(best_pid)
            obligation = self.ledger.open("active_after", (round_index,))
            self.event(obligation, "snapshot", tuple(sorted(active)))
            round_index += 1
        else:
            obligation = self.ledger.open("termination", (round_index,))
            self.event(obligation, "empty_active")
        obligation = self.ledger.open("completion_mode", ())
        self.event(obligation, "enabled" if self.bound.binding.completion else "disabled")
        if self.bound.binding.completion:
            self.complete()
        winners = tuple(sorted((*self.payment, *self.completion)))
        obligation = self.ledger.open("final_outcome", ())
        self.event(obligation, "winners", winners)
        self.ledger.validate(self.events)
        audit = self.audit(winners)
        compiled = _factory_create(
            CompiledTrace, bound=self.bound, surface=self.surface, events=tuple(self.events),
            obligations=tuple(self.ledger._opened), payment_order=tuple(self.payment),
            completion_order=tuple(self.completion), winners=winners,
            final_balances=tuple(self.balances), histories=tuple(tuple(h) for h in self.histories),
            audit=audit, float_runtime_proved=False, cache_safe=False, payload_digest="",
        )
        object.__setattr__(compiled, "payload_digest", _compiled_payload_digest(compiled))
        return compiled

    def complete(self):
        def compare_projects(left, right):
            obligation = self.ledger.open("completion_count", (left, right))
            side = self.static(obligation,
                               -len(self.supporters[left]), -len(self.supporters[right]))
            if side == 0:
                obligation = self.ledger.open("completion_cost", (left, right))
                side = self.static(obligation, self.costs[left], self.costs[right])
                if side == 0:
                    obligation = self.ledger.open("completion_id", (left, right))
                    side = self.static(obligation, left, right)
            return side

        ordered = sorted(self.costs, key=cmp_to_key(compare_projects))
        runtime_order = tuple(step.project for step in self.bound.trace.steps if step.phase == "completion-order")
        if tuple(ordered) != runtime_order:
            raise _ReplayAbstention("static_runtime_disagreement:completion_order")
        runtime_candidates = {step.project: step for step in self.bound.trace.steps
                              if step.phase == "completion-candidate"}
        remaining = self.spec.budget - sum((self.costs[pid] for pid in self.payment), Fraction(0))
        obligation = self.ledger.open("completion_remaining", ())
        self.event(obligation, "initial", (remaining,))
        winners = set(self.payment)
        for pid in ordered:
            obligation = self.ledger.open("completion_visit", (pid,))
            self.event(obligation, "visit")
            if pid in winners:
                obligation = self.ledger.open("completion_skip", (pid,))
                self.event(obligation, "winner")
                self.completion_decisions.append((pid, "winner"))
                continue
            if not self.supporters[pid]:
                obligation = self.ledger.open("completion_skip", (pid,))
                self.event(obligation, "zero_support")
                self.completion_decisions.append((pid, "zero_support"))
                continue
            cost = self.costs[pid]
            obligation = self.ledger.open("completion_positive", (pid,))
            if self.static(obligation, cost, 0) <= 0:
                outcome = "zero_cost"
            else:
                runtime_step = runtime_candidates.get(pid)
                if runtime_step is None:
                    raise _ReplayAbstention(f"runtime_completion_candidate_missing:{pid}")
                obligation = self.ledger.open("completion_affordability", (pid,))
                side = self.static(obligation, remaining, cost,
                                   runtime_left=runtime_step.remaining_budget)
                outcome = "fund" if side >= 0 else "unaffordable"
            obligation = self.ledger.open("completion_decision", (pid,))
            self.event(obligation, outcome)
            self.completion_decisions.append((pid, outcome))
            if outcome == "fund":
                previous = remaining
                remaining -= cost
                obligation = self.ledger.open("completion_remaining", (pid,))
                self.event(obligation, "transition", (previous, remaining))
                self.completion.append(pid)
                winners.add(pid)

    def audit(self, winners):
        trace = self.bound.trace
        payment_matches = tuple(self.payment) == tuple(trace.payment_order)
        removed_matches = tuple(self.removed) == tuple(self.runtime_removed)
        runtime_completion = []
        payment_winners = set(trace.payment_order)
        candidates = {step.project: dict(step.comparison_margins) for step in trace.steps
                      if step.phase == "completion-candidate"}
        for step in trace.steps:
            if step.phase != "completion-order":
                continue
            pid = step.project
            if pid in payment_winners:
                outcome = "winner"
            elif not self.supporters[pid]:
                outcome = "zero_support"
            else:
                margins = candidates[pid]
                outcome = ("zero_cost" if margins["positive_cost_result"] == "reject" else
                           "fund" if margins["affordability_result"] == "fund" else "unaffordable")
            runtime_completion.append((pid, outcome))
        completion_matches = tuple(self.completion_decisions) == tuple(runtime_completion)
        winners_match = winners == tuple(sorted(trace.winners))
        if not all((payment_matches, removed_matches, completion_matches, winners_match)):
            raise _ReplayAbstention(
                f"runtime_projection_disagreement:payment={payment_matches},removed={removed_matches},"
                f"completion={completion_matches},winners={winners_match}")
        omitted = [("candidate_rho_order", *event.event_id[2]) for event in self.events
                   if event.site == "rho_order"]
        for loc, order in self.payer_orders.items():
            margins = dict(self.runtime_candidates[loc].comparison_margins)
            runtime_order = tuple(margins[f"payer:{i}:identity"][0] for i in range(len(order)))
            # Breakpoint indices follow payer micro-order, so no full identity
            # assertion is made even when the resulting chosen project agrees.
            omitted.append(("payer_order_and_breakpoint", *loc,
                            "same_order" if order == runtime_order else "nonmatching_order"))
        matched = []
        for (round_index, pid, voter), side in self.charge_sides.items():
            step = self.runtime_charges[round_index]
            margins = dict(step.comparison_margins)
            before = margins[f"approver:{voter}:budget_before"]
            runtime_side = (before > step.rho) - (before < step.rho)
            if side == runtime_side:
                matched.append((round_index, pid, voter, side))
            else:
                omitted.append(("charge_side", round_index, pid, voter, side, runtime_side))
        return _factory_create(
            RuntimeAuditProjection, payment_order_matches=payment_matches,
            compared_events=("payment_winner_sequence", "removed_project_sets_by_round",
                             "completion_decisions_in_order", "final_winners", "matching_charge_sides",
                             "static_comparison_resolutions"),
            removed_project_sets_match=removed_matches, completion_decisions_match=completion_matches,
            final_winners_match=winners_match, matched_charge_sides=len(matched),
            charge_side_matches=tuple(matched),
            omitted_micro_events=tuple(omitted),
            production_difference="production <1e-12 balance clamp is audit-only; absent from exact model",
            full_trace_identity=False,
        )


def compile_signed_trace(bound: BoundTrace) -> CompiledTrace | UncertifiedTrace:
    """Compile all exact-real MES branches from a process-valid bound context."""
    binding = _require_bound(bound)
    surface = ResidualClassSurface.from_bound(bound)
    try:
        return _SymbolicReplay(bound, surface).run()
    except _ReplayAbstention as exc:
        return _factory_create(
            UncertifiedTrace, certified=False, reason=str(exc), context_digest=binding.context_digest,
            instance_semantics_sha256=binding.instance_semantics_sha256,
            endowment_bits=binding.endowment_bits, runtime=binding.runtime,
            source_role_sha256=binding.source_role_sha256,
            input_file_sha256=binding.input_file_sha256, float_runtime_proved=False, cache_safe=False,
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stat_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )


def _read_regular_nofollow(path: Path, label: str) -> bytes:
    """Read one stable regular file through a no-follow descriptor chain."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts[1:]):
        raise RuntimeError(f"{label} path is not a canonical file path")
    descriptors: list[int] = []
    try:
        try:
            directory = os.open(absolute.anchor, _DIRECTORY_FLAGS)
        except OSError as exc:
            raise RuntimeError(f"{label} has an unsafe path root") from exc
        descriptors.append(directory)
        for component in parts[1:-1]:
            try:
                identity = os.stat(component, dir_fd=directory, follow_symlinks=False)
            except OSError as exc:
                raise RuntimeError(f"{label} has an unsafe or missing ancestor") from exc
            if stat.S_ISLNK(identity.st_mode) or not stat.S_ISDIR(identity.st_mode):
                raise RuntimeError(f"{label} has a symlink or non-directory ancestor")
            try:
                child = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory)
            except OSError as exc:
                raise RuntimeError(f"{label} has an unstable ancestor") from exc
            descriptors.append(child)
            directory = child

        name = parts[-1]
        try:
            lexical = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"{label} is missing or inaccessible") from exc
        if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
            raise RuntimeError(f"{label} must be a regular nonsymlink file")
        try:
            descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory)
        except OSError as exc:
            raise RuntimeError(f"{label} cannot be opened without following links") from exc
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (lexical.st_dev, lexical.st_ino):
            raise RuntimeError(f"{label} changed during snapshot opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise RuntimeError(f"{label} changed during snapshot reading")
        content = b"".join(chunks)
        if len(content) != before.st_size:
            raise RuntimeError(f"{label} changed during snapshot reading")
        return content
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _snapshot_source_roles() -> tuple[tuple[str, str], ...]:
    paths = tuple(_SOURCE_ROLE_PATHS)
    roles = tuple(role for role, _ in paths)
    if len(roles) != len(set(roles)) or set(roles) != _EXPECTED_SOURCE_ROLES:
        raise RuntimeError("source-role map is incomplete or duplicated")
    return tuple(
        sorted(
            (
                role,
                hashlib.sha256(
                    _read_regular_nofollow(Path(path), f"source role {role}")
                ).hexdigest(),
            )
            for role, path in paths
        )
    )


def _snapshot_inputs(
    input_paths: Mapping[str, Path] | None,
) -> tuple[tuple[str, str], ...]:
    if input_paths is None:
        return ()
    if not isinstance(input_paths, Mapping):
        raise TypeError("input_paths must be a mapping from roles to paths")
    rows: list[tuple[str, str]] = []
    for role, path in input_paths.items():
        if not isinstance(role, str) or not role:
            raise TypeError("input file roles must be nonempty strings")
        content = _read_regular_nofollow(Path(path), f"input file {role}")
        rows.append((role, hashlib.sha256(content).hexdigest()))
    return tuple(sorted(rows))


def _runtime_snapshot() -> NumericRuntime:
    return _factory_create(
        NumericRuntime,
        python_implementation=str(sys.implementation.name),
        python_version=str(sys.version),
        python_cache_tag=str(sys.implementation.cache_tag or ""),
        python_version_info=tuple(str(value) for value in sys.version_info[:5]),
        numpy_version=str(np.__version__),
        byteorder=str(sys.byteorder),
        float_mant_dig=int(sys.float_info.mant_dig),
        float_max_exp=int(sys.float_info.max_exp),
        float_rounds=int(sys.float_info.rounds),
        process_id=os.getpid(),
        process_nonce=_PROCESS_NONCE,
    )


def _runtime_payload(runtime: NumericRuntime) -> dict[str, object]:
    return {item.name: getattr(runtime, item.name) for item in fields(runtime)}


def _freeze_instance(inst: PBInstance) -> tuple[dict[str, object], tuple[object, ...]]:
    if not isinstance(inst, PBInstance):
        raise TypeError("inst must be a PBInstance")
    if not isinstance(inst.meta, Mapping):
        raise TypeError("instance metadata must be a mapping")
    metadata: list[tuple[str, str]] = []
    for key, value in inst.meta.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("metadata keys and values must be strings")
        metadata.append((key, value))
    metadata_snapshot = tuple(sorted(metadata))
    raw_budget = dict(metadata_snapshot).get("budget", "0")
    try:
        budget = float(raw_budget.replace(",", "."))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("instance budget must be finite and positive") from None
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError("instance budget must be finite and positive")
    budget_bits = _float_bits(budget, "instance budget")

    if not isinstance(inst.projects, Mapping):
        raise TypeError("instance projects must be a mapping")
    project_rows: list[tuple[object, ...]] = []
    project_ids: list[str] = []
    for index, (mapping_key, project) in enumerate(inst.projects.items()):
        if not isinstance(mapping_key, str) or not isinstance(project, Project):
            raise TypeError("project mappings require string keys and Project values")
        if not isinstance(project.pid, str):
            raise TypeError("project IDs must be strings")
        if mapping_key != project.pid:
            raise ValueError("project mapping key disagrees with embedded project ID")
        if isinstance(project.cost, bool) or not isinstance(project.cost, Real):
            raise ValueError("project cost must be finite and nonnegative")
        cost = float(project.cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("project cost must be finite and nonnegative")
        if project.selected is not None and type(project.selected) is not int:
            raise TypeError("project selected values must be integers or None")
        string_fields = (
            project.name,
            project.category,
            project.target,
            project.neighborhood,
        )
        if any(not isinstance(value, str) for value in string_fields):
            raise TypeError("project descriptive fields must be strings")
        project_ids.append(mapping_key)
        project_rows.append(
            (
                index,
                mapping_key,
                project.pid,
                _float_bits(cost, f"project {mapping_key} cost"),
                project.selected,
                *string_fields,
            )
        )

    if isinstance(inst.votes, (str, bytes)) or not isinstance(inst.votes, Sequence):
        raise TypeError("instance votes must be a sequence")
    if not inst.votes:
        raise ValueError("at least one voter is required for a bound trace")
    known_projects = set(project_ids)
    vote_rows: list[tuple[object, ...]] = []
    for index, vote in enumerate(inst.votes):
        if not isinstance(vote, Vote):
            raise TypeError("instance votes must contain Vote values")
        if not isinstance(vote.vid, str):
            raise TypeError("voter IDs must be strings")
        if isinstance(vote.projects, (str, bytes)) or not isinstance(
            vote.projects, Sequence
        ):
            raise TypeError("vote approvals must be a sequence")
        approvals = tuple(vote.projects)
        if any(not isinstance(pid, str) for pid in approvals):
            raise TypeError("approved project IDs must be strings")
        if len(approvals) != len(set(approvals)):
            raise ValueError("duplicate project approval within a ballot")
        unknown = tuple(pid for pid in approvals if pid not in known_projects)
        if unknown:
            raise ValueError(f"unknown project ID in ballot: {unknown[0]}")
        if vote.age is not None and type(vote.age) is not int:
            raise TypeError("voter age must be an integer or None")
        if not isinstance(vote.sex, str) or not isinstance(vote.neighborhood, str):
            raise TypeError("voter demographic fields must be strings")
        vote_rows.append(
            (
                index,
                vote.vid,
                approvals,
                vote.age,
                vote.sex,
                vote.neighborhood,
            )
        )

    semantic_payload = {
        "schema": _INSTANCE_SCHEMA,
        "metadata": metadata_snapshot,
        "parsed_budget_bits": budget_bits,
        "projects": tuple(project_rows),
        "votes": tuple(vote_rows),
    }
    snapshot = (
        metadata_snapshot,
        budget_bits,
        tuple(project_rows),
        tuple(vote_rows),
    )
    return semantic_payload, snapshot


def _materialize_instance(snapshot: tuple[object, ...], semantic_sha256: str) -> PBInstance:
    metadata, _, project_rows, vote_rows = snapshot
    projects = {
        row[1]: Project(
            pid=row[2],
            cost=_float_from_bits(row[3]),
            selected=row[4],
            name=row[5],
            category=row[6],
            target=row[7],
            neighborhood=row[8],
        )
        for row in project_rows
    }
    votes = [
        Vote(
            vid=row[1],
            projects=tuple(row[2]),
            age=row[3],
            sex=row[4],
            neighborhood=row[5],
        )
        for row in vote_rows
    ]
    return PBInstance(
        path=f"certificate://v1/{semantic_sha256}",
        meta=dict(metadata),
        projects=projects,
        votes=votes,
    )


def _derive_partition_snapshot(
    votes: Sequence[Vote],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    members: dict[str, list[int]] = {}
    for index, vote in enumerate(votes):
        cohort = cohorts.cohort_of(vote, "age_sex")
        label = cohort if cohort is not None else _MISSING_DEMOGRAPHIC
        members.setdefault(label, []).append(index)
    return tuple((label, tuple(members[label])) for label in sorted(members))


def _validate_partition(
    partition: object, voter_count: int
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if isinstance(partition, (str, bytes)) or not isinstance(partition, Sequence):
        raise ValueError("class partition must be a sequence")
    rows: list[tuple[str, tuple[int, ...]]] = []
    seen: set[int] = set()
    labels: set[str] = set()
    for raw in partition:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError("class partition rows require a label and members")
        label, raw_members = raw
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("class partition labels must be unique nonempty strings")
        if isinstance(raw_members, (str, bytes)) or not isinstance(
            raw_members, Sequence
        ):
            raise ValueError("class partition members must be a sequence")
        members = tuple(raw_members)
        if not members:
            raise ValueError("class partition cannot contain a zero-count class")
        for member in members:
            if type(member) is not int or not 0 <= member < voter_count:
                raise ValueError("class partition contains an invalid voter index")
            if member in seen:
                raise ValueError("class partition classes overlap")
            seen.add(member)
        labels.add(label)
        rows.append((label, members))
    if seen != set(range(voter_count)):
        raise ValueError("class partition is incomplete")
    if tuple(label for label, _ in rows) != tuple(sorted(labels)):
        raise ValueError("class partition labels are not canonical")
    return tuple(rows)


def _freeze_anchor(anchor: residual.AnchorSpec) -> tuple[tuple[object, ...], residual.AnchorSpec]:
    if not isinstance(anchor, residual.AnchorSpec):
        raise TypeError("anchor must be an AnchorSpec")
    logits = tuple(
        _float_bits(value, f"anchor logit {index}")
        for index, value in enumerate(anchor.free_logits)
    )
    if not isinstance(anchor.family, str) or (
        anchor.reference_cell is not None and not isinstance(anchor.reference_cell, str)
    ):
        raise TypeError("anchor family and reference cell must be strings")
    alpha_bits = None if anchor.alpha is None else _float_bits(anchor.alpha, "anchor alpha")
    rebuilt = residual.AnchorSpec(
        anchor.family,
        tuple(_float_from_bits(value) for value in logits),
        anchor.reference_cell,
    )
    rebuilt_alpha = None if rebuilt.alpha is None else _float_bits(rebuilt.alpha, "anchor alpha")
    if alpha_bits != rebuilt_alpha:
        raise ValueError("anchor derived fields are inconsistent")
    return (anchor.family, logits, anchor.reference_cell, alpha_bits), rebuilt


def _freeze_scaler(
    scaler: residual.FeatureScaler,
) -> tuple[tuple[object, ...], residual.FeatureScaler]:
    if not isinstance(scaler, residual.FeatureScaler):
        raise TypeError("scaler must be a FeatureScaler")
    names = tuple(scaler.feature_names)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("scaler feature names must be strings")
    mean_bits = tuple(
        _float_bits(value, f"scaler mean {index}")
        for index, value in enumerate(scaler.mean)
    )
    scale_bits = tuple(
        _float_bits(value, f"scaler scale {index}")
        for index, value in enumerate(scaler.scale)
    )
    clip_bits = _float_bits(scaler.clip, "scaler clip")
    if type(scaler.row_count) is not int or type(scaler.instance_count) is not int:
        raise TypeError("scaler counts must be integers")
    rebuilt = residual.FeatureScaler(
        names,
        tuple(_float_from_bits(value) for value in mean_bits),
        tuple(_float_from_bits(value) for value in scale_bits),
        _float_from_bits(clip_bits),
        row_count=scaler.row_count,
        instance_count=scaler.instance_count,
    )
    if rebuilt.sha256 != scaler.sha256:
        raise ValueError("scaler digest is inconsistent with scaler contents")
    snapshot = (
        names,
        mean_bits,
        scale_bits,
        clip_bits,
        scaler.row_count,
        scaler.instance_count,
        scaler.sha256,
    )
    return snapshot, rebuilt


def _freeze_weights(weights: Sequence[float]) -> tuple[str, ...]:
    if isinstance(weights, np.ndarray):
        if weights.ndim != 1:
            raise ValueError("base weights must be a finite four-value sequence")
        materialized = tuple(weights.tolist())
    elif isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence):
        raise ValueError("base weights must be a finite four-value sequence")
    else:
        materialized = tuple(weights)
    if len(materialized) != len(residual.FEATURE_NAMES):
        raise ValueError("base weights must have one value per residual feature")
    return tuple(
        _float_bits(value, f"base weights value {index}")
        for index, value in enumerate(materialized)
    )


def _freeze_state(state: RolloutState) -> tuple[object, ...]:
    if not isinstance(state, RolloutState):
        raise TypeError("state must be a RolloutState")
    if type(state.year_index) is not int or state.year_index < 0:
        raise ValueError("rollout state year must be a nonnegative integer")

    def freeze_mapping(value: object, name: str) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, Mapping):
            raise TypeError(f"rollout state {name} must be a mapping")
        rows: list[tuple[str, str]] = []
        for key, number in value.items():
            if not isinstance(key, str):
                raise TypeError(f"rollout state {name} keys must be strings")
            rows.append((key, _float_bits(number, f"rollout state {name} {key}")))
        return tuple(sorted(rows))

    return (
        freeze_mapping(state.deficits, "deficits"),
        freeze_mapping(state.entitlements, "entitlements"),
        state.year_index,
    )


def _materialize_state(snapshot: tuple[object, ...]) -> RolloutState:
    deficits, entitlements, year_index = snapshot
    return RolloutState(
        deficits={key: _float_from_bits(value) for key, value in deficits},
        entitlements={key: _float_from_bits(value) for key, value in entitlements},
        year_index=year_index,
    )


def _trace_value(value: object) -> object:
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return value
    if type(value) is int:
        return value
    if isinstance(value, Real):
        return ("binary64", _float_bits(value, "trace floating value"))
    if isinstance(value, tuple):
        return tuple(_trace_value(item) for item in value)
    raise TypeError(f"unsupported trace payload value: {type(value).__name__}")


def _freeze_trace(trace: actuation.EqualSharesTrace) -> tuple[object, ...]:
    if not isinstance(trace, actuation.EqualSharesTrace):
        raise TypeError("witness solver did not return an EqualSharesTrace")
    return (
        tuple(trace.winners),
        tuple(trace.payment_order),
        tuple(trace.completion_order),
        tuple(
            (
                step.phase,
                step.project,
                None if step.rho is None else _float_bits(step.rho, "trace rho"),
                _float_bits(step.remaining_budget, "trace remaining budget"),
                tuple((name, _trace_value(value)) for name, value in step.comparison_margins),
            )
            for step in trace.steps
        ),
        trace.strict,
        tuple(
            (index, voter_id, _float_bits(budget, "trace initial budget"))
            for index, voter_id, budget in trace.initial_budget_state
        ),
    )


def _uncertified(
    reason: str,
    *,
    instance_semantics_sha256: str,
    endowment_bits: tuple[str, ...],
    runtime: NumericRuntime,
    source_role_sha256: tuple[tuple[str, str], ...],
    input_file_sha256: tuple[tuple[str, str], ...],
) -> UncertifiedTrace:
    payload = {
        "schema": BINDING_SCHEMA,
        "status": "uncertified",
        "reason": reason,
        "instance_semantics_sha256": instance_semantics_sha256,
        "endowment_bits": endowment_bits,
        "runtime": _runtime_payload(runtime),
        "source_role_sha256": source_role_sha256,
        "input_file_sha256": input_file_sha256,
        "float_runtime_proved": False,
        "cache_safe": False,
    }
    return _factory_create(
        UncertifiedTrace,
        certified=False,
        reason=reason,
        context_digest=_canonical_sha256(payload),
        instance_semantics_sha256=instance_semantics_sha256,
        endowment_bits=endowment_bits,
        runtime=runtime,
        source_role_sha256=source_role_sha256,
        input_file_sha256=input_file_sha256,
        float_runtime_proved=False,
        cache_safe=False,
    )


def compile_bound_trace(
    inst: PBInstance,
    anchor: residual.AnchorSpec,
    scaler: residual.FeatureScaler,
    base_weights: Sequence[float],
    state: RolloutState,
    *,
    completion: bool,
    input_paths: Mapping[str, Path] | None = None,
) -> BoundTrace | UncertifiedTrace:
    """Freeze inputs and compile one process-scoped three-kernel trace binding."""

    if type(completion) is not bool:
        raise TypeError("completion must be a bool")
    instance_payload, instance_snapshot = _freeze_instance(inst)
    anchor_snapshot, frozen_anchor = _freeze_anchor(anchor)
    scaler_snapshot, frozen_scaler = _freeze_scaler(scaler)
    weight_bits = _freeze_weights(base_weights)
    state_snapshot = _freeze_state(state)
    frozen_weights = tuple(_float_from_bits(value) for value in weight_bits)
    semantic_sha256 = _canonical_sha256(instance_payload)
    frozen_instance = _materialize_instance(instance_snapshot, semantic_sha256)
    partition = _validate_partition(
        _derive_partition_snapshot(tuple(frozen_instance.votes)),
        len(frozen_instance.votes),
    )
    source_role_sha256 = _snapshot_source_roles()
    input_file_sha256 = _snapshot_inputs(input_paths)
    runtime = _runtime_snapshot()

    policy = residual.residual_policy(frozen_anchor, frozen_scaler, frozen_weights)
    try:
        raw_endowments = policy(
            _materialize_instance(instance_snapshot, semantic_sha256),
            _materialize_state(state_snapshot),
        )
    except ValueError as exc:
        if str(exc) not in _POLICY_NORMALIZATION_FAILURES:
            raise
        return _uncertified(
            f"residual policy normalization failed: {exc}",
            instance_semantics_sha256=semantic_sha256,
            endowment_bits=(),
            runtime=runtime,
            source_role_sha256=source_role_sha256,
            input_file_sha256=input_file_sha256,
        )
    try:
        materialized = tuple(raw_endowments)
    except TypeError:
        return _uncertified(
            "derived endowments are not an iterable runtime vector",
            instance_semantics_sha256=semantic_sha256,
            endowment_bits=(),
            runtime=runtime,
            source_role_sha256=source_role_sha256,
            input_file_sha256=input_file_sha256,
        )
    if len(materialized) != len(frozen_instance.votes):
        return _uncertified(
            "derived endowments do not contain one value per voter",
            instance_semantics_sha256=semantic_sha256,
            endowment_bits=(),
            runtime=runtime,
            source_role_sha256=source_role_sha256,
            input_file_sha256=input_file_sha256,
        )
    try:
        endowment_bits = tuple(
            _float_bits(value, f"derived endowment {index}")
            for index, value in enumerate(materialized)
        )
    except (TypeError, ValueError):
        return _uncertified(
            "derived endowments are not a finite binary64 runtime vector",
            instance_semantics_sha256=semantic_sha256,
            endowment_bits=(),
            runtime=runtime,
            source_role_sha256=source_role_sha256,
            input_file_sha256=input_file_sha256,
        )
    endowments = tuple(_float_from_bits(value) for value in endowment_bits)
    budget = _float_from_bits(instance_snapshot[1])
    if any(value < 0.0 for value in endowments):
        return _uncertified(
            "derived endowments contain a negative runtime value",
            instance_semantics_sha256=semantic_sha256,
            endowment_bits=endowment_bits,
            runtime=runtime,
            source_role_sha256=source_role_sha256,
            input_file_sha256=input_file_sha256,
        )
    if math.fsum(endowments) != budget:
        return _uncertified(
            "math.fsum(endowments) does not exactly equal the parsed budget",
            instance_semantics_sha256=semantic_sha256,
            endowment_bits=endowment_bits,
            runtime=runtime,
            source_role_sha256=source_role_sha256,
            input_file_sha256=input_file_sha256,
        )

    reference_winners = tuple(
        sorted(
            rules.mes_with_endowments(
                _materialize_instance(instance_snapshot, semantic_sha256),
                list(endowments),
                completion=completion,
            )
        )
    )
    trace = actuation.trace_equal_shares(
        _materialize_instance(instance_snapshot, semantic_sha256),
        endowments,
        completion=completion,
    )
    production_winners = tuple(
        sorted(
            production.fast_mes_with_endowments(
                _materialize_instance(instance_snapshot, semantic_sha256),
                endowments,
                completion=completion,
            )
        )
    )
    trace_winners = tuple(sorted(trace.winners))
    if not reference_winners == trace_winners == production_winners:
        return _uncertified(
            "three-kernel runtime winner parity could not be established",
            instance_semantics_sha256=semantic_sha256,
            endowment_bits=endowment_bits,
            runtime=runtime,
            source_role_sha256=source_role_sha256,
            input_file_sha256=input_file_sha256,
        )

    trace_payload = _freeze_trace(trace)
    metadata, budget_bits, projects_snapshot, votes_snapshot = instance_snapshot
    project_id_sha256 = _canonical_sha256(
        {"project_ids": tuple(row[2] for row in projects_snapshot)}
    )
    binding_values: dict[str, object] = {
        "schema": BINDING_SCHEMA,
        "scope": "process",
        "runtime": runtime,
        "instance_semantics_sha256": semantic_sha256,
        "instance_path": f"certificate://v1/{semantic_sha256}",
        "project_id_sha256": project_id_sha256,
        "metadata": metadata,
        "budget_bits": budget_bits,
        "projects": projects_snapshot,
        "votes": votes_snapshot,
        "class_partition": partition,
        "anchor": anchor_snapshot,
        "scaler": scaler_snapshot,
        "base_weight_bits": weight_bits,
        "endowment_bits": endowment_bits,
        "completion": completion,
        "rollout_state": state_snapshot,
        "trace_payload": trace_payload,
        "reference_winners": reference_winners,
        "production_winners": production_winners,
        "source_role_sha256": source_role_sha256,
        "input_file_sha256": input_file_sha256,
        "residual_policy_state_semantics": "ignored-but-bound",
        "source_hashes_are_execution_proof": False,
        "float_runtime_proved": False,
        "cache_safe": False,
    }
    digest_payload = {
        key: (_runtime_payload(value) if key == "runtime" else value)
        for key, value in binding_values.items()
    }
    binding_values["context_digest"] = _canonical_sha256(digest_payload)
    binding = _factory_create(TraceBinding, **binding_values)
    return _factory_create(BoundTrace, binding=binding, trace=trace)
