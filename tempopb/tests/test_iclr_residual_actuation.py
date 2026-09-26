"""Executable contracts for residual winner-signature actuation diagnostics."""

from __future__ import annotations

import copy
from dataclasses import asdict, FrozenInstanceError
import hashlib
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

import iclr_residual_actuation as actuation
from iclr_residual_actuation import (
    ActuationRecord,
    EqualSharesTrace,
    TraceStep,
    scan_actuation_path,
    scan_fold_actuation,
    trace_equal_shares,
    winner_signature,
)
from iclr_train import SeriesData
from parse_pb import PBInstance, Project, Vote, parse_pb_file
from rules import mes_with_endowments


FIXTURE_SIGNATURE_SHA256 = (
    "2019fec8514f49689f73882f2eb56cc2625f8e3527a6f2ca9cec48f0bc83ad51"
)
WARSAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "pb"
MULTICITY_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "pb_multicity"
)
KARWINY_2025_SHA256 = (
    "4258b40a8bd93931fe6b7fbe555fd61df445c5e002db53a90a5a4099a9348a79"
)
BIALOLEKA_LOGICAL_2025_SHA256 = (
    "64e8e0f7dadaf334e64c51609ac6e656f8654d638c3887b594c28c2d21f7963f"
)


def _warsaw_sentinel_path(year: int, district: str) -> Path:
    return WARSAW_DATA_DIR / f"Poland_Warszawa_{year}_{district}.pb"


def _interior_fixture() -> PBInstance:
    return PBInstance(
        path="fixture://interior",
        meta={"budget": "10"},
        projects={
            "large": Project("large", 6.0, None),
            "pay": Project("pay", 4.0, None),
            "fill": Project("fill", 5.0, None),
        },
        votes=[
            Vote("0", ("pay", "large"), age=30, sex="F"),
            Vote("1", ("pay", "fill"), age=70, sex="M"),
        ],
    )


def _affordability_boundary_fixture() -> PBInstance:
    return PBInstance(
        path="fixture://affordability-boundary",
        meta={"budget": "5"},
        projects={
            "boundary": Project("boundary", 5.000000001, None),
        },
        votes=[Vote("0", ("boundary",), age=30, sex="F")],
    )


def _project_order_boundary_fixture() -> PBInstance:
    return PBInstance(
        path="fixture://project-order-boundary",
        meta={"budget": "10"},
        projects={
            "r": Project("r", 4.0, None),
            "q": Project("q", 4.0, None),
            "p": Project("p", 2.0, None),
        },
        votes=[
            Vote("0", ("p", "q", "r"), age=30, sex="F"),
            Vote("1", ("q", "r"), age=70, sex="M"),
        ],
    )


def _completion_change_fixture() -> PBInstance:
    return PBInstance(
        path="fixture://completion-change",
        meta={"budget": "12"},
        projects={
            "D": Project("D", 2.0, None),
            "C": Project("C", 2.0, None),
            "B": Project("B", 4.0, None),
            "A": Project("A", 7.0, None),
        },
        votes=[
            Vote("0", ("A",), age=30, sex="F"),
            Vote("1", ("A", "B"), age=70, sex="M"),
            Vote("2", ("C",), age=30, sex="M"),
            Vote("3", ("D",), age=70, sex="F"),
        ],
    )


def fixture_signature_sha256() -> str:
    """Canonical identity of the four literal, hand-checked elections."""

    payload = [
        {
            "completion": [False, True],
            "endowments": [5.2, 4.8],
            "name": "interior",
            "projects": [["fill", 5.0], ["large", 6.0], ["pay", 4.0]],
            "votes": [
                ["0", ["large", "pay"], 30, "F"],
                ["1", ["fill", "pay"], 70, "M"],
            ],
        },
        {
            "completion": [False, True],
            "endowments": [5.0],
            "name": "affordability-boundary",
            "projects": [["boundary", 5.000000001]],
            "votes": [["0", ["boundary"], 30, "F"]],
        },
        {
            "completion": [False, True],
            "endowments": [5.0, 5.0],
            "name": "project-order-boundary",
            "projects": [["p", 2.0], ["q", 4.0], ["r", 4.0]],
            "votes": [
                ["0", ["p", "q", "r"], 30, "F"],
                ["1", ["q", "r"], 70, "M"],
            ],
        },
        {
            "completion": [False, True],
            "endowments": [5.0, 5.0, 1.0, 1.0],
            "name": "completion-change",
            "projects": [["A", 7.0], ["B", 4.0], ["C", 2.0], ["D", 2.0]],
            "votes": [
                ["0", ["A"], 30, "F"],
                ["1", ["A", "B"], 70, "M"],
                ["2", ["C"], 30, "M"],
                ["3", ["D"], 70, "F"],
            ],
        },
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("instance", "endowments"),
    (
        (_interior_fixture(), (5.2, 4.8)),
        (_affordability_boundary_fixture(), (5.0,)),
        (_project_order_boundary_fixture(), (5.0, 5.0)),
        (_completion_change_fixture(), (5.0, 5.0, 1.0, 1.0)),
        (_completion_change_fixture(), (3.0, 7.0, 1.0, 1.0)),
    ),
    ids=("interior", "affordability", "project-order", "completion-anchor", "completion-shift"),
)
@pytest.mark.parametrize("completion", (False, True))
def test_trace_winners_match_reference_for_every_fixture_and_completion_mode(
    instance: PBInstance,
    endowments: tuple[float, ...],
    completion: bool,
) -> None:
    trace = trace_equal_shares(instance, endowments, completion=completion)

    assert set(trace.winners) == mes_with_endowments(
        instance, list(endowments), completion=completion
    )


@pytest.mark.parametrize(
    ("year", "district"),
    (
        (2020, "Bielany"),
        (2020, "Targowek"),
        (2026, "Bemowo"),
        (2026, "Bielany"),
        (2026, "Targowek"),
    ),
    ids=(
        "2020-Bielany",
        "2020-Targowek",
        "2026-Bemowo",
        "2026-Bielany",
        "2026-Targowek",
    ),
)
@pytest.mark.parametrize(
    "completion", (False, True), ids=("without-completion", "completion")
)
def test_warsaw_validator_arithmetic_sentinels(
    year: int, district: str, completion: bool
) -> None:
    inst = parse_pb_file(_warsaw_sentinel_path(year, district))
    endowments = [inst.budget / len(inst.votes)] * len(inst.votes)

    trace = trace_equal_shares(inst, endowments, completion=completion)

    assert set(trace.winners) == mes_with_endowments(
        inst, endowments, completion=completion
    )


def _authenticated_karwiny_uniform_case() -> tuple[PBInstance, tuple[float, ...]]:
    path = MULTICITY_DATA_DIR / "Poland_Gdynia_2025_Karwiny__large.pb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == KARWINY_2025_SHA256
    inst = parse_pb_file(path)
    endowments = (inst.budget / len(inst.votes),) * len(inst.votes)
    assert len(endowments) == 372
    assert math.fsum(endowments) == 281_142.0
    return inst, endowments


def _authenticated_bialoleka_logical_2025_uniform_case(
) -> tuple[PBInstance, tuple[float, ...]]:
    path = MULTICITY_DATA_DIR / "Poland_Warszawa_2026_Bialoleka.pb"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == BIALOLEKA_LOGICAL_2025_SHA256
    )
    inst = parse_pb_file(path)
    endowments = (inst.budget / len(inst.votes),) * len(inst.votes)
    assert len(endowments) == 6_542
    assert math.fsum(endowments) == 2_902_417.0
    return inst, endowments


def test_bialoleka_validator_replays_rho_instead_of_exact_charge_sum() -> None:
    inst, endowments = _authenticated_bialoleka_logical_2025_uniform_case()

    trace = trace_equal_shares(inst, endowments, completion=False)

    final_charge = tuple(
        step for step in trace.steps if step.phase == "payment-charge"
    )[-1]
    final_candidate = next(
        step
        for step in trace.steps
        if step.phase == "payment-candidate"
        and step.project == final_charge.project
        and step.rho == final_charge.rho
    )
    charges = [
        value
        for name, value in final_charge.comparison_margins
        if name.endswith(":charge")
    ]
    selected_cost = dict(final_candidate.comparison_margins)["cost"]
    assert math.fsum(charges) - selected_cost == 2.3865140974521637e-09
    assert trace.winners == (
        "1010",
        "1089",
        "1091",
        "1096",
        "1105",
        "1267",
        "1608",
        "1991",
        "2025",
        "2177",
        "283",
        "300",
        "69",
        "708",
        "713",
        "767",
        "916",
        "994",
    )
    assert set(trace.winners) == mes_with_endowments(
        inst, list(endowments), completion=False
    )


def test_payment_equation_rejects_one_ulp_forged_selected_cost() -> None:
    forged_cost = math.nextafter(1.0, math.inf)

    with pytest.raises(ValueError, match="payment equation"):
        candidate = _valid_payment_candidate(cost=forged_cost)
        charge = _valid_payment_charge()
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            True,
            ((0, "v0", 2.0),),
        )


def test_payment_equation_rejects_one_ulp_forged_selected_rho() -> None:
    forged_rho = math.nextafter(1.0, math.inf)

    with pytest.raises(ValueError, match="payment equation"):
        candidate = _valid_payment_candidate(
            rho=forged_rho,
            cost=1.0,
            breakpoint_candidate=forged_rho,
        )
        charge = _valid_payment_charge(
            rho=forged_rho,
            remaining=2.0 - forged_rho,
        )
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            True,
            ((0, "v0", 2.0),),
        )


def test_karwiny_validator_replays_binary64_payer_subtractions() -> None:
    inst, endowments = _authenticated_karwiny_uniform_case()

    trace = trace_equal_shares(inst, endowments, completion=False)

    charge = next(step for step in trace.steps if step.phase == "payment-charge")
    assert charge.remaining_budget == 391.9999999999818
    assert set(trace.winners) == mes_with_endowments(
        inst, list(endowments), completion=False
    )


def test_subset_payer_trace_replays_unchanged_balance_without_cancellation() -> None:
    endowments = (13_732.478382470053, 100_232_234.32186723)
    inst = PBInstance(
        path="fixture://subset-rounding",
        meta={"budget": repr(sum(endowments))},
        projects={"p": Project("p", endowments[1], None)},
        votes=[
            Vote("0", (), age=30, sex="F"),
            Vote("1", ("p",), age=30, sex="F"),
        ],
    )

    trace = trace_equal_shares(inst, endowments, completion=False)

    charge = next(step for step in trace.steps if step.phase == "payment-charge")
    assert charge.remaining_budget == endowments[0]
    assert trace.winners == ("p",)


@pytest.mark.parametrize("field", ("charge", "budget_after"))
def test_karwiny_charge_rows_reject_binary64_payer_mutations(field: str) -> None:
    inst, endowments = _authenticated_karwiny_uniform_case()
    trace = trace_equal_shares(inst, endowments, completion=False)
    charge = next(step for step in trace.steps if step.phase == "payment-charge")
    margins = dict(charge.comparison_margins)
    key = f"approver:0:{field}"
    margins[key] = math.nextafter(float(margins[key]), math.inf)

    with pytest.raises(ValueError, match=f"approver:0:{field}"):
        TraceStep(
            charge.phase,
            charge.project,
            charge.rho,
            charge.remaining_budget,
            margins,
        )


def test_karwiny_trace_rejects_mutated_post_charge_total() -> None:
    inst, endowments = _authenticated_karwiny_uniform_case()
    trace = trace_equal_shares(inst, endowments, completion=False)
    steps = list(trace.steps)
    charge_index = next(
        index for index, step in enumerate(steps) if step.phase == "payment-charge"
    )
    charge = steps[charge_index]
    steps[charge_index] = TraceStep(
        charge.phase,
        charge.project,
        charge.rho,
        charge.remaining_budget + 1e-6,
        charge.comparison_margins,
    )

    with pytest.raises(ValueError, match="remaining budget transition"):
        EqualSharesTrace(
            trace.winners,
            trace.payment_order,
            trace.completion_order,
            steps,
            trace.strict,
            trace.initial_budget_state,
        )


def test_payment_charge_validation_uses_fsum_for_high_cardinality_rows() -> None:
    rho = 1_000_000_000.0
    budgets_before = (rho,) + (0.1,) * 4095
    expected_cost = 1_000_000_409.5
    margins = {}
    ordinary_total = 0.0
    for index, before in enumerate(budgets_before):
        charge = min(before, rho)
        ordinary_total += charge
        margins.update(
            {
                f"approver:{index}:branch": abs(before - rho),
                f"approver:{index}:budget_before": before,
                f"approver:{index}:charge": charge,
                f"approver:{index}:budget_after": before - charge,
            }
        )

    _, _, validated_total = actuation._payment_charge_rows(margins, rho)

    assert abs(ordinary_total - expected_cost) > 1e-9
    assert math.fsum(min(before, rho) for before in budgets_before) == expected_cost
    assert validated_total == expected_cost


def test_trace_uses_reference_ordered_sum_at_float_affordability_boundary() -> None:
    instance = PBInstance(
        path="fixture://ordered-sum-boundary",
        meta={"budget": "10000000000000002"},
        projects={
            "p": Project("p", 10000000000000002.0, None),
        },
        votes=[Vote(str(index), ("p",), age=30, sex="F") for index in range(11)],
    )
    endowments = (1e16,) + (0.1,) * 10

    trace = trace_equal_shares(instance, endowments, completion=False)
    first_step = trace.steps[0]
    margins = dict(first_step.comparison_margins)

    assert trace.winners == ()
    assert margins["total_available"] == 1e16
    assert margins["affordability_result"] == "remove"
    assert set(trace.winners) == mes_with_endowments(
        instance, list(endowments), completion=False
    )


def test_payment_candidate_rejects_one_ulp_total_available_forgery() -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    forged_total = math.nextafter(float(margins["total_available"]), math.inf)
    margins["total_available"] = forged_total
    margins["affordability"] = abs(forged_total - (margins["cost"] - 1e-9))

    with pytest.raises(ValueError, match="payer budgets.*total_available"):
        TraceStep("payment-candidate", "p", 1.0, forged_total, margins)


def test_generated_candidate_replays_builtin_sum_in_voter_index_order() -> None:
    endowments = (
        9.823399221920027e125,
        5.273618287841217e-213,
        1.8034749212529832e276,
        5.8869629619189225e177,
        9.034780734224347e-57,
        5.880184847098048e-272,
        7.091003101946137e86,
        4.117125571154831e168,
        6.045561400178794e276,
        6.213468828249906e-97,
        1.8132752175636547e152,
        7.553477150490519e75,
    )
    reference_total = sum(endowments)
    instance = PBInstance(
        path="fixture://builtin-sum-order",
        meta={"budget": repr(reference_total)},
        projects={
            "p": Project("p", math.nextafter(reference_total, math.inf), None)
        },
        votes=[
            Vote(str(index), ("p",), age=30, sex="F")
            for index in range(len(endowments))
        ],
    )

    trace = trace_equal_shares(instance, endowments, completion=False)

    assert reference_total != math.fsum(endowments)
    assert trace.winners == ()
    assert mes_with_endowments(instance, list(endowments), completion=False) == set()


def test_interior_trace_is_strict_and_locally_order_stable() -> None:
    instance = _interior_fixture()

    anchor = trace_equal_shares(instance, (5.2, 4.8), completion=True)
    perturbed = trace_equal_shares(instance, (5.1, 4.9), completion=True)
    without_completion = trace_equal_shares(
        instance, (5.2, 4.8), completion=False
    )

    assert anchor.winners == ("fill", "pay")
    assert anchor.payment_order == ("pay",)
    assert anchor.completion_order == ("fill",)
    assert anchor.strict is True
    assert perturbed.payment_order == anchor.payment_order
    assert perturbed.completion_order == anchor.completion_order
    assert perturbed.strict is True
    assert without_completion.winners == ("pay",)
    assert without_completion.completion_order == ()


def test_generated_trace_retains_canonical_initial_budget_state() -> None:
    trace = trace_equal_shares(_interior_fixture(), (5.2, 4.8), completion=False)

    assert trace.initial_budget_state == ((0, "0", 5.2), (1, "1", 4.8))


def test_equal_payer_budgets_record_identity_and_zero_order_margin() -> None:
    trace = trace_equal_shares(_interior_fixture(), (5.0, 5.0), completion=True)
    payment = next(
        step
        for step in trace.steps
        if step.phase == "payment-candidate" and step.project == "pay"
    )
    margins = dict(payment.comparison_margins)

    assert margins["budget_order:0"] == 0.0
    assert margins["budget_order:0:voters"] == ((0, "0"), (1, "1"))
    assert trace.payment_order == ("pay",)
    assert trace.completion_order == ("fill",)
    assert trace.strict is False


def test_duplicate_voter_labels_preserve_generated_reference_parity() -> None:
    instance = PBInstance(
        path="fixture://duplicate-voter-labels",
        meta={"budget": "2"},
        projects={"p": Project("p", 2.0, None)},
        votes=[
            Vote("same", ("p",), age=30, sex="F"),
            Vote("same", ("p",), age=70, sex="M"),
        ],
    )
    endowments = (1.0, 1.0)

    trace = trace_equal_shares(instance, endowments, completion=False)

    assert trace.winners == ("p",)
    assert set(trace.winners) == mes_with_endowments(
        instance, list(endowments), completion=False
    )


def test_affordability_equality_is_recorded_and_trace_is_not_strict() -> None:
    trace = trace_equal_shares(
        _affordability_boundary_fixture(), (5.0,), completion=False
    )
    boundary = next(
        step
        for step in trace.steps
        if step.phase == "payment-candidate" and step.project == "boundary"
    )

    assert dict(boundary.comparison_margins)["affordability"] == 0.0
    assert trace.strict is False
    assert trace.winners == ()


def test_rho_cost_and_project_id_ties_are_recorded_canonically() -> None:
    trace = trace_equal_shares(
        _project_order_boundary_fixture(), (5.0, 5.0), completion=False
    )
    exact_id_tie = next(
        step
        for step in trace.steps
        if step.phase == "payment-candidate"
        and step.project == "r"
        and dict(step.comparison_margins).get("incumbent_key")
        == (2.0, 4.0, "q")
    )
    margins = dict(exact_id_tie.comparison_margins)

    assert trace.payment_order == ("p", "q", "r")
    assert margins["rho_order"] == 0.0
    assert margins["cost_order"] == 0.0
    assert margins["candidate_key"] == (2.0, 4.0, "r")
    assert margins["project_id_order"] == ("q", "r")
    assert trace.strict is False
    final_r = next(
        step
        for step in trace.steps
        if step.phase == "payment-candidate"
        and step.project == "r"
        and step.rho == 3.0
    )
    final_margins = dict(final_r.comparison_margins)
    assert final_margins["breakpoint:0:result"] == "continue"
    assert final_margins["breakpoint:1:result"] == "accept"


def test_payment_phase_change_alters_greedy_completion_set() -> None:
    instance = _completion_change_fixture()

    anchor = trace_equal_shares(instance, (5.0, 5.0, 1.0, 1.0))
    shifted = trace_equal_shares(instance, (3.0, 7.0, 1.0, 1.0))

    assert anchor.payment_order == ("A",)
    assert anchor.completion_order == ("C", "D")
    assert anchor.winners == ("A", "C", "D")
    assert shifted.payment_order == ("B",)
    assert shifted.completion_order == ("A",)
    assert shifted.winners == ("A", "B")
    anchor_c = next(
        step
        for step in anchor.steps
        if step.phase == "completion-candidate" and step.project == "C"
    )
    anchor_d = next(
        step
        for step in anchor.steps
        if step.phase == "completion-candidate" and step.project == "D"
    )
    shifted_a = next(
        step
        for step in shifted.steps
        if step.phase == "completion-candidate" and step.project == "A"
    )
    assert anchor_c.remaining_budget == 5.0
    assert dict(anchor_c.comparison_margins)["affordability"] == 3.0
    assert anchor_d.remaining_budget == 3.0
    assert dict(anchor_d.comparison_margins)["affordability"] == 1.0
    assert shifted_a.remaining_budget == 8.0
    assert dict(shifted_a.comparison_margins)["affordability_result"] == "fund"


def test_signature_is_a_sorted_tuple_independent_of_project_insertion_order() -> None:
    forward = _project_order_boundary_fixture()
    reverse = PBInstance(
        path="fixture://project-order-boundary-reversed",
        meta=dict(forward.meta),
        projects=dict(reversed(tuple(forward.projects.items()))),
        votes=list(forward.votes),
    )

    def policy(inst, state):
        del inst, state
        return [5.0, 5.0]

    assert winner_signature(forward, policy) == ("p", "q", "r")
    assert winner_signature(reverse, policy) == ("p", "q", "r")


def _valid_payment_candidate(
    project="p",
    rho=1.0,
    remaining=2.0,
    *,
    cost=None,
    total_available=None,
    breakpoint_budget=2.0,
    breakpoint_candidate=None,
    extra=None,
) -> TraceStep:
    if cost is None:
        cost = 1.0 if rho is None else rho
    if total_available is None:
        total_available = 2.0 if rho is not None else 0.0
    affordability = abs(total_available - (cost - 1e-9))
    margins = {
        "affordability": affordability,
        "affordability_result": (
            "remove" if total_available < cost - 1e-9 else "continue"
        ),
        "cost": cost,
        "payer_count": 1,
        "payer:0:identity": (0, "v0"),
        "payer:0:budget": total_available,
        "total_available": total_available,
    }
    if rho is not None:
        if breakpoint_candidate is None:
            breakpoint_candidate = rho
        margins.update(
            {
                "breakpoint:0": abs(
                    (breakpoint_budget + 1e-12) - breakpoint_candidate
                ),
                "breakpoint:0:budget": breakpoint_budget,
                "breakpoint:0:candidate": breakpoint_candidate,
                "breakpoint:0:result": (
                    "accept"
                    if breakpoint_candidate <= breakpoint_budget + 1e-12
                    else "continue"
                ),
                "candidate_key": (rho, cost, project),
            }
        )
    if extra:
        margins.update(extra)
    return TraceStep("payment-candidate", project, rho, remaining, margins)


def _valid_payment_charge(
    project="p",
    rho=1.0,
    remaining=1.0,
    *,
    budget_before=2.0,
    extra=None,
) -> TraceStep:
    charge = min(budget_before, rho)
    margins = {
        "approver:0:branch": abs(budget_before - rho),
        "approver:0:budget_before": budget_before,
        "approver:0:charge": charge,
        "approver:0:budget_after": budget_before - charge,
    }
    if extra:
        margins.update(extra)
    return TraceStep("payment-charge", project, rho, remaining, margins)


def _valid_two_payer_candidate(project="p", *, extra=None) -> TraceStep:
    margins = {
        "affordability": abs(4.0 - (3.0 - 1e-9)),
        "affordability_result": "continue",
        "budget_order:0": 2.0,
        "budget_order:0:voters": ((0, "v0"), (1, "v1")),
        "breakpoint:0": abs((1.0 + 1e-12) - 1.5),
        "breakpoint:0:budget": 1.0,
        "breakpoint:0:candidate": 1.5,
        "breakpoint:0:result": "continue",
        "breakpoint:1": abs((3.0 + 1e-12) - 2.0),
        "breakpoint:1:budget": 3.0,
        "breakpoint:1:candidate": 2.0,
        "breakpoint:1:result": "accept",
        "candidate_key": (2.0, 3.0, project),
        "cost": 3.0,
        "payer_count": 2,
        "payer:0:budget": 1.0,
        "payer:0:identity": (0, "v0"),
        "payer:1:budget": 3.0,
        "payer:1:identity": (1, "v1"),
        "total_available": 4.0,
    }
    if extra:
        margins.update(extra)
    return TraceStep("payment-candidate", project, 2.0, 4.0, margins)


def _valid_two_payer_charge(project="p", *, swapped=False) -> TraceStep:
    budgets = (3.0, 1.0) if swapped else (1.0, 3.0)
    margins = {}
    for index, before in enumerate(budgets):
        charge = min(before, 2.0)
        margins.update(
            {
                f"approver:{index}:branch": abs(before - 2.0),
                f"approver:{index}:budget_before": before,
                f"approver:{index}:charge": charge,
                f"approver:{index}:budget_after": before - charge,
            }
        )
    return TraceStep("payment-charge", project, 2.0, 1.0, margins)


def _valid_completion_order(
    project="c",
    remaining=2.0,
    *,
    approval_count=1,
    cost=1.0,
    spent=0.0,
    payment_spend_order=(),
    previous=None,
    extra=None,
) -> TraceStep:
    margins = {
        "approval_count": approval_count,
        "sort_key": (-approval_count, cost, project),
        "spent_before_completion": spent,
    }
    if previous is None and payment_spend_order is not None:
        margins["payment_spend_order"] = payment_spend_order
    if previous is not None:
        previous_count, previous_cost, previous_project = previous
        margins["previous_sort_key"] = (
            -previous_count,
            previous_cost,
            previous_project,
        )
        if approval_count != previous_count:
            margins["approval_count_order"] = abs(
                approval_count - previous_count
            )
        else:
            margins["cost_order"] = abs(cost - previous_cost)
            if cost == previous_cost:
                margins["project_id_order"] = tuple(
                    sorted((previous_project, project))
                )
    if extra:
        margins.update(extra)
    return TraceStep(
        "completion-order", project, None, remaining, margins
    )


def _valid_completion_candidate(
    project="c",
    result="fund",
    remaining=2.0,
    *,
    approval_count=1,
    cost=1.0,
    extra=None,
) -> TraceStep:
    margins = {
        "approval_count": approval_count,
        "positive_cost": abs(cost),
        "positive_cost_result": "continue" if cost > 0.0 else "reject",
        "sort_key": (-approval_count, cost, project),
    }
    if cost > 0.0:
        margins.update(
            {
                "affordability": abs(remaining - cost),
                "affordability_result": result,
            }
        )
    if extra:
        margins.update(extra)
    return TraceStep("completion-candidate", project, None, remaining, margins)


class _StringLike:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


def test_records_are_defensive_immutable_canonical_tuples() -> None:
    margins = {"z": 2.0, "a": 1.0, "voters": [[0, "0"], [1, "1"]]}
    step = _valid_payment_candidate("b", extra=margins)
    charge = _valid_payment_charge("b")
    completion_order_step = _valid_completion_order(
        "a", spent=1.0, payment_spend_order=("b",)
    )
    payment_winner_order_step = _valid_completion_order(
        "b", spent=1.0, previous=(1, 1.0, "a")
    )
    completion_step = _valid_completion_candidate("a")
    winners = ["b", "a"]
    payment_order = ["b"]
    completion_order = ["a"]
    initial_budget_state = [[0, "v0", 2.0]]
    steps = [
        step,
        charge,
        completion_order_step,
        payment_winner_order_step,
        completion_step,
    ]
    trace = EqualSharesTrace(
        winners,
        payment_order,
        completion_order,
        steps,
        False,
        initial_budget_state,
    )
    record = ActuationRecord("fixture", False, None, None, None, None)
    margins["a"] = 9.0
    margins["voters"][0][1] = "mutated"
    winners.append("c")
    payment_order.append("a")
    completion_order.clear()
    initial_budget_state[0][2] = 99.0
    steps.clear()

    assert step.comparison_margins == (
        ("a", 1.0),
        ("affordability", abs(2.0 - (1.0 - 1e-9))),
        ("affordability_result", "continue"),
        ("breakpoint:0", abs((2.0 + 1e-12) - 1.0)),
        ("breakpoint:0:budget", 2.0),
        ("breakpoint:0:candidate", 1.0),
        ("breakpoint:0:result", "accept"),
        ("candidate_key", (1.0, 1.0, "b")),
        ("cost", 1.0),
        ("payer:0:budget", 2.0),
        ("payer:0:identity", (0, "v0")),
        ("payer_count", 1),
        ("total_available", 2.0),
        ("voters", ((0, "0"), (1, "1"))),
        ("z", 2.0),
    )
    assert trace.winners == ("a", "b")
    assert trace.payment_order == ("b",)
    assert trace.completion_order == ("a",)
    assert trace.steps == (
        step,
        charge,
        completion_order_step,
        payment_winner_order_step,
        completion_step,
    )
    assert trace.initial_budget_state == ((0, "v0", 2.0),)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        step.phase = "changed"
    with pytest.raises((FrozenInstanceError, AttributeError)):
        trace.winners = ()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        record.actuated = True


@pytest.mark.parametrize("strict", (0, 1, "false", None))
def test_trace_requires_an_actual_boolean(strict) -> None:
    with pytest.raises(TypeError, match="strict must be a bool"):
        EqualSharesTrace((), (), (), (), strict)


@pytest.mark.parametrize(
    ("winners", "payment", "completion"),
    (
        (("a", "a"), ("a",), ()),
        (("a",), ("a", "a"), ()),
        (("a",), (), ("a", "a")),
        (("a",), ("a",), ("a",)),
        (("a", "b"), ("a",), ()),
    ),
)
def test_trace_rejects_contradictory_winner_and_order_states(
    winners, payment, completion
) -> None:
    with pytest.raises(ValueError, match="winner|order"):
        EqualSharesTrace(winners, payment, completion, (), True)


def test_trace_requires_trace_steps_and_supports_copy_pickle_and_json() -> None:
    with pytest.raises(TypeError, match="TraceStep"):
        EqualSharesTrace(("p",), ("p",), (), ({"mutable": []},), True)

    step = _valid_payment_candidate(extra={"nested": [1, [2.0, True]]})
    charge = _valid_payment_charge()
    trace = EqualSharesTrace(
        ("p",),
        ("p",),
        (),
        (step, charge),
        True,
        ((0, "v0", 2.0),),
    )
    for cloned in (copy.copy(trace), copy.deepcopy(trace), pickle.loads(pickle.dumps(trace))):
        assert cloned == trace
        assert cloned.steps == (step, charge)
    json.dumps(asdict(trace), sort_keys=True, allow_nan=False)


def test_trace_step_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="trace phase"):
        TraceStep("payment", "p", 1.0, 1.0, {})


@pytest.mark.parametrize(
    "margins",
    (
        [("affordability", 0.0), ("affordability", 2.0)],
        [(1, 0.0), ("1", 2.0)],
    ),
)
def test_trace_step_rejects_duplicate_canonical_names(margins) -> None:
    with pytest.raises(ValueError, match="duplicate trace margin"):
        TraceStep("payment-candidate", "p", 1.0, 1.0, margins)


@pytest.mark.parametrize(
    ("phase", "rho"),
    (
        ("payment-charge", None),
        ("completion-order", 1.0),
        ("completion-candidate", 1.0),
    ),
)
def test_trace_step_rejects_phase_inconsistent_rho(phase, rho) -> None:
    with pytest.raises(ValueError, match="rho"):
        TraceStep(phase, "p", rho, 1.0, {})


def test_payment_candidate_allows_defined_or_missing_rho() -> None:
    assert _valid_payment_candidate(rho=None).rho is None
    assert _valid_payment_candidate(rho=1.0).rho == 1.0


def _complete_payment_trace(
    candidate_margins=(),
    charge_margins=(),
    *,
    strict=True,
    candidate_kwargs=None,
) -> EqualSharesTrace:
    candidate = _valid_payment_candidate(
        extra=dict(candidate_margins), **(candidate_kwargs or {})
    )
    charge = _valid_payment_charge(extra=dict(charge_margins))
    return EqualSharesTrace(
        ("p",),
        ("p",),
        (),
        (candidate, charge),
        strict,
        ((0, "v0", 2.0),),
    )


def _zero_affordability_terminal(strict) -> EqualSharesTrace:
    budget = 1.0 - 1e-9
    candidate = TraceStep(
        "payment-candidate",
        "p",
        None,
        budget,
        {
            "affordability": 0.0,
            "affordability_result": "continue",
            "breakpoint:0": abs((budget + 1e-12) - 1.0),
            "breakpoint:0:budget": budget,
            "breakpoint:0:candidate": 1.0,
            "breakpoint:0:result": "continue",
            "cost": 1.0,
            "payer_count": 1,
            "payer:0:budget": budget,
            "payer:0:identity": (0, "v0"),
            "rho_result": "remove",
            "total_available": budget,
        },
    )
    return EqualSharesTrace(
        (), (), (), (candidate,), strict, ((0, "v0", budget),)
    )


def test_strict_trace_rejects_zero_affordability_certificate() -> None:
    with pytest.raises(ValueError, match="strict flag"):
        _zero_affordability_terminal(True)


def test_strict_trace_rejects_zero_candidate_key_certificates() -> None:
    first = _valid_payment_candidate("a")
    second = _valid_payment_candidate(
        "b",
        extra={
            "incumbent_key": (1.0, 1.0, "a"),
            "rho_order": 0.0,
            "cost_order": 0.0,
            "project_id_order": ("a", "b"),
        },
    )
    charge = _valid_payment_charge("a")
    with pytest.raises(ValueError, match="strict flag"):
        EqualSharesTrace(
            ("a",),
            ("a",),
            (),
            (first, second, charge),
            True,
            ((0, "v0", 2.0),),
        )


def test_strict_trace_rejects_zero_charge_certificate() -> None:
    candidate = _valid_payment_candidate(
        remaining=1.0, total_available=1.0, breakpoint_budget=1.0
    )
    charge = _valid_payment_charge(remaining=0.0, budget_before=1.0)
    with pytest.raises(ValueError, match="strict flag"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            True,
            ((0, "v0", 1.0),),
        )


def test_strict_trace_rejects_zero_completion_certificate() -> None:
    order = _valid_completion_order(remaining=1.0, cost=0.0)
    funded = TraceStep(
        "completion-candidate",
        "c",
        None,
        1.0,
        {
            "approval_count": 1,
            "positive_cost": 0.0,
            "positive_cost_result": "reject",
            "sort_key": (-1, 0.0, "c"),
        },
    )
    with pytest.raises(ValueError, match="strict flag"):
        EqualSharesTrace((), (), (), (order, funded), True)


def test_non_strict_trace_accepts_retained_zero_certificate() -> None:
    assert _zero_affordability_terminal(False).strict is False


@pytest.mark.parametrize("value", (-1.0, "positive", True))
def test_recognized_certificate_margin_requires_nonnegative_real_number(value) -> None:
    with pytest.raises((TypeError, ValueError), match="certificate margin"):
        _complete_payment_trace({"affordability": value})


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_recognized_certificate_margin_rejects_nonfinite_values(value) -> None:
    with pytest.raises(ValueError, match="finite"):
        _complete_payment_trace({"affordability": value})


@pytest.mark.parametrize("value", (1, 1.0))
def test_recognized_certificate_margin_accepts_ordinary_int_and_float(value) -> None:
    order = _valid_completion_order()
    candidate = _valid_completion_candidate(extra={"positive_cost": value})
    trace = EqualSharesTrace(("c",), (), ("c",), (order, candidate), True)
    assert trace.strict is True


def test_trace_authenticates_payment_and_completion_orders_from_steps() -> None:
    candidate = _valid_payment_candidate()
    charge = _valid_payment_charge()
    completion_order = _valid_completion_order(
        spent=1.0, payment_spend_order=("p",)
    )
    payment_winner_order = _valid_completion_order(
        "p", spent=1.0, previous=(1, 1.0, "c")
    )
    funded = _valid_completion_candidate()
    trace = EqualSharesTrace(
        ("p", "c"),
        ("p",),
        ("c",),
        (candidate, charge, completion_order, payment_winner_order, funded),
        False,
        ((0, "v0", 2.0),),
    )
    assert trace.payment_order == ("p",)
    assert trace.completion_order == ("c",)


@pytest.mark.parametrize(
    "steps",
    (
        (_valid_payment_candidate(),),
        (
            _valid_payment_candidate(),
            _valid_payment_charge(),
            _valid_payment_charge(),
        ),
    ),
)
def test_trace_rejects_missing_or_duplicate_payment_charge_evidence(steps) -> None:
    with pytest.raises(ValueError, match="payment order.*charge"):
        EqualSharesTrace(("p",), ("p",), (), steps, True)


@pytest.mark.parametrize(
    ("winners", "completion_order", "steps"),
    (
        (
            (),
            (),
            (
                _valid_completion_order(),
                _valid_completion_candidate(),
            ),
        ),
        (
            ("c",),
            ("c",),
            (
                _valid_completion_order(),
                _valid_completion_candidate(),
                _valid_completion_candidate(),
            ),
        ),
    ),
)
def test_trace_rejects_extra_or_duplicate_funded_completion_evidence(
    winners, completion_order, steps
) -> None:
    with pytest.raises(ValueError, match="completion order.*funded"):
        EqualSharesTrace(winners, (), completion_order, steps, True)


@pytest.mark.parametrize(
    "steps",
    (
        (_valid_payment_charge(),),
        (
            _valid_payment_candidate(rho=2.0),
            _valid_payment_charge(rho=1.0),
        ),
    ),
)
def test_trace_rejects_charge_without_matching_candidate_evidence(steps) -> None:
    with pytest.raises(ValueError, match="payment charge.*candidate"):
        EqualSharesTrace(("p",), ("p",), (), steps, True)


def test_trace_rejects_stale_candidate_from_an_earlier_payment_round() -> None:
    steps = (
        _valid_payment_candidate("p", rho=2.0, remaining=3.0),
        _valid_payment_candidate(
            "q",
            rho=1.0,
            remaining=3.0,
            extra={
                "incumbent_key": (2.0, 2.0, "p"),
                "rho_order": 1.0,
            },
        ),
        _valid_payment_charge("q", rho=1.0, remaining=2.0),
        _valid_payment_charge("p", rho=2.0, remaining=1.0),
    )
    with pytest.raises(ValueError, match="payment charge.*current-round candidate"):
        EqualSharesTrace(("p", "q"), ("q", "p"), (), steps, True)


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        ({"affordability_result": "banana"}, "affordability_result"),
        ({"breakpoint:0:result": "banana"}, "breakpoint.*result"),
        ({"affordability_result": "remove"}, "defined rho.*continue"),
        ({"breakpoint:0:result": "continue"}, "accepted breakpoint"),
        (
            {
                "breakpoint:1": abs((2.0 + 1e-12) - 1.0),
                "breakpoint:1:budget": 2.0,
                "breakpoint:1:candidate": 1.0,
                "breakpoint:1:result": "accept",
            },
            "accepted breakpoint|payer vector",
        ),
    ),
)
def test_payment_candidate_rejects_impossible_generated_results(extra, message) -> None:
    with pytest.raises(ValueError, match=message):
        _valid_payment_candidate(extra=extra)


@pytest.mark.parametrize(
    "candidate_key",
    (
        (1.0, 1.0),
        (1.0, 1.0, "q"),
        (2.0, 1.0, "p"),
        (1.0, 0.0, "p"),
        (1.0, -1.0, "p"),
        (1.0, "1.0", "p"),
        (1.0, True, "p"),
    ),
)
def test_payment_candidate_requires_canonical_candidate_key(candidate_key) -> None:
    with pytest.raises((TypeError, ValueError), match="candidate_key"):
        _valid_payment_candidate(extra={"candidate_key": candidate_key})


def test_payment_candidate_cross_authenticates_cost_breakpoint_and_key() -> None:
    with pytest.raises(
        ValueError,
        match="candidate_key|final accepted breakpoint|explicit payer budget",
    ):
        _valid_payment_candidate(
            rho=1.0,
            remaining=10.0,
            cost=9.0,
            total_available=10.0,
            breakpoint_budget=2.0,
            breakpoint_candidate=2.0,
            extra={"candidate_key": (1.0, 1.0, "p")},
        )


@pytest.mark.parametrize("missing", ("cost", "total_available"))
def test_payment_candidate_requires_retained_affordability_inputs(missing) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    del margins[missing]
    with pytest.raises(ValueError, match=missing):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("cost", True),
        ("cost", "1.0"),
        ("cost", 0.0),
        ("cost", -1.0),
        ("total_available", True),
        ("total_available", "2.0"),
        ("total_available", -1.0),
    ),
)
def test_payment_candidate_rejects_invalid_affordability_inputs(field, value) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize(
    "extra",
    (
        {"affordability": 99.0},
        {
            "total_available": 0.0,
            "affordability": abs(0.0 - (1.0 - 1e-9)),
            "affordability_result": "continue",
        },
    ),
)
def test_payment_candidate_authenticates_affordability_margin_and_result(extra) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins.update(extra)
    with pytest.raises(ValueError, match="affordability"):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize("missing", ("budget", "candidate"))
def test_payment_candidate_requires_complete_breakpoint_quartet(missing) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    del margins[f"breakpoint:0:{missing}"]
    with pytest.raises(ValueError, match=f"breakpoint:0:{missing}"):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("budget", True),
        ("budget", "2.0"),
        ("budget", -1.0),
        ("candidate", True),
        ("candidate", "1.0"),
    ),
)
def test_payment_candidate_rejects_invalid_breakpoint_numbers(field, value) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins[f"breakpoint:0:{field}"] = value
    with pytest.raises((TypeError, ValueError), match=f"breakpoint:0:{field}"):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize(
    "extra",
    (
        {"breakpoint:0": 99.0},
        {
            "breakpoint:0:budget": 0.0,
            "breakpoint:0:candidate": 1.0,
            "breakpoint:0": abs((0.0 + 1e-12) - 1.0),
            "breakpoint:0:result": "accept",
        },
    ),
)
def test_payment_candidate_authenticates_breakpoint_margin_and_result(extra) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins.update(extra)
    with pytest.raises(ValueError, match="breakpoint"):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


def test_payment_candidate_requires_nondecreasing_visited_budgets() -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins.update(
        {
            "breakpoint:0": abs((2.0 + 1e-12) - 3.0),
            "breakpoint:0:budget": 2.0,
            "breakpoint:0:candidate": 3.0,
            "breakpoint:0:result": "continue",
            "breakpoint:1": abs((1.0 + 1e-12) - 1.0),
            "breakpoint:1:budget": 1.0,
            "breakpoint:1:candidate": 1.0,
            "breakpoint:1:result": "accept",
        }
    )
    with pytest.raises(
        ValueError, match="breakpoint budgets.*nondecreasing|payment equation"
    ):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


def test_payment_candidate_defined_rho_matches_final_accepted_candidate() -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins.update(
        {
            "breakpoint:0": abs((2.0 + 1e-12) - 2.0),
            "breakpoint:0:candidate": 2.0,
        }
    )
    with pytest.raises(
        ValueError, match="rho.*final accepted breakpoint|payment equation"
    ):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize(
    "margins",
    (
        {
            "affordability": 1.0,
            "affordability_result": "continue",
            "breakpoint:0": 1.0,
            "breakpoint:0:budget": 1.0,
            "breakpoint:0:candidate": 1.0 + 5e-10,
            "breakpoint:0:result": "continue",
        },
        {
            "affordability": 1.0,
            "affordability_result": "continue",
            "breakpoint:0": 1.0,
            "breakpoint:0:budget": 1.0,
            "breakpoint:0:candidate": 1.0 + 5e-10,
            "breakpoint:0:result": "continue",
            "rho_result": "banana",
        },
    ),
)
def test_payment_candidate_without_rho_requires_generated_terminal_result(margins) -> None:
    margins = {
        "cost": 1.0 + 5e-10,
        "total_available": 1.0,
        **margins,
        "affordability": abs(1.0 - ((1.0 + 5e-10) - 1e-9)),
        "breakpoint:0": abs((1.0 + 1e-12) - (1.0 + 5e-10)),
        "payer_count": 1,
        "payer:0:budget": 1.0,
        "payer:0:identity": (0, "v0"),
    }
    with pytest.raises(ValueError, match="rho_result"):
        TraceStep("payment-candidate", "p", None, 2.0, margins)


def test_payment_candidate_preserves_legitimate_no_rho_paths() -> None:
    removed = _valid_payment_candidate(rho=None)
    cost = 1.0 + 5e-10
    budget = 1.0
    candidate = cost
    all_continue = TraceStep(
        "payment-candidate",
        "p",
        None,
        2.0,
        {
            "affordability": abs(budget - (cost - 1e-9)),
            "affordability_result": "continue",
            "breakpoint:0": abs((budget + 1e-12) - candidate),
            "breakpoint:0:budget": budget,
            "breakpoint:0:candidate": candidate,
            "breakpoint:0:result": "continue",
            "cost": cost,
            "payer_count": 1,
            "payer:0:budget": budget,
            "payer:0:identity": (0, "v0"),
            "rho_result": "remove",
            "total_available": budget,
        },
    )
    assert dict(removed.comparison_margins)["affordability_result"] == "remove"
    assert dict(all_continue.comparison_margins)["rho_result"] == "remove"


@pytest.mark.parametrize(
    "extra",
    (
        {"breakpoint:1:result": "continue"},
        {"breakpoint:not-an-index:result": "accept"},
    ),
)
def test_payment_candidate_rejects_orphan_or_malformed_breakpoint_result(extra) -> None:
    with pytest.raises(ValueError, match="breakpoint.*key shape"):
        _valid_payment_candidate(extra=extra)


def test_trace_rejects_duplicate_project_in_one_payment_round() -> None:
    steps = (
        _valid_payment_candidate(),
        _valid_payment_candidate(),
        _valid_payment_charge(),
    )
    with pytest.raises(ValueError, match="project.*payment round"):
        EqualSharesTrace(("p",), ("p",), (), steps, True)


def test_trace_rejects_charge_for_nonminimal_current_round_key() -> None:
    steps = (
        _valid_payment_candidate("a", rho=1.0),
        _valid_payment_candidate(
            "b",
            rho=2.0,
            extra={
                "incumbent_key": (1.0, 1.0, "a"),
                "rho_order": 1.0,
            },
        ),
        _valid_payment_charge("b", rho=2.0),
    )
    with pytest.raises(ValueError, match="minimum candidate key"):
        EqualSharesTrace(("b",), ("b",), (), steps, True)


def test_trace_rejects_terminal_defined_rho_candidate_without_charge() -> None:
    with pytest.raises(ValueError, match="defined-rho.*charge"):
        EqualSharesTrace((), (), (), (_valid_payment_candidate(),), True)


@pytest.mark.parametrize(
    "missing",
    ("budget_before", "charge", "budget_after"),
)
def test_payment_charge_requires_complete_approver_quartet(missing) -> None:
    margins = dict(_valid_payment_charge().comparison_margins)
    del margins[f"approver:0:{missing}"]
    with pytest.raises(ValueError, match=f"approver:0:{missing}"):
        TraceStep("payment-charge", "p", 1.0, 1.0, margins)


def test_payment_charge_rejects_orphan_indexed_field() -> None:
    margins = dict(_valid_payment_charge().comparison_margins)
    margins["approver:1:charge"] = 0.5
    with pytest.raises(ValueError, match="approver.*quartet"):
        TraceStep("payment-charge", "p", 1.0, 1.0, margins)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("budget_before", True),
        ("budget_before", "2.0"),
        ("budget_before", -1.0),
        ("charge", True),
        ("charge", "1.0"),
        ("budget_after", True),
        ("budget_after", "1.0"),
    ),
)
def test_payment_charge_rejects_invalid_approver_numbers(field, value) -> None:
    margins = dict(_valid_payment_charge().comparison_margins)
    margins[f"approver:0:{field}"] = value
    with pytest.raises((TypeError, ValueError), match=f"approver:0:{field}"):
        TraceStep("payment-charge", "p", 1.0, 1.0, margins)


@pytest.mark.parametrize(
    "extra",
    (
        {"approver:0:branch": 99.0},
        {"approver:0:charge": 0.5},
        {"approver:0:budget_after": 0.5},
    ),
)
def test_payment_charge_authenticates_approver_arithmetic(extra) -> None:
    margins = dict(_valid_payment_charge().comparison_margins)
    margins.update(extra)
    with pytest.raises(ValueError, match="approver"):
        TraceStep("payment-charge", "p", 1.0, 1.0, margins)


def test_payment_charge_authenticates_round_remaining_transition() -> None:
    candidate = _valid_payment_candidate(remaining=2.0)
    charge = _valid_payment_charge(remaining=0.5)
    with pytest.raises(ValueError, match="remaining budget transition"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            True,
            ((0, "v0", 2.0),),
        )


def test_payment_trace_requires_initial_budget_state() -> None:
    candidate = _valid_payment_candidate()
    charge = _valid_payment_charge()

    with pytest.raises(ValueError, match="initial budget state"):
        EqualSharesTrace(("p",), ("p",), (), (candidate, charge), True)


@pytest.mark.parametrize(
    "initial_budget_state",
    (
        ((0, "v0", 2.0), (0, "v1", 0.0)),
        ((1, "v0", 2.0),),
    ),
    ids=("duplicate-index", "index-gap"),
)
def test_payment_trace_rejects_noncontiguous_initial_budget_indices(
    initial_budget_state,
) -> None:
    candidate = _valid_payment_candidate()
    charge = _valid_payment_charge()

    with pytest.raises(ValueError, match="initial budget state indices"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            True,
            initial_budget_state,
        )


def test_payment_trace_rejects_initial_voter_identity_tamper() -> None:
    candidate = _valid_payment_candidate()
    charge = _valid_payment_charge()

    with pytest.raises(ValueError, match="payer identity.*initial budget state"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            True,
            ((0, "altered", 2.0),),
        )


def test_payment_trace_rejects_initial_budget_tamper() -> None:
    candidate = _valid_payment_candidate()
    charge = _valid_payment_charge()

    with pytest.raises(ValueError, match="remaining budget.*initial budget state"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            True,
            ((0, "v0", 2.5),),
        )


def test_payment_trace_rejects_forged_removed_candidate_total() -> None:
    forged = _valid_payment_candidate(
        rho=None,
        remaining=2.0,
        cost=1.5,
        total_available=0.0,
    )

    with pytest.raises(ValueError, match="replayed budget state"):
        EqualSharesTrace(
            (),
            (),
            (),
            (forged,),
            True,
            ((0, "v0", 2.0),),
        )


def test_payment_trace_accepts_exact_zero_support_removed_candidate() -> None:
    candidate = TraceStep(
        "payment-candidate",
        "p",
        None,
        2.0,
        {
            "affordability": abs(0.0 - (1.0 - 1e-9)),
            "affordability_result": "remove",
            "cost": 1.0,
            "payer_count": 0,
            "total_available": 0.0,
        },
    )

    trace = EqualSharesTrace(
        (),
        (),
        (),
        (candidate,),
        True,
        ((0, "v0", 2.0),),
    )

    assert trace.steps == (candidate,)


@pytest.mark.parametrize("payer_count", (True, "1", -1, 0))
def test_payment_candidate_rejects_invalid_or_inconsistent_payer_count(
    payer_count,
) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins["payer_count"] = payer_count

    with pytest.raises((TypeError, ValueError), match="payer_count|payer row"):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize(
    "margins",
    (
        {
            "affordability": abs(2.0 - (3.0 - 1e-9)),
            "affordability_result": "remove",
            "cost": 3.0,
            "payer_count": 2,
            "payer:0:budget": 1.0,
            "payer:0:identity": (1, "v1"),
            "payer:1:budget": 1.0,
            "payer:1:identity": (0, "v0"),
            "total_available": 2.0,
        },
        {
            "affordability": abs(2.0 - (2.0 - 1e-9)),
            "affordability_result": "continue",
            "breakpoint:0": abs((1.0 + 1e-12) - 1.0),
            "breakpoint:0:budget": 1.0,
            "breakpoint:0:candidate": 1.0,
            "breakpoint:0:result": "accept",
            "budget_order:0": 0.0,
            "budget_order:0:voters": ((1, "v1"), (0, "v0")),
            "candidate_key": (1.0, 2.0, "p"),
            "cost": 2.0,
            "payer_count": 2,
            "payer:0:budget": 1.0,
            "payer:0:identity": (1, "v1"),
            "payer:1:budget": 1.0,
            "payer:1:identity": (0, "v0"),
            "total_available": 2.0,
        },
    ),
    ids=("removed", "continue"),
)
def test_payment_candidate_rejects_noncanonical_equal_budget_payer_order(
    margins,
) -> None:
    with pytest.raises(ValueError, match=r"canonical.*budget.*voter index"):
        TraceStep(
            "payment-candidate",
            "p",
            None if margins["affordability_result"] == "remove" else 1.0,
            2.0,
            margins,
        )


def test_payment_round_candidates_share_one_remaining_budget() -> None:
    steps = (
        _valid_payment_candidate("a", rho=1.0, remaining=2.0),
        _valid_payment_candidate(
            "b",
            rho=2.0,
            remaining=3.0,
            extra={
                "incumbent_key": (1.0, 1.0, "a"),
                "rho_order": 1.0,
            },
        ),
        _valid_payment_charge("a", rho=1.0, remaining=1.0),
    )
    with pytest.raises(ValueError, match="payment round.*remaining budget"):
        EqualSharesTrace(("a",), ("a",), (), steps, True)


def test_next_payment_round_starts_at_previous_charge_remaining_budget() -> None:
    steps = (
        _valid_payment_candidate("a", remaining=2.0),
        _valid_payment_charge("a", remaining=1.0),
        _valid_payment_candidate("b", remaining=2.0),
        _valid_payment_charge("b", remaining=1.0),
    )
    with pytest.raises(ValueError, match="next payment round.*remaining budget"):
        EqualSharesTrace(("a", "b"), ("a", "b"), (), steps, True)


def test_terminal_payment_round_rejects_one_ulp_remaining_budget_forgery() -> None:
    steps = (
        _valid_payment_candidate("a", remaining=2.0),
        _valid_payment_charge("a", remaining=1.0),
        _valid_payment_candidate(
            "b", rho=None, remaining=math.nextafter(1.0, math.inf)
        ),
    )

    with pytest.raises(ValueError, match="next payment round.*remaining budget"):
        EqualSharesTrace(("a",), ("a",), (), steps, True)


def test_payment_round_candidate_projects_follow_generated_order() -> None:
    steps = (
        _valid_payment_candidate("b", remaining=2.0),
        _valid_payment_candidate("a", remaining=2.0),
        _valid_payment_charge("a", remaining=1.0),
    )
    with pytest.raises(ValueError, match="payment round projects.*ascending"):
        EqualSharesTrace(("a",), ("a",), (), steps, True)


@pytest.mark.parametrize("reappearance", ("removed", "charged"))
def test_removed_or_charged_project_cannot_reappear_in_later_payment_round(
    reappearance,
) -> None:
    if reappearance == "removed":
        steps = (
            _valid_payment_candidate("a", rho=None, remaining=2.0),
            _valid_payment_candidate("b", remaining=2.0),
            _valid_payment_charge("b", remaining=1.0),
            _valid_payment_candidate("a", rho=None, remaining=1.0),
        )
        winners = payment_order = ("b",)
    else:
        steps = (
            _valid_payment_candidate("a", remaining=2.0),
            _valid_payment_charge("a", remaining=1.0),
            _valid_payment_candidate("a", rho=None, remaining=1.0),
        )
        winners = payment_order = ("a",)
    with pytest.raises(ValueError, match="cannot reappear.*payment round"):
        EqualSharesTrace(winners, payment_order, (), steps, True)


def test_defined_nonwinning_candidate_may_reappear_in_later_payment_round() -> None:
    steps = (
        _valid_payment_candidate("a", remaining=3.0),
        _valid_payment_candidate(
            "b",
            remaining=3.0,
            extra={
                "incumbent_key": (1.0, 1.0, "a"),
                "rho_order": 0.0,
                "cost_order": 0.0,
                "project_id_order": ("a", "b"),
            },
        ),
        _valid_payment_charge("a", remaining=2.0),
        _valid_payment_candidate(
            "b",
            remaining=2.0,
            total_available=1.0,
            breakpoint_budget=1.0,
        ),
        _valid_payment_charge("b", remaining=1.0, budget_before=1.0),
    )
    trace = EqualSharesTrace(
        ("a", "b"),
        ("a", "b"),
        (),
        steps,
        False,
        ((0, "v0", 2.0), (1, "v1", 1.0)),
    )
    assert trace.payment_order == ("a", "b")


def test_defined_payment_rho_must_be_strictly_positive() -> None:
    margins = {
        "affordability": abs(2.0 - (1.0 - 1e-9)),
        "affordability_result": "continue",
        "breakpoint:0": abs((2.0 + 1e-12) - 0.0),
        "breakpoint:0:budget": 2.0,
        "breakpoint:0:candidate": 0.0,
        "breakpoint:0:result": "accept",
        "candidate_key": (0.0, 1.0, "p"),
        "cost": 1.0,
        "total_available": 2.0,
    }
    with pytest.raises(ValueError, match="payment rho.*positive"):
        TraceStep("payment-candidate", "p", 0.0, 2.0, margins)


@pytest.mark.parametrize(
    ("phase", "rho", "margins"),
    (
        (
            "payment-candidate",
            None,
            dict(_valid_payment_candidate(rho=None).comparison_margins),
        ),
        (
            "payment-charge",
            1.0,
            dict(_valid_payment_charge().comparison_margins),
        ),
    ),
)
def test_payment_steps_require_nonnegative_remaining_budget(
    phase, rho, margins
) -> None:
    with pytest.raises(ValueError, match="payment.*remaining budget.*nonnegative"):
        TraceStep(phase, "p", rho, -1.0, margins)


def test_payment_candidate_total_available_cannot_exceed_remaining_budget() -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    with pytest.raises(ValueError, match="total_available.*remaining budget"):
        TraceStep("payment-candidate", "p", 1.0, 1.0, margins)


def test_payment_candidate_total_available_allows_tiny_roundoff() -> None:
    total_available = 1.0 + 5e-13
    candidate = _valid_payment_candidate(
        remaining=1.0,
        total_available=total_available,
        breakpoint_budget=total_available,
    )
    assert dict(candidate.comparison_margins)["total_available"] == total_available


def test_payment_equation_rejects_reviewed_one_payer_underpayment() -> None:
    with pytest.raises(ValueError, match="payment equation|breakpoint candidate"):
        _valid_payment_candidate(
            cost=2.0,
            total_available=2.0,
            breakpoint_budget=2.0,
            breakpoint_candidate=1.0,
        )


def test_selected_candidate_and_charge_require_same_supporter_budgets() -> None:
    candidate = _valid_payment_candidate(
        remaining=2.0,
        total_available=1.0,
        breakpoint_budget=1.0,
    )
    charge = _valid_payment_charge(remaining=1.0, budget_before=2.0)
    with pytest.raises(ValueError, match="candidate.*charge.*budget"):
        EqualSharesTrace(("p",), ("p",), (), (candidate, charge), True)


def test_two_payer_breakpoint_recurrence_is_authenticated() -> None:
    with pytest.raises(ValueError, match="payment equation|breakpoint candidate"):
        _valid_two_payer_candidate(
            extra={
                "breakpoint:0": abs((1.0 + 1e-12) - 2.0),
                "breakpoint:0:candidate": 2.0,
            }
        )


@pytest.mark.parametrize(
    "extra",
    (
        {
            "budget_order:1": 1.0,
            "budget_order:1:voters": ((1, "v1"), (2, "v2")),
        },
        {"budget_order:0:voters": ((True, "v0"), (1, "v1"))},
        {"budget_order:0:voters": ((0, 0), (1, "v1"))},
        {
            "budget_order:0": 1.0,
            "budget_order:0:voters": ((0, "v0"), (1, "v1")),
            "budget_order:1": 1.0,
            "budget_order:1:voters": ((2, "v2"), (3, "v3")),
        },
        {
            "budget_order:0": 1.0,
            "budget_order:0:voters": ((0, "v0"), (1, "v1")),
            "budget_order:1": 1.0,
            "budget_order:1:voters": ((1, "changed-v1"), (2, "v2")),
        },
        {
            "budget_order:0": 1.0,
            "budget_order:0:voters": ((0, "v0"), (1, "v1")),
            "budget_order:1": 1.0,
            "budget_order:1:voters": ((1, "v1"), (0, "v0")),
        },
    ),
)
def test_budget_order_voter_links_form_one_contiguous_collision_free_chain(
    extra,
) -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins.update(extra)
    with pytest.raises((TypeError, ValueError), match="budget_order|voter"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


@pytest.mark.parametrize("missing", ("identity", "budget"))
def test_payer_rows_require_identity_and_budget(missing) -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    del margins[f"payer:1:{missing}"]

    with pytest.raises(ValueError, match=r"payer:1|payer row"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


def test_payer_row_indices_must_be_contiguous() -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins["payer:2:identity"] = margins.pop("payer:1:identity")
    margins["payer:2:budget"] = margins.pop("payer:1:budget")

    with pytest.raises(ValueError, match=r"payer.*contiguous"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


def test_payer_rows_reject_duplicate_voter_indices() -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins["payer:1:identity"] = (0, "v1")

    with pytest.raises(ValueError, match=r"payer.*duplicate|voter indices"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


def test_payer_rows_authenticate_adjacent_voter_identities() -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins["payer:0:identity"] = (1, "v1")
    margins["payer:1:identity"] = (0, "v0")

    with pytest.raises(ValueError, match=r"budget_order.*voter|payer.*identity"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


def test_payer_rows_authenticate_retained_total_available() -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins["payer:0:budget"] = 1.25

    with pytest.raises(ValueError, match=r"payer.*total_available|payer.*budget"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


def test_budget_order_separation_is_derived_from_explicit_payer_budgets() -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins["budget_order:0"] = 1.0
    margins["breakpoint:0"] = abs((1.5 + 1e-12) - 1.5)
    margins["breakpoint:0:budget"] = 1.5
    margins["breakpoint:0:candidate"] = 1.5
    margins["breakpoint:0:result"] = "accept"
    for name in (
        "breakpoint:1",
        "breakpoint:1:budget",
        "breakpoint:1:candidate",
        "breakpoint:1:result",
    ):
        del margins[name]
    margins["candidate_key"] = (1.5, 3.0, "p")

    with pytest.raises(ValueError, match=r"budget_order.*separation"):
        TraceStep("payment-candidate", "p", 1.5, 4.0, margins)


def test_generated_high_cardinality_near_equal_payers_have_canonical_rows() -> None:
    payer_count = 512
    endowments = tuple(1.0 + index * 1e-13 for index in range(payer_count))
    instance = PBInstance(
        path="fixture://high-cardinality-near-equal-payers",
        meta={"budget": str(math.fsum(endowments))},
        projects={"p": Project("p", payer_count / 2.0, None)},
        votes=[
            Vote(f"v{index}", ("p",), age=30, sex="F")
            for index in range(payer_count)
        ],
    )

    trace = trace_equal_shares(instance, endowments, completion=False)
    candidate = next(
        step for step in trace.steps if step.phase == "payment-candidate"
    )
    margins = dict(candidate.comparison_margins)

    assert trace.winners == ("p",)
    for position, budget in enumerate(endowments):
        assert margins[f"payer:{position}:identity"] == (
            position,
            f"v{position}",
        )
        assert margins[f"payer:{position}:budget"] == budget


@pytest.mark.parametrize("token", ("00", "\u0660"), ids=("leading-zero", "unicode-digit"))
def test_budget_order_rejects_noncanonical_index_overwrite(token) -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins["budget_order:0"] = 999.0
    margins[f"budget_order:{token}"] = 2.0

    with pytest.raises(ValueError, match="budget_order.*index|canonical.*index"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


@pytest.mark.parametrize("token", ("00", "\u0660"), ids=("leading-zero", "unicode-digit"))
def test_breakpoint_rejects_noncanonical_complete_contradictory_row(token) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins.update(
        {
            f"breakpoint:{token}": 999.0,
            f"breakpoint:{token}:budget": 999.0,
            f"breakpoint:{token}:candidate": 999.0,
            f"breakpoint:{token}:result": "continue",
        }
    )

    with pytest.raises(ValueError, match="breakpoint.*index|canonical.*index"):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


@pytest.mark.parametrize("token", ("00", "\u0660"), ids=("leading-zero", "unicode-digit"))
def test_approver_rejects_noncanonical_complete_contradictory_row(token) -> None:
    margins = dict(_valid_payment_charge().comparison_margins)
    margins.update(
        {
            f"approver:{token}:branch": 999.0,
            f"approver:{token}:budget_before": 999.0,
            f"approver:{token}:charge": 999.0,
            f"approver:{token}:budget_after": 0.0,
        }
    )

    with pytest.raises(ValueError, match="approver.*index|canonical.*index"):
        TraceStep("payment-charge", "p", 1.0, 1.0, margins)


@pytest.mark.parametrize("token", ("+0", "-0", " 0", "0 "))
def test_indexed_history_names_reject_signs_and_whitespace(token) -> None:
    margins = dict(_valid_payment_candidate().comparison_margins)
    margins[f"breakpoint:{token}:result"] = "continue"
    with pytest.raises(ValueError, match="breakpoint.*index|key shape|canonical.*index"):
        TraceStep("payment-candidate", "p", 1.0, 2.0, margins)


def test_payer_budget_must_be_nonnegative() -> None:
    margins = dict(_valid_two_payer_candidate().comparison_margins)
    margins["payer:0:budget"] = -1.0

    with pytest.raises(ValueError, match=r"payer:0:budget.*nonnegative"):
        TraceStep("payment-candidate", "p", 2.0, 4.0, margins)


def test_candidate_charge_rejects_swapped_explicit_voter_budgets() -> None:
    candidate = _valid_two_payer_candidate()
    charge = _valid_two_payer_charge(swapped=True)
    with pytest.raises(ValueError, match="candidate.*charge.*budget"):
        EqualSharesTrace(("p",), ("p",), (), (candidate, charge), True)


def test_payment_incumbent_evidence_rejects_false_strict_equal_keys() -> None:
    first = _valid_payment_candidate("a")
    second = _valid_payment_candidate(
        "b",
        extra={
            "incumbent_key": (1.0, 1.0, "a"),
            "rho_order": 1.0,
            "cost_order": 1.0,
            "project_id_order": ("a", "b"),
        },
    )
    charge = _valid_payment_charge("a")
    with pytest.raises(ValueError, match="incumbent|rho_order|cost_order|strict"):
        EqualSharesTrace(("a",), ("a",), (), (first, second, charge), True)


@pytest.mark.parametrize(
    "incumbent_key",
    (
        (True, 1.0, "a"),
        (1.0, True, "a"),
        (0.0, 1.0, "a"),
        (-1.0, 1.0, "a"),
        (1.0, 0.0, "a"),
        (1.0, -1.0, "a"),
        (math.inf, 1.0, "a"),
        (1.0, math.inf, "a"),
        (math.nan, 1.0, "a"),
        (1.0, math.nan, "a"),
        (1.0, 1.0),
        (1.0, 1.0, "a", "extra"),
        "not-a-history-key",
        (1.0, 1.0, True),
    ),
)
def test_payment_incumbent_key_requires_canonical_typed_history_tuple(
    incumbent_key,
) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="incumbent_key|trace margin values"
    ):
        EqualSharesTrace(
            ("a",),
            ("a",),
            (),
            (
                _valid_payment_candidate("a"),
                _valid_payment_candidate(
                    "b",
                    extra={
                        "incumbent_key": incumbent_key,
                        "rho_order": 0.0,
                        "cost_order": 0.0,
                        "project_id_order": ("a", "b"),
                    },
                ),
                _valid_payment_charge("a"),
            ),
            False,
        )


@pytest.mark.parametrize(
    "second_extra",
    (
        {},
        {"incumbent_key": (1.0, 1.0, "a"), "rho_order": 0.0},
        {
            "incumbent_key": (1.0, 1.0, "a"),
            "rho_order": 0.0,
            "cost_order": 0.0,
        },
        {
            "incumbent_key": (1.0, 1.0, "a"),
            "rho_order": 0.0,
            "cost_order": 0.0,
            "project_id_order": ("b", "a"),
        },
    ),
)
def test_payment_incumbent_evidence_requires_exact_generated_tie_branch(
    second_extra,
) -> None:
    steps = (
        _valid_payment_candidate("a"),
        _valid_payment_candidate("b", extra=second_extra),
        _valid_payment_charge("a"),
    )
    with pytest.raises(ValueError, match="incumbent|rho_order|cost_order|project_id"):
        EqualSharesTrace(("a",), ("a",), (), steps, False)


def test_first_defined_candidate_rejects_stray_incumbent_evidence() -> None:
    candidate = _valid_payment_candidate(
        extra={"incumbent_key": (2.0, 1.0, "q"), "rho_order": 1.0}
    )
    charge = _valid_payment_charge()
    with pytest.raises(ValueError, match="first.*incumbent|stray.*incumbent"):
        EqualSharesTrace(("p",), ("p",), (), (candidate, charge), True)


def test_strict_flag_must_equal_derived_payment_certificates() -> None:
    candidate = _valid_payment_candidate()
    charge = _valid_payment_charge()
    with pytest.raises(ValueError, match="strict.*derived|strict flag"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge),
            False,
            ((0, "v0", 2.0),),
        )


def test_deliberate_empty_trace_requires_strict_true() -> None:
    with pytest.raises(ValueError, match="empty.*strict|strict flag"):
        EqualSharesTrace((), (), (), (), False)


def test_terminal_payment_round_requires_one_common_remaining_budget() -> None:
    steps = (
        _valid_payment_candidate("a", rho=None, remaining=3.0),
        _valid_payment_candidate("b", rho=None, remaining=4.0),
    )
    with pytest.raises(ValueError, match="terminal payment round.*remaining budget"):
        EqualSharesTrace((), (), (), steps, True)


def test_trace_rejects_funded_completion_without_order_evidence() -> None:
    funded = _valid_completion_candidate()
    with pytest.raises(ValueError, match="funded completion.*order"):
        EqualSharesTrace(("c",), (), ("c",), (funded,), True)


def test_trace_rejects_rejected_completion_without_order_evidence() -> None:
    rejected = _valid_completion_candidate(result="reject", remaining=0.0)
    with pytest.raises(ValueError, match="completion candidate.*order"):
        EqualSharesTrace((), (), (), (rejected,), True)


def test_trace_rejects_duplicate_completion_order_evidence() -> None:
    order = _valid_completion_order()
    with pytest.raises(ValueError, match="completion-order evidence.*unique"):
        EqualSharesTrace((), (), (), (order, order), True)


def test_trace_rejects_completion_block_before_payment_block() -> None:
    steps = (
        _valid_completion_order(),
        _valid_completion_candidate(),
        _valid_payment_candidate(),
        _valid_payment_charge(),
    )
    with pytest.raises(ValueError, match="payment phases.*completion"):
        EqualSharesTrace(("p", "c"), ("p",), ("c",), steps, True)


def test_trace_rejects_duplicate_rejected_completion_candidates() -> None:
    steps = (
        _valid_completion_order(),
        _valid_completion_candidate(result="reject", remaining=0.0),
        _valid_completion_candidate(result="reject", remaining=0.0),
    )
    with pytest.raises(ValueError, match="completion-candidate projects.*unique"):
        EqualSharesTrace((), (), (), steps, True)


def test_completion_candidates_must_follow_completion_order_subsequence() -> None:
    steps = (
        _valid_completion_order("a"),
        _valid_completion_order("b", previous=(1, 1.0, "a")),
        _valid_completion_candidate("b", result="reject", remaining=0.0),
        _valid_completion_candidate("a", result="reject", remaining=0.0),
    )
    with pytest.raises(ValueError, match="order-preserving subsequence"):
        EqualSharesTrace((), (), (), steps, True)


@pytest.mark.parametrize(
    ("phase", "project", "margins"),
    (
        (
            "completion-order",
            "p",
            {"approval_count": 1, "sort_key": (-1, 1.0)},
        ),
        (
            "completion-order",
            "p",
            {"approval_count": 1, "sort_key": (-1, 1.0, "q")},
        ),
        (
            "completion-order",
            "p",
            {"approval_count": 2, "sort_key": (-1, 1.0, "p")},
        ),
        (
            "completion-order",
            "p",
            {"approval_count": 1.5, "sort_key": (-1.5, 1.0, "p")},
        ),
        (
            "completion-order",
            "p",
            {"approval_count": True, "sort_key": (-1, 1.0, "p")},
        ),
        (
            "completion-order",
            "p",
            {"approval_count": 1, "sort_key": (-1, True, "p")},
        ),
        (
            "completion-candidate",
            "p",
            {
                "approval_count": 1,
                "sort_key": (-1, 1.0, "q"),
                "positive_cost": 1.0,
                "positive_cost_result": "continue",
                "affordability": 1.0,
                "affordability_result": "fund",
            },
        ),
    ),
)
def test_completion_steps_require_canonical_sort_key(phase, project, margins) -> None:
    with pytest.raises((TypeError, ValueError), match="sort_key"):
        TraceStep(phase, project, None, 2.0, margins)


def test_trace_rejects_inverted_completion_sort_keys() -> None:
    steps = (
        _valid_completion_order("b", remaining=0.0, cost=2.0),
        _valid_completion_order(
            "a", remaining=0.0, cost=1.0, payment_spend_order=None
        ),
        _valid_completion_candidate("b", result="reject", remaining=0.0, cost=2.0),
        _valid_completion_candidate("a", result="reject", remaining=0.0, cost=1.0),
    )
    with pytest.raises(ValueError, match="completion-order keys.*nondecreasing"):
        EqualSharesTrace((), (), (), steps, True)


def test_completion_order_rows_share_one_initial_remaining_budget() -> None:
    steps = (
        _valid_completion_order("a", remaining=2.0, approval_count=2),
        _valid_completion_order(
            "b",
            remaining=3.0,
            approval_count=1,
            payment_spend_order=None,
        ),
    )
    with pytest.raises(ValueError, match="completion-order.*initial remaining budget"):
        EqualSharesTrace((), (), (), steps, True)


def test_completion_order_rows_reject_one_ulp_remaining_budget_forgery() -> None:
    steps = (
        _valid_completion_order("a", remaining=3.0, approval_count=2),
        _valid_completion_order(
            "b",
            remaining=math.nextafter(3.0, math.inf),
            approval_count=1,
            previous=(2, 1.0, "a"),
        ),
        _valid_completion_candidate("a", remaining=3.0, approval_count=2),
        _valid_completion_candidate("b", remaining=2.0, approval_count=1),
    )

    with pytest.raises(ValueError, match="completion-order.*initial remaining budget"):
        EqualSharesTrace(("a", "b"), (), ("a", "b"), steps, True)


def test_first_completion_candidate_starts_at_initial_remaining_budget() -> None:
    steps = (
        _valid_completion_order("a", remaining=2.0, cost=2.0),
        _valid_completion_candidate(
            "a", result="reject", remaining=1.0, cost=2.0
        ),
    )
    with pytest.raises(ValueError, match="first completion candidate.*remaining budget"):
        EqualSharesTrace((), (), (), steps, True)


def test_completion_candidates_update_remaining_budget_after_each_fund() -> None:
    steps = (
        _valid_completion_order("a", remaining=2.0),
        _valid_completion_order(
            "b", remaining=2.0, previous=(1, 1.0, "a")
        ),
        _valid_completion_candidate("a", remaining=2.0),
        _valid_completion_candidate("b", remaining=2.0),
    )
    with pytest.raises(ValueError, match="completion candidate.*remaining budget transition"):
        EqualSharesTrace(("a", "b"), (), ("a", "b"), steps, True)


def test_completion_candidate_rejects_one_ulp_remaining_budget_forgery() -> None:
    steps = (
        _valid_completion_order("c", remaining=2.0),
        _valid_completion_candidate(
            "c", remaining=math.nextafter(2.0, math.inf)
        ),
    )

    with pytest.raises(ValueError, match="first completion candidate.*remaining budget"):
        EqualSharesTrace(("c",), (), ("c",), steps, True)


def test_completion_candidate_cannot_forge_one_ulp_budget_to_change_winner() -> None:
    unaffordable_cost = math.nextafter(1.0, math.inf)
    steps = (
        _valid_completion_order(
            "c", remaining=1.0, cost=unaffordable_cost, spent=0.0
        ),
        _valid_completion_candidate(
            "c",
            result="fund",
            remaining=unaffordable_cost,
            cost=unaffordable_cost,
        ),
    )

    with pytest.raises(ValueError, match="first completion candidate.*remaining budget"):
        EqualSharesTrace(("c",), (), ("c",), steps, False)


@pytest.mark.parametrize(
    ("first_cost", "first_result", "strict"),
    (
        (3.0, "reject", False),
        (0.0, "fund", False),
    ),
)
def test_completion_remaining_budget_is_unchanged_after_nonfunding_branch(
    first_cost, first_result, strict
) -> None:
    steps = (
        _valid_completion_order(
            "a", remaining=2.0, approval_count=2, cost=first_cost
        ),
        _valid_completion_order(
            "b",
            remaining=2.0,
            approval_count=1,
            previous=(2, first_cost, "a"),
        ),
        _valid_completion_candidate(
            "a",
            result=first_result,
            remaining=2.0,
            approval_count=2,
            cost=first_cost,
        ),
        _valid_completion_candidate("b", remaining=1.0),
    )
    with pytest.raises(ValueError, match="completion candidate.*remaining budget transition"):
        EqualSharesTrace(("b",), (), ("b",), steps, strict)


def test_completion_candidate_cannot_repeat_a_payment_winner() -> None:
    steps = (
        _valid_payment_candidate("a"),
        _valid_payment_charge("a"),
        _valid_completion_order(
            "a", remaining=0.0, spent=1.0, payment_spend_order=("a",)
        ),
        _valid_completion_candidate("a", result="reject", remaining=0.0),
    )
    with pytest.raises(ValueError, match="completion candidate.*payment winner"):
        EqualSharesTrace(("a",), ("a",), (), steps, True)


def test_completion_evidence_orders_every_payment_winner() -> None:
    steps = (
        _valid_payment_candidate("a"),
        _valid_payment_charge("a"),
        _valid_completion_order(
            "b", remaining=0.0, spent=1.0, payment_spend_order=("a",)
        ),
        _valid_completion_candidate("b", result="reject", remaining=0.0),
    )
    with pytest.raises(ValueError, match="payment winner.*completion-order evidence"):
        EqualSharesTrace(("a",), ("a",), (), steps, True)


def test_retained_eligible_completion_order_requires_candidate() -> None:
    order = _valid_completion_order(
        "p", remaining=2.0, extra={"spent_before_completion": 0.0}
    )
    with pytest.raises(ValueError, match="eligible completion.*candidate|candidate sequence"):
        EqualSharesTrace((), (), (), (order,), True)


def test_completion_candidates_equal_exact_retained_eligible_sequence() -> None:
    orders = (
        _valid_completion_order(
            "a",
            remaining=3.0,
            approval_count=2,
            extra={"spent_before_completion": 0.0},
        ),
        _valid_completion_order(
            "b",
            remaining=3.0,
            approval_count=1,
            payment_spend_order=None,
            extra={
                "spent_before_completion": 0.0,
                "previous_sort_key": (-2, 1.0, "a"),
                "approval_count_order": 1,
            },
        ),
    )
    candidate = _valid_completion_candidate(
        "a", remaining=3.0, approval_count=2
    )
    with pytest.raises(ValueError, match="completion candidate sequence"):
        EqualSharesTrace(("a",), (), ("a",), (*orders, candidate), True)


def test_completion_order_rejects_false_strict_equal_costs() -> None:
    first = _valid_completion_order(
        "a", remaining=3.0, extra={"spent_before_completion": 0.0}
    )
    second = _valid_completion_order(
        "b",
        remaining=3.0,
        payment_spend_order=None,
        extra={
            "spent_before_completion": 0.0,
            "previous_sort_key": (-1, 1.0, "a"),
            "cost_order": 1.0,
            "project_id_order": ("a", "b"),
        },
    )
    candidates = (
        _valid_completion_candidate("a", remaining=3.0),
        _valid_completion_candidate("b", remaining=2.0),
    )
    with pytest.raises(ValueError, match="cost_order|completion.*strict"):
        EqualSharesTrace(
            ("a", "b"), (), ("a", "b"), (first, second, *candidates), True
        )


@pytest.mark.parametrize(
    ("first_extra", "second_extra"),
    (
        ({"previous_sort_key": (-1, 1.0, "z")}, {}),
        ({}, {}),
        (
            {},
            {
                "previous_sort_key": (-1, 1.0, "a"),
                "approval_count_order": 1,
                "cost_order": 1.0,
            },
        ),
        (
            {},
            {
                "previous_sort_key": (-1, 1.0, "a"),
                "cost_order": 0.0,
            },
        ),
    ),
)
def test_completion_order_history_requires_exact_generated_branch(
    first_extra, second_extra
) -> None:
    first = _valid_completion_order(
        "a",
        remaining=3.0,
        extra={"spent_before_completion": 0.0, **first_extra},
    )
    second = _valid_completion_order(
        "b",
        remaining=3.0,
        payment_spend_order=None,
        extra={"spent_before_completion": 0.0, **second_extra},
    )
    candidates = (
        _valid_completion_candidate("a", remaining=3.0),
        _valid_completion_candidate("b", remaining=2.0),
    )
    with pytest.raises(
        ValueError,
        match="previous_sort_key|completion-order.*evidence|approval_count_order|project_id_order",
    ):
        EqualSharesTrace(
            ("a", "b"), (), ("a", "b"), (first, second, *candidates), False
        )


@pytest.mark.parametrize(
    ("previous_sort_key", "approval_count"),
    (
        ((False, 1.0, "a"), 0),
        ((-1, True, "a"), 1),
        ((1, 1.0, "a"), 1),
        ((-1.0, 1.0, "a"), 1),
        ((-1, math.inf, "a"), 1),
        ((-1, math.nan, "a"), 1),
        ((-1, 1.0), 1),
        ((-1, 1.0, "a", "extra"), 1),
        ("not-a-history-key", 1),
        ((-1, 1.0, True), 1),
    ),
)
def test_previous_sort_key_requires_canonical_typed_history_tuple(
    previous_sort_key, approval_count
) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="previous_sort_key|trace margin values"
    ):
        first = _valid_completion_order(
            "a", remaining=3.0, approval_count=approval_count
        )
        second = _valid_completion_order(
            "b",
            remaining=3.0,
            approval_count=approval_count,
            previous=(approval_count, 1.0, "a"),
            extra={"previous_sort_key": previous_sort_key},
        )
        if approval_count:
            candidates = (
                _valid_completion_candidate(
                    "a", remaining=3.0, approval_count=approval_count
                ),
                _valid_completion_candidate(
                    "b", remaining=2.0, approval_count=approval_count
                ),
            )
            winners = completion_order = ("a", "b")
        else:
            candidates = ()
            winners = completion_order = ()
        EqualSharesTrace(
            winners,
            (),
            completion_order,
            (first, second, *candidates),
            False,
        )


def test_completion_spend_matches_selected_payment_candidate_costs() -> None:
    candidate = _valid_payment_candidate("p")
    charge = _valid_payment_charge("p")
    order = _valid_completion_order(
        "p",
        remaining=1.0,
        payment_spend_order=("p",),
        extra={"spent_before_completion": 99.0},
    )
    with pytest.raises(ValueError, match="spent_before_completion.*payment.*cost"):
        EqualSharesTrace(("p",), ("p",), (), (candidate, charge, order), True)


def test_completion_spend_rejects_one_ulp_payment_cost_forgery() -> None:
    candidate = _valid_payment_candidate("p")
    charge = _valid_payment_charge("p")
    order = _valid_completion_order(
        "p",
        remaining=1.0,
        spent=math.nextafter(1.0, math.inf),
        payment_spend_order=("p",),
    )

    with pytest.raises(ValueError, match="spent_before_completion.*payment.*cost"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge, order),
            True,
            ((0, "v0", 2.0),),
        )


def test_completion_rows_require_one_common_spent_before_completion() -> None:
    first = _valid_completion_order(
        "a", remaining=3.0, extra={"spent_before_completion": 0.0}
    )
    second = _valid_completion_order(
        "b",
        remaining=3.0,
        payment_spend_order=None,
        extra={
            "spent_before_completion": 1.0,
            "previous_sort_key": (-1, 1.0, "a"),
            "cost_order": 0.0,
            "project_id_order": ("a", "b"),
        },
    )
    candidates = (
        _valid_completion_candidate("a", remaining=3.0),
        _valid_completion_candidate("b", remaining=2.0),
    )
    with pytest.raises(ValueError, match="spent_before_completion.*common"):
        EqualSharesTrace(
            ("a", "b"), (), ("a", "b"), (first, second, *candidates), False
        )


def test_completion_rows_reject_one_ulp_common_spend_forgery() -> None:
    first = _valid_completion_order(
        "a", remaining=3.0, approval_count=2, spent=0.0
    )
    second = _valid_completion_order(
        "b",
        remaining=3.0,
        approval_count=1,
        spent=math.nextafter(0.0, math.inf),
        previous=(2, 1.0, "a"),
    )
    candidates = (
        _valid_completion_candidate("a", remaining=3.0, approval_count=2),
        _valid_completion_candidate("b", remaining=2.0, approval_count=1),
    )

    with pytest.raises(ValueError, match="spent_before_completion.*common"):
        EqualSharesTrace(
            ("a", "b"), (), ("a", "b"), (first, second, *candidates), True
        )


def test_generated_completion_retains_payment_spend_iteration_order() -> None:
    trace = trace_equal_shares(_interior_fixture(), (5.2, 4.8), completion=True)
    first_order = next(
        step for step in trace.steps if step.phase == "completion-order"
    )

    assert dict(first_order.comparison_margins)["payment_spend_order"] == ("pay",)


@pytest.mark.parametrize(
    "forged_order",
    (None, (), ("p", "p"), ("q",), (1,)),
    ids=("missing", "omitted-winner", "duplicate", "wrong-winner", "non-string"),
)
def test_completion_rejects_forged_payment_spend_order(forged_order) -> None:
    candidate = _valid_payment_candidate("p")
    charge = _valid_payment_charge("p")
    order = _valid_completion_order(
        "p", remaining=1.0, spent=1.0, payment_spend_order=("p",)
    )
    margins = dict(order.comparison_margins)
    if forged_order is None:
        margins.pop("payment_spend_order", None)
    else:
        margins["payment_spend_order"] = forged_order
    forged = TraceStep(
        order.phase,
        order.project,
        order.rho,
        order.remaining_budget,
        margins,
    )

    with pytest.raises((TypeError, ValueError), match="payment_spend_order"):
        EqualSharesTrace(
            ("p",),
            ("p",),
            (),
            (candidate, charge, forged),
            True,
            ((0, "v0", 2.0),),
        )


def test_payment_spend_order_must_match_replayed_winner_set_iteration() -> None:
    payment_order = ("a", "b", "c")
    replayed_winners: set[str] = set()
    for project in payment_order:
        replayed_winners.add(project)
    executed_order = tuple(replayed_winners)
    forged_order = tuple(reversed(executed_order))
    assert forged_order != executed_order

    with pytest.raises(ValueError, match="replayed winner-set iteration"):
        actuation._payment_spend_order(forged_order, payment_order)


@pytest.mark.parametrize(
    "extra",
    (
        {"positive_cost": 2.0},
        {"affordability": 0.5},
        {"affordability_result": "reject"},
    ),
)
def test_completion_candidate_authenticates_cost_and_affordability(extra) -> None:
    margins = {
        "approval_count": 1,
        "positive_cost": 1.0,
        "positive_cost_result": "continue",
        "sort_key": (-1, 1.0, "c"),
        "affordability": 1.0,
        "affordability_result": "fund",
    }
    margins.update(extra)
    with pytest.raises((TypeError, ValueError), match="completion|positive_cost|affordability"):
        TraceStep("completion-candidate", "c", None, 2.0, margins)


def test_completion_candidate_sort_key_must_match_order_row() -> None:
    order = _valid_completion_order(cost=1.0)
    candidate = _valid_completion_candidate(
        cost=2.0,
        remaining=3.0,
    )
    with pytest.raises(ValueError, match="sort_key.*completion-order"):
        EqualSharesTrace(("c",), (), ("c",), (order, candidate), True)


def test_positive_completion_sort_cost_cannot_take_reject_branch() -> None:
    with pytest.raises(ValueError, match="positive_cost_result"):
        TraceStep(
            "completion-candidate",
            "z",
            None,
            2.0,
            {
                "approval_count": 1,
                "sort_key": (-1, 1.0, "z"),
                "positive_cost": 1.0,
                "positive_cost_result": "reject",
            },
        )


@pytest.mark.parametrize("cost", (0.0, -1.0))
def test_nonpositive_completion_cost_branch_remains_authentic(cost) -> None:
    order = _valid_completion_order(cost=cost)
    candidate = _valid_completion_candidate(cost=cost)
    trace = EqualSharesTrace((), (), (), (order, candidate), cost != 0.0)
    assert trace.completion_order == ()


@pytest.mark.parametrize("value", (1, _StringLike("payment-candidate")))
def test_trace_step_phase_requires_actual_string(value) -> None:
    with pytest.raises(TypeError, match="phase must be a string"):
        TraceStep(
            value,
            "p",
            1.0,
            2.0,
            {
                "affordability": 1.0,
                "breakpoint:0": 1.0,
                "candidate_key": (1.0, 1.0, "p"),
            },
        )


@pytest.mark.parametrize("value", (1, _StringLike("p")))
def test_trace_step_project_requires_actual_string(value) -> None:
    with pytest.raises(TypeError, match="project must be a string"):
        TraceStep(
            "payment-candidate",
            value,
            1.0,
            2.0,
            {
                "affordability": 1.0,
                "breakpoint:0": 1.0,
                "candidate_key": (1.0, 1.0, "p"),
            },
        )


@pytest.mark.parametrize("value", (1, _StringLike("p")))
def test_trace_winner_and_order_identifiers_require_actual_strings(value) -> None:
    canonical = str(value)
    with pytest.raises(TypeError, match="winner and order identifiers must be strings"):
        EqualSharesTrace(
            (value,),
            (value,),
            (),
            (
                _valid_payment_candidate(canonical),
                _valid_payment_charge(canonical),
            ),
            True,
        )


@pytest.mark.parametrize("value", (1, _StringLike("fixture")))
def test_actuation_instance_requires_actual_string(value) -> None:
    with pytest.raises(TypeError, match="instance must be a string"):
        ActuationRecord(value, False, None, None, None, None)


@pytest.mark.parametrize("value", (True, "1", "positive"))
def test_trace_rho_rejects_booleans_and_strings(value) -> None:
    with pytest.raises(TypeError, match="rho must be numeric"):
        TraceStep(
            "payment-candidate",
            "p",
            value,
            2.0,
            {
                "affordability": 1.0,
                "breakpoint:0": 1.0,
                "candidate_key": (1.0, 1.0, "p"),
            },
        )


@pytest.mark.parametrize("value", (True, "1", "positive"))
def test_trace_remaining_budget_rejects_booleans_and_strings(value) -> None:
    with pytest.raises(TypeError, match="remaining_budget must be numeric"):
        TraceStep("payment-candidate", "p", None, value, {"affordability": 1.0})


@pytest.mark.parametrize("value", (True, "1", "positive"))
def test_actuation_threshold_rejects_booleans_and_strings(value) -> None:
    with pytest.raises(TypeError, match="first_grid_t must be numeric"):
        ActuationRecord(
            "fixture", True, value, 63.0 / 64.0, 1.0, None, None, None
        )


@pytest.mark.parametrize(
    ("low", "high"),
    (
        ("0.984375", 1.0),
        (63.0 / 64.0, "1"),
        (False, 1.0),
        (63.0 / 64.0, True),
    ),
)
def test_actuation_refined_endpoints_require_numeric_nonbooleans(low, high) -> None:
    with pytest.raises(TypeError, match="refined_(low|high) must be numeric"):
        ActuationRecord("fixture", True, 1.0, low, high, None, None, None)


@pytest.mark.parametrize("value", (True, "0.1", "positive"))
def test_actuation_csd_rejects_booleans_and_strings(value) -> None:
    with pytest.raises(TypeError, match="anchor_csd must be numeric"):
        ActuationRecord("fixture", False, None, None, None, None, value, None)


@pytest.mark.parametrize("value", (True, "0.2", "positive"))
def test_actuation_changed_csd_rejects_booleans_and_strings(value) -> None:
    with pytest.raises(TypeError, match="changed_csd must be numeric"):
        ActuationRecord(
            "fixture",
            True,
            1.0,
            63.0 / 64.0,
            1.0,
            "worsens",
            0.1,
            value,
        )


def test_public_numeric_fields_accept_ordinary_ints_and_floats() -> None:
    step = TraceStep(
        "payment-candidate",
        "p",
        1,
        2,
        {
            "affordability": abs(2 - (1 - 1e-9)),
            "affordability_result": "continue",
            "breakpoint:0": abs((2 + 1e-12) - 1),
            "breakpoint:0:budget": 2,
            "breakpoint:0:candidate": 1,
            "breakpoint:0:result": "accept",
            "candidate_key": (1, 1, "p"),
            "cost": 1,
            "payer_count": 1,
            "payer:0:budget": 2,
            "payer:0:identity": (0, "v0"),
            "total_available": 2,
        },
    )
    record = ActuationRecord("fixture", False, None, None, None, None, 1, None)
    assert step.rho == 1.0
    assert step.remaining_budget == 2.0
    assert record.anchor_csd == 1.0


@pytest.mark.parametrize(
    ("phase", "rho", "margins", "message"),
    (
        (
            "payment-candidate",
            None,
            {},
            "payment-candidate.*affordability",
        ),
        (
                "payment-candidate",
                1.0,
                {
                    "affordability": abs(2.0 - (1.0 - 1e-9)),
                    "affordability_result": "continue",
                    "candidate_key": (1.0, 1.0, "p"),
                    "cost": 1.0,
                    "payer_count": 1,
                    "payer:0:budget": 2.0,
                    "payer:0:identity": (0, "v0"),
                    "total_available": 2.0,
                },
                "accepted payment-candidate.*breakpoint",
            ),
            (
                "payment-candidate",
                1.0,
                {
                    "affordability": abs(2.0 - (1.0 - 1e-9)),
                    "affordability_result": "continue",
                    "breakpoint:0": abs((2.0 + 1e-12) - 1.0),
                    "breakpoint:0:budget": 2.0,
                    "breakpoint:0:candidate": 1.0,
                    "breakpoint:0:result": "accept",
                    "cost": 1.0,
                    "payer_count": 1,
                    "payer:0:budget": 2.0,
                    "payer:0:identity": (0, "v0"),
                    "total_available": 2.0,
                },
                "accepted payment-candidate.*candidate_key",
        ),
        ("payment-charge", 1.0, {}, "payment-charge.*approver branch"),
        ("completion-order", None, {}, "completion-order.*sort_key"),
        (
            "completion-candidate",
            None,
            {"positive_cost_result": "reject"},
            "completion-candidate.*positive_cost",
        ),
    ),
)
def test_trace_step_requires_generated_phase_evidence(phase, rho, margins, message) -> None:
    with pytest.raises(ValueError, match=message):
        TraceStep(phase, "p", rho, 2.0, margins)


@pytest.mark.parametrize(
    "margins",
    (
        {
            "positive_cost": 1.0,
            "positive_cost_result": "reject",
            "affordability": 1.0,
            "affordability_result": "reject",
        },
        {
            "positive_cost": 0.0,
            "positive_cost_result": "continue",
            "affordability": 1.0,
            "affordability_result": "fund",
        },
        {
            "positive_cost": 1.0,
            "positive_cost_result": "continue",
        },
    ),
)
def test_completion_candidate_requires_coherent_result_evidence(margins) -> None:
    with pytest.raises(ValueError, match="completion-candidate evidence"):
        TraceStep("completion-candidate", "p", None, 1.0, margins)


def test_generated_boundary_phase_evidence_shapes_remain_valid() -> None:
    unaffordable = _valid_payment_candidate(rho=None, remaining=1.0)
    zero_cost = TraceStep(
        "completion-candidate",
        "z",
        None,
        1.0,
        {
            "approval_count": 1,
            "positive_cost": 0.0,
            "positive_cost_result": "reject",
            "sort_key": (-1, 0.0, "z"),
        },
    )
    assert unaffordable.rho is None
    assert zero_cost.rho is None
    assert _valid_payment_candidate().rho == 1.0
    assert _valid_payment_charge().rho == 1.0
    assert _valid_completion_order().rho is None
    assert _valid_completion_candidate().rho is None


@pytest.mark.parametrize("actuated", (0, 1, "false", None))
def test_actuation_record_requires_an_actual_boolean(actuated) -> None:
    with pytest.raises(TypeError, match="actuated must be a bool"):
        ActuationRecord("fixture", actuated, None, None, None, None)


@pytest.mark.parametrize(
    "record",
    (
        ("bad-order", True, 0.25, 0.30, 0.20, None, None, None),
        ("bad-equal-bracket", True, 0.25, 0.20, 0.20, None, None, None),
        ("bad-high", True, 0.25, 0.20, 0.30, None, None, None),
        ("bad-range", True, 1.25, -0.1, 2.0, None, None, None),
        ("bad-grid", True, 0.20, 0.10, 0.20, None, None, None),
        ("impossible-predecessor", True, 0.25, 0.0, 0.1, None, None, None),
        ("non-dyadic-bracket", True, 0.25, 0.24, 0.245, None, None, None),
        (
            "bad-direction",
            True,
            0.25,
            15.0 / 64.0,
            0.25,
            "improves",
            0.1,
            0.2,
        ),
        (
            "bad-partial-csd",
            True,
            0.25,
            15.0 / 64.0,
            0.25,
            "ties",
            None,
            0.2,
        ),
        ("bad-inert-changed", False, None, None, None, None, 0.1, 0.2),
        ("bad-inert-direction", False, None, None, None, "worsens", 0.1, None),
    ),
)
def test_actuation_record_rejects_semantically_contradictory_states(record) -> None:
    with pytest.raises(ValueError):
        ActuationRecord(*record)


def test_actuation_record_derives_direction_and_is_restartable_json() -> None:
    record = ActuationRecord(
        "fixture", True, 0.25, 15.0 / 64.0, 0.25, "worsens", 0.1, 0.2
    )

    assert record.csd_direction == "worsens"
    for cloned in (copy.copy(record), copy.deepcopy(record), pickle.loads(pickle.dumps(record))):
        assert cloned == record
    json.dumps(asdict(record), sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    "record",
    (
        ActuationRecord("first-grid", True, 1.0 / 64.0, 0.0, 1.0 / 64.0, None),
        ActuationRecord("last-grid", True, 1.0, 63.0 / 64.0, 1.0, None),
        ActuationRecord(
            "refined",
            True,
            0.25,
            3840.0 / 16384.0,
            3968.0 / 16384.0,
            None,
        ),
    ),
)
def test_actuation_record_accepts_grid_boundaries_and_dyadic_roundtrips(record) -> None:
    for cloned in (
        copy.copy(record),
        copy.deepcopy(record),
        pickle.loads(pickle.dumps(record)),
    ):
        assert cloned == record
    json.dumps(asdict(record), sort_keys=True, allow_nan=False)


def test_actuation_record_rejects_jointly_unreachable_refinement_state() -> None:
    with pytest.raises(ValueError, match="eight-step bisection"):
        ActuationRecord(
            "unreachable",
            True,
            0.25,
            3841.0 / 16384.0,
            4095.0 / 16384.0,
            None,
        )


def test_actuation_record_accepts_reachable_refinement_state() -> None:
    record = ActuationRecord(
        "reachable",
        True,
        0.25,
        3840.0 / 16384.0,
        3968.0 / 16384.0,
        None,
    )
    assert pickle.loads(pickle.dumps(record)) == record


@pytest.mark.parametrize(
    "record",
    (
        ("actuated-anchor", True, 0.25, 15.0 / 64.0, 0.25, "improves", 1.0 + 1e-12, 1.0),
        ("actuated-changed", True, 0.25, 15.0 / 64.0, 0.25, "worsens", 1.0, 1.0 + 1e-12),
        ("inert-anchor", False, None, None, None, None, 1.0 + 1e-12, None),
    ),
)
def test_actuation_record_rejects_raw_csd_above_one(record) -> None:
    with pytest.raises(ValueError, match="CSD.*at most 1"):
        ActuationRecord(*record)


@pytest.mark.parametrize(
    "record",
    (
        ("actuated-boundary", True, 0.25, 15.0 / 64.0, 0.25, "ties", 1.0, 1.0),
        ("inert-boundary", False, None, None, None, None, 1.0, None),
        ("inert-negative", False, None, None, None, None, -100.0, None),
        ("actuated-negative", True, 0.25, 15.0 / 64.0, 0.25, "worsens", -2.0, -1.0),
    ),
)
def test_actuation_record_accepts_csd_upper_boundary_and_unbounded_lower(record) -> None:
    assert ActuationRecord(*record).anchor_csd == record[6]


def test_empty_voters_and_malformed_endowments_are_deliberate() -> None:
    empty = PBInstance(
        path="fixture://empty",
        meta={"budget": "10"},
        projects={"p": Project("p", 1.0, None)},
        votes=[],
    )
    empty_trace = trace_equal_shares(empty, (), completion=True)
    assert empty_trace.winners == ()
    assert empty_trace.payment_order == ()
    assert empty_trace.completion_order == ()
    assert winner_signature(empty, lambda inst, state: []) == ()
    assert mes_with_endowments(empty, [], completion=True) == set()

    instance = _interior_fixture()
    for malformed in ((5.0,), (5.0, math.nan), (5.0, -1.0)):
        with pytest.raises(ValueError, match="endowment"):
            trace_equal_shares(instance, malformed)
    with pytest.raises(ValueError, match="endowment"):
        winner_signature(instance, lambda inst, state: [5.0])


def _patch_piecewise_endowments(monkeypatch, observed: list[float]) -> None:
    def fake_anchor_policy(anchor):
        del anchor

        def policy(inst, state):
            del inst, state
            observed.append(0.0)
            return [5.0, 5.0, 1.0, 1.0]

        return policy

    def fake_residual_policy(anchor, scaler, weights):
        del anchor, scaler
        t = float(weights[0])

        def policy(inst, state):
            del inst, state
            observed.append(t)
            return [5.0 - 4.0 * t, 5.0 + 4.0 * t, 1.0, 1.0]

        return policy

    monkeypatch.setattr(actuation, "anchor_policy", fake_anchor_policy)
    monkeypatch.setattr(actuation, "residual_policy", fake_residual_policy)


def test_dense_grid_and_exact_eight_step_first_change_refinement(monkeypatch) -> None:
    observed: list[float] = []
    _patch_piecewise_endowments(monkeypatch, observed)

    record = scan_actuation_path(
        _completion_change_fixture(), object(), object(), (1.0, 0.0, 0.0, 0.0)
    )

    assert observed[:65] == [0.0] + [k / 64.0 for k in range(1, 65)]
    low = 31.0 / 64.0
    high = 32.0 / 64.0
    expected_bisections = []
    for _ in range(8):
        midpoint = (low + high) / 2.0
        expected_bisections.append(midpoint)
        low = midpoint
    assert observed[65:] == expected_bisections
    assert record.instance == "fixture://completion-change"
    assert record.actuated is True
    assert record.first_grid_t == 0.5
    assert record.refined_low == 8191.0 / 16384.0
    assert record.refined_high == 0.5
    assert record.anchor_csd == pytest.approx(3.0 / 11.0)
    assert record.changed_csd == 1.0
    assert record.csd_direction == "worsens"
    assert record.refined_low * 16384.0 == 8191.0
    assert record.refined_high * 16384.0 == 8192.0
    for cloned in (
        copy.copy(record),
        copy.deepcopy(record),
        pickle.loads(pickle.dumps(record)),
    ):
        assert cloned == record
    json.dumps(asdict(record), sort_keys=True, allow_nan=False)


def test_inert_path_evaluates_the_complete_grid_and_reports_no_threshold(
    monkeypatch,
) -> None:
    observed: list[float] = []

    def fake_anchor_policy(anchor):
        del anchor
        return lambda inst, state: [5.0, 5.0]

    def fake_residual_policy(anchor, scaler, weights):
        del anchor, scaler
        t = float(weights[0])
        observed.append(t)
        return lambda inst, state: [5.0, 5.0]

    monkeypatch.setattr(actuation, "anchor_policy", fake_anchor_policy)
    monkeypatch.setattr(actuation, "residual_policy", fake_residual_policy)

    record = scan_actuation_path(
        _interior_fixture(), object(), object(), (1.0, 0.0, 0.0, 0.0)
    )

    assert observed == [k / 64.0 for k in range(1, 65)]
    assert record.actuated is False
    assert record.first_grid_t is None
    assert record.refined_low is None
    assert record.refined_high is None
    assert record.changed_csd is None
    assert record.csd_direction is None


def test_grid_change_followed_by_reversion_remains_an_observed_event(
    monkeypatch,
) -> None:
    grid_signatures: list[tuple[float, tuple[str, ...]]] = []

    def fake_anchor_policy(anchor):
        del anchor
        return lambda inst, state: [5.0, 5.0, 1.0, 1.0]

    def fake_residual_policy(anchor, scaler, weights):
        del anchor, scaler
        t = float(weights[0])
        endowments = (
            [3.0, 7.0, 1.0, 1.0]
            if t == 1.0 / 64.0
            else [5.0, 5.0, 1.0, 1.0]
        )

        def policy(inst, state):
            del state
            signature = tuple(sorted(mes_with_endowments(inst, endowments)))
            grid_signatures.append((t, signature))
            return endowments

        return policy

    monkeypatch.setattr(actuation, "anchor_policy", fake_anchor_policy)
    monkeypatch.setattr(actuation, "residual_policy", fake_residual_policy)

    record = scan_actuation_path(
        _completion_change_fixture(), object(), object(), (1.0, 0.0, 0.0, 0.0)
    )

    observed_grid = grid_signatures[:64]
    assert observed_grid[0][1] == ("A", "B")
    assert observed_grid[1][1] == ("A", "C", "D")
    assert record.actuated is True
    assert record.first_grid_t == 1.0 / 64.0
    assert record.refined_low == 255.0 / 16384.0
    assert record.refined_high == 1.0 / 64.0


def test_refined_high_retains_the_first_grid_observed_differing_signature(
    monkeypatch,
) -> None:
    def policy_with_marker(marker):
        def policy(inst, state):
            del inst, state
            return []

        policy.marker = marker
        return policy

    monkeypatch.setattr(
        actuation, "anchor_policy", lambda anchor: policy_with_marker(0.0)
    )
    monkeypatch.setattr(
        actuation,
        "residual_policy",
        lambda anchor, scaler, weights: policy_with_marker(float(weights[0])),
    )

    def fake_signature(inst, policy):
        del inst
        if policy.marker == 0.0:
            return ("anchor",)
        if policy.marker == 2.0 / 64.0:
            return ("first-grid-change",)
        if 1.0 / 64.0 < policy.marker < 2.0 / 64.0:
            return ("different-between-grid",)
        return ("anchor",)

    monkeypatch.setattr(actuation, "winner_signature", fake_signature)
    monkeypatch.setattr(
        actuation,
        "group_outcome",
        lambda inst, winners, scheme: {
            "group": {"utility": 1.0, "entitlement": 1.0, "voters": 1.0}
        },
    )

    record = scan_actuation_path(
        PBInstance("fixture://third-signature", {"budget": "1"}, {}, []),
        object(),
        object(),
        (1.0, 0.0, 0.0, 0.0),
    )

    assert record.first_grid_t == 2.0 / 64.0
    assert record.refined_low == 1.0 / 64.0
    assert record.refined_high == 2.0 / 64.0


def _patch_marker_scan(monkeypatch, anchor_csd, changed_csd) -> None:
    def policy_with_marker(marker):
        def policy(inst, state):
            del inst, state
            return []

        policy.marker = marker
        return policy

    monkeypatch.setattr(
        actuation, "anchor_policy", lambda anchor: policy_with_marker(0.0)
    )
    monkeypatch.setattr(
        actuation,
        "residual_policy",
        lambda anchor, scaler, weights: policy_with_marker(float(weights[0])),
    )
    monkeypatch.setattr(
        actuation,
        "winner_signature",
        lambda inst, policy: ("anchor",)
        if policy.marker == 0.0
        else ("changed",),
    )

    def fake_group_outcome(inst, winners, scheme):
        del inst
        assert scheme == "age_sex"
        csd = anchor_csd if set(winners) == {"anchor"} else changed_csd
        if csd is None:
            return {"group": {"utility": 0.0, "entitlement": 0.0, "voters": 1.0}}
        return {
            "group": {
                "utility": 100.0 * (1.0 - csd),
                "entitlement": 100.0,
                "voters": 1.0,
            }
        }

    monkeypatch.setattr(actuation, "group_outcome", fake_group_outcome)


@pytest.mark.parametrize(
    ("changed_csd", "expected"),
    ((0.1, "improves"), (0.2 + 5e-13, "ties"), (0.3, "worsens")),
)
def test_csd_direction_uses_exact_absolute_tolerance_and_retains_raw_values(
    monkeypatch,
    changed_csd: float,
    expected: str,
) -> None:
    _patch_marker_scan(monkeypatch, 0.2, changed_csd)

    record = scan_actuation_path(
        PBInstance("fixture://csd", {"budget": "1"}, {}, []),
        object(),
        object(),
        (1.0, 0.0, 0.0, 0.0),
    )

    assert record.anchor_csd == pytest.approx(0.2)
    assert record.changed_csd == pytest.approx(changed_csd)
    assert record.csd_direction == expected


def test_missing_defined_csd_is_explicit_instead_of_fabricated(monkeypatch) -> None:
    _patch_marker_scan(monkeypatch, None, 0.3)

    record = scan_actuation_path(
        PBInstance("fixture://missing-csd", {"budget": "1"}, {}, []),
        object(),
        object(),
        (1.0, 0.0, 0.0, 0.0),
    )

    assert record.actuated is True
    assert record.anchor_csd is None
    assert record.changed_csd == pytest.approx(0.3)
    assert record.csd_direction is None


def test_scan_fold_uses_supplied_instances_and_primary_training_direction_only(
    monkeypatch,
) -> None:
    calls = []
    first = PBInstance("fixture://b", {"budget": "1"}, {}, [])
    second = PBInstance("fixture://a", {"budget": "1"}, {}, [])
    direction = (0.25, -0.5, 0.75, 1.0)
    fit = SimpleNamespace(
        anchor_fit=SimpleNamespace(selected=SimpleNamespace(anchor="anchor")),
        scaler="scaler",
        primary_seed=2,
        seeds=(
            SimpleNamespace(seed=1, selected=SimpleNamespace(endpoint=(9.0,) * 4)),
            SimpleNamespace(seed=2, selected=SimpleNamespace(endpoint=direction)),
            SimpleNamespace(seed=42, selected=SimpleNamespace(endpoint=(-9.0,) * 4)),
        ),
    )

    def fake_scan(inst, anchor, scaler, passed_direction):
        calls.append((inst.path, anchor, scaler, passed_direction))
        return ActuationRecord(inst.path, False, None, None, None, None)

    monkeypatch.setattr(actuation, "scan_actuation_path", fake_scan)

    records = scan_fold_actuation((first, second), fit)

    assert [record.instance for record in records] == ["fixture://a", "fixture://b"]
    assert calls == [
        ("fixture://a", "anchor", "scaler", direction),
        ("fixture://b", "anchor", "scaler", direction),
    ]


def test_scan_fold_rejects_series_and_mixed_view_wrappers(monkeypatch) -> None:
    scored = PBInstance("scored://2023", {"budget": "1"}, {}, [])
    training = PBInstance("train://2022", {"budget": "1"}, {}, [])
    series = SeriesData(
        SimpleNamespace(),
        (2022,),
        (2023,),
        {2022: training},
        {2022: training, 2023: scored},
    )
    wrapper = SimpleNamespace(
        train_only={2022: training},
        all_years={2022: training, 2023: scored},
        test_years=(2023,),
    )
    fit = SimpleNamespace(
        anchor_fit=SimpleNamespace(selected=SimpleNamespace(anchor="anchor")),
        scaler="scaler",
        primary_seed=1,
        seeds=(SimpleNamespace(seed=1, selected=SimpleNamespace(endpoint=(0.0,) * 4)),),
    )
    monkeypatch.setattr(
        actuation,
        "scan_actuation_path",
        lambda *args: pytest.fail("ambiguous wrappers must fail before scanning"),
    )

    with pytest.raises(TypeError, match="explicit sequence of PBInstance"):
        scan_fold_actuation((series,), fit)
    with pytest.raises(TypeError, match="explicit sequence of PBInstance"):
        scan_fold_actuation((scored, wrapper), fit)


def test_scan_fold_scans_only_explicit_scored_instance_paths(monkeypatch) -> None:
    calls: list[str] = []
    scored_b = PBInstance("scored://b", {"budget": "1"}, {}, [])
    scored_a = PBInstance("scored://a", {"budget": "1"}, {}, [])
    fit = SimpleNamespace(
        anchor_fit=SimpleNamespace(selected=SimpleNamespace(anchor="anchor")),
        scaler="scaler",
        primary_seed=42,
        seeds=(
            SimpleNamespace(seed=42, selected=SimpleNamespace(endpoint=(1.0,) * 4)),
        ),
    )

    def fake_scan(inst, anchor, scaler, direction):
        del anchor, scaler, direction
        calls.append(inst.path)
        return ActuationRecord(inst.path, False, None, None, None, None)

    monkeypatch.setattr(actuation, "scan_actuation_path", fake_scan)

    records = scan_fold_actuation((scored_b, scored_a), fit)

    assert calls == ["scored://a", "scored://b"]
    assert [record.instance for record in records] == calls


def test_fixture_signature_is_canonical_and_frozen() -> None:
    assert fixture_signature_sha256() == FIXTURE_SIGNATURE_SHA256
