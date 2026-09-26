"""Winner-signature actuation diagnostics for the residual endowment policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from cohorts import group_outcome
from iclr_env import EndowmentPolicy, RolloutState
from iclr_residual_policy import AnchorSpec, FeatureScaler, anchor_policy, residual_policy
from parse_pb import PBInstance


__all__ = [
    "TraceStep",
    "EqualSharesTrace",
    "ActuationRecord",
    "trace_equal_shares",
    "winner_signature",
    "scan_actuation_path",
    "scan_fold_actuation",
]


MarginValue = float | int | str | bool | tuple[Any, ...]
_TRACE_PHASES = frozenset(
    {
        "payment-candidate",
        "payment-charge",
        "completion-order",
        "completion-candidate",
    }
)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric and not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _freeze_margin_value(value: object) -> MarginValue:
    if isinstance(value, tuple):
        return tuple(_freeze_margin_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_margin_value(item) for item in value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("trace margin values must be finite")
        return value
    raise TypeError(f"unsupported trace margin value: {type(value).__name__}")


def _margin_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must contain a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must contain a finite number")
    return number


def _canonical_index_token(token: str, namespace: str) -> int:
    if not token or any(character not in "0123456789" for character in token):
        raise ValueError(
            f"{namespace} key shape requires a canonical ASCII nonnegative decimal index"
        )
    index = int(token)
    if token != str(index):
        raise ValueError(
            f"{namespace} key shape requires a canonical ASCII nonnegative decimal index"
        )
    return index


def _payment_history_key(
    value: object, name: str
) -> tuple[float, float, str]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-field tuple")
    rho = _margin_real(value[0], f"{name} rho")
    cost = _margin_real(value[1], f"{name} cost")
    if rho <= 0.0 or cost <= 0.0:
        raise ValueError(f"{name} rho and cost must be strictly positive")
    project = value[2]
    if not isinstance(project, str):
        raise TypeError(f"{name} project must be a string")
    return rho, cost, project


def _completion_history_key(
    value: object, name: str
) -> tuple[int, float, str]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-field tuple")
    negative_count = value[0]
    if type(negative_count) is not int:
        raise TypeError(f"{name} count must be an integer and not boolean")
    if negative_count > 0:
        raise ValueError(f"{name} count must be nonpositive")
    cost = _margin_real(value[1], f"{name} cost")
    project = value[2]
    if not isinstance(project, str):
        raise TypeError(f"{name} project must be a string")
    return negative_count, cost, project


def _payment_spend_order(
    value: object, payment_order: Sequence[str]
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("payment_spend_order must be a tuple of project identifiers")
    if any(not isinstance(project, str) for project in value):
        raise TypeError("payment_spend_order project identifiers must be strings")
    if len(set(value)) != len(value):
        raise ValueError("payment_spend_order project identifiers must be unique")
    if set(value) != set(payment_order):
        raise ValueError("payment_spend_order must cover exactly the payment winners")
    replayed_winners: set[str] = set()
    for project in payment_order:
        replayed_winners.add(project)
    if value != tuple(replayed_winners):
        raise ValueError(
            "payment_spend_order must match replayed winner-set iteration"
        )
    return value


def _payment_payer_vector(
    margins: Mapping[str, MarginValue],
    total_available: float,
    *,
    require_order: bool,
) -> tuple[tuple[int, str, float], ...]:
    payer_count = margins.get("payer_count")
    if type(payer_count) is not int:
        raise TypeError("payer_count must be an integer and not boolean")
    if payer_count < 0:
        raise ValueError("payer_count must be nonnegative")
    payer_rows: dict[int, dict[str, MarginValue]] = {}
    order_rows: dict[int, dict[str, MarginValue]] = {}
    for name, value in margins.items():
        if name.startswith("payer:"):
            parts = name.split(":")
            if len(parts) != 3 or parts[2] not in {"identity", "budget"}:
                raise ValueError(
                    "payer keys must form indexed identity/budget pairs"
                )
            index = _canonical_index_token(parts[1], "payer")
            payer_rows.setdefault(index, {})[parts[2]] = value
            continue
        if name.startswith("budget_order:"):
            parts = name.split(":")
            if len(parts) == 2:
                index = _canonical_index_token(parts[1], "budget_order")
                field = "separation"
            elif len(parts) == 3 and parts[2] == "voters":
                index = _canonical_index_token(parts[1], "budget_order")
                field = "voters"
            else:
                raise ValueError(
                    "budget_order keys must form indexed separation/voter pairs"
                )
            order_rows.setdefault(index, {})[field] = value
    payer_indices = sorted(payer_rows)
    if payer_indices != list(range(payer_count)):
        raise ValueError(
            "payer rows must exactly match the contiguous payer_count range"
        )
    identities: list[tuple[int, str]] = []
    budgets: list[float] = []
    for index in payer_indices:
        row = payer_rows[index]
        if set(row) != {"identity", "budget"}:
            raise ValueError(
                f"payer row requires payer:{index}:identity and payer:{index}:budget"
            )
        raw_identity = row["identity"]
        if not isinstance(raw_identity, tuple) or len(raw_identity) != 2:
            raise TypeError(f"payer:{index}:identity must be an index/id pair")
        voter_index, voter_id = raw_identity
        if type(voter_index) is not int or voter_index < 0:
            raise TypeError(
                f"payer:{index}:identity voter index must be a nonnegative integer"
            )
        if not isinstance(voter_id, str):
            raise TypeError(f"payer:{index}:identity voter ID must be a string")
        budget = _margin_real(row["budget"], f"payer:{index}:budget")
        if budget < 0.0:
            raise ValueError(f"payer:{index}:budget must be nonnegative")
        identities.append((voter_index, voter_id))
        budgets.append(budget)
    voter_indices = [identity[0] for identity in identities]
    if len(set(voter_indices)) != len(voter_indices):
        raise ValueError("payer rows must have unique voter indices")
    canonical_payer_keys = tuple(
        (budget, identity[0])
        for identity, budget in zip(identities, budgets)
    )
    if any(
        left > right
        for left, right in zip(canonical_payer_keys, canonical_payer_keys[1:])
    ):
        raise ValueError(
            "payer rows must follow canonical (budget, voter index) order"
        )
    budgets_in_supporter_order = (
        budget
        for (_, _), budget in sorted(
            zip(identities, budgets), key=lambda item: item[0][0]
        )
    )
    if sum(budgets_in_supporter_order) != total_available:
        raise ValueError("payer budgets do not sum to total_available")

    order_indices = sorted(order_rows)
    expected_order_count = max(0, payer_count - 1) if require_order else 0
    if order_indices != list(range(expected_order_count)):
        raise ValueError("budget_order indices must be contiguous")
    for index in order_indices:
        row = order_rows[index]
        if set(row) != {"separation", "voters"}:
            raise ValueError("each budget_order row requires separation and voters")
        separation = _margin_real(
            row["separation"], f"budget_order:{index}"
        )
        if separation < 0.0:
            raise ValueError("budget_order separations must be nonnegative")
        expected_separation = budgets[index + 1] - budgets[index]
        if separation != expected_separation:
            raise ValueError(
                f"budget_order:{index} separation contradicts explicit payer budgets"
            )
        raw_link = row["voters"]
        if not isinstance(raw_link, tuple) or len(raw_link) != 2:
            raise TypeError("budget_order voter links must contain two voters")
        parsed: list[tuple[int, str]] = []
        for raw_voter in raw_link:
            if not isinstance(raw_voter, tuple) or len(raw_voter) != 2:
                raise TypeError("budget_order voters must be index/id pairs")
            voter_index, voter_id = raw_voter
            if type(voter_index) is not int or voter_index < 0:
                raise TypeError("budget_order voter indices must be nonnegative integers")
            if not isinstance(voter_id, str):
                raise TypeError("budget_order voter IDs must be strings")
            parsed.append((voter_index, voter_id))
        expected_link = (identities[index], identities[index + 1])
        if tuple(parsed) != expected_link:
            raise ValueError(
                f"budget_order:{index} voter link contradicts explicit payer identities"
            )
    return tuple(
        (voter_index, voter_id, budget)
        for (voter_index, voter_id), budget in zip(identities, budgets)
    )


def _payment_candidate_key(
    margins: Mapping[str, MarginValue], project: str, rho: float
) -> tuple[float, float, str]:
    value = margins.get("candidate_key")
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("candidate_key must be a three-field tuple")
    key_rho = _margin_real(value[0], "candidate_key rho")
    key_cost = _margin_real(value[1], "candidate_key cost")
    key_project = value[2]
    if not isinstance(key_project, str):
        raise TypeError("candidate_key project must be a string")
    retained_cost = _margin_real(margins.get("cost"), "candidate_key retained cost")
    if (
        key_rho != rho
        or key_cost != retained_cost
        or key_project != project
    ):
        raise ValueError("candidate_key must equal (rho, retained cost, project)")
    return key_rho, key_cost, key_project


def _payment_charge_rows(
    margins: Mapping[str, MarginValue], rho: float
) -> tuple[dict[int, float], dict[int, float], float]:
    fields = {"branch", "budget_before", "charge", "budget_after"}
    by_index: dict[int, dict[str, MarginValue]] = {}
    for name, value in margins.items():
        if not name.startswith("approver:"):
            continue
        parts = name.split(":")
        if len(parts) != 3 or parts[2] not in fields:
            raise ValueError("payment-charge approver key shape must be an indexed quartet")
        index = _canonical_index_token(parts[1], "approver")
        by_index.setdefault(index, {})[parts[2]] = value
    if not by_index:
        raise ValueError("payment-charge requires approver branch evidence")
    budgets_before: dict[int, float] = {}
    budgets_after: dict[int, float] = {}
    charges: list[float] = []
    for index in sorted(by_index):
        row = by_index[index]
        missing = fields - set(row)
        if missing:
            field = sorted(missing)[0]
            raise ValueError(
                f"payment-charge approver quartet requires approver:{index}:{field}"
            )
        prefix = f"approver:{index}"
        branch = _margin_real(row["branch"], f"{prefix}:branch")
        before = _margin_real(
            row["budget_before"], f"{prefix}:budget_before"
        )
        charge = _margin_real(row["charge"], f"{prefix}:charge")
        after = _margin_real(
            row["budget_after"], f"{prefix}:budget_after"
        )
        if before < 0.0:
            raise ValueError(f"{prefix}:budget_before must be nonnegative")
        if branch != abs(before - rho):
            raise ValueError(f"{prefix}:branch contradicts budget_before and rho")
        expected_charge = min(before, rho)
        if charge != expected_charge:
            raise ValueError(f"{prefix}:charge must equal min(budget_before, rho)")
        if after != before - charge:
            raise ValueError(f"{prefix}:budget_after contradicts budget_before and charge")
        budgets_before[index] = before
        budgets_after[index] = after
        charges.append(charge)
    return budgets_before, budgets_after, math.fsum(charges)


def _payment_charge_total(
    margins: Mapping[str, MarginValue], rho: float
) -> float:
    return _payment_charge_rows(margins, rho)[2]


def _initial_budget_rows(
    raw_rows: Sequence[tuple[int, str, float]], *, required: bool
) -> tuple[tuple[int, str, float], ...]:
    if isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
        raise TypeError("initial budget state must be a sequence of indexed rows")
    rows: list[tuple[int, str, float]] = []
    for position, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, (tuple, list)) or len(raw_row) != 3:
            raise TypeError(
                "initial budget state rows must contain voter index, voter ID, and budget"
            )
        voter_index, voter_id, raw_budget = raw_row
        if type(voter_index) is not int or voter_index < 0:
            raise TypeError(
                "initial budget state voter indices must be nonnegative integers"
            )
        if voter_index != position:
            raise ValueError("initial budget state indices must be contiguous")
        if not isinstance(voter_id, str):
            raise TypeError("initial budget state voter IDs must be strings")
        budget = _margin_real(raw_budget, f"initial budget state voter {voter_index}")
        if budget < 0.0:
            raise ValueError("initial budget state budgets must be nonnegative")
        rows.append((voter_index, voter_id, budget))
    if required and not rows:
        raise ValueError("payment trace requires an initial budget state")
    return tuple(rows)


def _completion_sort_key(
    margins: Mapping[str, MarginValue], project: str, *, candidate: bool
) -> tuple[int, float, str]:
    approval_count = margins.get("approval_count")
    if type(approval_count) is not int or approval_count < 0:
        raise TypeError("sort_key requires a nonnegative integer approval_count")
    if candidate and approval_count == 0:
        raise ValueError("completion-candidate sort_key requires positive approval_count")
    value = margins.get("sort_key")
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("sort_key must be a three-field tuple")
    negative_count, raw_cost, key_project = value
    if type(negative_count) is not int or negative_count != -approval_count:
        raise ValueError("sort_key count must equal negative approval_count")
    cost = _margin_real(raw_cost, "sort_key cost")
    if not isinstance(key_project, str) or key_project != project:
        raise ValueError("sort_key project must match the trace-step project")
    return negative_count, cost, key_project


_PAYMENT_INCUMBENT_FIELDS = frozenset(
    {"incumbent_key", "rho_order", "cost_order", "project_id_order"}
)
_COMPLETION_HISTORY_FIELDS = frozenset(
    {
        "previous_sort_key",
        "approval_count_order",
        "cost_order",
        "project_id_order",
    }
)


def _reject_present(
    margins: Mapping[str, MarginValue], names: frozenset[str], message: str
) -> None:
    present = names.intersection(margins)
    if present:
        raise ValueError(f"{message}: {sorted(present)}")


def _validate_payment_round_history(payment_round: Sequence[TraceStep]) -> None:
    incumbent: tuple[float, float, str] | None = None
    for candidate in payment_round:
        margins = dict(candidate.comparison_margins)
        if candidate.rho is None:
            _reject_present(
                margins,
                _PAYMENT_INCUMBENT_FIELDS,
                "undefined-rho candidate has stray incumbent evidence",
            )
            continue
        key = _payment_candidate_key(margins, candidate.project, candidate.rho)
        if incumbent is None:
            _reject_present(
                margins,
                _PAYMENT_INCUMBENT_FIELDS,
                "first defined candidate has stray incumbent evidence",
            )
        else:
            if "incumbent_key" not in margins or "rho_order" not in margins:
                raise ValueError(
                    "later defined candidate requires incumbent_key and rho_order"
                )
            retained_incumbent = _payment_history_key(
                margins.get("incumbent_key"), "incumbent_key"
            )
            if retained_incumbent != incumbent:
                raise ValueError("incumbent_key must equal the current generated incumbent")
            rho_order = _margin_real(margins.get("rho_order"), "rho_order")
            if rho_order != abs(key[0] - incumbent[0]):
                raise ValueError("rho_order contradicts candidate and incumbent keys")
            if key[0] == incumbent[0]:
                if "cost_order" not in margins:
                    raise ValueError(
                        "equal-rho candidate requires cost_order evidence"
                    )
                cost_order = _margin_real(margins.get("cost_order"), "cost_order")
                if cost_order != abs(key[1] - incumbent[1]):
                    raise ValueError("cost_order contradicts candidate and incumbent keys")
                if key[1] == incumbent[1]:
                    expected_projects = tuple(sorted((incumbent[2], key[2])))
                    if margins.get("project_id_order") != expected_projects:
                        raise ValueError(
                            "project_id_order contradicts candidate and incumbent keys"
                        )
                elif "project_id_order" in margins:
                    raise ValueError("project_id_order is stray when candidate costs differ")
            else:
                _reject_present(
                    margins,
                    frozenset({"cost_order", "project_id_order"}),
                    "cost/project evidence is stray when candidate rhos differ",
                )
        if incumbent is None or key < incumbent:
            incumbent = key


def _validate_completion_order_history(
    steps: Sequence[TraceStep],
    keys: Sequence[tuple[int, float, str]],
) -> tuple[float, ...]:
    spent_values: list[float] = []
    for index, (step, key) in enumerate(zip(steps, keys)):
        margins = dict(step.comparison_margins)
        if "spent_before_completion" not in margins:
            raise ValueError("completion-order requires spent_before_completion")
        spent_values.append(
            _margin_real(
                margins["spent_before_completion"], "spent_before_completion"
            )
        )
        if index == 0:
            _reject_present(
                margins,
                _COMPLETION_HISTORY_FIELDS,
                "first completion-order row has stray history evidence",
            )
            continue
        previous = keys[index - 1]
        retained_previous = _completion_history_key(
            margins.get("previous_sort_key"), "previous_sort_key"
        )
        if retained_previous != previous:
            raise ValueError(
                "previous_sort_key must equal the preceding completion sort_key"
            )
        previous_count = -previous[0]
        count = -key[0]
        if count != previous_count:
            approval_order = _margin_real(
                margins.get("approval_count_order"), "approval_count_order"
            )
            if approval_order != abs(count - previous_count):
                raise ValueError(
                    "approval_count_order contradicts adjacent completion keys"
                )
            _reject_present(
                margins,
                frozenset({"cost_order", "project_id_order"}),
                "completion-order cost/project evidence is stray when counts differ",
            )
        else:
            if "approval_count_order" in margins:
                raise ValueError(
                    "approval_count_order is stray when completion counts are equal"
                )
            cost_order = _margin_real(margins.get("cost_order"), "cost_order")
            if cost_order != abs(key[1] - previous[1]):
                raise ValueError("cost_order contradicts adjacent completion keys")
            if key[1] == previous[1]:
                expected_projects = tuple(sorted((previous[2], key[2])))
                if margins.get("project_id_order") != expected_projects:
                    raise ValueError(
                        "project_id_order contradicts adjacent completion keys"
                    )
            elif "project_id_order" in margins:
                raise ValueError(
                    "project_id_order is stray when completion costs differ"
                )
    return tuple(spent_values)


@dataclass(frozen=True)
class TraceStep:
    """One canonical payment or completion branch observation."""

    phase: str
    project: str
    rho: float | None
    remaining_budget: float
    comparison_margins: Sequence[tuple[str, MarginValue]] | Mapping[str, MarginValue]

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str):
            raise TypeError("phase must be a string")
        if not isinstance(self.project, str):
            raise TypeError("project must be a string")
        phase = self.phase
        project = self.project
        if phase not in _TRACE_PHASES:
            raise ValueError(f"unsupported trace phase: {phase}")
        if isinstance(self.comparison_margins, Mapping):
            source = self.comparison_margins.items()
        else:
            source = self.comparison_margins
        canonical_margins = tuple(
            (str(name), _freeze_margin_value(value)) for name, value in source
        )
        margin_names = tuple(name for name, _ in canonical_margins)
        if len(set(margin_names)) != len(margin_names):
            raise ValueError("duplicate trace margin names are not allowed")
        margins = tuple(sorted(canonical_margins, key=lambda item: item[0]))
        for name, value in margins:
            if _is_certificate_margin(name):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(
                        f"certificate margin {name} must be an actual number"
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(f"certificate margin {name} must be finite")
                if value < 0:
                    raise ValueError(
                        f"certificate margin {name} must be nonnegative"
                    )
        rho = None if self.rho is None else _finite_real(self.rho, "rho")
        remaining = _finite_real(self.remaining_budget, "remaining_budget")
        if phase.startswith("payment-") and remaining < 0.0:
            raise ValueError("payment remaining budget must be nonnegative")
        if phase.startswith("payment-") and rho is not None and rho <= 0.0:
            raise ValueError("defined payment rho must be strictly positive")
        if phase == "payment-charge" and rho is None:
            raise ValueError("payment-charge rho is required")
        if phase in {"completion-order", "completion-candidate"} and rho is not None:
            raise ValueError(f"{phase} rho must be None")
        margin_map = dict(margins)
        if phase == "payment-candidate":
            if "affordability" not in margin_map:
                raise ValueError("payment-candidate requires affordability evidence")
            if "cost" not in margin_map:
                raise ValueError("payment-candidate requires retained cost")
            if "total_available" not in margin_map:
                raise ValueError("payment-candidate requires retained total_available")
            cost = _margin_real(margin_map["cost"], "cost")
            total_available = _margin_real(
                margin_map["total_available"], "total_available"
            )
            if cost <= 0.0:
                raise ValueError("cost must be positive for a payment candidate")
            if total_available < 0.0:
                raise ValueError("total_available must be nonnegative")
            if total_available - remaining > 1e-12:
                raise ValueError(
                    "total_available cannot exceed the payment remaining budget"
                )
            affordability = _margin_real(
                margin_map["affordability"], "affordability"
            )
            expected_affordability = abs(total_available - (cost - 1e-9))
            if affordability != expected_affordability:
                raise ValueError(
                    "payment-candidate affordability contradicts cost and total_available"
                )
            affordability_result = margin_map.get("affordability_result")
            if affordability_result not in {"remove", "continue"}:
                raise ValueError(
                    "payment-candidate affordability_result must be remove or continue"
                )
            if rho is not None and affordability_result != "continue":
                raise ValueError(
                    "payment-candidate with defined rho must continue affordability"
                )
            expected_affordability_result = (
                "remove"
                if total_available < cost - 1e-9
                else "continue"
            )
            if affordability_result != expected_affordability_result:
                raise ValueError(
                    "payment-candidate affordability_result contradicts generated comparison"
                )
            payer_vector = _payment_payer_vector(
                margin_map,
                total_available,
                require_order=affordability_result == "continue",
            )
            breakpoint_fields = {"base", "budget", "candidate", "result"}
            breakpoint_rows: dict[int, dict[str, MarginValue]] = {}
            for name, value in margin_map.items():
                if not name.startswith("breakpoint:"):
                    continue
                parts = name.split(":")
                if len(parts) == 2:
                    index = _canonical_index_token(parts[1], "breakpoint")
                    field = "base"
                elif (
                    len(parts) == 3
                    and parts[2] in breakpoint_fields - {"base"}
                ):
                    index = _canonical_index_token(parts[1], "breakpoint")
                    field = parts[2]
                else:
                    raise ValueError(
                        "payment-candidate breakpoint result key shape is invalid"
                    )
                breakpoint_rows.setdefault(index, {})[field] = value
            breakpoint_indices = sorted(breakpoint_rows)
            if breakpoint_indices and breakpoint_indices != list(
                range(len(breakpoint_indices))
            ):
                raise ValueError("payment-candidate breakpoint keys must be contiguous")
            breakpoint_results = []
            breakpoint_budgets = []
            breakpoint_candidates = []
            paid_so_far = 0.0
            for index in breakpoint_indices:
                row = breakpoint_rows[index]
                missing = breakpoint_fields - set(row)
                if missing:
                    field = sorted(missing)[0]
                    if field == "base":
                        raise ValueError(
                            "payment-candidate breakpoint result key shape is invalid"
                        )
                    name = f"breakpoint:{index}" if field == "base" else f"breakpoint:{index}:{field}"
                    raise ValueError(f"payment-candidate requires {name}")
                budget = _margin_real(
                    row["budget"],
                    f"breakpoint:{index}:budget",
                )
                candidate = _margin_real(
                    row["candidate"],
                    f"breakpoint:{index}:candidate",
                )
                if budget < 0.0:
                    raise ValueError(f"breakpoint:{index}:budget must be nonnegative")
                if index >= len(payer_vector):
                    raise ValueError("breakpoint position exceeds explicit payer vector")
                expected_budget = payer_vector[index][2]
                if budget != expected_budget:
                    raise ValueError(
                        f"breakpoint:{index} budget contradicts explicit payer budget"
                    )
                expected_candidate = (cost - paid_so_far) / (
                    len(payer_vector) - index
                )
                if candidate != expected_candidate:
                    raise ValueError(
                        f"breakpoint:{index} candidate contradicts the payment equation"
                    )
                separation = _margin_real(
                    row["base"], f"breakpoint:{index}"
                )
                if separation != abs((budget + 1e-12) - candidate):
                    raise ValueError(
                        f"breakpoint:{index} margin contradicts budget and candidate"
                    )
                result = row["result"]
                if result not in {"accept", "continue"}:
                    raise ValueError(
                        f"breakpoint:{index}:result must be accept or continue"
                    )
                expected_result = (
                    "accept" if candidate <= budget + 1e-12 else "continue"
                )
                if result != expected_result:
                    raise ValueError(
                        f"breakpoint:{index}:result contradicts generated comparison for accepted breakpoint"
                    )
                breakpoint_results.append(result)
                breakpoint_budgets.append(budget)
                breakpoint_candidates.append(candidate)
                if result == "continue":
                    paid_so_far += expected_budget
            if any(
                left > right
                for left, right in zip(
                    breakpoint_budgets, breakpoint_budgets[1:]
                )
            ):
                raise ValueError("payment-candidate breakpoint budgets must be nondecreasing")
            if rho is not None:
                if not breakpoint_indices:
                    raise ValueError(
                        "accepted payment-candidate requires breakpoint evidence"
                    )
                if (
                    breakpoint_results[-1] != "accept"
                    or breakpoint_results.count("accept") != 1
                ):
                    raise ValueError(
                        "accepted payment-candidate requires one final accepted breakpoint"
                    )
                if "candidate_key" not in margin_map:
                    raise ValueError(
                        "accepted payment-candidate requires candidate_key evidence"
                    )
                if rho != breakpoint_candidates[-1]:
                    raise ValueError(
                        "rho must equal the final accepted breakpoint candidate"
                    )
                _payment_candidate_key(margin_map, project, rho)
                if "rho_result" in margin_map:
                    raise ValueError("defined-rho candidate cannot retain rho_result")
            else:
                if "candidate_key" in margin_map:
                    raise ValueError("candidate_key requires a defined rho")
                if affordability_result == "remove":
                    if breakpoint_indices or "rho_result" in margin_map:
                        raise ValueError(
                            "removed affordability candidate cannot retain rho evidence"
                        )
                elif (
                    not breakpoint_indices
                    or any(result != "continue" for result in breakpoint_results)
                    or margin_map.get("rho_result") != "remove"
                ):
                    raise ValueError(
                        "payment-candidate without rho requires rho_result remove"
                    )
                elif len(breakpoint_indices) != len(payer_vector):
                    raise ValueError(
                        "payment-candidate without rho must visit every payer breakpoint"
                    )
        elif phase == "payment-charge":
            _payment_charge_total(margin_map, rho)
        elif phase == "completion-order":
            if "sort_key" not in margin_map:
                raise ValueError("completion-order requires sort_key evidence")
            _completion_sort_key(margin_map, project, candidate=False)
        else:
            if "positive_cost" not in margin_map:
                raise ValueError(
                    "completion-candidate requires positive_cost evidence"
                )
            positive_result = margin_map.get("positive_cost_result")
            if positive_result == "continue":
                coherent = (
                    margin_map["positive_cost"] > 0
                    and "affordability" in margin_map
                    and margin_map.get("affordability_result") in {"fund", "reject"}
                )
            elif positive_result == "reject":
                coherent = (
                    "affordability" not in margin_map
                    and "affordability_result" not in margin_map
                )
            else:
                coherent = False
            if not coherent:
                raise ValueError("completion-candidate evidence is incoherent")
            _, cost, _ = _completion_sort_key(
                margin_map, project, candidate=True
            )
            positive_cost = _margin_real(
                margin_map["positive_cost"], "positive_cost"
            )
            if positive_cost != abs(cost):
                raise ValueError("completion positive_cost must equal abs(sort cost)")
            expected_positive_result = "continue" if cost > 0.0 else "reject"
            if positive_result != expected_positive_result:
                raise ValueError(
                    "completion positive_cost_result contradicts sort_key cost"
                )
            if cost > 0.0:
                affordability = _margin_real(
                    margin_map["affordability"], "completion affordability"
                )
                if affordability != abs(remaining - cost):
                    raise ValueError(
                        "completion affordability contradicts cost and remaining budget"
                    )
                expected_affordability_result = (
                    "fund" if cost <= remaining else "reject"
                )
                if (
                    margin_map.get("affordability_result")
                    != expected_affordability_result
                ):
                    raise ValueError(
                        "completion affordability_result contradicts cost and remaining budget"
                    )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "project", self.project)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "remaining_budget", remaining)
        object.__setattr__(self, "comparison_margins", margins)


@dataclass(frozen=True)
class EqualSharesTrace:
    """Immutable executed Equal Shares payment and completion trace."""

    winners: Sequence[str]
    payment_order: Sequence[str]
    completion_order: Sequence[str]
    steps: Sequence[TraceStep]
    strict: bool
    initial_budget_state: Sequence[tuple[int, str, float]] = ()

    def __post_init__(self) -> None:
        if type(self.strict) is not bool:
            raise TypeError("strict must be a bool")
        winners = tuple(self.winners)
        payment_order = tuple(self.payment_order)
        completion_order = tuple(self.completion_order)
        if any(
            not isinstance(value, str)
            for values in (winners, payment_order, completion_order)
            for value in values
        ):
            raise TypeError("winner and order identifiers must be strings")
        steps = tuple(self.steps)
        if len(set(winners)) != len(winners):
            raise ValueError("winners must be unique")
        if len(set(payment_order)) != len(payment_order):
            raise ValueError("payment order must be unique")
        if len(set(completion_order)) != len(completion_order):
            raise ValueError("completion order must be unique")
        if set(payment_order) & set(completion_order):
            raise ValueError("payment and completion orders must be disjoint")
        if set(payment_order) | set(completion_order) != set(winners):
            raise ValueError("payment and completion orders must cover winners")
        if any(not isinstance(step, TraceStep) for step in steps):
            raise TypeError("every trace step must be a TraceStep")
        has_payment_trace = any(step.phase.startswith("payment-") for step in steps)
        initial_budget_state = _initial_budget_rows(
            self.initial_budget_state, required=False
        )
        replay_voter_ids = tuple(row[1] for row in initial_budget_state)
        replay_budgets = [row[2] for row in initial_budget_state]
        payment_evidence = tuple(
            step.project for step in steps if step.phase == "payment-charge"
        )
        if payment_order != payment_evidence:
            raise ValueError("payment order must exactly match payment-charge evidence")
        funded_completion_evidence = tuple(
            step.project
            for step in steps
            if step.phase == "completion-candidate"
            and dict(step.comparison_margins).get("affordability_result") == "fund"
        )
        if completion_order != funded_completion_evidence:
            raise ValueError(
                "completion order must exactly match funded completion-candidate evidence"
            )
        completion_order_evidence = tuple(
            step.project for step in steps if step.phase == "completion-order"
        )
        if len(set(completion_order_evidence)) != len(completion_order_evidence):
            raise ValueError("completion-order evidence must be unique by project")
        completion_candidate_evidence = tuple(
            step.project for step in steps if step.phase == "completion-candidate"
        )
        if len(set(completion_candidate_evidence)) != len(
            completion_candidate_evidence
        ):
            raise ValueError("completion-candidate projects must be unique")
        completion_positions = {
            project: index
            for index, project in enumerate(completion_order_evidence)
        }
        candidate_positions = [
            completion_positions.get(project, -1)
            for project in completion_candidate_evidence
        ]
        if all(position >= 0 for position in candidate_positions) and any(
            left >= right
            for left, right in zip(candidate_positions, candidate_positions[1:])
        ):
            raise ValueError(
                "completion candidates must form an order-preserving subsequence"
            )
        completion_started = False
        completion_candidates_started = False
        for step in steps:
            if step.phase.startswith("payment-"):
                if completion_started:
                    raise ValueError("payment phases cannot follow completion phases")
            else:
                completion_started = True
                if step.phase == "completion-order":
                    if completion_candidates_started:
                        raise ValueError(
                            "completion-order rows cannot follow completion candidates"
                        )
                else:
                    completion_candidates_started = True
        payment_round: list[TraceStep] = []
        unavailable_projects: set[str] = set()
        preceding_charge_remaining: float | None = None
        selected_payment_costs: dict[str, float] = {}
        for step in steps:
            if step.phase == "payment-candidate":
                if (
                    initial_budget_state
                    and step.remaining_budget != sum(replay_budgets)
                ):
                    raise ValueError(
                        "payment candidate remaining budget contradicts "
                        "the replayed initial budget state"
                    )
                candidate_margins = dict(step.comparison_margins)
                if initial_budget_state:
                    candidate_total = _margin_real(
                        candidate_margins["total_available"], "total_available"
                    )
                    candidate_payers = _payment_payer_vector(
                        candidate_margins,
                        candidate_total,
                        require_order=(
                            candidate_margins.get("affordability_result")
                            == "continue"
                        ),
                    )
                    payer_indices: list[int] = []
                    for voter_index, voter_id, budget in candidate_payers:
                        if voter_index >= len(replay_budgets):
                            raise ValueError(
                                "candidate payer index exceeds the initial budget state"
                            )
                        if replay_voter_ids[voter_index] != voter_id:
                            raise ValueError(
                                "candidate payer identity contradicts the initial budget state"
                            )
                        if replay_budgets[voter_index] != budget:
                            raise ValueError(
                                "candidate payer budget contradicts the replayed budget state"
                            )
                        payer_indices.append(voter_index)
                    expected_candidate_total = sum(
                        replay_budgets[index] for index in sorted(payer_indices)
                    )
                    if candidate_total != expected_candidate_total:
                        raise ValueError(
                            "candidate total_available contradicts the replayed payer state"
                        )
                if step.project in unavailable_projects:
                    raise ValueError(
                        "a removed or charged project cannot reappear in a later payment round"
                    )
                payment_round.append(step)
                continue
            if step.phase != "payment-charge":
                break
            if not payment_round:
                raise ValueError(
                    "payment charge requires matching current-round candidate evidence"
                )
            projects = [candidate.project for candidate in payment_round]
            if len(set(projects)) != len(projects):
                raise ValueError("a project may occur only once in a payment round")
            if projects != sorted(projects):
                raise ValueError(
                    "payment round projects must follow ascending generated order"
                )
            _validate_payment_round_history(payment_round)
            round_remaining = {
                candidate.remaining_budget for candidate in payment_round
            }
            if len(round_remaining) > 1:
                raise ValueError(
                    "all payment round candidates must share one remaining budget"
                )
            before_remaining = next(iter(round_remaining))
            if (
                preceding_charge_remaining is not None
                and before_remaining != preceding_charge_remaining
            ):
                raise ValueError(
                    "next payment round must start at the prior charge remaining budget"
                )
            defined = [candidate for candidate in payment_round if candidate.rho is not None]
            matching = [
                candidate
                for candidate in defined
                if candidate.project == step.project and candidate.rho == step.rho
            ]
            if len(matching) != 1:
                raise ValueError(
                    "payment charge requires matching current-round candidate evidence"
                )
            minimum = min(
                defined,
                key=lambda candidate: _payment_candidate_key(
                    dict(candidate.comparison_margins),
                    candidate.project,
                    candidate.rho,
                ),
            )
            if minimum is not matching[0]:
                raise ValueError("payment charge must use the minimum candidate key")
            (
                charge_budgets,
                post_charge_budgets,
                _,
            ) = _payment_charge_rows(dict(step.comparison_margins), step.rho)
            selected_margins = dict(matching[0].comparison_margins)
            selected_total = _margin_real(
                selected_margins["total_available"], "total_available"
            )
            selected_payers = _payment_payer_vector(
                selected_margins, selected_total, require_order=True
            )
            explicit_payer_budgets = {
                payer_index: budget
                for payer_index, _, budget in selected_payers
            }
            payer_budgets_match = set(explicit_payer_budgets) == set(
                charge_budgets
            ) and all(
                explicit_payer_budgets[index] == charge_budgets[index]
                for index in explicit_payer_budgets
            )
            if not payer_budgets_match:
                raise ValueError(
                    "selected candidate and charge payer budgets are inconsistent"
                )
            selected_cost = _margin_real(selected_margins["cost"], "cost")
            # The payment candidate has already replayed the forward binary64
            # rho equation exactly.  Summing rounded per-voter charges is not
            # its inverse and can differ from the retained cost by several ulps.
            if initial_budget_state:
                for voter_index, budget_after in post_charge_budgets.items():
                    replay_budgets[voter_index] = budget_after
                expected_remaining = sum(replay_budgets)
                if step.remaining_budget != expected_remaining:
                    raise ValueError(
                        "payment-charge remaining budget transition is inconsistent"
                    )
            unavailable_projects.update(
                candidate.project
                for candidate in payment_round
                if candidate.rho is None
            )
            unavailable_projects.add(step.project)
            preceding_charge_remaining = step.remaining_budget
            selected_payment_costs[step.project] = selected_cost
            payment_round = []
        terminal_projects = [candidate.project for candidate in payment_round]
        if len(set(terminal_projects)) != len(terminal_projects):
            raise ValueError("a project may occur only once in a payment round")
        if terminal_projects != sorted(terminal_projects):
            raise ValueError(
                "payment round projects must follow ascending generated order"
            )
        _validate_payment_round_history(payment_round)
        terminal_remaining = {
            candidate.remaining_budget for candidate in payment_round
        }
        if len(terminal_remaining) > 1:
            raise ValueError(
                "terminal payment round candidates must share one remaining budget"
            )
        if payment_round and preceding_charge_remaining is not None:
            terminal_value = next(iter(terminal_remaining))
            if terminal_value != preceding_charge_remaining:
                raise ValueError(
                    "next payment round must start at the prior charge remaining budget"
                )
        if any(candidate.rho is not None for candidate in payment_round):
            raise ValueError("terminal defined-rho payment candidates require a charge")

        completion_order_steps = tuple(
            step for step in steps if step.phase == "completion-order"
        )
        if completion_order_steps:
            initial_completion_remaining = completion_order_steps[0].remaining_budget
            first_completion_margins = dict(
                completion_order_steps[0].comparison_margins
            )
            payment_spend_order = _payment_spend_order(
                first_completion_margins.get("payment_spend_order"), payment_order
            )
            if any(
                "payment_spend_order" in dict(step.comparison_margins)
                for step in completion_order_steps[1:]
            ):
                raise ValueError(
                    "payment_spend_order belongs only on the first completion-order row"
                )
            if any(
                step.remaining_budget != initial_completion_remaining
                for step in completion_order_steps[1:]
            ):
                raise ValueError(
                    "completion-order rows must share one initial remaining budget"
                )
            if not set(payment_order).issubset(completion_order_evidence):
                raise ValueError(
                    "every payment winner requires completion-order evidence"
                )
        else:
            initial_completion_remaining = None
            payment_spend_order = ()
        completion_keys = tuple(
            _completion_sort_key(
                dict(step.comparison_margins), step.project, candidate=False
            )
            for step in completion_order_steps
        )
        if any(
            left > right
            for left, right in zip(completion_keys, completion_keys[1:])
        ):
            raise ValueError("completion-order keys must be nondecreasing")
        completion_spent = _validate_completion_order_history(
            completion_order_steps, completion_keys
        )
        if completion_spent:
            if any(
                value != completion_spent[0]
                for value in completion_spent[1:]
            ):
                raise ValueError(
                    "spent_before_completion must be common to all completion rows"
                )
            # Replay the reference's built-in sum over the retained winner-set
            # iteration order.  Reordering or replacing this with fsum would
            # use different binary64 accumulator transitions.
            expected_spent = sum(
                selected_payment_costs[project]
                for project in payment_spend_order
            )
            if completion_spent[0] != expected_spent:
                raise ValueError(
                    "spent_before_completion must equal selected payment candidate costs"
                )
        completion_key_by_project = {
            step.project: key
            for step, key in zip(completion_order_steps, completion_keys)
        }
        expected_completion_remaining = initial_completion_remaining
        completion_candidate_count = 0
        for index, step in enumerate(steps):
            if step.phase == "completion-candidate":
                margins = dict(step.comparison_margins)
                if step.project in payment_order:
                    raise ValueError(
                        "a completion candidate cannot repeat a payment winner"
                    )
                has_order_evidence = any(
                    order_step.phase == "completion-order"
                    and order_step.project == step.project
                    for order_step in steps[:index]
                )
                if not has_order_evidence:
                    if margins.get("affordability_result") == "fund":
                        raise ValueError(
                            "funded completion requires prior completion-order evidence"
                        )
                    raise ValueError(
                        "completion candidate requires prior completion-order evidence"
                    )
                candidate_key = _completion_sort_key(
                    margins, step.project, candidate=True
                )
                if completion_key_by_project.get(step.project) != candidate_key:
                    raise ValueError(
                        "completion-candidate sort_key must match completion-order evidence"
                    )
                if (
                    expected_completion_remaining is not None
                    and step.remaining_budget != expected_completion_remaining
                ):
                    if completion_candidate_count == 0:
                        raise ValueError(
                            "first completion candidate must start at the initial remaining budget"
                        )
                    raise ValueError(
                        "completion candidate remaining budget transition is inconsistent"
                    )
                if margins.get("affordability_result") == "fund":
                    expected_completion_remaining -= candidate_key[1]
                completion_candidate_count += 1
        required_completion_candidates = tuple(
            step.project
            for step, key in zip(completion_order_steps, completion_keys)
            if -key[0] > 0 and step.project not in payment_order
        )
        if completion_candidate_evidence != required_completion_candidates:
            raise ValueError(
                "completion candidate sequence must equal retained eligible projects"
            )
        if has_payment_trace and not initial_budget_state:
            raise ValueError("payment trace requires an initial budget state")
        derived_strict = all(
            value > 0
            for step in steps
            for name, value in step.comparison_margins
            if _is_certificate_margin(name)
        )
        if self.strict != derived_strict:
            raise ValueError("strict flag must equal derived certificate strictness")
        object.__setattr__(self, "winners", tuple(sorted(winners)))
        object.__setattr__(self, "payment_order", payment_order)
        object.__setattr__(self, "completion_order", completion_order)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "initial_budget_state", initial_budget_state)


@dataclass(frozen=True)
class ActuationRecord:
    """First observed directional winner-signature change for one election."""

    instance: str
    actuated: bool
    first_grid_t: float | None
    refined_low: float | None
    refined_high: float | None
    csd_direction: str | None
    anchor_csd: float | None = None
    changed_csd: float | None = None

    def __post_init__(self) -> None:
        if type(self.actuated) is not bool:
            raise TypeError("actuated must be a bool")
        if not isinstance(self.instance, str):
            raise TypeError("instance must be a string")
        optional_numbers = {}
        for name in ("first_grid_t", "refined_low", "refined_high", "anchor_csd", "changed_csd"):
            value = getattr(self, name)
            if value is None:
                optional_numbers[name] = None
            else:
                optional_numbers[name] = _finite_real(value, name)
        direction = self.csd_direction
        if direction not in {None, "improves", "ties", "worsens"}:
            raise ValueError("csd_direction must be improves, ties, worsens, or None")
        actuated = self.actuated
        threshold_values = (
            optional_numbers["first_grid_t"],
            optional_numbers["refined_low"],
            optional_numbers["refined_high"],
        )
        if actuated and any(value is None for value in threshold_values):
            raise ValueError("actuated records require a complete threshold bracket")
        if not actuated and any(value is not None for value in threshold_values):
            raise ValueError("inert records cannot contain a threshold bracket")
        if actuated:
            first_grid_t, refined_low, refined_high = threshold_values
            assert first_grid_t is not None
            assert refined_low is not None
            assert refined_high is not None
            grid_index = round(first_grid_t * 64.0)
            if not 1 <= grid_index <= 64 or first_grid_t != grid_index / 64.0:
                raise ValueError("first_grid_t must lie on the k/64 grid")
            preceding_grid = (grid_index - 1) / 64.0
            if not (
                preceding_grid
                <= refined_low
                < refined_high
                <= first_grid_t
                <= 1.0
            ):
                raise ValueError("actuation threshold bracket is invalid")
            for name, value in (
                ("refined_low", refined_low),
                ("refined_high", refined_high),
            ):
                if value != round(value * 16384.0) / 16384.0:
                    raise ValueError(f"{name} must lie on the 1/16384 lattice")
            base_units = round(preceding_grid * 16384.0)
            low_units = round(refined_low * 16384.0) - base_units
            high_units = round(refined_high * 16384.0) - base_units
            width_units = high_units - low_units
            reachable_widths = {256 // (2**depth) for depth in range(9)}
            if (
                width_units not in reachable_widths
                or low_units % width_units != 0
            ):
                raise ValueError(
                    "actuation bracket is unreachable after eight-step bisection"
                )
        anchor_csd = optional_numbers["anchor_csd"]
        changed_csd = optional_numbers["changed_csd"]
        if any(
            value is not None and value > 1.0
            for value in (anchor_csd, changed_csd)
        ):
            raise ValueError("raw CSD values must be at most 1.0")
        if not actuated and (changed_csd is not None or direction is not None):
            raise ValueError("inert records cannot contain changed CSD fields")
        if anchor_csd is None or changed_csd is None:
            if direction is not None:
                raise ValueError("CSD direction requires both raw CSD values")
        else:
            difference = changed_csd - anchor_csd
            expected_direction = (
                "improves"
                if difference < -1e-12
                else "worsens"
                if difference > 1e-12
                else "ties"
            )
            if direction != expected_direction:
                raise ValueError("csd_direction contradicts the raw CSD values")
            direction = expected_direction
        object.__setattr__(self, "instance", self.instance)
        object.__setattr__(self, "actuated", actuated)
        object.__setattr__(self, "csd_direction", direction)
        for name, value in optional_numbers.items():
            object.__setattr__(self, name, value)


def _separation(value: float) -> float:
    return abs(float(value))


def _is_certificate_margin(name: str) -> bool:
    if name in {"affordability", "rho_order", "cost_order", "positive_cost"}:
        return True
    if name.startswith("budget_order:"):
        parts = name.split(":")
        _canonical_index_token(parts[1], "budget_order")
        return len(parts) == 2
    if name.startswith("breakpoint:"):
        parts = name.split(":")
        _canonical_index_token(parts[1], "breakpoint")
        return len(parts) == 2
    if name.startswith("approver:"):
        parts = name.split(":")
        _canonical_index_token(parts[1], "approver")
        return len(parts) == 3 and parts[2] == "branch"
    return False


def _validate_endowments(inst: PBInstance, endowments: Sequence[float]) -> list[float]:
    if len(endowments) != len(inst.votes):
        raise ValueError("one endowment is required for every voter")
    budgets = [float(value) for value in endowments]
    if any(not math.isfinite(value) or value < 0.0 for value in budgets):
        raise ValueError("endowments must be finite and nonnegative")
    return budgets


def trace_equal_shares(
    inst: PBInstance,
    endowments: Sequence[float],
    completion: bool = True,
) -> EqualSharesTrace:
    """Mirror ``rules.mes_with_endowments`` and retain its executed branches."""

    if not inst.votes:
        return EqualSharesTrace((), (), (), (), True)
    budgets = _validate_endowments(inst, endowments)
    initial_budget_state = tuple(
        (index, str(vote.vid), budgets[index])
        for index, vote in enumerate(inst.votes)
    )
    approver_ids: dict[str, list[int]] = {pid: [] for pid in inst.projects}
    for index, vote in enumerate(inst.votes):
        for project in vote.projects:
            if project in approver_ids:
                approver_ids[project].append(index)

    active = {
        pid
        for pid, supporters in approver_ids.items()
        if supporters and inst.projects[pid].cost > 0
    }
    winners: set[str] = set()
    payment_order: list[str] = []
    completion_order: list[str] = []
    steps: list[TraceStep] = []
    strict = True

    while active:
        best_pid: str | None = None
        best_rho: float | None = None
        best_key: tuple[float, float, str] | None = None
        for pid in sorted(active):
            cost = float(inst.projects[pid].cost)
            supporters = approver_ids[pid]
            total_available = sum(budgets[index] for index in supporters)
            sorted_payers = sorted((budgets[index], index) for index in supporters)
            affordability_gap = total_available - (cost - 1e-9)
            affordability_margin = _separation(affordability_gap)
            strict = strict and affordability_margin > 0.0
            margins: dict[str, MarginValue] = {
                "affordability": affordability_margin,
                "affordability_result": (
                    "remove" if total_available < cost - 1e-9 else "continue"
                ),
                "cost": cost,
                "payer_count": len(sorted_payers),
                "total_available": total_available,
            }
            for position, (budget, voter_index) in enumerate(sorted_payers):
                margins[f"payer:{position}:identity"] = (
                    voter_index,
                    str(inst.votes[voter_index].vid),
                )
                margins[f"payer:{position}:budget"] = budget
            if total_available < cost - 1e-9:
                steps.append(
                    TraceStep(
                        "payment-candidate",
                        pid,
                        None,
                        sum(budgets),
                        margins,
                    )
                )
                active.discard(pid)
                continue

            for position, (left, right) in enumerate(
                zip(sorted_payers, sorted_payers[1:])
            ):
                order_margin = _separation(right[0] - left[0])
                margins[f"budget_order:{position}"] = order_margin
                margins[f"budget_order:{position}:voters"] = (
                    (left[1], str(inst.votes[left[1]].vid)),
                    (right[1], str(inst.votes[right[1]].vid)),
                )
                strict = strict and order_margin > 0.0
            sorted_budgets = [budget for budget, _ in sorted_payers]
            payer_count = len(sorted_budgets)
            paid_so_far = 0.0
            rho: float | None = None
            for position, budget in enumerate(sorted_budgets):
                remaining_payers = payer_count - position
                candidate = (cost - paid_so_far) / remaining_payers
                breakpoint_gap = (budget + 1e-12) - candidate
                breakpoint_margin = _separation(breakpoint_gap)
                strict = strict and breakpoint_margin > 0.0
                prefix = f"breakpoint:{position}"
                margins[prefix] = breakpoint_margin
                margins[f"{prefix}:budget"] = budget
                margins[f"{prefix}:candidate"] = candidate
                margins[f"{prefix}:result"] = (
                    "accept" if candidate <= budget + 1e-12 else "continue"
                )
                if candidate <= budget + 1e-12:
                    rho = candidate
                    break
                paid_so_far += budget
            if rho is None:
                margins["rho_result"] = "remove"
                steps.append(
                    TraceStep(
                        "payment-candidate",
                        pid,
                        None,
                        sum(budgets),
                        margins,
                    )
                )
                active.discard(pid)
                continue

            candidate_key = (rho, cost, pid)
            margins["candidate_key"] = candidate_key
            if best_key is not None:
                margins["incumbent_key"] = best_key
                rho_margin = _separation(rho - best_key[0])
                margins["rho_order"] = rho_margin
                strict = strict and rho_margin > 0.0
                if rho == best_key[0]:
                    cost_margin = _separation(cost - best_key[1])
                    margins["cost_order"] = cost_margin
                    strict = strict and cost_margin > 0.0
                    if cost == best_key[1]:
                        margins["project_id_order"] = tuple(
                            sorted((best_key[2], pid))
                        )
            steps.append(
                TraceStep(
                    "payment-candidate",
                    pid,
                    rho,
                    sum(budgets),
                    margins,
                )
            )
            if best_key is None or candidate_key < best_key:
                best_pid, best_rho, best_key = pid, rho, candidate_key

        if best_pid is None or best_rho is None:
            break
        charge_margins: dict[str, MarginValue] = {}
        for index in approver_ids[best_pid]:
            budget_before = budgets[index]
            branch_margin = _separation(budget_before - best_rho)
            strict = strict and branch_margin > 0.0
            charge = min(budget_before, best_rho)
            budgets[index] -= charge
            charge_margins[f"approver:{index}:branch"] = branch_margin
            charge_margins[f"approver:{index}:budget_before"] = budget_before
            charge_margins[f"approver:{index}:charge"] = charge
            charge_margins[f"approver:{index}:budget_after"] = budgets[index]
        steps.append(
            TraceStep(
                "payment-charge",
                best_pid,
                best_rho,
                sum(budgets),
                charge_margins,
            )
        )
        winners.add(best_pid)
        payment_order.append(best_pid)
        active.discard(best_pid)

    if completion:
        counts = {pid: len(approver_ids[pid]) for pid in inst.projects}
        payment_spend_order = tuple(winners)
        spent = sum(
            float(inst.projects[pid].cost) for pid in payment_spend_order
        )
        remaining = float(inst.budget) - spent
        ordered = sorted(
            counts,
            key=lambda pid: (-counts[pid], inst.projects[pid].cost, pid),
        )
        previous: str | None = None
        for pid in ordered:
            cost = float(inst.projects[pid].cost)
            order_margins: dict[str, MarginValue] = {
                "approval_count": counts[pid],
                "sort_key": (-counts[pid], cost, pid),
                "spent_before_completion": spent,
            }
            if previous is None:
                order_margins["payment_spend_order"] = payment_spend_order
            if previous is not None:
                previous_count = counts[previous]
                order_margins["previous_sort_key"] = (
                    -previous_count,
                    float(inst.projects[previous].cost),
                    previous,
                )
                if counts[pid] == previous_count:
                    cost_margin = _separation(
                        cost - float(inst.projects[previous].cost)
                    )
                    order_margins["cost_order"] = cost_margin
                    strict = strict and cost_margin > 0.0
                    if cost == float(inst.projects[previous].cost):
                        order_margins["project_id_order"] = tuple(
                            sorted((previous, pid))
                        )
                else:
                    order_margins["approval_count_order"] = abs(
                        counts[pid] - previous_count
                    )
            steps.append(
                TraceStep("completion-order", pid, None, remaining, order_margins)
            )
            previous = pid

        for pid in ordered:
            if pid in winners or counts[pid] == 0:
                continue
            cost = float(inst.projects[pid].cost)
            positive_margin = _separation(cost)
            strict = strict and positive_margin > 0.0
            margins = {
                "approval_count": counts[pid],
                "positive_cost": positive_margin,
                "positive_cost_result": "continue" if cost > 0.0 else "reject",
                "sort_key": (-counts[pid], cost, pid),
            }
            if cost > 0.0:
                affordability_margin = _separation(remaining - cost)
                strict = strict and affordability_margin > 0.0
                margins["affordability"] = affordability_margin
                margins["affordability_result"] = (
                    "fund" if cost <= remaining else "reject"
                )
            steps.append(
                TraceStep("completion-candidate", pid, None, remaining, margins)
            )
            if 0.0 < cost <= remaining:
                winners.add(pid)
                completion_order.append(pid)
                remaining -= cost

    return EqualSharesTrace(
        winners,
        payment_order,
        completion_order,
        steps,
        strict,
        initial_budget_state,
    )


def winner_signature(inst: PBInstance, policy: EndowmentPolicy) -> tuple[str, ...]:
    """Return the canonical completed-Equal-Shares winner signature."""

    endowments = policy(inst, RolloutState())
    return trace_equal_shares(inst, endowments, completion=True).winners


def _one_election_worst_csd(
    inst: PBInstance, signature: Sequence[str]
) -> float | None:
    outcome = group_outcome(inst, set(signature), scheme="age_sex")
    values = []
    for row in outcome.values():
        entitlement = float(row["entitlement"])
        utility = float(row["utility"])
        if entitlement > 0.0 and math.isfinite(entitlement) and math.isfinite(utility):
            values.append((entitlement - utility) / entitlement)
    if not values or any(not math.isfinite(value) for value in values):
        return None
    return max(values)


def _csd_direction(anchor_csd: float | None, changed_csd: float | None) -> str | None:
    if anchor_csd is None or changed_csd is None:
        return None
    difference = changed_csd - anchor_csd
    if difference < -1e-12:
        return "improves"
    if difference > 1e-12:
        return "worsens"
    return "ties"


def scan_actuation_path(
    inst: PBInstance,
    anchor: AnchorSpec,
    scaler: FeatureScaler,
    direction: Sequence[float],
) -> ActuationRecord:
    """Scan and refine the first grid-observed directional signature change."""

    direction_values = tuple(float(value) for value in direction)
    if len(direction_values) != 4 or any(
        not math.isfinite(value) for value in direction_values
    ):
        raise ValueError("direction must contain four finite values")
    anchor_signature = winner_signature(inst, anchor_policy(anchor))
    anchor_csd = _one_election_worst_csd(inst, anchor_signature)
    first_index: int | None = None
    first_signature: tuple[str, ...] | None = None
    for index in range(1, 65):
        t = index / 64.0
        signature = winner_signature(
            inst,
            residual_policy(
                anchor,
                scaler,
                tuple(t * value for value in direction_values),
            ),
        )
        if first_index is None and signature != anchor_signature:
            first_index = index
            first_signature = signature

    if first_index is None or first_signature is None:
        return ActuationRecord(
            inst.path,
            False,
            None,
            None,
            None,
            None,
            anchor_csd,
            None,
        )

    low = (first_index - 1) / 64.0
    high = first_index / 64.0
    for _ in range(8):
        midpoint = (low + high) / 2.0
        signature = winner_signature(
            inst,
            residual_policy(
                anchor,
                scaler,
                tuple(midpoint * value for value in direction_values),
            ),
        )
        if signature == anchor_signature:
            low = midpoint
        elif signature == first_signature:
            high = midpoint
    changed_csd = _one_election_worst_csd(inst, first_signature)
    return ActuationRecord(
        inst.path,
        True,
        first_index / 64.0,
        low,
        high,
        _csd_direction(anchor_csd, changed_csd),
        anchor_csd,
        changed_csd,
    )


def _supplied_instances(data: Sequence[PBInstance]) -> tuple[PBInstance, ...]:
    instances: list[PBInstance] = []
    for item in data:
        if not isinstance(item, PBInstance):
            raise TypeError(
                "actuation data must be an explicit sequence of PBInstance objects"
            )
        instances.append(item)
    return tuple(sorted(instances, key=lambda inst: inst.path))


def scan_fold_actuation(
    data: Sequence[PBInstance], fit: object
) -> list[ActuationRecord]:
    """Scan supplied instances along the training-selected primary direction."""

    primary_seed = int(getattr(fit, "primary_seed"))
    matches = [row for row in getattr(fit, "seeds") if int(row.seed) == primary_seed]
    if len(matches) != 1:
        raise ValueError("fit must contain exactly one primary seed result")
    direction = tuple(float(value) for value in matches[0].selected.endpoint)
    anchor = fit.anchor_fit.selected.anchor
    scaler = fit.scaler
    return [
        scan_actuation_path(inst, anchor, scaler, direction)
        for inst in _supplied_instances(data)
    ]
