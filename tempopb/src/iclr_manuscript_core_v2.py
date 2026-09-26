"""Pure formatting of core manuscript macros from captured corrected-v2 consumers.

The caller authenticates the consumer bundle. This module selects exact saved
rows, checks their identities, and formats values with field-level provenance.
It performs no I/O and computes no statistical tests or confidence intervals.
"""

from __future__ import annotations

from itertools import product
from math import isfinite
from typing import Any


Record = dict[str, Any]
OUTCOME_FLOORS = (0.0, 0.85, 0.95, 1.0, 1.02, 1.05)
OUTCOME_SEEDS = (1, 2, 3, 42)
ENDOWMENT_SEEDS = (1, 2, 42)
BASELINES = ("mes", "res-0.25", "res-0.5", "res-0.75", "res-1.0",
             "greedy-count", "greedy-cost", "historical", "greedy")
SIGNIFICANCE_BASELINES = ("greedy-cost", "greedy-count", "llmrule-card",
                          "llmrule-cost", "mes", "res-1.0")
KERNEL_TAGS = {"direct": "Direct", "static-floor": "StaticFloor",
               "payment-only": "PaymentOnly", "payment+completion": "PaymentComplete"}


def _number(row: dict, field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field}: expected a finite number, found {value!r}")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field}: expected a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{field}: expected a finite number")
    return result


def _integer(row: dict, field: str, *, positive: bool = False) -> int:
    value = _number(row, field)
    if not value.is_integer() or value < 0 or (positive and value == 0):
        raise ValueError(f"{field}: expected a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise ValueError("macro ratio has a zero denominator")
    result = numerator / denominator
    if not isfinite(result):
        raise ValueError("macro ratio is not finite")
    return result


def _fmt(value: float, nd: int = 4, *, signed: bool = False) -> str:
    if not isfinite(value):
        raise ValueError("macro result is not finite")
    return f"{value:+.{nd}f}" if signed else f"{value:.{nd}f}"


def _pct(value: float, nd: int = 0) -> str:
    return _fmt(100 * value, nd) + r"\%"


def _p(value: float) -> str:
    if not 0 <= value <= 1:
        raise ValueError("p-value must be between zero and one")
    return "< 0.001" if value < 1e-3 else f"= {value:.3f}"


def _record(value: str, sources: list[dict], transformation: str) -> Record:
    return {"value": value, "sources": sources, "transformation": transformation}


class _Table:
    def __init__(self, consumers: dict, consumer: str, table: str, schema: str,
                 count_field: str | None):
        payload = consumers.get(consumer)
        if not isinstance(payload, dict) or payload.get("consumer_id") != consumer:
            raise ValueError(f"{consumer}: missing or mismatched consumer identity")
        values = payload.get("values")
        if not isinstance(values, dict) or values.get("schema") != schema:
            raise ValueError(f"{consumer}: expected schema {schema}")
        rows = values.get(table)
        if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) for r in rows):
            raise ValueError(f"{consumer}.{table}: expected nonempty object rows")
        self.consumer, self.table, self.rows, self.values = consumer, table, rows, values
        if count_field is not None and _integer(values, count_field, positive=True) != len(rows):
            raise ValueError(f"{consumer}.{table}: {count_field} does not match rows")

    def index(self, fields: tuple[str, ...], expected: set[tuple]) -> dict[tuple, dict]:
        indexed = {}
        for row in self.rows:
            key = tuple(row.get(field) for field in fields)
            if any(isinstance(value, bool) or not isinstance(value, (str, int, float))
                   or value == "" for value in key):
                raise ValueError(f"{self.consumer}.{self.table}: invalid row identity {key}")
            if "seed" in fields and not isinstance(row["seed"], int):
                raise ValueError(f"{self.consumer}.{self.table}: seed must be an integer")
            if key in indexed:
                raise ValueError(f"{self.consumer}.{self.table}: duplicate row {key}")
            indexed[key] = row
        if set(indexed) != expected:
            raise ValueError(f"{self.consumer}.{self.table}: missing or unexpected row identities")
        return indexed

    def source(self, coordinates: dict, *fields: str) -> dict:
        return {"consumer": self.consumer, "table": self.table,
                "coordinates": dict(coordinates), "fields": list(fields)}

    def record(self, value: str, coordinates: dict, fields: tuple[str, ...],
               transformation: str) -> Record:
        return _record(value, [self.source(coordinates, *fields)], transformation)


def _expect(row: dict, field: str, expected: object) -> None:
    if row.get(field) != expected or isinstance(row.get(field), bool):
        raise ValueError(f"{field}: expected {expected!r}, found {row.get(field)!r}")


def _frontier_id(arm: str, floor: float, seed: int = 42) -> str:
    return f"{arm}_frontier/temporal_2022/target-{floor:g}/seed-{seed}"


def _frontier(consumers: dict, arm: str) -> tuple[_Table, dict]:
    table = _Table(consumers, f"frontier.{arm}", "point_rows", f"{arm}-frontier-v2", "point_count")
    _expect(table.values, "arm", arm)
    floors = OUTCOME_FLOORS if arm == "outcome" else (0.0, 0.99, 1.0)
    seeds = OUTCOME_SEEDS if arm == "outcome" else (42,)
    rows = table.index(("floor", "seed"), set(product(floors, seeds)))
    for (floor, seed), row in rows.items():
        _expect(row, "arm", arm)
        _expect(row, "fit_id", _frontier_id(arm, floor, seed))
        for field in ("test_worst_csd", "test_welfare", "test_exclusion", "train_worst_csd"):
            _number(row, field)
    return table, rows


def _seed_table(consumers: dict, arm: str) -> tuple[_Table, dict]:
    consumer = "outcome.seed_stability" if arm == "outcome" else "summary.endowment_seed_stability"
    table = _Table(consumers, consumer, "seed_rows", f"{arm}-seed-stability-v2", "seed_count")
    seeds = OUTCOME_SEEDS if arm == "outcome" else ENDOWMENT_SEEDS
    rows = table.index(("seed",), {(seed,) for seed in seeds})
    for (seed,), row in rows.items():
        fit = (_frontier_id("outcome", 1.0, seed) if arm == "outcome"
               else f"temporal_endowment/temporal_2022/seed-{seed}")
        _expect(row, "fit_id", fit)
        for field in ("test_worst_csd", "test_welfare", "test_exclusion"):
            _number(row, field)
    return table, rows


def _interval(row: dict, lo: str, hi: str) -> str:
    low, high = _number(row, lo), _number(row, hi)
    if low > high:
        raise ValueError("confidence interval endpoints are reversed")
    return f"[{low:+.4f}, {high:+.4f}]"


def _wins(row: dict, field: str, denominator: str = "n") -> str:
    wins, n = _integer(row, field), _integer(row, denominator, positive=True)
    if wins > n:
        raise ValueError("wins exceed the comparison count")
    return f"{wins}/{n}"


def _significance(consumers: dict) -> dict[str, Record]:
    out = {}
    outcome = _Table(consumers, "outcome.significance", "contrast_rows",
                     "outcome-significance-v2", "contrast_count")
    rows = outcome.index(("floor", "baseline"), set(product(OUTCOME_FLOORS, SIGNIFICANCE_BASELINES)))
    for (floor, _baseline), row in rows.items():
        _expect(row, "arm", "outcome")
        _expect(row, "source_id", _frontier_id("outcome", floor))
        _number(row, "diff")
        _interval(row, "ci_lo", "ci_hi")
        _p(_number(row, "p_two_sided"))
        _wins(row, "wins")
    selected = [
        (outcome, rows[(floor, "mes")], {"floor": floor, "baseline": "mes"}, f"Sig{tag}")
        for floor, tag in ((1.0, "Head"), (0.85, "Aggr"), (1.02, "Tight"), (1.05, "Tighter"))
    ]
    endow = _Table(consumers, "summary.endowment_significance", "contrast_rows",
                   "endowment-significance-suite-v2", "contrast_count")
    tags = {"mes": "MES", "res-1.0": "RES", "llmrule-cost": "LLMRuleCost",
            "llmrule-card": "LLMRuleCard", "learned-outcome": "TwoArm"}
    rows = endow.index(("contrast",), {(f"learned-endowment vs {baseline}",) for baseline in tags})
    for baseline, tag in tags.items():
        contrast = f"learned-endowment vs {baseline}"
        row = rows[(contrast,)]
        _expect(row, "learned_source_id", "temporal_endowment/temporal_2022/seed-42")
        _expect(row, "baseline_source_id", _frontier_id("outcome", 1.0) if baseline == "learned-outcome" else baseline)
        selected.append((endow, row, {"contrast": contrast}, f"EndowSig{tag}"))
    for table, row, coordinates, prefix in selected:
        for suffix, value, fields, explanation in (
            ("Diff", _fmt(_number(row, "diff")), ("diff",), "Format the saved paired difference to four decimals."),
            ("CI", _interval(row, "ci_lo", "ci_hi"), ("ci_lo", "ci_hi"),
             "Format saved confidence interval endpoints with signs and four decimals."),
            ("P", _p(_number(row, "p_two_sided")), ("p_two_sided",),
             "Format the saved p-value as < 0.001 below the threshold, otherwise to three decimals."),
            ("Wins", _wins(row, "wins"), ("wins", "n"), "Format saved wins over the comparison count."),
        ):
            out[prefix + suffix] = table.record(value, coordinates, fields, explanation)
    return out


def format_core_macros(consumers: dict) -> dict[str, Record]:
    """Return exactly the core macro family, rejecting incomplete or ambiguous rows."""
    if not isinstance(consumers, dict):
        raise ValueError("consumers must be a dictionary")
    outcome, points = _frontier(consumers, "outcome")
    endow_front, endow_points = _frontier(consumers, "endowment")
    base = _Table(consumers, "frontier.outcome", "baseline_rows", "outcome-frontier-v2", "baseline_count")
    baselines = base.index(("policy", "view"), set(product(BASELINES, ("train", "test"))))
    for (policy, _view), row in baselines.items():
        _expect(row, "source_id", "greedy-count" if policy == "greedy" else policy)
        for field in ("worst_csd", "welfare", "exclusion"):
            _number(row, field)
    out: dict[str, Record] = {}
    for policy, tag in (("historical", "Historical"), ("greedy-count", "Greedy"),
                        ("greedy-cost", "GreedyCost"), ("mes", "MES"), ("res-1.0", "RES")):
        row = baselines[(policy, "test")]
        for suffix, field in (("CSD", "worst_csd"), ("Welfare", "welfare"), ("Excl", "exclusion")):
            value = f"{_number(row, field):,.0f}" if suffix == "Welfare" else _fmt(_number(row, field))
            out[f"Test{tag}{suffix}"] = base.record(value, {"policy": policy, "view": "test"}, (field,),
                "Format welfare with comma grouping and no decimals." if suffix == "Welfare" else "Format to four decimals.")
    hand_policies = ("greedy-cost", "mes", "res-0.5", "res-1.0")
    hand = [_number(baselines[(policy, "test")], "worst_csd") for policy in hand_policies]
    hand_sources = [base.source({"policy": policy, "view": "test"}, "worst_csd") for policy in hand_policies]
    for suffix, value, transform in (("Lo", min(hand), "minimum"), ("Hi", max(hand), "maximum"),
                                     ("Width", max(hand) - min(hand), "maximum minus minimum")):
        out["HandBand" + suffix] = _record(_fmt(value), hand_sources,
            f"Take the {transform} held-out CSD of the four named reference rules and format to four decimals.")
    for floor, tag in ((1.0, "Head"), (0.85, "Aggr"), (0.0, "Corner")):
        row = points[(floor, 42)]
        coordinates = {"floor": floor, "seed": 42}
        for suffix, field, value in (
            ("Floor", "floor", _fmt(floor, 2)),
            ("CSD", "test_worst_csd", _fmt(_number(row, "test_worst_csd"))),
            ("Welfare", "test_welfare", f"{_number(row, 'test_welfare'):,.0f}"),
            ("Excl", "test_exclusion", _fmt(_number(row, "test_exclusion"))),
            ("TrainCSD", "train_worst_csd", _fmt(_number(row, "train_worst_csd"))),
        ):
            out[f"Learned{tag}{suffix}"] = outcome.record(value, coordinates, (field,),
                "Format the exact seed-42 target: floor to two decimals, welfare with commas and no decimals, CSD/exclusion to four decimals.")
        out[f"Learned{tag}Gap"] = outcome.record(
            _fmt(_number(row, "test_worst_csd") - _number(row, "train_worst_csd"), signed=True),
            coordinates, ("test_worst_csd", "train_worst_csd"),
            "Subtract training CSD from test CSD and format with sign to four decimals.")
    mes = baselines[("mes", "test")]
    mes_coordinates = {"policy": "mes", "view": "test"}
    mes_csd, mes_welfare = _number(mes, "worst_csd"), _number(mes, "welfare")
    for floor, tag in ((1.0, "Head"), (0.85, "Aggr")):
        out[tag + "ReductionPct"] = _record(
            _pct(_ratio(mes_csd - _number(points[(floor, 42)], "test_worst_csd"), mes_csd)),
            [outcome.source({"floor": floor, "seed": 42}, "test_worst_csd"), base.source(mes_coordinates, "worst_csd")],
            "Subtract policy CSD from MES CSD, divide by MES CSD, and format as percent with no decimals.")
    out["AggrWelfareCostPct"] = _record(
        _pct(1 - _ratio(_number(points[(0.85, 42)], "test_welfare"), mes_welfare)),
        [outcome.source({"floor": 0.85, "seed": 42}, "test_welfare"), base.source(mes_coordinates, "welfare")],
        "Subtract the policy/MES welfare ratio from one and format as percent with no decimals.")
    corner = points[(0.0, 42)]
    for policy, tag in (("greedy-count", "Deployed"), ("mes", "MES")):
        out["CornerExclVs" + tag] = _record(
            _fmt(_ratio(_number(corner, "test_exclusion"), _number(baselines[(policy, "test")], "exclusion")), 1),
            [outcome.source({"floor": 0.0, "seed": 42}, "test_exclusion"),
             base.source({"policy": policy, "view": "test"}, "exclusion")],
            "Divide unpenalized seed-42 outcome exclusion by the named baseline exclusion and format to one decimal.")
    for table, rows, tag in ((outcome, points, "Outcome"), (endow_front, endow_points, "Endow")):
        values = [_number(row, "test_worst_csd") for (_, seed), row in rows.items() if seed == 42]
        for suffix, value, transform in (("Lo", min(values), "minimum"), ("Hi", max(values), "maximum"),
                                         ("Width", max(values) - min(values), "maximum minus minimum")):
            out[f"Range{tag}{suffix}"] = table.record(_fmt(value), {"seed": 42}, ("test_worst_csd",),
                f"Take the {transform} across the complete seed-42 target grid and format to four decimals.")
    endow, endow_seeds = _seed_table(consumers, "endowment")
    outcome_seeds, outcome_seed_rows = _seed_table(consumers, "outcome")
    for table, rows, names in (
        (endow, endow_seeds, ("EndowNumSeeds", "EndowSeedLo", "EndowSeedHi", "EndowSeedSpread")),
        (outcome_seeds, outcome_seed_rows, ("NumSeeds", "SeedMin", "SeedMax", "SeedSpread")),
    ):
        values = [_number(row, "test_worst_csd") for row in rows.values()]
        out[names[0]] = table.record(str(len(values)), {}, ("seed",), "Count the complete declared seed set.")
        for name, value, transform in zip(names[1:], (min(values), max(values), max(values) - min(values)),
                                         ("minimum", "maximum", "maximum minus minimum")):
            out[name] = table.record(_fmt(value), {}, ("test_worst_csd",),
                f"Take the {transform} held-out CSD over the declared seeds and format to four decimals.")
    primary = endow_seeds[(42,)]
    for suffix, field in (("CSD", "test_worst_csd"), ("Welfare", "test_welfare"), ("Excl", "test_exclusion")):
        value = f"{_number(primary, field):,.0f}" if suffix == "Welfare" else _fmt(_number(primary, field))
        out["Endow" + suffix] = endow.record(value, {"seed": 42}, (field,),
            "Format the temporal-endowment seed-42 result: welfare with commas and no decimals, CSD/exclusion to four decimals.")
    endow_csd, endow_welfare = _number(primary, "test_worst_csd"), _number(primary, "test_welfare")
    for suffix, value, own_field, baseline_field, transform in (
        ("VsMES", _fmt(endow_csd - mes_csd), "test_worst_csd", "worst_csd", "Subtract MES CSD and format to four decimals."),
        ("ReductionPct", _pct(_ratio(mes_csd - endow_csd, mes_csd)), "test_worst_csd", "worst_csd",
         "Divide MES-minus-endowment CSD by MES CSD and format as percent with no decimals."),
        ("WelfareRatio", _fmt(_ratio(endow_welfare, mes_welfare), 3), "test_welfare", "welfare",
         "Divide endowment welfare by MES welfare and format to three decimals."),
        ("WelfareCostPct", _pct(1 - _ratio(endow_welfare, mes_welfare), 1), "test_welfare", "welfare",
         "Subtract the endowment/MES welfare ratio from one and format as percent to one decimal."),
    ):
        out["Endow" + suffix] = _record(value,
            [endow.source({"seed": 42}, own_field), base.source(mes_coordinates, baseline_field)], transform)
    out.update(_significance(consumers))
    out.update(_mechanism(consumers, outcome, corner, base, mes))
    return out


def _mechanism(consumers: dict, outcome: _Table, corner: dict, base: _Table, mes: dict) -> dict[str, Record]:
    out = {}
    kernel = _Table(consumers, "mechanism.payment_kernel", "kernel_summary_rows",
                    "payment-kernel-intervention-v2", None)
    # This table has a Cartesian-product count rather than a row-count field.
    _expect(kernel.values, "floor_count", 3)
    _expect(kernel.values, "kernel_count", 4)
    rows = kernel.index(("floor", "kernel"), set(product((0.0, 0.85, 1.0), KERNEL_TAGS)))
    for (floor, name), row in rows.items():
        source = _frontier_id("outcome", floor)
        _expect(row, "source_id", source if name == "direct" else f"payment/{source}/{name}")
        for field in ("worst_csd", "welfare", "exclusion", "budget_utilization", "welfare_ratio_vs_direct",
                      "diff_vs_direct", "exclusion_diff_vs_direct"):
            _number(row, field)
        for lo, hi, p, wins in (("ci_lo", "ci_hi", "p_two_sided", "wins_vs_direct"),
                               ("exclusion_ci_lo", "exclusion_ci_hi", "exclusion_p_two_sided", "exclusion_wins_vs_direct")):
            _interval(row, lo, hi)
            _p(_number(row, p))
            _wins(row, wins)
    for name, tag in KERNEL_TAGS.items():
        row, coordinates = rows[(0.0, name)], {"floor": 0.0, "kernel": name}
        for suffix, field, value, transform in (
            ("CSD", "worst_csd", _fmt(_number(row, "worst_csd")), "Format CSD to four decimals."),
            ("Welfare", "welfare", f"{_number(row, 'welfare'):,.0f}", "Format welfare with commas and no decimals."),
            ("Excl", "exclusion", _fmt(_number(row, "exclusion")), "Format exclusion to four decimals."),
            ("Util", "budget_utilization", _pct(_number(row, "budget_utilization"), 1), "Format budget utilization as percent to one decimal."),
            ("WelfareFactor", "welfare_ratio_vs_direct", _fmt(_number(row, "welfare_ratio_vs_direct"), 2), "Format the saved welfare ratio to two decimals."),
        ):
            out[f"Kernel{tag}{suffix}"] = kernel.record(value, coordinates, (field,), transform)
    complete = rows[(0.0, "payment+completion")]
    complete_coordinates = {"floor": 0.0, "kernel": "payment+completion"}
    for tag, diff, lo, hi, p in (
        ("CSD", "diff_vs_direct", "ci_lo", "ci_hi", "p_two_sided"),
        ("Excl", "exclusion_diff_vs_direct", "exclusion_ci_lo", "exclusion_ci_hi", "exclusion_p_two_sided"),
    ):
        for suffix, value, fields, transform in (
            ("Diff", _fmt(_number(complete, diff)), (diff,), "Format the saved paired difference to four decimals."),
            ("CI", _interval(complete, lo, hi), (lo, hi), "Format saved confidence interval endpoints with signs and four decimals."),
            ("P", _p(_number(complete, p)), (p,), "Format the saved p-value as < 0.001 below the threshold, otherwise to three decimals."),
        ):
            out[f"KernelPaymentComplete{tag}{suffix}"] = kernel.record(value, complete_coordinates, fields, transform)
    out["KernelPaymentCompleteExclWins"] = kernel.record(_wins(complete, "exclusion_wins_vs_direct"),
        complete_coordinates, ("exclusion_wins_vs_direct", "n"), "Format saved wins over the comparison count.")
    out["KernelPaymentCompleteExclReductionPct"] = _record(
        _pct(1 - _ratio(_number(complete, "exclusion"), _number(rows[(0.0, "direct")], "exclusion")), 1),
        [kernel.source(complete_coordinates, "exclusion"), kernel.source({"floor": 0.0, "kernel": "direct"}, "exclusion")],
        "Subtract the completed-payment/direct exclusion ratio from one and format as percent to one decimal.")
    identification = _Table(consumers, "mechanism.support_floor", "identification_rows",
                            "static-support-floor-identification-v2", "row_count")
    floor_rows = identification.index(("kappa",), {(1.0,), (2.0,)})
    for kappa, tag in ((1.0, "KOne"), (2.0, "KTwo")):
        row = floor_rows[(kappa,)]
        _expect(row, "fit_id", f"static_support_floor/temporal_2022/kappa-{kappa:g}/seed-42")
        _expect(row, "welfare_floor", "none")
        for suffix, field in (("CSD", "test_worst_csd"), ("Welfare", "test_welfare"), ("Excl", "test_exclusion")):
            value = f"{_number(row, field):,.0f}" if suffix == "Welfare" else _fmt(_number(row, field))
            out[f"Ident{tag}{suffix}"] = identification.record(value, {"kappa": kappa}, (field,),
                "Format the saved static-floor result: welfare with commas and no decimals, CSD/exclusion to four decimals.")
    k1 = floor_rows[(1.0,)]
    for name, field, base_field, numerator, denominator in (
        ("Excl", "test_exclusion", "exclusion", _number(corner, "test_exclusion") - _number(k1, "test_exclusion"),
         _number(corner, "test_exclusion") - _number(mes, "exclusion")),
        ("Welfare", "test_welfare", "welfare", _number(k1, "test_welfare") - _number(corner, "test_welfare"),
         _number(mes, "welfare") - _number(corner, "test_welfare")),
    ):
        out[f"FloorCloses{name}Pct"] = _record(_pct(_ratio(numerator, denominator)),
            [identification.source({"kappa": 1.0}, field), outcome.source({"floor": 0.0, "seed": 42}, field),
             base.source({"policy": "mes", "view": "test"}, base_field)],
            "Divide the kappa-one static floor's repair of the corner result by the corner-to-MES gap and format as percent with no decimals.")
    return out
