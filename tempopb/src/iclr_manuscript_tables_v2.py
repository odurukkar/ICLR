"""Pure, fail-closed reconstruction of table macros from captured v2 consumers.

This module formats saved results only. It performs no I/O, imports no legacy
generator, and computes no statistical test. Every output identifies its input
rows and fields; authentication of the containing artifact belongs to the caller.
"""

from __future__ import annotations

from itertools import product
from math import isfinite
from typing import Any


Record = dict[str, Any]


def _number(row: dict, field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{field}: expected a finite number, found {value!r}")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field}: expected a finite number, found {value!r}") from exc
    if not isfinite(result):
        raise ValueError(f"{field}: expected a finite number, found {value!r}")
    return result


def _integer(row: dict, field: str) -> str:
    value = _number(row, field)
    if value < 0 or not value.is_integer():
        raise ValueError(f"{field}: expected a nonnegative integer")
    return str(int(value))


def _fmt(value: float, nd: int = 4) -> str:
    # Preserve Python's signed-zero behavior, as the existing generator does.
    return f"{value:.{nd}f}"


def _pct(value: float, nd: int = 0) -> str:
    return f"{100 * value:.{nd}f}\\%"


def _p(value: float) -> str:
    if not 0 <= value <= 1:
        raise ValueError("p-value must be between zero and one")
    return "< 0.001" if value < 1e-3 else f"= {value:.3f}"


def _record(value: str, sources: list[dict], transformation: str) -> Record:
    return {"value": value, "sources": sources, "transformation": transformation}


class _Table:
    def __init__(self, consumers: dict, consumer: str, table: str, schema: str):
        payload = consumers.get(consumer)
        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, dict) or values.get("schema") != schema:
            raise ValueError(f"{consumer}: expected values schema {schema}")
        rows = values.get(table)
        if not isinstance(rows, list) or not rows or any(
            not isinstance(row, dict) for row in rows
        ):
            raise ValueError(f"{consumer}.{table}: expected nonempty object rows")
        self.consumer, self.table, self.rows = consumer, table, rows
        if "row_count" in values and (
            isinstance(values["row_count"], bool) or values["row_count"] != len(rows)
        ):
            raise ValueError(f"{consumer}.{table}: row_count does not match rows")

    def index(self, *fields: str, expected: set[tuple] | None = None) -> dict:
        indexed = {}
        for row in self.rows:
            key = tuple(row.get(field) for field in fields)
            if any(
                (not isinstance(value, int) or isinstance(value, bool) or value < 1)
                if field == "year" else (not isinstance(value, str) or not value)
                for field, value in zip(fields, key)
            ):
                raise ValueError(f"{self.consumer}.{self.table}: invalid row identity {key}")
            if key in indexed:
                raise ValueError(f"{self.consumer}.{self.table}: duplicate row {key}")
            indexed[key] = row
        if expected is not None and set(indexed) != expected:
            raise ValueError(
                f"{self.consumer}.{self.table}: incomplete or unexpected categories; "
                f"missing={sorted(expected - set(indexed))}, "
                f"unexpected={sorted(set(indexed) - expected)}"
            )
        return indexed

    def source(self, coordinates: dict, *fields: str) -> dict:
        return {"consumer": self.consumer, "table": self.table,
                "coordinates": dict(coordinates), "fields": list(fields)}

    def record(self, value: str, coordinates: dict, fields: tuple[str, ...],
               transformation: str) -> Record:
        return _record(value, [self.source(coordinates, *fields)], transformation)


def _district(consumers: dict) -> dict[str, Record]:
    expected = {(name,) for name in (
        "greedy-cost", "greedy-count", "llmrule-card", "llmrule-cost", "mes", "res-1.0"
    )}
    tables = [_Table(consumers, f"cross_district.bound{bound}", "contrast_rows",
                     f"cross-district-bound{bound}-v2") for bound in (10, 40)]
    primary, wide = [table.index("baseline", expected=expected)[("mes",)]
                     for table in tables]
    for field in ("baseline_mean", "n", "n_folds"):
        if primary.get(field) != wide.get(field):
            raise ValueError(f"district bound sensitivity changed comparison field {field}")
    out: dict[str, Record] = {}
    coordinates = {"baseline": "mes"}
    for table, row, prefix in ((tables[0], primary, "DistrictCV"),
                               (tables[1], wide, "DistrictCVWide")):
        for suffix, field in (("LearnedCSD", "learned_mean"), ("Diff", "diff"),
                              ("LearnedExcl", "learned_exclusion")):
            out[prefix + suffix] = table.record(
                _fmt(_number(row, field)), coordinates, (field,), "Format to four decimals.")
        out[prefix + "CI"] = table.record(
            f"[{_number(row, 'ci_lo'):+.4f}, {_number(row, 'ci_hi'):+.4f}]",
            coordinates, ("ci_lo", "ci_hi"),
            "Format saved confidence interval endpoints with signs and four decimals.")
        fold_p = _number(row, "fold_p_two_sided")
        if not 0 <= fold_p <= 1:
            raise ValueError("fold_p_two_sided must be between zero and one")
        out[prefix + "P"] = table.record(
            f"= {fold_p:.4f}", coordinates, ("fold_p_two_sided",),
            "Format the saved fold-level p-value with four informative decimals.")
        for suffix, numerator, denominator in (("Wins", "wins", "n"),
                                               ("FoldWins", "fold_wins", "n_folds")):
            out[prefix + suffix] = table.record(
                f"{_integer(row, numerator)}/{_integer(row, denominator)}", coordinates,
                (numerator, denominator), "Format saved wins over the comparison count.")
        ratio = _number(row, "welfare_ratio")
        out[prefix + "WelfareRatio"] = table.record(
            _fmt(ratio, 3), coordinates, ("welfare_ratio",), "Format to three decimals.")
        out[prefix + "WelfareCostPct"] = table.record(
            _pct(max(0.0, 1.0 - ratio), 1), coordinates, ("welfare_ratio",),
            "Subtract the welfare ratio from one, clamp below at zero, and format as percent to one decimal.")
    table = tables[0]
    for suffix, field in (("MESCSD", "baseline_mean"), ("MESExcl", "baseline_exclusion"),
                          ("FoldMeanDiff", "fold_mean_diff")):
        out["DistrictCV" + suffix] = table.record(
            _fmt(_number(primary, field)), coordinates, (field,), "Format to four decimals.")
    for suffix, field in (("Ties", "ties"), ("N", "n")):
        out["DistrictCV" + suffix] = table.record(
            _integer(primary, field), coordinates, (field,), "Format the saved integer count.")
    out["DistrictCVSeriesP"] = table.record(
        _p(_number(primary, "p_two_sided")), coordinates, ("p_two_sided",),
        "Format the saved series p-value as < 0.001 below that threshold, otherwise to three decimals.")
    out["DistrictCVWelfareDeltaPct"] = table.record(
        f"{100 * (_number(primary, 'welfare_ratio') - 1):+.1f}\\%", coordinates,
        ("welfare_ratio",), "Subtract one from the welfare ratio and format signed percent to one decimal.")
    out["DistrictCVExclDiff"] = table.record(
        f"{_number(primary, 'learned_exclusion') - _number(primary, 'baseline_exclusion'):+.4f}",
        coordinates, ("learned_exclusion", "baseline_exclusion"),
        "Subtract MES exclusion from learned exclusion and format with sign to four decimals.")
    out["DistrictCVWideCSDShift"] = _record(
        f"{_number(wide, 'learned_mean') - _number(primary, 'learned_mean'):+.4f}",
        [item.source(coordinates, "learned_mean", "baseline_mean", "n", "n_folds")
         for item in tables],
        "Check equal MES baseline mean and series/fold counts; subtract bound-10 learned CSD from bound-40 learned CSD and format signed to four decimals.")
    return out


def _composition(consumers: dict) -> dict[str, Record]:
    table = _Table(consumers, "summary.composition", "composition_rows", "allocation-composition-v2")
    tags = {"mes": "MES", "learned-endowment": "Endow", "greedy-cost": "GreedyCost",
            "llmrule-cost": "LLMRuleCost", "llmrule-card": "LLMRuleCard",
            "outcome-f1.00": "OutcomeHead", "outcome-f0.00": "OutcomeCorner"}
    rows = table.index("policy", expected={(policy,) for policy in tags})
    out = {}
    for policy, tag in tags.items():
        for suffix, field, nd in (("Funded", "mean_projects_funded", 1),
                                  ("Cost", "mean_cost_share", 4),
                                  ("Conc", "mean_concentration", 4),
                                  ("RepTV", "cost_weighted_representation_tv", 4)):
            out[f"Comp{tag}{suffix}"] = table.record(
                _fmt(_number(rows[(policy,)], field), nd), {"policy": policy}, (field,),
                f"Format the saved composition statistic to {nd} decimals.")
    return out


def _partitions(consumers: dict) -> dict[str, Record]:
    table = _Table(consumers, "robustness.demographic_partition", "scheme_rows",
                   "demographic-partition-robustness-v2")
    schemes = {"age_sex": "AgeSex", "age": "Age", "sex": "Sex"}
    policies = {"mes": "MES", "res-1.0": "RES", "learned-endowment": "Endow",
                "learned-outcome": "Outcome"}
    rows = table.index("scheme", "policy", expected=set(product(schemes, (*policies, "greedy-cost"))))
    out, learned_ps, p_sources = {}, [], []
    for scheme, scheme_tag in schemes.items():
        counts = {_integer(row, "n_groups") for (s, _), row in rows.items() if s == scheme}
        if len(counts) != 1:
            raise ValueError(f"{scheme}: inconsistent n_groups")
        out[f"Part{scheme_tag}Groups"] = table.record(
            next(iter(counts)), {"scheme": scheme}, ("n_groups",),
            "Check equal group counts across policies and format the saved count.")
        for policy, tag in policies.items():
            row, coords = rows[(scheme, policy)], {"scheme": scheme, "policy": policy}
            out[f"Part{scheme_tag}{tag}CSD"] = table.record(
                _fmt(_number(row, "worst_csd")), coords, ("worst_csd",), "Format to four decimals.")
            if policy.startswith("learned-"):
                p = _number(row, "vs_mes_p")
                learned_ps.append(p)
                p_sources.append(table.source(coords, "vs_mes_p"))
                out[f"Part{scheme_tag}{tag}P"] = table.record(
                    _p(p), coords, ("vs_mes_p",),
                    "Format the saved p-value as < 0.001 below that threshold, otherwise to three decimals.")
            elif row.get("vs_mes_p") not in (None, ""):
                raise ValueError(f"{scheme}/{policy}: unexpected baseline vs_mes_p")
    out["PartMaxLearnedP"] = _record(
        _fmt(max(learned_ps)), p_sources,
        "Take the maximum of the six saved learned-policy p-values and format to four decimals.")
    return out


def _ablation(consumers: dict) -> dict[str, Record]:
    table = _Table(consumers, "outcome.ablation", "ablation_rows", "outcome-ablation-table-v2")
    tags = {"none (full model)": "Full", "approval_share": "ApprShare",
            "approvals_per_cost": "PerCost", "cost_share": "CostShare",
            "deficit_weighted": "Deficit", "cohort_concentration": "Conc"}
    rows = table.index("ablated", expected={(name,) for name in tags})
    out = {}
    for name, tag in tags.items():
        row, coords = rows[(name,)], {"ablated": name}
        out[f"Abl{tag}CSD"] = table.record(
            _fmt(_number(row, "test_worst_csd")), coords, ("test_worst_csd",), "Format to four decimals.")
        out[f"Abl{tag}Delta"] = table.record(
            f"{_number(row, 'delta_vs_full'):+.4f}", coords, ("delta_vs_full",),
            "Format the saved difference from the full model with sign to four decimals.")
    out["AblationFullCSD"] = table.record(
        _fmt(_number(rows[("none (full model)",)], "test_worst_csd")),
        {"ablated": "none (full model)"}, ("test_worst_csd",), "Format to four decimals.")
    candidates = [row for row in table.rows if row["ablated"] != "none (full model)"]
    worst = max(candidates, key=lambda row: _number(row, "delta_vs_full"))
    sources = [table.source({"ablated": row["ablated"]}, "ablated", "delta_vs_full")
               for row in candidates]
    out["AblationWorstFeature"] = _record(
        worst["ablated"].replace("_", "\\_"), sources,
        "Select the ablated feature with the greatest saved delta (first row on a tie); escape underscores for LaTeX.")
    out["AblationWorstDelta"] = _record(
        f"{_number(worst, 'delta_vs_full'):+.4f}", sources,
        "Take the greatest saved non-full-model delta and format with sign to four decimals.")
    return out


def _theory(consumers: dict) -> dict[str, Record]:
    cid, schema = "mechanism.theory_floor", "empirical-support-floor-v2"
    table = _Table(consumers, cid, "infeasible_spend_rows", schema)
    tags = {"mes": "MES", "res-1.0": "RES", "learned-endowment": "LearnedEndow",
            "greedy-cost": "GreedyCost", "learned-outcome-f1.0": "OutcomeHead",
            "learned-outcome-f0.0": "OutcomeCorner"}
    rows = table.index("policy", expected={(policy,) for policy in tags})
    out = {}
    for policy, tag in tags.items():
        for suffix, field in (("Budget", "pct_budget"), ("Proj", "pct_projects")):
            out[f"Infeas{tag}{suffix}"] = table.record(
                f"{_number(rows[(policy,)], field):.2f}\\%", {"policy": policy}, (field,),
                "Format the saved value, already in percent units, to two decimals with a percent sign.")
    theory = _Table(consumers, cid, "theory_rows", schema)
    indexed = theory.index("source_series", "year", "policy")
    corpus = _Table(consumers, "primary.corpus", "instance_rows", "primary-corpus-inventory-v2")
    # The captured corpus supplies an independent edition universe. Deriving it
    # from theory rows would let removal of all six policies for an edition pass.
    editions = {(series, year) for series, year in corpus.index("series", "year")
                if series.startswith("Poland/Warszawa/") and year > 2022}
    if len(editions) != 54 or len({series for series, _ in editions}) != 18:
        raise ValueError("primary.corpus: expected 54 scored Warsaw editions across 18 series")
    expected = {(series, year, policy) for series, year in editions for policy in tags}
    if set(indexed) != expected:
        raise ValueError("theory_rows: incomplete or unexpected edition/policy categories")
    values = consumers[cid]["values"]
    for field, expected_count in (("edition_row_count", len(expected)),
                                   ("policy_count", len(tags))):
        if type(values.get(field)) is not int or values[field] != expected_count:
            raise ValueError(f"{cid}: {field} must equal {expected_count}")
    ks = [_number(row, "kappa") for row in theory.rows if row["policy"] == "learned-endowment"]
    source = [theory.source({"policy": "learned-endowment"}, "kappa")]
    out["KappaEndowMean"] = _record(
        _fmt(sum(ks) / len(ks), 2), source,
        "Compute the arithmetic mean of saved learned-endowment kappa values and format to two decimals.")
    out["KappaEndowMax"] = _record(
        _fmt(max(ks), 2), source,
        "Take the maximum saved learned-endowment kappa and format to two decimals.")
    return out


def _prior_rules(consumers: dict) -> dict[str, Record]:
    table = _Table(consumers, "baseline.prior_rules", "rule_view_rows", "prior-rule-table-v2")
    tags = {"llmrule-cost": "LLMRuleCost", "llmrule-card": "LLMRuleCard"}
    indexed = table.index("series", "view", "rule")
    series = {series for series, _, _ in indexed}
    if "__aggregate__" not in series or set(indexed) != set(product(series, ("train", "test"), tags)):
        raise ValueError("rule_view_rows: incomplete or unexpected series/view/rule categories")
    out = {}
    for rule, tag in tags.items():
        coords = {"series": "__aggregate__", "view": "test", "rule": rule}
        row = indexed[("__aggregate__", "test", rule)]
        for suffix, field in (("CSD", "worst_csd"), ("Welfare", "welfare"),
                              ("CostWelfare", "cost_welfare"), ("Excl", "exclusion")):
            value = _number(row, field)
            monetary = suffix in ("Welfare", "CostWelfare")
            out[tag + suffix] = table.record(
                f"{value:,.0f}" if monetary else _fmt(value), coords, (field,),
                "Format the saved aggregate test value " +
                ("as an integer with thousands separators." if monetary else "to four decimals."))
    return out


def _transfer(consumers: dict) -> dict[str, Record]:
    table = _Table(consumers, "transfer.lodz", "transfer_rows", "lodz-transfer-table-v2")
    tags = {"historical": "Hist", "greedy-cost": "GreedyCost", "mes": "MES",
            "learned-endowment": "Endow", "learned-outcome-f1.00": "Outcome"}
    city = "Poland/Łódź/CITYWIDE"
    rows = table.index("series", "policy", expected={
        (city, policy) for policy in (*tags, "greedy-count", "res-1.0", "learned-outcome-f0.85")})
    out = {}
    for policy, tag in tags.items():
        row, coords = rows[(city, policy)], {"series": city, "policy": policy}
        out[f"Transfer{tag}CSD"] = table.record(
            _fmt(_number(row, "worst_csd")), coords, ("worst_csd",), "Format to four decimals.")
        out[f"Transfer{tag}Welfare"] = table.record(
            f"{_number(row, 'welfare'):,.0f}", coords, ("welfare",),
            "Format the saved welfare as an integer with thousands separators.")
    return out


def format_table_macros(consumers: dict) -> dict[str, Record]:
    """Reconstruct all owned table families from authenticated consumer values.

    Missing tables, row categories, duplicate coordinates, non-finite consumed
    values, or changed district baselines raise ``ValueError``. No partial result
    is returned. Magnitude twins are deliberately left to the caller.
    """
    if not isinstance(consumers, dict):
        raise ValueError("consumers must be an object")
    out: dict[str, Record] = {}
    for formatter in (_district, _composition, _partitions, _ablation, _theory,
                      _prior_rules, _transfer):
        out.update(formatter(consumers))
    return out
