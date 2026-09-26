"""Formatting-only regression tests over captured consumer values."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from iclr_manuscript_tables_v2 import format_table_macros


CAPTURE = (Path(__file__).resolve().parents[1] / "results" /
           "iclr_primary_warsaw_v2/evaluation/paper_consumers.json")
TABLES = (
    ("cross_district.bound10", "contrast_rows"),
    ("cross_district.bound40", "contrast_rows"),
    ("summary.composition", "composition_rows"),
    ("robustness.demographic_partition", "scheme_rows"),
    ("outcome.ablation", "ablation_rows"),
    ("mechanism.theory_floor", "infeasible_spend_rows"),
    ("mechanism.theory_floor", "theory_rows"),
    ("baseline.prior_rules", "rule_view_rows"),
    ("transfer.lodz", "transfer_rows"),
)


@pytest.fixture
def payload() -> dict:
    return json.loads(CAPTURE.read_text())


@pytest.fixture
def consumers(payload: dict) -> dict:
    return payload["consumers"]


def _values(consumers: dict) -> dict[str, str]:
    return {name: record["value"] for name, record in format_table_macros(consumers).items()}


def _rows(consumers: dict, cid: str, table: str) -> list[dict]:
    return consumers[cid]["values"][table]


def _find(consumers: dict, cid: str, table: str, **coordinates: object) -> dict:
    found = [row for row in _rows(consumers, cid, table)
             if all(row.get(key) == value for key, value in coordinates.items())]
    assert len(found) == 1
    return found[0]


def _update_count(consumers: dict, cid: str, table: str) -> None:
    values = consumers[cid]["values"]
    if "row_count" in values:
        values["row_count"] = len(values[table])


def test_saved_values_and_complete_auditable_records(payload: dict, consumers: dict) -> None:
    records = format_table_macros(consumers)
    assert len(records) == 124
    assert not any(name.endswith(("DiffAbs", "DeltaAbs")) for name in records)
    expected_subset = {name: value for name, value in payload["manuscript_macros"].items()
                       if name in records}
    assert len(expected_subset) == 22
    assert {name: records[name]["value"] for name in expected_subset} == expected_subset
    for name, record in records.items():
        assert set(record) == {"value", "sources", "transformation"}, name
        assert isinstance(record["value"], str) and record["value"]
        assert record["sources"] and record["transformation"]
        for source in record["sources"]:
            assert set(source) == {"consumer", "table", "coordinates", "fields"}
            assert source["fields"]
            selected = [row for row in _rows(consumers, source["consumer"], source["table"])
                        if all(row.get(key) == value for key, value in source["coordinates"].items())]
            assert selected, (name, source)
            assert all(field in row for row in selected for field in source["fields"])


def test_saved_table_rounding_and_transformations(consumers: dict) -> None:
    values = _values(consumers)
    expected = {
        "DistrictCVP": "= 0.0625", "DistrictCVWideP": "= 0.0625",
        "DistrictCVSeriesP": "< 0.001", "DistrictCVWideCSDShift": "-0.0054",
        "DistrictCVWins": "17/19", "DistrictCVFoldWins": "5/5",
        "DistrictCVCI": "[-0.0396, -0.0205]", "DistrictCVExclDiff": "-0.0031",
        "DistrictCVWelfareDeltaPct": r"-1.6\%", "DistrictCVWelfareCostPct": r"1.6\%",
        "CompMESFunded": "30.1", "CompOutcomeCornerRepTV": "0.0664",
        "PartAgeRESCSD": "0.1106", "PartSexRESCSD": "0.0558",
        "PartAgeGroups": "4", "PartMaxLearnedP": "0.0130",
        "PartSexEndowP": "< 0.001", "PartAgeSexEndowP": "= 0.012",
        "AblFullDelta": "+0.0000", "AblConcDelta": "-0.0095",
        "AblationWorstFeature": r"cost\_share", "AblationWorstDelta": "+0.0063",
        "InfeasMESBudget": r"0.29\%", "InfeasOutcomeCornerBudget": r"25.06\%",
        "KappaEndowMean": "7.40", "KappaEndowMax": "13.63",
        "LLMRuleCostWelfare": "1,009,756", "LLMRuleCardCostWelfare": "263,710,950,650",
        "TransferOutcomeCSD": "0.0620", "TransferEndowWelfare": "447,075",
    }
    assert {name: values[name] for name in expected} == expected


def test_res_values_are_selected_by_scheme_and_policy(consumers: dict) -> None:
    before = _values(consumers)
    for scheme, replacement in (("age", 0.123456), ("sex", 0.065432)):
        _find(consumers, "robustness.demographic_partition", "scheme_rows",
              scheme=scheme, policy="res-1.0")["worst_csd"] = replacement
    after = _values(consumers)
    assert after["PartAgeRESCSD"] == "0.1235"
    assert after["PartSexRESCSD"] == "0.0654"
    assert {name for name in before if before[name] != after[name]} == {
        "PartAgeRESCSD", "PartSexRESCSD"}


@pytest.mark.parametrize("cid,table", TABLES)
@pytest.mark.parametrize("mutation", ("missing", "duplicate", "unknown"))
def test_category_changes_fail_closed(consumers: dict, cid: str, table: str, mutation: str) -> None:
    rows = _rows(consumers, cid, table)
    if mutation == "missing":
        rows.pop(0)
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        field = next(key for key in ("baseline", "policy", "ablated", "rule") if key in rows[0])
        rows[0][field] = "unexpected-category"
    _update_count(consumers, cid, table)
    with pytest.raises(ValueError, match="categories|duplicate"):
        format_table_macros(consumers)


@pytest.mark.parametrize("cid", tuple(dict.fromkeys(cid for cid, _ in TABLES)))
@pytest.mark.parametrize("mutation", ("consumer", "schema"))
def test_missing_consumers_and_changed_schemas_fail_closed(consumers: dict, cid: str, mutation: str) -> None:
    if mutation == "consumer":
        del consumers[cid]
    else:
        consumers[cid]["values"]["schema"] = "unknown"
    with pytest.raises(ValueError, match="schema"):
        format_table_macros(consumers)


@pytest.mark.parametrize("field", ("baseline_mean", "n", "n_folds"))
def test_district_bound_comparisons_must_match(consumers: dict, field: str) -> None:
    row = _find(consumers, "cross_district.bound40", "contrast_rows", baseline="mes")
    row[field] += 1
    with pytest.raises(ValueError, match=f"comparison field {field}"):
        format_table_macros(consumers)


def test_fold_p_retains_four_decimals_and_series_p_retains_threshold(consumers: dict) -> None:
    row = _find(consumers, "cross_district.bound10", "contrast_rows", baseline="mes")
    row["fold_p_two_sided"], row["p_two_sided"] = 0.125, 0.001
    wide = _find(consumers, "cross_district.bound40", "contrast_rows", baseline="mes")
    wide["fold_p_two_sided"] = 0.3125
    values = _values(consumers)
    assert values["DistrictCVP"] == "= 0.1250"
    assert values["DistrictCVWideP"] == "= 0.3125"
    assert values["DistrictCVSeriesP"] == "= 0.001"


def test_signed_zero_and_welfare_cost_clamp_match_existing_formatting(consumers: dict) -> None:
    district = _find(consumers, "cross_district.bound10", "contrast_rows", baseline="mes")
    district.update(diff=-0.0000001, welfare_ratio=1.01, learned_exclusion=0.0,
                    baseline_exclusion=0.0000001, ci_lo=-0.0, ci_hi=0.0)
    _find(consumers, "outcome.ablation", "ablation_rows", ablated="none (full model)")["delta_vs_full"] = -0.0
    values = _values(consumers)
    assert values["DistrictCVDiff"] == "-0.0000"
    assert values["DistrictCVExclDiff"] == "-0.0000"
    assert values["DistrictCVCI"] == "[-0.0000, +0.0000]"
    assert values["DistrictCVWelfareDeltaPct"] == r"+1.0\%"
    assert values["DistrictCVWelfareCostPct"] == r"0.0\%"
    assert values["AblFullDelta"] == "-0.0000"


@pytest.mark.parametrize("bad", (None, "", "nan", float("inf"), True, "invalid"))
def test_invalid_consumed_numbers_fail_closed(consumers: dict, bad: object) -> None:
    _find(consumers, "summary.composition", "composition_rows", policy="mes")["mean_cost_share"] = bad
    with pytest.raises(ValueError, match="finite number"):
        format_table_macros(consumers)


def test_missing_learned_p_and_inconsistent_group_counts_fail_closed(consumers: dict) -> None:
    row = _find(consumers, "robustness.demographic_partition", "scheme_rows",
                scheme="age", policy="learned-endowment")
    p = row.pop("vs_mes_p")
    with pytest.raises(ValueError, match="vs_mes_p"):
        format_table_macros(consumers)
    row["vs_mes_p"] = p
    row["n_groups"] = 8
    with pytest.raises(ValueError, match="inconsistent n_groups"):
        format_table_macros(consumers)


def test_prior_rule_macros_use_only_aggregate_test_rows(consumers: dict) -> None:
    expected = _values(consumers)
    for row in _rows(consumers, "baseline.prior_rules", "rule_view_rows"):
        if row["series"] != "__aggregate__" or row["view"] != "test":
            row["worst_csd"] = 999
            row["welfare"] = 999
    assert _values(consumers) == expected


def test_kappa_uses_only_saved_learned_values(consumers: dict) -> None:
    rows = _rows(consumers, "mechanism.theory_floor", "theory_rows")
    for row in rows:
        row["kappa"] = 3.125 if row["policy"] == "learned-endowment" else None
    values = _values(consumers)
    assert values["KappaEndowMean"] == values["KappaEndowMax"] == "3.12"
    next(row for row in rows if row["policy"] == "learned-endowment")["kappa"] = None
    with pytest.raises(ValueError, match="kappa"):
        format_table_macros(consumers)


@pytest.mark.parametrize("rewrite_count", (False, True))
def test_theory_cannot_drop_an_entire_edition(consumers: dict, rewrite_count: bool) -> None:
    values = consumers["mechanism.theory_floor"]["values"]
    rows = values["theory_rows"]
    removed = (rows[0]["source_series"], rows[0]["year"])
    values["theory_rows"] = [row for row in rows
                             if (row["source_series"], row["year"]) != removed]
    assert len(values["theory_rows"]) == 318
    if rewrite_count:
        values["edition_row_count"] = 318
    with pytest.raises(ValueError, match="edition/policy categories"):
        format_table_macros(consumers)


@pytest.mark.parametrize("field,expected", (("edition_row_count", 324), ("policy_count", 6)))
@pytest.mark.parametrize("mutation", ("missing", "wrong_count", "bool", "string"))
def test_theory_saved_counts_must_match(consumers: dict, field: str, expected: int,
                                      mutation: str) -> None:
    values = consumers["mechanism.theory_floor"]["values"]
    if mutation == "missing":
        del values[field]
    else:
        values[field] = {"wrong_count": expected - 1, "bool": True,
                         "string": str(expected)}[mutation]
    with pytest.raises(ValueError, match=field):
        format_table_macros(consumers)


def test_theory_edition_identity_must_match_independent_corpus(consumers: dict) -> None:
    rows = _rows(consumers, "mechanism.theory_floor", "theory_rows")
    changed = (rows[0]["source_series"], rows[0]["year"])
    for row in rows:
        if (row["source_series"], row["year"]) == changed:
            row["year"] = 2022
    with pytest.raises(ValueError, match="edition/policy categories"):
        format_table_macros(consumers)


def test_theory_requires_all_54_corpus_editions(consumers: dict) -> None:
    rows = _rows(consumers, "primary.corpus", "instance_rows")
    rows.remove(next(row for row in rows if row["series"].startswith("Poland/Warszawa/")
                     and row["year"] > 2022))
    with pytest.raises(ValueError, match="54 scored Warsaw editions"):
        format_table_macros(consumers)


def test_worst_ablation_is_selected_from_saved_deltas(consumers: dict) -> None:
    _find(consumers, "outcome.ablation", "ablation_rows", ablated="deficit_weighted")["delta_vs_full"] = 0.031234
    values = _values(consumers)
    assert values["AblationWorstFeature"] == r"deficit\_weighted"
    assert values["AblationWorstDelta"] == "+0.0312"


def test_table_order_does_not_change_values_without_maximum_ties(consumers: dict) -> None:
    before = _values(consumers)
    for cid, table in TABLES:
        _rows(consumers, cid, table).reverse()
    assert _values(consumers) == before


def test_formatter_is_read_only_and_needs_no_file_access(consumers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    before = deepcopy(consumers)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("formatter attempted filesystem access")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    assert len(format_table_macros(consumers)) == 124
    assert consumers == before
