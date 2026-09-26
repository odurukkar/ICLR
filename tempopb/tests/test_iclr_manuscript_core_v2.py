"""Read-only regression checks for the pure corrected-primary macro formatter."""

from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path

import pytest

from iclr_manuscript_core_v2 import format_core_macros


CAPTURE = Path(__file__).resolve().parents[1] / "results/iclr_primary_warsaw_v2/evaluation/paper_consumers.json"
TABLES = (
    ("frontier.outcome", "baseline_rows", "baseline_count"),
    ("frontier.outcome", "point_rows", "point_count"),
    ("frontier.endowment", "point_rows", "point_count"),
    ("summary.endowment_seed_stability", "seed_rows", "seed_count"),
    ("outcome.seed_stability", "seed_rows", "seed_count"),
    ("outcome.significance", "contrast_rows", "contrast_count"),
    ("summary.endowment_significance", "contrast_rows", "contrast_count"),
    ("mechanism.payment_kernel", "kernel_summary_rows", None),
    ("mechanism.support_floor", "identification_rows", "row_count"),
)


@pytest.fixture
def consumers():
    # The bundle is captured input only. No generator, evaluation, or fitting runs.
    return json.loads(CAPTURE.read_text(encoding="utf-8"))["consumers"]


def _row(consumers, cid, table, **coordinates):
    matches = [row for row in consumers[cid]["values"][table]
               if all(row[key] == value for key, value in coordinates.items())]
    assert len(matches) == 1
    return matches[0]


def test_actual_capture_values_and_complete_provenance(consumers):
    before = copy.deepcopy(consumers)
    records = format_core_macros(consumers)
    assert consumers == before
    assert len(records) == 134
    assert not any(name.endswith(("DiffAbs", "DeltaAbs", "HolmP")) for name in records)
    expected = {
        "TestMESCSD": "0.1992", "TestMESWelfare": "1,293,610", "TestMESExcl": "0.0438",
        "LearnedHeadCSD": "0.1728", "LearnedHeadFloor": "1.00",
        "LearnedAggrFloor": "0.85", "LearnedCornerFloor": "0.00",
        "LearnedCornerCSD": "0.1228", "LearnedCornerWelfare": "586,114",
        "EndowCSD": "0.1736", "EndowWelfare": "1,273,257", "EndowExcl": "0.0414",
        "EndowNumSeeds": "3", "NumSeeds": "4", "EndowReductionPct": r"13\%",
        "EndowWelfareRatio": "0.984", "EndowWelfareCostPct": r"1.6\%",
        "KernelPaymentCompleteCSDCI": "[-0.0227, +0.0341]",
        "KernelPaymentCompleteExclCI": "[-0.1313, -0.0773]",
        "KernelPaymentCompleteCSDP": "= 0.721",
        "KernelPaymentCompleteExclP": "< 0.001",
        "KernelPaymentCompleteExclWins": "17/18",
        "KernelPaymentCompleteExclReductionPct": r"70.9\%",
        "FloorClosesExclPct": r"59\%", "FloorClosesWelfarePct": r"27\%",
        "CornerExclVsDeployed": "1.8", "CornerExclVsMES": "3.4",
    }
    for name, expected_value in expected.items():
        assert records[name]["value"] == expected_value, name
    for name, record in records.items():
        assert set(record) == {"value", "sources", "transformation"}, name
        assert isinstance(record["value"], str) and record["value"]
        assert record["transformation"] and record["sources"]
        for source in record["sources"]:
            assert set(source) == {"consumer", "table", "coordinates", "fields"}
            assert source["fields"]
            rows = consumers[source["consumer"]]["values"][source["table"]]
            selected = [row for row in rows if all(row.get(key) == value
                        for key, value in source["coordinates"].items())]
            assert selected, (name, source)
            assert all(field in row for row in selected for field in source["fields"])


def test_no_io_and_no_input_mutation(consumers, monkeypatch):
    before = copy.deepcopy(consumers)
    def forbidden(*args, **kwargs):
        raise AssertionError("formatter attempted I/O")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    assert format_core_macros(consumers)["EndowCSD"]["value"] == "0.1736"
    assert consumers == before


@pytest.mark.parametrize("cid,table,count", TABLES)
@pytest.mark.parametrize("mutation", ("delete", "duplicate"))
def test_missing_and_duplicate_rows_fail_even_with_updated_counts(consumers, cid, table, count, mutation):
    values = consumers[cid]["values"]
    rows = values[table]
    if mutation == "delete":
        rows.pop()
    else:
        rows.append(copy.deepcopy(rows[0]))
    if count:
        values[count] = len(rows)
    with pytest.raises(ValueError):
        format_core_macros(consumers)


@pytest.mark.parametrize("cid,table,count", TABLES)
def test_missing_consumers_fail(consumers, cid, table, count):
    del consumers[cid]
    with pytest.raises(ValueError):
        format_core_macros(consumers)


def test_primary_endowment_is_temporal_seed42_not_frontier(consumers):
    frontier = _row(consumers, "frontier.endowment", "point_rows", floor=0.0, seed=42)
    frontier["test_worst_csd"] = 0.987654
    records = format_core_macros(consumers)
    assert records["EndowCSD"]["value"] == "0.1736"
    assert records["RangeEndowHi"]["value"] == "0.9877"
    assert records["EndowCSD"]["sources"] == [{
        "consumer": "summary.endowment_seed_stability", "table": "seed_rows",
        "coordinates": {"seed": 42}, "fields": ["test_worst_csd"],
    }]


def test_frontier_ranges_and_headline_use_only_seed42(consumers):
    original = format_core_macros(consumers)
    alternative = _row(consumers, "frontier.outcome", "point_rows", floor=1.0, seed=1)
    alternative["test_worst_csd"] = 0.99
    records = format_core_macros(consumers)
    assert records["LearnedHeadCSD"]["value"] == "0.1728"
    assert records["RangeOutcomeHi"]["value"] != "0.9900"
    assert records["RangeOutcomeHi"]["sources"][0]["coordinates"] == {"seed": 42}
    assert records["SeedMax"] == original["SeedMax"]


def test_hand_band_uses_exact_four_test_references(consumers):
    _row(consumers, "frontier.outcome", "baseline_rows", policy="greedy-count", view="test")["worst_csd"] = 9
    _row(consumers, "frontier.outcome", "baseline_rows", policy="mes", view="train")["worst_csd"] = 8
    _row(consumers, "frontier.outcome", "baseline_rows", policy="res-0.5", view="test")["worst_csd"] = 0.4
    records = format_core_macros(consumers)
    assert records["HandBandHi"]["value"] == "0.4000"
    assert {source["coordinates"]["policy"] for source in records["HandBandHi"]["sources"]} == {
        "greedy-cost", "mes", "res-0.5", "res-1.0",
    }


@pytest.mark.parametrize("cid,table,count", [entry for entry in TABLES if entry[2]])
def test_declared_row_counts_are_checked(consumers, cid, table, count):
    consumers[cid]["values"][count] += 1
    with pytest.raises(ValueError, match="does not match"):
        format_core_macros(consumers)


@pytest.mark.parametrize("field", ("floor_count", "kernel_count"))
def test_kernel_grid_counts_are_checked(consumers, field):
    consumers["mechanism.payment_kernel"]["values"][field] += 1
    with pytest.raises(ValueError):
        format_core_macros(consumers)


@pytest.mark.parametrize("cid,table,selector,field,value", (
    ("frontier.outcome", "point_rows", {"floor": 1.0, "seed": 42}, "floor", 1.000001),
    ("frontier.outcome", "point_rows", {"floor": 1.0, "seed": 42}, "seed", 41),
    ("frontier.outcome", "point_rows", {"floor": 1.0, "seed": 42}, "seed", 42.0),
    ("outcome.seed_stability", "seed_rows", {"seed": 42}, "fit_id", "outcome_frontier/temporal_2022/target-0/seed-42"),
    ("summary.endowment_seed_stability", "seed_rows", {"seed": 42}, "fit_id", "endowment_frontier/temporal_2022/target-0/seed-42"),
    ("mechanism.payment_kernel", "kernel_summary_rows", {"floor": 0.0, "kernel": "direct"}, "source_id", "wrong"),
    ("mechanism.support_floor", "identification_rows", {"kappa": 1.0}, "welfare_floor", 1.0),
))
def test_wrong_row_identity_is_rejected(consumers, cid, table, selector, field, value):
    _row(consumers, cid, table, **selector)[field] = value
    with pytest.raises(ValueError):
        format_core_macros(consumers)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), True, None, "bad"))
def test_invalid_numbers_are_rejected(consumers, value):
    _row(consumers, "frontier.outcome", "point_rows", floor=1.0, seed=42)["test_worst_csd"] = value
    with pytest.raises(ValueError):
        format_core_macros(consumers)


@pytest.mark.parametrize("field", ("worst_csd", "welfare", "exclusion"))
def test_zero_baseline_denominators_fail(consumers, field):
    _row(consumers, "frontier.outcome", "baseline_rows", policy="mes", view="test")[field] = 0
    with pytest.raises(ValueError, match="denominator"):
        format_core_macros(consumers)


@pytest.mark.parametrize("field,base_field", (("test_welfare", "welfare"), ("test_exclusion", "exclusion")))
def test_zero_repair_gap_denominators_fail(consumers, field, base_field):
    mes = _row(consumers, "frontier.outcome", "baseline_rows", policy="mes", view="test")
    _row(consumers, "frontier.outcome", "point_rows", floor=0.0, seed=42)[field] = mes[base_field]
    with pytest.raises(ValueError, match="denominator"):
        format_core_macros(consumers)


def test_signed_intervals_and_exact_p_threshold(consumers):
    row = _row(consumers, "outcome.significance", "contrast_rows", floor=1.0, baseline="mes")
    row.update(ci_lo=-0.00001, ci_hi=0.0, p_two_sided=0.001)
    records = format_core_macros(consumers)
    assert records["SigHeadCI"]["value"] == "[-0.0000, +0.0000]"
    assert records["SigHeadP"]["value"] == "= 0.001"
    row["p_two_sided"] = 0.000999
    assert format_core_macros(consumers)["SigHeadP"]["value"] == "< 0.001"


@pytest.mark.parametrize("fields", ({"ci_lo": 1, "ci_hi": -1}, {"p_two_sided": -0.1},
                                    {"p_two_sided": 1.01}, {"wins": 19}, {"n": 0}))
def test_invalid_saved_inference_is_rejected(consumers, fields):
    _row(consumers, "outcome.significance", "contrast_rows", floor=1.0, baseline="mes").update(fields)
    with pytest.raises(ValueError):
        format_core_macros(consumers)


def test_percentages_use_raw_values_not_rounded_macros(consumers):
    mes = _row(consumers, "frontier.outcome", "baseline_rows", policy="mes", view="test")
    primary = _row(consumers, "summary.endowment_seed_stability", "seed_rows", seed=42)
    mes["welfare"] = 1000.49
    primary["test_welfare"] = 995.04
    records = format_core_macros(consumers)
    assert records["EndowWelfareCostPct"]["value"] == r"0.5\%"
    assert records["EndowWelfareRatio"]["value"] == "0.995"
    # The support-floor saved welfare ratio uses mes/train. It must not enter
    # the requested held-out repair calculation, which uses raw test welfare.
    _row(consumers, "mechanism.support_floor", "identification_rows", kappa=1.0)["welfare_ratio_vs_mes"] = 999
    assert format_core_macros(consumers)["FloorClosesWelfarePct"] == records["FloorClosesWelfarePct"]
