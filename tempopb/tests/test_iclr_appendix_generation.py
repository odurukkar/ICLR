"""Regression tests for paper-facing appendix table semantics."""

import json
from pathlib import Path
import re
import shutil

import pytest

import gen_iclr_numbers
from gen_iclr_numbers import (
    _age_lookup_macros,
    _cross_district_bound_macros,
    _cross_district_macros,
    _external_support_set_macros,
    _holm_adjust,
    _load_external_support_set_inputs,
    _warsaw_holm_macros,
)
from gen_iclr_appendix import (
    _cross_district_bound_table,
    _cross_district_table,
    _external_city_seed_table,
    _external_gate_table,
    _external_protocol_lineage_table,
    _frontier_table,
    _payment_intervention_table,
    _per_series_table,
    _res_grid_table,
)

ROOT = Path(__file__).resolve().parents[1]
NUMBERS_TEX = ROOT.parent / "iclr_paper" / "tex" / "numbers.tex"
APPENDIX_TABLES_TEX = ROOT.parent / "iclr_paper" / "tex" / "appendix_tables.tex"
EXTERNAL_RESULT_ROOT = ROOT / "results" / "iclr_external_support_set_amendment"
EXTERNAL_ANALYSIS_DIR = ROOT / "analysis-output" / "external-support-set"
RESULTS_ROOT = ROOT / "results"


def test_number_generation_requires_the_primary_seed42_frontier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "frontier_outcome_temporal_2022_seed42.json"
    output = tmp_path / "numbers.tex"
    monkeypatch.setattr(gen_iclr_numbers, "PRIMARY_FRONTIER", missing)
    monkeypatch.setattr(gen_iclr_numbers, "OUT", output)

    with pytest.raises(
        FileNotFoundError,
        match=r"required primary frontier artifact missing: .*seed42\.json",
    ):
        gen_iclr_numbers.main()

    assert not output.exists()


def test_primary_frontier_loader_uses_only_the_temporal_2022_seed42_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = tmp_path / "frontier_outcome_temporal_2022_seed42.json"
    stale = tmp_path / "frontier_outcome_district_seed42.json"
    canonical.write_text(
        json.dumps(
            {"source": "canonical", "split": "temporal_2022", "seed": 42}
        )
    )
    stale.write_text(json.dumps({"source": "stale"}))
    monkeypatch.setattr(gen_iclr_numbers, "PRIMARY_FRONTIER", canonical)

    assert gen_iclr_numbers._load_primary_frontier() == {
        "source": "canonical",
        "split": "temporal_2022",
        "seed": 42,
    }


def test_seed_sweep_loader_uses_only_declared_temporal_2022_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seed_one = tmp_path / "frontier_outcome_temporal_2022_seed1.json"
    seed_42 = tmp_path / "frontier_outcome_temporal_2022_seed42.json"
    stale = tmp_path / "frontier_outcome_district_seed42.json"
    seed_one.write_text(
        json.dumps({"source": "seed1", "split": "temporal_2022", "seed": 1})
    )
    seed_42.write_text(
        json.dumps({"source": "seed42", "split": "temporal_2022", "seed": 42})
    )
    stale.write_text(json.dumps({"source": "stale"}))
    monkeypatch.setattr(
        gen_iclr_numbers,
        "FRONTIER_SEED_PATHS",
        (seed_one, seed_42),
    )

    assert gen_iclr_numbers._load_frontier_seed_sweep() == [
        {"source": "seed1", "split": "temporal_2022", "seed": 1},
        {"source": "seed42", "split": "temporal_2022", "seed": 42},
    ]


def test_number_generation_writes_the_canonical_frontier_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "numbers.tex"
    monkeypatch.setattr(gen_iclr_numbers, "OUT", output)

    gen_iclr_numbers.main()

    assert output.is_file()
    assert (
        f"% source frontier: {gen_iclr_numbers.PRIMARY_FRONTIER.name}"
        in output.read_text()
    )


def test_age_lookup_macros_preserve_the_post_lock_audit() -> None:
    macros = _age_lookup_macros()
    expected = {
        "AgeLookupCSD": "0.1624",
        "AgeLookupExcl": "0.0452",
        "AgeLookupWelfare": "1,261,342",
        "AgeLookupWelfareCostPct": r"2.5\%",
        "AgeLookupExclVsEndowDiff": "+0.0038",
        "AgeLookupExclVsMESDiff": "+0.0014",
        "AgeLookupMultipliers": "0.0564, 0.0528, 0.0895, 20.0855",
        "AgeLookupVsSeniorDiff": "-0.0217",
        "AgeLookupVsSeniorCI": "[-0.0381, -0.0079]",
        "AgeLookupVsSeniorP": "= 0.007",
        "AgeLookupVsSeniorWins": "13/18",
        "AgeLookupVsSeniorRecord": "13/0/5",
        "AgeLookupVsSeniorDz": "-0.642",
        "ContextualVsAgeLookupDiff": "+0.0069",
        "ContextualVsAgeLookupCI": "[-0.0160, +0.0237]",
        "ContextualVsAgeLookupP": "= 0.600",
        "ContextualVsAgeLookupWins": "4/18",
        "ContextualVsAgeLookupRecord": "4/1/13",
        "ContextualVsAgeLookupDz": "+0.153",
    }
    assert macros == expected


def test_holm_adjustment_is_monotone_and_installed_for_warsaw_family() -> None:
    assert _holm_adjust([0.01304, 0.01151, 0.0072479248046875]) == pytest.approx(
        [0.02302, 0.02302, 0.0217437744140625]
    )
    assert _warsaw_holm_macros() == {
        "WarsawHolmFamilyN": "3",
        "SigHeadHolmP": "= 0.023",
        "EndowSigMESHolmP": "= 0.023",
        "TiltHolmP": "= 0.022",
    }


def _external_inputs() -> tuple[dict, dict, dict]:
    bundle, diagnostics = _load_external_support_set_inputs(
        EXTERNAL_RESULT_ROOT, EXTERNAL_ANALYSIS_DIR
    )
    return dict(bundle.decision), dict(bundle.evidence), diagnostics


def test_external_macros_preserve_every_frozen_reporting_value() -> None:
    decision, evidence, diagnostics = _external_inputs()
    macros = _external_support_set_macros(decision, evidence, diagnostics)
    expected = {
        "ExternalDecision": "falsified",
        "ExternalElectionN": "228",
        "ExternalSeriesN": "32",
        "ExternalScoreN": "96",
        "ExternalCoverageMin": r"99.12\%",
        "ExternalEndowmentActuated": "12",
        "ExternalPriorityActuated": "3",
        "ExternalSeedOneMacro": "-0.0131",
        "ExternalSeedTwoMacro": "-0.0054",
        "ExternalSeedFortyTwoMacro": "-0.0072",
        "ExternalSeedSpread": "0.0078",
        "ExternalPrimaryWins": "8",
        "ExternalPrimaryTies": "20",
        "ExternalPrimaryLosses": "4",
        "ExternalAlwaysWins": "4",
        "ExternalAlwaysTies": "13",
        "ExternalSignSwitches": "5",
        "ExternalTopThreeShare": r"78.90\%",
        "ExternalGatePassN": "4",
        "ExternalGateFailN": "4",
        "ExternalKatowiceSeedOneDiff": "-0.0125",
        "ExternalKrakowSeedOneDiff": "-0.0138",
        "ExternalKatowiceSeedTwoDiff": "-0.0027",
        "ExternalKrakowSeedTwoDiff": "-0.0080",
        "ExternalKatowiceSeedFortyTwoDiff": "-0.0117",
        "ExternalKrakowSeedFortyTwoDiff": "-0.0026",
        "ExternalKatowiceSeedOneExclDelta": "+0.0078",
        "ExternalKrakowSeedOneExclDelta": "+0.0037",
        "ExternalKatowiceSeedTwoExclDelta": "+0.0010",
        "ExternalKrakowSeedTwoExclDelta": "-0.0014",
        "ExternalKatowiceSeedFortyTwoExclDelta": "+0.0085",
        "ExternalKrakowSeedFortyTwoExclDelta": "+0.0015",
        "ExternalKatowiceSeedOneWelfareRatio": "0.9913",
        "ExternalKrakowSeedOneWelfareRatio": "1.0014",
        "ExternalKatowiceSeedTwoWelfareRatio": "1.0002",
        "ExternalKrakowSeedTwoWelfareRatio": "1.0015",
        "ExternalKatowiceSeedFortyTwoWelfareRatio": "0.9924",
        "ExternalKrakowSeedFortyTwoWelfareRatio": "1.0041",
        "ExternalKatowicePrimaryDiff": "-0.0117",
        "ExternalKatowicePrimaryCILow": "-0.0282",
        "ExternalKatowicePrimaryCIHigh": "0.0000",
        "ExternalKatowicePrimaryHolmP": "0.5000",
        "ExternalKatowicePrimaryExclDelta": "+0.0085",
        "ExternalKatowicePrimaryWelfareRatio": "0.9924",
        "ExternalKrakowPrimaryDiff": "-0.0026",
        "ExternalKrakowPrimaryCILow": "-0.0259",
        "ExternalKrakowPrimaryCIHigh": "0.0166",
        "ExternalKrakowPrimaryHolmP": "0.8398",
        "ExternalKrakowPrimaryExclDelta": "+0.0015",
        "ExternalKrakowPrimaryWelfareRatio": "1.0041",
    }
    assert macros == expected


def test_external_inputs_reject_manifest_hash_disagreement(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    for name in ("diagnostics.json", "manifest.json"):
        shutil.copy2(EXTERNAL_ANALYSIS_DIR / name, analysis_dir / name)
    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_sha256"]["protocol_lock.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="input manifest hash disagreement"):
        _load_external_support_set_inputs(EXTERNAL_RESULT_ROOT, analysis_dir)


def test_external_city_seed_table_reports_all_six_frozen_rows() -> None:
    _, evidence, _ = _external_inputs()
    table = _external_city_seed_table(evidence)
    expected_rows = (
        "1 & Katowice & -0.0125 & [-0.0307, 0.0016] & -- & +0.0078 & 0.9913",
        "1 & Krakow & -0.0138 & [-0.0407, 0.0082] & -- & +0.0037 & 1.0014",
        "2 & Katowice & -0.0027 & [-0.0080, 0.0000] & -- & +0.0010 & 1.0002",
        "2 & Krakow & -0.0080 & [-0.0278, 0.0051] & -- & -0.0014 & 1.0015",
        "42 & Katowice & -0.0117 & [-0.0282, 0.0000] & 0.5000 & +0.0085 & 0.9924",
        "42 & Krakow & -0.0026 & [-0.0259, 0.0166] & 0.8398 & +0.0015 & 1.0041",
    )
    for row in expected_rows:
        assert row in table
    assert r"$\Delta$ CSD $\downarrow$" in table
    assert r"$\Delta$ exclusion $\downarrow$" in table
    assert r"Welfare ratio $\uparrow$" in table
    assert "significance" not in table
    assert "*" not in table


def test_external_gate_table_reports_every_pass_and_failure() -> None:
    decision, _, _ = _external_inputs()
    table = _external_gate_table(decision)
    assert table.count(r"\checkmark") == 4
    assert table.count(r"\texttimes") == 4
    for phrase in (
        "city macro",
        "each city",
        "paired-bootstrap",
        "Holm",
        "welfare ratio",
        "exclusion increase",
        "actuation",
        "seed spread",
    ):
        assert phrase in table
    assert "falsified" in table


def test_external_protocol_lineage_distinguishes_aborts_from_result() -> None:
    table = _external_protocol_lineage_table(RESULTS_ROOT)
    assert table.count(r"\texttt{v") == 3
    assert "approval-only abort" in table
    assert "Unicode-label technical abort" in table
    assert table.count("no scientific result") == 2
    assert "valid amended run" in table
    assert "falsified" in table
    assert r"protocol\_invalid" in table
    assert r"technical\_invalid" in table
    for digest_prefix in ("064957412947", "1464a34e2274", "a6d7f945b6f6"):
        assert digest_prefix in table
        assert re.fullmatch(r"[0-9a-f]{12}", digest_prefix)


def test_external_protocol_lineage_rejects_hash_disagreement(tmp_path: Path) -> None:
    copied_results = tmp_path / "results"
    for name in (
        "iclr_external_validation",
        "iclr_external_support_set",
        "iclr_external_support_set_amendment",
    ):
        shutil.copytree(RESULTS_ROOT / name, copied_results / name)
    v1_abort = copied_results / "iclr_external_validation" / "abort_record.json"
    v1_abort.write_text(v1_abort.read_text().replace("protocol_invalid", "invalid"))

    with pytest.raises(ValueError, match="lineage SHA-256 mismatch"):
        _external_protocol_lineage_table(copied_results)


def test_generated_numbers_install_external_macros() -> None:
    numbers = NUMBERS_TEX.read_text()
    for line in (
        r"\newcommand{\ExternalDecision}{falsified}",
        r"\newcommand{\ExternalElectionN}{228}",
        r"\newcommand{\ExternalPrimaryTies}{20}",
        r"\newcommand{\ExternalKrakowPrimaryHolmP}{0.8398}",
        r"\newcommand{\ExternalGateFailN}{4}",
    ):
        assert line in numbers


def test_generated_appendix_installs_three_external_tables() -> None:
    tables = APPENDIX_TABLES_TEX.read_text()
    for label in (
        r"\label{tab:external-city-seed}",
        r"\label{tab:external-gates}",
        r"\label{tab:external-lineage}",
    ):
        assert label in tables
    assert "42 & Krakow & -0.0026 & [-0.0259, 0.0166]" in tables
    assert tables.count(r"\checkmark") == 4
    assert tables.count(r"\texttimes") == 4
    assert "No---no scientific result" in tables


def test_per_series_table_names_the_policy_and_estimand() -> None:
    table = _per_series_table()
    assert "learned outcome policy" in table
    assert "mean district-level" in table


def test_frontier_table_is_generated_and_describes_soft_targets() -> None:
    table = _frontier_table()
    assert "mean district-level worst-group" in table
    assert "soft welfare targets" in table
    assert "Repeated rows" not in table


def test_res_grid_table_reports_every_tested_rollover_intensity() -> None:
    payload = {
        "baselines": {
            "mes/test": {"worst_csd": 0.2, "welfare": 1000, "exclusion": 0.05},
            "res-0.25/test": {"worst_csd": 0.19, "welfare": 990, "exclusion": 0.04},
            "res-0.5/test": {"worst_csd": 0.18, "welfare": 980, "exclusion": 0.03},
            "res-0.75/test": {"worst_csd": 0.17, "welfare": 970, "exclusion": 0.02},
            "res-1.0/test": {"worst_csd": 0.16, "welfare": 960, "exclusion": 0.01},
        }
    }
    table = _res_grid_table(payload)
    for intensity in ("0.00", "0.25", "0.50", "0.75", "1.00"):
        assert intensity in table
    assert "All tested rollover intensities" in table
    assert r"\label{tab:res-grid}" in table


def test_payment_intervention_table_calls_welfare_knob_a_soft_target() -> None:
    table = _payment_intervention_table()
    assert table.startswith(r"\begin{table}[H]")
    assert r"\clearpage" not in table
    assert r"\begin{table}[H]" in table
    assert r"\begin{table}[t]" not in table
    assert r"\begin{table}[!b]" not in table
    assert "Soft target" in table
    assert "welfare-floor" not in table


def test_generated_appendix_places_immediate_kernel_table_before_queued_floats() -> None:
    tables = APPENDIX_TABLES_TEX.read_text()
    assert tables.index(r"\label{tab:kernel-intervention}") < tables.index(
        r"\label{tab:perseries}"
    )


def test_generated_numbers_include_recorded_outcome_baseline() -> None:
    numbers = NUMBERS_TEX.read_text()
    assert r"\newcommand{\TestHistoricalCSD}" in numbers
    assert r"\newcommand{\TestHistoricalWelfare}" in numbers
    assert r"\newcommand{\TestHistoricalExcl}" in numbers


def test_generated_numbers_include_recorded_runtime_ranges() -> None:
    numbers = NUMBERS_TEX.read_text()
    for macro in (
        "EndowRuntimeLoMin",
        "EndowRuntimeHiMin",
        "OutcomeRuntimeLoMin",
        "OutcomeRuntimeHiMin",
    ):
        assert rf"\newcommand{{\{macro}}}" in numbers


def test_generated_numbers_include_primary_corpus_scale() -> None:
    numbers = NUMBERS_TEX.read_text()
    expected = {
        "PrimaryBallotN": "821,572",
        "PrimaryProjectN": "8,366",
        "PrimaryYearLo": "2016",
        "PrimaryYearHi": "2025",
    }
    for macro, value in expected.items():
        assert rf"\newcommand{{\{macro}}}{{{value}}}" in numbers


def _district_summary_rows() -> list[dict[str, str]]:
    return [
        {
            "baseline": "mes",
            "learned_mean": "0.17",
            "baseline_mean": "0.20",
            "diff": "-0.03",
            "ci_lo": "-0.05",
            "ci_hi": "-0.01",
            "p_two_sided": "0.012",
            "fold_mean_diff": "-0.028",
            "fold_p_two_sided": "0.250",
            "fold_wins": "3",
            "n_folds": "3",
            "wins": "13",
            "ties": "0",
            "n": "19",
            "learned_welfare": "1000",
            "baseline_welfare": "1020",
            "welfare_ratio": "0.980",
            "learned_exclusion": "0.040",
            "baseline_exclusion": "0.050",
        },
        {
            "baseline": "greedy-cost",
            "learned_mean": "0.17",
            "baseline_mean": "0.19",
            "diff": "-0.02",
            "ci_lo": "-0.04",
            "ci_hi": "0.00",
            "p_two_sided": "0.070",
            "fold_mean_diff": "-0.018",
            "fold_p_two_sided": "0.500",
            "fold_wins": "2",
            "n_folds": "3",
            "wins": "12",
            "ties": "0",
            "n": "19",
            "learned_welfare": "1000",
            "baseline_welfare": "1100",
            "welfare_ratio": "0.909",
            "learned_exclusion": "0.040",
            "baseline_exclusion": "0.060",
        },
    ]


def test_cross_district_macros_expose_pooled_mes_contrast() -> None:
    macros = _cross_district_macros(_district_summary_rows())
    assert macros["DistrictCVLearnedCSD"] == "0.1700"
    assert macros["DistrictCVMESCSD"] == "0.2000"
    assert macros["DistrictCVDiff"] == "-0.0300"
    assert macros["DistrictCVCI"] == "[-0.0500, -0.0100]"
    assert macros["DistrictCVP"] == "= 0.2500"
    assert macros["DistrictCVSeriesP"] == "= 0.012"
    assert macros["DistrictCVFoldWins"] == "3/3"
    assert macros["DistrictCVWins"] == "13/19"
    assert macros["DistrictCVWelfareRatio"] == "0.980"
    assert macros["DistrictCVWelfareCostPct"] == r"2.0\%"


def test_cross_district_table_states_generalization_scope() -> None:
    table = _cross_district_table(_district_summary_rows())
    assert r"Learned $\csd$" in table
    assert r"95\% $\Delta$ interval" in table
    assert "fitted on the other districts" in table
    assert "not a temporal holdout" in table
    assert "not cross-city validation" in table
    assert "0.2500" in table
    assert "dependence sensitivity check" in table
    assert "Series wins" in table
    assert "Fold wins" in table
    assert "greedy-cost" in table


def test_cross_district_bound_table_marks_posthoc_outcome_sensitivity() -> None:
    primary = _district_summary_rows()
    sensitivity = [dict(row) for row in primary]
    sensitivity[0]["learned_mean"] = "0.168"
    sensitivity[0]["diff"] = "-0.032"
    table = _cross_district_bound_table(primary, sensitivity)
    assert "Post hoc fourfold box-expansion sensitivity" in table
    assert "primary analysis" in table
    assert "outcome stability" in table
    assert "coefficient identification" in table
    assert "0.1700" in table
    assert "0.1680" in table
    assert r"\label{tab:district-bound}" in table


def test_cross_district_bound_macros_compare_outcomes_not_weights() -> None:
    primary = _district_summary_rows()
    sensitivity = [dict(row) for row in primary]
    sensitivity[0].update(
        learned_mean="0.168",
        diff="-0.032",
        ci_lo="-0.052",
        ci_hi="-0.012",
        fold_p_two_sided="0.125",
        wins="15",
        fold_wins="3",
        welfare_ratio="0.975",
        learned_exclusion="0.038",
    )
    macros = _cross_district_bound_macros(primary, sensitivity)
    assert macros["DistrictCVWideLearnedCSD"] == "0.1680"
    assert macros["DistrictCVWideDiff"] == "-0.0320"
    assert macros["DistrictCVWideCI"] == "[-0.0520, -0.0120]"
    assert macros["DistrictCVWideP"] == "= 0.1250"
    assert macros["DistrictCVWideWins"] == "15/19"
    assert macros["DistrictCVWideFoldWins"] == "3/3"
    assert macros["DistrictCVWideWelfareRatio"] == "0.975"
    assert macros["DistrictCVWideWelfareCostPct"] == r"2.5\%"
    assert macros["DistrictCVWideLearnedExcl"] == "0.0380"
    assert macros["DistrictCVWideCSDShift"] == "-0.0020"
