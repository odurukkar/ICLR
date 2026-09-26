"""Generate iclr_paper/tex/numbers.tex from the results directory.

Headline experimental values and corpus counts resolve to macros defined here,
which in turn resolve to frozen result files or the executable corpus index.
Methodological constants remain stated directly in the manuscript. If a result
or corpus filter changes, the generated values change with it or the build fails
loudly.

Sources:
  results/iclr_baselines.csv                    whole-corpus baseline table
  results/iclr_frontier/frontier_*_seed*.json   frontier points + test baselines
  results/iclr_significance.csv                 paired CIs and exact tests
  results/iclr_per_series.csv                   per-series wins and losses
  results/iclr_ablation.csv                     leave-one-feature-out
  results/iclr_degeneracy.csv                   representative-support theorem
  results/iclr_llmrule_baseline.csv             frozen closest-prior rules
  results/iclr_payment_intervention_summary.csv fixed-score kernel ablation
  results/iclr_cross_district_summary.csv       pooled disjoint-series check
  results/iclr_cross_district_b40_summary.csv   post hoc district bound check
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from analyze_external_support_set_results import (
    EXPECTED_SHA256,
    VerifiedExternalBundle,
    load_and_verify_bundle,
    sha256_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "iclr_paper" / "tex" / "numbers.tex"
HEADLINE_FLOOR = 1.0
AGGRESSIVE_FLOOR = 0.85
EXTERNAL_RESULT_ROOT = RESULTS / "iclr_external_support_set_amendment"
EXTERNAL_ANALYSIS_DIR = ROOT / "analysis-output" / "external-support-set"
PRIMARY_FRONTIER = (
    RESULTS
    / "iclr_frontier"
    / "frontier_outcome_temporal_2022_seed42.json"
)
FRONTIER_SEED_PATHS = tuple(
    RESULTS
    / "iclr_frontier"
    / f"frontier_outcome_temporal_2022_seed{seed}.json"
    for seed in (1, 2, 3, 42)
)
ENDOWMENT_SEED_PATHS = (
    RESULTS / "iclr_train" / "run_main_seed42.json",
    RESULTS / "iclr_train" / "run_endow_seed1.json",
    RESULTS / "iclr_train" / "run_endow_seed2.json",
)

_ENDOWMENT_SEED_BY_NAME = {
    "run_main_seed42.json": 42,
    "run_endow_seed1.json": 1,
    "run_endow_seed2.json": 2,
}


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        logger.warning("missing %s", path)
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _validate_artifact_metadata(
    payload: object,
    *,
    path: Path,
    expected_seed: int,
    nested_config: bool = False,
) -> dict[str, Any]:
    """Bind a declared artifact path to its temporal split and seed payload."""
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    metadata = payload.get("config") if nested_config else payload
    if not isinstance(metadata, dict):
        location = "config object" if nested_config else "metadata"
        raise ValueError(f"{path}: expected {location}")
    found_split = metadata.get("split")
    if found_split != "temporal_2022":
        raise ValueError(
            f"{path}: expected split temporal_2022, found {found_split!r}"
        )
    found_seed = metadata.get("seed")
    if found_seed != expected_seed or isinstance(found_seed, bool):
        raise ValueError(
            f"{path}: expected seed {expected_seed}, found {found_seed!r}"
        )
    return payload


def _load_frontier_artifact(path: Path, *, expected_seed: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required frontier artifact missing: {path}")
    return _validate_artifact_metadata(
        json.loads(path.read_text()),
        path=path,
        expected_seed=expected_seed,
    )


def _load_primary_frontier() -> dict[str, Any]:
    """Load only the canonical temporal-split seed-42 outcome frontier."""
    if not PRIMARY_FRONTIER.is_file():
        raise FileNotFoundError(
            f"required primary frontier artifact missing: {PRIMARY_FRONTIER}"
        )
    return _load_frontier_artifact(PRIMARY_FRONTIER, expected_seed=42)


def _load_frontier_seed_sweep() -> list[dict[str, Any]]:
    """Load the four declared temporal-split outcome-frontier seeds."""
    missing = [path for path in FRONTIER_SEED_PATHS if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"required temporal-2022 frontier seed artifacts missing: {joined}"
        )
    loaded = []
    for path in FRONTIER_SEED_PATHS:
        name = path.name
        prefix = "frontier_outcome_temporal_2022_seed"
        if not name.startswith(prefix) or not name.endswith(".json"):
            raise ValueError(f"unexpected declared frontier artifact name: {path}")
        seed_text = name[len(prefix) : -len(".json")]
        if not seed_text.isdigit():
            raise ValueError(f"unexpected declared frontier artifact name: {path}")
        loaded.append(
            _load_frontier_artifact(path, expected_seed=int(seed_text))
        )
    return loaded


def _load_endowment_seed_artifact(path: Path) -> dict[str, Any]:
    """Load one explicitly declared endowment run and verify its identity."""
    expected_seed = _ENDOWMENT_SEED_BY_NAME.get(path.name)
    if expected_seed is None:
        raise ValueError(f"unexpected declared endowment artifact name: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"required endowment seed artifact missing: {path}")
    return _validate_artifact_metadata(
        json.loads(path.read_text()),
        path=path,
        expected_seed=expected_seed,
        nested_config=True,
    )


def _load_endowment_seed_runs() -> list[dict[str, Any]]:
    """Load only the three declared temporal-split endowment runs."""
    missing = [path for path in ENDOWMENT_SEED_PATHS if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"required endowment seed artifacts missing: {joined}"
        )
    return [_load_endowment_seed_artifact(path) for path in ENDOWMENT_SEED_PATHS]


def _exact_frontier_point(
    frontier: Mapping[str, Any], floor: float
) -> dict[str, Any]:
    """Return the sole point at ``floor``; never substitute a nearby target."""
    points = frontier.get("points")
    if not isinstance(points, list):
        raise ValueError("frontier payload has no points list")
    matches = [
        point
        for point in points
        if isinstance(point, dict) and point.get("floor") == floor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one frontier point at floor {floor:.2f}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _endowment_seed_summary_macros() -> Dict[str, str]:
    """Summarize the declared endowment runs without discovering stale files."""
    seed_vals = [
        float(run["learned"]["test"]["worst_csd"])
        for run in _load_endowment_seed_runs()
    ]
    return {
        "EndowNumSeeds": str(len(seed_vals)),
        "EndowSeedLo": _fmt(min(seed_vals)),
        "EndowSeedHi": _fmt(max(seed_vals)),
        "EndowSeedSpread": _fmt(max(seed_vals) - min(seed_vals)),
    }


def _runtime_macros() -> Dict[str, str]:
    """Report runtimes from only the declared paper-facing seed artifacts."""
    endow_elapsed = [
        float(run["elapsed_sec"])
        for run in _load_endowment_seed_runs()
        if run.get("elapsed_sec") is not None
    ]
    endow_frontier = (
        RESULTS
        / "iclr_frontier"
        / "frontier_endowment_temporal_2022_seed42.json"
    )
    if endow_frontier.exists():
        endow_elapsed.extend(
            float(point["elapsed_sec"])
            for point in json.loads(endow_frontier.read_text())["points"]
        )

    outcome_elapsed = [
        float(point["elapsed_sec"])
        for frontier in _load_frontier_seed_sweep()
        for point in frontier["points"]
    ]
    macros: Dict[str, str] = {}
    for tag, values in (("Endow", endow_elapsed), ("Outcome", outcome_elapsed)):
        if values:
            macros[f"{tag}RuntimeLoMin"] = f"{min(values) / 60:.0f}"
            macros[f"{tag}RuntimeHiMin"] = f"{max(values) / 60:.0f}"
    return macros


def _pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f}\\%"


def _p(value: float) -> str:
    """Format a p-value without ever rendering a small one as literally zero."""
    if value < 1e-3:
        return "< 0.001"
    return f"= {value:.3f}"


def _holm_adjust(values: Sequence[float]) -> List[float]:
    """Return Holm--Bonferroni adjusted p-values in the input order."""
    count = len(values)
    order = sorted(range(count), key=lambda idx: values[idx])
    adjusted = [0.0] * count
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (count - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def _age_lookup_macros() -> Dict[str, str]:
    """Read the post-lock static age-lookup audit for paper-facing reporting."""
    audit_path = (
        ROOT
        / "analysis-output"
        / "ml-contribution-audit"
        / "static_age_lookup_seed42_g30.json"
    )
    audit = json.loads(audit_path.read_text())
    summary = audit["summary"]
    metrics = summary["policy_metrics"]["age_lookup"]
    age_vs_senior = summary["age_lookup_minus_static_60plus"]
    history_free_vs_age = summary["history_free_minus_age_lookup"]

    main_run = json.loads(
        (RESULTS / "iclr_train" / "run_main_seed42.json").read_text()
    )
    mes_metrics = main_run["baselines"]["mes/test"]
    learned_metrics = main_run["learned"]["test"]

    multipliers = ", ".join(_fmt(float(value)) for value in summary["best_multipliers"])
    return {
        "AgeLookupCSD": _fmt(float(metrics["mean_csd"])),
        "AgeLookupExcl": _fmt(float(metrics["mean_exclusion"])),
        "AgeLookupWelfare": f"{float(metrics['welfare']):,.0f}",
        "AgeLookupWelfareCostPct": _pct(
            1 - float(metrics["welfare"]) / float(mes_metrics["welfare"]), 1
        ),
        "AgeLookupExclVsEndowDiff": f"{float(metrics['mean_exclusion']) - float(learned_metrics['exclusion']):+.4f}",
        "AgeLookupExclVsMESDiff": f"{float(metrics['mean_exclusion']) - float(mes_metrics['exclusion']):+.4f}",
        "AgeLookupMultipliers": multipliers,
        "AgeLookupVsSeniorDiff": f"{float(age_vs_senior['difference']):+.4f}",
        "AgeLookupVsSeniorCI": (
            f"[{float(age_vs_senior['ci_low']):+.4f}, "
            f"{float(age_vs_senior['ci_high']):+.4f}]"
        ),
        "AgeLookupVsSeniorP": _p(
            float(age_vs_senior["p_two_sided_exact_sign_flip"])
        ),
        "AgeLookupVsSeniorWins": (
            f"{age_vs_senior['wins']}/{age_vs_senior['n_series']}"
        ),
        "AgeLookupVsSeniorRecord": (
            f"{age_vs_senior['wins']}/{age_vs_senior['ties']}/{age_vs_senior['losses']}"
        ),
        "AgeLookupVsSeniorDz": f"{float(age_vs_senior['paired_cohens_dz']):+.3f}",
        "HistoryFreeVsAgeLookupDiff": f"{float(history_free_vs_age['difference']):+.4f}",
        "HistoryFreeVsAgeLookupCI": (
            f"[{float(history_free_vs_age['ci_low']):+.4f}, "
            f"{float(history_free_vs_age['ci_high']):+.4f}]"
        ),
        "HistoryFreeVsAgeLookupP": _p(
            float(history_free_vs_age["p_two_sided_exact_sign_flip"])
        ),
        "HistoryFreeVsAgeLookupWins": (
            f"{history_free_vs_age['wins']}/{history_free_vs_age['n_series']}"
        ),
        "HistoryFreeVsAgeLookupRecord": (
            f"{history_free_vs_age['wins']}/{history_free_vs_age['ties']}/{history_free_vs_age['losses']}"
        ),
        "HistoryFreeVsAgeLookupDz": f"{float(history_free_vs_age['paired_cohens_dz']):+.3f}",
    }


def _contextual_age_lookup_macros() -> Dict[str, str]:
    """Read the separately verified contextual contrast, never the history-free one."""
    path = ROOT / "analysis-output/contextual-age-lookup-20260920/summary.json"
    raw = path.read_bytes()
    expected = "2b0a3a7b59cbd2d6952a6dca341b657c7b5269747a3cdfc7785a87ad15a6242c"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("contextual/lookup correction artifact identity changed")
    payload = json.loads(raw)
    if (
        payload["contextual_fit_id"] != "temporal_endowment/temporal_2022/seed-42"
        or payload["age_lookup_fit_id"] != "static_age_lookup/temporal_2022/seed-42"
        or payload["statistics"]["n_series"] != 18
    ):
        raise ValueError("wrong policy identity in contextual/lookup correction")
    row = payload["statistics"]
    return {
        "ContextualVsAgeLookupDiff": f"{float(row['difference']):+.4f}",
        "ContextualVsAgeLookupDiffAbs": f"{abs(float(row['difference'])):.4f}",
        "ContextualVsAgeLookupCI": (
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"
        ),
        "ContextualVsAgeLookupP": _p(float(row["p_two_sided_exact_sign_flip"])),
        "ContextualVsAgeLookupRecord": f"{row['wins']}/{row['ties']}/{row['losses']}",
        "ContextualVsAgeLookupWins": f"{row['wins']}/{row['n_series']}",
        "ContextualVsAgeLookupDz": f"{float(row['paired_cohens_dz']):+.3f}",
    }


def _warsaw_holm_macros() -> Dict[str, str]:
    """Adjust the three focal Warsaw-versus-reference contrasts as one family."""
    outcome = next(
        row
        for row in _read_csv(RESULTS / "iclr_significance.csv")
        if row["baseline"] == "mes" and float(row["floor"]) == HEADLINE_FLOOR
    )
    endowment = next(
        row
        for row in _read_csv(RESULTS / "iclr_endow_significance.csv")
        if row["contrast"] == "learned-endowment vs mes"
    )
    senior = next(
        row
        for row in _read_csv(RESULTS / "iclr_senior_tilt.csv")
        if row.get("test_worst_csd")
    )
    adjusted = _holm_adjust(
        [
            float(outcome["p_two_sided"]),
            float(endowment["p_two_sided"]),
            float(senior["p_value"]),
        ]
    )
    return {
        "WarsawHolmFamilyN": str(len(adjusted)),
        "SigHeadHolmP": _p(adjusted[0]),
        "EndowSigMESHolmP": _p(adjusted[1]),
        "TiltHolmP": _p(adjusted[2]),
    }


def _crosscity_macros() -> Dict[str, str]:
    """Macros for the native-approval cross-city evaluation.

    This study needs no ballot projection: Gdynia and Lodz record approval
    ballots directly, so it complements the projected support-set study rather
    than duplicating it. Every value is read from the frozen manifest written
    by ``iclr_crosscity.py``.
    """
    manifest_path = RESULTS / "iclr_crosscity_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} missing; run iclr_crosscity.py before generating numbers"
        )
    manifest = json.loads(manifest_path.read_text())
    corpus = manifest["corpus"]
    actuation = manifest["actuation"]
    macros = {
        "CrossCitySeriesN": str(corpus["n_series"]),
        "CrossCityElectionN": str(corpus["n_elections"]),
        "CrossCityScoredElectionN": str(corpus["n_scored_elections"]),
        "CrossCityCityN": str(len(corpus["cities"])),
        "CrossCityInertN": str(actuation["n_inert"]),
        "CrossCityActiveN": str(actuation["n_active"]),
        "CrossCityMedianProjectsInert": _fmt(actuation["median_projects_inert"], 1),
        "CrossCityMedianProjectsActive": _fmt(actuation["median_projects_active"], 1),
    }
    scope_names = {"ALL": "All", "Poland/Gdynia": "Gdynia", "Poland/Łódź": "Lodz"}
    for key, row in manifest["contrasts"].items():
        scope, _, baseline = key.split("|")
        if baseline != "mes" or scope not in scope_names:
            continue
        prefix = f"CrossCity{scope_names[scope]}"
        macros[f"{prefix}Learned"] = _fmt(row["learned_csd"])
        macros[f"{prefix}MES"] = _fmt(row["baseline_csd"])
        macros[f"{prefix}Diff"] = f"{row['difference']:+.4f}"
        macros[f"{prefix}DiffAbs"] = _fmt(abs(row["difference"]))
        macros[f"{prefix}CILow"] = _fmt(row["ci_low"])
        macros[f"{prefix}CIHigh"] = _fmt(row["ci_high"])
        macros[f"{prefix}P"] = _p(row["p_value"])
        macros[f"{prefix}Wins"] = f"{row['wins']}/{row['n_series']}"
        macros[f"{prefix}SeriesN"] = str(row["n_series"])

    strata = manifest["actuation_by_granularity"]
    coarsest = max(strata, key=lambda r: r["min_projects_per_election"])
    macros["CrossCityStratMaxThreshold"] = str(coarsest["min_projects_per_election"])
    macros["CrossCityStratMaxN"] = str(coarsest["n_series"])
    macros["CrossCityStratMaxDiff"] = f"{coarsest['difference']:+.4f}"
    macros["CrossCityStratMinP"] = _fmt(min(r["p_value"] for r in strata), 2)
    return macros


def _reviewer_check_macros() -> Dict[str, str]:
    """Macros for the matched-target, coverage, tilt and actuation checks."""
    macros: Dict[str, str] = {}

    matched = _read_csv(RESULTS / "iclr_matched_target.csv")
    if matched:
        row = matched[0]
        macros["MatchedTargetEndowCSD"] = _fmt(float(row["endowment_csd"]))
        macros["MatchedTargetDirectCSD"] = _fmt(float(row["direct_csd"]))
        macros["MatchedTargetDiff"] = f"{float(row['difference']):+.4f}"
        macros["MatchedTargetCILow"] = _fmt(float(row["ci_low"]))
        macros["MatchedTargetCIHigh"] = _fmt(float(row["ci_high"]))
        macros["MatchedTargetP"] = _p(float(row["p_value"]))
        macros["MatchedTargetWins"] = f"{row['endowment_wins']}/{row['n_series']}"
        macros["MatchedTargetEndowExcl"] = _fmt(float(row["endowment_exclusion"]))
        macros["MatchedTargetDirectExcl"] = _fmt(float(row["direct_exclusion"]))

    coverage = _read_csv(RESULTS / "iclr_attribution_coverage.csv")
    policy_rows = [r for r in coverage if r["policy"] != "series_clean_for_mes_and_learned_endowment"]
    if policy_rows:
        macros["CoverageScoredEditions"] = policy_rows[0]["scored_editions"]
        worst = max(policy_rows, key=lambda r: float(r["uncovered_budget_share"]))
        macros["CoverageMaxUncoveredShare"] = _pct(float(worst["uncovered_budget_share"]), 2)
        macros["CoverageAffectedEditions"] = str(
            max(int(r["editions_with_uncovered_funded_project"]) for r in policy_rows)
        )
        macros["CoveragePolicyN"] = str(len(policy_rows))

    tilt = _read_csv(RESULTS / "iclr_senior_tilt.csv")
    fitted = [r for r in tilt if r.get("test_worst_csd")]
    if fitted:
        row = fitted[0]
        macros["TiltAlpha"] = _fmt(float(row["alpha"]), 2)
        macros["TiltCSD"] = _fmt(float(row["test_worst_csd"]))
        macros["TiltMES"] = _fmt(float(row["mes_worst_csd"]))
        macros["TiltDiff"] = f"{float(row['difference']):+.4f}"
        macros["TiltCILow"] = _fmt(float(row["ci_low"]))
        macros["TiltCIHigh"] = _fmt(float(row["ci_high"]))
        macros["TiltP"] = _p(float(row["p_value"]))
        macros["TiltWins"] = f"{row['wins']}/{row['n_series']}"

    granularity = _read_csv(RESULTS / "iclr_instance_granularity.csv")
    label_by_corpus = {"warsaw_fitting": "Warsaw", "external_projected": "External"}
    for row in granularity:
        label = label_by_corpus.get(row["corpus"])
        if label:
            macros[f"{label}MedianProjects"] = _fmt(float(row["median_projects"]), 1)
            macros[f"{label}ScoredElections"] = row["scored_elections"]

    actuating = _read_csv(RESULTS / "iclr_actuating_subset.csv")
    primary = next((r for r in actuating if r["seed"] == "42"), None)
    if primary:
        macros["ActuatingSeries"] = f"{primary['n_actuating']}/{primary['n_series']}"
        macros["ActuatingInert"] = primary["n_inert"]
        macros["ActuatingWins"] = primary["wins_where_actuating"]
        macros["ActuatingLosses"] = primary["losses_where_actuating"]
        macros["ActuatingMeanAll"] = f"{float(primary['mean_all_series']):+.4f}"
        macros["ActuatingMeanSubset"] = f"{float(primary['mean_where_actuating']):+.4f}"
    if actuating:
        subset = [float(r["mean_where_actuating"]) for r in actuating]
        macros["ActuatingMeanSubsetLo"] = f"{min(subset):+.4f}"
        macros["ActuatingMeanSubsetHi"] = f"{max(subset):+.4f}"
    return macros


def _load_external_support_set_inputs(
    result_root: Path = EXTERNAL_RESULT_ROOT,
    analysis_dir: Path = EXTERNAL_ANALYSIS_DIR,
) -> tuple[VerifiedExternalBundle, dict[str, Any]]:
    """Load verified external results and their hash-bound diagnostics."""
    bundle = load_and_verify_bundle(result_root)
    manifest_path = analysis_dir / "manifest.json"
    diagnostics_path = analysis_dir / "diagnostics.json"
    if not manifest_path.is_file() or not diagnostics_path.is_file():
        raise ValueError("external analysis manifest and diagnostics must exist")

    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("external analysis manifest must be a JSON object")
    live_input_hashes = {
        relative: sha256_file(bundle.root / relative)
        for relative in sorted(EXPECTED_SHA256)
    }
    if manifest.get("input_sha256") != live_input_hashes:
        raise ValueError("external input manifest hash disagreement")

    diagnostics_bytes = diagnostics_path.read_bytes()
    diagnostics_digest = hashlib.sha256(diagnostics_bytes).hexdigest()
    registered_digest = manifest.get("output_sha256", {}).get("diagnostics.json")
    if registered_digest != diagnostics_digest:
        raise ValueError("external diagnostics hash disagreement")
    diagnostics = json.loads(diagnostics_bytes)
    if not isinstance(diagnostics, dict):
        raise ValueError("external diagnostics must be a JSON object")
    if (
        manifest.get("classification") != bundle.decision["classification"]
        or diagnostics.get("decision") != bundle.decision["classification"]
    ):
        raise ValueError("external classification disagreement")

    protocol_counts = bundle.lock["protocol"]["expected_counts"]
    enriched_diagnostics = dict(diagnostics)
    enriched_diagnostics["protocol_expected_counts"] = dict(protocol_counts)
    return bundle, enriched_diagnostics


def _external_support_set_macros(
    decision: Mapping[str, Any],
    evidence: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, str]:
    """Format every external-study manuscript value from verified mappings."""
    if decision.get("classification") != "falsified":
        raise ValueError("external decision must remain falsified")
    if evidence.get("primary_seed") != "42":
        raise ValueError("external primary seed must remain 42")

    conditions = decision.get("conditions")
    if not isinstance(conditions, Mapping) or len(conditions) != 8:
        raise ValueError("external decision must contain exactly eight conditions")
    gate_passes = sum(bool(values["passed"]) for values in conditions.values())
    gate_failures = len(conditions) - gate_passes

    counts = diagnostics["protocol_expected_counts"]
    primary = diagnostics["primary"]
    across_seeds = diagnostics["across_seeds"]
    actuation = diagnostics["actuation"]
    seed_macros = diagnostics["city_macro_differences"]
    if set(seed_macros) != {"1", "2", "42"}:
        raise ValueError("external diagnostics must contain seeds 1, 2, and 42")
    seed_spread = max(float(value) for value in seed_macros.values()) - min(
        float(value) for value in seed_macros.values()
    )

    macros = {
        "ExternalDecision": str(decision["classification"]),
        "ExternalElectionN": str(counts["elections"]),
        "ExternalSeriesN": str(counts["series"]),
        "ExternalScoreN": str(counts["score_elections"]),
        "ExternalCoverageMin": (
            f"{100 * float(diagnostics['minimum_demographic_coverage']):.2f}\\%"
        ),
        "ExternalEndowmentActuated": str(actuation["endowment"]),
        "ExternalPriorityActuated": str(actuation["priority"]),
        "ExternalSeedOneMacro": f"{float(seed_macros['1']):.4f}",
        "ExternalSeedTwoMacro": f"{float(seed_macros['2']):.4f}",
        "ExternalSeedFortyTwoMacro": f"{float(seed_macros['42']):.4f}",
        "ExternalSeedSpread": f"{seed_spread:.4f}",
        "ExternalPrimaryWins": str(primary["wins"]),
        "ExternalPrimaryTies": str(primary["ties"]),
        "ExternalPrimaryLosses": str(primary["losses"]),
        "ExternalAlwaysWins": str(across_seeds["always_win"]),
        "ExternalAlwaysTies": str(across_seeds["always_tie"]),
        "ExternalSignSwitches": str(across_seeds["switches_sign"]),
        "ExternalTopThreeShare": (
            f"{100 * float(diagnostics['top3_improvement_concentration']):.2f}\\%"
        ),
        "ExternalGatePassN": str(gate_passes),
        "ExternalGateFailN": str(gate_failures),
    }

    seed_tags = {"1": "One", "2": "Two", "42": "FortyTwo"}
    city_tags = {"Poland/Katowice": "Katowice", "Poland/Krakow": "Krakow"}
    seeds = evidence.get("seeds")
    if not isinstance(seeds, Mapping) or set(seeds) != set(seed_tags):
        raise ValueError("external evidence must contain seeds 1, 2, and 42")
    for seed, seed_tag in seed_tags.items():
        cities = seeds[seed]["cities"]
        if set(cities) != set(city_tags):
            raise ValueError(f"external seed {seed} must contain both frozen cities")
        for city, city_tag in city_tags.items():
            values = cities[city]
            prefix = f"External{city_tag}Seed{seed_tag}"
            macros[f"{prefix}Diff"] = f"{float(values['difference']):.4f}"
            macros[f"{prefix}ExclDelta"] = (
                f"{float(values['exclusion_difference']):+.4f}"
            )
            macros[f"{prefix}WelfareRatio"] = (
                f"{float(values['welfare_ratio']):.4f}"
            )

    primary_cities = seeds["42"]["cities"]
    for city, city_tag in city_tags.items():
        values = primary_cities[city]
        prefix = f"External{city_tag}Primary"
        macros[f"{prefix}Diff"] = f"{float(values['difference']):.4f}"
        macros[f"{prefix}CILow"] = f"{float(values['ci_low']):.4f}"
        macros[f"{prefix}CIHigh"] = f"{float(values['ci_high']):.4f}"
        macros[f"{prefix}HolmP"] = f"{float(values['holm_p_value']):.4f}"
        macros[f"{prefix}ExclDelta"] = (
            f"{float(values['exclusion_difference']):+.4f}"
        )
        macros[f"{prefix}WelfareRatio"] = f"{float(values['welfare_ratio']):.4f}"
    return macros


def _multicity_macros(rows: List[dict]) -> Dict[str, str]:
    """Format the descriptive no-refit city stress test for LaTeX."""
    scope_tags = {
        "Poland/Gdynia": "Gdynia",
        "Poland/Lublin": "Lublin",
        "Poland/Łódź": "Lodz",
        "all-non-Warsaw": "All",
    }
    policy_tags = {
        "mes": "MES",
        "learned-endowment": "Endow",
        "learned-outcome-f1.00": "Outcome",
    }
    macros: Dict[str, str] = {}
    city_scopes = {
        row["scope"]
        for row in rows
        if row.get("row_type") == "city" and row.get("scope") in scope_tags
    }
    if city_scopes:
        macros["MultiCityN"] = str(len(city_scopes))
    for row in rows:
        scope_tag = scope_tags.get(row.get("scope", ""))
        policy_tag = policy_tags.get(row.get("policy", ""))
        if scope_tag is None or policy_tag is None:
            continue
        prefix = f"Multi{scope_tag}"
        macros[f"{prefix}SeriesN"] = row["n_series"]
        macros[f"{prefix}ElectionN"] = row["n_elections"]
        macros[f"{prefix}{policy_tag}CSD"] = _fmt(float(row["worst_csd"]))
        macros[f"{prefix}{policy_tag}Diff"] = f"{float(row['delta_vs_mes']):+.4f}"
        macros[f"{prefix}{policy_tag}WelfareRatio"] = (
            f"{float(row['welfare_ratio_vs_mes']):.3f}"
        )
        macros[f"{prefix}{policy_tag}ExclDelta"] = (
            f"{float(row['exclusion_delta_vs_mes']):+.4f}"
        )
        if policy_tag in {"Endow", "Outcome"}:
            macros[f"{prefix}{policy_tag}Gain"] = _fmt(
                abs(float(row["delta_vs_mes"]))
            )
            macros[f"{prefix}{policy_tag}WelfareCostPct"] = _pct(
                max(0.0, 1.0 - float(row["welfare_ratio_vs_mes"])), 1
            )
    return macros


def _cross_district_macros(rows: List[dict]) -> Dict[str, str]:
    """Format the primary pooled district-held-out contrast against MES."""
    row = next((item for item in rows if item.get("baseline") == "mes"), None)
    if row is None:
        return {}
    welfare_ratio = float(row["welfare_ratio"])
    learned_exclusion = float(row["learned_exclusion"])
    baseline_exclusion = float(row["baseline_exclusion"])
    return {
        "DistrictCVLearnedCSD": _fmt(float(row["learned_mean"])),
        "DistrictCVMESCSD": _fmt(float(row["baseline_mean"])),
        "DistrictCVDiff": _fmt(float(row["diff"])),
        "DistrictCVCI": (
            f"[{float(row['ci_lo']):+.4f}, {float(row['ci_hi']):+.4f}]"
        ),
        # Five fold-level sign flips have a coarse exact grid (minimum
        # two-sided p = 0.0625), so retain all four informative decimals.
        "DistrictCVP": f"= {float(row['fold_p_two_sided']):.4f}",
        "DistrictCVSeriesP": _p(float(row["p_two_sided"])),
        "DistrictCVWins": f"{row['wins']}/{row['n']}",
        "DistrictCVFoldWins": f"{row['fold_wins']}/{row['n_folds']}",
        "DistrictCVFoldMeanDiff": _fmt(float(row["fold_mean_diff"])),
        "DistrictCVTies": str(row["ties"]),
        "DistrictCVN": str(row["n"]),
        "DistrictCVWelfareRatio": f"{welfare_ratio:.3f}",
        "DistrictCVWelfareDeltaPct": f"{100 * (welfare_ratio - 1):+.1f}\\%",
        "DistrictCVWelfareCostPct": (
            f"{100 * max(0.0, 1.0 - welfare_ratio):.1f}\\%"
        ),
        "DistrictCVLearnedExcl": _fmt(learned_exclusion),
        "DistrictCVMESExcl": _fmt(baseline_exclusion),
        "DistrictCVExclDiff": f"{learned_exclusion - baseline_exclusion:+.4f}",
    }


def _cross_district_bound_macros(
    primary_rows: List[dict], sensitivity_rows: List[dict]
) -> Dict[str, str]:
    """Format held-out outcome stability under the post hoc wider weight box."""
    primary = next(
        (item for item in primary_rows if item.get("baseline") == "mes"), None
    )
    sensitivity = next(
        (item for item in sensitivity_rows if item.get("baseline") == "mes"), None
    )
    if primary is None or sensitivity is None:
        return {}
    for field in ("baseline_mean", "n", "n_folds"):
        if primary[field] != sensitivity[field]:
            raise ValueError(
                f"district bound sensitivity changed comparison field {field}: "
                f"{primary[field]} != {sensitivity[field]}"
            )
    welfare_ratio = float(sensitivity["welfare_ratio"])
    return {
        "DistrictCVWideLearnedCSD": _fmt(float(sensitivity["learned_mean"])),
        "DistrictCVWideDiff": _fmt(float(sensitivity["diff"])),
        "DistrictCVWideCI": (
            f"[{float(sensitivity['ci_lo']):+.4f}, "
            f"{float(sensitivity['ci_hi']):+.4f}]"
        ),
        "DistrictCVWideP": (
            f"= {float(sensitivity['fold_p_two_sided']):.4f}"
        ),
        "DistrictCVWideWins": f"{sensitivity['wins']}/{sensitivity['n']}",
        "DistrictCVWideFoldWins": (
            f"{sensitivity['fold_wins']}/{sensitivity['n_folds']}"
        ),
        "DistrictCVWideWelfareRatio": f"{welfare_ratio:.3f}",
        "DistrictCVWideWelfareCostPct": (
            f"{100 * max(0.0, 1.0 - welfare_ratio):.1f}\\%"
        ),
        "DistrictCVWideLearnedExcl": _fmt(
            float(sensitivity["learned_exclusion"])
        ),
        "DistrictCVWideCSDShift": (
            f"{float(sensitivity['learned_mean']) - float(primary['learned_mean']):+.4f}"
        ),
    }


def main() -> None:
    macros: Dict[str, str] = {}

    # ---- authenticated fresh-city support-set falsification ----
    external_bundle, external_diagnostics = _load_external_support_set_inputs()
    macros.update(
        _external_support_set_macros(
            external_bundle.decision,
            external_bundle.evidence,
            external_diagnostics,
        )
    )

    # ---- native-approval cross-city evaluation ----
    macros.update(_crosscity_macros())

    # ---- reviewer-requested matched-target, coverage, tilt, actuation checks ----
    macros.update(_reviewer_check_macros())

    # ---- post-lock static demographic audit and focal Warsaw multiplicity ----
    macros.update(_age_lookup_macros())
    macros.update(_contextual_age_lookup_macros())
    macros.update(_warsaw_holm_macros())

    # ---- frontier + test baselines (primary seed) ----
    front = _load_primary_frontier()
    base = front["baselines"]

    head = _exact_frontier_point(front, HEADLINE_FLOOR)
    aggr = _exact_frontier_point(front, AGGRESSIVE_FLOOR)
    corner = _exact_frontier_point(front, 0.0)

    for name, key in [
        ("Historical", "historical"),
        ("Greedy", "greedy-count"),
        ("GreedyCost", "greedy-cost"),
        ("MES", "mes"),
        ("RES", "res-1.0"),
    ]:
        s = base[f"{key}/test"]
        macros[f"Test{name}CSD"] = _fmt(s["worst_csd"])
        macros[f"Test{name}Welfare"] = f"{s['welfare']:,.0f}"
        macros[f"Test{name}Excl"] = _fmt(s["exclusion"], 4)

    hand = [base[f"{k}/test"]["worst_csd"] for k in ("greedy-cost", "mes", "res-0.5", "res-1.0")]
    macros["HandBandLo"] = _fmt(min(hand))
    macros["HandBandHi"] = _fmt(max(hand))
    macros["HandBandWidth"] = _fmt(max(hand) - min(hand))

    for label, pt in [("Head", head), ("Aggr", aggr), ("Corner", corner)]:
        macros[f"Learned{label}Floor"] = f"{pt['floor']:.2f}"
        macros[f"Learned{label}CSD"] = _fmt(pt["test_worst_csd"])
        macros[f"Learned{label}Welfare"] = f"{pt['test_welfare']:,.0f}"
        macros[f"Learned{label}Excl"] = _fmt(pt["test_exclusion"], 4)
        macros[f"Learned{label}TrainCSD"] = _fmt(pt["train_worst_csd"])
        macros[f"Learned{label}Gap"] = f"{pt['test_worst_csd'] - pt['train_worst_csd']:+.4f}"

    mes_csd = base["mes/test"]["worst_csd"]
    macros["HeadReductionPct"] = _pct((mes_csd - head["test_worst_csd"]) / mes_csd)
    macros["AggrReductionPct"] = _pct((mes_csd - aggr["test_worst_csd"]) / mes_csd)
    macros["AggrWelfareCostPct"] = _pct(
        1 - aggr["test_welfare"] / base["mes/test"]["welfare"]
    )

    # ---- exclusion ratios of the degenerate corner ----
    # These were prose ("three times") and were attached to the wrong baseline:
    # the deployed rule is greedy-by-count, not Equal Shares.
    gc_excl = base["greedy-count/test"]["exclusion"]
    mes_excl = base["mes/test"]["exclusion"]
    macros["CornerExclVsDeployed"] = f"{corner['test_exclusion'] / gc_excl:.1f}"
    macros["CornerExclVsMES"] = f"{corner['test_exclusion'] / mes_excl:.1f}"

    # ---- significance ----
    for row in _read_csv(RESULTS / "iclr_significance.csv"):
        if row["baseline"] != "mes":
            continue
        fl = float(row["floor"])
        tag = {HEADLINE_FLOOR: "Head", AGGRESSIVE_FLOOR: "Aggr", 1.02: "Tight", 1.05: "Tighter"}.get(fl)
        if tag is None:
            continue
        macros[f"Sig{tag}Diff"] = _fmt(float(row["diff"]))
        macros[f"Sig{tag}CI"] = f"[{float(row['ci_lo']):+.4f}, {float(row['ci_hi']):+.4f}]"
        macros[f"Sig{tag}P"] = _p(float(row["p_two_sided"]))
        macros[f"Sig{tag}Wins"] = f"{row['wins']}/{row['n']}"

    # ---- seed spread at the headline floor ----
    vals = []
    for d in _load_frontier_seed_sweep():
        pt = _exact_frontier_point(d, HEADLINE_FLOOR)
        vals.append(pt["test_worst_csd"])
    macros["NumSeeds"] = str(len(vals))
    macros["SeedSpread"] = _fmt(max(vals) - min(vals))
    macros["SeedMin"] = _fmt(min(vals))
    macros["SeedMax"] = _fmt(max(vals))

    # ---- learned endowment arm (mechanism retained) ----
    endow_path = RESULTS / "iclr_train" / "run_main_seed42.json"
    if endow_path.exists():
        e = json.loads(endow_path.read_text())
        t = e["learned"]["test"]
        macros["EndowCSD"] = _fmt(t["worst_csd"])
        macros["EndowWelfare"] = f"{t['welfare']:,.0f}"
        macros["EndowExcl"] = _fmt(t["exclusion"], 4)
        mes_t = e["baselines"]["mes/test"]
        macros["EndowVsMES"] = _fmt(t["worst_csd"] - mes_t["worst_csd"])
        macros["EndowReductionPct"] = _pct(
            (mes_t["worst_csd"] - t["worst_csd"]) / mes_t["worst_csd"]
        )
        macros["EndowWelfareRatio"] = f"{t['welfare'] / mes_t['welfare']:.3f}"
        macros["EndowWelfareCostPct"] = _pct(1 - t["welfare"] / mes_t["welfare"], 1)

    # ---- identification: is the support floor the operative constraint? ----
    # Impose Proposition 2's floor without the mechanism and with no welfare
    # penalty. If this still degenerates, the support floor is only a correlate.
    for row in _read_csv(RESULTS / "iclr_identification.csv"):
        kappa = float(row["kappa"])
        # LaTeX control sequences may not contain digits, so spell them out.
        tags = {1.0: "KOne", 2.0: "KTwo"}
        if kappa not in tags:
            raise ValueError(f"unexpected identification kappa: {kappa}")
        tag = tags[kappa]
        macros[f"Ident{tag}CSD"] = _fmt(float(row["test_worst_csd"]))
        macros[f"Ident{tag}Welfare"] = f"{float(row['test_welfare']):,.0f}"
        macros[f"Ident{tag}Excl"] = _fmt(float(row["test_exclusion"]), 4)

    # ---- how much of the damage the support floor alone repairs ----
    # "It still degenerates" is true against MES but hides that the floor does
    # most of the coverage repair and little of the welfare repair. Quantify.
    ident = {float(r["kappa"]): r for r in _read_csv(RESULTS / "iclr_identification.csv")}
    if 1.0 in ident:
        k1 = ident[1.0]
        hack_ex, mes_ex = corner["test_exclusion"], base["mes/test"]["exclusion"]
        hack_w, mes_w = corner["test_welfare"], base["mes/test"]["welfare"]
        got_ex, got_w = float(k1["test_exclusion"]), float(k1["test_welfare"])
        if hack_ex != mes_ex:
            macros["FloorClosesExclPct"] = _pct((hack_ex - got_ex) / (hack_ex - mes_ex))
        if hack_w != mes_w:
            macros["FloorClosesWelfarePct"] = _pct((got_w - hack_w) / (mes_w - hack_w))

    # ---- frozen-score allocation-kernel intervention ----
    # Hold the learned outcome weights and feature definitions fixed, then
    # insert either the theorem's one-shot support filter or iterative
    # approver-funded payment. The floor-0 point is the observed hacked corner.
    kernel_tags = {
        "direct": "Direct",
        "static-floor": "StaticFloor",
        "payment-only": "PaymentOnly",
        "payment+completion": "PaymentComplete",
    }
    kernel_rows = {
        row["kernel"]: row
        for row in _read_csv(RESULTS / "iclr_payment_intervention_summary.csv")
        if abs(float(row["floor"])) < 1e-12 and row.get("kernel") in kernel_tags
    }
    for kernel, row in kernel_rows.items():
        tag = kernel_tags[kernel]
        macros[f"Kernel{tag}CSD"] = _fmt(float(row["worst_csd"]))
        macros[f"Kernel{tag}Welfare"] = f"{float(row['welfare']):,.0f}"
        macros[f"Kernel{tag}Excl"] = _fmt(float(row["exclusion"]))
        macros[f"Kernel{tag}Util"] = _pct(float(row["budget_utilization"]), 1)
        macros[f"Kernel{tag}WelfareFactor"] = (
            f"{float(row['welfare_ratio_vs_direct']):.2f}"
        )
    complete = kernel_rows.get("payment+completion")
    if complete is not None:
        macros["KernelPaymentCompleteCSDDiff"] = _fmt(
            float(complete["diff_vs_direct"])
        )
        macros["KernelPaymentCompleteCSDCI"] = (
            f"[{float(complete['ci_lo']):+.4f}, {float(complete['ci_hi']):+.4f}]"
        )
        macros["KernelPaymentCompleteCSDP"] = _p(
            float(complete["p_two_sided"])
        )
        macros["KernelPaymentCompleteExclDiff"] = _fmt(
            float(complete["exclusion_diff_vs_direct"])
        )
        macros["KernelPaymentCompleteExclCI"] = (
            f"[{float(complete['exclusion_ci_lo']):+.4f}, "
            f"{float(complete['exclusion_ci_hi']):+.4f}]"
        )
        macros["KernelPaymentCompleteExclP"] = _p(
            float(complete["exclusion_p_two_sided"])
        )
        macros["KernelPaymentCompleteExclWins"] = (
            f"{complete['exclusion_wins_vs_direct']}/{complete['n']}"
        )
        direct = kernel_rows.get("direct")
        if direct is not None and float(direct["exclusion"]) > 0:
            macros["KernelPaymentCompleteExclReductionPct"] = _pct(
                1.0 - float(complete["exclusion"]) / float(direct["exclusion"]),
                1,
            )

    # ---- static-tilt control: does memory contribute anything? ----
    for row in _read_csv(RESULTS / "iclr_static_tilt.csv"):
        if row["policy"].startswith("static"):
            macros["StaticTiltCSD"] = _fmt(float(row["test_worst_csd"]))
            macros["StaticTiltWelfare"] = f"{float(row['test_welfare']):,.0f}"
            macros["StaticTiltExcl"] = _fmt(float(row["test_exclusion"]), 4)

    # ---- realized kappa of the fitted endowment map ----
    theory = _read_csv(RESULTS / "iclr_theory_check.csv")
    ks = [
        float(r["kappa"]) for r in theory
        if r.get("policy") == "learned-endowment" and r.get("kappa") not in ("", "nan")
    ]
    if ks:
        macros["KappaEndowMean"] = f"{sum(ks) / len(ks):.2f}"
        macros["KappaEndowMax"] = f"{max(ks):.2f}"

    # ---- endowment seed replication ----
    macros.update(_endowment_seed_summary_macros())

    # ---- closest published learned PB rules, replayed without refitting ----
    for row in _read_csv(RESULTS / "iclr_llmrule_baseline.csv"):
        if row.get("series") != "__aggregate__" or row.get("view") != "test":
            continue
        tag = {
            "llmrule-cost": "LLMRuleCost",
            "llmrule-card": "LLMRuleCard",
        }.get(row.get("rule", ""))
        if tag is None:
            continue
        macros[f"{tag}CSD"] = _fmt(float(row["worst_csd"]))
        macros[f"{tag}Welfare"] = f"{float(row['welfare']):,.0f}"
        macros[f"{tag}CostWelfare"] = f"{float(row['cost_welfare']):,.0f}"
        macros[f"{tag}Excl"] = _fmt(float(row["exclusion"]), 4)

    # ---- who the objective actually favours (ethics audit) ----
    mix_path = RESULTS / "iclr_worst_cohort_mix.json"
    if mix_path.exists():
        macros.update(json.loads(mix_path.read_text()))

    # ---- appendix: split sizes and optimizer provenance ----
    split_path = RESULTS / "iclr_splits" / "temporal_2022.json"
    if split_path.exists():
        sp = json.loads(split_path.read_text())
        macros["SplitTrainEditions"] = str(sum(len(y) for _, y in sp["train"]))
        macros["SplitTestEditions"] = str(sum(len(y) for _, y in sp["test"]))
    cmaes = ROOT / "src" / "iclr_cmaes.py"
    if cmaes.exists():
        code = [
            l for l in cmaes.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        macros["OptimizerLines"] = str(len(code))

    # ---- recorded CPU wall-clock ranges for the frozen primary fits ----
    macros.update(_runtime_macros())

    # ---- representative-support degeneracy theorem ----
    degeneracy = _read_csv(RESULTS / "iclr_degeneracy.csv")
    if degeneracy:
        largest = max(degeneracy, key=lambda row: int(row["voters"]))
        macros["DegGroups"] = largest["groups"]
        macros["DegSupporters"] = largest["supporters"]
        macros["DegVoters"] = largest["voters"]
        macros["DegCSD"] = _fmt(float(largest["worst_abs_csd"]))
        macros["DegExcl"] = _pct(float(largest["exclusion"]), 1)
        row_tags = {80: "Eighty", 200: "TwoHundred", 400: "FourHundred", 800: "EightHundred"}
        for row in degeneracy:
            voters = int(row["voters"])
            tag = row_tags.get(voters)
            if tag is None:
                continue
            macros[f"Deg{tag}Voters"] = str(voters)
            macros[f"Deg{tag}CSD"] = _fmt(float(row["worst_abs_csd"]))
            macros[f"Deg{tag}Excl"] = _pct(float(row["exclusion"]), 1)

    # ---- measured frontier-range comparison between the two arms ----
    # This is an empirical result for the fitted family and search budget, not
    # a universal reachable-set consequence of the support-floor proposition.
    for tag, pattern in (
        ("Endow", "frontier_endowment_temporal_2022_seed42.csv"),
        ("Outcome", "frontier_outcome_temporal_2022_seed42.csv"),
    ):
        rows_f = _read_csv(RESULTS / "iclr_frontier" / pattern)
        vals = [float(r["test_worst_csd"]) for r in rows_f if r.get("test_worst_csd")]
        if vals:
            macros[f"Range{tag}Lo"] = _fmt(min(vals))
            macros[f"Range{tag}Hi"] = _fmt(max(vals))
            macros[f"Range{tag}Width"] = _fmt(max(vals) - min(vals))

    # ---- endowment-arm significance ----
    sig_tags = {
        "learned-endowment vs mes": "MES",
        "learned-endowment vs res-1.0": "RES",
        "learned-endowment vs llmrule-cost": "LLMRuleCost",
        "learned-endowment vs llmrule-card": "LLMRuleCard",
        "learned-endowment vs learned-outcome": "TwoArm",
    }
    for row in _read_csv(RESULTS / "iclr_endow_significance.csv"):
        tag = sig_tags.get(row["contrast"])
        if tag is None:
            logger.warning("unmapped endowment contrast: %s", row["contrast"])
            continue
        macros[f"EndowSig{tag}Diff"] = _fmt(float(row["diff"]))
        macros[f"EndowSig{tag}CI"] = (
            f"[{float(row['ci_lo']):+.4f}, {float(row['ci_hi']):+.4f}]"
        )
        macros[f"EndowSig{tag}P"] = _p(float(row["p_two_sided"]))
        macros[f"EndowSig{tag}Wins"] = f"{row['wins']}/{row['n']}"

    # ---- allocation composition (few-and-large, not unpopular) ----
    comp_map = {
        "mes": "MES", "learned-endowment": "Endow", "greedy-cost": "GreedyCost",
        "llmrule-cost": "LLMRuleCost", "llmrule-card": "LLMRuleCard",
        "outcome-f1.00": "OutcomeHead", "outcome-f0.00": "OutcomeCorner",
    }
    for row in _read_csv(RESULTS / "iclr_composition.csv"):
        tag = comp_map.get(row["policy"])
        if not tag:
            continue
        macros[f"Comp{tag}Funded"] = f"{float(row['mean_projects_funded']):.1f}"
        macros[f"Comp{tag}Cost"] = _fmt(float(row["mean_cost_share"]))
        macros[f"Comp{tag}Conc"] = _fmt(float(row["mean_concentration"]))
        macros[f"Comp{tag}RepTV"] = _fmt(
            float(row["cost_weighted_representation_tv"])
        )

    # ---- transfer across demographic partitions ----
    # These values used to be typed directly into an appendix table. Generate
    # every cell, including the exact paired-test p-value, from
    # the same CSV that supports the prose.
    scheme_rows = _read_csv(RESULTS / "iclr_scheme_robustness.csv")
    scheme_tags = {"age_sex": "AgeSex", "age": "Age", "sex": "Sex"}
    policy_tags = {
        "mes": "MES",
        "res-1.0": "RES",
        "learned-endowment": "Endow",
        "learned-outcome": "Outcome",
    }
    learned_ps: List[float] = []
    for row in scheme_rows:
        scheme_tag = scheme_tags.get(row.get("scheme", ""))
        policy_tag = policy_tags.get(row.get("policy", ""))
        if scheme_tag is None or policy_tag is None:
            continue
        macros[f"Part{scheme_tag}Groups"] = row["n_groups"]
        macros[f"Part{scheme_tag}{policy_tag}CSD"] = _fmt(
            float(row["worst_csd"])
        )
        if row.get("vs_mes_p") not in (None, ""):
            p = float(row["vs_mes_p"])
            learned_ps.append(p)
            macros[f"Part{scheme_tag}{policy_tag}P"] = _p(p)
    if learned_ps:
        macros["PartMaxLearnedP"] = _fmt(max(learned_ps), 4)

    # ---- ablation (matched optimization budget) ----
    ab_map = {
        "none (full model)": "Full", "approval_share": "ApprShare",
        "approvals_per_cost": "PerCost", "cost_share": "CostShare",
        "deficit_weighted": "Deficit", "cohort_concentration": "Conc",
    }
    for row in _read_csv(RESULTS / "iclr_ablation.csv"):
        tag = ab_map.get(row["ablated"])
        if not tag:
            continue
        macros[f"Abl{tag}CSD"] = _fmt(float(row["test_worst_csd"]))
        macros[f"Abl{tag}Delta"] = f"{float(row['delta_vs_full']):+.4f}"

    # ---- theory: budget spent above the kappa=1 payment-floor reference ----
    name_map = {
        "mes": "MES", "res-1.0": "RES", "learned-endowment": "LearnedEndow",
        "greedy-cost": "GreedyCost", "learned-outcome-f1.0": "OutcomeHead",
        "learned-outcome-f0.0": "OutcomeCorner",
    }
    for row in _read_csv(RESULTS / "iclr_infeasible_spend.csv"):
        tag = name_map.get(row["policy"])
        if tag:
            macros[f"Infeas{tag}Budget"] = f"{float(row['pct_budget']):.2f}\\%"
            macros[f"Infeas{tag}Proj"] = f"{float(row['pct_projects']):.2f}\\%"

    # ---- cross-city transfer (Łódź) ----
    tmap = {
        "historical": "Hist", "greedy-cost": "GreedyCost", "mes": "MES",
        "learned-endowment": "Endow", "learned-outcome-f1.00": "Outcome",
    }
    for row in _read_csv(RESULTS / "iclr_transfer.csv"):
        tag = tmap.get(row["policy"])
        if tag and row["worst_csd"]:
            macros[f"Transfer{tag}CSD"] = _fmt(float(row["worst_csd"]))
            macros[f"Transfer{tag}Welfare"] = f"{float(row['welfare']):,.0f}"

    # ---- all eligible shorter non-Warsaw series, frozen policies/no refit ----
    # This is a descriptive stress test. The city rows are kept visible so an
    # improved pooled mean cannot hide exact ties in Gdynia and Lublin.
    macros.update(
        _multicity_macros(_read_csv(RESULTS / "iclr_multicity_transfer.csv"))
    )

    # ---- disjoint-series transfer: each series scored out of its training fold ----
    macros.update(
        _cross_district_macros(
            _read_csv(RESULTS / "iclr_cross_district_summary.csv")
        )
    )
    macros.update(
        _cross_district_bound_macros(
            _read_csv(RESULTS / "iclr_cross_district_summary.csv"),
            _read_csv(RESULTS / "iclr_cross_district_b40_summary.csv"),
        )
    )

    # ---- city composition of the EVALUATED split ----
    # Lodz has no pre-2023 editions, so it is absent from temporal_2022
    # entirely. Stating "17 of 18" here would understate the limitation.
    try:
        from iclr_corpus import build_series_index, load_split
        from parse_pb import parse_pb_file

        idx = build_series_index()
        evaluated = [k for k, _ in load_split("temporal_2022").test]
        cities = {idx[k].city.split("/")[-1] for k in evaluated}
        macros["EvalSeriesN"] = str(len(evaluated))
        macros["EvalCityN"] = str(len(cities))
        macros["EvalCityList"] = ", ".join(sorted(cities))
        macros["DynamicsStartYear"] = str(
            max(min(idx[key].years) for key in evaluated)
        )
        frozen_corpus = [
            raw.split("\t")
            for raw in (RESULTS / "frozen_instances.txt").read_text().splitlines()
            if raw and not raw.startswith("#")
        ]
        corpus_instances = [
            parse_pb_file(ROOT / "data" / "pb" / filename)
            for filename, _series, _year in frozen_corpus
        ]
        corpus_years = [int(year) for _filename, _series, year in frozen_corpus]
        macros["PrimaryElectionN"] = str(len(corpus_instances))
        macros["PrimaryBallotN"] = f"{sum(len(inst.votes) for inst in corpus_instances):,}"
        macros["PrimaryProjectN"] = f"{sum(len(inst.projects) for inst in corpus_instances):,}"
        macros["PrimaryYearLo"] = str(min(corpus_years))
        macros["PrimaryYearHi"] = str(max(corpus_years))
    except Exception as exc:  # noqa: BLE001 - macro is optional
        logger.warning("could not derive evaluated-split composition: %s", exc)

    # ---- whole-corpus baselines ----
    rows = _read_csv(RESULTS / "iclr_baselines.csv")
    if rows:
        macros["NumSeriesTotal"] = str(len(rows))

    # ---- per-series wins / failure cases ----
    per = _read_csv(RESULTS / "iclr_per_series.csv")
    if per:
        losses = [r for r in per if r.get("learned_wins") == "0"]
        macros["PerSeriesN"] = str(len(per))
        macros["PerSeriesWins"] = str(len(per) - len(losses))
        macros["PerSeriesLosses"] = str(len(losses))
        macros["LossSeriesList"] = ", ".join(
            r["series"].split("/")[-1] for r in losses
        ) or "none"

    # ---- ablation ----
    ab = _read_csv(RESULTS / "iclr_ablation.csv")
    if ab:
        full = next((r for r in ab if r["ablated"].startswith("none")), None)
        worst = max(
            (r for r in ab if not r["ablated"].startswith("none")),
            key=lambda r: float(r["delta_vs_full"]),
            default=None,
        )
        if full and worst:
            macros["AblationFullCSD"] = _fmt(float(full["test_worst_csd"]))
            macros["AblationWorstFeature"] = worst["ablated"].replace("_", "\\_")
            macros["AblationWorstDelta"] = f"{float(worst['delta_vs_full']):+.4f}"

    # Signed deltas read backwards in prose: "improves fairness by -0.0095" is
    # confusing when lower is better. Emit magnitude twins for use in sentences.
    for key in [k for k in list(macros) if k.endswith(("Diff", "Delta"))]:
        try:
            macros[key + "Abs"] = f"{abs(float(macros[key])):.4f}"
        except ValueError:
            pass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% AUTO-GENERATED by tempopb/src/gen_iclr_numbers.py -- do not edit by hand",
        f"% source frontier: {PRIMARY_FRONTIER.name}",
        "",
    ]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    OUT.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s (%d macros)", OUT, len(macros))
    for k in sorted(macros):
        logger.info("  \\%-24s %s", k, macros[k])


if __name__ == "__main__":
    main()
