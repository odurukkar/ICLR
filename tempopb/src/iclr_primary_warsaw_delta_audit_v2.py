"""Authenticate and compare the protected primary-Warsaw v1/v2 lineages.

The corrected lineage changes both approval-set normalization and deterministic
project ordering.  Consequently this audit reports observed deltas, but never
labels them as the causal effect of duplicate removal.  The audit is read-only
except for two append-only outputs: the row CSV is installed first and the
hash-sealing JSON is installed last.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import iclr_primary_warsaw_protocol_v2 as protocol_v2
import iclr_primary_warsaw_train_v2 as train_v2
import iclr_primary_warsaw_evaluate_v2 as evaluate_v2


ROOT = Path(__file__).resolve().parent.parent
V2_RESULT_RELATIVE = Path("results/iclr_primary_warsaw_v2")
JSON_OUTPUT = Path("v1_v2_delta_audit.json")
CSV_OUTPUT = Path("v1_v2_delta_rows.csv")

PROTECTED_PDF_RELATIVE = Path("output/pdf/ICLR2027_TempoPB_anonymous.pdf")
PROTECTED_REPRO_RELATIVE = Path(
    "output/artifact/ICLR2027_TempoPB_reproducibility.zip"
)
PROTECTED_SOURCE_RELATIVE = Path("output/artifact/ICLR2027_TempoPB_source.zip")
PROTECTED_PDF_SHA256 = (
    "7fed4760ea5b75205590d7eb47c2f3084d2f3efe0c39b90d0fbb4d8c13f3e88a"
)
PROTECTED_REPRO_SHA256 = (
    "4adfa80db1467d8b7b4fd314699b3fbf69cc9057a640ef8816611e86500b3d20"
)
PROTECTED_SOURCE_SHA256 = (
    "6608114d3ff0a9166264f1ff88fbb6f76813f1b649ef4840306ed2d7ad174251"
)
PROTECTED_NUMBERS_SHA256 = (
    "67d19404f169a5a53d3f714171297f2431ce8a7922cac2d80cf8ea06b4013bb9"
)
PROTECTED_NUMBERS_MEMBER = "numbers.tex"

METRIC_FIELDS = (
    "worst_csd",
    "mean_csd",
    "welfare",
    "cost_welfare",
    "exclusion",
)
EPISODE_REQUIRED_FIELDS = frozenset(
    {
        "fit_id",
        "split",
        "series",
        "scored_years",
        "worst_cohort",
        "worst_csd",
        "mean_csd",
        "welfare",
        "cost_welfare",
        "exclusion",
        "scored_outcomes",
    }
)
DELTA_CSV_FIELDS = (
    "fit_id",
    "split",
    "series",
    "winner_exact",
    "worst_cohort_exact",
    "worst_csd_delta_v2_minus_v1",
    "mean_csd_delta_v2_minus_v1",
    "welfare_delta_v2_minus_v1",
    "cost_welfare_delta_v2_minus_v1",
    "exclusion_delta_v2_minus_v1",
    "v1_scored_outcomes_sha256",
    "v2_scored_outcomes_sha256",
)
METRIC_DELTA_CSV_FIELDS = (
    "fit_id",
    "family",
    "split",
    "metric",
    "status",
    "v1_value",
    "v2_value",
    "delta_v2_minus_v1",
    "changed",
)
CAUSAL_ATTRIBUTION_REASON = (
    "The corrected lineage changes both approval-set normalization and "
    "deterministic project ordering; observed deltas cannot be assigned "
    "to duplicate removal alone."
)

NON_PRIMARY_MACRO_PREFIXES = ("External", "CrossCity", "Multi", "Actuating")

# These are semantic table interfaces, not permissive header intersections.  A
# protected table must have the exact legacy fields below, and every listed
# numeric field must exist at the same typed v2 coordinate.  V2-only provenance
# fields (for example ``source_id``) are intentionally not projected into v1.
CONSUMER_CSV_COMPARISON_SPECS: dict[str, tuple[dict[str, object], ...]] = {
    "baseline.primary": (
        {
            "source_member": "results/iclr_baselines.csv",
            "table": "series_rows",
            "legacy_fields": (
                "series",
                "greedy-count_worst", "greedy-count_welfare",
                "greedy-cost_worst", "greedy-cost_welfare",
                "llmrule-cost_worst", "llmrule-cost_welfare",
                "llmrule-card_worst", "llmrule-card_welfare",
                "mes_worst", "mes_welfare",
                "res-0.25_worst", "res-0.25_welfare",
                "res-0.5_worst", "res-0.5_welfare",
                "res-0.75_worst", "res-0.75_welfare",
                "res-1.0_worst", "res-1.0_welfare",
            ),
            "identity_fields": ("series",),
            "numeric_fields": (
                "greedy-count_worst", "greedy-count_welfare",
                "greedy-cost_worst", "greedy-cost_welfare",
                "llmrule-cost_worst", "llmrule-cost_welfare",
                "llmrule-card_worst", "llmrule-card_welfare",
                "mes_worst", "mes_welfare",
                "res-0.25_worst", "res-0.25_welfare",
                "res-0.5_worst", "res-0.5_welfare",
                "res-0.75_worst", "res-0.75_welfare",
                "res-1.0_worst", "res-1.0_welfare",
            ),
        },
    ),
    "baseline.prior_rules": (
        {
            "source_member": "results/iclr_llmrule_baseline.csv",
            "table": "rule_view_rows",
            "legacy_fields": (
                "rule", "series", "view", "worst_csd", "mean_csd",
                "welfare", "cost_welfare", "exclusion", "n_series",
                "source_doi", "score",
            ),
            "identity_fields": ("rule", "series", "view"),
            "numeric_fields": (
                "worst_csd", "mean_csd", "welfare", "cost_welfare",
                "exclusion", "n_series",
            ),
        },
    ),
    "control.history_free": (
        {
            "source_member": "results/iclr_static_tilt.csv",
            "table": "policy_rows",
            "legacy_fields": (
                "policy", "test_worst_csd", "test_welfare", "test_exclusion",
            ),
            "identity_fields": ("policy",),
            "numeric_fields": (
                "test_worst_csd", "test_welfare", "test_exclusion",
            ),
        },
    ),
    "cross_district.bound10": (
        {
            "source_member": "results/iclr_cross_district_per_series.csv",
            "table": "series_policy_rows",
            "legacy_fields": (
                "fold", "split", "series", "policy", "n_years", "worst_csd",
                "mean_csd", "welfare", "cost_welfare", "exclusion",
            ),
            "identity_fields": ("fold", "split", "series", "policy"),
            "numeric_fields": (
                "n_years", "worst_csd", "mean_csd", "welfare",
                "cost_welfare", "exclusion",
            ),
        },
        {
            "source_member": "results/iclr_cross_district_summary.csv",
            "table": "contrast_rows",
            "legacy_fields": (
                "baseline", "learned_mean", "baseline_mean", "diff", "ci_lo",
                "ci_hi", "p_two_sided", "wins", "ties", "n",
                "fold_mean_diff", "fold_p_two_sided", "fold_wins", "n_folds",
                "learned_welfare", "baseline_welfare", "welfare_ratio",
                "learned_exclusion", "baseline_exclusion",
            ),
            "identity_fields": ("baseline",),
            "numeric_fields": (
                "learned_mean", "baseline_mean", "diff", "ci_lo", "ci_hi",
                "p_two_sided", "wins", "ties", "n", "fold_mean_diff",
                "fold_p_two_sided", "fold_wins", "n_folds", "learned_welfare",
                "baseline_welfare", "welfare_ratio", "learned_exclusion",
                "baseline_exclusion",
            ),
        },
    ),
    "cross_district.bound40": (
        {
            "source_member": "results/iclr_cross_district_b40_per_series.csv",
            "table": "series_policy_rows",
            "legacy_fields": (
                "fold", "split", "series", "policy", "n_years", "worst_csd",
                "mean_csd", "welfare", "cost_welfare", "exclusion",
            ),
            "identity_fields": ("fold", "split", "series", "policy"),
            "numeric_fields": (
                "n_years", "worst_csd", "mean_csd", "welfare",
                "cost_welfare", "exclusion",
            ),
        },
        {
            "source_member": "results/iclr_cross_district_b40_summary.csv",
            "table": "contrast_rows",
            "legacy_fields": (
                "baseline", "learned_mean", "baseline_mean", "diff", "ci_lo",
                "ci_hi", "p_two_sided", "wins", "ties", "n",
                "fold_mean_diff", "fold_p_two_sided", "fold_wins", "n_folds",
                "learned_welfare", "baseline_welfare", "welfare_ratio",
                "learned_exclusion", "baseline_exclusion",
            ),
            "identity_fields": ("baseline",),
            "numeric_fields": (
                "learned_mean", "baseline_mean", "diff", "ci_lo", "ci_hi",
                "p_two_sided", "wins", "ties", "n", "fold_mean_diff",
                "fold_p_two_sided", "fold_wins", "n_folds", "learned_welfare",
                "baseline_welfare", "welfare_ratio", "learned_exclusion",
                "baseline_exclusion",
            ),
        },
    ),
    "failure.outcome": (
        {
            "source_member": "results/iclr_failure_analysis.csv",
            "table": "series_diagnostics",
            "legacy_fields": (
                "series", "learned_minus_mes", "loses", "worst_cohort",
                "is_senior_worst", "mes_csd", "cohort_divergence",
                "ballot_len_dispersion", "mean_voters", "n_test_years",
            ),
            "identity_fields": ("series",),
            "numeric_fields": (
                "learned_minus_mes", "loses", "is_senior_worst", "mes_csd",
                "cohort_divergence", "ballot_len_dispersion", "mean_voters",
                "n_test_years",
            ),
        },
    ),
    "frontier.endowment": (
        {
            "source_member": "results/iclr_frontier/frontier_endowment_temporal_2022_seed42.csv",
            "table": "point_rows",
            "legacy_fields": (
                "arm", "floor", "penalty", "train_worst_csd",
                "train_welfare_ratio", "test_worst_csd", "test_mean_csd",
                "test_welfare", "test_exclusion", "elapsed_sec",
            ),
            "identity_fields": ("seed", "floor"),
            "numeric_fields": (
                "penalty", "train_worst_csd", "train_welfare_ratio",
                "test_worst_csd", "test_mean_csd", "test_welfare",
                "test_exclusion",
            ),
            "context": {"seed": 42},
        },
    ),
    "frontier.outcome": tuple(
        {
            "source_member": (
                f"results/iclr_frontier/"
                f"frontier_outcome_temporal_2022_seed{seed}.csv"
            ),
            "table": "point_rows",
            "legacy_fields": (
                "arm", "floor", "penalty", "train_worst_csd",
                "train_welfare_ratio", "test_worst_csd", "test_mean_csd",
                "test_welfare", "test_exclusion", "elapsed_sec",
            ),
            "identity_fields": ("seed", "floor"),
            "numeric_fields": (
                "penalty", "train_worst_csd", "train_welfare_ratio",
                "test_worst_csd", "test_mean_csd", "test_welfare",
                "test_exclusion",
            ),
            "context": {"seed": seed},
        }
        for seed in (1, 2, 3, 42)
    ),
    "mechanism.attribution_coverage": (
        {
            "source_member": "results/iclr_attribution_coverage.csv",
            "table": "policy_rows",
            "legacy_fields": (
                "policy", "scored_editions",
                "editions_with_uncovered_funded_project", "funded_projects",
                "uncovered_funded_projects", "uncovered_budget_share",
            ),
            "identity_fields": ("policy",),
            "numeric_fields": (
                "scored_editions", "editions_with_uncovered_funded_project",
                "funded_projects", "uncovered_funded_projects",
                "uncovered_budget_share",
            ),
            "legacy_exclude": {
                "policy": "series_clean_for_mes_and_learned_endowment"
            },
        },
    ),
    "mechanism.payment_kernel": (
        {
            "source_member": "results/iclr_payment_intervention_per_series.csv",
            "table": "series_kernel_rows",
            "legacy_fields": (
                "floor", "kernel", "series", "n_years", "worst_csd",
                "mean_csd", "welfare", "exclusion", "spent",
            ),
            "identity_fields": ("floor", "kernel", "series"),
            "numeric_fields": (
                "n_years", "worst_csd", "mean_csd", "welfare", "exclusion",
                "spent",
            ),
        },
        {
            "source_member": "results/iclr_payment_intervention_summary.csv",
            "table": "kernel_summary_rows",
            "legacy_fields": (
                "floor", "kernel", "worst_csd", "mean_csd", "welfare",
                "exclusion", "spent", "budget_utilization",
                "welfare_ratio_vs_direct", "diff_vs_direct", "ci_lo", "ci_hi",
                "p_two_sided", "wins_vs_direct", "n",
                "exclusion_diff_vs_direct", "exclusion_ci_lo", "exclusion_ci_hi",
                "exclusion_p_two_sided", "exclusion_wins_vs_direct",
            ),
            "identity_fields": ("floor", "kernel"),
            "numeric_fields": (
                "worst_csd", "mean_csd", "welfare", "exclusion", "spent",
                "budget_utilization", "welfare_ratio_vs_direct", "diff_vs_direct",
                "ci_lo", "ci_hi", "p_two_sided", "wins_vs_direct", "n",
                "exclusion_diff_vs_direct", "exclusion_ci_lo", "exclusion_ci_hi",
                "exclusion_p_two_sided", "exclusion_wins_vs_direct",
            ),
        },
    ),
    "mechanism.support_floor": (
        {
            "source_member": "results/iclr_identification.csv",
            "table": "identification_rows",
            "legacy_fields": (
                "kappa", "welfare_floor", "train_worst_csd", "test_worst_csd",
                "test_welfare", "test_exclusion", "welfare_ratio_vs_mes",
            ),
            "identity_fields": ("kappa",),
            "numeric_fields": (
                "train_worst_csd", "test_worst_csd", "test_welfare",
                "test_exclusion", "welfare_ratio_vs_mes",
            ),
        },
    ),
    "mechanism.theory_floor": (
        {
            "source_member": "results/iclr_theory_check.csv",
            "table": "theory_rows",
            "legacy_fields": (
                "series", "year", "policy", "kappa", "min_approval_share",
                "implied_floor", "violates_floor",
            ),
            "identity_fields": ("series", "year", "policy"),
            "numeric_fields": (
                "kappa", "min_approval_share", "implied_floor", "violates_floor",
            ),
        },
        {
            "source_member": "results/iclr_infeasible_spend.csv",
            "table": "infeasible_spend_rows",
            "legacy_fields": ("policy", "pct_projects", "pct_budget"),
            "identity_fields": ("policy",),
            "numeric_fields": ("pct_projects", "pct_budget"),
        },
    ),
    "outcome.ablation": (
        {
            "source_member": "results/iclr_ablation.csv",
            "table": "ablation_rows",
            "legacy_fields": (
                "ablated",
                "test_worst_csd",
                "test_welfare",
                "delta_vs_full",
            ),
            "identity_fields": ("ablated",),
            "numeric_fields": (
                "test_worst_csd",
                "test_welfare",
                "delta_vs_full",
            ),
        },
    ),
    "outcome.per_series": (
        {
            "source_member": "results/iclr_per_series.csv",
            "table": "learned_rows",
            "legacy_fields": (
                "series", "test_years", "greedy-count", "greedy-cost", "mes",
                "res-1.0", "learned", "worst_cohort", "learned_minus_mes",
                "learned_wins",
            ),
            "identity_fields": ("series",),
            "numeric_fields": (
                "greedy-count", "greedy-cost", "mes", "res-1.0", "learned",
                "learned_minus_mes", "learned_wins",
            ),
        },
        {
            "source_member": "results/iclr_two_arm_per_series.csv",
            "table": "two_arm_rows",
            "legacy_fields": (
                "series", "mes", "learned_endowment", "learned_outcome",
                "endowment_minus_mes", "outcome_minus_mes", "endowment_wins",
                "outcome_wins",
            ),
            "identity_fields": ("series",),
            "numeric_fields": (
                "mes", "learned_endowment", "learned_outcome",
                "endowment_minus_mes", "outcome_minus_mes", "endowment_wins",
                "outcome_wins",
            ),
        },
    ),
    "outcome.significance": (
        {
            "source_member": "results/iclr_significance.csv",
            "table": "contrast_rows",
            "legacy_fields": (
                "arm", "floor", "baseline", "learned_mean", "baseline_mean",
                "diff", "ci_lo", "ci_hi", "p_two_sided", "wins", "n",
            ),
            "identity_fields": ("arm", "floor", "baseline"),
            "numeric_fields": (
                "learned_mean", "baseline_mean", "diff", "ci_lo", "ci_hi",
                "p_two_sided", "wins", "n",
            ),
        },
    ),
    "primary.corpus": (
        {
            "source_member": "results/iclr_instance_granularity.csv",
            "table": "granularity_rows",
            "legacy_fields": (
                "corpus", "scored_elections", "median_projects", "mean_projects",
            ),
            "identity_fields": ("corpus",),
            "numeric_fields": (
                "scored_elections", "median_projects", "mean_projects",
            ),
            "legacy_include": {"corpus": "warsaw_fitting"},
        },
    ),
    "robustness.demographic_partition": (
        {
            "source_member": "results/iclr_scheme_robustness.csv",
            "table": "scheme_rows",
            "legacy_fields": (
                "scheme", "n_groups", "policy", "worst_csd", "vs_mes_diff",
                "vs_mes_p", "vs_mes_wins", "n",
            ),
            "identity_fields": ("scheme", "policy"),
            "numeric_fields": (
                "n_groups", "worst_csd", "vs_mes_diff", "vs_mes_p",
                "vs_mes_wins", "n",
            ),
        },
    ),
    "summary.composition": (
        {
            "source_member": "results/iclr_composition.csv",
            "table": "composition_rows",
            "legacy_fields": (
                "policy", "mean_projects_funded", "mean_cost_share",
                "mean_concentration", "cost_weighted_representation_tv",
            ),
            "identity_fields": ("policy",),
            "numeric_fields": (
                "mean_projects_funded", "mean_cost_share", "mean_concentration",
                "cost_weighted_representation_tv",
            ),
        },
    ),
    "summary.endowment_significance": (
        {
            "source_member": "results/iclr_endow_significance.csv",
            "table": "contrast_rows",
            "legacy_fields": (
                "contrast", "learned_mean", "baseline_mean", "diff", "ci_lo",
                "ci_hi", "p_two_sided", "wins", "n",
            ),
            "identity_fields": ("contrast",),
            "numeric_fields": (
                "learned_mean", "baseline_mean", "diff", "ci_lo", "ci_hi",
                "p_two_sided", "wins", "n",
            ),
        },
        {
            "source_member": "results/iclr_two_arm_per_series.csv",
            "table": "two_arm_rows",
            "legacy_fields": (
                "series", "mes", "learned_endowment", "learned_outcome",
                "endowment_minus_mes", "outcome_minus_mes", "endowment_wins",
                "outcome_wins",
            ),
            "identity_fields": ("series",),
            "numeric_fields": (
                "mes", "learned_endowment", "learned_outcome",
                "endowment_minus_mes", "outcome_minus_mes", "endowment_wins",
                "outcome_wins",
            ),
        },
        {
            "source_member": "results/iclr_composition.csv",
            "table": "composition_rows",
            "legacy_fields": (
                "policy", "mean_projects_funded", "mean_cost_share",
                "mean_concentration", "cost_weighted_representation_tv",
            ),
            "identity_fields": ("policy",),
            "numeric_fields": (
                "mean_projects_funded", "mean_cost_share", "mean_concentration",
                "cost_weighted_representation_tv",
            ),
        },
    ),
    "summary.matched_target": (
        {
            "source_member": "results/iclr_matched_target.csv",
            "table": "matched_rows",
            "legacy_fields": (
                "soft_target", "n_series", "endowment_csd", "direct_csd",
                "difference", "ci_low", "ci_high", "p_value", "endowment_wins",
                "endowment_welfare", "direct_welfare", "endowment_exclusion",
                "direct_exclusion",
            ),
            "identity_fields": ("soft_target",),
            "numeric_fields": (
                "n_series", "endowment_csd", "direct_csd", "difference",
                "ci_low", "ci_high", "p_value", "endowment_wins",
                "endowment_welfare", "direct_welfare", "endowment_exclusion",
                "direct_exclusion",
            ),
        },
    ),
    "transfer.lodz": (
        {
            "source_member": "results/iclr_transfer.csv",
            "table": "transfer_rows",
            "legacy_fields": (
                "series",
                "policy",
                "worst_csd",
                "welfare",
                "exclusion",
                "worst_cohort",
            ),
            "identity_fields": ("series", "policy"),
            "numeric_fields": ("worst_csd", "welfare", "exclusion"),
        },
    ),
}

CONSUMER_SPECIAL_COMPARISON_PATHS: dict[str, tuple[str, ...]] = {
    "control.history_free": (
        "analysis-output/ml-contribution-audit/history_free_seed42_g30.json",
    ),
    "control.senior_scalar": ("results/iclr_senior_tilt.csv",),
    "control.static_age_lookup": (
        "analysis-output/ml-contribution-audit/static_age_lookup_seed42_g30.json",
    ),
    "cross_district.bound10": ("results/iclr_cross_district_manifest.json",),
    "cross_district.bound40": ("results/iclr_cross_district_b40_manifest.json",),
    "failure.outcome": ("results/iclr_worst_cohort_mix.json",),
    "frontier.endowment": (
        "results/iclr_frontier/frontier_endowment_temporal_2022_seed42.json",
    ),
    "frontier.outcome": tuple(
        f"results/iclr_frontier/frontier_outcome_temporal_2022_seed{seed}.json"
        for seed in (1, 2, 3, 42)
    ),
    # This path is also a four-row CSV table; the special comparison covers its
    # fifth, joint-clean-series record.
    "mechanism.attribution_coverage": ("results/iclr_attribution_coverage.csv",),
    "mechanism.payment_kernel": (
        "results/iclr_payment_intervention_manifest.json",
    ),
    "outcome.seed_stability": tuple(
        f"results/iclr_frontier/frontier_outcome_temporal_2022_seed{seed}.json"
        for seed in (1, 2, 3, 42)
    ),
    "primary.corpus": (
        "results/frozen_instances.txt",
        "results/iclr_instance_granularity.csv",
    ),
    "summary.endowment_seed_stability": (
        "results/iclr_train/run_endow_seed1.json",
        "results/iclr_train/run_endow_seed2.json",
        "results/iclr_train/run_main_seed42.json",
    ),
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_json_constant(value: str) -> None:
    raise RuntimeError(f"non-finite JSON token: {value}")


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise RuntimeError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _validate_json_tree(value: object, *, label: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{label} must contain only finite numbers")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_json_tree(child, label=f"{label}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise RuntimeError(f"{label} contains a non-string JSON key")
            _validate_json_tree(child, label=f"{label}.{key}")
        return
    raise RuntimeError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _load_json_bytes(content: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if type(payload) is not dict:
        raise RuntimeError(f"{label} must be a JSON object")
    _validate_json_tree(payload, label=label)
    return payload


def _load_csv_bytes(content: bytes, *, label: str) -> list[dict[str, str]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 CSV") from exc
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise RuntimeError(f"{label} is not valid CSV") from exc
    if not rows or not rows[0] or any(header == "" for header in rows[0]):
        raise RuntimeError(f"{label} must have a nonempty CSV header")
    headers = rows[0]
    if len(headers) != len(set(headers)):
        raise RuntimeError(f"{label} has a duplicate CSV header")
    result: list[dict[str, str]] = []
    for line_number, values in enumerate(rows[1:], start=2):
        if len(values) != len(headers):
            raise RuntimeError(f"{label} row {line_number} has the wrong width")
        result.append(dict(zip(headers, values, strict=True)))
    return result


def _finite_float_exact(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise RuntimeError(f"{label} must contain exact JSON floats")
    if not math.isfinite(value):
        raise RuntimeError(f"{label} must be finite")
    return value


def _float_vector_exact(value: object, *, label: str) -> list[float]:
    if type(value) is not list or not value:
        raise RuntimeError(f"{label} must be a nonempty list")
    return [
        _finite_float_exact(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _path_get(payload: object, path: Sequence[object], *, label: str) -> object:
    value = payload
    for component in path:
        if type(component) is str:
            if type(value) is not dict or component not in value:
                raise RuntimeError(f"{label} missing path component {component!r}")
            value = value[component]
        elif type(component) is int:
            if type(value) is not list or not 0 <= component < len(value):
                raise RuntimeError(f"{label} missing path index {component}")
            value = value[component]
        else:
            raise RuntimeError(f"{label} contains an invalid path component")
    return value


def _target_from_fit(row: Mapping[str, object]) -> float:
    target = row.get("soft_welfare_target")
    if type(target) is not float:
        raise RuntimeError("frontier inventory target must be an exact JSON float")
    return target


def v1_fit_mapping() -> list[dict[str, object]]:
    """Return the explicit protected-v1 source mapping for all 49 v2 fits."""

    rows: list[dict[str, object]] = []
    for fit in protocol_v2.locked_fit_inventory_v2():
        fit_id = str(fit["fit_id"])
        family = str(fit["family"])
        seed = int(fit["seed"])
        selector: dict[str, object]
        weight_path: list[object] | None
        objective_path: list[object] | None
        recorded_test_metric_paths: dict[str, list[object]]

        if family == "outcome_frontier":
            source_member = (
                f"results/iclr_frontier/"
                f"frontier_outcome_temporal_2022_seed{seed}.json"
            )
            selector = {
                "kind": "json_list_match",
                "container_path": ["points"],
                "field": "floor",
                "value": _target_from_fit(fit),
            }
            weight_path = ["weights"]
            objective_path = ["train_worst_csd"]
            recorded_test_metric_paths = {
                "worst_csd": ["test_worst_csd"],
                "mean_csd": ["test_mean_csd"],
                "welfare": ["test_welfare"],
                "exclusion": ["test_exclusion"],
            }
        elif family == "endowment_frontier":
            source_member = (
                "results/iclr_frontier/"
                "frontier_endowment_temporal_2022_seed42.json"
            )
            selector = {
                "kind": "json_list_match",
                "container_path": ["points"],
                "field": "floor",
                "value": _target_from_fit(fit),
            }
            weight_path = ["weights"]
            objective_path = ["train_worst_csd"]
            recorded_test_metric_paths = {
                "worst_csd": ["test_worst_csd"],
                "mean_csd": ["test_mean_csd"],
                "welfare": ["test_welfare"],
                "exclusion": ["test_exclusion"],
            }
        elif family == "temporal_endowment":
            source_member = (
                "results/iclr_train/run_main_seed42.json"
                if seed == 42
                else f"results/iclr_train/run_endow_seed{seed}.json"
            )
            selector = {"kind": "json_root"}
            weight_path = ["best_weights"]
            objective_path = ["best_train_loss"]
            recorded_test_metric_paths = {
                metric: ["learned", "test", metric] for metric in METRIC_FIELDS
            }
        elif family == "district_endowment":
            split = str(fit["split"])
            fold_match = re.fullmatch(r"district_out_f([0-4])of5", split)
            if fold_match is None:
                raise RuntimeError(f"unexpected district split in inventory: {split}")
            suffix = "_b40" if fit["bound"] == 40.0 else ""
            source_member = (
                "results/iclr_train/"
                f"run_district_endow_f{fold_match.group(1)}_g25{suffix}.json"
            )
            selector = {"kind": "json_root"}
            weight_path = ["best_weights"]
            objective_path = ["best_train_loss"]
            recorded_test_metric_paths = {
                metric: ["learned", "test", metric] for metric in METRIC_FIELDS
            }
        elif family == "outcome_loo":
            masked = fit["masked_feature_names"]
            if type(masked) is not list or len(masked) != 1 or type(masked[0]) is not str:
                raise RuntimeError("outcome-LOO inventory mask differs")
            source_member = "results/iclr_ablation.csv"
            selector = {"kind": "csv_row", "field": "ablated", "value": masked[0]}
            weight_path = None
            objective_path = None
            recorded_test_metric_paths = {
                "worst_csd": ["test_worst_csd"],
                "welfare": ["test_welfare"],
            }
        elif family == "static_support_floor":
            source_member = "results/iclr_identification.csv"
            selector = {
                "kind": "csv_row",
                "field": "kappa",
                "value": str(fit["support_floor_kappa"]),
            }
            weight_path = None
            objective_path = ["train_worst_csd"]
            recorded_test_metric_paths = {
                "worst_csd": ["test_worst_csd"],
                "welfare": ["test_welfare"],
                "exclusion": ["test_exclusion"],
            }
        elif family == "history_free":
            source_member = (
                "analysis-output/ml-contribution-audit/"
                "history_free_seed42_g30.json"
            )
            selector = {"kind": "json_root"}
            weight_path = ["summary", "best_weights"]
            objective_path = ["summary", "best_train_loss"]
            recorded_test_metric_paths = {
                "worst_csd": ["summary", "history_free_mean"],
                "welfare": ["summary", "history_free_welfare"],
                "exclusion": ["summary", "history_free_exclusion"],
            }
        elif family == "static_age_lookup":
            source_member = (
                "analysis-output/ml-contribution-audit/"
                "static_age_lookup_seed42_g30.json"
            )
            selector = {"kind": "json_root"}
            weight_path = ["summary", "best_logits"]
            objective_path = ["summary", "best_train_loss"]
            recorded_test_metric_paths = {
                "worst_csd": ["summary", "policy_metrics", "age_lookup", "mean_csd"],
                "welfare": ["summary", "policy_metrics", "age_lookup", "welfare"],
                "exclusion": [
                    "summary",
                    "policy_metrics",
                    "age_lookup",
                    "mean_exclusion",
                ],
            }
        else:
            raise RuntimeError(f"unmapped primary-Warsaw family: {family}")

        rows.append(
            {
                "fit_id": fit_id,
                "family": family,
                "source_member": source_member,
                "selector": selector,
                "weight_path": weight_path,
                "train_objective_path": objective_path,
                "recorded_test_metric_paths": recorded_test_metric_paths,
            }
        )
    rows.sort(key=lambda row: str(row["fit_id"]))
    return rows


_MAPPING_FIELDS = frozenset(
    {
        "fit_id",
        "family",
        "source_member",
        "selector",
        "weight_path",
        "train_objective_path",
        "recorded_test_metric_paths",
    }
)


def _safe_member_name(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise RuntimeError(f"{label} must be a nonempty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or value != pure.as_posix():
        raise RuntimeError(f"{label} is not a safe relative archive member")
    return value


def validate_v1_fit_mapping(
    mapping: object, inventory: object
) -> list[dict[str, object]]:
    expected_inventory = protocol_v2.validate_locked_fit_inventory_v2(inventory)
    if type(mapping) is not list or len(mapping) != 49:
        raise RuntimeError("v1 fit mapping must cover exactly 49 coordinates")
    expected = {
        str(row["fit_id"]): str(row["family"]) for row in expected_inventory
    }
    validated: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in mapping:
        if type(row) is not dict or set(row) != _MAPPING_FIELDS:
            raise RuntimeError("v1 fit mapping row schema differs")
        fit_id = row["fit_id"]
        family = row["family"]
        if type(fit_id) is not str or fit_id in seen:
            raise RuntimeError("v1 fit mapping contains a duplicate or invalid fit_id")
        if fit_id not in expected or family != expected[fit_id]:
            raise RuntimeError("v1 fit mapping identity differs from locked inventory")
        _safe_member_name(row["source_member"], label=f"mapping[{fit_id}].source_member")
        if type(row["selector"]) is not dict:
            raise RuntimeError("v1 fit mapping selector must be an object")
        for path_field in ("weight_path", "train_objective_path"):
            path = row[path_field]
            if path is not None and (
                type(path) is not list
                or not path
                or any(type(part) not in (str, int) for part in path)
            ):
                raise RuntimeError(f"v1 fit mapping {path_field} differs")
        if type(row["recorded_test_metric_paths"]) is not dict:
            raise RuntimeError("v1 recorded metric mapping must be an object")
        seen.add(fit_id)
        validated.append(dict(row))
    if seen != set(expected):
        raise RuntimeError("v1 fit mapping coordinate coverage differs")
    if [row["fit_id"] for row in validated] != sorted(seen):
        raise RuntimeError("v1 fit mapping must be sorted by fit_id")
    unavailable = {
        str(row["fit_id"]) for row in validated if row["weight_path"] is None
    }
    expected_unavailable = {
        fit_id
        for fit_id, family in expected.items()
        if family in {"outcome_loo", "static_support_floor"}
    }
    if unavailable != expected_unavailable:
        raise RuntimeError("v1 missing-weight disclosure differs from protected evidence")
    return validated


def _exact_scalar_equal(left: object, right: object) -> bool:
    """Compare selector scalars without accepting bool/int/float aliases."""

    return type(left) is type(right) and left == right


def _selected_v1_object(
    mapping: Mapping[str, object], content: bytes
) -> tuple[Mapping[str, object], bool]:
    """Select exactly one legacy object and report whether it came from CSV."""

    member = str(mapping["source_member"])
    selector = mapping["selector"]
    if type(selector) is not dict or type(selector.get("kind")) is not str:
        raise RuntimeError("v1 selector schema differs")
    kind = selector["kind"]
    if kind == "json_root":
        if set(selector) != {"kind"}:
            raise RuntimeError("v1 JSON-root selector schema differs")
        return _load_json_bytes(content, label=member), False
    if kind == "json_list_match":
        if set(selector) != {"kind", "container_path", "field", "value"}:
            raise RuntimeError("v1 JSON-list selector schema differs")
        payload = _load_json_bytes(content, label=member)
        container = _path_get(
            payload, selector["container_path"], label=f"{member} selector"
        )
        field = selector["field"]
        if type(container) is not list or type(field) is not str:
            raise RuntimeError("v1 JSON-list selector target differs")
        matches = [
            row
            for row in container
            if type(row) is dict
            and field in row
            and _exact_scalar_equal(row[field], selector["value"])
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"v1 selector must identify exactly one row; found {len(matches)}"
            )
        return matches[0], False
    if kind == "csv_row":
        if set(selector) != {"kind", "field", "value"}:
            raise RuntimeError("v1 CSV-row selector schema differs")
        field = selector["field"]
        value = selector["value"]
        if type(field) is not str or type(value) is not str:
            raise RuntimeError("v1 CSV-row selector values must be strings")
        rows = _load_csv_bytes(content, label=member)
        matches = [row for row in rows if row.get(field) == value]
        if len(matches) != 1:
            raise RuntimeError(
                f"v1 selector must identify exactly one row; found {len(matches)}"
            )
        return matches[0], True
    raise RuntimeError(f"unsupported v1 selector kind: {kind}")


def _legacy_float(value: object, *, from_csv: bool, label: str) -> float:
    if from_csv:
        if type(value) is not str or not value:
            raise RuntimeError(f"{label} must be a nonempty CSV number")
        try:
            converted = float(value)
        except ValueError as exc:
            raise RuntimeError(f"{label} must be numeric") from exc
        if not math.isfinite(converted):
            raise RuntimeError(f"{label} must be finite")
        return converted
    if type(value) not in (int, float):
        raise RuntimeError(f"{label} must be an exact JSON number")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"{label} must be finite")
    return converted


def extract_v1_fit_record(
    mapping: object, content: bytes
) -> dict[str, object]:
    """Extract one fit record solely from its authenticated protected bytes."""

    if type(mapping) is not dict or set(mapping) != _MAPPING_FIELDS:
        raise RuntimeError("v1 fit mapping row schema differs")
    if type(content) is not bytes:
        raise RuntimeError("protected v1 fit content must be bytes")
    fit_id = mapping["fit_id"]
    family = mapping["family"]
    member = mapping["source_member"]
    if any(type(value) is not str or not value for value in (fit_id, family, member)):
        raise RuntimeError("v1 fit mapping identity differs")
    _safe_member_name(member, label="v1 source member")
    selected, from_csv = _selected_v1_object(mapping, content)

    weight_path = mapping["weight_path"]
    if weight_path is None:
        weight_status = "not_recorded_in_protected_release"
        weights = None
    else:
        raw_weights = _path_get(selected, weight_path, label=f"{fit_id} v1 weights")
        if from_csv:
            raise RuntimeError("protected CSV mappings may not claim recorded weights")
        weights = _float_vector_exact(raw_weights, label=f"{fit_id} v1 weights")
        weight_status = "recorded"

    objective_path = mapping["train_objective_path"]
    train_objective = (
        None
        if objective_path is None
        else _legacy_float(
            _path_get(selected, objective_path, label=f"{fit_id} train objective"),
            from_csv=from_csv,
            label=f"{fit_id} train objective",
        )
    )
    metric_paths = mapping["recorded_test_metric_paths"]
    if type(metric_paths) is not dict:
        raise RuntimeError("v1 recorded-test-metric mapping differs")
    recorded_metrics: dict[str, float] = {}
    for metric in sorted(metric_paths):
        path = metric_paths[metric]
        if type(metric) is not str or type(path) is not list or not path:
            raise RuntimeError("v1 recorded-test-metric path differs")
        recorded_metrics[metric] = _legacy_float(
            _path_get(selected, path, label=f"{fit_id} metric {metric}"),
            from_csv=from_csv,
            label=f"{fit_id} metric {metric}",
        )
    return _validated_v1_fit_record(
        {
            "fit_id": fit_id,
            "family": family,
            "source_member": member,
            "source_sha256": _sha256(content),
            "weight_status": weight_status,
            "weights": weights,
            "train_objective": train_objective,
            "recorded_test_metrics": recorded_metrics,
        }
    )


def _validated_v1_fit_record(record: object) -> dict[str, object]:
    fields = frozenset(
        {
            "fit_id",
            "family",
            "source_member",
            "source_sha256",
            "weight_status",
            "weights",
            "train_objective",
            "recorded_test_metrics",
        }
    )
    if type(record) is not dict or set(record) != fields:
        raise RuntimeError("v1 fit record schema differs")
    for field in ("fit_id", "family", "source_member"):
        if type(record[field]) is not str or not record[field]:
            raise RuntimeError(f"v1 fit record {field} differs")
    if not _is_sha256(record["source_sha256"]):
        raise RuntimeError("v1 fit record source SHA-256 differs")
    status = record["weight_status"]
    if status == "recorded":
        _float_vector_exact(record["weights"], label="v1 weights")
    elif status == "not_recorded_in_protected_release":
        if record["weights"] is not None:
            raise RuntimeError("unrecorded v1 weights must remain null")
    else:
        raise RuntimeError("v1 fit record weight status differs")
    if record["train_objective"] is not None:
        _finite_float_exact(record["train_objective"], label="v1 train objective")
    metrics = record["recorded_test_metrics"]
    if type(metrics) is not dict:
        raise RuntimeError("v1 recorded test metrics must be an object")
    for name, value in metrics.items():
        if type(name) is not str:
            raise RuntimeError("v1 test metric name differs")
        _finite_float_exact(value, label=f"v1 test metric {name}")
    return record


def compare_fit_record(
    v1_record: object, v2_payload: object, *, v2_sha256: str
) -> dict[str, object]:
    """Compare one mapped fit without fabricating absent protected-v1 weights."""

    v1 = _validated_v1_fit_record(v1_record)
    if not _is_sha256(v2_sha256):
        raise RuntimeError("v2 fit SHA-256 differs")
    if type(v2_payload) is not dict:
        raise RuntimeError("v2 fit must be a JSON object")
    fit_id = v1["fit_id"]
    if v2_payload.get("fit_id") != fit_id:
        raise RuntimeError("v1/v2 fit identity differs")
    config = v2_payload.get("config")
    if type(config) is not dict or config.get("fit_id") != fit_id:
        raise RuntimeError("v2 config identity differs")
    if v2_payload.get("schema_version") != 2 or type(v2_payload.get("schema_version")) is not int:
        raise RuntimeError("v2 fit schema version differs")
    if v2_payload.get("training_only") is not True:
        raise RuntimeError("v2 fit must be training-only")
    if v2_payload.get("execution_mode") != "serial":
        raise RuntimeError("v2 fit execution mode differs")
    result = v2_payload.get("result")
    if type(result) is not dict:
        raise RuntimeError("v2 fit result schema differs")
    v2_weights = _float_vector_exact(
        result.get("selected_weights"), label="v2 selected_weights"
    )
    v2_loss = _finite_float_exact(result.get("best_loss"), label="v2 best_loss")
    v1_weights = v1["weights"]
    if v1_weights is None:
        weights_comparable = False
        weights_exact: bool | None = None
        weight_linf_delta: float | None = None
    else:
        assert type(v1_weights) is list
        if len(v1_weights) != len(v2_weights):
            raise RuntimeError("v1/v2 weight dimension differs")
        weights_comparable = True
        weights_exact = v1_weights == v2_weights
        weight_linf_delta = max(
            abs(v2_value - v1_value)
            for v1_value, v2_value in zip(v1_weights, v2_weights, strict=True)
        )
    v1_loss = v1["train_objective"]
    loss_delta = None if v1_loss is None else v2_loss - v1_loss
    return {
        "fit_id": fit_id,
        "family": v1["family"],
        "v1_source_member": v1["source_member"],
        "v1_source_sha256": v1["source_sha256"],
        "v2_source_sha256": v2_sha256,
        "v1_weight_status": v1["weight_status"],
        "v1_weights": v1_weights,
        "v2_weights": v2_weights,
        "weights_comparable": weights_comparable,
        "weights_exact": weights_exact,
        "weight_linf_delta": weight_linf_delta,
        "v1_train_objective": v1_loss,
        "v2_train_objective": v2_loss,
        "train_objective_delta_v2_minus_v1": loss_delta,
        "v1_recorded_test_metrics": v1["recorded_test_metrics"],
    }


def authenticate_complete_v2_matrix(
    repo_root: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Authenticate, independently reread, and validate the exact 49-fit matrix."""

    root = Path(repo_root).resolve()
    result_root = root / V2_RESULT_RELATIVE
    data_dir = root / "data" / "pb"
    complete = train_v2.verify_complete_matrix_v2(
        repo_root=root,
        result_root=result_root,
        data_dir=data_dir,
    )
    expected_fields = {
        "schema_version",
        "status",
        "protocol_lock_sha256",
        "grid_sha256",
        "fit_count",
        "fit_sha256",
    }
    if type(complete) is not dict or set(complete) != expected_fields:
        raise RuntimeError("complete primary-v2 matrix receipt schema differs")
    if (
        type(complete["schema_version"]) is not int
        or complete["schema_version"] != 2
        or complete["status"] != "complete"
        or type(complete["fit_count"]) is not int
        or complete["fit_count"] != 49
        or not _is_sha256(complete["protocol_lock_sha256"])
        or not _is_sha256(complete["grid_sha256"])
    ):
        raise RuntimeError("complete primary-v2 matrix receipt identity differs")
    inventory = protocol_v2.locked_fit_inventory_v2()
    protocol_v2.validate_locked_fit_inventory_v2(inventory)
    expected_fit_ids = [str(spec["fit_id"]) for spec in inventory]
    fit_sha = complete["fit_sha256"]
    if (
        type(fit_sha) is not dict
        or list(fit_sha) != expected_fit_ids
        or any(not _is_sha256(value) for value in fit_sha.values())
    ):
        raise RuntimeError("complete primary-v2 fit SHA-256 map differs")

    lock = protocol_v2.verify_primary_warsaw_protocol_lock_snapshot_v2(
        result_root / "protocol_lock.json", root, result_root, data_dir
    )
    if lock.sha256 != complete["protocol_lock_sha256"]:
        raise RuntimeError("primary-v2 protocol lock changed after matrix verification")
    grid_path = protocol_v2.canonical_grid_path_v2(result_root)
    grid_content = _read_regular(grid_path, label="primary-v2 training grid")
    if _sha256(grid_content) != complete["grid_sha256"]:
        raise RuntimeError("primary-v2 training grid changed after matrix verification")

    payloads: dict[str, dict[str, object]] = {}
    byte_snapshots: dict[str, bytes] = {}
    for spec in inventory:
        fit_id = str(spec["fit_id"])
        path = protocol_v2.canonical_fit_path_v2(result_root, spec)
        content = _read_regular(path, label=f"primary-v2 fit {fit_id}")
        if _sha256(content) != fit_sha[fit_id]:
            raise RuntimeError(f"primary-v2 fit digest changed: {fit_id}")
        payload = _load_json_bytes(content, label=f"primary-v2 fit {fit_id}")
        train_v2.validate_fit_payload_v2(
            payload,
            expected_spec=spec,
            expected_protocol_lock_sha256=lock.sha256,
        )
        payloads[fit_id] = payload
        byte_snapshots[fit_id] = content
    for spec in inventory:
        fit_id = str(spec["fit_id"])
        path = protocol_v2.canonical_fit_path_v2(result_root, spec)
        if _read_regular(path, label=f"primary-v2 fit {fit_id}") != byte_snapshots[fit_id]:
            raise RuntimeError(f"primary-v2 fit changed during delta authentication: {fit_id}")
    if _read_regular(grid_path, label="primary-v2 training grid") != grid_content:
        raise RuntimeError("primary-v2 training grid changed during delta authentication")
    return dict(complete), payloads


def _episode_coordinate(row: Mapping[str, object]) -> tuple[str, str, str]:
    return str(row["fit_id"]), str(row["split"]), str(row["series"])


def _canonical_outcomes(value: object, *, label: str) -> list[dict[str, object]]:
    if type(value) is not list or not value:
        raise RuntimeError(f"{label} scored_outcomes must be a nonempty list")
    rows: list[dict[str, object]] = []
    previous_year: int | None = None
    for index, outcome in enumerate(value):
        if type(outcome) is not dict or set(outcome) != {"year", "winners"}:
            raise RuntimeError(f"{label} scored outcome row schema differs")
        year = outcome["year"]
        winners = outcome["winners"]
        if type(year) is not int or isinstance(year, bool):
            raise RuntimeError(f"{label} scored outcome year differs")
        if previous_year is not None and year <= previous_year:
            raise RuntimeError(f"{label} scored outcome years are noncanonical")
        if (
            type(winners) is not list
            or any(type(winner) is not str or not winner for winner in winners)
            or winners != sorted(set(winners))
        ):
            raise RuntimeError(f"{label} winners are noncanonical")
        rows.append({"year": year, "winners": list(winners)})
        previous_year = year
    return rows


def _validated_episode(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or not EPISODE_REQUIRED_FIELDS.issubset(value):
        raise RuntimeError(f"{label} evaluation row schema differs")
    for name in ("fit_id", "split", "series", "worst_cohort"):
        if type(value[name]) is not str or not value[name]:
            raise RuntimeError(f"{label} {name} differs")
    years = value["scored_years"]
    if (
        type(years) is not list
        or not years
        or any(type(year) is not int or isinstance(year, bool) for year in years)
        or years != sorted(set(years))
    ):
        raise RuntimeError(f"{label} scored years are noncanonical")
    for metric in METRIC_FIELDS:
        _finite_float_exact(value[metric], label=f"{label} {metric}")
    outcomes = _canonical_outcomes(value["scored_outcomes"], label=label)
    if [row["year"] for row in outcomes] != years:
        raise RuntimeError(f"{label} scored outcome coverage differs")
    result = dict(value)
    result["scored_outcomes"] = outcomes
    return result


def _outcome_sha(outcomes: object) -> str:
    return _sha256(
        json.dumps(
            outcomes,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _episode_index(rows: object, *, label: str) -> dict[tuple[str, str, str], dict[str, object]]:
    if type(rows) is not list:
        raise RuntimeError(f"{label} evaluation rows must be a list")
    indexed: dict[tuple[str, str, str], dict[str, object]] = {}
    for index, value in enumerate(rows):
        row = _validated_episode(value, label=f"{label}[{index}]")
        coordinate = _episode_coordinate(row)
        if coordinate in indexed:
            raise RuntimeError(f"{label} contains a duplicate evaluation coordinate")
        indexed[coordinate] = row
    return indexed


def compare_evaluation_rows(
    v1_rows: object, v2_rows: object
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compare every matched winner trace and all five reported metrics."""

    v1 = _episode_index(v1_rows, label="v1")
    v2 = _episode_index(v2_rows, label="v2")
    if set(v1) != set(v2):
        missing = sorted(set(v1) - set(v2))
        extra = sorted(set(v2) - set(v1))
        raise RuntimeError(
            f"v1/v2 evaluation coordinate coverage differs: missing={missing}, extra={extra}"
        )
    maxima = {metric: 0.0 for metric in METRIC_FIELDS}
    result: list[dict[str, object]] = []
    winner_changed = 0
    metric_changed = 0
    for coordinate in sorted(v1):
        old = v1[coordinate]
        new = v2[coordinate]
        if old["scored_years"] != new["scored_years"]:
            raise RuntimeError(f"scored-year coverage differs at {coordinate}")
        winner_exact = old["scored_outcomes"] == new["scored_outcomes"]
        winner_changed += int(not winner_exact)
        deltas: dict[str, float] = {}
        for metric in METRIC_FIELDS:
            delta = new[metric] - old[metric]
            deltas[metric] = delta
            maxima[metric] = max(maxima[metric], abs(delta))
        metric_changed += int(any(delta != 0.0 for delta in deltas.values()))
        result.append(
            {
                "fit_id": coordinate[0],
                "split": coordinate[1],
                "series": coordinate[2],
                "winner_exact": winner_exact,
                "worst_cohort_exact": old["worst_cohort"] == new["worst_cohort"],
                **{
                    f"{metric}_delta_v2_minus_v1": deltas[metric]
                    for metric in METRIC_FIELDS
                },
                "v1_scored_outcomes_sha256": _outcome_sha(old["scored_outcomes"]),
                "v2_scored_outcomes_sha256": _outcome_sha(new["scored_outcomes"]),
            }
        )
    return (
        {
            "total_rows": len(result),
            "winner_changed_rows": winner_changed,
            "metric_changed_rows": metric_changed,
            "max_absolute_metric_delta": maxima,
        },
        result,
    )


def _normalized_per_series_csv_bytes(payload: Mapping[str, object]) -> bytes:
    """Independently reproduce the evaluator's normalized long CSV bytes."""

    fields = (
        "fit_id",
        "source_kind",
        "split",
        "view",
        "scheme",
        "series",
        "scored_years",
        "worst_csd",
        "worst_cohort",
        "mean_csd",
        "welfare",
        "cost_welfare",
        "exclusion",
        "winner_trace_sha256",
    )
    rows = payload.get("rows")
    if type(rows) is not list:
        raise RuntimeError("primary-v2 normalized rows differ")
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if type(row) is not dict or type(row.get("metrics")) is not dict:
            raise RuntimeError("primary-v2 normalized row schema differs")
        metrics = row["metrics"]
        trace = json.dumps(
            row.get("year_outcomes"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        writer.writerow(
            {
                "fit_id": row.get("fit_id"),
                "source_kind": row.get("source_kind"),
                "split": row.get("split"),
                "view": row.get("view"),
                "scheme": row.get("scheme"),
                "series": row.get("series"),
                "scored_years": "|".join(map(str, row.get("scored_years", []))),
                "worst_csd": metrics.get("worst_csd"),
                "worst_cohort": metrics.get("worst_cohort"),
                "mean_csd": metrics.get("mean_csd"),
                "welfare": metrics.get("welfare"),
                "cost_welfare": metrics.get("cost_welfare"),
                "exclusion": metrics.get("exclusion"),
                "winner_trace_sha256": _sha256(trace),
            }
        )
    return handle.getvalue().encode("utf-8")


def _validate_matrix_receipt_for_evaluation(value: object) -> dict[str, object]:
    fields = {
        "schema_version",
        "status",
        "protocol_lock_sha256",
        "grid_sha256",
        "fit_count",
        "fit_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise RuntimeError("primary-v2 matrix receipt schema differs")
    fit_sha = value["fit_sha256"]
    expected_ids = [
        str(row["fit_id"]) for row in protocol_v2.locked_fit_inventory_v2()
    ]
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["status"] != "complete"
        or type(value["fit_count"]) is not int
        or value["fit_count"] != 49
        or not _is_sha256(value["protocol_lock_sha256"])
        or not _is_sha256(value["grid_sha256"])
        or type(fit_sha) is not dict
        or list(fit_sha) != expected_ids
        or any(not _is_sha256(digest) for digest in fit_sha.values())
    ):
        raise RuntimeError("primary-v2 matrix receipt identity differs")
    return value


def authenticate_v2_evaluation_commit(
    repo_root: Path, matrix_receipt: object
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    """Authenticate the final-summary commit and every byte it transitively binds."""

    matrix = _validate_matrix_receipt_for_evaluation(matrix_receipt)
    root = Path(repo_root).resolve()
    result_root = root / V2_RESULT_RELATIVE
    evaluation_root = result_root / "evaluation"
    relative_paths = (
        "evaluation/per_series.json",
        "evaluation/per_series.csv",
        "evaluation/paper_consumers.json",
        "evaluation/artifact_manifest.json",
        "evaluation/summary.json",
    )
    if not evaluation_root.is_dir() or evaluation_root.is_symlink():
        raise RuntimeError("primary-v2 evaluation commit directory is missing or unsafe")
    expected_paths = {(result_root / relative).absolute() for relative in relative_paths}
    actual_paths: set[Path] = set()
    for path in evaluation_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"primary-v2 evaluation commit contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(path.absolute())
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - expected_paths)
        raise RuntimeError(
            f"primary-v2 evaluation commit path coverage differs: missing={missing}, extra={extra}"
        )
    content = {
        relative: _read_regular(result_root / relative, label=f"primary-v2 {relative}")
        for relative in relative_paths
    }
    receipt_path = result_root / evaluate_v2.REPLAY_RECEIPT_RELATIVE_PATH
    receipt_content = _read_regular(receipt_path, label="primary-v2 replay receipt")
    receipt = _load_json_bytes(receipt_content, label="primary-v2 replay receipt")
    per_series = _load_json_bytes(
        content["evaluation/per_series.json"], label="primary-v2 per-series JSON"
    )
    consumers = _load_json_bytes(
        content["evaluation/paper_consumers.json"],
        label="primary-v2 paper consumers JSON",
    )
    manifest = _load_json_bytes(
        content["evaluation/artifact_manifest.json"],
        label="primary-v2 evaluation manifest",
    )
    summary = _load_json_bytes(
        content["evaluation/summary.json"], label="primary-v2 evaluation summary"
    )
    expected_fit_ids = list(matrix["fit_sha256"])
    evaluate_v2.validate_per_series_payload_v2(
        per_series, expected_fit_ids=expected_fit_ids
    )
    inventory = protocol_v2.locked_fit_inventory_v2()
    evaluate_v2.validate_paper_consumers_payload_v2(
        consumers, inventory=inventory
    )
    for label, payload in (("per-series", per_series), ("paper consumers", consumers)):
        provenance = payload.get("provenance")
        if (
            type(provenance) is not dict
            or provenance.get("protocol_lock_sha256")
            != matrix["protocol_lock_sha256"]
        ):
            raise RuntimeError(f"primary-v2 {label} protocol provenance differs")

    summary_fields = {
        "schema_version",
        "status",
        "classification",
        "successful_process_exit_required_before_consumption",
        "fit_count",
        "consumer_count",
        "per_series_row_count",
        "protocol_lock_sha256",
        "replay_receipt_sha256",
        "training_grid_sha256",
        "fit_sha256",
        "output_sha256",
    }
    expected_output_names = set(relative_paths[:-1])
    output_sha = summary.get("output_sha256")
    if (
        set(summary) != summary_fields
        or type(summary.get("schema_version")) is not int
        or summary.get("schema_version") != 2
        or summary.get("status") != "complete"
        or summary.get("classification") != "corrected post-hoc replay"
        or summary.get("successful_process_exit_required_before_consumption") is not True
        or type(summary.get("fit_count")) is not int
        or summary.get("fit_count") != 49
        or type(summary.get("consumer_count")) is not int
        or summary.get("consumer_count") != len(evaluate_v2.PAPER_CONSUMER_CONTRACT_V2)
        or type(summary.get("per_series_row_count")) is not int
        or summary.get("per_series_row_count") != len(per_series["rows"])
        or summary.get("protocol_lock_sha256") != matrix["protocol_lock_sha256"]
        or summary.get("training_grid_sha256") != matrix["grid_sha256"]
        or not protocol_v2.exact_json_equal_v2(
            summary.get("fit_sha256"), matrix["fit_sha256"]
        )
        or type(output_sha) is not dict
        or set(output_sha) != expected_output_names
    ):
        raise RuntimeError("primary-v2 evaluation summary commit differs")
    for relative in sorted(expected_output_names):
        if output_sha[relative] != _sha256(content[relative]):
            raise RuntimeError(f"primary-v2 summary digest differs for {relative}")

    receipt_sha = _sha256(receipt_content)
    receipt_fields = {
        "schema_version",
        "event",
        "classification",
        "heldout_outcomes_already_known",
        "fresh_holdout",
        "preregistered",
        "disclosure",
        "protocol_lock_sha256",
        "semantics_profile",
        "semantics_receipt_sha256",
        "corpus_semantic_sha256",
        "structural_gates_sha256",
        "training_grid_sha256",
        "fit_count",
        "fit_sha256",
    }
    if (
        set(receipt) != receipt_fields
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 2
        or receipt.get("event") != "evaluation replay started"
        or receipt.get("classification") != "corrected post-hoc replay"
        or receipt.get("heldout_outcomes_already_known") is not True
        or receipt.get("fresh_holdout") is not False
        or receipt.get("preregistered") is not False
        or receipt.get("protocol_lock_sha256") != matrix["protocol_lock_sha256"]
        or receipt.get("training_grid_sha256") != matrix["grid_sha256"]
        or receipt.get("fit_count") != 49
        or not protocol_v2.exact_json_equal_v2(
            receipt.get("fit_sha256"), matrix["fit_sha256"]
        )
        or summary.get("replay_receipt_sha256") != receipt_sha
    ):
        raise RuntimeError("primary-v2 replay receipt or summary binding differs")

    manifest_fields = {"schema_version", "classification", "files", "provenance"}
    manifest_files = manifest.get("files")
    manifest_source_names = {
        "evaluation/per_series.json",
        "evaluation/per_series.csv",
        "evaluation/paper_consumers.json",
    }
    expected_provenance = {
        "protocol_lock_sha256": matrix["protocol_lock_sha256"],
        "replay_receipt_sha256": receipt_sha,
        "training_grid_sha256": matrix["grid_sha256"],
        "fit_sha256": matrix["fit_sha256"],
    }
    if (
        set(manifest) != manifest_fields
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
        or manifest.get("classification") != "corrected post-hoc replay"
        or type(manifest_files) is not dict
        or set(manifest_files) != manifest_source_names
        or not protocol_v2.exact_json_equal_v2(
            manifest.get("provenance"), expected_provenance
        )
    ):
        raise RuntimeError("primary-v2 evaluation artifact manifest differs")
    for relative in sorted(manifest_source_names):
        row = manifest_files[relative]
        if (
            type(row) is not dict
            or set(row) != {"bytes", "sha256"}
            or type(row["bytes"]) is not int
            or row["bytes"] != len(content[relative])
            or row["sha256"] != _sha256(content[relative])
        ):
            raise RuntimeError(f"primary-v2 manifest identity differs for {relative}")
    if content["evaluation/per_series.csv"] != _normalized_per_series_csv_bytes(per_series):
        raise RuntimeError("primary-v2 per-series CSV differs from independent serialization")
    for relative, original in content.items():
        if _read_regular(result_root / relative, label=f"primary-v2 {relative}") != original:
            raise RuntimeError(f"primary-v2 evaluation output changed during audit: {relative}")
    if _read_regular(receipt_path, label="primary-v2 replay receipt") != receipt_content:
        raise RuntimeError("primary-v2 replay receipt changed during audit")
    authenticated = {
        "evaluation_summary_sha256": _sha256(content["evaluation/summary.json"]),
        "evaluation_manifest_sha256": _sha256(
            content["evaluation/artifact_manifest.json"]
        ),
        "evaluation_replay_receipt_sha256": receipt_sha,
        "per_series_sha256": _sha256(content["evaluation/per_series.json"]),
        "paper_consumers_sha256": _sha256(
            content["evaluation/paper_consumers.json"]
        ),
        "per_series_row_count": len(per_series["rows"]),
        "consumer_count": len(consumers["consumer_ids"]),
    }
    return authenticated, list(per_series["rows"]), consumers


def _validate_normalized_fit_test_row(
    value: object, *, label: str
) -> dict[str, object]:
    fields = {
        "fit_id",
        "source_kind",
        "split",
        "view",
        "scheme",
        "series",
        "scored_years",
        "year_outcomes",
        "metrics",
    }
    if type(value) is not dict or set(value) != fields:
        raise RuntimeError(f"{label} normalized evaluation row schema differs")
    for field in ("fit_id", "split", "scheme", "series"):
        if type(value[field]) is not str or not value[field]:
            raise RuntimeError(f"{label} normalized {field} differs")
    if value["source_kind"] != "fit" or value["view"] != "test":
        raise RuntimeError(f"{label} is not a fitted test-view row")
    if value["scheme"] != "age_sex":
        raise RuntimeError(f"{label} is not the canonical age_sex scheme")
    years = value["scored_years"]
    if (
        type(years) is not list
        or not years
        or any(type(year) is not int for year in years)
        or years != sorted(set(years))
    ):
        raise RuntimeError(f"{label} scored-year inventory differs")
    outcomes = value["year_outcomes"]
    year_fields = {
        "year",
        "scored",
        "winners",
        "spent",
        "welfare",
        "cost_welfare",
        "exclusion",
    }
    if type(outcomes) is not list or not outcomes:
        raise RuntimeError(f"{label} winner trace is empty")
    seen_years: list[int] = []
    scored: list[int] = []
    for outcome in outcomes:
        if type(outcome) is not dict or set(outcome) != year_fields:
            raise RuntimeError(f"{label} year outcome schema differs")
        year = outcome["year"]
        winners = outcome["winners"]
        if type(year) is not int or type(outcome["scored"]) is not bool:
            raise RuntimeError(f"{label} year outcome identity differs")
        if (
            type(winners) is not list
            or any(type(winner) is not str for winner in winners)
            or winners != sorted(set(winners))
        ):
            raise RuntimeError(f"{label} winners are noncanonical")
        for metric in ("spent", "welfare", "cost_welfare", "exclusion"):
            _finite_float_exact(outcome[metric], label=f"{label} year {metric}")
        seen_years.append(year)
        if outcome["scored"]:
            scored.append(year)
    if seen_years != sorted(set(seen_years)) or scored != years:
        raise RuntimeError(f"{label} year outcome coverage differs")
    metrics = value["metrics"]
    if type(metrics) is not dict or set(metrics) != set(METRIC_FIELDS) | {"worst_cohort"}:
        raise RuntimeError(f"{label} episode metric schema differs")
    if metrics["worst_cohort"] is not None and type(metrics["worst_cohort"]) is not str:
        raise RuntimeError(f"{label} worst cohort differs")
    for metric in METRIC_FIELDS:
        _finite_float_exact(metrics[metric], label=f"{label} {metric}")
    return value


def compare_protected_evaluation(
    v1_fit_records: object,
    v2_rows: object,
    *,
    expected_fit_ids: Sequence[str],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Compare protected aggregate metrics and disclose absent v1 winner traces."""

    if (
        type(expected_fit_ids) not in (list, tuple)
        or not expected_fit_ids
        or any(type(fit_id) is not str or not fit_id for fit_id in expected_fit_ids)
        or len(set(expected_fit_ids)) != len(expected_fit_ids)
    ):
        raise RuntimeError("expected protected-evaluation fit ids differ")
    expected = set(expected_fit_ids)
    if type(v1_fit_records) is not list:
        raise RuntimeError("v1 fit records must be a list")
    old_by_id: dict[str, dict[str, object]] = {}
    for record in v1_fit_records:
        validated = _validated_v1_fit_record(record)
        fit_id = str(validated["fit_id"])
        if fit_id in old_by_id:
            raise RuntimeError("v1 fit records contain a duplicate identity")
        old_by_id[fit_id] = validated
    if set(old_by_id) != expected:
        raise RuntimeError("v1 fit record coverage differs")
    if type(v2_rows) is not list:
        raise RuntimeError("v2 normalized evaluation rows must be a list")
    by_fit: dict[str, list[dict[str, object]]] = {fit_id: [] for fit_id in expected}
    seen_rows: set[tuple[str, str, str, str, str]] = set()
    for index, value in enumerate(v2_rows):
        if type(value) is not dict:
            raise RuntimeError("v2 normalized evaluation row must be an object")
        if not (
            value.get("source_kind") == "fit"
            and value.get("view") == "test"
            and value.get("scheme") == "age_sex"
        ):
            continue
        row = _validate_normalized_fit_test_row(value, label=f"v2[{index}]")
        fit_id = str(row["fit_id"])
        if fit_id not in expected:
            raise RuntimeError("v2 normalized evaluation names an unexpected fit")
        coordinate = (
            fit_id,
            str(row["split"]),
            str(row["view"]),
            str(row["scheme"]),
            str(row["series"]),
        )
        if coordinate in seen_rows:
            raise RuntimeError("v2 normalized evaluation contains a duplicate coordinate")
        seen_rows.add(coordinate)
        by_fit[fit_id].append(row)
    missing = sorted(fit_id for fit_id, rows in by_fit.items() if not rows)
    if missing:
        raise RuntimeError(f"v2 canonical test evaluation fit coverage differs: {missing}")

    metric_rows: list[dict[str, object]] = []
    winner_rows: list[dict[str, object]] = []
    metric_comparable = 0
    metric_changed = 0
    for fit_id in sorted(expected):
        rows = sorted(
            by_fit[fit_id],
            key=lambda row: (str(row["split"]), str(row["series"])),
        )
        split_names = {str(row["split"]) for row in rows}
        if len(split_names) != 1:
            raise RuntimeError(f"v2 fit spans multiple canonical test splits: {fit_id}")
        aggregated: dict[str, float] = {}
        for metric in METRIC_FIELDS:
            values = [float(row["metrics"][metric]) for row in rows]
            aggregated[metric] = (
                math.fsum(values)
                if metric in {"welfare", "cost_welfare"}
                else math.fsum(values) / len(values)
            )
        old_metrics = old_by_id[fit_id]["recorded_test_metrics"]
        assert type(old_metrics) is dict
        unknown_metrics = set(old_metrics) - set(METRIC_FIELDS)
        if unknown_metrics:
            raise RuntimeError(f"v1 fit records contain unknown metrics: {unknown_metrics}")
        for metric in METRIC_FIELDS:
            old_value = old_metrics.get(metric)
            comparable = old_value is not None
            delta = aggregated[metric] - old_value if comparable else None
            changed = (delta != 0.0) if comparable else None
            metric_comparable += int(comparable)
            metric_changed += int(changed is True)
            metric_rows.append(
                {
                    "fit_id": fit_id,
                    "family": old_by_id[fit_id]["family"],
                    "split": next(iter(split_names)),
                    "metric": metric,
                    "status": (
                        "comparable"
                        if comparable
                        else "not_recorded_in_protected_release"
                    ),
                    "v1_value": old_value,
                    "v2_value": aggregated[metric],
                    "delta_v2_minus_v1": delta,
                    "changed": changed,
                }
            )
        for row in rows:
            for outcome in row["year_outcomes"]:
                winner_rows.append(
                    {
                        "fit_id": fit_id,
                        "family": old_by_id[fit_id]["family"],
                        "split": row["split"],
                        "view": row["view"],
                        "scheme": row["scheme"],
                        "series": row["series"],
                        "year": outcome["year"],
                        "scored": outcome["scored"],
                        "status": "not_recorded_in_protected_release",
                        "v1_winners": None,
                        "v2_winners": list(outcome["winners"]),
                        "winner_exact": None,
                    }
                )
    winner_rows.sort(
        key=lambda row: (
            str(row["fit_id"]),
            str(row["split"]),
            str(row["series"]),
            int(row["year"]),
        )
    )
    metric_total = len(metric_rows)
    winner_total = len(winner_rows)
    summary = {
        "fit_count": len(expected),
        "metric_coordinates": metric_total,
        "metric_comparable": metric_comparable,
        "metric_unavailable": metric_total - metric_comparable,
        "metric_changed": metric_changed,
        "winner_coordinates": winner_total,
        "winner_comparable": 0,
        "winner_unavailable": winner_total,
        "protected_winner_trace_status": "not_recorded_in_protected_release",
    }
    return summary, metric_rows, winner_rows


_MACRO_LINE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")


def parse_numbers_tex(content: bytes) -> dict[str, str]:
    """Parse the protected generated macro file without accepting silent loss."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("numbers.tex is not valid UTF-8") from exc
    macros: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        match = _MACRO_LINE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"unparsed manuscript macro at line {line_number}")
        name, value = match.groups()
        if name in macros:
            raise RuntimeError(f"duplicate manuscript macro: {name}")
        macros[name] = value
    if not macros:
        raise RuntimeError("numbers.tex contains no manuscript macros")
    return macros


def compare_manuscript_values(
    v1_values: object, v2_values: object
) -> tuple[dict[str, int], list[dict[str, object]]]:
    if type(v1_values) is not dict or type(v2_values) is not dict:
        raise RuntimeError("manuscript values must be JSON objects")
    if not v1_values or set(v1_values) != set(v2_values):
        raise RuntimeError("v1/v2 manuscript macro coverage differs")
    rows: list[dict[str, object]] = []
    for name in sorted(v1_values):
        if type(name) is not str or re.fullmatch(r"[A-Za-z]+", name) is None:
            raise RuntimeError("manuscript macro name differs")
        old = v1_values[name]
        new = v2_values[name]
        if type(old) is not str or type(new) is not str:
            raise RuntimeError("manuscript macro values must be strings")
        rows.append(
            {"name": name, "v1_value": old, "v2_value": new, "changed": old != new}
        )
    changed = sum(int(row["changed"]) for row in rows)
    return {"total": len(rows), "changed": changed, "exact": len(rows) - changed}, rows


def classify_protected_manuscript_values(
    protected_values: object,
    corrected_values: object,
    macro_coordinates: object,
) -> tuple[dict[str, int], list[dict[str, object]]]:
    """Classify every protected macro; v1-only names may never disappear silently."""

    if (
        type(protected_values) is not dict
        or not protected_values
        or type(corrected_values) is not dict
        or not corrected_values
        or type(macro_coordinates) is not dict
    ):
        raise RuntimeError("manuscript macro inventories differ")
    protected_names = set(protected_values)
    corrected_names = set(corrected_values)
    if not corrected_names <= protected_names:
        missing = sorted(corrected_names - protected_names)
        raise RuntimeError(
            f"primary-v2 manuscript macros lack a protected-v1 baseline: {missing}"
        )
    if set(macro_coordinates) != corrected_names:
        raise RuntimeError("primary-v2 macro coordinate coverage differs")

    rows: list[dict[str, object]] = []
    for name in sorted(protected_values):
        if type(name) is not str or re.fullmatch(r"[A-Za-z]+", name) is None:
            raise RuntimeError("protected manuscript macro name differs")
        old = protected_values[name]
        if type(old) is not str:
            raise RuntimeError("protected manuscript macro value must be a string")
        if name in corrected_values:
            new = corrected_values[name]
            coordinate = macro_coordinates[name]
            if type(new) is not str or type(coordinate) is not dict or not coordinate:
                raise RuntimeError("primary-v2 manuscript macro provenance differs")
            _validate_json_tree(
                coordinate, label=f"primary-v2 manuscript coordinate {name}"
            )
            rows.append(
                {
                    "name": name,
                    "classification": "compared",
                    "reason": "exact_corrected_primary_macro_reconstruction",
                    "v1_value": old,
                    "v2_value": new,
                    "changed": old != new,
                    "v2_coordinate": coordinate,
                }
            )
        elif name.startswith(NON_PRIMARY_MACRO_PREFIXES):
            rows.append(
                {
                    "name": name,
                    "classification": "unaffected_non_primary",
                    "reason": (
                        "outside_corrected_primary_warsaw_lineage: external, "
                        "cross-city, multicity, or actuation analysis"
                    ),
                    "v1_value": old,
                    "v2_value": None,
                    "changed": None,
                    "v2_coordinate": None,
                }
            )
        else:
            rows.append(
                {
                    "name": name,
                    "classification": "unavailable",
                    "reason": "no_exact_specialized_primary_v2_macro_reconstruction",
                    "v1_value": old,
                    "v2_value": None,
                    "changed": None,
                    "v2_coordinate": None,
                }
            )

    counts = {
        classification: sum(
            row["classification"] == classification for row in rows
        )
        for classification in (
            "compared",
            "unaffected_non_primary",
            "unavailable",
        )
    }
    if sum(counts.values()) != len(protected_values) or len(rows) != len(protected_values):
        raise RuntimeError("protected manuscript macro classification coverage differs")
    compared = [row for row in rows if row["classification"] == "compared"]
    return {
        "protected_macro_total": len(rows),
        "compared": counts["compared"],
        "compared_changed": sum(row["changed"] is True for row in compared),
        "compared_exact": sum(row["changed"] is False for row in compared),
        "unaffected_non_primary": counts["unaffected_non_primary"],
        "unavailable": counts["unavailable"],
    }, rows


def explicit_legacy_fit_gaps(fit_rows: object) -> list[dict[str, str]]:
    """Return the seven protected fits without serialized weights or winners."""

    if type(fit_rows) is not list or any(type(row) is not dict for row in fit_rows):
        raise RuntimeError("legacy fit-gap rows differ")
    expected = {
        str(row["fit_id"]): str(row["family"])
        for row in v1_fit_mapping()
        if row["weight_path"] is None
    }
    actual: dict[str, str] = {}
    for row in fit_rows:
        fit_id = row.get("fit_id")
        family = row.get("family")
        if (
            type(fit_id) is not str
            or type(family) is not str
            or row.get("weights_comparable") is not False
            or fit_id in actual
        ):
            raise RuntimeError("legacy fit-gap row schema differs")
        actual[fit_id] = family
    if actual != expected:
        raise RuntimeError("legacy fit-gap coverage differs")
    reason = (
        "protected release does not serialize this fit's selected weights or "
        "per-year winner trace"
    )
    return [
        {
            "fit_id": fit_id,
            "family": expected[fit_id],
            "weight_status": "not_recorded_in_protected_release",
            "winner_trace_status": "not_recorded_in_protected_release",
            "reason": reason,
        }
        for fit_id in sorted(expected)
    ]


def _legacy_numeric(value: object, *, label: str) -> float | None:
    if value is None or value == "" or value == "NA":
        return None
    if type(value) not in (str, int, float) or type(value) is bool:
        raise RuntimeError(f"{label} numeric value differs")
    try:
        numeric = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} must be finite")
    return numeric


def parse_protected_ratio(value: object) -> tuple[int, int]:
    """Parse the released ``N/D`` ratio, with an optional rounded percent suffix."""

    if type(value) is not str:
        raise RuntimeError("protected ratio value differs")
    match = re.fullmatch(
        r"([0-9]+)/([0-9]+)(?: \(([0-9]+(?:\.[0-9]+)?)\\?%\))?",
        value,
    )
    if match is None:
        raise RuntimeError("protected ratio grammar differs")
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator == 0:
        raise RuntimeError("protected ratio denominator must be positive")
    if numerator > denominator:
        raise RuntimeError("protected ratio numerator exceeds denominator")
    annotated = match.group(3)
    if annotated is not None:
        expected = round(100.0 * numerator / denominator, 1)
        if float(annotated) != expected:
            raise RuntimeError("protected ratio annotation differs")
    return numerator, denominator


def _identity_atom(value: object, *, label: str) -> tuple[str, object]:
    if type(value) in (int, float) and type(value) is not bool:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError(f"{label} identity is non-finite")
        return ("number", numeric)
    if type(value) is not str or not value:
        raise RuntimeError(f"{label} identity differs")
    try:
        numeric = float(value)
    except ValueError:
        return ("string", value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} identity is non-finite")
    return ("number", numeric)


def _coordinate_key(
    row: Mapping[str, object], fields: Sequence[str], *, label: str
) -> tuple[tuple[str, object], ...]:
    return tuple(
        _identity_atom(row.get(field), label=f"{label}.{field}") for field in fields
    )


def _coordinate_text(row: Mapping[str, object], fields: Sequence[str]) -> str:
    return "|".join(f"{field}={row[field]}" for field in fields)


def _validate_typed_consumer_values(
    consumer_id: str, values: object
) -> Mapping[str, object]:
    contracts = getattr(evaluate_v2, "PAPER_CONSUMER_VALUE_CONTRACT_V2", None)
    if type(contracts) is not dict or set(contracts) != set(
        evaluate_v2.PAPER_CONSUMER_CONTRACT_V2
    ):
        raise RuntimeError("primary-v2 typed consumer contract coverage differs")
    contract = contracts.get(consumer_id)
    if (
        type(contract) is not dict
        or set(contract) != {"schema", "lists"}
        or type(contract["schema"]) is not str
        or type(contract["lists"]) is not dict
    ):
        raise RuntimeError(f"primary-v2 typed consumer contract differs: {consumer_id}")
    if type(values) is not dict or values.get("schema") != contract["schema"]:
        raise RuntimeError(f"primary-v2 typed consumer schema differs: {consumer_id}")
    forbidden = {
        "computed_from_normalized_rows",
        "normalized_row_count",
        "source_metrics",
        "policy_metrics",
        "paired_test_contrasts",
        "rows",
        "summary",
    }
    if set(values) & forbidden:
        raise RuntimeError(f"primary-v2 typed consumer contains a generic summary: {consumer_id}")
    for table, expected_count in contract["lists"].items():
        rows = values.get(table)
        if (
            type(table) is not str
            or type(expected_count) is not int
            or expected_count < 0
            or type(rows) is not list
            or len(rows) != expected_count
            or any(type(row) is not dict or not row for row in rows)
        ):
            raise RuntimeError(
                f"primary-v2 typed consumer table schema/count differs: "
                f"{consumer_id}/{table}"
            )
    validator = getattr(evaluate_v2, "validate_consumer_values_v2", None)
    if validator is not None:
        validator(consumer_id, values)
    _validate_json_tree(values, label=f"primary-v2 typed consumer {consumer_id}")
    return values


def _compare_consumer_csv_table(
    *,
    consumer_id: str,
    artifact_schema: str,
    spec: Mapping[str, object],
    snapshot: bytes,
    values: Mapping[str, object],
) -> list[dict[str, object]]:
    source_member = spec.get("source_member")
    table = spec.get("table")
    legacy_fields = spec.get("legacy_fields")
    identity_fields = spec.get("identity_fields")
    numeric_fields = spec.get("numeric_fields")
    numeric_field_map = spec.get("numeric_field_map", {})
    context = spec.get("context", {})
    legacy_include = spec.get("legacy_include", {})
    legacy_exclude = spec.get("legacy_exclude", {})
    v2_include = spec.get("v2_include", context)
    if (
        type(source_member) is not str
        or type(table) is not str
        or type(legacy_fields) is not tuple
        or type(identity_fields) is not tuple
        or type(numeric_fields) is not tuple
        or any(type(field) is not str for field in legacy_fields)
        or any(type(field) is not str for field in identity_fields + numeric_fields)
        or type(numeric_field_map) is not dict
        or any(
            type(old) is not str or type(new) is not str
            for old, new in numeric_field_map.items()
        )
        or any(type(value) is not dict for value in (context, legacy_include, legacy_exclude, v2_include))
    ):
        raise RuntimeError(f"protected consumer comparison contract differs: {consumer_id}")
    parsed_old_rows = _load_csv_bytes(
        snapshot, label=f"protected {consumer_id} {source_member}"
    )
    if not parsed_old_rows or tuple(parsed_old_rows[0]) != legacy_fields:
        raise RuntimeError(f"protected consumer legacy table schema differs: {source_member}")
    if any(tuple(row) != legacy_fields for row in parsed_old_rows):
        raise RuntimeError(f"protected consumer legacy row schema differs: {source_member}")
    old_rows = [
        {**row, **context}
        for row in parsed_old_rows
        if all(row.get(field) == value for field, value in legacy_include.items())
        and not any(row.get(field) == value for field, value in legacy_exclude.items())
    ]
    if not old_rows:
        raise RuntimeError(f"protected consumer filtered table is empty: {source_member}")
    new_rows = values.get(table)
    if type(new_rows) is not list or any(type(row) is not dict for row in new_rows):
        raise RuntimeError(f"primary-v2 typed table differs: {consumer_id}/{table}")
    new_rows = [
        row
        for row in new_rows
        if all(
            _identity_atom(row.get(field), label=f"primary-v2 filter {field}")
            == _identity_atom(value, label=f"protected filter {field}")
            for field, value in v2_include.items()
        )
    ]
    if not new_rows:
        raise RuntimeError(f"primary-v2 filtered table is empty: {consumer_id}/{table}")
    for row in new_rows:
        mapped_numeric = {str(numeric_field_map.get(field, field)) for field in numeric_fields}
        missing = (set(identity_fields) | mapped_numeric) - set(row)
        if missing:
            raise RuntimeError(
                f"primary-v2 typed table field coverage differs: "
                f"{consumer_id}/{table}/{sorted(missing)}"
            )
    old_by_coordinate: dict[tuple[tuple[str, object], ...], dict[str, str]] = {}
    for row in old_rows:
        key = _coordinate_key(row, identity_fields, label=f"protected {consumer_id}")
        if key in old_by_coordinate:
            raise RuntimeError(f"protected consumer contains a duplicate coordinate: {consumer_id}")
        old_by_coordinate[key] = row
    new_by_coordinate: dict[tuple[tuple[str, object], ...], Mapping[str, object]] = {}
    for row in new_rows:
        key = _coordinate_key(row, identity_fields, label=f"primary-v2 {consumer_id}")
        if key in new_by_coordinate:
            raise RuntimeError(f"primary-v2 consumer contains a duplicate coordinate: {consumer_id}")
        new_by_coordinate[key] = row
    if set(old_by_coordinate) != set(new_by_coordinate):
        raise RuntimeError(f"protected/v2 consumer coordinate coverage differs: {consumer_id}/{table}")

    result: list[dict[str, object]] = []
    for key in sorted(old_by_coordinate, key=repr):
        old = old_by_coordinate[key]
        new = new_by_coordinate[key]
        coordinate = _coordinate_text(old, identity_fields)
        for field in numeric_fields:
            v2_field = str(numeric_field_map.get(field, field))
            old_value = _legacy_numeric(
                old[field], label=f"protected {consumer_id}/{coordinate}/{field}"
            )
            new_value = _legacy_numeric(
                new[v2_field], label=f"primary-v2 {consumer_id}/{coordinate}/{v2_field}"
            )
            if old_value is None and new_value is None:
                continue
            if old_value is None or new_value is None:
                raise RuntimeError(
                    f"protected/v2 consumer numeric coverage differs: "
                    f"{consumer_id}/{coordinate}/{field}"
                )
            result.append(
                {
                    "consumer_id": consumer_id,
                    "artifact_schema": artifact_schema,
                    "source_member": source_member,
                    "table": table,
                    "coordinate": coordinate,
                    "field": field,
                    "v1_value": old_value,
                    "v2_value": new_value,
                    "delta_v2_minus_v1": new_value - old_value,
                    "changed": new_value != old_value,
                }
            )
    return result


def _consumer_numeric_row(
    *,
    consumer_id: str,
    artifact_schema: str,
    source_member: str,
    table: str,
    coordinate: str,
    field: str,
    v1_value: object,
    v2_value: object,
) -> dict[str, object]:
    old = _legacy_numeric(v1_value, label=f"protected {consumer_id}/{coordinate}/{field}")
    new = _legacy_numeric(v2_value, label=f"primary-v2 {consumer_id}/{coordinate}/{field}")
    if old is None or new is None:
        raise RuntimeError(
            f"protected/v2 consumer numeric coverage differs: "
            f"{consumer_id}/{coordinate}/{field}"
        )
    return {
        "consumer_id": consumer_id,
        "artifact_schema": artifact_schema,
        "source_member": source_member,
        "table": table,
        "coordinate": coordinate,
        "field": field,
        "v1_value": old,
        "v2_value": new,
        "delta_v2_minus_v1": new - old,
        "changed": new != old,
    }


def _consumer_vector_rows(
    *,
    consumer_id: str,
    artifact_schema: str,
    source_member: str,
    table: str,
    coordinate: str,
    field: str,
    v1_values: object,
    v2_values: object,
) -> list[dict[str, object]]:
    if (
        type(v1_values) is not list
        or type(v2_values) is not list
        or not v1_values
        or len(v1_values) != len(v2_values)
    ):
        raise RuntimeError(
            f"protected/v2 consumer vector coverage differs: {consumer_id}/{field}"
        )
    return [
        _consumer_numeric_row(
            consumer_id=consumer_id,
            artifact_schema=artifact_schema,
            source_member=source_member,
            table=table,
            coordinate=coordinate,
            field=f"{field}[{index}]",
            v1_value=old,
            v2_value=new,
        )
        for index, (old, new) in enumerate(zip(v1_values, v2_values, strict=True))
    ]


def _one_record(
    records: object, *, field: str, value: object, label: str
) -> Mapping[str, object]:
    if type(records) is not list:
        raise RuntimeError(f"{label} records differ")
    matches = [
        record
        for record in records
        if type(record) is dict
        and _identity_atom(record.get(field), label=f"{label}.{field}")
        == _identity_atom(value, label=f"{label} selector")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{label} must match exactly one record")
    return matches[0]


def _special_consumer_comparisons(
    *,
    consumer_id: str,
    artifact_schema: str,
    expected_paths: Sequence[str],
    snapshots: Mapping[str, bytes],
    values: Mapping[str, object],
) -> tuple[list[dict[str, object]], set[str], list[dict[str, str]]]:
    """Compare non-tabular legacy structures and disclose non-reconstructible fields."""

    rows: list[dict[str, object]] = []
    covered: set[str] = set()
    unavailable: list[dict[str, str]] = []

    def add(
        source: str,
        table: str,
        coordinate: str,
        field: str,
        old: object,
        new: object,
    ) -> None:
        rows.append(
            _consumer_numeric_row(
                consumer_id=consumer_id,
                artifact_schema=artifact_schema,
                source_member=source,
                table=table,
                coordinate=coordinate,
                field=field,
                v1_value=old,
                v2_value=new,
            )
        )

    def gap(source: str, field: str, reason: str) -> None:
        unavailable.append(
            {"source_member": source, "field": field, "reason": reason}
        )

    if consumer_id == "control.history_free":
        source = "analysis-output/ml-contribution-audit/history_free_seed42_g30.json"
        old = _load_json_bytes(snapshots[source], label=f"protected {source}")
        summary = old.get("summary")
        if type(summary) is not dict:
            raise RuntimeError("protected history-free summary differs")
        record = _one_record(
            values["fit_records"],
            field="family",
            value="history_free",
            label="primary-v2 history-free fit",
        )
        add(source, "fit_records", "family=history_free", "best_train_loss", summary.get("best_train_loss"), record.get("best_train_loss"))
        rows.extend(
            _consumer_vector_rows(
                consumer_id=consumer_id,
                artifact_schema=artifact_schema,
                source_member=source,
                table="fit_records",
                coordinate="family=history_free",
                field="selected_weights",
                v1_values=summary.get("best_weights"),
                v2_values=record.get("selected_weights"),
            )
        )
        covered.add(source)
        gap(source, "history", "optimizer trajectories are not a typed manuscript consumer")

    elif consumer_id == "control.senior_scalar":
        source = "results/iclr_senior_tilt.csv"
        old_rows = _load_csv_bytes(snapshots[source], label=f"protected {source}")
        expected = (
            "alpha", "train_worst_csd", "test_worst_csd", "mes_worst_csd",
            "difference", "ci_low", "ci_high", "p_value", "wins", "n_series",
            "test_welfare", "test_exclusion",
        )
        if len(old_rows) != 2 or any(tuple(row) != expected for row in old_rows):
            raise RuntimeError("protected senior-scalar table schema/count differs")
        selected_rows = [row for row in old_rows if row["alpha"] != "GRID"]
        grid_rows = [row for row in old_rows if row["alpha"] == "GRID"]
        if len(selected_rows) != 1 or len(grid_rows) != 1:
            raise RuntimeError("protected senior-scalar row identities differ")
        selected = values.get("selected")
        grid = values.get("training_grid")
        if type(selected) is not dict or type(grid) is not dict:
            raise RuntimeError("primary-v2 senior-scalar typed objects differ")
        for field in expected:
            add(source, "selected", "selected", field, selected_rows[0][field], selected.get(field))
        add(
            source,
            "training_grid",
            "grid",
            "candidate_count",
            grid_rows[0]["train_worst_csd"],
            grid.get("candidate_count"),
        )
        covered.add(source)

    elif consumer_id == "control.static_age_lookup":
        source = "analysis-output/ml-contribution-audit/static_age_lookup_seed42_g30.json"
        old = _load_json_bytes(snapshots[source], label=f"protected {source}")
        summary = old.get("summary")
        if type(summary) is not dict:
            raise RuntimeError("protected static-age summary differs")
        policy_metrics = summary.get("policy_metrics")
        if type(policy_metrics) is not dict:
            raise RuntimeError("protected static-age policy metrics differ")
        new_policies = values.get("policy_rows")
        for policy, old_metrics in sorted(policy_metrics.items()):
            if type(old_metrics) is not dict:
                raise RuntimeError("protected static-age policy row differs")
            new = _one_record(
                new_policies,
                field="policy",
                value=policy,
                label="primary-v2 static-age policy",
            )
            for field in ("mean_csd", "welfare", "mean_exclusion"):
                add(source, "policy_rows", f"policy={policy}", field, old_metrics.get(field), new.get(field))
        new_contrasts = values.get("contrast_rows")
        for contrast in (
            "age_lookup_minus_static_60plus",
            "history_free_minus_age_lookup",
        ):
            old_contrast = summary.get(contrast)
            if type(old_contrast) is not dict:
                raise RuntimeError("protected static-age contrast differs")
            new = _one_record(
                new_contrasts,
                field="contrast",
                value=contrast,
                label="primary-v2 static-age contrast",
            )
            aliases = {
                "difference": "diff", "ci_low": "ci_lo", "ci_high": "ci_hi",
                "p_two_sided_exact_sign_flip": "p_two_sided", "wins": "wins",
                "ties": "ties", "n_series": "n",
            }
            for old_field, new_field in aliases.items():
                add(source, "contrast_rows", f"contrast={contrast}", new_field, old_contrast.get(old_field), new.get(new_field))
        fit = values.get("fit_record")
        if type(fit) is not dict:
            raise RuntimeError("primary-v2 static-age fit record differs")
        add(source, "fit_record", "family=static_age_lookup", "best_train_loss", summary.get("best_train_loss"), fit.get("best_train_loss"))
        rows.extend(
            _consumer_vector_rows(
                consumer_id=consumer_id,
                artifact_schema=artifact_schema,
                source_member=source,
                table="fit_record",
                coordinate="family=static_age_lookup",
                field="selected_weights",
                v1_values=summary.get("best_logits"),
                v2_values=fit.get("selected_weights"),
            )
        )
        covered.add(source)
        gap(source, "history", "optimizer trajectories are not a typed manuscript consumer")

    elif consumer_id.startswith("cross_district."):
        bound40 = consumer_id.endswith("bound40")
        source = (
            "results/iclr_cross_district_b40_manifest.json"
            if bound40
            else "results/iclr_cross_district_manifest.json"
        )
        old = _load_json_bytes(snapshots[source], label=f"protected {source}")
        for old_field, new_field in (
            ("n_folds", "n_folds"),
            ("n_series", "n_series"),
            ("n_policy_series_rows", "n_policy_series_rows"),
            ("expected_training_bound", "bound"),
        ):
            add(source, "manifest", "manifest", new_field, old.get(old_field), values.get(new_field))
        old_runs = old.get("runs")
        new_runs = values.get("fold_records")
        if type(old_runs) is not list:
            raise RuntimeError("protected cross-district run manifest differs")
        for old_run in old_runs:
            if type(old_run) is not dict:
                raise RuntimeError("protected cross-district run row differs")
            fold = old_run.get("fold")
            new_run = _one_record(new_runs, field="fold", value=fold, label="primary-v2 district fold")
            fit = new_run.get("fit")
            if type(fit) is not dict:
                raise RuntimeError("primary-v2 district fit record differs")
            add(source, "fold_records", f"fold={fold}", "best_train_loss", old_run.get("best_train_loss"), fit.get("best_train_loss"))
            rows.extend(
                _consumer_vector_rows(
                    consumer_id=consumer_id,
                    artifact_schema=artifact_schema,
                    source_member=source,
                    table="fold_records",
                    coordinate=f"fold={fold}",
                    field="selected_weights",
                    v1_values=old_run.get("best_weights"),
                    v2_values=fit.get("selected_weights"),
                )
            )
        covered.add(source)
        gap(source, "operational metadata", "paths, hashes, and run-time settings authenticate v1 but are not numerical outcomes")

    elif consumer_id == "failure.outcome":
        source = "results/iclr_worst_cohort_mix.json"
        old = _load_json_bytes(snapshots[source], label=f"protected {source}")
        mix = values.get("worst_cohort_mix")
        if type(mix) is not dict or set(old) != set(mix):
            raise RuntimeError("protected/v2 worst-cohort mix coverage differs")
        for name, rendered in sorted(old.items()):
            if type(rendered) is not str:
                raise RuntimeError("protected worst-cohort mix value differs")
            new = mix[name]
            numerator, denominator = parse_protected_ratio(rendered)
            if type(new) is not dict:
                raise RuntimeError("protected/v2 worst-cohort mix schema differs")
            add(source, "worst_cohort_mix", f"name={name}", "numerator", numerator, new.get("numerator"))
            add(source, "worst_cohort_mix", f"name={name}", "denominator", denominator, new.get("denominator"))
        covered.add(source)

    elif consumer_id in {"frontier.endowment", "frontier.outcome"}:
        for source in expected_paths:
            if not source.endswith(".json"):
                continue
            old = _load_json_bytes(snapshots[source], label=f"protected {source}")
            seed = old.get("seed")
            points = old.get("points")
            if type(points) is not list:
                raise RuntimeError("protected frontier points differ")
            for old_point in points:
                if type(old_point) is not dict:
                    raise RuntimeError("protected frontier point differs")
                floor = old_point.get("floor")
                new = next(
                    (
                        item
                        for item in values["point_rows"]
                        if item.get("seed") == seed and item.get("floor") == floor
                    ),
                    None,
                )
                if type(new) is not dict:
                    raise RuntimeError("protected/v2 frontier point coverage differs")
                coordinate = f"seed={seed}|floor={floor}"
                for field in (
                    "penalty", "train_worst_csd", "train_welfare_ratio",
                    "test_worst_csd", "test_mean_csd", "test_welfare",
                    "test_exclusion",
                ):
                    add(source, "point_rows", coordinate, field, old_point.get(field), new.get(field))
                rows.extend(
                    _consumer_vector_rows(
                        consumer_id=consumer_id,
                        artifact_schema=artifact_schema,
                        source_member=source,
                        table="point_rows",
                        coordinate=coordinate,
                        field="weights",
                        v1_values=old_point.get("weights"),
                        v2_values=new.get("weights"),
                    )
                )
            baselines = old.get("baselines")
            if type(baselines) is not dict:
                raise RuntimeError("protected frontier baseline table differs")
            for key, old_baseline in sorted(baselines.items()):
                if type(old_baseline) is not dict or "/" not in key:
                    raise RuntimeError("protected frontier baseline row differs")
                policy, view = key.rsplit("/", 1)
                new = next(
                    (
                        item
                        for item in values["baseline_rows"]
                        if item.get("policy") == policy and item.get("view") == view
                    ),
                    None,
                )
                if type(new) is not dict:
                    raise RuntimeError("protected/v2 frontier baseline coverage differs")
                for field in (
                    "n_series", "worst_csd", "mean_csd", "welfare",
                    "cost_welfare", "exclusion",
                ):
                    add(source, "baseline_rows", f"policy={policy}|view={view}", field, old_baseline.get(field), new.get(field))
            covered.add(source)
            gap(source, "elapsed_sec", "runtime is machine-dependent and is not a semantic v2 consumer value")

    elif consumer_id == "mechanism.attribution_coverage":
        source = "results/iclr_attribution_coverage.csv"
        old_rows = _load_csv_bytes(snapshots[source], label=f"protected {source}")
        old = _one_record(
            old_rows,
            field="policy",
            value="series_clean_for_mes_and_learned_endowment",
            label="protected attribution joint-clean row",
        )
        joint = values.get("joint_clean_series")
        if type(joint) is not dict:
            raise RuntimeError("primary-v2 attribution joint-clean summary differs")
        add(source, "joint_clean_series", "policies=mes+learned-endowment", "n_series", old.get("scored_editions"), joint.get("n_series"))

    elif consumer_id == "mechanism.payment_kernel":
        source = "results/iclr_payment_intervention_manifest.json"
        old = _load_json_bytes(snapshots[source], label=f"protected {source}")
        metadata = values.get("analysis_metadata")
        if type(metadata) is not dict:
            raise RuntimeError("primary-v2 payment metadata differs")
        add(source, "analysis_metadata", "manifest", "floor_count", len(old.get("floors", [])), values.get("floor_count"))
        add(source, "analysis_metadata", "manifest", "kernel_count", 4, values.get("kernel_count"))
        series_rows = values.get("series_kernel_rows")
        if type(series_rows) is not list:
            raise RuntimeError("primary-v2 payment series table differs")
        add(source, "analysis_metadata", "manifest", "n_series", old.get("n_series"), len({row["series"] for row in series_rows}))
        for field in ("bootstrap_replicates", "bootstrap_seed"):
            add(source, "analysis_metadata", "manifest", field, old.get(field), metadata.get(field))
        covered.add(source)
        gap(source, "source_sha256", "the v1 source hash authenticates lineage but has no numerical v2 outcome counterpart")

    elif consumer_id == "outcome.seed_stability":
        for source in expected_paths:
            old = _load_json_bytes(snapshots[source], label=f"protected {source}")
            seed = old.get("seed")
            points = old.get("points")
            if type(points) is not list:
                raise RuntimeError("protected outcome seed frontier differs")
            selected = [point for point in points if type(point) is dict and point.get("floor") == 1.0]
            if len(selected) != 1:
                raise RuntimeError("protected outcome seed headline point differs")
            new = _one_record(values["seed_rows"], field="seed", value=seed, label="primary-v2 outcome seed")
            for field in (
                "test_worst_csd", "test_mean_csd", "test_welfare", "test_exclusion",
            ):
                add(source, "seed_rows", f"seed={seed}", field, selected[0].get(field), new.get(field))
            rows.extend(
                _consumer_vector_rows(
                    consumer_id=consumer_id,
                    artifact_schema=artifact_schema,
                    source_member=source,
                    table="seed_rows",
                    coordinate=f"seed={seed}",
                    field="weights",
                    v1_values=selected[0].get("weights"),
                    v2_values=new.get("weights"),
                )
            )
            covered.add(source)

    elif consumer_id == "primary.corpus":
        source = "results/frozen_instances.txt"
        try:
            lines = snapshots[source].decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise RuntimeError("protected corpus inventory is not UTF-8") from exc
        old_coordinates: set[tuple[str, str, int]] = set()
        for line in lines:
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                raise RuntimeError("protected corpus inventory row schema differs")
            try:
                year = int(fields[2])
            except ValueError as exc:
                raise RuntimeError("protected corpus inventory year differs") from exc
            old_coordinates.add((fields[0], fields[1], year))
        new_rows = values.get("instance_rows")
        if type(new_rows) is not list:
            raise RuntimeError("primary-v2 corpus instance rows differ")
        new_coordinates = {
            (str(row["file"]), str(row["series"]), int(row["year"]))
            for row in new_rows
        }
        if len(old_coordinates) != len(new_coordinates) or old_coordinates != new_coordinates:
            raise RuntimeError("protected/v2 corpus instance coordinate coverage differs")
        add(source, "instance_rows", "inventory", "row_count", len(old_coordinates), len(new_coordinates))
        covered.add(source)
        granularity_source = "results/iclr_instance_granularity.csv"
        old_granularity = _load_csv_bytes(
            snapshots[granularity_source], label=f"protected {granularity_source}"
        )
        excluded = _one_record(
            old_granularity,
            field="corpus",
            value="external_projected",
            label="protected excluded corpus row",
        )
        gap(
            granularity_source,
            "external_projected numerical row",
            (
                "explicitly outside locked primary-Warsaw v2 corpus; protected values "
                f"were {excluded.get('scored_elections')}/"
                f"{excluded.get('median_projects')}/{excluded.get('mean_projects')}"
            ),
        )

    elif consumer_id == "summary.endowment_seed_stability":
        for source in expected_paths:
            old = _load_json_bytes(snapshots[source], label=f"protected {source}")
            config = old.get("config")
            learned = old.get("learned")
            if type(config) is not dict or type(learned) is not dict:
                raise RuntimeError("protected endowment seed record differs")
            seed = config.get("seed")
            new = _one_record(values["seed_rows"], field="seed", value=seed, label="primary-v2 endowment seed")
            train = learned.get("train")
            test = learned.get("test")
            if type(train) is not dict or type(test) is not dict:
                raise RuntimeError("protected endowment seed summaries differ")
            add(source, "seed_rows", f"seed={seed}", "train_worst_csd", train.get("worst_csd"), new.get("train_worst_csd"))
            for old_field, new_field in (
                ("worst_csd", "test_worst_csd"),
                ("mean_csd", "test_mean_csd"),
                ("welfare", "test_welfare"),
                ("cost_welfare", "test_cost_welfare"),
                ("exclusion", "test_exclusion"),
            ):
                add(source, "seed_rows", f"seed={seed}", new_field, test.get(old_field), new.get(new_field))
            add(source, "seed_rows", f"seed={seed}", "best_train_loss", old.get("best_train_loss"), new.get("best_train_loss"))
            rows.extend(
                _consumer_vector_rows(
                    consumer_id=consumer_id,
                    artifact_schema=artifact_schema,
                    source_member=source,
                    table="seed_rows",
                    coordinate=f"seed={seed}",
                    field="weights",
                    v1_values=old.get("best_weights"),
                    v2_values=new.get("weights"),
                )
            )
            covered.add(source)
            gap(source, "elapsed_sec/history", "runtime and optimizer trajectories are not typed manuscript outcomes")

    return rows, covered, unavailable


def validate_consumer_comparison_registry() -> dict[str, int]:
    """Prove that all 25 consumers and all 55 legacy path associations are mapped."""

    expected_contract = evaluate_v2.PAPER_CONSUMER_CONTRACT_V2
    expected_ids = set(expected_contract)
    registered_ids = set(CONSUMER_CSV_COMPARISON_SPECS) | set(
        CONSUMER_SPECIAL_COMPARISON_PATHS
    )
    if registered_ids != expected_ids:
        raise RuntimeError("protected consumer comparison registry coverage differs")
    association_count = 0
    for consumer_id, contract in expected_contract.items():
        expected_paths = set(contract["legacy_paths"])
        direct_paths = {
            str(spec["source_member"])
            for spec in CONSUMER_CSV_COMPARISON_SPECS.get(consumer_id, ())
        }
        special_paths = set(
            CONSUMER_SPECIAL_COMPARISON_PATHS.get(consumer_id, ())
        )
        if direct_paths | special_paths != expected_paths:
            raise RuntimeError(
                f"protected consumer artifact registry coverage differs: {consumer_id}"
            )
        if not direct_paths <= expected_paths or not special_paths <= expected_paths:
            raise RuntimeError(
                f"protected consumer registry names an unknown path: {consumer_id}"
            )
        association_count += len(expected_paths)
    return {
        "consumer_count": len(expected_ids),
        "legacy_path_association_count": association_count,
    }


def compare_protected_consumers(
    v1_snapshots: object, consumers_payload: object
) -> tuple[
    dict[str, int], list[dict[str, object]], list[dict[str, object]]
]:
    """Numerically compare each authenticated legacy consumer to its typed v2 table."""

    registry = validate_consumer_comparison_registry()
    if registry["consumer_count"] != len(evaluate_v2.PAPER_CONSUMER_CONTRACT_V2):
        raise RuntimeError("protected consumer comparison registry count differs")
    if type(v1_snapshots) is not dict or any(
        type(path) is not str or type(content) is not bytes
        for path, content in v1_snapshots.items()
    ):
        raise RuntimeError("protected consumer snapshots differ")
    expected_ids = sorted(evaluate_v2.PAPER_CONSUMER_CONTRACT_V2)
    if (
        type(consumers_payload) is not dict
        or consumers_payload.get("consumer_ids") != expected_ids
        or type(consumers_payload.get("consumers")) is not dict
        or set(consumers_payload["consumers"]) != set(expected_ids)
    ):
        raise RuntimeError("primary-v2 paper consumer coverage differs")

    numeric_rows: list[dict[str, object]] = []
    consumer_rows: list[dict[str, object]] = []
    for consumer_id in expected_ids:
        legacy_contract = evaluate_v2.PAPER_CONSUMER_CONTRACT_V2[consumer_id]
        consumer = consumers_payload["consumers"][consumer_id]
        expected_paths = list(legacy_contract["legacy_paths"])
        if (
            type(consumer) is not dict
            or consumer.get("consumer_id") != consumer_id
            or consumer.get("legacy_paths") != expected_paths
        ):
            raise RuntimeError(
                f"primary-v2 paper consumer source coverage differs: {consumer_id}"
            )
        missing_snapshots = [path for path in expected_paths if path not in v1_snapshots]
        if missing_snapshots:
            raise RuntimeError(
                f"protected consumer snapshots are incomplete: {consumer_id}/{missing_snapshots}"
            )
        values = _validate_typed_consumer_values(consumer_id, consumer.get("values"))
        schema = str(values["schema"])
        specs = CONSUMER_CSV_COMPARISON_SPECS.get(consumer_id, ())
        covered_paths: set[str] = set()
        before = len(numeric_rows)
        for spec in specs:
            source_member = str(spec["source_member"])
            if source_member not in expected_paths:
                raise RuntimeError(
                    f"consumer comparison names an unregistered legacy path: {consumer_id}"
                )
            covered_paths.add(source_member)
            numeric_rows.extend(
                _compare_consumer_csv_table(
                    consumer_id=consumer_id,
                    artifact_schema=schema,
                    spec=spec,
                    snapshot=v1_snapshots[source_member],
                    values=values,
                )
            )
        special_rows, special_paths, unavailable = _special_consumer_comparisons(
            consumer_id=consumer_id,
            artifact_schema=schema,
            expected_paths=expected_paths,
            snapshots=v1_snapshots,
            values=values,
        )
        numeric_rows.extend(special_rows)
        covered_paths.update(special_paths)
        if covered_paths != set(expected_paths):
            raise RuntimeError(
                f"protected consumer artifact coverage differs: {consumer_id}; "
                f"covered={sorted(covered_paths)}, expected={sorted(expected_paths)}"
            )
        consumer_rows.append(
            {
                "consumer_id": consumer_id,
                "artifact_schema": schema,
                "status": "compared",
                "legacy_paths": expected_paths,
                "numeric_coordinates": len(numeric_rows) - before,
                "reason": "exact_typed_specialized_v1_v2_reconstruction",
                "unavailable_fields": unavailable,
            }
        )
    coordinates = [
        (
            row["consumer_id"],
            row["source_member"],
            row["table"],
            row["coordinate"],
            row["field"],
        )
        for row in numeric_rows
    ]
    if len(coordinates) != len(set(coordinates)):
        raise RuntimeError("protected consumer numeric coordinates contain duplicates")
    numeric_rows.sort(
        key=lambda row: (
            str(row["consumer_id"]),
            str(row["source_member"]),
            str(row["table"]),
            str(row["coordinate"]),
            str(row["field"]),
        )
    )
    changed = sum(row["changed"] is True for row in numeric_rows)
    return {
        "consumer_total": len(consumer_rows),
        "consumer_compared": len(consumer_rows),
        "numeric_coordinates": len(numeric_rows),
        "numeric_changed": changed,
        "numeric_exact": len(numeric_rows) - changed,
    }, numeric_rows, consumer_rows


def _safe_zip_members(archive: zipfile.ZipFile, *, label: str) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = _safe_member_name(info.filename, label=f"{label} member")
        if name in members:
            raise RuntimeError(f"{label} contains a duplicate member: {name}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"{label} contains a symbolic-link member: {name}")
        members[name] = info
    return members


def _read_regular(path: Path, *, label: str) -> bytes:
    content = protocol_v2.read_regular_bytes_artifact_v2(path, label=label)
    if content is None:
        raise RuntimeError(f"{label} is missing")
    return content


def _protected_path(repo_root: Path, relative: Path) -> Path:
    return repo_root.resolve().parent / relative


def verify_protected_v1_release(
    repo_root: Path, required_members: Sequence[str]
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Hash-authenticate the frozen paper, both archives, and mapped v1 bytes."""

    root = Path(repo_root).resolve()
    pdf = _read_regular(
        _protected_path(root, PROTECTED_PDF_RELATIVE), label="protected paper PDF"
    )
    repro = _read_regular(
        _protected_path(root, PROTECTED_REPRO_RELATIVE),
        label="protected reproducibility ZIP",
    )
    source = _read_regular(
        _protected_path(root, PROTECTED_SOURCE_RELATIVE), label="protected source ZIP"
    )
    actual = (_sha256(pdf), _sha256(repro), _sha256(source))
    expected = (
        PROTECTED_PDF_SHA256,
        PROTECTED_REPRO_SHA256,
        PROTECTED_SOURCE_SHA256,
    )
    if actual != expected:
        raise RuntimeError(
            "protected v1 release authentication failed: "
            f"actual={actual}, expected={expected}"
        )
    member_names = list(required_members)
    if member_names != sorted(set(member_names)):
        raise RuntimeError("required v1 member list must be unique and sorted")
    for name in member_names:
        _safe_member_name(name, label="required v1 member")

    snapshots: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(repro), "r") as archive:
            members = _safe_zip_members(archive, label="protected reproducibility ZIP")
            for name in member_names:
                if name not in members:
                    raise RuntimeError(f"protected reproducibility ZIP missing {name}")
                released = archive.read(members[name])
                local = _read_regular(root / name, label=f"local protected-v1 member {name}")
                if released != local:
                    raise RuntimeError(f"local v1 member differs from protected ZIP: {name}")
                snapshots[name] = released
        with zipfile.ZipFile(io.BytesIO(source), "r") as archive:
            members = _safe_zip_members(archive, label="protected source ZIP")
            if PROTECTED_NUMBERS_MEMBER not in members:
                raise RuntimeError("protected source ZIP is missing numbers.tex")
            numbers = archive.read(members[PROTECTED_NUMBERS_MEMBER])
    except zipfile.BadZipFile as exc:
        raise RuntimeError("protected release contains an invalid ZIP") from exc
    if _sha256(numbers) != PROTECTED_NUMBERS_SHA256:
        raise RuntimeError("protected numbers.tex SHA-256 differs")
    local_numbers = _read_regular(
        root.parent / "iclr_paper/tex/numbers.tex", label="local protected numbers.tex"
    )
    if numbers != local_numbers:
        raise RuntimeError("local numbers.tex differs from protected source ZIP")
    verified = {
        "pdf_sha256": actual[0],
        "pdf_size_bytes": len(pdf),
        "reproducibility_zip_sha256": actual[1],
        "reproducibility_zip_size_bytes": len(repro),
        "source_zip_sha256": actual[2],
        "source_zip_size_bytes": len(source),
        "numbers_tex_sha256": _sha256(numbers),
        "verified_member_count": len(snapshots),
        "verified_members": {
            name: {"sha256": _sha256(content), "size_bytes": len(content)}
            for name, content in sorted(snapshots.items())
        },
    }
    return verified, snapshots


def serialize_delta_csv(rows: object) -> str:
    if type(rows) is not list:
        raise RuntimeError("delta CSV rows must be a list")
    if not rows:
        raise RuntimeError("delta CSV rows must not be empty")
    first = rows[0]
    if type(first) is not dict:
        raise RuntimeError("delta CSV row schema differs")
    schema = tuple(first)
    if schema not in (DELTA_CSV_FIELDS, METRIC_DELTA_CSV_FIELDS):
        raise RuntimeError("delta CSV row schema differs")
    validated: list[dict[str, object]] = []
    for row in rows:
        if type(row) is not dict or tuple(row) != schema:
            raise RuntimeError("delta CSV row schema differs")
        validated.append(row)
    if schema == DELTA_CSV_FIELDS:
        coordinates = [
            (str(row["fit_id"]), str(row["split"]), str(row["series"]))
            for row in validated
        ]
        sort_key = lambda item: (
            str(item["fit_id"]), str(item["split"]), str(item["series"])
        )
    else:
        coordinates = [
            (str(row["fit_id"]), str(row["metric"])) for row in validated
        ]
        for row in validated:
            if (
                row["metric"] not in METRIC_FIELDS
                or row["status"]
                not in {"comparable", "not_recorded_in_protected_release"}
                or type(row["v2_value"]) is not float
                or not math.isfinite(row["v2_value"])
            ):
                raise RuntimeError("delta CSV metric row schema differs")
            comparable = row["status"] == "comparable"
            if comparable:
                if (
                    type(row["v1_value"]) is not float
                    or type(row["delta_v2_minus_v1"]) is not float
                    or type(row["changed"]) is not bool
                ):
                    raise RuntimeError("delta CSV comparable metric row differs")
            elif any(
                row[field] is not None
                for field in ("v1_value", "delta_v2_minus_v1", "changed")
            ):
                raise RuntimeError("delta CSV unavailable metric row differs")
        sort_key = lambda item: (str(item["fit_id"]), str(item["metric"]))
    if len(coordinates) != len(set(coordinates)):
        raise RuntimeError("delta CSV contains a duplicate coordinate")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=schema, lineterminator="\n")
    writer.writeheader()
    for row in sorted(validated, key=sort_key):
        writer.writerow(row)
    return output.getvalue()


def assemble_delta_payload(
    *,
    release_audit: object,
    authenticated_inputs: object,
    semantic_correction: object,
    fit_audit: object,
    evaluation_audit: object,
    consumer_audit: object,
    manuscript_audit: object,
    csv_sha256: str,
) -> dict[str, object]:
    for label, value in (
        ("release_audit", release_audit),
        ("authenticated_inputs", authenticated_inputs),
        ("semantic_correction", semantic_correction),
        ("fit_audit", fit_audit),
        ("evaluation_audit", evaluation_audit),
        ("consumer_audit", consumer_audit),
        ("manuscript_audit", manuscript_audit),
    ):
        if type(value) is not dict:
            raise RuntimeError(f"{label} must be a JSON object")
    if not _is_sha256(csv_sha256):
        raise RuntimeError("delta CSV SHA-256 differs")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "interpretation": "complete_corrected_lineage_rerun",
        "causal_attribution": {
            "duplicate_only_attribution_allowed": False,
            "reason": CAUSAL_ATTRIBUTION_REASON,
        },
        "release_audit": release_audit,
        "authenticated_inputs": authenticated_inputs,
        "semantic_correction": semantic_correction,
        "fit_audit": fit_audit,
        "evaluation_audit": evaluation_audit,
        "consumer_audit": consumer_audit,
        "manuscript_audit": manuscript_audit,
        "delta_csv": {"relative_path": CSV_OUTPUT.as_posix(), "sha256": csv_sha256},
    }
    _validate_json_tree(payload, label="delta audit payload")
    return payload


def _json_bytes(value: object) -> bytes:
    _validate_json_tree(value, label="JSON output")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _summarize_fit_rows(
    rows: object, *, expected_fit_ids: Sequence[str]
) -> dict[str, object]:
    if type(rows) is not list or len(rows) != len(expected_fit_ids):
        raise RuntimeError("fit delta row count differs")
    if any(type(row) is not dict for row in rows):
        raise RuntimeError("fit delta row schema differs")
    by_id = {str(row.get("fit_id")): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected_fit_ids):
        raise RuntimeError("fit delta coordinate coverage differs")
    ordered = [by_id[fit_id] for fit_id in sorted(expected_fit_ids)]
    comparable_weights = [row for row in ordered if row.get("weights_comparable") is True]
    unavailable_weights = [row for row in ordered if row.get("weights_comparable") is False]
    if len(comparable_weights) + len(unavailable_weights) != len(ordered):
        raise RuntimeError("fit weight comparison status differs")
    for row in comparable_weights:
        if (
            type(row.get("weights_exact")) is not bool
            or type(row.get("weight_linf_delta")) is not float
            or not math.isfinite(row["weight_linf_delta"])
        ):
            raise RuntimeError("comparable fit weight delta differs")
    for row in unavailable_weights:
        if row.get("weights_exact") is not None or row.get("weight_linf_delta") is not None:
            raise RuntimeError("unavailable fit weight delta must remain null")
    comparable_objectives = [
        row for row in ordered if row.get("train_objective_delta_v2_minus_v1") is not None
    ]
    for row in comparable_objectives:
        delta = row["train_objective_delta_v2_minus_v1"]
        if type(delta) is not float or not math.isfinite(delta):
            raise RuntimeError("fit objective delta differs")
    return {
        "total": len(ordered),
        "weights_comparable": len(comparable_weights),
        "weights_unavailable": len(unavailable_weights),
        "weights_exact": sum(int(row["weights_exact"] is True) for row in comparable_weights),
        "weights_changed": sum(int(row["weights_exact"] is False) for row in comparable_weights),
        "max_weight_linf_delta": max(
            (float(row["weight_linf_delta"]) for row in comparable_weights),
            default=0.0,
        ),
        "train_objectives_comparable": len(comparable_objectives),
        "train_objectives_unavailable": len(ordered) - len(comparable_objectives),
        "train_objectives_changed": sum(
            int(row["train_objective_delta_v2_minus_v1"] != 0.0)
            for row in comparable_objectives
        ),
        "max_absolute_train_objective_delta": max(
            (
                abs(float(row["train_objective_delta_v2_minus_v1"]))
                for row in comparable_objectives
            ),
            default=0.0,
        ),
        "rows": ordered,
    }


def _required_v1_lineage_members(mapping: Sequence[Mapping[str, object]]) -> list[str]:
    members = {str(row["source_member"]) for row in mapping}
    for consumer_id, contract in evaluate_v2.PAPER_CONSUMER_CONTRACT_V2.items():
        if type(consumer_id) is not str or type(contract) is not dict or set(contract) != {"legacy_paths"}:
            raise RuntimeError("primary-v2 consumer legacy-path contract differs")
        paths = contract["legacy_paths"]
        if type(paths) is not tuple or not paths:
            raise RuntimeError("primary-v2 consumer legacy-path inventory differs")
        for path in paths:
            members.add(_safe_member_name(path, label=f"legacy path for {consumer_id}"))
    return sorted(members)


def build_delta_audit(repo_root: Path = ROOT) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Independently authenticate both lineages and construct the complete audit."""

    root = Path(repo_root).resolve()
    inventory = protocol_v2.locked_fit_inventory_v2()
    mapping = validate_v1_fit_mapping(v1_fit_mapping(), inventory)
    expected_fit_ids = [str(row["fit_id"]) for row in mapping]
    required_members = _required_v1_lineage_members(mapping)
    release, v1_snapshots = verify_protected_v1_release(root, required_members)
    if (
        release.get("verified_member_count") != len(required_members)
        or set(v1_snapshots) != set(required_members)
    ):
        raise RuntimeError("protected v1 lineage member coverage differs")
    v1_records = [
        extract_v1_fit_record(row, v1_snapshots[str(row["source_member"])])
        for row in mapping
    ]

    matrix, v2_payloads = authenticate_complete_v2_matrix(root)
    if set(v2_payloads) != set(expected_fit_ids):
        raise RuntimeError("authenticated primary-v2 fit payload coverage differs")
    fit_rows = [
        compare_fit_record(
            v1_record,
            v2_payloads[str(v1_record["fit_id"])],
            v2_sha256=matrix["fit_sha256"][str(v1_record["fit_id"])],
        )
        for v1_record in v1_records
    ]
    fit_audit = _summarize_fit_rows(fit_rows, expected_fit_ids=expected_fit_ids)
    fit_audit["explicit_unserialized_fit_gaps"] = explicit_legacy_fit_gaps(
        [row for row in fit_rows if row["weights_comparable"] is False]
    )

    evaluation_auth, normalized_rows, consumers = authenticate_v2_evaluation_commit(
        root, matrix
    )
    evaluation_summary, metric_rows, winner_rows = compare_protected_evaluation(
        v1_records,
        normalized_rows,
        expected_fit_ids=expected_fit_ids,
    )
    evaluation_audit = {
        **evaluation_summary,
        "winner_changed_rows": None,
        "metric_rows": metric_rows,
        "winner_rows": winner_rows,
    }
    consumer_summary, consumer_numeric_rows, consumer_rows = (
        compare_protected_consumers(v1_snapshots, consumers)
    )
    consumer_audit = {
        **consumer_summary,
        "consumers": consumer_rows,
        "numeric_rows": consumer_numeric_rows,
    }

    protected_numbers = _read_regular(
        root.parent / "iclr_paper" / "tex" / "numbers.tex",
        label="protected manuscript numbers.tex",
    )
    if _sha256(protected_numbers) != PROTECTED_NUMBERS_SHA256:
        raise RuntimeError("protected manuscript numbers.tex changed after release verification")
    protected_macros = parse_numbers_tex(protected_numbers)
    v2_macros = consumers.get("manuscript_macros")
    if type(v2_macros) is not dict or not v2_macros:
        raise RuntimeError("primary-v2 manuscript macro mapping differs")
    provenance = consumers.get("provenance")
    macro_coordinates = (
        provenance.get("macro_coordinates") if type(provenance) is dict else None
    )
    macro_summary, macro_rows = classify_protected_manuscript_values(
        protected_macros, v2_macros, macro_coordinates
    )
    manuscript_audit = {
        **macro_summary,
        "rows": macro_rows,
    }

    csv_bytes = serialize_delta_csv(metric_rows).encode("utf-8")
    release_audit = {
        **release,
        "required_lineage_member_count": len(required_members),
        "fit_mapping_count": len(mapping),
        "paper_consumer_count": len(evaluate_v2.PAPER_CONSUMER_CONTRACT_V2),
    }
    authenticated_inputs = {
        "v2_protocol_lock_sha256": matrix["protocol_lock_sha256"],
        "v2_training_grid_sha256": matrix["grid_sha256"],
        "v2_fit_count": matrix["fit_count"],
        "v2_fit_sha256": matrix["fit_sha256"],
        "v2_evaluation": evaluation_auth,
    }
    semantic_correction = {
        "semantics_profile": "approval-set-first-occurrence-v2",
        "approval_set_normalization_changed": True,
        "deterministic_project_ordering_changed": True,
        "counterfactual_isolation": "not_identified",
    }
    payload = assemble_delta_payload(
        release_audit=release_audit,
        authenticated_inputs=authenticated_inputs,
        semantic_correction=semantic_correction,
        fit_audit=fit_audit,
        evaluation_audit=evaluation_audit,
        consumer_audit=consumer_audit,
        manuscript_audit=manuscript_audit,
        csv_sha256=_sha256(csv_bytes),
    )
    return payload, metric_rows


def _install_audit_pair(
    *,
    json_path: Path,
    json_bytes: bytes,
    csv_path: Path,
    csv_bytes: bytes,
) -> tuple[str, str]:
    csv_sha = protocol_v2.write_immutable_bytes_artifact_v2(
        csv_path, csv_bytes, label="primary-Warsaw v1/v2 delta CSV"
    )
    json_sha = protocol_v2.write_immutable_bytes_artifact_v2(
        json_path, json_bytes, label="primary-Warsaw v1/v2 delta sealing JSON"
    )
    return json_sha, csv_sha


def write_delta_audit(repo_root: Path = ROOT) -> dict[str, object]:
    """Recompute twice, preflight both outputs, then commit CSV before JSON."""

    root = Path(repo_root).resolve()
    payload, rows = build_delta_audit(root)
    csv_bytes = serialize_delta_csv(rows).encode("utf-8")
    payload = dict(payload)
    if type(payload.get("delta_csv")) is dict:
        if payload["delta_csv"].get("sha256") != _sha256(csv_bytes):
            raise RuntimeError("built payload does not bind the serialized delta CSV")
    json_bytes = _json_bytes(payload)

    replay_payload, replay_rows = build_delta_audit(root)
    replay_csv = serialize_delta_csv(replay_rows).encode("utf-8")
    replay_json = _json_bytes(replay_payload)
    if replay_csv != csv_bytes or replay_json != json_bytes:
        raise RuntimeError("independent delta-audit recomputation changed output bytes")

    output_root = root / V2_RESULT_RELATIVE
    csv_path = output_root / CSV_OUTPUT
    json_path = output_root / JSON_OUTPUT
    protocol_v2.preflight_immutable_bytes_artifact_v2(
        csv_path, csv_bytes, label="primary-Warsaw v1/v2 delta CSV"
    )
    protocol_v2.preflight_immutable_bytes_artifact_v2(
        json_path, json_bytes, label="primary-Warsaw v1/v2 delta sealing JSON"
    )
    json_sha, csv_sha = _install_audit_pair(
        json_path=json_path,
        json_bytes=json_bytes,
        csv_path=csv_path,
        csv_bytes=csv_bytes,
    )
    fit_audit = payload.get("fit_audit")
    evaluation_audit = payload.get("evaluation_audit")
    consumer_audit = payload.get("consumer_audit")
    manuscript_audit = payload.get("manuscript_audit")
    if not all(
        type(value) is dict
        for value in (fit_audit, evaluation_audit, consumer_audit, manuscript_audit)
    ):
        raise RuntimeError("delta audit summary schema differs")
    return {
        "status": payload.get("status"),
        "fit_count": fit_audit.get("total"),
        "evaluation_rows": evaluation_audit.get("total_rows"),
        "winner_changed_rows": evaluation_audit.get("winner_changed_rows"),
        "consumer_numeric_coordinates": consumer_audit.get("numeric_coordinates"),
        "manuscript_value_count": manuscript_audit.get("protected_macro_total"),
        "json_sha256": json_sha,
        "csv_sha256": csv_sha,
    }


def check_delta_audit(repo_root: Path = ROOT) -> dict[str, object]:
    """Recompute and compare against the immutable committed pair without writes."""

    root = Path(repo_root).resolve()
    payload, rows = build_delta_audit(root)
    expected_csv = serialize_delta_csv(rows).encode("utf-8")
    expected_json = _json_bytes(payload)
    output_root = root / V2_RESULT_RELATIVE
    actual_csv = _read_regular(output_root / CSV_OUTPUT, label="committed delta CSV")
    actual_json = _read_regular(output_root / JSON_OUTPUT, label="committed delta JSON")
    if actual_csv != expected_csv or actual_json != expected_json:
        raise RuntimeError("committed delta audit differs from independent recomputation")
    return {
        "status": "pass",
        "json_sha256": _sha256(actual_json),
        "csv_sha256": _sha256(actual_csv),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="immutably install the audit")
    parser.add_argument(
        "--check-only", action="store_true", help="verify the already committed audit"
    )
    args = parser.parse_args(argv)
    if args.write == args.check_only:
        parser.error("choose exactly one of --write or --check-only")
    result = write_delta_audit(ROOT) if args.write else check_delta_audit(ROOT)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
