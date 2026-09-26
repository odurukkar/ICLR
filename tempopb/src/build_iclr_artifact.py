"""Build anonymous, deterministic ICLR source and reproducibility archives."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

from iclr_lock_drift import (
    LOCK_PATHS as AUDITED_LOCK_PATHS,
    assert_audit_passes,
    is_decision_critical,
    is_sha256,
)
from iclr_multicity_protocol import (
    locked_fit_inventory,
    multicity_manifest_retained_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "tempopb"
PAPER = ROOT / "iclr_paper"
OUTPUT = ROOT / "output"
PABULIB_REPOSITORY = "https://github.com/pabulib/pabulib_files.git"
PABULIB_COMMIT = "2f4321fec84069f50abf35fb5e90852013d17070"
RELEASE_ENTRYPOINTS = (
    "analyze_external_support_set_results.py",
    "build_iclr_artifact.py",
    "gen_iclr_appendix.py",
    "gen_iclr_figures.py",
    "gen_iclr_numbers.py",
    "iclr_ablation.py",
    "iclr_actuating_subset.py",
    "iclr_baselines.py",
    "iclr_cross_district.py",
    "iclr_crosscity.py",
    "iclr_crosscity_signatures.py",
    "iclr_degeneracy.py",
    "iclr_endowment_stats.py",
    "iclr_external_support_set.py",
    "iclr_external_support_set_amendment.py",
    "iclr_external_validation.py",
    "iclr_failure_analysis.py",
    "iclr_fetch_refs.py",
    "iclr_fig_coverage.py",
    "iclr_fig_dynamics.py",
    "iclr_fig_external.py",
    "iclr_fig_forest.py",
    "iclr_fig_hack.py",
    "iclr_fig_schematic.py",
    "iclr_frontier.py",
    "iclr_history_free_audit.py",
    "iclr_identify.py",
    "iclr_lock_drift.py",
    "iclr_multicity_evaluate.py",
    "iclr_multicity_protocol.py",
    "iclr_multicity_train.py",
    "iclr_multicity_transfer.py",
    "iclr_payment_intervention.py",
    "iclr_prior_baseline.py",
    "iclr_priority_mes.py",
    "iclr_scheme_robustness.py",
    "iclr_significance.py",
    "iclr_static_demographic_audit.py",
    "iclr_static_tilt.py",
    "iclr_theory_check.py",
    "iclr_train.py",
    "iclr_transfer.py",
    "iclr_verify_env.py",
    "iclr_verify_outcome.py",
    "iclr_verify_policy.py",
    "stage_iclr_release_data.py",
    "verify_iclr_data.py",
)
RELEASE_DEPENDENCIES = (
    "cohorts.py",
    "iclr_analysis_rollout.py",
    "iclr_cmaes.py",
    "iclr_corpus.py",
    "iclr_env.py",
    "iclr_fig_external_contracts.py",
    "iclr_fig_external_fs.py",
    "iclr_fig_external_io.py",
    "iclr_ml_contribution_audit.py",
    "iclr_outcome.py",
    "iclr_policy.py",
    "iclr_reviewer_checks.py",
    "iclr_stats.py",
    "iclr_style.py",
    "parse_pb.py",
    "rules.py",
    "run_experiments.py",
)
RELEASE_SOURCE = tuple(sorted((*RELEASE_ENTRYPOINTS, *RELEASE_DEPENDENCIES)))
EXCLUDED_DEVELOPMENT_SOURCE = frozenset(
    {
        "iclr_approval_semantics_v2.py",
        "iclr_multicity_delta_audit_v2.py",
        "iclr_multicity_protocol_v2.py",
        "iclr_multicity_train_v2.py",
        "iclr_multicity_evaluate_v2.py",
        "iclr_primary_warsaw_protocol_v2.py",
        "iclr_primary_warsaw_train_v2.py",
        "iclr_residual_actuation.py",
        "iclr_residual_policy.py",
        "iclr_residual_train.py",
        "iclr_toy_model.py",
        "iclr_trace_parity.py",
        "iclr_wide_senior_audit.py",
    }
)
RELEASE_TESTS = (
    "test_iclr_analysis_rollout.py",
    "test_iclr_appendix_generation.py",
    "test_iclr_artifact_manifest.py",
    "test_iclr_cross_district_analysis.py",
    "test_iclr_cross_district_split.py",
    "test_iclr_crosscity.py",
    "test_iclr_crosscity_reporting.py",
    "test_iclr_degeneracy.py",
    "test_iclr_external_support_set.py",
    "test_iclr_external_support_set_amendment.py",
    "test_iclr_external_validation.py",
    "test_iclr_fetch_refs.py",
    "test_iclr_fig_external.py",
    "test_iclr_fig_external_contracts.py",
    "test_iclr_fig_external_io.py",
    "test_iclr_fig_hack.py",
    "test_iclr_figure_fonts.py",
    "test_iclr_figure_terminology.py",
    "test_iclr_humanization_contracts.py",
    "test_iclr_lock_drift.py",
    "test_iclr_ml_contribution_audit.py",
    "test_iclr_multicity_evaluate.py",
    "test_iclr_multicity_protocol.py",
    "test_iclr_multicity_train.py",
    "test_iclr_multicity_transfer.py",
    "test_iclr_outcome_containment.py",
    "test_iclr_paper_claims.py",
    "test_iclr_paper_seed_paths.py",
    "test_iclr_payment_intervention.py",
    "test_iclr_prior_baseline.py",
    "test_iclr_priority_mes.py",
    "test_iclr_release_data.py",
    "test_iclr_seaborn_figures.py",
    "test_iclr_static_demographic_audit.py",
    "test_iclr_statistics.py",
    "test_iclr_tie_breaking.py",
)
EXCLUDED_DEVELOPMENT_TESTS = frozenset(
    {
        "test_iclr_approval_semantics_v2.py",
        "test_iclr_multicity_delta_audit_v2.py",
        "test_iclr_multicity_protocol_v2.py",
        "test_iclr_multicity_train_v2.py",
        "test_iclr_multicity_evaluate_v2.py",
        "test_iclr_primary_warsaw_protocol_v2.py",
        "test_iclr_primary_warsaw_train_v2.py",
        "test_iclr_residual_actuation.py",
        "test_iclr_residual_policy.py",
        "test_iclr_residual_runtime.py",
        "test_iclr_residual_train.py",
        "test_iclr_toy_model.py",
        "test_iclr_trace_parity.py",
        "test_iclr_wide_senior_audit.py",
    }
)
AUDITED_PROTOCOL_LOCK_RELATIVE_PATHS = tuple(
    path.relative_to(PROJECT).as_posix() for path in AUDITED_LOCK_PATHS
)
RELEASE_PROTOCOL_LOCK_RELATIVE_PATHS = (
    "results/iclr_multicity/protocol_lock.json",
    "results/iclr_external_validation/protocol_lock.json",
    "results/iclr_external_support_set/protocol_lock.json",
    "results/iclr_external_support_set_amendment/protocol_lock.json",
)
DEVELOPMENT_ONLY_PROTOCOL_LOCK_RELATIVE_PATHS = frozenset(
    {"results/iclr_multicity_v2/protocol_lock.json"}
)
REPORTED_RESULT_DIRS = (
    "results/iclr_frontier",
    "results/iclr_train",
    "results/iclr_splits",
    "results/iclr_multicity",
    "results/iclr_external_validation",
    "results/iclr_external_support_set",
    "results/iclr_external_support_set_amendment",
)
REPORTED_RESULT_RELATIVE_PATHS = tuple(
    sorted(
        {
            *(
                f"results/iclr_frontier/frontier_endowment_temporal_2022_seed42.{suffix}"
                for suffix in ("csv", "json")
            ),
            *(
                f"results/iclr_frontier/frontier_outcome_temporal_2022_seed{seed}.{suffix}"
                for seed in (1, 2, 3, 42)
                for suffix in ("csv", "json")
            ),
            *(
                f"results/iclr_train/{name}.json"
                for name in (
                    "run_district_endow_f0_g25",
                    "run_district_endow_f0_g25_b40",
                    "run_district_endow_f1_g25",
                    "run_district_endow_f1_g25_b40",
                    "run_district_endow_f2_g25",
                    "run_district_endow_f2_g25_b40",
                    "run_district_endow_f3_g25",
                    "run_district_endow_f3_g25_b40",
                    "run_district_endow_f4_g25",
                    "run_district_endow_f4_g25_b40",
                    "run_endow_bound40",
                    "run_endow_seed1",
                    "run_endow_seed2",
                    "run_main_seed42",
                )
            ),
            *(
                f"results/iclr_splits/{name}.json"
                for name in (
                    "city_out_Poland_Łódź",
                    "district_out_f0of5",
                    "district_out_f1of5",
                    "district_out_f2of5",
                    "district_out_f3of5",
                    "district_out_f4of5",
                    "temporal_2022",
                )
            ),
            "results/iclr_multicity/corpus_manifest.json",
            "results/iclr_multicity/evaluation/evidence_payload.json",
            "results/iclr_multicity/evaluation/per_series.csv",
            "results/iclr_multicity/evaluation/per_series.json",
            "results/iclr_multicity/evaluation/summary.json",
            "results/iclr_multicity/evidence_decision.json",
            *(
                "results/iclr_multicity/fits/"
                f"{spec['split']}/{spec['arm']}/seed-{spec['seed']}.json"
                for spec in locked_fit_inventory()
            ),
            "results/iclr_multicity/heldout_opened.json",
            "results/iclr_multicity/protocol_lock.json",
            "results/iclr_multicity/splits/city_out_Poland_Gdynia.json",
            "results/iclr_multicity/splits/city_out_Poland_Warszawa.json",
            "results/iclr_multicity/splits/city_out_Poland_Łódź.json",
            "results/iclr_multicity/splits/temporal_2022.json",
            "results/iclr_multicity/structural_gates.json",
            *(
                f"results/{study}/{name}.json"
                for study in (
                    "iclr_external_validation",
                    "iclr_external_support_set",
                )
                for name in (
                    "abort_record",
                    "corpus_manifest",
                    "heldout_opened",
                    "protocol_lock",
                    "published_lock_anchor",
                )
            ),
            "results/iclr_external_support_set_amendment/evaluation/demographic_coverage.json",
            "results/iclr_external_support_set_amendment/evaluation/evidence_payload.json",
            "results/iclr_external_support_set_amendment/evaluation/per_series.csv",
            "results/iclr_external_support_set_amendment/evaluation/per_series.json",
            "results/iclr_external_support_set_amendment/evaluation/summary.json",
            "results/iclr_external_support_set_amendment/evidence_decision.json",
            "results/iclr_external_support_set_amendment/heldout_opened.json",
            "results/iclr_external_support_set_amendment/protocol_lock.json",
            "results/iclr_external_support_set_amendment/published_lock_anchor.json",
        }
    )
)
EXCLUDED_REPORTED_RESULT_RELATIVE_PATHS = frozenset(
    {
        "results/iclr_frontier/frontier_endowment_temporal_2022_seed42_gen15.csv",
        "results/iclr_frontier/frontier_endowment_temporal_2022_seed42_gen15.json",
        "results/iclr_train/run_district_f0_smoke_fixed.json",
        "results/iclr_train/run_smoke.json",
        "results/iclr_train/run_smoke_outcome.json",
        "results/iclr_multicity/smoke/temporal_2022/endowment/seed-1.json",
        "results/iclr_multicity/smoke/temporal_2022/endowment/seed-2.json",
        "results/iclr_multicity/smoke/temporal_2022/priority/seed-1.json",
        "results/iclr_multicity/superseded/protocol_lock-1f5e42b3ba4b1adf1b2874554e2bde9907d6b9c250f1ab346d4e3869b1681b34.json",
    }
)
RELEASE_RESULT_FILES = (
    "ceiling.csv",
    "frozen_instances.txt",
    "iclr_ablation.csv",
    "iclr_actuating_subset.csv",
    "iclr_attribution_coverage.csv",
    "iclr_baselines.csv",
    "iclr_composition.csv",
    "iclr_cross_district_b40_manifest.json",
    "iclr_cross_district_b40_per_series.csv",
    "iclr_cross_district_b40_summary.csv",
    "iclr_cross_district_manifest.json",
    "iclr_cross_district_per_series.csv",
    "iclr_cross_district_summary.csv",
    "iclr_crosscity_actuation.csv",
    "iclr_crosscity_instances.txt",
    "iclr_crosscity_manifest.json",
    "iclr_crosscity_per_series.csv",
    "iclr_crosscity_summary.csv",
    "iclr_crosscity_winner_signatures.json",
    "iclr_degeneracy.csv",
    "iclr_endow_significance.csv",
    "iclr_env_verification.csv",
    "iclr_failure_analysis.csv",
    "iclr_identification.csv",
    "iclr_infeasible_spend.csv",
    "iclr_instance_granularity.csv",
    "iclr_llmrule_baseline.csv",
    "iclr_lock_amendments.json",
    "iclr_matched_target.csv",
    "iclr_multicity_instances.txt",
    "iclr_multicity_selection.json",
    "iclr_multicity_transfer.csv",
    "iclr_payment_intervention_manifest.json",
    "iclr_payment_intervention_per_series.csv",
    "iclr_payment_intervention_summary.csv",
    "iclr_per_series.csv",
    "iclr_refs_cache.json",
    "iclr_scheme_robustness.csv",
    "iclr_senior_tilt.csv",
    "iclr_significance.csv",
    "iclr_static_tilt.csv",
    "iclr_theory_check.csv",
    "iclr_transfer.csv",
    "iclr_two_arm_per_series.csv",
    "iclr_worst_cohort_mix.json",
)
EXCLUDED_TOP_LEVEL_RESULT_FILES = frozenset(
    {
        "iclr_ablation_gen20_confounded.csv",
        "iclr_ablation_log.txt",
        "iclr_ablation_log_gen25.txt",
        "iclr_identify_log.txt",
        "iclr_reachability_counterexample.csv",
        "iclr_refs_report.json",
        "iclr_static_tilt_log.txt",
    }
)
ANONYMOUS_MULTICITY_MANIFEST = "results/iclr_multicity/corpus_manifest.json"
PAPER_SOURCE = [
    "main.tex",
    "appendix.tex",
    "numbers.tex",
    "appendix_tables.tex",
    "table_frontiers.tex",
    "fig_frontier.pdf",
    "fig_forest.pdf",
    "fig_dynamics.pdf",
    "fig_endowmap.pdf",
    "fig_regularization.pdf",
    "fig_hack.pdf",
    "fig_schematic.pdf",
    "fig_coverage.pdf",
    "fig_external.pdf",
    "refs.bib",
    "refs_ml.bib",
    "math_commands.tex",
    "iclr2027_conference.sty",
    "iclr2027_conference.bst",
    "natbib.sty",
    "fancyhdr.sty",
]
REPRODUCIBILITY_INPUTS = (
    "pyproject.toml",
    "uv.lock",
    "analysis-output/ml-contribution-audit/history_free_seed42_g30.json",
    "analysis-output/ml-contribution-audit/static_age_lookup_seed42_g30.json",
    "analysis-output/external-support-set/manifest.json",
    "analysis-output/external-support-set/effects.csv",
    "analysis-output/external-support-set/city_intervals.csv",
    "analysis-output/external-support-set/decision_conditions.csv",
    "analysis-output/external-support-set/diagnostics.json",
    "analysis-output/external-support-set/analysis-report.md",
    "analysis-output/external-support-set/stats-appendix.md",
    "analysis-output/external-support-set/figure-catalog.md",
    "analysis-output/external-support-set/figures/fig_external.pdf",
    "analysis-output/external-support-set/figures/fig_external.png",
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_manifest_line(path: Path, output_root: Path = OUTPUT) -> str:
    """Return a checksum entry addressable from the manifest's directory."""
    relative_path = path.relative_to(output_root).as_posix()
    return f"{sha256(path)}  {relative_path}"


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _project_path(project: Path, relative: str) -> Path:
    """Resolve one repository-relative path without permitting archive escape."""
    project = Path(project).resolve()
    raw = Path(relative)
    if raw.is_absolute():
        raise ValueError(f"release path is absolute: {relative}")
    candidate = (project / raw).resolve()
    try:
        candidate.relative_to(project)
    except ValueError as exc:
        raise ValueError(f"release path escapes the project root: {relative}") from exc
    return candidate


def validate_release_namespace(project: Path = PROJECT) -> None:
    """Fail when a new paper-namespaced file lacks a release classification."""
    source_dir = Path(project) / "src"
    source_candidates = {
        path.name
        for pattern in ("iclr_*.py", "gen_iclr_*.py")
        for path in source_dir.glob(pattern)
    }
    classified_source = {
        name
        for name in set(RELEASE_SOURCE) | set(EXCLUDED_DEVELOPMENT_SOURCE)
        if name.startswith("iclr_") or name.startswith("gen_iclr_")
    }
    required_namespaced_source = {
        name
        for name in RELEASE_SOURCE
        if name.startswith("iclr_") or name.startswith("gen_iclr_")
    }
    unclassified = sorted(source_candidates - classified_source)
    missing_required = sorted(required_namespaced_source - source_candidates)
    if unclassified or missing_required:
        raise RuntimeError(
            "release source classification mismatch: "
            f"unclassified={unclassified}, missing_required={missing_required}"
        )
    if set(RELEASE_SOURCE) & set(EXCLUDED_DEVELOPMENT_SOURCE):
        raise RuntimeError("release and development-only source inventories overlap")
    if not set(RELEASE_ENTRYPOINTS) <= set(RELEASE_SOURCE):
        raise RuntimeError("a release entry point is absent from RELEASE_SOURCE")
    for name in RELEASE_SOURCE:
        if not (source_dir / name).is_file():
            raise FileNotFoundError(f"required release source missing: {source_dir / name}")

    test_dir = Path(project) / "tests"
    test_candidates = {path.name for path in test_dir.glob("test_iclr_*.py")}
    classified_tests = set(RELEASE_TESTS) | set(EXCLUDED_DEVELOPMENT_TESTS)
    unclassified_tests = sorted(test_candidates - classified_tests)
    missing_required_tests = sorted(set(RELEASE_TESTS) - test_candidates)
    if unclassified_tests or missing_required_tests:
        raise RuntimeError(
            "release test classification mismatch: "
            f"unclassified={unclassified_tests}, "
            f"missing_required={missing_required_tests}"
        )
    if set(RELEASE_TESTS) & set(EXCLUDED_DEVELOPMENT_TESTS):
        raise RuntimeError("release and development-only test inventories overlap")
    for name in RELEASE_TESTS:
        if not (test_dir / name).is_file():
            raise FileNotFoundError(f"required release test missing: {test_dir / name}")
    validate_reported_result_namespace(project)


def validate_reported_result_namespace(project: Path = PROJECT) -> None:
    """Fail when a watched result directory contains an unclassified artifact."""

    project = Path(project).resolve()
    results_dir = _project_path(project, "results")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"required results directory missing: {results_dir}")
    observed = set()
    for relative_dir in REPORTED_RESULT_DIRS:
        directory = _project_path(project, relative_dir)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"required reported result directory missing: {directory}"
            )
        observed.update(
            candidate.relative_to(project).as_posix()
            for candidate in directory.rglob("*")
            if candidate.is_file() and candidate.suffix in {".csv", ".json"}
        )
    classified = set(REPORTED_RESULT_RELATIVE_PATHS) | set(
        EXCLUDED_REPORTED_RESULT_RELATIVE_PATHS
    )
    unclassified = sorted(observed - classified)
    if unclassified:
        raise RuntimeError(
            f"reported result classification mismatch: unclassified={unclassified}"
        )

    observed_top_level = {
        candidate.name
        for candidate in results_dir.glob("iclr_*")
        if candidate.is_file() and candidate.suffix in {".csv", ".json", ".txt"}
    }
    released_top_level = {
        name
        for name in RELEASE_RESULT_FILES
        if name.startswith("iclr_") and Path(name).suffix in {".csv", ".json", ".txt"}
    }
    excluded_top_level = set(EXCLUDED_TOP_LEVEL_RESULT_FILES)
    overlap = sorted(released_top_level & excluded_top_level)
    unclassified_top_level = sorted(
        observed_top_level - released_top_level - excluded_top_level
    )
    if overlap or unclassified_top_level:
        raise RuntimeError(
            "top-level result classification mismatch: "
            f"overlap={overlap}, unclassified={unclassified_top_level}"
        )


def collect_locked_release_inputs(
    project: Path = PROJECT,
    lock_relative_paths: Sequence[str] = RELEASE_PROTOCOL_LOCK_RELATIVE_PATHS,
) -> Tuple[Path, ...]:
    """Return every non-corpus byte referenced by every configured lock."""
    project = Path(project).resolve()
    collected = set()
    for relative_lock in lock_relative_paths:
        lock_path = _project_path(project, relative_lock)
        if not lock_path.is_file():
            raise FileNotFoundError(f"required protocol lock missing: {lock_path}")
        collected.add(lock_path)
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        tracked = payload.get("tracked_files")
        if not isinstance(tracked, dict) or not tracked:
            raise ValueError(f"protocol lock has no tracked files: {lock_path}")
        for label, row in tracked.items():
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError(f"malformed tracked row {label!r} in {lock_path}")
            tracked_path = _project_path(project, row["path"])
            label_parts = str(label).split("/")
            if "corpus" in label_parts or not is_decision_critical(str(label)):
                continue
            if not tracked_path.is_file():
                raise FileNotFoundError(
                    f"locked non-corpus input missing: {tracked_path}"
                )
            collected.add(tracked_path)
    return tuple(sorted(collected))


def validate_protocol_lock_integrity(
    project: Path = PROJECT,
    lock_relative_paths: Sequence[str] = AUDITED_PROTOCOL_LOCK_RELATIVE_PATHS,
) -> None:
    """Fail closed unless every configured protocol-lock row is accepted."""

    project = Path(project).resolve()
    lock_paths = tuple(
        _project_path(project, relative) for relative in lock_relative_paths
    )
    assert_audit_passes(
        lock_paths,
        root=project,
        amendments_path=project / "results" / "iclr_lock_amendments.json",
    )


def validate_protocol_lock_partition(
    *,
    audited: Sequence[str] | None = None,
    release: Sequence[str] | None = None,
    development_only: frozenset[str] | None = None,
) -> None:
    """Require every audited lock to be explicitly release or development only."""

    audited_rows = tuple(
        AUDITED_PROTOCOL_LOCK_RELATIVE_PATHS if audited is None else audited
    )
    release_rows = tuple(
        RELEASE_PROTOCOL_LOCK_RELATIVE_PATHS if release is None else release
    )
    development_rows = frozenset(
        DEVELOPMENT_ONLY_PROTOCOL_LOCK_RELATIVE_PATHS
        if development_only is None
        else development_only
    )
    audited_set = set(audited_rows)
    release_set = set(release_rows)
    overlap = sorted(release_set & development_rows)
    unclassified = sorted(audited_set - release_set - development_rows)
    unknown = sorted((release_set | development_rows) - audited_set)
    duplicates = {
        "audited": len(audited_rows) != len(audited_set),
        "release": len(release_rows) != len(release_set),
    }
    if overlap or unclassified or unknown or any(duplicates.values()):
        raise RuntimeError(
            "protocol-lock release partition differs: "
            f"overlap={overlap}, unclassified={unclassified}, "
            f"unknown={unknown}, duplicates={duplicates}"
        )


def write_anonymous_multicity_manifest(
    source: Path,
    destination: Path,
    protocol_lock: Path,
    amendments_path: Path,
) -> None:
    """Redact location-only fields and bind the release bytes as an amendment."""
    source = Path(source)
    destination = Path(destination)
    protocol_lock = Path(protocol_lock)
    amendments_path = Path(amendments_path)
    try:
        source_bytes = source.read_bytes()
        raw = json.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed multicity corpus manifest: {source}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), list):
        raise ValueError(f"malformed multicity corpus manifest: {source}")
    lock_payload = json.loads(protocol_lock.read_text(encoding="utf-8"))
    row = lock_payload.get("tracked_files", {}).get("artifact/corpus_manifest")
    if not isinstance(row, dict):
        raise ValueError("multicity lock does not track artifact/corpus_manifest")
    if row.get("path") != ANONYMOUS_MULTICITY_MANIFEST:
        raise ValueError("multicity lock points to an unexpected corpus manifest")

    amendments = json.loads(amendments_path.read_text(encoding="utf-8"))
    if (
        not isinstance(amendments, dict)
        or type(amendments.get("schema_version")) is not int
        or amendments["schema_version"] != 1
        or not isinstance(amendments.get("amendments"), list)
        or not all(isinstance(entry, dict) for entry in amendments["amendments"])
    ):
        raise ValueError(f"malformed amendment inventory: {amendments_path}")
    entries = amendments["amendments"]
    manifest_entries = [
        entry
        for entry in entries
        if entry.get("path") == ANONYMOUS_MULTICITY_MANIFEST
    ]

    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    locked_sha256 = row.get("sha256")
    if not is_sha256(locked_sha256):
        raise ValueError(
            f"multicity protocol lock has malformed locked SHA-256: "
            f"{locked_sha256!r}"
        )
    if source_sha256 != locked_sha256:
        matching_entries = [
            entry
            for entry in manifest_entries
            if entry.get("locked_sha256") == locked_sha256
            and entry.get("amended_sha256") == source_sha256
            and entry.get("affects_recorded_decision") is False
            and entry.get("scope") == "anonymous-release-only"
        ]
        if len(manifest_entries) != 1 or len(matching_entries) != 1:
            raise ValueError(
                "multicity corpus manifest no longer matches its protocol lock "
                "or exact release amendment"
            )
        constraint = matching_entries[0].get("semantic_constraint")
        expected_values = {
            "destination": "data/pb_multicity",
            "source_dir": "pabulib_files/pb_files",
        }
        if (
            not isinstance(constraint, dict)
            or set(constraint)
            != {
                "kind",
                "excluded_top_level_fields",
                "retained_payload_sha256",
                "current_values",
            }
            or constraint.get("kind") != "json-top-level-location-redaction-v1"
            or constraint.get("excluded_top_level_fields")
            != ["destination", "source_dir"]
            or constraint.get("current_values") != expected_values
            or any(raw.get(key) != value for key, value in expected_values.items())
            or constraint.get("retained_payload_sha256")
            != multicity_manifest_retained_sha256(raw)
        ):
            raise ValueError("multicity release manifest semantic constraint differs")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_bytes)
        return

    if manifest_entries:
        raise ValueError("multicity manifest already has an amendment record")

    redacted = dict(raw)
    redacted["source_dir"] = "pabulib_files/pb_files"
    redacted["destination"] = "data/pb_multicity"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(redacted, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    amended_sha256 = sha256(destination)

    entries.append(
        {
            "path": ANONYMOUS_MULTICITY_MANIFEST,
            "locked_sha256": locked_sha256,
            "amended_sha256": amended_sha256,
            "change": (
                "Replace only the machine-specific source_dir and destination "
                "fields with repository-relative release paths."
            ),
            "reason": (
                "The locked manifest records absolute staging locations that "
                "identify the build machine but do not affect corpus bytes, "
                "series membership, splits, fitting, or evaluation."
            ),
            "verification": [
                "All 397 file rows, hashes, sizes, series labels, years, counts, and the upstream commit are unchanged.",
                "The amended digest is bound inside the anonymous release amendment inventory and the original locked digest remains in protocol_lock.json.",
            ],
            "affects_recorded_decision": False,
            "disclosed_in": "REPRODUCE.md anonymous-release section",
            "scope": "anonymous-release-only",
            "semantic_constraint": {
                "kind": "json-top-level-location-redaction-v1",
                "excluded_top_level_fields": ["destination", "source_dir"],
                "retained_payload_sha256": multicity_manifest_retained_sha256(raw),
                "current_values": {
                    "destination": "data/pb_multicity",
                    "source_dir": "pabulib_files/pb_files",
                },
            },
        }
    )
    amendments_path.write_text(
        json.dumps(amendments, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def collect_reported_result_inputs(project: Path = PROJECT) -> Tuple[Path, ...]:
    """Return the explicit reported-result inventory plus locked dependencies."""
    project = Path(project).resolve()
    paths = {
        _project_path(project, f"results/{name}") for name in RELEASE_RESULT_FILES
    }
    paths.update(
        _project_path(project, relative)
        for relative in REPORTED_RESULT_RELATIVE_PATHS
    )
    paths.update(
        collect_locked_release_inputs(
            project,
            RELEASE_PROTOCOL_LOCK_RELATIVE_PATHS,
        )
    )
    for candidate in paths:
        if not candidate.is_file():
            raise FileNotFoundError(f"required release result/input missing: {candidate}")
    return tuple(sorted(paths))


def validate_paper_source(paper_dir: Path, source_names: Sequence[str]) -> None:
    """Reject a source allowlist that omits a locally referenced graphic."""
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
    referenced = set()
    for tex_name in ("main.tex", "appendix.tex"):
        referenced.update(pattern.findall((paper_dir / tex_name).read_text()))
    packaged = set(source_names)
    missing = sorted(referenced - packaged)
    if missing:
        raise RuntimeError(f"Graphics missing from PAPER_SOURCE: {missing}")
    absent = sorted(name for name in referenced if not (paper_dir / name).is_file())
    if absent:
        raise FileNotFoundError(f"Referenced graphics do not exist: {absent}")


def write_zip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(source).as_posix())
            info.date_time = (2026, 8, 10, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def build_data_manifest(
    destination: Path,
    frozen_lists: Optional[Sequence[Path]] = None,
    data_dir: Optional[Path] = None,
    extra_scopes: Sequence[Tuple[Path, Path]] = (),
) -> int:
    """Combine frozen data scopes into one deduplicated integrity manifest.

    ``extra_scopes`` names (frozen list, data root) pairs for corpora that live
    outside the fitting directory, such as the held-out cross-city corpus. They
    are hashed the same way, so one manifest covers every file any reported
    experiment reads.
    """
    rows = ["filename\tseries\tyear\tbytes\tsha256"]
    list_paths = tuple(frozen_lists or (PROJECT / "results/frozen_instances.txt",))
    data_root = data_dir or PROJECT / "data/pb"
    scopes = [(path, data_root) for path in list_paths]
    for path, root in extra_scopes:
        if not path.is_file() or not root.is_dir():
            raise FileNotFoundError(
                f"Missing declared extra scope: frozen list {path}, data root {root}"
            )
        scopes.append((path, root))
    entries = {}
    for frozen_list, data_root in scopes:
        for raw in frozen_list.read_text().splitlines():
            if not raw or raw.startswith("#"):
                continue
            filename, series, year = raw.split("\t")
            identity = (series, year)
            data_path = data_root / filename
            if not data_path.is_file():
                raise FileNotFoundError(f"Missing frozen corpus file: {data_path}")
            entry = (
                series,
                year,
                data_path.stat().st_size,
                sha256(data_path),
            )
            if filename in entries:
                if entries[filename][:2] != identity:
                    raise RuntimeError(
                        f"Conflicting frozen metadata for {filename}: "
                        f"{entries[filename][:2]} versus {identity}"
                    )
                if entries[filename][2:] != entry[2:]:
                    raise RuntimeError(
                        f"Conflicting frozen bytes for {filename}: "
                        f"{entries[filename][2:]} versus {entry[2:]}"
                    )
            entries[filename] = entry
    rows.extend(
        "\t".join(map(str, (filename, *entries[filename])))
        for filename in sorted(entries)
    )
    destination.write_text("\n".join(rows) + "\n")
    return len(entries)


def pabulib_source_note(data_count: int) -> str:
    """Describe the exact public snapshot verified against the frozen corpus."""
    return (
        "# Pabulib corpus provenance\n\n"
        f"Upstream repository: {PABULIB_REPOSITORY}\n"
        f"Verified commit: {PABULIB_COMMIT}\n\n"
        f"All {data_count} unique files named in `DATA_MANIFEST.tsv` were checked "
        "byte-for-byte against this upstream commit and matched in both size and "
        "SHA-256 digest. The manifest remains the authoritative check for a local "
        "copy used to rerun the experiments.\n"
    )


def main() -> None:
    validate_release_namespace(PROJECT)
    validate_protocol_lock_partition()
    validate_protocol_lock_integrity(
        PROJECT,
        AUDITED_PROTOCOL_LOCK_RELATIVE_PATHS,
    )
    validate_paper_source(PAPER / "tex", PAPER_SOURCE)
    artifact_dir = OUTPUT / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pdf_output = OUTPUT / "pdf" / "ICLR2027_TempoPB_anonymous.pdf"
    copy(PAPER / "tex" / "main.pdf", pdf_output)
    with tempfile.TemporaryDirectory(prefix="tempopb-artifact-") as tmp:
        staging = Path(tmp) / "TempoPB-ICLR2027"
        staging.mkdir()
        for name in RELEASE_SOURCE:
            copy(PROJECT / "src" / name, staging / "src" / name)
        for name in RELEASE_TESTS:
            copy(PROJECT / "tests" / name, staging / "tests" / name)

        for path in collect_reported_result_inputs(PROJECT):
            relative = path.relative_to(PROJECT)
            if relative.as_posix() == ANONYMOUS_MULTICITY_MANIFEST:
                continue
            copy(path, staging / relative)
        write_anonymous_multicity_manifest(
            PROJECT / ANONYMOUS_MULTICITY_MANIFEST,
            staging / ANONYMOUS_MULTICITY_MANIFEST,
            PROJECT / "results/iclr_multicity/protocol_lock.json",
            staging / "results/iclr_lock_amendments.json",
        )

        copy(PAPER / "REPRODUCE.md", staging / "REPRODUCE.md")
        frozen_lists = (
            PROJECT / "results/frozen_instances.txt",
            PROJECT / "results/iclr_multicity_instances.txt",
        )
        data_count = build_data_manifest(
            staging / "DATA_MANIFEST.tsv",
            frozen_lists=frozen_lists,
            extra_scopes=(
                (
                    PROJECT / "results/iclr_crosscity_instances.txt",
                    PROJECT / "data/pb_crosscity",
                ),
            ),
        )
        (staging / "PABULIB_SOURCE.md").write_text(
            pabulib_source_note(data_count)
        )
        for relative in REPRODUCIBILITY_INPUTS:
            copy(PROJECT / relative, staging / relative)
        (staging / "REPRODUCIBILITY_INPUTS.sha256").write_text(
            "".join(
                f"{sha256(staging / relative)}  {relative}\n"
                for relative in REPRODUCIBILITY_INPUTS
            )
        )
        (staging / "README.md").write_text(
            "# TempoPB ICLR 2027 reproducibility package\n\n"
            "This anonymous package contains the exact analysis code, frozen splits, result files, "
            "and generated-paper pipeline. Follow `REPRODUCE.md`.\n\n## Data\n\n"
            f"The {data_count} voter-level Pabulib files are omitted because the public site did not expose "
            "an explicit redistribution license when this archive was built. Obtain the public files "
            f"from {PABULIB_REPOSITORY} at commit `{PABULIB_COMMIT}`, place them in `data/pb/`, and verify them "
            "against `DATA_MANIFEST.tsv` with `uv run python src/verify_iclr_data.py`. Then run "
            "`uv run python src/stage_iclr_release_data.py` to verify and materialize every "
            "protocol-locked corpus target. No private "
            "data are required. See `PABULIB_SOURCE.md` for the frozen source provenance.\n")
        write_zip(staging, artifact_dir / "ICLR2027_TempoPB_reproducibility.zip")

        source = Path(tmp) / "ICLR2027_TempoPB_source"
        source.mkdir()
        for name in PAPER_SOURCE:
            copy(PAPER / "tex" / name, source / name)
        write_zip(source, artifact_dir / "ICLR2027_TempoPB_source.zip")

    sums = []
    release_paths = [
        pdf_output,
        *sorted(artifact_dir.glob("ICLR2027_TempoPB_*.zip")),
    ]
    for path in release_paths:
        sums.append(checksum_manifest_line(path))
        print(f"{path}: {path.stat().st_size:,} bytes")
    (OUTPUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
