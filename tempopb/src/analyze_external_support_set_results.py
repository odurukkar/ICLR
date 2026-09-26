"""Load and analyse the immutable external support-set result bundle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd


EXPECTED_SHA256 = {
    "protocol_lock.json": "a6d7f945b6f6181f01ad30b9593d9d5f1cdbf1ec4cff977ad71488e76d17595b",
    "published_lock_anchor.json": "bac6ea9f36e45d0672906a24f2891bb077b8f37789f35262cfe916e62b40bae6",
    "heldout_opened.json": "80c7c0244d1f4493455764241e359fc42c2443c1dfb963ac441b26d2478dea9c",
    "evaluation/demographic_coverage.json": "52ffd19c0a8a87e0fdb798c21c5eb7dda8ee7bc1a20e8950fdd2cd1e250ed4b2",
    "evaluation/per_series.json": "ef568a4af2f22a44d0c6542c4bcc4f304a072efceeea16d69bcfae0a7353f7dc",
    "evaluation/per_series.csv": "46af97a12867b9183683cac02b17932d504ddd19ae804ead30e257775ec4e954",
    "evaluation/summary.json": "4b25bf289b731ebd06b315fa7820237a3b576a4e91074e2dfa2066f5e78f344c",
    "evaluation/evidence_payload.json": "41d399d7c59bcc70032d347fab71f1f4ec450e20fb2005cc185382575e6ace2c",
    "evidence_decision.json": "8a75c30f39361dad541022ba764facda3db5cbfbd9923e93fb2bda2ef9d32294",
}

EXPECTED_POLICIES = frozenset(
    {
        "mes",
        "endowment/seed-1",
        "endowment/seed-2",
        "endowment/seed-42",
        "priority/seed-1",
        "priority/seed-2",
        "priority/seed-42",
    }
)
LEARNED_POLICIES = EXPECTED_POLICIES - {"mes"}
EFFECT_COLUMNS = [
    "policy",
    "arm",
    "seed",
    "series",
    "city",
    "delta_csd",
    "delta_exclusion",
    "welfare_ratio",
    "actuated",
    "direction",
]
CITY_INTERVAL_COLUMNS = [
    "seed",
    "city",
    "baseline_csd",
    "learned_csd",
    "difference",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_kind",
    "holm_p_value",
    "exclusion_difference",
    "welfare_ratio",
    "wins",
    "ties",
    "losses",
    "n_series",
]
DECISION_CONDITION_COLUMNS = ["condition", "threshold", "passed", "observed_json"]
OUTPUT_FILENAMES = (
    "manifest.json",
    "effects.csv",
    "city_intervals.csv",
    "decision_conditions.csv",
    "diagnostics.json",
    "analysis-report.md",
    "stats-appendix.md",
    "figure-catalog.md",
)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VerifiedExternalBundle:
    root: Path
    lock: Mapping[str, Any]
    anchor: Mapping[str, Any]
    receipt: Mapping[str, Any]
    coverage: Mapping[str, Any]
    per_series: Mapping[str, Sequence[Mapping[str, Any]]]
    summary: Mapping[str, Any]
    evidence: Mapping[str, Any]
    decision: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _series_city(series: str) -> str:
    parts = series.split("/")
    if len(parts) != 3:
        raise ValueError(f"invalid series identity: {series!r}")
    return "/".join(parts[:2])


def _validate_policy_rows(per_series: Mapping[str, Any]) -> tuple[str, ...]:
    if set(per_series) != EXPECTED_POLICIES:
        raise ValueError("per_series policy keys do not match the frozen policy keys")

    mes_rows = per_series["mes"]
    if not isinstance(mes_rows, list) or len(mes_rows) != 32:
        raise ValueError("policy mes must contain exactly 32 rows")
    if not all(isinstance(row, dict) and "series" in row for row in mes_rows):
        raise ValueError("policy mes rows must contain series identities")
    series_order = tuple(str(row["series"]) for row in mes_rows)
    if len(set(series_order)) != 32:
        raise ValueError("policy mes must contain 32 unique series identities")

    for policy in EXPECTED_POLICIES:
        rows = per_series[policy]
        if not isinstance(rows, list) or len(rows) != 32:
            raise ValueError(f"policy {policy} must contain exactly 32 rows")
        if not all(isinstance(row, dict) and "series" in row for row in rows):
            raise ValueError(f"policy {policy} rows must contain series identities")
        if tuple(str(row["series"]) for row in rows) != series_order:
            raise ValueError(f"policy {policy} series order and identities do not match mes")
    return series_order


def _validate_authenticated_election_count(
    lock: Mapping[str, Any], per_series: Mapping[str, Any]
) -> None:
    expected = lock.get("protocol", {}).get("expected_counts", {}).get("elections")
    if expected != 228:
        raise ValueError("protocol lock must require exactly 228 authenticated elections")

    total = 0
    for row in per_series["mes"]:
        years = row.get("years")
        if not isinstance(years, list):
            raise ValueError("MES rows must include authenticated election years")
        total += len(years)
    if total != 228:
        raise ValueError("MES must contain exactly 228 authenticated elections")


def _validate_coverage(coverage: Mapping[str, Any]) -> None:
    rows = coverage.get("files")
    if not isinstance(rows, list) or len(rows) != 96:
        raise ValueError("demographic coverage must contain exactly 96 coverage rows")
    if coverage.get("n_scored_elections") != 96:
        raise ValueError("demographic coverage must declare exactly 96 scored elections")

    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("demographic coverage rows must be JSON objects")
        value = row.get("coverage")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("demographic coverage rows must contain finite numeric coverage")
        values.append(float(value))
    if min(values) < 0.50:
        raise ValueError("demographic coverage minimum must be at least 0.50")


def _merge_verified_csv(
    per_series: Mapping[str, Any], csv_bytes: bytes, series_order: tuple[str, ...]
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    table = pd.read_csv(io.BytesIO(csv_bytes))
    required_columns = {
        "policy",
        "series",
        "city",
        "worst_csd",
        "exclusion",
        "welfare",
        "outcome_sha256",
    }
    if not required_columns.issubset(table.columns):
        raise ValueError("per_series.csv is missing required columns")
    if len(table) != 32 * len(EXPECTED_POLICIES):
        raise ValueError("per_series.csv must contain exactly 224 rows")
    if set(table["policy"]) != EXPECTED_POLICIES:
        raise ValueError("per_series.csv policy keys do not match the frozen policy keys")

    merged: dict[str, Sequence[Mapping[str, Any]]] = {}
    for policy in EXPECTED_POLICIES:
        csv_rows = table.loc[table["policy"] == policy]
        if len(csv_rows) != 32 or tuple(csv_rows["series"]) != series_order:
            raise ValueError(f"per_series.csv {policy} rows do not match mes identities")

        enriched_rows: list[Mapping[str, Any]] = []
        for json_row, (_, csv_row) in zip(per_series[policy], csv_rows.iterrows(), strict=True):
            series = str(json_row["series"])
            city = _series_city(series)
            if csv_row["city"] != city:
                raise ValueError(f"per_series.csv city does not match series for {series}")
            outcome_sha256 = str(csv_row["outcome_sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", outcome_sha256):
                raise ValueError(f"invalid outcome SHA-256 for {policy}/{series}")
            enriched = dict(json_row)
            enriched["outcome_sha256"] = outcome_sha256
            enriched_rows.append(_freeze(enriched))
        merged[policy] = tuple(enriched_rows)
    return MappingProxyType(merged)


def load_and_verify_bundle(result_root: Path) -> VerifiedExternalBundle:
    root = result_root.resolve()
    verified_bytes: dict[str, bytes] = {}
    for relative, expected_digest in EXPECTED_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing required bundle file: {relative}")
        file_bytes = path.read_bytes()
        actual_digest = hashlib.sha256(file_bytes).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(f"SHA-256 mismatch for {relative}")
        verified_bytes[relative] = file_bytes

    lock = _require_mapping(json.loads(verified_bytes["protocol_lock.json"]), "protocol_lock")
    anchor = _require_mapping(
        json.loads(verified_bytes["published_lock_anchor.json"]), "published_lock_anchor"
    )
    receipt = _require_mapping(json.loads(verified_bytes["heldout_opened.json"]), "heldout_opened")
    coverage = _require_mapping(
        json.loads(verified_bytes["evaluation/demographic_coverage.json"]),
        "demographic_coverage",
    )
    per_series = _require_mapping(
        json.loads(verified_bytes["evaluation/per_series.json"]), "per_series"
    )
    summary = _require_mapping(json.loads(verified_bytes["evaluation/summary.json"]), "summary")
    evidence = _require_mapping(
        json.loads(verified_bytes["evaluation/evidence_payload.json"]), "evidence_payload"
    )
    decision = _require_mapping(
        json.loads(verified_bytes["evidence_decision.json"]), "evidence_decision"
    )

    if anchor.get("lock_sha256") != EXPECTED_SHA256["protocol_lock.json"]:
        raise ValueError("published lock anchor does not match the frozen protocol lock")
    if receipt.get("protocol_lock_sha256") != EXPECTED_SHA256["protocol_lock.json"]:
        raise ValueError("heldout receipt does not match the frozen protocol lock")
    if lock.get("protocol", {}).get("policies", {}).get("primary_seed") != 42:
        raise ValueError("protocol lock primary_seed must be 42")
    if evidence.get("primary_seed") != "42" or decision.get("primary_seed") != "42":
        raise ValueError("evidence and decision primary_seed must be '42'")
    if decision.get("classification") != "falsified":
        raise ValueError("frozen decision classification must be falsified")

    series_order = _validate_policy_rows(per_series)
    _validate_authenticated_election_count(lock, per_series)
    _validate_coverage(coverage)
    merged_per_series = _merge_verified_csv(
        per_series, verified_bytes["evaluation/per_series.csv"], series_order
    )
    return VerifiedExternalBundle(
        root=root,
        lock=_freeze(lock),
        anchor=_freeze(anchor),
        receipt=_freeze(receipt),
        coverage=_freeze(coverage),
        per_series=merged_per_series,
        summary=_freeze(summary),
        evidence=_freeze(evidence),
        decision=_freeze(decision),
    )


def _parse_policy(policy: str) -> tuple[str, int]:
    arm, seed_token = policy.split("/")
    return arm, int(seed_token.removeprefix("seed-"))


def _direction(delta: float) -> str:
    if delta < 0.0:
        return "win"
    if delta > 0.0:
        return "loss"
    return "tie"


def build_series_effects(bundle: VerifiedExternalBundle) -> pd.DataFrame:
    mes_rows = bundle.per_series["mes"]
    mes_by_series = {str(row["series"]): row for row in mes_rows}
    if len(mes_by_series) != 32:
        raise ValueError("missing or duplicate MES baseline series")

    effects: list[dict[str, Any]] = []
    for policy in sorted(LEARNED_POLICIES):
        arm, seed = _parse_policy(policy)
        for row in bundle.per_series[policy]:
            series = str(row["series"])
            baseline = mes_by_series.get(series)
            if baseline is None:
                raise ValueError(f"missing MES baseline for {series}")
            baseline_welfare = float(baseline["welfare"])
            if baseline_welfare == 0.0:
                raise ValueError(f"zero MES welfare baseline for {series}")
            delta_csd = float(row["worst_csd"]) - float(baseline["worst_csd"])
            effects.append(
                {
                    "policy": policy,
                    "arm": arm,
                    "seed": seed,
                    "series": series,
                    "city": _series_city(series),
                    "delta_csd": delta_csd,
                    "delta_exclusion": float(row["exclusion"]) - float(baseline["exclusion"]),
                    "welfare_ratio": float(row["welfare"]) / baseline_welfare,
                    "actuated": row["outcome_sha256"] != baseline["outcome_sha256"],
                    "direction": _direction(delta_csd),
                }
            )
    return pd.DataFrame(effects, columns=EFFECT_COLUMNS).sort_values(
        ["arm", "seed", "city", "series"], ignore_index=True
    )


def build_city_intervals(bundle: VerifiedExternalBundle) -> pd.DataFrame:
    """Copy frozen city-level interval evidence without recomputation."""
    rows: list[dict[str, Any]] = []
    for seed_token, seed_data in sorted(bundle.evidence["seeds"].items(), key=lambda item: int(item[0])):
        for city, values in sorted(seed_data["cities"].items()):
            rows.append(
                {
                    "seed": int(seed_token),
                    "city": city,
                    "baseline_csd": values["baseline_csd"],
                    "learned_csd": values["learned_csd"],
                    "difference": values["difference"],
                    "ci_low": values["ci_low"],
                    "ci_high": values["ci_high"],
                    "p_value": values["p_value"],
                    "p_value_kind": values["p_value_kind"],
                    "holm_p_value": values.get("holm_p_value"),
                    "exclusion_difference": values["exclusion_difference"],
                    "welfare_ratio": values["welfare_ratio"],
                    "wins": values["wins"],
                    "ties": values["ties"],
                    "losses": values["losses"],
                    "n_series": values["n_series"],
                }
            )
    intervals = pd.DataFrame(rows, columns=CITY_INTERVAL_COLUMNS)
    intervals["holm_p_value"] = intervals["holm_p_value"].astype(object).where(
        intervals["holm_p_value"].notna(), None
    )
    return intervals


def build_decision_conditions(bundle: VerifiedExternalBundle) -> pd.DataFrame:
    """Copy the frozen mechanical gate flags and their authenticated evidence."""
    rows = [
        {
            "condition": name,
            "threshold": values["threshold"],
            "passed": values["passed"],
            "observed_json": json.dumps(
                _json_ready(values["observed"]),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        for name, values in sorted(bundle.decision["conditions"].items())
    ]
    return pd.DataFrame(rows, columns=DECISION_CONDITION_COLUMNS)


def compute_diagnostics(
    bundle: VerifiedExternalBundle, effects: pd.DataFrame
) -> dict[str, Any]:
    """Derive descriptive seed-consistency diagnostics from verified effects."""
    primary = effects.loc[(effects["arm"] == "endowment") & (effects["seed"] == 42)]
    endowment = effects.loc[effects["arm"] == "endowment"]
    by_seed = endowment.pivot(index="series", columns="seed", values="delta_csd")
    if list(by_seed.columns) != [1, 2, 42] or len(by_seed) != 32:
        raise ValueError("endowment effects must contain 32 series for seeds 1, 2, and 42")

    always_win = int((by_seed < 0.0).all(axis=1).sum())
    always_tie = int((by_seed == 0.0).all(axis=1).sum())
    switches_sign = int(((by_seed < 0.0).any(axis=1) & (by_seed > 0.0).any(axis=1)).sum())
    mixed_zero = len(by_seed) - always_win - always_tie - switches_sign

    improvements = sorted(
        (
            Decimal(str(value)).copy_abs()
            for value in primary.loc[primary["delta_csd"] < 0, "delta_csd"]
        ),
        reverse=True,
    )
    calculated_top3_concentration = sum(improvements[:3]) / sum(improvements)
    if not calculated_top3_concentration.is_finite():
        raise ValueError("top-three improvement concentration must be finite")
    top3_concentration = math.nextafter(float(calculated_top3_concentration), -math.inf)
    seed_evidence = bundle.evidence["seeds"]
    return {
        "decision": bundle.decision["classification"],
        "descriptive_only": True,
        "primary": {
            "seed": 42,
            "wins": int((primary["direction"] == "win").sum()),
            "ties": int((primary["direction"] == "tie").sum()),
            "losses": int((primary["direction"] == "loss").sum()),
        },
        "across_seeds": {
            "always_win": always_win,
            "always_tie": always_tie,
            "switches_sign": switches_sign,
            "mixed_zero": mixed_zero,
            "total_series": len(by_seed),
        },
        "top3_improvement_concentration": top3_concentration,
        "minimum_demographic_coverage": bundle.coverage["minimum_observed"],
        "city_macro_differences": {
            seed: seed_evidence[seed]["city_macro_difference"]
            for seed in sorted(seed_evidence, key=int)
        },
        "actuation": {
            "endowment": bundle.evidence["endowment_actuated_series"],
            "priority": bundle.evidence["priority_actuated_series"],
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def _format_number(value: Any) -> str:
    return repr(value)


def _render_analysis_report(
    bundle: VerifiedExternalBundle,
    diagnostics: Mapping[str, Any],
    intervals: pd.DataFrame,
    conditions: pd.DataFrame,
) -> str:
    protocol = bundle.lock["protocol"]
    projection = protocol["ballot_projection"]
    lines = [
        "# External Support-Set Analysis",
        "",
        "## 1. Immutable protocol identity and support-set projection",
        "",
        f"Study: `{protocol['study']}`.",
        f"Protocol lock SHA-256: `{EXPECTED_SHA256['protocol_lock.json']}`.",
        f"Katowice projection: {projection['Katowice']}.",
        f"Krakow projection: {projection['Krakow']}.",
        "",
        "## 2. Authenticated corpus and demographic coverage",
        "",
        "228 authenticated elections; 32 series; 96 scored-election demographic checks.",
        "Observed minimum coverage: "
        f"{_format_number(diagnostics['minimum_demographic_coverage'])}.",
        "",
        "## 3. City-macro differences and frozen city intervals",
        "",
    ]
    for seed, difference in diagnostics["city_macro_differences"].items():
        lines.append(f"- Seed {seed} city-macro CSD difference: {_format_number(difference)}.")
    for row in intervals.itertuples(index=False):
        holm = "not applicable" if pd.isna(row.holm_p_value) else _format_number(row.holm_p_value)
        lines.append(
            f"- Seed {row.seed}, {row.city}: 95% interval "
            f"[{_format_number(row.ci_low)}, {_format_number(row.ci_high)}]; "
            f"exact sign-flip p-value {_format_number(row.p_value)}; Holm p-value {holm}."
        )
    lines.extend(
        [
            "",
            "## 4. Descriptive series diagnostics",
            "",
            f"Primary seed 42 decomposition: {diagnostics['primary']['wins']} wins, "
            f"{diagnostics['primary']['ties']} ties, {diagnostics['primary']['losses']} losses.",
            f"Across seeds: {diagnostics['across_seeds']['always_win']} always-wins, "
            f"{diagnostics['across_seeds']['always_tie']} all-ties, and "
            f"{diagnostics['across_seeds']['switches_sign']} sign switches.",
            "Top-three improvement concentration: "
            f"{_format_number(diagnostics['top3_improvement_concentration'])}.",
            "",
            "## 5. Frozen mechanical gate conditions",
            "",
            "Frozen gates: 4 passed and 4 failed.",
        ]
    )
    for row in conditions.itertuples(index=False):
        status = "passed" if row.passed else "failed"
        lines.append(f"- {row.condition}: {status} ({row.threshold}).")
    lines.extend(["", "## 6. Mechanical decision", "", "Mechanical decision: falsified.", ""])
    return "\n".join(lines)


def _render_stats_appendix(
    bundle: VerifiedExternalBundle, intervals: pd.DataFrame, conditions: pd.DataFrame
) -> str:
    protocol = bundle.lock["protocol"]
    lines = [
        "# Statistics Appendix",
        "",
        "All values below are copied from the authenticated evidence and decision files.",
        f"Bootstrap draws: {protocol['statistics']['bootstrap_draws']}.",
        f"Exact sign-flip draws: {protocol['statistics']['sign_flip_draws']}.",
        "",
        "## City intervals and p-values",
        "",
    ]
    for row in intervals.itertuples(index=False):
        holm = "not applicable" if pd.isna(row.holm_p_value) else _format_number(row.holm_p_value)
        lines.append(
            f"- Seed {row.seed}; {row.city}; interval "
            f"[{_format_number(row.ci_low)}, {_format_number(row.ci_high)}]; "
            f"exact sign-flip p-value {_format_number(row.p_value)}; Holm p-value {holm}."
        )
    lines.extend(["", "## Mechanical conditions", ""])
    for row in conditions.itertuples(index=False):
        lines.append(f"- {row.condition}: {'passed' if row.passed else 'failed'}.")
    lines.append("")
    return "\n".join(lines)


def _render_figure_catalog() -> str:
    return "\n".join(
        [
            "# Figure Catalog",
            "",
            "- External dashboard panel A: city-by-seed frozen bootstrap intervals.",
            "- External dashboard panel B: 32-series by three-seed CSD-effect heatmap.",
            "- External dashboard panel C: primary-seed CSD-versus-exclusion descriptive tradeoff.",
            "",
        ]
    )


def _reject_unsafe_output_dir(result_root: Path, output_dir: Path) -> None:
    resolved_result_root = result_root.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir == resolved_result_root or resolved_result_root in resolved_output_dir.parents:
        raise ValueError("output_dir must not be inside result_root")


def _validate_existing_output_entries(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("output_dir must be a non-symlink directory")
    names = {entry.name for entry in output_dir.iterdir()}
    unexpected_names = names - set(OUTPUT_FILENAMES)
    if unexpected_names:
        raise ValueError("analysis output directory must contain only strict bundle names")
    for name in names:
        entry = output_dir / name
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("pre-existing output entries must be regular non-symlink files")


def _validate_complete_output_entries(output_dir: Path) -> None:
    names = {entry.name for entry in output_dir.iterdir()}
    if names != set(OUTPUT_FILENAMES):
        raise ValueError("analysis output directory must contain exactly the strict bundle names")
    for name in OUTPUT_FILENAMES:
        entry = output_dir / name
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("strict bundle entries must be regular non-symlink files")


def refresh_output_manifest(output_dir: Path) -> Path:
    """Hash each strict-bundle output except the manifest itself."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest.json must exist before manifest hashes are refreshed")
    _validate_complete_output_entries(output_dir)
    manifest = _require_mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "output manifest")
    refreshed = dict(manifest)
    refreshed["output_sha256"] = {
        name: sha256_file(output_dir / name)
        for name in OUTPUT_FILENAMES
        if name != "manifest.json"
    }
    _write_json(manifest_path, refreshed)
    return manifest_path


def write_analysis_bundle(result_root: Path, output_dir: Path) -> dict[str, Path]:
    """Write the deterministic strict analysis bundle from authenticated inputs."""
    _reject_unsafe_output_dir(result_root, output_dir)
    _validate_existing_output_entries(output_dir)
    bundle = load_and_verify_bundle(result_root)
    effects = build_series_effects(bundle)
    intervals = build_city_intervals(bundle)
    conditions = build_decision_conditions(bundle)
    diagnostics = compute_diagnostics(bundle, effects)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    _write_csv(paths["effects.csv"], effects)
    _write_csv(paths["city_intervals.csv"], intervals)
    _write_csv(paths["decision_conditions.csv"], conditions)
    _write_json(paths["diagnostics.json"], diagnostics)
    paths["analysis-report.md"].write_text(
        _render_analysis_report(bundle, diagnostics, intervals, conditions),
        encoding="utf-8",
        newline="\n",
    )
    paths["stats-appendix.md"].write_text(
        _render_stats_appendix(bundle, intervals, conditions), encoding="utf-8", newline="\n"
    )
    paths["figure-catalog.md"].write_text(
        _render_figure_catalog(), encoding="utf-8", newline="\n"
    )
    _write_json(
        paths["manifest.json"],
        {
            "classification": bundle.decision["classification"],
            "entry_point": "python src/analyze_external_support_set_results.py",
            "input_sha256": {
                relative: EXPECTED_SHA256[relative] for relative in sorted(EXPECTED_SHA256)
            },
            "output_sha256": {},
            "package_versions": {"pandas": pd.__version__},
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "schema_version": 1,
        },
    )
    refresh_output_manifest(output_dir)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "results" / "iclr_external_support_set_amendment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis-output" / "external-support-set",
    )
    args = parser.parse_args(argv)
    write_analysis_bundle(args.result_root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
