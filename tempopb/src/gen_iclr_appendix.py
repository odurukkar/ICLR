"""Generate the appendix tables from results/, so the main text can be trimmed.

ICLR allows an unlimited appendix, and the main text is close enough to the page
limit that material will have to move. This emits the per-series, per-seed and
full-robustness tables as a single \\input-able file, plus the complete two-arm
frontier table. Both are generated from the same result files the main text
reads, so they cannot drift apart.

Writes iclr_paper/tex/appendix_tables.tex and table_frontiers.tex.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping

from analyze_external_support_set_results import load_and_verify_bundle, sha256_file
from gen_iclr_numbers import (
    _load_external_support_set_inputs,
    _load_frontier_seed_sweep,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT.parent / "iclr_paper" / "tex" / "appendix_tables.tex"
FRONTIER_OUT = ROOT.parent / "iclr_paper" / "tex" / "table_frontiers.tex"


def _read(path: Path) -> List[dict]:
    if not path.exists():
        logger.warning("missing %s", path)
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _esc(text: str) -> str:
    """Escape LaTeX specials. Cohort keys look like `age60+|M`; the pipe is
    fragile in text mode, so render these as readable labels instead."""
    text = (
        text.replace("age", "")
            .replace("|F", " (F)")
            .replace("|M", " (M)")
    )
    return (
        text.replace("&", r"\&")
            .replace("_", r"\_")
            .replace("|", r"\textbar{}")
    )


SCHEME_LABEL = {
    "age_sex": r"age $\times$ sex",
    "age": "age",
    "sex": "sex",
}


def _pfmt(value: float) -> str:
    """Match the exact-test reporting precision used in the paper."""
    if value < 1e-3:
        return "$<0.001$"
    return f"{value:.4f}"


def _per_series_table() -> str:
    rows = _read(RESULTS / "iclr_per_series.csv")
    if not rows:
        return ""
    out = [
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Series & Test years & \MES{} & Learned outcome & $\Delta$ & Worst group \\",
        r"\midrule",
    ]
    for r in rows:
        name = _esc(r["series"].split("/")[-1])
        delta = float(r["learned_minus_mes"])
        mark = r"\textbf{" + f"{delta:+.4f}" + "}" if delta > 0 else f"{delta:+.4f}"
        out.append(
            f"{name} & {r['test_years'].replace('|', ', ')} & "
            f"{float(r['mes']):.4f} & {float(r['learned']):.4f} & {mark} & "
            f"{_esc(str(r.get('worst_cohort', '')))} \\\\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Per-series worst-group CSD for the learned outcome policy at "
        r"soft target 1.00 on held-out Warsaw editions. $\Delta$ is learned minus "
        r"Equal Shares; negative values favor learned and positive losses are bold. "
        r"Rows are descriptive without intervals. The aggregate is the equal-weight "
        r"mean district-level worst-group CSD, all from one city.}",
        r"\label{tab:perseries}\end{table}",
    ]
    return "\n".join(out)


def _res_grid_table(payload: dict | None = None) -> str:
    """Expose every tested hand-designed rollover intensity in the PDF."""
    if payload is None:
        path = RESULTS / "iclr_train" / "run_main_seed42.json"
        if not path.exists():
            return ""
        payload = json.loads(path.read_text())

    baselines = payload.get("baselines", {})
    rows = (
        ("0.00", "mes/test"),
        ("0.25", "res-0.25/test"),
        ("0.50", "res-0.5/test"),
        ("0.75", "res-0.75/test"),
        ("1.00", "res-1.0/test"),
    )
    missing = [key for _, key in rows if key not in baselines]
    if missing:
        raise ValueError(f"rollover grid is missing baseline rows: {missing}")

    out = [
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Rollover intensity $\lambda$ & Worst $\csd$ & Welfare & Exclusion \\",
        r"\midrule",
    ]
    for intensity, key in rows:
        values = baselines[key]
        out.append(
            f"{intensity} & {float(values['worst_csd']):.4f} & "
            f"{float(values['welfare']):,.0f} & "
            f"{float(values['exclusion']):.4f} \\\\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{All tested rollover intensities on held-out Warsaw editions. Lower CSD and "
        r"exclusion and higher welfare are favorable; the former are equal-weight "
        r"district-series means and welfare is summed. Rows are descriptive "
        r"without intervals. $\lambda=0$ is Equal Shares; $\lambda=1$ is the "
        r"main-text reference and optimizer initial mean, not selected on held-out "
        r"results.}",
        r"\label{tab:res-grid}\end{table}",
    ]
    return "\n".join(out)


def _seed_table() -> str:
    per_floor: Dict[float, Dict[str, float]] = {}
    for frontier in _load_frontier_seed_sweep():
        seed = str(frontier["seed"])
        for p in frontier["points"]:
            per_floor.setdefault(p["floor"], {})[seed] = p["test_worst_csd"]
    if not per_floor:
        return ""
    seeds = sorted({s for v in per_floor.values() for s in v}, key=int)
    out = [
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{l" + "c" * (len(seeds) + 1) + "}",
        r"\toprule",
        "Soft welfare target & " + " & ".join(f"seed {s}" for s in seeds) + r" & spread \\",
        r"\midrule",
    ]
    for floor in sorted(per_floor):
        vals = [per_floor[floor].get(s) for s in seeds]
        cells = " & ".join(f"{v:.4f}" if v is not None else "--" for v in vals)
        got = [v for v in vals if v is not None]
        spread = max(got) - min(got) if len(got) > 1 else float("nan")
        out.append(f"{floor:.2f} & {cells} & {spread:.4f} \\\\")
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Outcome-frontier seed sensitivity on held-out Warsaw district "
        r"series. Cells are equal-weight mean district-level worst-group CSD "
        r"(lower is favorable); spread is the fitted-seed range and describes "
        r"optimization sensitivity, not sampling uncertainty.}",
        r"\label{tab:seeds}\end{table}",
    ]
    return "\n".join(out)


def _scheme_table() -> str:
    rows = _read(RESULTS / "iclr_scheme_robustness.csv")
    if not rows:
        return ""
    out = [
        r"\begin{table}[t]\centering\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Partition & Policy & Worst $\csd$ & vs \MES{} & $p$ & Wins \\",
        r"\midrule",
    ]
    last = None
    for r in rows:
        scheme = r["scheme"]
        label = (
            f"{SCHEME_LABEL.get(scheme, _esc(scheme))} ({r['n_groups']})"
            if scheme != last else ""
        )
        last = scheme
        diff = r.get("vs_mes_diff") or ""
        pval = r.get("vs_mes_p") or ""
        wins = r.get("vs_mes_wins") or ""
        n = r.get("n") or ""
        extra = (
            f"{float(diff):+.4f} & {_pfmt(float(pval))} & {wins}/{n}"
            if diff else "-- & -- & --"
        )
        out.append(
            f"{label} & {r['policy'].replace('_', chr(92) + '_')} & "
            f"{float(r['worst_csd']):.4f} & {extra} \\\\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Held-out Warsaw district series rescored under each group "
        r"partition. ``vs \MES{}'' is policy minus Equal Shares; negative values, "
        r"paired exact sign-flip $p$ values, and wins use series. Age~$\times$~sex "
        r"policies are rescored without refitting, so rows are dependent robustness "
        r"checks, not transfer evidence.}",
        r"\label{tab:schemefull}\end{table}",
    ]
    return "\n".join(out)


def _payment_intervention_table() -> str:
    rows = _read(RESULTS / "iclr_payment_intervention_summary.csv")
    if not rows:
        return ""
    wanted_floors = {0.0, 0.85, 1.0}
    order = {
        "direct": 0,
        "static-floor": 1,
        "payment-only": 2,
        "payment+completion": 3,
    }
    rows = sorted(
        (
            row
            for row in rows
            if float(row["floor"]) in wanted_floors and row["kernel"] in order
        ),
        key=lambda row: (float(row["floor"]), order[row["kernel"]]),
    )
    labels = {
        "direct": "direct greedy fill",
        "static-floor": r"direct + static floor ($\kappa=1$)",
        "payment-only": "iterative payment only",
        "payment+completion": "iterative payment + completion",
    }
    out = [
        r"\begin{table}[H]\centering\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Soft target & Allocation kernel & Worst $\csd$ & $\Delta$ direct & Welfare & W/direct & Exclusion \\",
        r"\midrule",
    ]
    previous_floor = None
    for row in rows:
        floor = float(row["floor"])
        floor_cell = f"{floor:.2f}" if floor != previous_floor else ""
        previous_floor = floor
        out.append(
            f"{floor_cell} & {labels[row['kernel']]} & "
            f"{float(row['worst_csd']):.4f} & {float(row['diff_vs_direct']):+.4f} & "
            f"{float(row['welfare']):,.0f} & "
            f"{float(row['welfare_ratio_vs_direct']):.2f} & "
            f"{float(row['exclusion']):.4f} \\\\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Allocation-kernel replay over held-out Warsaw district series. "
        r"Within each target, $\Delta$ is kernel minus direct-greedy CSD; negative "
        r"values favor the kernel, as do lower exclusion and higher welfare. Values "
        r"are descriptive without intervals. Score weights and feature definitions "
        r"are fixed, but history-dependent values and scores may diverge. The static "
        r"floor uses Proposition~\ref{prop:support} at $\kappa=1$; uniform-balance "
        r"score-prioritized payment is diagnostic, not standard \MES{}.}",
        r"\label{tab:kernel-intervention}\end{table}",
    ]
    return "\n".join(out)


def _cross_district_table(rows: List[dict] | None = None) -> str:
    """Render all pooled out-of-district contrasts with their scope visible."""
    if rows is None:
        rows = _read(RESULTS / "iclr_cross_district_summary.csv")
    if not rows:
        return ""
    labels = {
        "mes": r"\MES{}",
        "res-1.0": r"\RES{} $(\lambda=1)$",
        "greedy-count": "greedy-count",
        "greedy-cost": "greedy-cost",
        "llmrule-card": "LLMRule count",
        "llmrule-cost": "LLMRule cost",
    }
    out = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Baseline & Learned $\csd$ & Base $\csd$ & $\Delta$ & 95\% $\Delta$ interval & $p_{\rm fold}$ & Series wins & Fold wins \\",
        r"\midrule",
    ]
    for row in rows:
        baseline = row["baseline"]
        out.append(
            f"{labels.get(baseline, _esc(baseline))} & "
            f"{float(row['learned_mean']):.4f} & "
            f"{float(row['baseline_mean']):.4f} & "
            f"{float(row['diff']):+.4f} & "
            f"[{float(row['ci_lo']):+.4f}, {float(row['ci_hi']):+.4f}] & "
            f"{float(row['fold_p_two_sided']):.4f} & "
            f"{row['wins']}/{row['n']} & "
            f"{row['fold_wins']}/{row['n_folds']} \\\\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Five-fold out-of-district contrasts over 19 series, each scored "
        r"by a map fitted on the other districts. This is not a temporal holdout and "
        r"not cross-city validation: 18 of the 19 series are Warsaw districts. $\Delta$ is learned "
        r"minus baseline; negative values favor learned. Paired-series bootstrap "
        r"intervals are descriptive over 19 series. $p_{\rm fold}$ is an exact "
        r"sign-flip dependence sensitivity check on fold means (minimum two-sided resolution 0.0625). "
        r"Overlapping training sets make folds dependent.}",
        r"\label{tab:district-cv}\end{table}",
    ]
    return "\n".join(out)


def _two_arm_per_series_table(rows: List[dict] | None = None) -> str:
    """Every district under both learned surfaces, so losses stay checkable."""
    if rows is None:
        rows = _read(RESULTS / "iclr_two_arm_per_series.csv")
    if not rows:
        return ""
    ordered = sorted(rows, key=lambda r: float(r["endowment_minus_mes"]))
    out = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Series & \MES{} & Endowment & $\Delta$ & Direct & $\Delta$ \\",
        r"\midrule",
    ]
    for row in ordered:
        endow_delta = float(row["endowment_minus_mes"])
        direct_delta = float(row["outcome_minus_mes"])
        name = _esc(row["series"].split("/")[-1])
        out.append(
            f"{name} & {float(row['mes']):.4f} & "
            f"{float(row['learned_endowment']):.4f} & "
            + (f"\\textbf{{{endow_delta:+.4f}}}" if endow_delta > 0 else f"{endow_delta:+.4f}")
            + f" & {float(row['learned_outcome']):.4f} & "
            + (f"\\textbf{{{direct_delta:+.4f}}}" if direct_delta > 0 else f"{direct_delta:+.4f}")
            + r" \\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Headline learned endowment and direct-outcome policies by "
        r"held-out Warsaw district, ordered by endowment contrast. Each $\Delta$ "
        r"is policy minus Equal Shares; negative favors learned and positive losses "
        r"are bold. Rows are descriptive without intervals.}",
        r"\label{tab:twoarm}\end{table}",
    ]
    return "\n".join(out)


def _crosscity_table(manifest: Mapping[str, Any] | None = None) -> str:
    """Render the native-approval cross-city contrast and its actuation grid."""
    path = RESULTS / "iclr_crosscity_manifest.json"
    if manifest is None:
        if not path.is_file():
            return ""
        manifest = json.loads(path.read_text())
    labels = {"Poland/Gdynia": "Gdynia", "Poland/Łódź": r"\L{}\'od\'z", "ALL": "Both cities"}
    out = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\begin{tabular}{lrrrrlr}",
        r"\toprule",
        r"Scope & Learned $\csd$ & \MES{} $\csd$ & $\Delta$ & 95\% $\Delta$ interval & $p$ & Wins \\",
        r"\midrule",
    ]
    for scope in ("Poland/Gdynia", "Poland/Łódź", "ALL"):
        row = manifest["contrasts"].get(f"{scope}|learned-endowment|mes")
        if row is None:
            continue
        out.append(
            f"{labels[scope]} & {float(row['learned_csd']):.4f} & "
            f"{float(row['baseline_csd']):.4f} & {float(row['difference']):+.4f} & "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] & "
            f"{float(row['p_value']):.3f} & {row['wins']}/{row['n_series']} \\\\"
        )
    out += [
        r"\midrule",
        r"\multicolumn{7}{l}{\emph{Actuation by instance granularity (whole prespecified grid)}} \\",
        r"Min.\ projects/election & Series & Inert & $\Delta$ & 95\% $\Delta$ interval & $p$ & Wins/losses \\",
        r"\midrule",
    ]
    for row in manifest["actuation_by_granularity"]:
        out.append(
            f"$\\ge$ {row['min_projects_per_election']} & {row['n_series']} & "
            f"{row['n_inert']} & {float(row['difference']):+.4f} & "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] & "
            f"{float(row['p_value']):.3f} & {row['wins']}/{row['losses']} \\\\"
        )
    actuation = manifest["actuation"]
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Native-approval Gdynia and \L{}\'od\'z series scored from 2023 "
        r"with the frozen Warsaw map, without refitting or support-set projection. "
        r"$\Delta$ is learned minus Equal Shares; negative favors learned. "
        r"Paired-series bootstrap intervals summarize uncertainty. \emph{Inert} "
        f"means the Equal Shares allocation (median projects: "
        f"{actuation['median_projects_inert']:.1f} inert, "
        f"{actuation['median_projects_active']:.1f} active). Effects stay near zero "
        r"across the prespecified grid, so low actuation alone does not explain the "
        r"null. $p$ uses exact sign flips when feasible and 200{,}000 draws "
        r"otherwise. Two cities do not establish wider validity.}",
        r"\label{tab:crosscity}\end{table}",
    ]
    return "\n".join(out)


def _cross_district_bound_table(
    primary_rows: List[dict] | None = None,
    sensitivity_rows: List[dict] | None = None,
) -> str:
    """Compare the primary district fits with the wider-box reruns."""
    if primary_rows is None:
        primary_rows = _read(RESULTS / "iclr_cross_district_summary.csv")
    if sensitivity_rows is None:
        sensitivity_rows = _read(RESULTS / "iclr_cross_district_b40_summary.csv")
    if not primary_rows or not sensitivity_rows:
        return ""

    def mes_row(rows: List[dict]) -> dict:
        row = next((item for item in rows if item["baseline"] == "mes"), None)
        if row is None:
            raise ValueError("district-bound sensitivity is missing the MES row")
        return row

    compared = [
        ("10 (primary)", mes_row(primary_rows)),
        ("40 (post hoc)", mes_row(sensitivity_rows)),
    ]
    out = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\toprule",
        r"Weight box & Learned & \MES{} & $\Delta$ & 95\% $\Delta$ interval & $p_{\rm fold}$ & S-win & F-win & W/\MES{} & Excl. \\",
        r"\midrule",
    ]
    for label, row in compared:
        out.append(
            f"{label} & {float(row['learned_mean']):.4f} & "
            f"{float(row['baseline_mean']):.4f} & "
            f"{float(row['diff']):+.4f} & "
            f"[{float(row['ci_lo']):+.4f}, {float(row['ci_hi']):+.4f}] & "
            f"{float(row['fold_p_two_sided']):.4f} & "
            f"{row['wins']}/{row['n']} & "
            f"{row['fold_wins']}/{row['n_folds']} & "
            f"{float(row['welfare_ratio']):.3f} & "
            f"{float(row['learned_exclusion']):.4f} \\\\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Post hoc fourfold box-expansion sensitivity on held-out series. Bound-10 "
        r"is the primary analysis; bound 40 is post hoc. $\Delta$ is learned minus Equal Shares; "
        r"negative favors learned. Intervals use paired series; $p_{\rm fold}$ tests "
        r"fold-mean signs under dependence. S-win and F-win count series and "
        r"fold-mean wins. This is an outcome stability check, not coefficient identification or "
        r"independent replication.}",
        r"\label{tab:district-bound}\end{table}",
    ]
    return "\n".join(out)


def _frontier_table() -> str:
    """Generate the complete two-arm soft welfare-target sweep."""
    families = (
        (
            "Endowment family (mechanism retained)",
            RESULTS
            / "iclr_frontier"
            / "frontier_endowment_temporal_2022_seed42.json",
        ),
        (
            "Outcome family (mechanism discarded)",
            RESULTS
            / "iclr_frontier"
            / "frontier_outcome_temporal_2022_seed42.json",
        ),
    )
    payloads = []
    for label, path in families:
        if not path.exists():
            logger.warning("missing %s", path)
            return ""
        payloads.append((label, json.loads(path.read_text())))

    out = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        (
            r"& Welfare target $\tau$ & Worst $\csd$ & Mean $\csd$ "
            r"& Welfare & Exclusion \\"
        ),
        r"\midrule",
    ]
    for family_index, (label, payload) in enumerate(payloads):
        out.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{label}}}}}\\")
        for point in sorted(
            payload["points"], key=lambda row: float(row["floor"])
        ):
            out.append(
                f"& {float(point['floor']):.2f} & "
                f"{float(point['test_worst_csd']):.4f} & "
                f"{float(point['test_mean_csd']):.4f} & "
                f"{float(point['test_welfare']):,.0f} & "
                f"{float(point['test_exclusion']):.4f} \\\\"
            )
        if family_index + 1 < len(payloads):
            out.append(r"\midrule")
    out += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Sweeps over soft welfare targets for both arms at seed 42 on held-out "
        r"Warsaw districts. Worst $\csd$ is the equal-weight mean district-level worst-group "
        r"value; lower CSD and exclusion and higher welfare are "
        r"favorable. Targets are penalties, not hard constraints. This single-seed "
        r"descriptive sweep has no intervals; corpus- and budget-specific ranges "
        r"are not universal reachable sets.}",
        r"\label{tab:endowfrontier}",
        r"\end{table}",
    ]
    return "\n".join(out)


def _external_city_seed_table(evidence: Mapping[str, Any]) -> str:
    """Render all authenticated city-by-seed evidence without significance stars."""
    seeds = evidence.get("seeds")
    if not isinstance(seeds, Mapping) or set(seeds) != {"1", "2", "42"}:
        raise ValueError("external evidence must contain seeds 1, 2, and 42")
    city_order = ("Poland/Katowice", "Poland/Krakow")
    out = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        (
            r"Seed & City & $\Delta$ CSD $\downarrow$ & "
            r"95\% paired-bootstrap interval & Holm $p$ & "
            r"$\Delta$ exclusion $\downarrow$ & Welfare ratio $\uparrow$ \\"
        ),
        r"\midrule",
    ]
    for seed in ("1", "2", "42"):
        cities = seeds[seed]["cities"]
        if set(cities) != set(city_order):
            raise ValueError(f"external seed {seed} must contain both frozen cities")
        for city in city_order:
            values = cities[city]
            holm = (
                "--"
                if values.get("holm_p_value") is None
                else f"{float(values['holm_p_value']):.4f}"
            )
            out.append(
                f"{seed} & {city.split('/')[-1]} & "
                f"{float(values['difference']):.4f} & "
                f"[{float(values['ci_low']):.4f}, {float(values['ci_high']):.4f}] & "
                f"{holm} & {float(values['exclusion_difference']):+.4f} & "
                f"{float(values['welfare_ratio']):.4f} \\\\"
            )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Projected-support-set results by city and frozen seed, with "
        r"series as paired analysis units. $\Delta$ is learned endowment minus "
        r"Equal Shares, so negative values favor the learned map. Intervals use "
        r"paired-series bootstrap resampling, and Holm-adjusted $p$ values are "
        r"shown only for the primary seed. Ballot projection "
        r"limits the transfer interpretation.}",
        r"\label{tab:external-city-seed}\end{table}",
    ]
    return "\n".join(out)


def _external_gate_table(decision: Mapping[str, Any]) -> str:
    """Render every frozen mechanical condition with its pass/fail status."""
    conditions = decision.get("conditions")
    labels = {
        "macro_at_most_minus_0_005_all_seeds": (
            "city macro $\\leq -0.005$ for every seed"
        ),
        "negative_each_city_all_seeds": (
            "negative in each city for every seed"
        ),
        "primary_ci_excludes_zero_each_city": (
            "primary paired-bootstrap interval excludes zero in each city"
        ),
        "primary_holm_p_each_city": (
            "primary Holm $p \\leq 0.05$ in each city"
        ),
        "welfare_ratio_all_cities_seeds": (
            "welfare ratio $\\geq 0.98$ in each city and seed"
        ),
        "exclusion_increase_all_cities_seeds": (
            "exclusion increase $\\leq 0.005$ in each city and seed"
        ),
        "endowment_actuation": "endowment actuation $\\geq 8$ and greater than priority",
        "macro_seed_spread": "city-macro seed spread $\\leq 0.005$",
    }
    order = tuple(labels)
    if not isinstance(conditions, Mapping) or set(conditions) != set(order):
        raise ValueError("external decision must contain the eight frozen conditions")

    macro_values = conditions["macro_at_most_minus_0_005_all_seeds"]["observed"]
    city_differences = conditions["negative_each_city_all_seeds"]["observed"]
    ci_highs = conditions["primary_ci_excludes_zero_each_city"]["observed"]
    holm_values = conditions["primary_holm_p_each_city"]["observed"]
    welfare_values = conditions["welfare_ratio_all_cities_seeds"]["observed"]
    exclusion_values = conditions["exclusion_increase_all_cities_seeds"]["observed"]
    actuation = conditions["endowment_actuation"]["observed"]
    observed = {
        "macro_at_most_minus_0_005_all_seeds": (
            f"maximum {max(float(value) for value in macro_values.values()):.4f}"
        ),
        "negative_each_city_all_seeds": (
            f"{sum(float(value) < 0 for seed in city_differences.values() for value in seed.values())}/6 negative"
        ),
        "primary_ci_excludes_zero_each_city": ", ".join(
            f"{city.split('/')[-1]} {float(value):.4f}"
            for city, value in sorted(ci_highs.items())
        ),
        "primary_holm_p_each_city": ", ".join(
            f"{city.split('/')[-1]} {float(value):.4f}"
            for city, value in sorted(holm_values.items())
        ),
        "welfare_ratio_all_cities_seeds": (
            f"minimum {min(float(value) for seed in welfare_values.values() for value in seed.values()):.4f}"
        ),
        "exclusion_increase_all_cities_seeds": (
            f"maximum {max(float(value) for seed in exclusion_values.values() for value in seed.values()):.4f}"
        ),
        "endowment_actuation": (
            f"{actuation['endowment']} versus {actuation['priority']} series"
        ),
        "macro_seed_spread": (
            f"{float(conditions['macro_seed_spread']['observed']):.4f}"
        ),
    }
    out = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\begin{tabular}{lrl}",
        r"\toprule",
        r"Frozen condition & Observed & Status \\",
        r"\midrule",
    ]
    for name in order:
        status = r"\checkmark" if conditions[name]["passed"] else r"\texttimes"
        out.append(f"{labels[name]} & {observed[name]} & {status} \\\\ ")
    out += [
        r"\bottomrule\end{tabular}",
        rf"\caption{{One row per preregistered condition for the projected-support-set "
        rf"external study, judged in the direction printed in the condition. The "
        rf"deterministic joint status is \texttt{{{decision['classification']}}} "
        rf"because the verdict is conjunctive and not all conditions pass.}}",
        r"\label{tab:external-gates}\end{table}",
    ]
    return "\n".join(out)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_bound_lineage_files(
    results_root: Path,
    directory: str,
    version: str,
    tracked_files: Mapping[str, Any],
) -> dict[str, tuple[Path, str]]:
    names = {
        "protocol_lock": "protocol_lock.json",
        "published_anchor": "published_lock_anchor.json",
        "heldout_receipt": "heldout_opened.json",
        "abort_record": "abort_record.json",
    }
    verified: dict[str, tuple[Path, str]] = {}
    for key, filename in names.items():
        tracked_key = f"{version}/{key}"
        tracked = tracked_files.get(tracked_key)
        if not isinstance(tracked, Mapping):
            raise ValueError(f"missing bound lineage entry {tracked_key}")
        expected_path = f"results/{directory}/{filename}"
        if tracked.get("path") != expected_path:
            raise ValueError(f"lineage path disagreement for {tracked_key}")
        path = results_root / directory / filename
        actual_digest = sha256_file(path)
        if actual_digest != tracked.get("sha256"):
            raise ValueError(f"lineage SHA-256 mismatch for {tracked_key}")
        verified[key] = (path, actual_digest)
    return verified


def _verify_attempt_links(
    version: str,
    records: Mapping[str, Mapping[str, Any]],
    digests: Mapping[str, str],
) -> None:
    lock_digest = digests["protocol_lock"]
    if records["published_anchor"].get("lock_sha256") != lock_digest:
        raise ValueError(f"{version} anchor does not bind its protocol lock")
    if records["heldout_receipt"].get("protocol_lock_sha256") != lock_digest:
        raise ValueError(f"{version} receipt does not bind its protocol lock")
    if records["abort_record"].get("protocol_lock_sha256") != lock_digest:
        raise ValueError(f"{version} abort does not bind its protocol lock")
    if version == "v2":
        if (
            records["abort_record"].get("published_anchor_sha256")
            != digests["published_anchor"]
            or records["abort_record"].get("heldout_receipt_sha256")
            != digests["heldout_receipt"]
        ):
            raise ValueError("v2 abort does not bind its anchor and receipt")


def _external_protocol_lineage_table(results_root: Path) -> str:
    """Authenticate and render the v1/v2 aborts and valid v3 result."""
    bundle = load_and_verify_bundle(
        results_root / "iclr_external_support_set_amendment"
    )
    v2_files = _verify_bound_lineage_files(
        results_root,
        "iclr_external_support_set",
        "v2",
        bundle.lock["tracked_files"],
    )
    v2_records = {
        key: _read_json_object(path, f"v2 {key}")
        for key, (path, _) in v2_files.items()
    }
    v2_digests = {key: digest for key, (_, digest) in v2_files.items()}
    _verify_attempt_links("v2", v2_records, v2_digests)
    if (
        bundle.receipt.get("v2_protocol_lock_sha256")
        != v2_digests["protocol_lock"]
        or bundle.receipt.get("v2_abort_record_sha256")
        != v2_digests["abort_record"]
    ):
        raise ValueError("v3 receipt does not bind the v2 abort lineage")

    v1_files = _verify_bound_lineage_files(
        results_root,
        "iclr_external_validation",
        "v1",
        v2_records["protocol_lock"]["tracked_files"],
    )
    v1_records = {
        key: _read_json_object(path, f"v1 {key}")
        for key, (path, _) in v1_files.items()
    }
    v1_digests = {key: digest for key, (_, digest) in v1_files.items()}
    _verify_attempt_links("v1", v1_records, v1_digests)

    attempts = (
        (
            "v1",
            v1_digests["protocol_lock"][:12],
            "approval-only abort",
            "No---no scientific result",
            str(v1_records["abort_record"]["classification"]),
        ),
        (
            "v2",
            v2_digests["protocol_lock"][:12],
            "Unicode-label technical abort",
            "No---no scientific result",
            str(v2_records["abort_record"]["classification"]),
        ),
        (
            "v3",
            sha256_file(bundle.root / "protocol_lock.json")[:12],
            "valid amended run",
            "Yes",
            str(bundle.decision["classification"]),
        ),
    )
    out = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Version & Lock SHA-256 prefix & Stage & Metrics computed & Classification \\",
        r"\midrule",
    ]
    for version, digest, stage, metrics, classification in attempts:
        out.append(
            rf"\texttt{{{version}}} & \texttt{{{digest}}} & {stage} & {metrics} & {_esc(classification)} \\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{v1 and v2 ended before metrics were computed. Only v3 "
        r"contributes results. Matching hashes authenticate the artifacts, not "
        r"the transfer claim.}",
        r"\label{tab:external-lineage}\end{table}",
    ]
    return "\n".join(out)


def main() -> None:
    external_bundle, _ = _load_external_support_set_inputs()
    blocks = [
        "% AUTO-GENERATED by tempopb/src/gen_iclr_appendix.py -- do not hand-edit.",
        "% Regenerate after any change to the underlying results CSVs.",
        "",
        _payment_intervention_table(),
        "",
        _external_city_seed_table(external_bundle.evidence),
        "",
        _external_gate_table(external_bundle.decision),
        "",
        _external_protocol_lineage_table(RESULTS),
        "",
        _per_series_table(),
        "",
        _res_grid_table(),
        "",
        _seed_table(),
        "",
        _scheme_table(),
        "",
        _cross_district_table(),
        "",
        _cross_district_bound_table(),
        "",
        _crosscity_table(),
        "",
        _two_arm_per_series_table(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(b for b in blocks if b is not None) + "\n")
    logger.info("wrote %s", OUT)
    FRONTIER_OUT.write_text(_frontier_table() + "\n")
    logger.info("wrote %s", FRONTIER_OUT)


if __name__ == "__main__":
    main()
