"""Locked multi-city evaluation statistics and mechanical evidence decision."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from iclr_corpus import (
    SeriesRef,
    Split,
    load_series,
)
from iclr_env import (
    EnvConfig,
    EpisodeResult,
    SelectorPolicy,
    endowment_selector,
    res_policy,
    rollout_selector,
    uniform_policy,
)
from iclr_multicity_protocol import (
    FIT_SOURCE_FILES,
    PABULIB_COMMIT,
    _sha256,
    evidence_gate_config,
    load_canonical_index_from_manifest,
    load_frozen_split,
    locked_fit_inventory,
    open_heldout_once,
    verify_multicity_protocol_lock,
    verify_structural_gates,
    write_immutable_json_artifact,
    write_immutable_text_artifact,
)
from iclr_multicity_train import build_multicity_arm
from iclr_outcome import (
    cost_effective_weights,
    greedy_equivalent_weights,
    score_selector,
)
from iclr_stats import (
    exact_paired_sign_flip_pvalue,
    monte_carlo_paired_sign_flip_pvalue,
)
from parse_pb import PBInstance

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260812
SIGN_FLIP_DRAWS = 200_000
SIGN_FLIP_SEED = 20260812
EXACT_SIGN_FLIP_LIMIT = 24
TIE_TOLERANCE = 1e-12
PRIMARY_SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results" / "iclr_multicity"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluatedSeries:
    """Episode metrics plus exact winner fingerprints on scored editions."""

    episode: EpisodeResult
    scored_outcomes: Tuple[Tuple[int, Tuple[str, ...]], ...]


def _city(series: str) -> str:
    parts = series.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else series


def evaluate_series(
    ref: SeriesRef,
    selector: SelectorPolicy,
    score_years: Sequence[int],
    instances: Mapping[int, PBInstance],
    cfg: EnvConfig,
) -> EvaluatedSeries:
    """Replay one series once while recording only scored winner sets."""

    score_set = set(score_years)
    outcomes: Dict[int, Tuple[str, ...]] = {}

    def recording_selector(inst: PBInstance, state) -> set[str]:
        winners = set(selector(inst, state))
        if inst.year in score_set:
            outcomes[int(inst.year)] = tuple(sorted(winners))
        return winners

    episode = rollout_selector(
        ref,
        recording_selector,
        score_years=score_years,
        cfg=cfg,
        instances=dict(instances),
    )
    return EvaluatedSeries(
        episode=episode,
        scored_outcomes=tuple((year, outcomes[year]) for year in sorted(outcomes)),
    )


def count_actuated_series(
    learned: Sequence[EvaluatedSeries], baseline: Sequence[EvaluatedSeries]
) -> int:
    """Count series with any scored winner-set difference from the baseline."""

    baseline_by_series = {row.episode.series: row for row in baseline}
    count = 0
    for row in learned:
        other = baseline_by_series.get(row.episode.series)
        if other is None:
            raise ValueError(f"missing baseline series {row.episode.series}")
        if row.scored_outcomes != other.scored_outcomes:
            count += 1
    return count


def required_fit_specs() -> list[Dict[str, object]]:
    """Return the frozen 24-run primary and matched-control inventory."""

    return locked_fit_inventory()


def required_fit_inventory(
    result_root: Path,
) -> list[Tuple[Dict[str, object], Path]]:
    """Pair every required fit specification with its deterministic path."""

    result_root = Path(result_root)
    return [
        (
            spec,
            result_root
            / "fits"
            / str(spec["split"])
            / str(spec["arm"])
            / f"seed-{spec['seed']}.json",
        )
        for spec in required_fit_specs()
    ]


def verify_lock_fit_inventory(lock_payload: Mapping[str, object]) -> None:
    """Require the lock and evaluator to name the same 24 fit specifications."""

    def canonical(rows):
        return sorted(
            (str(row["split"]), str(row["arm"]), int(row["seed"]))
            for row in rows
        )

    locked = canonical(lock_payload.get("fit_inventory", []))
    evaluator = canonical(required_fit_specs())
    if locked != evaluator:
        raise RuntimeError(
            f"protocol-lock fit inventory differs from evaluator: "
            f"locked={locked}, evaluator={evaluator}"
        )


def verify_fit_inventory(
    inventory: Sequence[Tuple[Mapping[str, object], Path]],
    lock_payload: Mapping[str, object],
    lock_sha256: str,
) -> None:
    """Reject missing, non-training, or hyperparameter-drifted fit artifacts."""

    expected_fixed = {
        "generations": 30,
        "popsize": None,
        "sigma0": 0.4,
        "bound": 10.0,
        "welfare_penalty": 2.0,
        "welfare_floor": 1.0,
    }
    feature_counts = {"priority": 5, "endowment": 6, "outcome": 5}
    for spec, path in inventory:
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(f"missing required fit: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("training_only") is not True:
            raise RuntimeError(f"fit is not training-only: {path}")
        if payload.get("execution_mode") != "serial":
            raise RuntimeError(f"fit has invalid execution_mode: {path}")
        config = payload.get("config", {})
        for key in ("split", "arm", "seed"):
            if config.get(key) != spec[key]:
                raise RuntimeError(
                    f"fit {path} has {key}={config.get(key)!r}, expected {spec[key]!r}"
                )
        for key, expected in expected_fixed.items():
            if config.get(key) != expected:
                raise RuntimeError(
                    f"fit {path} has {key}={config.get(key)!r}, expected {expected!r}"
                )
        weights = np.asarray(payload.get("result", {}).get("best_weights", []), dtype=float)
        expected_count = feature_counts[str(spec["arm"])]
        if weights.shape != (expected_count,) or not np.all(np.isfinite(weights)):
            raise RuntimeError(f"fit {path} has invalid best_weights")
        arm = build_multicity_arm(str(spec["arm"]))
        expected_arm = {
            "name": arm.name,
            "feature_names": list(arm.feature_names),
            "init_name": arm.init_name,
            "initial": arm.initial.tolist(),
        }
        if payload.get("arm") != expected_arm:
            raise RuntimeError(f"fit {path} has invalid arm metadata")
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            raise RuntimeError(f"fit {path} is missing provenance")
        tracked = lock_payload.get("tracked_files", {})
        expected_sources = {
            name: tracked.get(f"source/{name}", {}).get("sha256")
            for name in FIT_SOURCE_FILES
        }
        if any(value is None for value in expected_sources.values()):
            raise RuntimeError("protocol lock is missing fit source hashes")
        expected_provenance = {
            "protocol_lock_sha256": lock_sha256,
            "pabulib_commit": PABULIB_COMMIT,
            "corpus_manifest_sha256": tracked.get(
                "artifact/corpus_manifest", {}
            ).get("sha256"),
            "split_sha256": tracked.get(
                f"split/{spec['split']}", {}
            ).get("sha256"),
            "source_sha256": expected_sources,
        }
        if provenance != expected_provenance:
            raise RuntimeError(
                f"fit {path} provenance differs from protocol lock"
            )


def _fit_selector(path: Path, cfg: EnvConfig) -> SelectorPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arm = build_multicity_arm(str(payload["config"]["arm"]))
    weights = np.asarray(payload["result"]["best_weights"], dtype=float)
    return arm.selector(weights, cfg)


def evaluate_policy_on_split(
    split: Split,
    index: Mapping[str, SeriesRef],
    instances: Mapping[str, Mapping[int, PBInstance]],
    selector: SelectorPolicy,
    cfg: EnvConfig,
) -> list[EvaluatedSeries]:
    """Evaluate one frozen policy on one scored split with full warm-up history."""

    rows = []
    for key, score_years in split.test:
        rows.append(
            evaluate_series(
                index[key], selector, score_years, instances[key], cfg
            )
        )
    return rows


def _evaluated_payload(row: EvaluatedSeries) -> Dict[str, object]:
    payload = asdict(row.episode)
    payload["scored_outcomes"] = [
        {"year": year, "winners": list(winners)}
        for year, winners in row.scored_outcomes
    ]
    return payload


def _evidence_payload(
    evaluated: Mapping[str, Mapping[str, Sequence[EvaluatedSeries]]],
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
    containment_pass: bool,
) -> Dict[str, object]:
    temporal_summary = summaries["temporal_2022"]
    temporal_seeds: Dict[str, object] = {}
    for seed in (1, 2, 42):
        rows = temporal_summary[f"priority/seed-{seed}"]
        temporal_seeds[str(seed)] = {
            "cities": {
                city: dict(row)
                for city, row in rows.items()
                if city != "CITY_MACRO"
            },
            "city_macro_difference": float(rows["CITY_MACRO"]["difference"]),
        }

    city_out_seeds: Dict[str, Dict[str, object]] = {
        str(seed): {"cities": {}} for seed in (1, 2, 42)
    }
    for city in ("Poland_Warszawa", "Poland_Gdynia", "Poland_Łódź"):
        split_name = f"city_out_{city}"
        for seed in (1, 2, 42):
            rows = summaries[split_name][f"priority/seed-{seed}"]
            city_rows = {
                name: dict(row) for name, row in rows.items() if name != "CITY_MACRO"
            }
            if len(city_rows) != 1:
                raise RuntimeError(
                    f"{split_name} seed {seed} should contain one held city, "
                    f"found {sorted(city_rows)}"
                )
            city_out_seeds[str(seed)]["cities"].update(city_rows)

    temporal_results = evaluated["temporal_2022"]
    mes = temporal_results["mes"]
    priority_actuated = count_actuated_series(
        temporal_results[f"priority/seed-{PRIMARY_SEED}"], mes
    )
    endowment_actuated = count_actuated_series(
        temporal_results[f"endowment/seed-{PRIMARY_SEED}"], mes
    )
    return {
        "containment_pass": bool(containment_pass),
        "primary_seed": str(PRIMARY_SEED),
        "temporal": {
            "seeds": temporal_seeds,
            "priority_actuated_series": priority_actuated,
            "endowment_actuated_series": endowment_actuated,
        },
        "city_out": {"seeds": city_out_seeds},
    }


def _write_per_series_csv(
    path: Path,
    evaluated: Mapping[str, Mapping[str, Sequence[EvaluatedSeries]]],
) -> None:
    fields = (
        "split",
        "policy",
        "series",
        "city",
        "scored_years",
        "worst_csd",
        "worst_cohort",
        "mean_csd",
        "welfare",
        "cost_welfare",
        "exclusion",
        "outcome_sha256",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for split_name, policies in sorted(evaluated.items()):
        for policy, rows in sorted(policies.items()):
            for row in rows:
                episode = row.episode
                encoded = json.dumps(
                    row.scored_outcomes,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                writer.writerow(
                    {
                        "split": split_name,
                        "policy": policy,
                        "series": episode.series,
                        "city": _city(episode.series),
                        "scored_years": " ".join(map(str, episode.scored_years)),
                        "worst_csd": episode.worst_csd,
                        "worst_cohort": episode.worst_cohort,
                        "mean_csd": episode.mean_csd,
                        "welfare": episode.welfare,
                        "cost_welfare": episode.cost_welfare,
                        "exclusion": episode.exclusion,
                        "outcome_sha256": hashlib.sha256(encoded).hexdigest(),
                    }
                )
    write_immutable_text_artifact(path, handle.getvalue())


def run_locked_evaluation(
    result_root: Path,
    index: Mapping[str, SeriesRef],
    splits: Mapping[str, Split],
    inventory: Sequence[Tuple[Mapping[str, object], Path]],
    containment_pass: bool,
    receipt_path: Path,
    evidence_gates: Mapping[str, object],
) -> Dict[str, object]:
    """Evaluate all frozen policies after the one-way receipt exists."""

    if not Path(receipt_path).is_file():
        raise RuntimeError("held-out receipt must exist before evaluation")
    cfg = EnvConfig()
    instances = {key: load_series(ref) for key, ref in sorted(index.items())}
    inventory_by_split: Dict[str, list[Tuple[Mapping[str, object], Path]]] = defaultdict(list)
    for spec, path in inventory:
        inventory_by_split[str(spec["split"])].append((spec, path))

    evaluated: Dict[str, Dict[str, list[EvaluatedSeries]]] = {}
    fixed = {
        "mes": endowment_selector(uniform_policy, cfg),
        "res-1.0": endowment_selector(res_policy(1.0), cfg),
        "greedy-count": score_selector(greedy_equivalent_weights()),
        "greedy-cost": score_selector(cost_effective_weights()),
    }
    for split_name, split in sorted(splits.items()):
        policies: Dict[str, SelectorPolicy] = dict(fixed)
        for spec, path in inventory_by_split[split_name]:
            name = f"{spec['arm']}/seed-{spec['seed']}"
            policies[name] = _fit_selector(path, cfg)
        evaluated[split_name] = {}
        for name, selector in sorted(policies.items()):
            logger.info("evaluating %s on %s", name, split_name)
            evaluated[split_name][name] = evaluate_policy_on_split(
                split, index, instances, selector, cfg
            )

    summaries: Dict[str, Dict[str, Dict[str, object]]] = {}
    for split_name, policies in evaluated.items():
        baseline = [row.episode for row in policies["mes"]]
        summaries[split_name] = {}
        for name, rows in policies.items():
            if name == "mes":
                continue
            summaries[split_name][name] = paired_city_summary(
                [row.episode for row in rows],
                baseline,
                include_inference=(split_name == "temporal_2022"),
            )

    primary_rows = summaries["temporal_2022"][f"priority/seed-{PRIMARY_SEED}"]
    primary_pvalues = {
        city: float(row["p_value"])
        for city, row in primary_rows.items()
        if city != "CITY_MACRO" and "p_value" in row
    }
    adjusted = holm_adjust(primary_pvalues)
    for city, value in adjusted.items():
        primary_rows[city]["holm_p_value"] = value

    evidence = _evidence_payload(evaluated, summaries, containment_pass)
    decision = evaluate_gold_gate(evidence, evidence_gates)
    evaluation_dir = Path(result_root) / "evaluation"
    per_series_payload = {
        split_name: {
            policy: [_evaluated_payload(row) for row in rows]
            for policy, rows in sorted(policies.items())
        }
        for split_name, policies in sorted(evaluated.items())
    }
    write_immutable_json_artifact(
        evaluation_dir / "per_series.json", per_series_payload
    )
    _write_per_series_csv(evaluation_dir / "per_series.csv", evaluated)
    write_immutable_json_artifact(evaluation_dir / "summary.json", summaries)
    write_immutable_json_artifact(
        evaluation_dir / "evidence_payload.json", evidence
    )
    write_immutable_json_artifact(
        Path(result_root) / "evidence_decision.json", decision
    )
    return decision


def main(argv: Sequence[str] | None = None) -> None:
    """Verify the lock, create the receipt, then open scored outcomes once."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result_root = args.result_root.resolve()
    inventory = required_fit_inventory(result_root)
    lock_path = result_root / "protocol_lock.json"
    receipt_path = result_root / "heldout_opened.json"
    structural_path = result_root / "structural_gates.json"
    lock_payload = verify_multicity_protocol_lock(
        lock_path,
        ROOT,
        result_root,
        ROOT / "data" / "pb_multicity",
    )
    verify_lock_fit_inventory(lock_payload)
    verify_structural_gates(structural_path)
    verify_fit_inventory(inventory, lock_payload, _sha256(lock_path))
    open_heldout_once(
        lock_path, receipt_path, inventory, structural_path, ROOT
    )
    index = load_canonical_index_from_manifest(
        result_root / "corpus_manifest.json", ROOT / "data" / "pb_multicity"
    )
    splits = {
        path.stem: load_frozen_split(path, index)
        for path in sorted((result_root / "splits").glob("*.json"))
    }
    structural = verify_structural_gates(structural_path)
    decision = run_locked_evaluation(
        result_root,
        index,
        splits,
        inventory,
        structural.get("status") == "pass",
        receipt_path,
        lock_payload["evidence_gates"],
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


def _paired_bootstrap_interval(
    differences: np.ndarray, draws: int, seed: int
) -> tuple[float, float]:
    if differences.size == 0:
        raise ValueError("paired bootstrap requires at least one difference")
    if differences.size == 1:
        value = float(differences[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    chunk = 1000
    for start in range(0, draws, chunk):
        width = min(chunk, draws - start)
        indices = rng.integers(0, differences.size, size=(width, differences.size))
        means[start : start + width] = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _paired_pvalue(differences: np.ndarray) -> tuple[float, str]:
    if differences.size <= EXACT_SIGN_FLIP_LIMIT:
        return exact_paired_sign_flip_pvalue(differences), "exact_sign_flip"
    return (
        monte_carlo_paired_sign_flip_pvalue(
            differences, draws=SIGN_FLIP_DRAWS, seed=SIGN_FLIP_SEED
        ),
        "monte_carlo_sign_flip",
    )


def paired_city_summary(
    learned: Sequence[EpisodeResult],
    baseline: Sequence[EpisodeResult],
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    include_inference: bool = True,
) -> Dict[str, Dict[str, float | int | str]]:
    """Compute paired series effects per city and an unweighted city macro."""

    learned_by_series = {episode.series: episode for episode in learned}
    baseline_by_series = {episode.series: episode for episode in baseline}
    common = sorted(set(learned_by_series) & set(baseline_by_series))
    grouped: Dict[str, list[tuple[EpisodeResult, EpisodeResult]]] = defaultdict(list)
    for series in common:
        left = learned_by_series[series]
        right = baseline_by_series[series]
        if left.worst_csd is None or right.worst_csd is None:
            continue
        grouped[_city(series)].append((left, right))
    if not grouped:
        raise ValueError("no paired scoreable series")

    rows: Dict[str, Dict[str, float | int | str]] = {}
    for city, pairs in sorted(grouped.items()):
        differences = np.asarray(
            [left.worst_csd - right.worst_csd for left, right in pairs],
            dtype=float,
        )
        low, high = _paired_bootstrap_interval(
            differences, bootstrap_draws, bootstrap_seed
        )
        learned_welfare = sum(left.welfare for left, _ in pairs)
        baseline_welfare = sum(right.welfare for _, right in pairs)
        row: Dict[str, float | int | str] = {
            "n_series": len(pairs),
            "learned_csd": float(np.mean([left.worst_csd for left, _ in pairs])),
            "baseline_csd": float(np.mean([right.worst_csd for _, right in pairs])),
            "difference": float(differences.mean()),
            "ci_low": low,
            "ci_high": high,
            "wins": int(np.count_nonzero(differences < -TIE_TOLERANCE)),
            "ties": int(np.count_nonzero(np.abs(differences) <= TIE_TOLERANCE)),
            "losses": int(np.count_nonzero(differences > TIE_TOLERANCE)),
            "welfare_ratio": (
                float(learned_welfare / baseline_welfare)
                if baseline_welfare > 0
                else float("nan")
            ),
            "exclusion_difference": float(
                np.mean([left.exclusion - right.exclusion for left, right in pairs])
            ),
        }
        if include_inference:
            pvalue, kind = _paired_pvalue(differences)
            row["p_value"] = float(pvalue)
            row["p_value_kind"] = kind
        rows[city] = row

    city_rows = list(rows.values())
    macro: Dict[str, float | int | str] = {
        "n_cities": len(city_rows),
        "n_series": int(sum(int(row["n_series"]) for row in city_rows)),
        "learned_csd": float(np.mean([float(row["learned_csd"]) for row in city_rows])),
        "baseline_csd": float(np.mean([float(row["baseline_csd"]) for row in city_rows])),
        "difference": float(np.mean([float(row["difference"]) for row in city_rows])),
        "welfare_ratio": float(np.mean([float(row["welfare_ratio"]) for row in city_rows])),
        "exclusion_difference": float(
            np.mean([float(row["exclusion_difference"]) for row in city_rows])
        ),
        "wins": int(sum(int(row["wins"]) for row in city_rows)),
        "ties": int(sum(int(row["ties"]) for row in city_rows)),
        "losses": int(sum(int(row["losses"]) for row in city_rows)),
    }
    rows["CITY_MACRO"] = macro
    return rows


def holm_adjust(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm family-wise correction with monotone adjusted p-values."""

    ordered = sorted(pvalues.items(), key=lambda item: (item[1], item[0]))
    adjusted: Dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * float(pvalue))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _condition(passed: bool, observed, threshold) -> Dict[str, object]:
    return {"passed": bool(passed), "observed": observed, "threshold": threshold}


def evaluate_gold_gate(
    payload: Mapping[str, object],
    evidence_gates: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Apply every frozen gold condition without researcher discretion."""

    gates = evidence_gates or evidence_gate_config()
    gold = gates["gold"]
    silver_gate = gates["silver"]
    primary_seed = str(payload["primary_seed"])
    temporal = payload["temporal"]
    temporal_seeds = temporal["seeds"]
    primary = temporal_seeds[primary_seed]
    primary_cities = primary["cities"]
    city_out_seeds = payload["city_out"]["seeds"]

    primary_differences = {
        city: float(row["difference"]) for city, row in primary_cities.items()
    }
    primary_macro = float(primary["city_macro_difference"])
    city_out_differences = {
        seed: {
            city: float(row["difference"])
            for city, row in seed_row["cities"].items()
        }
        for seed, seed_row in city_out_seeds.items()
    }
    welfare_ratios = {
        city: float(row["welfare_ratio"]) for city, row in primary_cities.items()
    }
    exclusion_differences = {
        city: float(row["exclusion_difference"])
        for city, row in primary_cities.items()
    }
    all_temporal_directions = {
        seed: {
            city: float(row["difference"])
            for city, row in seed_row["cities"].items()
        }
        for seed, seed_row in temporal_seeds.items()
    }
    macro_values = [
        float(seed_row["city_macro_difference"])
        for seed_row in temporal_seeds.values()
    ]
    macro_spread = max(macro_values) - min(macro_values)
    priority_actuation = int(temporal["priority_actuated_series"])
    endowment_actuation = int(temporal["endowment_actuated_series"])

    conditions = {
        "exact_mes_containment": _condition(
            bool(payload["containment_pass"]), bool(payload["containment_pass"]), True
        ),
        "temporal_negative_each_city": _condition(
            all(value < 0 for value in primary_differences.values()),
            primary_differences,
            "difference < 0 in every city at primary seed",
        ),
        "temporal_macro_at_most_minus_0_010": _condition(
            primary_macro <= float(gold["city_macro_difference_max"]),
            primary_macro,
            f"<= {gold['city_macro_difference_max']}",
        ),
        "city_out_negative_each_city_and_seed": _condition(
            all(
                value < 0
                for seed_row in city_out_differences.values()
                for value in seed_row.values()
            ),
            city_out_differences,
            "difference < 0 for every held city and seed",
        ),
        "welfare_ratio_each_city": _condition(
            all(
                value >= float(gold["welfare_ratio_min_each_city"])
                for value in welfare_ratios.values()
            ),
            welfare_ratios,
            f">= {gold['welfare_ratio_min_each_city']} in every city",
        ),
        "exclusion_increase_each_city": _condition(
            all(
                value <= float(gold["exclusion_increase_max_each_city"])
                for value in exclusion_differences.values()
            ),
            exclusion_differences,
            f"<= {gold['exclusion_increase_max_each_city']} in every city",
        ),
        "temporal_negative_each_city_all_seeds": _condition(
            all(
                value < 0
                for seed_row in all_temporal_directions.values()
                for value in seed_row.values()
            ),
            all_temporal_directions,
            "difference < 0 in every city and seed",
        ),
        "temporal_macro_seed_spread": _condition(
            macro_spread <= float(gold["city_macro_seed_spread_max"]),
            macro_spread,
            f"<= {gold['city_macro_seed_spread_max']}",
        ),
        "priority_actuation_greater_than_endowment": _condition(
            priority_actuation > endowment_actuation,
            {"priority": priority_actuation, "endowment": endowment_actuation},
            "priority > endowment",
        ),
    }
    gold_pass = all(row["passed"] for row in conditions.values())

    primary_city_out = city_out_differences.get(primary_seed, {})
    nonnegative_city_out = sum(value >= 0 for value in primary_city_out.values())
    macro_magnitude = abs(primary_macro)
    silver = (
        not gold_pass
        and all(value < 0 for value in primary_differences.values())
        and float(silver_gate["city_macro_difference_min_magnitude"])
        <= macro_magnitude
        <= float(silver_gate["city_macro_difference_max_magnitude"])
        and all(
            value >= float(gold["welfare_ratio_min_each_city"])
            for value in welfare_ratios.values()
        )
        and all(
            value <= float(gold["exclusion_increase_max_each_city"])
            for value in exclusion_differences.values()
        )
        and nonnegative_city_out
        <= int(silver_gate["maximum_null_city_out_folds"])
    )
    return {
        "classification": "gold" if gold_pass else ("silver" if silver else "falsified"),
        "gold_pass": gold_pass,
        "primary_seed": primary_seed,
        "conditions": conditions,
    }


if __name__ == "__main__":
    main()
