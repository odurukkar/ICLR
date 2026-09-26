"""Pool five district-held-out endowment fits into one out-of-district result.

Each district is evaluated only by the model fit on the other four folds.  The
script verifies exact fold coverage, replays every fold-specific policy, and
reports descriptive paired-series dispersion plus a fold-block dependence
sensitivity check against fixed executable baselines.

Default inputs:
    results/iclr_train/run_district_endow_f{fold}_g25.json

Outputs:
    results/iclr_cross_district_per_series.csv
    results/iclr_cross_district_summary.csv
    results/iclr_cross_district_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from iclr_corpus import build_series_index, load_split
from iclr_env import (
    EnvConfig,
    SelectorPolicy,
    endowment_selector,
    res_policy,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import (
    cost_effective_weights,
    greedy_equivalent_weights,
    llmrule_card_selector,
    llmrule_cost_selector,
    score_selector,
)
from iclr_stats import exact_paired_sign_flip_pvalue
from iclr_train import _load_series_data, build_arm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "results" / "iclr_train"
N_BOOT = 10_000


def output_paths(tag: str = "") -> tuple[Path, Path, Path]:
    """Return non-overlapping output paths for a primary or sensitivity run."""
    if tag and not all(char.isalnum() or char in {"-", "_"} for char in tag):
        raise ValueError(f"unsafe output tag: {tag!r}")
    stem = "iclr_cross_district" + (f"_{tag}" if tag else "")
    results = ROOT / "results"
    return (
        results / f"{stem}_per_series.csv",
        results / f"{stem}_summary.csv",
        results / f"{stem}_manifest.json",
    )


def boundary_feature_names(
    weights: Sequence[float],
    feature_names: Sequence[str],
    *,
    bound: float,
    atol: float = 1e-6,
) -> list[str]:
    """Return features whose fitted coefficient lies on the search box."""
    values = np.asarray(weights, dtype=float)
    if values.shape != (len(feature_names),):
        raise ValueError(
            f"weight/name shape mismatch: {values.shape} vs {len(feature_names)} names"
        )
    return [
        name
        for name, value in zip(feature_names, values)
        if abs(abs(float(value)) - bound) <= atol
    ]


def validate_replay_summary(
    rows: Sequence[Mapping[str, object]],
    saved: Mapping[str, object],
    *,
    atol: float = 1e-12,
) -> None:
    """Reject a replay that no longer reproduces its saved fold aggregate."""
    if not rows:
        raise ValueError("no learned replay rows for fold")
    observed = {
        "n_series": float(len(rows)),
        "worst_csd": float(np.mean([float(row["worst_csd"]) for row in rows])),
        "mean_csd": float(np.mean([float(row["mean_csd"]) for row in rows])),
        "welfare": float(sum(float(row["welfare"]) for row in rows)),
        "cost_welfare": float(sum(float(row["cost_welfare"]) for row in rows)),
        "exclusion": float(np.mean([float(row["exclusion"]) for row in rows])),
    }
    for metric, value in observed.items():
        try:
            target = float(saved[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing or invalid saved replay metric: {metric}") from exc
        if not np.isclose(value, target, rtol=0.0, atol=atol):
            raise ValueError(
                f"replayed {metric} differs from saved fold result: "
                f"{value:.16g} vs {target:.16g}"
            )


def validate_fold_coverage(
    memberships: Mapping[str, Sequence[str]], expected_keys: Sequence[str]
) -> None:
    """Raise unless folds form an exact, disjoint partition of expected keys."""
    observed = [key for keys in memberships.values() for key in keys]
    counts = Counter(observed)
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    expected = set(expected_keys)
    present = set(observed)
    missing = sorted(expected - present)
    unexpected = sorted(present - expected)
    if duplicated:
        raise ValueError(f"duplicated test-series keys across folds: {duplicated}")
    if missing:
        raise ValueError(f"missing test-series keys: {missing}")
    if unexpected:
        raise ValueError(f"unexpected test-series keys: {unexpected}")


def validate_run_config(
    config: Mapping[str, object],
    *,
    expected_split: str,
    expected_seed: int,
    expected_generations: int,
    expected_bound: float = 10.0,
    allow_missing_default_bound: bool = False,
) -> None:
    """Reject a run that does not match the frozen primary district-CV fit."""
    expected = {
        "arm": "endowment",
        "split": expected_split,
        "seed": expected_seed,
        "generations": expected_generations,
        "popsize": None,
        "sigma0": 0.4,
        "bound": expected_bound,
        "init": "res",
        "welfare_penalty": 0.0,
        "welfare_floor": 1.0,
    }
    for field, target in expected.items():
        if (
            field == "bound"
            and field not in config
            and allow_missing_default_bound
            and expected_bound == 10.0
        ):
            continue
        observed = config.get(field)
        if observed != target:
            raise ValueError(
                f"wrong {field} in district-CV run: expected {target!r}, "
                f"found {observed!r}"
            )


def validate_run_counts(
    payload: Mapping[str, object],
    *,
    expected_train: int,
    expected_test: int,
) -> None:
    """Reject a run whose recorded train/test population does not match its split."""
    expected = {
        "n_train_series": expected_train,
        "n_test_series": expected_test,
    }
    for field, target in expected.items():
        try:
            observed = int(payload.get(field, -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field} in district-CV run") from exc
        if observed != target:
            raise ValueError(
                f"wrong {field} in district-CV run: expected {target}, "
                f"found {observed}"
            )


def validate_feature_names(
    observed: Sequence[object],
    expected: Sequence[str],
) -> None:
    """Reject weights produced under a missing or reordered feature schema."""
    if list(observed) != list(expected):
        raise ValueError(
            f"wrong feature_names in district-CV run: expected {list(expected)!r}, "
            f"found {list(observed)!r}"
        )


def paired_summary(
    learned: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 42,
    n_boot: int = N_BOOT,
) -> Dict[str, float | int]:
    """Summarize learned-minus-baseline paired CSD differences."""
    a = np.asarray(learned, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired vectors have different shapes: {a.shape} vs {b.shape}")
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    if a.size == 0:
        raise ValueError("no finite paired observations")
    diff = a - b
    rng = np.random.default_rng(seed)
    draws = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
    return {
        "learned_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "diff": float(diff.mean()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_two_sided": exact_paired_sign_flip_pvalue(diff),
        "wins": int(np.count_nonzero(diff < 0)),
        "ties": int(np.count_nonzero(diff == 0)),
        "n": int(diff.size),
    }


def fold_block_summary(
    learned: Sequence[float],
    baseline: Sequence[float],
    folds: Sequence[int],
) -> Dict[str, float | int]:
    """Test fold-mean effects so shared fitted maps are not ignored.

    The ordinary pooled summary treats series as descriptive units. Series in
    one held-out fold share a fitted policy, so we also summarize the five fold
    means. With five folds the exact two-sided test has necessarily coarse
    resolution. The training sets overlap, so the fold fits are not independent
    replications; this is a dependence sensitivity check, not a replacement
    experiment with five independent training samples.
    """
    a = np.asarray(learned, dtype=float)
    b = np.asarray(baseline, dtype=float)
    fold_ids = np.asarray(folds)
    if a.shape != b.shape or a.shape != fold_ids.shape:
        raise ValueError(
            "learned, baseline, and fold vectors must have identical shapes"
        )
    keep = np.isfinite(a) & np.isfinite(b)
    a = a[keep]
    b = b[keep]
    fold_ids = fold_ids[keep]
    if a.size == 0:
        raise ValueError("no finite paired observations")
    unique_folds = sorted(set(fold_ids.tolist()))
    fold_diffs = np.asarray(
        [float(np.mean((a - b)[fold_ids == fold])) for fold in unique_folds]
    )
    return {
        "fold_mean_diff": float(fold_diffs.mean()),
        "fold_p_two_sided": exact_paired_sign_flip_pvalue(fold_diffs),
        "fold_wins": int(np.count_nonzero(fold_diffs < 0)),
        "n_folds": int(fold_diffs.size),
    }


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_baselines(cfg: EnvConfig) -> Dict[str, SelectorPolicy]:
    return {
        "mes": endowment_selector(uniform_policy, cfg),
        "res-1.0": endowment_selector(res_policy(1.0), cfg),
        "greedy-count": score_selector(greedy_equivalent_weights()),
        "greedy-cost": score_selector(cost_effective_weights()),
        "llmrule-card": llmrule_card_selector(),
        "llmrule-cost": llmrule_cost_selector(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tag-template", default="district_endow_f{fold}_g25")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--expected-bound", type=float, default=10.0)
    parser.add_argument(
        "--allow-missing-default-bound",
        action="store_true",
        help=(
            "accept legacy run JSONs that predate bound recording only when "
            "the expected bound is the historical default 10.0; the manifest "
            "will disclose this provenance exception"
        ),
    )
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument(
        "--output-tag",
        default="",
        help="optional safe suffix for sensitivity outputs (for example b40)",
    )
    args = parser.parse_args()

    per_series_csv, summary_csv, manifest_json = output_paths(args.output_tag)

    cfg = EnvConfig()
    arm = build_arm("endowment")
    index = build_series_index()
    memberships: Dict[str, Sequence[str]] = {}
    run_records = []
    rows = []

    for fold in range(args.folds):
        split_name = f"district_out_f{fold}of{args.folds}"
        split = load_split(split_name)
        memberships[split_name] = split.test_keys()
        run_file = RUN_DIR / f"run_{args.tag_template.format(fold=fold)}.json"
        if not run_file.exists():
            raise FileNotFoundError(f"missing completed fold run: {run_file}")
        payload = json.loads(run_file.read_text())
        run_cfg = payload.get("config", {})
        if payload.get("smoke"):
            raise ValueError(f"refusing smoke artifact: {run_file}")
        validate_run_config(
            run_cfg,
            expected_split=split_name,
            expected_seed=args.seed,
            expected_generations=args.generations,
            expected_bound=args.expected_bound,
            allow_missing_default_bound=args.allow_missing_default_bound,
        )
        validate_run_counts(
            payload,
            expected_train=len(split.train),
            expected_test=len(split.test),
        )

        weights = np.asarray(payload["best_weights"], dtype=float)
        if weights.shape != (arm.n_features,):
            raise ValueError(f"wrong weight shape in {run_file}: {weights.shape}")
        validate_feature_names(payload.get("feature_names", ()), arm.feature_names)
        policies = {"learned": arm.selector(weights, cfg), **_fixed_baselines(cfg)}
        data = _load_series_data(split, index)

        for district in data:
            if not district.test_years:
                continue
            for policy_name, selector in policies.items():
                episode = rollout_selector(
                    district.ref,
                    selector,
                    score_years=district.test_years,
                    cfg=cfg,
                    instances=district.all_years,
                )
                rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "series": district.ref.key,
                        "policy": policy_name,
                        "n_years": len(episode.scored_years),
                        "worst_csd": episode.worst_csd,
                        "mean_csd": episode.mean_csd,
                        "welfare": episode.welfare,
                        "cost_welfare": episode.cost_welfare,
                        "exclusion": episode.exclusion,
                    }
                )
        validate_replay_summary(
            [
                row
                for row in rows
                if row["fold"] == fold and row["policy"] == "learned"
            ],
            payload.get("learned", {}).get("test", {}),
        )
        run_records.append(
            {
                "fold": fold,
                "split": split_name,
                "file": str(run_file.relative_to(ROOT)),
                "sha256": _sha256(run_file),
                "test_keys": split.test_keys(),
                "best_weights": weights.tolist(),
                "best_train_loss": payload["best_train_loss"],
                "boundary_features": boundary_feature_names(
                    weights, arm.feature_names, bound=args.expected_bound
                ),
            }
        )

    validate_fold_coverage(memberships, expected_keys=sorted(index))
    expected_rows = len(index) * (1 + len(_fixed_baselines(cfg)))
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} policy-series rows, found {len(rows)}")

    per_series_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_series_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_policy = {
        name: {row["series"]: row for row in rows if row["policy"] == name}
        for name in {row["policy"] for row in rows}
    }
    keys = sorted(index)
    learned_rows = by_policy["learned"]
    fold_by_series = {
        row["series"]: int(row["fold"])
        for row in rows
        if row["policy"] == "learned"
    }
    summaries = []
    for baseline_name in sorted(name for name in by_policy if name != "learned"):
        baseline_rows = by_policy[baseline_name]
        learned_values = [learned_rows[key]["worst_csd"] for key in keys]
        baseline_values = [baseline_rows[key]["worst_csd"] for key in keys]
        summary = paired_summary(
            learned_values,
            baseline_values,
            seed=args.seed,
            n_boot=args.n_boot,
        )
        block_summary = fold_block_summary(
            learned_values,
            baseline_values,
            [fold_by_series[key] for key in keys],
        )
        learned_welfare = sum(float(learned_rows[key]["welfare"]) for key in keys)
        baseline_welfare = sum(float(baseline_rows[key]["welfare"]) for key in keys)
        summaries.append(
            {
                "baseline": baseline_name,
                **summary,
                **block_summary,
                "learned_welfare": learned_welfare,
                "baseline_welfare": baseline_welfare,
                "welfare_ratio": learned_welfare / baseline_welfare,
                "learned_exclusion": float(
                    np.mean([learned_rows[key]["exclusion"] for key in keys])
                ),
                "baseline_exclusion": float(
                    np.mean([baseline_rows[key]["exclusion"] for key in keys])
                ),
            }
        )

    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "analysis": "five-fold district-held-out endowment evaluation",
        "n_folds": args.folds,
        "n_series": len(index),
        "n_policy_series_rows": len(rows),
        "bootstrap_replicates": args.n_boot,
        "bootstrap_seed": args.seed,
        "expected_training_seed": args.seed,
        "expected_training_generations": args.generations,
        "expected_training_bound": args.expected_bound,
        "output_tag": args.output_tag,
        "legacy_missing_default_bound_allowed": args.allow_missing_default_bound,
        "bound_provenance": (
            "Legacy fold runs were launched without --bound while iclr_train.py "
            "used its documented default 10.0; their pre-fix TrainConfig JSON "
            "omitted that field. Numerical payloads are unmodified."
            if args.allow_missing_default_bound
            else "Recorded directly in each fold run config."
        ),
        "hypothesis_tests": {
            "pooled_series": (
                "descriptive exact two-sided paired sign-flip over series; "
                "series within one fold share a fitted map"
            ),
            "paired_series_interval": (
                "descriptive 2.5--97.5 percentile bootstrap over 19 paired "
                "series effects"
            ),
            "fold_block": (
                "coarse exact two-sided sign-flip sensitivity check over five "
                "fold-mean effects; fold fits have overlapping training sets "
                "and are not independent replications"
            ),
        },
        "coverage": memberships,
        "runs": run_records,
        "outputs": {
            "per_series": str(per_series_csv.relative_to(ROOT)),
            "summary": str(summary_csv.relative_to(ROOT)),
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    logger.info(
        "%-14s %9s %9s %9s %20s %8s %8s %8s",
        "baseline", "learned", "base", "diff", "95% CI", "p(series)",
        "p(fold)", "fold wins",
    )
    for row in summaries:
        logger.info(
            "%-14s %9.4f %9.4f %+9.4f [%+.4f,%+.4f] %8.4f %8.4f %4d/%d",
            row["baseline"], row["learned_mean"], row["baseline_mean"],
            row["diff"], row["ci_lo"], row["ci_hi"], row["p_two_sided"],
            row["fold_p_two_sided"], row["fold_wins"], row["n_folds"],
        )
    logger.info("wrote %s, %s, and %s", per_series_csv, summary_csv, manifest_json)


if __name__ == "__main__":
    main()
