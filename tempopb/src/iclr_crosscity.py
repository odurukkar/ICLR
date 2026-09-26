"""Cross-city temporal evaluation of the frozen Warsaw-fitted policies.

The submitted draft stated that no non-Warsaw series supports the main
longitudinal protocol. That was true of the local 697-file download and false
of the public corpus: the Pabulib repository holds 2015 instance files, and the
paper's own filter chain accepts 81 non-Warsaw series there. Gdynia runs six
consecutive editions (2020--2025) and Lodz four (2022--2025), both with
approval ballots, recorded winners and age/sex fields.

This module evaluates the *already fitted* Warsaw policies on those cities with
no refitting and no policy selection. Each series is replayed in order from its
first edition so every policy carries its own endogenous deficit history, and
only editions from 2023 on are scored -- the same warm-up/score protocol the
main experiment uses. The policies were fit on Warsaw editions through 2022, so
they have seen none of these ballots, projects, or cities, and every scored
edition postdates fitting.

Two corpus rules avoid counting the same voters twice, matching how the primary
corpus excludes Warsaw's citywide and Wawer-subunit aggregates:

* Gdynia runs a "small" and a "large" project ballot over one neighbourhood
  electorate in the same edition. We keep one track per neighbourhood, the one
  with the larger summed budget, and never treat the two as independent series.
* Citywide aggregates (Lodz, Wroclaw) are dropped because their ballots are
  already counted inside the district series.

Writes results/iclr_crosscity_per_series.csv, results/iclr_crosscity_summary.csv
and results/iclr_crosscity_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from iclr_corpus import CorpusConfig, SeriesRef, build_series_index, load_series
from iclr_env import (
    EnvConfig,
    EpisodeResult,
    SelectorPolicy,
    aggregate,
    endowment_selector,
    res_policy,
    rollout_reference,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import cost_effective_weights, greedy_equivalent_weights, score_selector
from iclr_stats import exact_paired_sign_flip_pvalue, monte_carlo_paired_sign_flip_pvalue
from iclr_transfer import _learned_policies

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA_DIR = ROOT / "data" / "pb_crosscity"
OUT_SERIES = RESULTS / "iclr_crosscity_per_series.csv"
OUT_SUMMARY = RESULTS / "iclr_crosscity_summary.csv"
OUT_MANIFEST = RESULTS / "iclr_crosscity_manifest.json"
OUT_FROZEN = RESULTS / "iclr_crosscity_instances.txt"
OUT_ACTUATION = RESULTS / "iclr_crosscity_actuation.csv"

# Granularity thresholds for the actuation analysis. These are instance
# properties, fixed here before any contrast is computed, and the whole grid is
# reported so no threshold can be selected after seeing an effect.
GRANULARITY_THRESHOLDS = (0, 5, 10, 15, 20, 25, 30)
ACTUATION_TOLERANCE = 1e-9

TRAINING_CITY = "Poland/Warszawa"
FIRST_SCORED_YEAR = 2023
LAST_TRAINING_YEAR = 2022
CITYWIDE_KEYS = ("CITYWIDE",)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260811
SIGNFLIP_DRAWS = 200_000
SIGNFLIP_SEED = 20260812
EXACT_SIGNFLIP_LIMIT = 24


def _budget_weight(ref: SeriesRef) -> float:
    """Total budget across a series' editions, used only to pick one track."""
    return float(sum(inst.budget for inst in load_series(ref).values()))


def _neighbourhood(key: str) -> str:
    """Series key with any Gdynia track suffix removed."""
    head, _, tail = key.rpartition("/")
    return f"{head}/{tail.rsplit(' | ', 1)[0]}" if " | " in tail else key


def select_crosscity_series(index: Mapping[str, SeriesRef]) -> Tuple[SeriesRef, ...]:
    """Eligible non-training-city series that support the temporal protocol."""
    candidates = [
        ref
        for _, ref in sorted(index.items())
        if ref.city != TRAINING_CITY
        and not ref.key.endswith(CITYWIDE_KEYS)
        and any(year <= LAST_TRAINING_YEAR for year in ref.years)
        and any(year >= FIRST_SCORED_YEAR for year in ref.years)
    ]
    grouped: Dict[str, List[SeriesRef]] = defaultdict(list)
    for ref in candidates:
        grouped[_neighbourhood(ref.key)].append(ref)

    selected: List[SeriesRef] = []
    for _, refs in sorted(grouped.items()):
        if len(refs) == 1:
            selected.append(refs[0])
            continue
        selected.append(max(refs, key=lambda r: (_budget_weight(r), r.key)))
    return tuple(sorted(selected, key=lambda r: r.key))


def stage_corpus(source_dir: Path, destination: Path = DATA_DIR) -> Tuple[int, int]:
    """Copy every file backing a selected series into the evaluation corpus.

    The primary corpus directory is left untouched: adding files there would
    silently change the frozen 19-series index the submitted results depend on.
    """
    index = build_series_index(CorpusConfig(data_dir=source_dir))
    refs = select_crosscity_series(index)
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ref in refs:
        for path in ref.paths:
            target = destination / path.name
            if not target.exists():
                shutil.copy2(path, target)
            copied += 1
    logger.info("staged %d series / %d instance files into %s", len(refs), copied, destination)
    return len(refs), copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


EXPECTED_LEARNED = (
    "learned-endowment",
    "learned-outcome-f1.00",
    "learned-outcome-f0.85",
)


def policies(cfg: EnvConfig) -> List[Tuple[str, SelectorPolicy]]:
    """Frozen Warsaw-fitted policies plus every hand-designed reference rule."""
    learned = _learned_policies(cfg)
    names = tuple(name for name, _ in learned)
    if names != EXPECTED_LEARNED:
        raise RuntimeError(f"Expected frozen policies {EXPECTED_LEARNED}, found {names}")
    return [
        ("greedy-count", score_selector(greedy_equivalent_weights())),
        ("greedy-cost", score_selector(cost_effective_weights())),
        ("mes", endowment_selector(uniform_policy, cfg)),
        ("res-1.0", endowment_selector(res_policy(1.0), cfg)),
        *learned,
    ]


def evaluate(
    refs: Sequence[SeriesRef], cfg: Optional[EnvConfig] = None
) -> Dict[str, List[EpisodeResult]]:
    """Replay every policy on every series, scoring only held-out editions."""
    cfg = cfg or EnvConfig()
    named = policies(cfg)
    episodes: Dict[str, List[EpisodeResult]] = {name: [] for name, _ in named}
    episodes["recorded"] = []
    for position, ref in enumerate(refs, start=1):
        instances = load_series(ref)
        scored = tuple(y for y in ref.years if y >= FIRST_SCORED_YEAR)
        for name, selector in named:
            episodes[name].append(
                rollout_selector(ref, selector, score_years=scored, cfg=cfg, instances=instances)
            )
        episodes["recorded"].append(
            rollout_reference(
                ref, "historical", score_years=scored, cfg=cfg, instances=instances
            )
        )
        logger.info(
            "[%d/%d] %-45s scored %s", position, len(refs), ref.key, list(scored)
        )
    return episodes


def _paired_bootstrap(differences: np.ndarray) -> Tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, differences.size, size=(BOOTSTRAP_DRAWS, differences.size))
    means = differences[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _pvalue(differences: np.ndarray) -> Tuple[float, str]:
    if differences.size <= EXACT_SIGNFLIP_LIMIT:
        return exact_paired_sign_flip_pvalue(differences), "exact"
    return (
        monte_carlo_paired_sign_flip_pvalue(
            differences, draws=SIGNFLIP_DRAWS, seed=SIGNFLIP_SEED
        ),
        f"monte-carlo({SIGNFLIP_DRAWS})",
    )


def contrast(
    learned: Sequence[EpisodeResult], baseline: Sequence[EpisodeResult]
) -> Dict[str, object]:
    """Series-level paired contrast in held-out worst-group CSD."""
    pairs = [
        (a.worst_csd, b.worst_csd)
        for a, b in zip(learned, baseline)
        if a.worst_csd is not None and b.worst_csd is not None
    ]
    diff = np.array([a - b for a, b in pairs], dtype=float)
    low, high = _paired_bootstrap(diff)
    pvalue, kind = _pvalue(diff)
    return {
        "n_series": diff.size,
        "learned_csd": float(np.mean([a for a, _ in pairs])),
        "baseline_csd": float(np.mean([b for _, b in pairs])),
        "difference": float(diff.mean()),
        "ci_low": low,
        "ci_high": high,
        "p_value": pvalue,
        "p_kind": kind,
        "wins": int(np.count_nonzero(diff < 0)),
    }


def granularity(refs: Sequence[SeriesRef]) -> Dict[str, float]:
    """Mean projects per scored election, the instance property we stratify on."""
    out: Dict[str, float] = {}
    for ref in refs:
        instances = load_series(ref)
        scored = [y for y in ref.years if y >= FIRST_SCORED_YEAR and y in instances]
        if scored:
            out[ref.key] = sum(len(instances[y].projects) for y in scored) / len(scored)
    return out


def actuation_rows(
    refs: Sequence[SeriesRef], episodes: Mapping[str, List[EpisodeResult]]
) -> List[Dict[str, object]]:
    """Contrast against Equal Shares within each granularity stratum.

    A series is *inert* when the learned surface returns exactly the Equal
    Shares outcome, so its paired difference is zero to machine precision.
    Reporting the whole threshold grid keeps this a description of where the
    surface can act, not a search for a stratum where it wins.
    """
    sizes = granularity(refs)
    paired = [
        (ref.key, learned.worst_csd - base.worst_csd)
        for ref, learned, base in zip(refs, episodes["learned-endowment"], episodes["mes"])
        if learned.worst_csd is not None and base.worst_csd is not None
    ]
    rows: List[Dict[str, object]] = []
    for threshold in GRANULARITY_THRESHOLDS:
        subset = [(k, d) for k, d in paired if sizes.get(k, 0.0) >= threshold]
        if len(subset) < 4:
            continue
        diff = np.array([d for _, d in subset], dtype=float)
        low, high = _paired_bootstrap(diff)
        pvalue, kind = _pvalue(diff)
        rows.append({
            "min_projects_per_election": threshold,
            "n_series": diff.size,
            "n_inert": int(np.count_nonzero(np.abs(diff) < ACTUATION_TOLERANCE)),
            "difference": float(diff.mean()),
            "ci_low": low,
            "ci_high": high,
            "p_value": pvalue,
            "p_kind": kind,
            "wins": int(np.count_nonzero(diff < -ACTUATION_TOLERANCE)),
            "losses": int(np.count_nonzero(diff > ACTUATION_TOLERANCE)),
        })
    return rows


def actuation_profile(
    refs: Sequence[SeriesRef], episodes: Mapping[str, List[EpisodeResult]]
) -> Dict[str, float]:
    """Median instance granularity for series the surface does and does not move."""
    sizes = granularity(refs)
    inert: List[float] = []
    active: List[float] = []
    for ref, learned, base in zip(refs, episodes["learned-endowment"], episodes["mes"]):
        if learned.worst_csd is None or base.worst_csd is None or ref.key not in sizes:
            continue
        bucket = inert if abs(learned.worst_csd - base.worst_csd) < ACTUATION_TOLERANCE else active
        bucket.append(sizes[ref.key])
    return {
        "n_inert": len(inert),
        "n_active": len(active),
        "median_projects_inert": float(np.median(inert)) if inert else float("nan"),
        "median_projects_active": float(np.median(active)) if active else float("nan"),
    }


def _scopes(refs: Sequence[SeriesRef]) -> List[Tuple[str, List[int]]]:
    by_city: Dict[str, List[int]] = defaultdict(list)
    for position, ref in enumerate(refs):
        by_city[ref.city].append(position)
    scopes = sorted(by_city.items())
    scopes.append(("ALL", list(range(len(refs)))))
    return scopes


def write_outputs(
    refs: Sequence[SeriesRef], episodes: Mapping[str, List[EpisodeResult]]
) -> None:
    OUT_SERIES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SERIES.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["series", "city", "scored_years", "policy", "worst_csd",
             "worst_cohort", "mean_csd", "welfare", "exclusion"]
        )
        for policy, results in sorted(episodes.items()):
            for ref, episode in zip(refs, results):
                writer.writerow([
                    ref.key, ref.city, " ".join(map(str, episode.scored_years)), policy,
                    episode.worst_csd, episode.worst_cohort, episode.mean_csd,
                    episode.welfare, episode.exclusion,
                ])

    rows: List[Dict[str, object]] = []
    for scope, positions in _scopes(refs):
        for policy, results in sorted(episodes.items()):
            subset = [results[i] for i in positions]
            summary = aggregate(subset)
            rows.append({"scope": scope, "policy": policy, **summary})
    with OUT_SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with OUT_FROZEN.open("w") as handle:
        for ref in refs:
            for year, path in zip(ref.years, ref.paths):
                handle.write(f"{path.name}\t{ref.key}\t{year}\n")

    rows = actuation_rows(refs, episodes)
    with OUT_ACTUATION.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    refs: Sequence[SeriesRef], episodes: Mapping[str, List[EpisodeResult]]
) -> Dict[str, object]:
    """Record the corpus, protocol and every headline contrast in one file."""
    contrasts: Dict[str, Dict[str, object]] = {}
    learned = "learned-endowment"
    for scope, positions in _scopes(refs):
        for baseline in ("mes", "res-1.0", "greedy-count", "greedy-cost"):
            if baseline not in episodes or learned not in episodes:
                continue
            key = f"{scope}|{learned}|{baseline}"
            contrasts[key] = contrast(
                [episodes[learned][i] for i in positions],
                [episodes[baseline][i] for i in positions],
            )
    manifest = {
        "protocol": {
            "training_city": TRAINING_CITY,
            "refitting": False,
            "warm_up_years": f"<= {LAST_TRAINING_YEAR}",
            "scored_years": f">= {FIRST_SCORED_YEAR}",
            "series_unit": "one (city, district/neighbourhood) series",
        },
        "corpus": {
            "n_series": len(refs),
            "n_scored_elections": sum(
                sum(1 for y in ref.years if y >= FIRST_SCORED_YEAR) for ref in refs
            ),
            "n_elections": sum(len(ref.years) for ref in refs),
            "cities": sorted({ref.city for ref in refs}),
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for ref in refs
                for path in ref.paths
            },
        },
        "contrasts": contrasts,
        "actuation": actuation_profile(refs, episodes),
        "actuation_by_granularity": actuation_rows(refs, episodes),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-from",
        type=Path,
        default=None,
        help="copy the eligible non-Warsaw corpus out of this Pabulib checkout first",
    )
    args = parser.parse_args()

    if args.stage_from is not None:
        stage_corpus(args.stage_from)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"{DATA_DIR} is missing; rerun with --stage-from <pabulib checkout>"
        )
    index = build_series_index(CorpusConfig(data_dir=DATA_DIR))
    refs = select_crosscity_series(index)
    logger.info(
        "cross-city corpus: %d series, %d elections, cities %s",
        len(refs),
        sum(len(r.years) for r in refs),
        sorted({r.city for r in refs}),
    )
    episodes = evaluate(refs)
    write_outputs(refs, episodes)
    manifest = write_manifest(refs, episodes)
    for key, row in sorted(manifest["contrasts"].items()):
        if key.endswith("|mes"):
            logger.info(
                "%-40s learned %.4f vs MES %.4f  diff %+.4f  CI [%.4f, %.4f]  p=%.4g (%s)  wins %d/%d",
                key, row["learned_csd"], row["baseline_csd"], row["difference"],
                row["ci_low"], row["ci_high"], row["p_value"], row["p_kind"],
                row["wins"], row["n_series"],
            )


if __name__ == "__main__":
    main()
