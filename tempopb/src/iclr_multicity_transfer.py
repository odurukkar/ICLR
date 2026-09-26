"""No-refit stress test on every eligible shorter non-Warsaw PB series.

The headline temporal protocol needs at least three pre-2023 editions and one
later edition. No city outside Warsaw meets that requirement in the local
Pabulib snapshot. This diagnostic therefore applies the already-fitted Warsaw
policies, without retraining or policy selection, to every non-Warsaw series
that passes the same approval-ballot, recorded-winner, demographic-coverage,
and voter-count filters when the minimum run length is relaxed to one.

The resulting 17 series (21 elections) are a descriptive breadth stress test,
not external validation: most series have one edition and the 17 series are
clustered in only three cities. The script reports no inferential p-values.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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
from iclr_transfer import _learned_policies

TRAINING_CITY = "Poland/Warszawa"
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "iclr_multicity_transfer.csv"
OUT_SELECTION = RESULTS / "iclr_multicity_selection.json"
OUT_FROZEN = RESULTS / "iclr_multicity_instances.txt"
EXPECTED_SERIES = 17
EXPECTED_ELECTIONS = 21
EXPECTED_CITIES = ("Poland/Gdynia", "Poland/Lublin", "Poland/Łódź")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def select_stress_series(
    index: Mapping[str, SeriesRef], training_city: str = TRAINING_CITY
) -> Tuple[SeriesRef, ...]:
    """Return every eligible series outside the city used to fit the policies."""
    return tuple(
        ref for _, ref in sorted(index.items()) if ref.city != training_city
    )


def aggregate_scope_rows(
    row_type: str,
    scope: str,
    episodes_by_policy: Mapping[str, Sequence[EpisodeResult]],
) -> List[Dict[str, object]]:
    """Summarize a city or the full stress set with series as the unit."""
    if "mes" not in episodes_by_policy:
        raise ValueError("MES episodes are required for reference deltas")
    mes = aggregate(episodes_by_policy["mes"])
    mes_welfare = mes["welfare"] or 1.0
    rows: List[Dict[str, object]] = []
    for policy, episodes in episodes_by_policy.items():
        stats = aggregate(episodes)
        rows.append(
            {
                "row_type": row_type,
                "scope": scope,
                "policy": policy,
                "n_series": int(stats["n_series"]),
                "n_elections": sum(len(ep.scored_years) for ep in episodes),
                "worst_csd": round(stats["worst_csd"], 12),
                "welfare": round(stats["welfare"], 6),
                "cost_welfare": round(stats["cost_welfare"], 6),
                "exclusion": round(stats["exclusion"], 12),
                "delta_vs_mes": round(stats["worst_csd"] - mes["worst_csd"], 12),
                "welfare_ratio_vs_mes": round(stats["welfare"] / mes_welfare, 12),
                "exclusion_delta_vs_mes": round(
                    stats["exclusion"] - mes["exclusion"], 12
                ),
            }
        )
    return rows


def frozen_instance_rows(
    refs: Sequence[SeriesRef],
) -> List[Tuple[str, str, int]]:
    """Return stable, deduplicated manifest rows for the stress-test files."""
    rows: Dict[str, Tuple[str, str, int]] = {}
    for ref in refs:
        for year, path in zip(ref.years, ref.paths):
            rows.setdefault(path.name, (path.name, ref.key, year))
    return [rows[name] for name in sorted(rows)]


def main() -> None:
    cfg = EnvConfig()
    index = build_series_index(CorpusConfig(min_run=1, exclude=frozenset()))
    refs = select_stress_series(index)
    cities = tuple(sorted({ref.city for ref in refs}))
    n_elections = sum(len(ref.years) for ref in refs)
    if (len(refs), n_elections, cities) != (
        EXPECTED_SERIES,
        EXPECTED_ELECTIONS,
        EXPECTED_CITIES,
    ):
        raise RuntimeError(
            "Eligible multi-city corpus drifted: "
            f"got {len(refs)} series, {n_elections} elections, {cities}"
        )

    learned = _learned_policies(cfg)
    learned_names = tuple(name for name, _ in learned)
    expected_learned = (
        "learned-endowment",
        "learned-outcome-f1.00",
        "learned-outcome-f0.85",
    )
    if learned_names != expected_learned:
        raise RuntimeError(
            f"Expected frozen policies {expected_learned}, found {learned_names}"
        )

    selectors: List[Tuple[str, SelectorPolicy]] = [
        ("greedy-count", score_selector(greedy_equivalent_weights())),
        ("greedy-cost", score_selector(cost_effective_weights())),
        ("mes", endowment_selector(uniform_policy, cfg)),
        ("res-1.0", endowment_selector(res_policy(1.0), cfg)),
        *learned,
    ]
    policy_order = ("historical", *(name for name, _ in selectors))
    all_episodes: Dict[str, List[EpisodeResult]] = {
        policy: [] for policy in policy_order
    }
    city_episodes: Dict[str, Dict[str, List[EpisodeResult]]] = defaultdict(
        lambda: {policy: [] for policy in policy_order}
    )
    rows: List[Dict[str, object]] = []

    for position, ref in enumerate(refs, start=1):
        instances = load_series(ref)
        per_series: Dict[str, List[EpisodeResult]] = {
            policy: [] for policy in policy_order
        }
        historical = rollout_reference(
            ref, "historical", cfg=cfg, instances=instances
        )
        per_series["historical"].append(historical)
        all_episodes["historical"].append(historical)
        city_episodes[ref.city]["historical"].append(historical)
        for name, selector in selectors:
            episode = rollout_selector(ref, selector, cfg=cfg, instances=instances)
            per_series[name].append(episode)
            all_episodes[name].append(episode)
            city_episodes[ref.city][name].append(episode)
        rows.extend(aggregate_scope_rows("series", ref.key, per_series))
        logger.info(
            "[%d/%d] %s: %d edition(s)",
            position,
            len(refs),
            ref.key,
            len(ref.years),
        )

    for city in cities:
        rows.extend(aggregate_scope_rows("city", city, city_episodes[city]))
    rows.extend(aggregate_scope_rows("all", "all-non-Warsaw", all_episodes))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selection = {
        "protocol": "all eligible non-Warsaw series; min_run relaxed to 1; no refit",
        "training_city": TRAINING_CITY,
        "n_series": len(refs),
        "n_elections": n_elections,
        "cities": list(cities),
        "inference": "descriptive only; no city-level or population inference",
        "series": [
            {
                "key": ref.key,
                "city": ref.city,
                "years": list(ref.years),
                "files": [path.name for path in ref.paths],
            }
            for ref in refs
        ],
    }
    OUT_SELECTION.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    frozen_rows = frozen_instance_rows(refs)
    OUT_FROZEN.write_text(
        "# filename\tseries\tparsed_year\n"
        + "\n".join("\t".join(map(str, row)) for row in frozen_rows)
        + "\n"
    )

    logger.info(
        "wrote %s (%d rows), %s, and %s (%d files)",
        OUT_CSV,
        len(rows),
        OUT_SELECTION,
        OUT_FROZEN,
        len(frozen_rows),
    )
    for row in rows:
        if row["row_type"] in {"city", "all"}:
            logger.info(
                "%-16s %-24s CSD=%0.6f delta=%+0.6f welfare/MES=%0.4f exclusion_delta=%+0.6f",
                row["scope"],
                row["policy"],
                row["worst_csd"],
                row["delta_vs_mes"],
                row["welfare_ratio_vs_mes"],
                row["exclusion_delta_vs_mes"],
            )


if __name__ == "__main__":
    main()
