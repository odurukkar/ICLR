"""Every hand-designed baseline, scored through one ledger.

Existing results compare the deployed rule against greedy-by-approval-count and
Equal Shares. The natural utilitarian competitor -- greedy by approvals per unit
cost -- has not been scored anywhere in this project, and it is the first
baseline a referee asks for when a paper claims a proportional rule improves on
"greedy". This script puts all of them on the same footing.

Writes results/iclr_baselines.csv (per series) and logs the aggregate table.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from iclr_corpus import build_series_index, load_series
from iclr_env import (
    EnvConfig,
    SelectorPolicy,
    aggregate,
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "iclr_baselines.csv"


def _rules(cfg: EnvConfig) -> List[Tuple[str, SelectorPolicy]]:
    """Name -> selector, in the order the table should read."""
    rules: List[Tuple[str, SelectorPolicy]] = [
        ("greedy-count", score_selector(greedy_equivalent_weights())),
        ("greedy-cost", score_selector(cost_effective_weights())),
        ("llmrule-cost", llmrule_cost_selector()),
        ("llmrule-card", llmrule_card_selector()),
        ("mes", endowment_selector(uniform_policy, cfg)),
    ]
    for lam in (0.25, 0.5, 0.75, 1.0):
        rules.append((f"res-{lam}", endowment_selector(res_policy(lam), cfg)))
    return rules


def main() -> None:
    cfg = EnvConfig()
    index = build_series_index()
    rules = _rules(cfg)

    rows: List[dict] = []
    episodes_by_rule: Dict[str, list] = {name: [] for name, _ in rules}

    for i, (key, ref) in enumerate(sorted(index.items()), start=1):
        instances = load_series(ref)
        row = {"series": key}
        for name, selector in rules:
            episode = rollout_selector(ref, selector, cfg=cfg, instances=instances)
            episodes_by_rule[name].append(episode)
            row[f"{name}_worst"] = round(episode.worst_csd, 6) if episode.worst_csd else None
            row[f"{name}_welfare"] = round(episode.welfare, 1)
        rows.append(row)
        logger.info("[%2d/%d] %s", i, len(index), key.split("/")[-1])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", OUT_CSV)

    ref_welfare = aggregate(episodes_by_rule["greedy-count"])["welfare"] or 1.0
    logger.info("=" * 74)
    logger.info(
        "%-14s %10s %10s %10s %10s", "rule", "worst_csd", "mean_csd", "welf/gc", "exclusion"
    )
    logger.info("-" * 74)
    for name, _ in rules:
        stats = aggregate(episodes_by_rule[name])
        logger.info(
            "%-14s %10.4f %10.4f %10.3f %10.4f",
            name, stats["worst_csd"], stats["mean_csd"],
            stats["welfare"] / ref_welfare, stats["exclusion"],
        )
    logger.info("=" * 74)
    logger.info("welf/gc = count welfare relative to greedy by approval count")


if __name__ == "__main__":
    main()
