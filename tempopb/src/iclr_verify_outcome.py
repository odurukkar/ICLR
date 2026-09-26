"""Phase-3 gate: the outcome-space family must contain the deployed greedy rule.

Two independent identities must hold election by election:

    score_selector(e_1) == rules.greedy_by_votes
    score_selector(e_2) == rules.greedy_by_approvals_per_cost

If either fails, the learned scorer is being compared against a greedy baseline
it does not actually generalize, and the frontier plot would mix different
rules. The second identity also verifies the utilitarian baseline that the audit
line of work is otherwise open to being criticized for omitting.
"""

from __future__ import annotations

import logging
from typing import List

from iclr_corpus import build_series_index, load_series
from iclr_env import EnvConfig, RolloutState, aggregate, rollout_selector
from iclr_outcome import (
    cost_effective_weights,
    greedy_equivalent_weights,
    score_selector,
)
from rules import greedy_by_approvals_per_cost, greedy_by_votes

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", force=True
)
logger = logging.getLogger(__name__)


def main() -> None:
    index = build_series_index()
    cfg = EnvConfig()
    count_selector = score_selector(greedy_equivalent_weights())
    cost_selector = score_selector(cost_effective_weights())
    failures: List[str] = []
    n_elections = 0

    for i, (key, ref) in enumerate(sorted(index.items()), start=1):
        instances = load_series(ref)
        mismatched_count = 0
        mismatched_cost = 0
        for year, inst in sorted(instances.items()):
            want = greedy_by_votes(inst)
            got = count_selector(inst, RolloutState())
            n_elections += 1
            if want != got:
                mismatched_count += 1
                failures.append(
                    f"{key} {year} greedy-count: |want|={len(want)} |got|={len(got)} "
                    f"sym_diff={len(want ^ got)}"
                )
            want_cost = greedy_by_approvals_per_cost(inst)
            got_cost = cost_selector(inst, RolloutState())
            if want_cost != got_cost:
                mismatched_cost += 1
                failures.append(
                    f"{key} {year} greedy-cost: |want|={len(want_cost)} "
                    f"|got|={len(got_cost)} sym_diff={len(want_cost ^ got_cost)}"
                )
        logger.info(
            "[%2d/%d] %-18s %d elections, mismatches count=%d cost=%d",
            i, len(index), key.split("/")[-1], len(instances),
            mismatched_count, mismatched_cost,
        )

    if failures:
        logger.error("OUTCOME FAMILY GATE FAILED (%d elections):", len(failures))
        for line in failures[:20]:
            logger.error("  %s", line)
        raise SystemExit(1)
    logger.info(
        "OUTCOME FAMILY GATE PASSED: e_1 == greedy-by-count and "
        "e_2 == greedy-by-approvals/cost on all %d elections across %d series",
        n_elections, len(index),
    )

    # The free extra baseline: greedy by approvals per unit cost.
    for name, weights in (
        ("greedy (approval count)", greedy_equivalent_weights()),
        ("greedy (approvals/cost)", cost_effective_weights()),
    ):
        episodes = [
            rollout_selector(ref, score_selector(weights), cfg=cfg)
            for ref in index.values()
        ]
        stats = aggregate(episodes)
        logger.info(
            "%-26s worst_csd=%.4f mean_csd=%.4f welfare=%.0f exclusion=%.4f",
            name, stats["worst_csd"], stats["mean_csd"],
            stats["welfare"], stats["exclusion"],
        )


if __name__ == "__main__":
    main()
