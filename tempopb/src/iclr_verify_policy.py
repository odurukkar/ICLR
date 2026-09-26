"""Phase-2 gate: the learned policy family must contain the hand-designed rules.

`linear_policy` is only an honest generalization of Rollover Equal Shares if it
reproduces the baselines exactly at the corresponding weight vectors:

    linear_policy(zeros)                  == uniform endowments == MES
    linear_policy(res_equivalent(lam))    == RES at intensity lam

If either identity fails, every later claim of the form "search found something
better than the hand-designed rule" is comparing two different mechanisms
rather than two points in one space. Phase 2 training must not start until this
passes.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from iclr_corpus import build_series_index, load_series
from iclr_env import res_policy, rollout, uniform_policy
from iclr_policy import N_FEATURES, linear_policy, res_equivalent_weights

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOL = 1e-9
LAMBDAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    index = build_series_index()
    failures: List[str] = []

    for i, (key, ref) in enumerate(sorted(index.items()), start=1):
        instances = load_series(ref)
        short = key.split("/")[-1]

        mes = rollout(ref, uniform_policy, instances=instances).worst_csd
        lin0 = rollout(
            ref, linear_policy(np.zeros(N_FEATURES)), instances=instances
        ).worst_csd
        if mes is None or lin0 is None or abs(mes - lin0) > TOL:
            failures.append(f"{key}: MES={mes} linear(0)={lin0}")

        deltas: List[str] = []
        for lam in LAMBDAS:
            res = rollout(ref, res_policy(lam), instances=instances).worst_csd
            lin = rollout(
                ref, linear_policy(res_equivalent_weights(lam)), instances=instances
            ).worst_csd
            if res is None or lin is None or abs(res - lin) > TOL:
                failures.append(f"{key} lam={lam}: RES={res} linear={lin}")
            deltas.append(f"{lam}:{abs((res or 0) - (lin or 0)):.1e}")

        logger.info(
            "[%2d/%d] %-18s MES|delta|=%.1e  RES deltas %s",
            i, len(index), short, abs((mes or 0) - (lin0 or 0)), " ".join(deltas),
        )

    if failures:
        logger.error("POLICY FAMILY GATE FAILED (%d):", len(failures))
        for line in failures:
            logger.error("  %s", line)
        raise SystemExit(1)
    logger.info(
        "POLICY FAMILY GATE PASSED: linear_policy contains MES and RES(%s) exactly "
        "on all %d series",
        ",".join(str(x) for x in LAMBDAS), len(index),
    )


if __name__ == "__main__":
    main()
