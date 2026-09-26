"""One post-lock contrast from sealed saved outcomes; never runs an allocation.

Run once with --write to save the correction bundle, or without it to inspect
the computed summary. Bootstrap conventions match the corrected-v2 evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from iclr_stats import exact_paired_sign_flip_pvalue

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "results/iclr_primary_warsaw_v2/evaluation"
OUTPUT = ROOT / "analysis-output/contextual-age-lookup-20260920"
CONTEXTUAL = "temporal_endowment/temporal_2022/seed-42"
LOOKUP = "static_age_lookup/temporal_2022/seed-42"
SOURCE_SHA256 = "a96c9b486f12702553165601e27d7fa6f04a10ae38510aedb7c9fc1059fd54f5"


def pair_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, dict[str, dict[str, str]]] = {CONTEXTUAL: {}, LOOKUP: {}}
    for row in rows:
        fit = row["fit_id"]
        if fit not in groups or (row["split"], row["view"], row["scheme"]) != (
            "temporal_2022", "test", "age_sex"
        ):
            continue
        key = row["series"]
        if key in groups[fit]:
            raise ValueError(f"duplicate series for {fit}: {key}")
        if row["source_kind"] != "fit" or row["scored_years"] != "2023|2024|2025":
            raise ValueError(f"wrong source or scored years: {key}")
        value = float(row["worst_csd"])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite CSD: {key}")
        groups[fit][key] = row
    a, b = groups[CONTEXTUAL], groups[LOOKUP]
    if len(a) != 18 or set(a) != set(b):
        raise ValueError("expected exactly 18 identical series for both policies")
    return [
        dict(series=key, scored_years=a[key]["scored_years"],
             contextual_csd=float(a[key]["worst_csd"]),
             age_lookup_csd=float(b[key]["worst_csd"]),
             difference=float(a[key]["worst_csd"]) - float(b[key]["worst_csd"]),
             contextual_winner_trace_sha256=a[key]["winner_trace_sha256"],
             age_lookup_winner_trace_sha256=b[key]["winner_trace_sha256"])
        for key in sorted(a)
    ]


def summarize(pairs: list[dict[str, object]]) -> dict[str, object]:
    a = np.asarray([row["contextual_csd"] for row in pairs], dtype=float)
    b = np.asarray([row["age_lookup_csd"] for row in pairs], dtype=float)
    if a.size != 18 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("expected 18 finite pairs")
    diff = a - b
    rng = np.random.default_rng(42)
    draws = diff[rng.integers(0, 18, size=(10_000, 18))].mean(axis=1)
    low, high = np.percentile(draws, (2.5, 97.5))
    sd = float(diff.std(ddof=1))
    return dict(
        n_series=18, contextual_mean=float(a.mean()), age_lookup_mean=float(b.mean()),
        contextual_sd=float(a.std(ddof=1)), age_lookup_sd=float(b.std(ddof=1)),
        difference=float(diff.mean()), paired_difference_sd=sd,
        median_difference=float(np.median(diff)), ci_low=float(low), ci_high=float(high),
        p_two_sided_exact_sign_flip=exact_paired_sign_flip_pvalue(diff),
        paired_cohens_dz=float(diff.mean() / sd) if sd > 0 else None,
        wins=int((diff < 0).sum()), ties=int((diff == 0).sum()), losses=int((diff > 0).sum()),
    )


def build_summary() -> tuple[dict[str, object], list[dict[str, object]]]:
    source = EVALUATION / "per_series.csv"
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((EVALUATION / "artifact_manifest.json").read_text())
    if digest != SOURCE_SHA256 or digest != manifest["files"]["evaluation/per_series.csv"]["sha256"]:
        raise ValueError("saved outcome digest does not match the pinned sealed input")
    pairs = pair_rows(list(csv.DictReader(io.StringIO(raw.decode("utf-8")))))
    summary = dict(
        schema_version=1,
        classification="post-lock exploratory contrast of corrected post-hoc replay",
        contextual_fit_id=CONTEXTUAL, age_lookup_fit_id=LOOKUP,
        split="temporal_2022", view="test", scheme="age_sex",
        metric="per-series worst-cohort cumulative share deficit; lower is better",
        unit="18 Warsaw series, equally weighted; one frozen fit per policy",
        source_csv=str(source.relative_to(ROOT)), source_csv_sha256=digest,
        fit_sha256={fit: manifest["provenance"]["fit_sha256"][fit] for fit in (CONTEXTUAL, LOOKUP)},
        bootstrap=dict(draws=10_000, seed=42, ordering="sorted series", interval="paired percentile 2.5/97.5"),
        exact_test=dict(assignments=2**18, two_sided=True, tail_tolerance=1e-15),
        multiple_comparisons="one requested exploratory contrast; p-value unadjusted",
        limitations=["not a fresh holdout or preregistered test", "common-city dependence",
                     "sign-flip inference assumes sign-exchangeability under the null",
                     "no equivalence conclusion from a non-significant result"],
        statistics=summarize(pairs),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistical_helper_sha256=hashlib.sha256((ROOT / "src/iclr_stats.py").read_bytes()).hexdigest(),
        numpy_version=np.__version__,
    )
    return summary, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    summary, pairs = build_summary()
    serialized = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        with (OUTPUT / "per_series.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
        with (OUTPUT / "summary.json").open("x") as handle:
            handle.write(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
