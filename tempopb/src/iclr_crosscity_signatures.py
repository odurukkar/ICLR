"""Post-hoc winner-set verification for the native cross-city replay.

This module is an audit artifact generator, not a protocol lock.  It replays
the already frozen Warsaw endowment fit and Equal Shares on the already
reported native-approval corpus, preserving canonical winner IDs so exact
allocation equality can be checked without a floating-point metric proxy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from cohorts import group_outcome
from iclr_corpus import CorpusConfig, SeriesRef, build_series_index, load_series
import iclr_crosscity
from iclr_env import (
    EnvConfig,
    RolloutState,
    SelectorPolicy,
    endowment_selector,
    uniform_policy,
)
from iclr_policy import linear_policy
from parse_pb import PBInstance


FROZEN_WARSAW_FIT = (
    iclr_crosscity.RESULTS / "iclr_train" / "run_main_seed42.json"
)
OUT_SIGNATURES = iclr_crosscity.RESULTS / "iclr_crosscity_winner_signatures.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_corpus(
    refs: Sequence[SeriesRef], manifest_files: Mapping[str, Mapping[str, object]]
) -> Dict[str, Dict[str, object]]:
    """Verify selected corpus bytes against the historical manifest hashes."""
    authenticated: Dict[str, Dict[str, object]] = {}
    for ref in refs:
        for path in ref.paths:
            if path.name in authenticated:
                raise ValueError(f"duplicate native corpus basename: {path.name}")
            expected = manifest_files.get(path.name)
            if expected is None:
                raise ValueError(f"manifest is missing native corpus file {path.name}")
            actual_sha = _sha256(path)
            if actual_sha != expected.get("sha256"):
                raise ValueError(f"sha256 mismatch for native corpus file {path.name}")
            actual_bytes = path.stat().st_size
            if actual_bytes != expected.get("bytes"):
                raise ValueError(f"byte-size mismatch for native corpus file {path.name}")
            authenticated[path.name] = {
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
    return dict(sorted(authenticated.items()))


def load_verification_selectors(
    fit_path: Path, cfg: EnvConfig
) -> Dict[str, SelectorPolicy]:
    """Load the learned selector from the same fit path recorded in provenance."""
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    learned_policy = linear_policy(fit["best_weights"], scheme=cfg.scheme)
    return {
        "learned-endowment": endowment_selector(learned_policy, cfg),
        "mes": endowment_selector(uniform_policy, cfg),
    }


def capture_series_winners(
    ref: SeriesRef,
    instances: Mapping[int, PBInstance],
    selectors: Mapping[str, SelectorPolicy],
    *,
    scored_years: Sequence[int],
    cfg: Optional[EnvConfig] = None,
) -> Dict[str, object]:
    """Replay both policies and retain sorted project IDs for every election."""
    cfg = cfg or EnvConfig()
    expected_years = set(ref.years)
    actual_years = set(instances)
    if actual_years != expected_years:
        missing_years = sorted(expected_years - actual_years)
        unexpected_years = sorted(actual_years - expected_years)
        raise ValueError(
            f"{ref.key}: missing elections={missing_years}, "
            f"unexpected elections={unexpected_years}"
        )
    policy_names = ("learned-endowment", "mes")
    missing = [name for name in policy_names if name not in selectors]
    if missing:
        raise ValueError(f"missing winner-signature policies: {missing}")

    winners_by_policy: Dict[str, Dict[int, list[str]]] = {}
    for name in policy_names:
        state = RolloutState()
        rows: Dict[int, list[str]] = {}
        for year in ref.years:
            inst = instances.get(year)
            if inst is None:
                continue
            winners = set(selectors[name](inst, state))
            rows[year] = sorted(winners)
            state.update(group_outcome(inst, winners, cfg.scheme))
        winners_by_policy[name] = rows

    scored = set(scored_years)
    elections = []
    for year in ref.years:
        if year not in winners_by_policy[policy_names[0]]:
            continue
        learned = winners_by_policy["learned-endowment"][year]
        mes = winners_by_policy["mes"][year]
        elections.append(
            {
                "year": year,
                "scored": year in scored,
                "exact_winner_match": learned == mes,
                "winners": {
                    "learned-endowment": learned,
                    "mes": mes,
                },
            }
        )

    scored_rows = [row for row in elections if row["scored"]]
    return {
        "series": ref.key,
        "city": ref.city,
        "exactly_unchanged_on_scored_elections": bool(scored_rows)
        and all(row["exact_winner_match"] for row in scored_rows),
        "exactly_unchanged_on_all_elections": bool(elections)
        and all(row["exact_winner_match"] for row in elections),
        "elections": elections,
    }


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(iclr_crosscity.ROOT.resolve()).as_posix()


def build_verification() -> Dict[str, object]:
    """Rebuild the post-hoc exact-winner audit from authenticated inputs."""
    cfg = EnvConfig()
    manifest = json.loads(iclr_crosscity.OUT_MANIFEST.read_text(encoding="utf-8"))
    index = build_series_index(CorpusConfig(data_dir=iclr_crosscity.DATA_DIR))
    refs = iclr_crosscity.select_crosscity_series(index)

    manifest_files = manifest["corpus"]["files"]
    authenticated_files = authenticate_corpus(refs, manifest_files)
    if set(authenticated_files) != set(manifest_files):
        missing = sorted(set(manifest_files) - set(authenticated_files))
        unexpected = sorted(set(authenticated_files) - set(manifest_files))
        raise ValueError(
            "native corpus file-set mismatch: "
            f"missing selected files={missing}, unexpected selected files={unexpected}"
        )

    selectors = load_verification_selectors(FROZEN_WARSAW_FIT, cfg)
    series = []
    for ref in refs:
        series.append(
            capture_series_winners(
                ref,
                load_series(ref),
                selectors,
                scored_years=tuple(
                    year for year in ref.years if year >= iclr_crosscity.FIRST_SCORED_YEAR
                ),
                cfg=cfg,
            )
        )
    series.sort(key=lambda row: row["series"])

    unchanged = [
        row["series"]
        for row in series
        if row["exactly_unchanged_on_scored_elections"]
    ]
    changed = [
        row["series"]
        for row in series
        if not row["exactly_unchanged_on_scored_elections"]
    ]
    unchanged_all = [
        row["series"]
        for row in series
        if row["exactly_unchanged_on_all_elections"]
    ]
    n_scored_elections = sum(
        1 for row in series for election in row["elections"] if election["scored"]
    )
    n_elections = sum(len(row["elections"]) for row in series)
    if len(series) != manifest["corpus"]["n_series"]:
        raise ValueError("native corpus series count differs from the source manifest")
    if n_elections != manifest["corpus"]["n_elections"]:
        raise ValueError("native corpus election count differs from the source manifest")
    if n_scored_elections != manifest["corpus"]["n_scored_elections"]:
        raise ValueError("native corpus scored-election count differs from the source manifest")

    return {
        "schema_version": 1,
        "artifact_kind": "post-hoc-verification",
        "protocol_lock": False,
        "replay_configuration": {
            "scheme": cfg.scheme,
            "completion": cfg.completion,
        },
        "verification_scope": {
            "comparison": ["learned-endowment", "mes"],
            "series_unchanged_definition": (
                "all scored elections have exactly equal sorted winner-ID lists"
            ),
            "scored_years": f">= {iclr_crosscity.FIRST_SCORED_YEAR}",
            "includes_warm_up_winner_ids": True,
        },
        "sources": {
            "source_manifest": {
                "path": _relative_path(iclr_crosscity.OUT_MANIFEST),
                "sha256": _sha256(iclr_crosscity.OUT_MANIFEST),
            },
            "frozen_warsaw_fit": {
                "path": _relative_path(FROZEN_WARSAW_FIT),
                "sha256": _sha256(FROZEN_WARSAW_FIT),
            },
        },
        "authenticated_corpus": {
            "n_files": len(authenticated_files),
            "files": authenticated_files,
        },
        "summary": {
            "n_series": len(series),
            "n_elections": n_elections,
            "n_scored_elections": n_scored_elections,
            "n_exactly_unchanged_on_scored_elections": len(unchanged),
            "n_changed_on_scored_elections": len(changed),
            "n_exactly_unchanged_on_all_elections": len(unchanged_all),
            "exactly_unchanged_series": unchanged,
            "changed_series": changed,
            "exactly_unchanged_on_all_elections_series": unchanged_all,
        },
        "series": series,
    }


def write_verification(path: Path = OUT_SIGNATURES) -> Dict[str, object]:
    """Write the deterministic post-hoc verification artifact."""
    payload = build_verification()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    path.write_bytes(serialized)
    return payload


def main() -> None:
    payload = write_verification()
    summary = payload["summary"]
    print(
        "wrote "
        f"{OUT_SIGNATURES}: "
        f"{summary['n_exactly_unchanged_on_scored_elections']}/"
        f"{summary['n_series']} series exactly unchanged on scored elections"
    )


if __name__ == "__main__":
    main()
