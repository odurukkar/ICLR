"""Regressions for the cross-city corpus rules and the sampled sign-flip test.

The cross-city evaluation only means anything if its corpus rules hold: no
refitting city, no citywide aggregate double-counting district voters, no
Gdynia neighbourhood contributing two ballots over one electorate, and no
series without both a warm-up edition and a held-out edition.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

import iclr_crosscity
from iclr_corpus import SeriesRef
from iclr_env import EnvConfig
from iclr_stats import (
    exact_paired_sign_flip_pvalue,
    monte_carlo_paired_sign_flip_pvalue,
)
from parse_pb import PBInstance, Project, Vote


def _ref(key: str, years) -> SeriesRef:
    return SeriesRef(key=key, years=tuple(years), paths=tuple(Path(f"{key}-{y}.pb") for y in years))


def _index(*refs: SeriesRef):
    return {ref.key: ref for ref in refs}


@pytest.fixture
def budget_by_key(monkeypatch):
    """Replace the on-disk budget probe used to pick one Gdynia track."""
    budgets = {}

    def fake(ref: SeriesRef) -> float:
        return budgets.get(ref.key, 0.0)

    monkeypatch.setattr(iclr_crosscity, "_budget_weight", fake)
    return budgets


def test_training_city_is_excluded(budget_by_key) -> None:
    index = _index(
        _ref("Poland/Warszawa/Wola", (2021, 2022, 2023)),
        _ref("Poland/Gdynia/Cisowa", (2021, 2022, 2023)),
    )
    keys = [ref.key for ref in iclr_crosscity.select_crosscity_series(index)]
    assert keys == ["Poland/Gdynia/Cisowa"]


def test_citywide_aggregates_are_dropped(budget_by_key) -> None:
    """Citywide ballots are already counted inside the district series."""
    index = _index(
        _ref("Poland/Łódź/CITYWIDE", (2022, 2023, 2024)),
        _ref("Poland/Łódź/Bałuty-Doły", (2022, 2023, 2024)),
    )
    keys = [ref.key for ref in iclr_crosscity.select_crosscity_series(index)]
    assert keys == ["Poland/Łódź/Bałuty-Doły"]


def test_one_gdynia_track_per_neighbourhood(budget_by_key) -> None:
    """The small and large ballots share one electorate, so only one survives."""
    small = _ref("Poland/Gdynia/Cisowa | small", (2021, 2022, 2023))
    large = _ref("Poland/Gdynia/Cisowa | large", (2021, 2022, 2023))
    budget_by_key[small.key] = 70_000.0
    budget_by_key[large.key] = 630_000.0
    keys = [ref.key for ref in iclr_crosscity.select_crosscity_series(_index(small, large))]
    assert keys == ["Poland/Gdynia/Cisowa | large"]


def test_series_needs_a_warm_up_and_a_held_out_edition(budget_by_key) -> None:
    index = _index(
        _ref("Poland/Gdynia/OnlyOld", (2020, 2021, 2022)),
        _ref("Poland/Łódź/OnlyNew", (2023, 2024, 2025)),
        _ref("Poland/Gdynia/Both", (2022, 2023, 2024)),
    )
    keys = [ref.key for ref in iclr_crosscity.select_crosscity_series(index)]
    assert keys == ["Poland/Gdynia/Both"]


def test_sampled_sign_flip_matches_exact_enumeration() -> None:
    rng = np.random.default_rng(7)
    diff = rng.normal(-0.02, 0.05, size=12)
    exact = exact_paired_sign_flip_pvalue(diff)
    sampled = monte_carlo_paired_sign_flip_pvalue(diff, draws=200_000, seed=3)
    assert sampled == pytest.approx(exact, abs=0.01)


def test_sampled_sign_flip_never_reports_zero() -> None:
    """The (1+extreme)/(1+draws) estimator keeps the p-value strictly positive."""
    diff = np.full(40, -0.1)
    assert monte_carlo_paired_sign_flip_pvalue(diff, draws=2_000, seed=1) > 0.0


def test_sampled_sign_flip_is_reproducible() -> None:
    diff = np.random.default_rng(11).normal(-0.01, 0.04, size=30)
    first = monte_carlo_paired_sign_flip_pvalue(diff, draws=20_000, seed=5)
    second = monte_carlo_paired_sign_flip_pvalue(diff, draws=20_000, seed=5)
    assert first == second


def test_winner_verification_uses_sorted_exact_winners_in_scored_elections() -> None:
    """A series is unchanged only when every scored winner set is identical."""
    try:
        signatures = importlib.import_module("iclr_crosscity_signatures")
    except ModuleNotFoundError:
        pytest.fail("native cross-city winner-signature verification is missing")

    ref = _ref("Poland/Gdynia/Fixture", (2022, 2023))
    instances = {
        year: PBInstance(
            path=f"fixture-{year}",
            meta={"budget": "10", "vote_type": "approval"},
            projects={
                "a": Project("a", 2.0, None),
                "z": Project("z", 2.0, None),
            },
            votes=[Vote("v", ("a", "z"), age=30, sex="F")],
        )
        for year in ref.years
    }

    state_ids = {"learned-endowment": [], "mes": []}

    def learned(inst, state):
        del inst
        state_ids["learned-endowment"].append(id(state))
        return {"z", "a"} if state.year_index == 0 else {"a"}

    def mes(inst, state):
        del inst
        state_ids["mes"].append(id(state))
        return {"a", "z"} if state.year_index == 0 else {"z"}

    got = signatures.capture_series_winners(
        ref,
        instances,
        {"learned-endowment": learned, "mes": mes},
        scored_years=(2023,),
        cfg=EnvConfig(),
    )

    assert got["exactly_unchanged_on_scored_elections"] is False
    assert got["exactly_unchanged_on_all_elections"] is False
    assert len(set(state_ids["learned-endowment"])) == 1
    assert len(set(state_ids["mes"])) == 1
    assert state_ids["learned-endowment"][0] != state_ids["mes"][0]
    assert got["elections"] == [
        {
            "year": 2022,
            "scored": False,
            "exact_winner_match": True,
            "winners": {
                "learned-endowment": ["a", "z"],
                "mes": ["a", "z"],
            },
        },
        {
            "year": 2023,
            "scored": True,
            "exact_winner_match": False,
            "winners": {
                "learned-endowment": ["a"],
                "mes": ["z"],
            },
        },
    ]


def test_winner_verification_rejects_a_manifest_data_hash_mismatch(tmp_path) -> None:
    """Copied manifest hashes are not authentication; current bytes must match."""
    signatures = importlib.import_module("iclr_crosscity_signatures")
    corpus_file = tmp_path / "fixture.pb"
    corpus_file.write_bytes(b"current native-approval data")
    ref = SeriesRef(key="Poland/Gdynia/Fixture", years=(2023,), paths=(corpus_file,))
    manifest_files = {
        corpus_file.name: {
            "bytes": corpus_file.stat().st_size,
            "sha256": "0" * 64,
        }
    }

    with pytest.raises(ValueError, match="sha256 mismatch.*fixture.pb"):
        signatures.authenticate_corpus((ref,), manifest_files)


def test_winner_verification_rejects_duplicate_corpus_basenames(tmp_path) -> None:
    """Filename-keyed provenance must not silently collapse two data files."""
    signatures = importlib.import_module("iclr_crosscity_signatures")
    first = tmp_path / "first" / "fixture.pb"
    second = tmp_path / "second" / "fixture.pb"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    refs = (
        SeriesRef("Poland/Gdynia/First", (2023,), (first,)),
        SeriesRef("Poland/Gdynia/Second", (2023,), (second,)),
    )
    manifest_files = {
        "fixture.pb": {
            "bytes": len(b"same bytes"),
            "sha256": hashlib.sha256(b"same bytes").hexdigest(),
        }
    }

    with pytest.raises(ValueError, match="duplicate native corpus basename.*fixture.pb"):
        signatures.authenticate_corpus(refs, manifest_files)


def test_winner_verification_rejects_a_missing_warm_up_election() -> None:
    """Dropping warm-up data would change the policy-specific carried history."""
    signatures = importlib.import_module("iclr_crosscity_signatures")
    ref = _ref("Poland/Gdynia/Fixture", (2022, 2023))
    scored = PBInstance(
        path="fixture-2023",
        meta={"budget": "10", "vote_type": "approval"},
        projects={"a": Project("a", 2.0, None)},
        votes=[Vote("v", ("a",), age=30, sex="F")],
    )

    def selector(inst, state):
        del inst, state
        return {"a"}

    with pytest.raises(ValueError, match=r"missing elections=\[2022\]"):
        signatures.capture_series_winners(
            ref,
            {2023: scored},
            {"learned-endowment": selector, "mes": selector},
            scored_years=(2023,),
        )


def test_winner_verification_rejects_an_alternate_replay_configuration() -> None:
    """The frozen audit must not silently change completion or cohort scheme."""
    signatures = importlib.import_module("iclr_crosscity_signatures")

    with pytest.raises(TypeError):
        signatures.build_verification(EnvConfig(completion=False))


def test_winner_verification_selector_is_loaded_from_the_declared_fit(tmp_path) -> None:
    """A zero-weight fit loaded from the given path must reproduce MES exactly."""
    signatures = importlib.import_module("iclr_crosscity_signatures")
    fit_path = tmp_path / "declared-fit.json"
    fit_path.write_text(json.dumps({"best_weights": [0.0] * 6}), encoding="utf-8")
    selectors = signatures.load_verification_selectors(fit_path, EnvConfig())
    ref = _ref("Poland/Gdynia/Fixture", (2023,))
    instances = {
        2023: PBInstance(
            path="fixture-2023",
            meta={"budget": "8", "vote_type": "approval"},
            projects={
                "a": Project("a", 4.0, None),
                "b": Project("b", 4.0, None),
            },
            votes=[
                Vote("young", ("a",), age=20, sex="F"),
                Vote("senior", ("b",), age=70, sex="M"),
            ],
        )
    }

    got = signatures.capture_series_winners(
        ref,
        instances,
        selectors,
        scored_years=(2023,),
    )

    assert got["exactly_unchanged_on_scored_elections"] is True
    assert got["elections"][0]["winners"] == {
        "learned-endowment": ["a", "b"],
        "mes": ["a", "b"],
    }


@pytest.mark.skipif(
    not (
        iclr_crosscity.OUT_MANIFEST.is_file()
        and iclr_crosscity.DATA_DIR.is_dir()
    ),
    reason="cross-city study or its unbundled public corpus is unavailable",
)
def test_winner_verification_rebuilds_the_exact_native_crosscity_classification() -> None:
    """The 36/58 claim must be derived from scored-election winner equality."""
    signatures = importlib.import_module("iclr_crosscity_signatures")
    payload = signatures.build_verification()
    canonical_bytes = (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert signatures.OUT_SIGNATURES.read_bytes() == canonical_bytes
    assert json.loads(canonical_bytes) == payload
    manifest = json.loads(iclr_crosscity.OUT_MANIFEST.read_text())
    fit_path = iclr_crosscity.RESULTS / "iclr_train" / "run_main_seed42.json"

    assert payload["artifact_kind"] == "post-hoc-verification"
    assert payload["protocol_lock"] is False
    assert payload["replay_configuration"] == {
        "scheme": "age_sex",
        "completion": True,
    }
    assert payload["verification_scope"] == {
        "comparison": ["learned-endowment", "mes"],
        "series_unchanged_definition": (
            "all scored elections have exactly equal sorted winner-ID lists"
        ),
        "scored_years": ">= 2023",
        "includes_warm_up_winner_ids": True,
    }
    assert payload["sources"] == {
        "source_manifest": {
            "path": "results/iclr_crosscity_manifest.json",
            "sha256": hashlib.sha256(iclr_crosscity.OUT_MANIFEST.read_bytes()).hexdigest(),
        },
        "frozen_warsaw_fit": {
            "path": "results/iclr_train/run_main_seed42.json",
            "sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
        },
    }
    assert payload["authenticated_corpus"] == {
        "n_files": len(manifest["corpus"]["files"]),
        "files": manifest["corpus"]["files"],
    }

    rows = payload["series"]
    assert [row["series"] for row in rows] == sorted(row["series"] for row in rows)
    unchanged = []
    unchanged_all = []
    n_scored_elections = 0
    for row in rows:
        assert [election["year"] for election in row["elections"]] == sorted(
            election["year"] for election in row["elections"]
        )
        scored_matches = []
        all_matches = []
        for election in row["elections"]:
            learned = election["winners"]["learned-endowment"]
            mes = election["winners"]["mes"]
            assert learned == sorted(learned)
            assert mes == sorted(mes)
            assert election["exact_winner_match"] is (learned == mes)
            all_matches.append(learned == mes)
            if election["scored"]:
                n_scored_elections += 1
                scored_matches.append(learned == mes)
        assert row["exactly_unchanged_on_scored_elections"] is (
            bool(scored_matches) and all(scored_matches)
        )
        assert row["exactly_unchanged_on_all_elections"] is (
            bool(all_matches) and all(all_matches)
        )
        if row["exactly_unchanged_on_scored_elections"]:
            unchanged.append(row["series"])
        if row["exactly_unchanged_on_all_elections"]:
            unchanged_all.append(row["series"])

    assert payload["summary"] == {
        "n_series": 58,
        "n_elections": 273,
        "n_scored_elections": 172,
        "n_exactly_unchanged_on_scored_elections": 36,
        "n_changed_on_scored_elections": 22,
        "n_exactly_unchanged_on_all_elections": 28,
        "exactly_unchanged_series": unchanged,
        "changed_series": sorted(set(row["series"] for row in rows) - set(unchanged)),
        "exactly_unchanged_on_all_elections_series": unchanged_all,
    }
    assert n_scored_elections == manifest["corpus"]["n_scored_elections"]
