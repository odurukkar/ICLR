from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

import iclr_external_validation as ext


def _city_row(
    n_series: int,
    difference: float = -0.01,
    ci_high: float = -0.001,
    holm: float = 0.04,
    welfare: float = 0.99,
    exclusion: float = 0.0,
) -> dict[str, float]:
    return {
        "n_series": n_series,
        "difference": difference,
        "ci_high": ci_high,
        "holm_p_value": holm,
        "welfare_ratio": welfare,
        "exclusion_difference": exclusion,
    }


def _gold_evidence() -> dict[str, object]:
    return {
        "seeds": {
            str(seed): {
                "cities": {
                    "Poland/Katowice": _city_row(14),
                    "Poland/Krakow": _city_row(18),
                },
                "city_macro_difference": -0.01,
            }
            for seed in ext.SEEDS
        },
        "primary_seed": str(ext.PRIMARY_SEED),
        "endowment_actuated_series": 12,
        "priority_actuated_series": 5,
        "inference": {
            "bootstrap_draws": ext.BOOTSTRAP_DRAWS,
            "bootstrap_seed": ext.BOOTSTRAP_SEED,
            "sign_flip_draws": ext.SIGN_FLIP_DRAWS,
            "sign_flip_seed": ext.SIGN_FLIP_SEED,
        },
    }


def test_filename_selection_is_exact_and_content_free(tmp_path: Path) -> None:
    for city, years, tokens in (
        ("Katowice", range(2020, 2026), ext.KATOWICE_SERIES),
        ("Krakow", range(2018, 2026), ext.KRAKOW_SERIES),
    ):
        for year in years:
            for token in tokens:
                (tmp_path / f"Poland_{city}_{year}_{token}.pb").write_bytes(b"opaque")
    (tmp_path / "Poland_Krakow_2025_CITYWIDE.pb").write_bytes(b"ignore")

    selected = ext.select_external_files(tmp_path)

    assert len(selected) == 228
    assert sum(row.year <= 2022 for row in selected) == 132
    assert sum(row.year >= 2023 for row in selected) == 96
    assert all("CITYWIDE" not in row.path.name for row in selected)


def test_filename_selection_rejects_missing_or_extra_expected_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="expected 228"):
        ext.select_external_files(tmp_path)


def test_gold_gate_passes_only_all_frozen_conditions() -> None:
    decision = ext.evaluate_external_gate(_gold_evidence())
    assert decision["classification"] == "gold"
    assert decision["gold_pass"] is True

    evidence = _gold_evidence()
    evidence["seeds"]["2"]["cities"]["Poland/Krakow"]["difference"] = 0.0
    decision = ext.evaluate_external_gate(evidence)
    assert decision["classification"] == "falsified"
    assert decision["conditions"]["negative_each_city_all_seeds"]["passed"] is False


def test_silver_does_not_count_as_gold() -> None:
    evidence = _gold_evidence()
    evidence["seeds"][str(ext.PRIMARY_SEED)]["cities"]["Poland/Krakow"][
        "holm_p_value"
    ] = 0.08
    decision = ext.evaluate_external_gate(evidence)
    assert decision["classification"] == "silver"
    assert decision["gold_pass"] is False


def test_gate_rejects_threshold_drift() -> None:
    gates = ext.evidence_gate_config()
    gates["gold"]["city_macro_difference_max"] = -0.001
    with pytest.raises(RuntimeError, match="canonical"):
        ext.evaluate_external_gate(_gold_evidence(), gates)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda evidence: evidence["seeds"].pop("1"),
        lambda evidence: evidence["seeds"]["1"]["cities"].pop(
            "Poland/Katowice"
        ),
        lambda evidence: evidence.update(primary_seed="2"),
        lambda evidence: evidence["seeds"]["1"]["cities"][
            "Poland/Katowice"
        ].update(n_series=13),
    ),
)
def test_gate_rejects_incomplete_seed_city_or_series_family(mutation) -> None:
    evidence = _gold_evidence()
    mutation(evidence)
    with pytest.raises(RuntimeError, match="external evidence family"):
        ext.evaluate_external_gate(evidence)


def test_production_paths_are_canonical(tmp_path: Path) -> None:
    ext.require_canonical_paths(ext.DATA_DIR, ext.RESULT_ROOT)
    with pytest.raises(RuntimeError, match="canonical external paths"):
        ext.require_canonical_paths(tmp_path / "data", ext.RESULT_ROOT)
    with pytest.raises(RuntimeError, match="canonical external paths"):
        ext.require_canonical_paths(ext.DATA_DIR, tmp_path / "results")


def test_source_checkout_authenticates_commit_and_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    source = checkout / "pb_files"
    source.mkdir(parents=True)
    ballot = source / "Poland_Krakow_2025_Bronowice.pb"
    ballot.write_bytes(b"opaque ballot bytes")
    subprocess.run(["git", "init", "-q", checkout], check=True)
    subprocess.run(["git", "-C", checkout, "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", checkout, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", checkout, "add", "pb_files"], check=True)
    subprocess.run(["git", "-C", checkout, "commit", "-qm", "fixture"], check=True)
    subprocess.run(
        ["git", "-C", checkout, "remote", "add", "origin", ext.PABULIB_REMOTE],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", checkout, "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(ext, "PABULIB_COMMIT", commit)
    selected = [ext.SelectedFile(ballot, "Krakow", "Bronowice", 2025)]

    blobs = ext.verify_source_checkout(source, selected)
    assert blobs[ballot.name]

    ballot.write_bytes(b"mutated after commit")
    with pytest.raises(RuntimeError, match="blob differs"):
        ext.verify_source_checkout(source, selected)


def test_external_split_has_chronological_warmup_and_score_counts() -> None:
    index = {}
    for city, years, tokens in (
        ("Katowice", tuple(range(2020, 2026)), ext.KATOWICE_SERIES),
        ("Krakow", tuple(range(2018, 2026)), ext.KRAKOW_SERIES),
    ):
        for token in tokens:
            key = f"Poland/{city}/{token}"
            index[key] = ext.SeriesRef(key, years, tuple(Path(str(y)) for y in years))
    split = ext.external_split(index)
    assert sum(len(years) for _, years in split.train) == 132
    assert sum(len(years) for _, years in split.test) == 96
    assert all(max(years) <= 2022 for _, years in split.train)
    assert all(min(years) >= 2023 for _, years in split.test)


def test_parser_cannot_run_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def parser_spy(_path):
        nonlocal called
        called = True
        raise AssertionError("parser should not run")

    monkeypatch.setattr(ext, "parse_pb_file", parser_spy)
    with pytest.raises(RuntimeError, match="receipt"):
        ext.load_external_corpus(
            tmp_path / "manifest.json",
            tmp_path / "data",
            tmp_path / "heldout_opened.json",
        )
    assert called is False


def test_original_fit_inventory_names_all_six_fixed_policies() -> None:
    assert set(ext.original_fit_paths()) == {
        f"fit/{arm}/seed-{seed}"
        for arm in ("endowment", "priority")
        for seed in ext.SEEDS
    }


def test_published_anchor_rejects_wrong_lock_hash(tmp_path: Path) -> None:
    lock = tmp_path / "protocol_lock.json"
    lock.write_text("{}\n", encoding="utf-8")
    actual = ext._sha256(lock)
    anchor = tmp_path / "published_lock_anchor.json"
    wrong = "0" * 64
    ext.write_published_anchor(anchor, wrong, f"confirm {wrong}")
    with pytest.raises(RuntimeError, match="published lock anchor"):
        ext.verify_published_anchor(anchor, lock)

    anchor.unlink()
    ext.write_published_anchor(anchor, actual, f"confirm {actual}")
    ext.verify_published_anchor(anchor, lock)


def test_evaluation_reentry_is_rejected(tmp_path: Path) -> None:
    ext.assert_evaluation_unopened(tmp_path)
    (tmp_path / "heldout_opened.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already been opened"):
        ext.assert_evaluation_unopened(tmp_path)


def test_verify_lock_detects_tracked_mutation(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    lock = tmp_path / "lock.json"
    ext.write_external_lock(lock, tmp_path, {"artifact": tracked})
    ext.verify_external_lock(lock, tmp_path, {"artifact": tracked})

    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ext.verify_external_lock(lock, tmp_path, {"artifact": tracked})
