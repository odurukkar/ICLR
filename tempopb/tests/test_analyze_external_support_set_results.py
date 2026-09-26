from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

import analyze_external_support_set_results as analysis
from analyze_external_support_set_results import (
    EXPECTED_SHA256,
    build_city_intervals,
    build_decision_conditions,
    build_series_effects,
    compute_diagnostics,
    load_and_verify_bundle,
    sha256_file,
    write_analysis_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "iclr_external_support_set_amendment"


def _copied_bundle(tmp_path: Path) -> Path:
    copied = tmp_path / "bundle"
    shutil.copytree(RESULT_ROOT, copied)
    return copied


def _accept_mutated_file(monkeypatch: pytest.MonkeyPatch, root: Path, relative: str) -> None:
    expected = dict(EXPECTED_SHA256)
    expected[relative] = sha256_file(root / relative)
    monkeypatch.setattr("analyze_external_support_set_results.EXPECTED_SHA256", expected)


def test_verified_bundle_builds_exact_frozen_effect_table() -> None:
    bundle = load_and_verify_bundle(RESULT_ROOT)
    effects = build_series_effects(bundle)
    assert bundle.decision["classification"] == "falsified"
    assert len(effects) == 192
    assert set(effects["seed"]) == {1, 2, 42}
    assert set(effects["arm"]) == {"endowment", "priority"}
    assert effects["series"].nunique() == 32
    assert set(effects["city"]) == {"Poland/Katowice", "Poland/Krakow"}
    assert list(effects.columns) == [
        "policy",
        "arm",
        "seed",
        "series",
        "city",
        "delta_csd",
        "delta_exclusion",
        "welfare_ratio",
        "actuated",
        "direction",
    ]


def test_effect_uses_the_literal_same_series_mes_delta_and_outcome_hash() -> None:
    effects = build_series_effects(load_and_verify_bundle(RESULT_ROOT))
    row = effects.loc[
        (effects["policy"] == "endowment/seed-1")
        & (effects["series"] == "Poland/Katowice/Giszowiec")
    ].iloc[0]

    assert row["delta_csd"] == -0.05439277845089753
    assert row["delta_exclusion"] == -0.028409090909090884
    assert row["welfare_ratio"] == 1.0541287386215865
    assert bool(row["actuated"]) is True
    assert row["direction"] == "win"


def test_hash_mismatch_aborts_before_analysis(tmp_path: Path) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evidence_decision.json"
    path.write_text(path.read_text().replace('"falsified"', '"gold"'))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_and_verify_bundle(copied)


def test_primary_seed_exact_zeros_remain_ties() -> None:
    effects = build_series_effects(load_and_verify_bundle(RESULT_ROOT))
    primary = effects.query("arm == 'endowment' and seed == 42")
    assert (primary["direction"] == "win").sum() == 8
    assert (primary["direction"] == "tie").sum() == 20
    assert (primary["direction"] == "loss").sum() == 4


def test_diagnostics_match_frozen_sparse_seed_sensitive_pattern() -> None:
    bundle = load_and_verify_bundle(RESULT_ROOT)
    effects = build_series_effects(bundle)
    diagnostics = compute_diagnostics(bundle, effects)

    assert diagnostics["decision"] == "falsified"
    assert diagnostics["primary"]["wins"] == 8
    assert diagnostics["primary"]["ties"] == 20
    assert diagnostics["primary"]["losses"] == 4
    assert diagnostics["across_seeds"]["always_win"] == 4
    assert diagnostics["across_seeds"]["always_tie"] == 13
    assert diagnostics["across_seeds"]["switches_sign"] == 5
    assert diagnostics["top3_improvement_concentration"] == 0.7889575669610045
    assert diagnostics["minimum_demographic_coverage"] == pytest.approx(
        0.9912280701754386
    )
    assert diagnostics["city_macro_differences"] == {
        "1": -0.013137966195055278,
        "2": -0.005357752651077373,
        "42": -0.007153800547544324,
    }
    assert diagnostics["actuation"] == {"endowment": 12, "priority": 3}


def test_top3_concentration_is_derived_from_primary_effect_values() -> None:
    bundle = load_and_verify_bundle(RESULT_ROOT)
    effects = build_series_effects(bundle)
    baseline = compute_diagnostics(bundle, effects)["top3_improvement_concentration"]
    mutated = effects.copy()
    changed_index = mutated.loc[
        (mutated["arm"] == "endowment")
        & (mutated["seed"] == 42)
        & (mutated["delta_csd"] < 0)
    ].index[0]
    mutated.loc[changed_index, "delta_csd"] *= 2

    changed = compute_diagnostics(bundle, mutated)["top3_improvement_concentration"]

    assert baseline == 0.7889575669610045
    assert changed != baseline


def test_city_intervals_copy_all_frozen_interval_and_p_value_fields() -> None:
    bundle = load_and_verify_bundle(RESULT_ROOT)
    intervals = build_city_intervals(bundle)

    assert intervals[
        ["seed", "city", "ci_low", "ci_high", "p_value", "p_value_kind", "holm_p_value"]
    ].to_dict("records") == [
        {
            "seed": 1,
            "city": "Poland/Katowice",
            "ci_low": -0.03069432927693753,
            "ci_high": 0.0016126922938029373,
            "p_value": 0.25,
            "p_value_kind": "exact_sign_flip",
            "holm_p_value": None,
        },
        {
            "seed": 1,
            "city": "Poland/Krakow",
            "ci_low": -0.040688923656287554,
            "ci_high": 0.008176038678425564,
            "p_value": 0.35882568359375,
            "p_value_kind": "exact_sign_flip",
            "holm_p_value": None,
        },
        {
            "seed": 2,
            "city": "Poland/Katowice",
            "ci_low": -0.00803283252667272,
            "ci_high": 0.0,
            "p_value": 1.0,
            "p_value_kind": "exact_sign_flip",
            "holm_p_value": None,
        },
        {
            "seed": 2,
            "city": "Poland/Krakow",
            "ci_low": -0.027793162678913012,
            "ci_high": 0.005117572627737076,
            "p_value": 0.59375,
            "p_value_kind": "exact_sign_flip",
            "holm_p_value": None,
        },
        {
            "seed": 42,
            "city": "Poland/Katowice",
            "ci_low": -0.028176841645247863,
            "ci_high": 0.0,
            "p_value": 0.25,
            "p_value_kind": "exact_sign_flip",
            "holm_p_value": 0.5,
        },
        {
            "seed": 42,
            "city": "Poland/Krakow",
            "ci_low": -0.025908745651570884,
            "ci_high": 0.0166219908152236,
            "p_value": 0.83984375,
            "p_value_kind": "exact_sign_flip",
            "holm_p_value": 0.83984375,
        },
    ]


def test_decision_conditions_copy_all_frozen_names_flags_and_observations() -> None:
    conditions = build_decision_conditions(load_and_verify_bundle(RESULT_ROOT))
    records = conditions.set_index("condition")

    assert records["passed"].to_dict() == {
        "endowment_actuation": True,
        "exclusion_increase_all_cities_seeds": False,
        "macro_at_most_minus_0_005_all_seeds": True,
        "macro_seed_spread": False,
        "negative_each_city_all_seeds": True,
        "primary_ci_excludes_zero_each_city": False,
        "primary_holm_p_each_city": False,
        "welfare_ratio_all_cities_seeds": True,
    }
    assert {name: json.loads(value) for name, value in records["observed_json"].items()} == {
        "endowment_actuation": {"endowment": 12, "priority": 3},
        "exclusion_increase_all_cities_seeds": {
            "1": {"Poland/Katowice": 0.0077914287919601056, "Poland/Krakow": 0.0037132806449263708},
            "2": {"Poland/Katowice": 0.001009266905220662, "Poland/Krakow": -0.0013574706308101699},
            "42": {"Poland/Katowice": 0.00850163782511327, "Poland/Krakow": 0.0015156386942720259},
        },
        "macro_at_most_minus_0_005_all_seeds": {
            "1": -0.013137966195055278,
            "2": -0.005357752651077373,
            "42": -0.007153800547544324,
        },
        "macro_seed_spread": 0.0077802135439779055,
        "negative_each_city_all_seeds": {
            "1": {"Poland/Katowice": -0.01246362199567651, "Poland/Krakow": -0.013812310394434046},
            "2": {"Poland/Katowice": -0.0026776108422242395, "Poland/Krakow": -0.008037894459930507},
            "42": {"Poland/Katowice": -0.011736055913730586, "Poland/Krakow": -0.002571545181358063},
        },
        "primary_ci_excludes_zero_each_city": {
            "Poland/Katowice": 0.0,
            "Poland/Krakow": 0.0166219908152236,
        },
        "primary_holm_p_each_city": {
            "Poland/Katowice": 0.5,
            "Poland/Krakow": 0.83984375,
        },
        "welfare_ratio_all_cities_seeds": {
            "1": {"Poland/Katowice": 0.9912512530757314, "Poland/Krakow": 1.001394276749159},
            "2": {"Poland/Katowice": 1.000169246592285, "Poland/Krakow": 1.0015259136676733},
            "42": {"Poland/Katowice": 0.9924229602530887, "Poland/Krakow": 1.0041371602961562},
        },
    }


def test_analysis_bundle_is_complete_and_deterministic(tmp_path: Path) -> None:
    first = write_analysis_bundle(RESULT_ROOT, tmp_path / "first")
    second = write_analysis_bundle(RESULT_ROOT, tmp_path / "second")
    expected = {
        "manifest.json",
        "effects.csv",
        "city_intervals.csv",
        "decision_conditions.csv",
        "diagnostics.json",
        "analysis-report.md",
        "stats-appendix.md",
        "figure-catalog.md",
    }
    assert set(first) == expected
    assert {path.name for path in first.values()} == expected
    assert {name: path.read_bytes() for name, path in first.items()} == {
        name: second[name].read_bytes() for name in second
    }

    manifest = json.loads(first["manifest.json"].read_text())
    assert manifest["classification"] == "falsified"
    assert manifest["entry_point"] == "python src/analyze_external_support_set_results.py"
    assert set(manifest["input_sha256"]) == set(EXPECTED_SHA256)
    assert set(manifest["output_sha256"]) == expected - {"manifest.json"}
    assert "timestamp" not in manifest
    assert str(tmp_path) not in first["manifest.json"].read_text()
    assert "--output-dir" not in first["manifest.json"].read_text()
    for name, digest in manifest["output_sha256"].items():
        assert digest == sha256_file(first[name])


def test_analysis_bundle_renders_exact_decimal_and_report_structure(tmp_path: Path) -> None:
    paths = write_analysis_bundle(RESULT_ROOT, tmp_path / "bundle")
    diagnostics = json.loads(paths["diagnostics.json"].read_text())
    report = paths["analysis-report.md"].read_text()

    assert diagnostics["top3_improvement_concentration"] == 0.7889575669610045
    assert "Top-three improvement concentration: 0.7889575669610045." in report
    headings = [
        "## 1. Immutable protocol identity and support-set projection",
        "## 2. Authenticated corpus and demographic coverage",
        "## 3. City-macro differences and frozen city intervals",
        "## 4. Descriptive series diagnostics",
        "## 5. Frozen mechanical gate conditions",
        "## 6. Mechanical decision",
    ]
    assert [report.index(heading) for heading in headings] == sorted(
        report.index(heading) for heading in headings
    )
    assert report.splitlines()[-1] == "Mechanical decision: falsified."


def test_analysis_bundle_rejects_result_tree_output_before_creation(tmp_path: Path) -> None:
    copied = _copied_bundle(tmp_path)
    descendant = copied / "disallowed-output"

    for output_dir in (copied, descendant):
        with pytest.raises(ValueError, match="inside result_root"):
            write_analysis_bundle(copied, output_dir)
    assert not descendant.exists()


@pytest.mark.parametrize("entry_kind", ["symlink", "directory"])
def test_analysis_bundle_rejects_nonregular_expected_output_entries(
    tmp_path: Path, entry_kind: str
) -> None:
    output_dir = tmp_path / entry_kind
    output_dir.mkdir()
    entry = output_dir / "effects.csv"
    if entry_kind == "symlink":
        target = tmp_path / "target.csv"
        target.write_text("preserve")
        entry.symlink_to(target)
    else:
        entry.mkdir()

    with pytest.raises(ValueError, match="regular non-symlink"):
        write_analysis_bundle(RESULT_ROOT, output_dir)
    if entry_kind == "symlink":
        assert target.read_text() == "preserve"


@pytest.mark.parametrize("entry_kind", ["extra-file", "extra-directory"])
def test_analysis_bundle_rejects_extra_output_entries(tmp_path: Path, entry_kind: str) -> None:
    output_dir = tmp_path / entry_kind
    output_dir.mkdir()
    extra = output_dir / "unexpected"
    if entry_kind == "extra-file":
        extra.write_text("unexpected")
    else:
        extra.mkdir()

    with pytest.raises(ValueError, match="strict bundle names"):
        write_analysis_bundle(RESULT_ROOT, output_dir)


def test_row_count_drift_is_rejected_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evaluation" / "per_series.json"
    payload = json.loads(path.read_text())
    payload["mes"].pop()
    path.write_text(json.dumps(payload))
    _accept_mutated_file(monkeypatch, copied, "evaluation/per_series.json")

    with pytest.raises(ValueError, match="32 rows"):
        load_and_verify_bundle(copied)


def test_missing_policy_is_rejected_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evaluation" / "per_series.json"
    payload = json.loads(path.read_text())
    del payload["priority/seed-42"]
    path.write_text(json.dumps(payload))
    _accept_mutated_file(monkeypatch, copied, "evaluation/per_series.json")

    with pytest.raises(ValueError, match="policy keys"):
        load_and_verify_bundle(copied)


def test_nonfalsified_decision_is_rejected_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evidence_decision.json"
    payload = json.loads(path.read_text())
    payload["classification"] = "gold"
    path.write_text(json.dumps(payload))
    _accept_mutated_file(monkeypatch, copied, "evidence_decision.json")

    with pytest.raises(ValueError, match="classification"):
        load_and_verify_bundle(copied)


def test_authenticated_election_count_must_be_exactly_228(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evaluation" / "per_series.json"
    payload = json.loads(path.read_text())
    payload["mes"][0]["years"].pop()
    path.write_text(json.dumps(payload))
    _accept_mutated_file(monkeypatch, copied, "evaluation/per_series.json")

    with pytest.raises(ValueError, match="228 authenticated elections"):
        load_and_verify_bundle(copied)


def test_coverage_row_count_drift_is_rejected_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evaluation" / "demographic_coverage.json"
    payload = json.loads(path.read_text())
    payload["files"].pop()
    path.write_text(json.dumps(payload))
    _accept_mutated_file(monkeypatch, copied, "evaluation/demographic_coverage.json")

    with pytest.raises(ValueError, match="96 coverage rows"):
        load_and_verify_bundle(copied)


def test_coverage_below_threshold_is_rejected_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evaluation" / "demographic_coverage.json"
    payload = json.loads(path.read_text())
    payload["files"][0]["coverage"] = 0.49
    path.write_text(json.dumps(payload))
    _accept_mutated_file(monkeypatch, copied, "evaluation/demographic_coverage.json")

    with pytest.raises(ValueError, match="coverage minimum"):
        load_and_verify_bundle(copied)


def test_nonfinite_coverage_is_rejected_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evaluation" / "demographic_coverage.json"
    payload = json.loads(path.read_text())
    payload["files"][0]["coverage"] = float("nan")
    path.write_text(json.dumps(payload))
    _accept_mutated_file(monkeypatch, copied, "evaluation/demographic_coverage.json")

    with pytest.raises(ValueError, match="finite numeric coverage"):
        load_and_verify_bundle(copied)


def test_all_hashes_finish_before_any_json_or_csv_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_sha256 = analysis.hashlib.sha256
    original_json_loads = analysis.json.loads
    original_read_csv = analysis.pd.read_csv

    class RecordingDigest:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._digest = original_sha256(*args, **kwargs)

        def update(self, data: bytes) -> None:
            self._digest.update(data)

        def hexdigest(self) -> str:
            events.append("hash")
            return self._digest.hexdigest()

    def spy_json_loads(*args: object, **kwargs: object) -> object:
        assert events[: len(EXPECTED_SHA256)] == ["hash"] * len(EXPECTED_SHA256)
        events.append("json")
        return original_json_loads(*args, **kwargs)

    def spy_read_csv(*args: object, **kwargs: object) -> object:
        assert events[: len(EXPECTED_SHA256)] == ["hash"] * len(EXPECTED_SHA256)
        assert events[len(EXPECTED_SHA256) :] == ["json"] * 8
        events.append("csv")
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(analysis.hashlib, "sha256", RecordingDigest)
    monkeypatch.setattr(analysis.json, "loads", spy_json_loads)
    monkeypatch.setattr(analysis.pd, "read_csv", spy_read_csv)

    load_and_verify_bundle(RESULT_ROOT)
    assert events == ["hash"] * len(EXPECTED_SHA256) + ["json"] * 8 + ["csv"]


def test_invalid_outcome_digest_format_is_rejected_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = _copied_bundle(tmp_path)
    path = copied / "evaluation" / "per_series.csv"
    original = "c4a59e0bd2b347fa655be4417379bee940d48f9e377ba3b85cbebb9402041855"
    path.write_text(path.read_text().replace(original, "A" * 64, 1))
    _accept_mutated_file(monkeypatch, copied, "evaluation/per_series.csv")

    with pytest.raises(ValueError, match="outcome SHA-256"):
        load_and_verify_bundle(copied)


def test_verified_bundle_payloads_are_recursively_immutable() -> None:
    bundle = load_and_verify_bundle(RESULT_ROOT)

    with pytest.raises(TypeError):
        bundle.lock["protocol"]["cities"]["Katowice"]["years"][0] = 0
    with pytest.raises(TypeError):
        bundle.per_series["mes"][0]["welfare"] = 0.0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "zero_welfare"])
def test_build_effects_rejects_invalid_mes_baselines(mutation: str) -> None:
    bundle = load_and_verify_bundle(RESULT_ROOT)
    rows = list(bundle.per_series["mes"])
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        duplicate = dict(rows[1])
        duplicate["series"] = rows[0]["series"]
        rows[1] = duplicate
    else:
        zero_welfare = dict(rows[0])
        zero_welfare["welfare"] = 0.0
        rows[0] = zero_welfare
    per_series = dict(bundle.per_series)
    per_series["mes"] = tuple(rows)
    malformed = replace(bundle, per_series=per_series)

    with pytest.raises(ValueError, match=r"MES (?:welfare )?baseline"):
        build_series_effects(malformed)
