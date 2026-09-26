"""Focused checks for the saved-results-only manuscript correction."""

import pytest

from iclr_contextual_lookup_correction import CONTEXTUAL, LOOKUP, pair_rows, summarize


def rows():
    return [
        dict(fit_id=fit, source_kind="fit", split="temporal_2022", view="test",
             scheme="age_sex", series=f"district-{i:02}",
             scored_years="2023|2024|2025", worst_csd=str(value),
             winner_trace_sha256="fixture")
        for fit, value in ((CONTEXTUAL, 0.2), (LOOKUP, 0.1))
        for i in range(18)
    ]


def test_pairing_is_by_sorted_identity_not_position():
    paired = pair_rows(list(reversed(rows())))
    assert len(paired) == 18
    assert paired[0]["series"] == "district-00"
    assert paired[0]["difference"] == pytest.approx(0.1)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "years", "nan", "wrong_model"])
def test_bad_inputs_fail_closed(mutation):
    data = rows()
    if mutation == "duplicate":
        data.append(dict(data[0]))
    elif mutation == "missing":
        data.pop()
    elif mutation == "years":
        data[0]["scored_years"] = "2024|2025"
    elif mutation == "nan":
        data[0]["worst_csd"] = "nan"
    else:
        data[0]["fit_id"] = "history_free/temporal_2022/seed-42"
    with pytest.raises(ValueError):
        pair_rows(data)


def test_direction_and_exact_test_on_known_vectors():
    s = summarize(pair_rows(rows()))
    assert s["difference"] == pytest.approx(0.1)
    assert (s["wins"], s["ties"], s["losses"]) == (0, 0, 18)
    assert s["p_two_sided_exact_sign_flip"] == 2 / 2**18
    assert s["ci_low"] == pytest.approx(0.1)
    assert s["ci_high"] == pytest.approx(0.1)


def test_generator_keeps_contextual_contrast_separate():
    from gen_iclr_numbers import _contextual_age_lookup_macros
    assert _contextual_age_lookup_macros() == {
        "ContextualVsAgeLookupDiff": "+0.0112",
        "ContextualVsAgeLookupDiffAbs": "0.0112",
        "ContextualVsAgeLookupCI": "[-0.0105, +0.0301]",
        "ContextualVsAgeLookupP": "= 0.311",
        "ContextualVsAgeLookupRecord": "5/1/12",
        "ContextualVsAgeLookupWins": "5/18",
        "ContextualVsAgeLookupDz": "+0.251",
    }


def test_generator_rejects_changed_correction_artifact(tmp_path, monkeypatch):
    import gen_iclr_numbers as generator
    folder = tmp_path / "analysis-output/contextual-age-lookup-20260920"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text('{"contrast":"history_free"}')
    monkeypatch.setattr(generator, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="artifact identity changed"):
        generator._contextual_age_lookup_macros()
