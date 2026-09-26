"""The native-approval cross-city study must stay visible in the manuscript.

A null result is the easiest thing in a paper to lose during a rewrite: nothing
downstream breaks when it disappears. These checks fail if the study stops
being reported, if its numbers stop coming from the frozen manifest, or if the
corpus rules that keep one electorate from being counted twice are relaxed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "iclr_paper" / "tex"
MAIN_TEX = PAPER / "main.tex"
NUMBERS_TEX = PAPER / "numbers.tex"
APPENDIX_TEX = PAPER / "appendix.tex"
APPENDIX_TABLES_TEX = PAPER / "appendix_tables.tex"
MANIFEST = ROOT / "results" / "iclr_crosscity_manifest.json"
WINNER_SIGNATURES = ROOT / "results" / "iclr_crosscity_winner_signatures.json"

REQUIRED_MAIN_MACROS = (
    "CrossCitySeriesN",
    "CrossCityAllDiff",
    "CrossCityAllCILow",
    "CrossCityAllCIHigh",
    "CrossCityAllP",
    "CrossCityInertN",
)


@pytest.mark.skipif(not MANIFEST.is_file(), reason="cross-city study not run")
def test_main_text_reports_the_null_and_its_uncertainty() -> None:
    """A bare point estimate is not enough: the interval must travel with it."""
    text = MAIN_TEX.read_text()
    for macro in REQUIRED_MAIN_MACROS:
        assert f"\\{macro}" in text, f"main text no longer cites \\{macro}"


@pytest.mark.skipif(not MANIFEST.is_file(), reason="cross-city study not run")
def test_reported_numbers_match_the_frozen_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert WINNER_SIGNATURES.is_file(), "exact winner-signature artifact is missing"
    signatures = json.loads(WINNER_SIGNATURES.read_text())
    row = manifest["contrasts"]["ALL|learned-endowment|mes"]
    numbers = NUMBERS_TEX.read_text()
    expected = {
        "CrossCitySeriesN": str(manifest["corpus"]["n_series"]),
        "CrossCityInertN": str(
            signatures["summary"]["n_exactly_unchanged_on_scored_elections"]
        ),
        "CrossCityAllDiff": f"{row['difference']:+.4f}",
        "CrossCityAllWins": f"{row['wins']}/{row['n_series']}",
    }
    for macro, value in expected.items():
        assert f"\\newcommand{{\\{macro}}}{{{value}}}" in numbers


@pytest.mark.skipif(not MANIFEST.is_file(), reason="cross-city study not run")
def test_winner_signature_artifact_is_current_and_explicitly_post_hoc() -> None:
    assert WINNER_SIGNATURES.is_file(), "exact winner-signature artifact is missing"
    payload = json.loads(WINNER_SIGNATURES.read_text())
    fit_path = ROOT / "results" / "iclr_train" / "run_main_seed42.json"

    assert payload["artifact_kind"] == "post-hoc-verification"
    assert payload["protocol_lock"] is False
    assert payload["sources"]["source_manifest"]["sha256"] == hashlib.sha256(
        MANIFEST.read_bytes()
    ).hexdigest()
    assert payload["sources"]["frozen_warsaw_fit"]["sha256"] == hashlib.sha256(
        fit_path.read_bytes()
    ).hexdigest()
    assert payload["summary"]["n_exactly_unchanged_on_scored_elections"] == 36
    assert payload["summary"]["n_changed_on_scored_elections"] == 22


@pytest.mark.skipif(not MANIFEST.is_file(), reason="cross-city study not run")
def test_appendix_documents_the_protocol_and_the_table() -> None:
    appendix = APPENDIX_TEX.read_text()
    assert "\\label{app:external}" in appendix
    assert "tab:crosscity" in APPENDIX_TABLES_TEX.read_text()
    external = appendix[appendix.index("\\label{app:external}") :]
    assert "larger-budget track" in external
    normalized = " ".join(external.split())
    assert re.search(
        r"replay the frozen Warsaw fit (?:with no|without) refitting "
        r"(?:and no|or) policy selection",
        normalized,
    ), "external protocol no longer rules out refitting and policy selection"


@pytest.mark.skipif(not MANIFEST.is_file(), reason="cross-city study not run")
def test_the_null_is_not_described_as_a_success() -> None:
    """Guard the wording the result cannot support."""
    text = MAIN_TEX.read_text().lower()
    for phrase in (
        "transfers to gdynia",
        "external validation",
        "confirms externally",
        "generalizes across cities",
    ):
        assert phrase not in text, f"unsupported transfer wording: {phrase}"


@pytest.mark.skipif(not MANIFEST.is_file(), reason="cross-city study not run")
def test_corpus_excludes_double_counted_electorates() -> None:
    """No citywide aggregate, and at most one Gdynia track per neighbourhood."""
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["protocol"]["refitting"] is False
    assert manifest["protocol"]["scored_years"] == ">= 2023"

    frozen = ROOT / "results" / "iclr_crosscity_instances.txt"
    keys = {line.split("\t")[1] for line in frozen.read_text().splitlines() if line}
    assert keys, "frozen cross-city instance list is empty"
    assert not [k for k in keys if k.endswith("CITYWIDE")]

    neighbourhoods = [k.rsplit(" | ", 1)[0] for k in keys]
    assert len(neighbourhoods) == len(set(neighbourhoods)), (
        "a neighbourhood contributes more than one ballot track"
    )
    assert len(keys) == manifest["corpus"]["n_series"]
