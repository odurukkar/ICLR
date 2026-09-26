"""Regression tests for the generated ML bibliography."""

from __future__ import annotations

import json
from pathlib import Path

import iclr_fetch_refs as refs


MOHSIN_BIBTEX = """@article{mohsin2022learning,
  title   = {Learning to Design Fair and Private Voting Rules},
  author  = {Farhad Mohsin and Ao Liu and Pin-Yu Chen and Francesca Rossi and Lirong Xia},
  year    = {2022},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {75},
  pages   = {1139--1176},
  doi     = {10.1613/jair.1.13734},
  url     = {https://doi.org/10.1613/jair.1.13734},
}"""


def test_crossref_bibtex_preserves_journal_provenance(monkeypatch) -> None:
    payload = {
        "message": {
            "items": [
                {
                    "title": ["Learning to Design Fair and Private Voting Rules"],
                    "issued": {"date-parts": [[2022, 11, 30]]},
                    "author": [
                        {"given": "Farhad", "family": "Mohsin"},
                        {"given": "Ao", "family": "Liu"},
                    ],
                    "container-title": ["Journal of Artificial Intelligence Research"],
                    "volume": "75",
                    "page": "1139-1176",
                    "DOI": "10.1613/jair.1.13734",
                    "URL": "https://doi.org/10.1613/jair.1.13734",
                }
            ]
        }
    }
    monkeypatch.setattr(refs, "_get", lambda _url: payload)

    candidate = refs._crossref_candidates("Learning to Design Fair and Private Voting Rules")[0]
    bibtex = refs._bibtex("mohsin2022learning", candidate)

    for field in (
        "volume  = {75}",
        "pages   = {1139--1176}",
        "doi     = {10.1613/jair.1.13734}",
        "url     = {https://doi.org/10.1613/jair.1.13734}",
    ):
        assert field in bibtex


def test_cached_mohsin_record_survives_network_failure(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    cache_path = results / "iclr_refs_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "mohsin2022learning": {
                    "bibtex": MOHSIN_BIBTEX,
                    "source": "CrossRef DOI record",
                    "score": 1.0,
                    "title": "Learning to Design Fair and Private Voting Rules",
                    "year": 2022,
                }
            }
        )
    )
    output = tmp_path / "refs_ml.bib"
    monkeypatch.setattr(refs, "ROOT", tmp_path)
    monkeypatch.setattr(refs, "CACHE", cache_path)
    monkeypatch.setattr(refs, "OUT_BIB", output)
    monkeypatch.setattr(
        refs,
        "WANTED",
        [
            refs.Want(
                "mohsin2022learning",
                "Learning to Design Fair and Private Voting Rules",
                2022,
            )
        ],
    )
    monkeypatch.setattr(refs, "_crossref_candidates", lambda _query: [])
    monkeypatch.setattr(refs, "_openalex_candidates", lambda _query: [])
    monkeypatch.setattr(refs.time, "sleep", lambda _seconds: None)

    refs.main()

    generated = output.read_text()
    assert MOHSIN_BIBTEX in generated
    assert "PLACEHOLDER-VERIFY-mohsin2022learning" not in generated


def test_checked_in_cache_uses_the_canonical_mohsin_key() -> None:
    cache = json.loads(refs.CACHE.read_text())
    assert "mohsin2022learning" in cache
    assert "dandekar2024learning" not in cache
    assert cache["mohsin2022learning"]["bibtex"] == MOHSIN_BIBTEX


def test_mohsin_key_is_consistent_across_paper_and_generator() -> None:
    repo = Path(__file__).resolve().parents[2]
    checked = (
        repo / "iclr_paper" / "tex" / "appendix.tex",
        repo / "iclr_paper" / "tex" / "refs.bib",
        repo / "iclr_paper" / "tex" / "refs_ml.bib",
    )
    for path in checked:
        text = path.read_text()
        assert "mohsin2022learning" in text
        assert "dandekar2024learning" not in text

    wanted = {want.key for want in refs.WANTED}
    assert "mohsin2022learning" in wanted
    assert "dandekar2024learning" not in wanted
    assert refs.METADATA_OVERRIDES["mohsin2022learning"] == {
        "venue": "Journal of Artificial Intelligence Research",
        "volume": "75",
        "pages": "1139-1176",
        "url": "https://doi.org/10.1613/jair.1.13734",
    }
