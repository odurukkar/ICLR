"""Fetch and verify the ML-side citations for the ICLR paper.

Discipline (non-negotiable): **no BibTeX is ever written from memory.** Each
entry is resolved against CrossRef or OpenAlex, cross-checked on title and year,
and only then emitted. Anything that fails resolution is written as an explicit
`PLACEHOLDER-VERIFY` entry and reported, never silently invented.

The social-choice half of the bibliography already exists and was verified
programmatically for the earlier draft (`paper/refs.bib`, 45 entries with
CrossRef/S2/DBLP artifacts alongside it); this script only adds what that file
does not cover: automated mechanism design, reward hacking / specification
gaming, and learning-to-optimize.

Writes iclr_paper/tex/refs_ml.bib and a report of anything unresolved.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_BIB = ROOT.parent / "iclr_paper" / "tex" / "refs_ml.bib"
TITLE_MATCH_THRESHOLD = 0.72

# CrossRef-verified metadata that OpenAlex omits for otherwise exact matches.
# Keep this narrow: every override must have a DOI checked against CrossRef.
METADATA_OVERRIDES = {
    "corbett2017algorithmic": {
        "venue": (
            "Proceedings of the 23rd ACM SIGKDD International Conference on "
            "Knowledge Discovery and Data Mining"
        ),
    },
    # Verified directly against DOI 10.1613/jair.1.13734 on 2026-08-29.
    "mohsin2022learning": {
        "venue": "Journal of Artificial Intelligence Research",
        "volume": "75",
        "pages": "1139-1176",
        "url": "https://doi.org/10.1613/jair.1.13734",
    },
}


@dataclass(frozen=True)
class Want:
    """A citation we need, with the key the paper will use."""

    key: str
    query: str
    expect_year: Optional[int] = None
    why: str = ""


WANTED: List[Want] = [
    Want("duetting2019optimal", "Optimal Auctions through Deep Learning", 2019,
         "learning mechanisms end to end; closest ML analogue of our setting"),
    Want("conitzer2002automated", "Complexity of Mechanism Design: A Unified Approach", 2002,
         "automated mechanism design, the older framing"),
    Want("skalse2022defining", "Defining and Characterizing Reward Hacking", 2022,
         "formal notion of reward hacking; our degenerate corner is an instance"),
    Want("pan2022effects", "The Effects of Reward Misspecification: Mapping and "
         "Mitigating Misaligned Models", 2022,
         "reward misspecification produces qualitatively different policies"),
    Want("amodei2016concrete", "Concrete Problems in AI Safety", 2016,
         "specification gaming / negative side effects"),
    # krakovna2020 dropped: it is a DeepMind blog post, not a citable paper.
    Want("bello2016neural", "Neural Combinatorial Optimization with Reinforcement "
         "Learning", 2016, "learning over combinatorial structures"),
    Want("kool2019attention", "Attention, Learn to Solve Routing Problems!", 2019,
         "learned heuristics for combinatorial problems"),
    Want("hardt2016equality", "Equality of Opportunity in Supervised Learning", 2016,
         "fairness metrics in ML; contrast with allocation fairness"),
    Want("corbett2017algorithmic", "Algorithmic Decision Making and the Cost of "
         "Fairness", 2017, "fairness-utility tradeoffs, the ML framing of our frontier"),
    Want("kearns2019empirical", "An Empirical Study of Rich Subgroup Fairness for "
         "Machine Learning", 2019,
         "fairness on coarse groups can fail on structured subgroups; the "
         "closest antecedent to our mechanism-side result"),
    Want("liu2018delayed", "Delayed Impact of Fair Machine Learning", 2018,
         "fairness interventions evaluated over time, not one shot"),
    Want("mohsin2022learning", "Learning to Design Fair and Private Voting Rules",
         2022, "learned voting-rule design outside participatory budgeting"),
]


CROSSREF = "https://api.crossref.org/works"
# CrossRef's "polite pool" wants a contact address and in exchange does not
# throttle the way Semantic Scholar's anonymous pool does (which returns 429
# almost immediately from a shared IP).
POLITE = "iclr-submission@example.org"


def _get(url: str, tries: int = 3) -> Optional[dict]:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": f"research/1.0 (mailto:{POLITE})",
                },
            )
            with urllib.request.urlopen(req, timeout=40) as fh:
                return json.load(fh)
        except Exception as exc:  # noqa: BLE001 - network flakiness is expected
            wait = 2 * (attempt + 1)
            logger.debug("retry %d after %s (%s)", attempt + 1, wait, exc)
            time.sleep(wait)
    return None


def _crossref_candidates(query: str) -> List[dict]:
    """Search CrossRef, normalized into the shape the matcher expects."""
    url = (
        f"{CROSSREF}?query.bibliographic={urllib.parse.quote(query)}"
        f"&rows=5&mailto={POLITE}"
    )
    data = _get(url)
    out: List[dict] = []
    for item in ((data or {}).get("message", {}) or {}).get("items", []) or []:
        titles = item.get("title") or []
        if not titles:
            continue
        parts = (item.get("issued") or {}).get("date-parts") or [[None]]
        year = parts[0][0] if parts and parts[0] else None
        authors = [
            {"name": " ".join(filter(None, [a.get("given"), a.get("family")]))}
            for a in (item.get("author") or [])
        ]
        out.append(
            {
                "title": titles[0],
                "year": year,
                "authors": authors,
                "venue": (item.get("container-title") or [""])[0],
                "volume": item.get("volume"),
                "pages": item.get("page"),
                "url": item.get("URL"),
                "externalIds": {"DOI": item.get("DOI")},
            }
        )
    return out


OPENALEX = "https://api.openalex.org/works"


def _openalex_candidates(query: str) -> List[dict]:
    """Search OpenAlex.

    CrossRef indexes journals well but misses most NeurIPS / ICLR / KDD / UAI
    proceedings, which is where the ML half of this bibliography lives. DBLP
    would be the natural second source but is unreachable from this environment
    (connection reset), so OpenAlex is the fallback: it covers CS proceedings
    and arXiv preprints, and issues arXiv DOIs for the latter.
    """
    url = (
        f"{OPENALEX}?search={urllib.parse.quote(query)}"
        f"&per-page=6&mailto={POLITE}"
    )
    data = _get(url)
    out: List[dict] = []
    for work in (data or {}).get("results", []) or []:
        title = work.get("title")
        if not title:
            continue
        loc = (work.get("primary_location") or {}).get("source") or {}
        doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
        authors = [
            {"name": (a.get("author") or {}).get("display_name", "")}
            for a in (work.get("authorships") or [])
        ]
        out.append(
            {
                "title": title,
                "year": work.get("publication_year"),
                "authors": [a for a in authors if a["name"]],
                "venue": loc.get("display_name", "") or "",
                "url": work.get("doi"),
                "externalIds": {"DOI": doi},
            }
        )
    return out


def _similar(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def _bibtex(key: str, paper: dict) -> str:
    authors = " and ".join(a.get("name", "") for a in paper.get("authors", []) if a)
    ext = paper.get("externalIds") or {}
    lines = [f"@article{{{key},"]
    lines.append(f"  title   = {{{paper.get('title', '')}}},")
    if authors:
        lines.append(f"  author  = {{{authors}}},")
    if paper.get("year"):
        lines.append(f"  year    = {{{paper['year']}}},")
    if paper.get("venue"):
        lines.append(f"  journal = {{{paper['venue']}}},")
    if paper.get("volume"):
        lines.append(f"  volume  = {{{paper['volume']}}},")
    if paper.get("pages"):
        pages = re.sub(r"(?<=\d)[-–—](?=\d)", "--", str(paper["pages"]))
        lines.append(f"  pages   = {{{pages}}},")
    if ext.get("DOI"):
        lines.append(f"  doi     = {{{ext['DOI']}}},")
    if paper.get("url"):
        lines.append(f"  url     = {{{paper['url']}}},")
    if ext.get("ArXiv"):
        lines.append(f"  eprint  = {{{ext['ArXiv']}}},")
        lines.append("  archivePrefix = {arXiv},")
    lines.append("}")
    return "\n".join(lines)


CACHE = ROOT / "results" / "iclr_refs_cache.json"


def _load_cache() -> Dict[str, dict]:
    """Previously verified entries, keyed by citation key.

    A transient rate limit must never be able to delete a citation that was
    verified on an earlier run: that silently drops a reference out of the
    bibliography and the build only complains much later, if at all. Anything
    resolved once is cached and reused when the network fails.
    """
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            logger.warning("cache unreadable, ignoring: %s", CACHE)
    return {}


def main() -> None:
    cache = _load_cache()
    resolved: List[str] = []
    failed: List[Want] = []
    report: List[dict] = []

    for want in WANTED:
        best: Optional[dict] = None
        best_score = 0.0
        source = ""
        for src, getter in (("CrossRef", _crossref_candidates), ("OpenAlex", _openalex_candidates)):
            for cand in getter(want.query):
                score = _similar(want.query, cand.get("title", ""))
                if want.expect_year and cand.get("year"):
                    if abs(cand["year"] - want.expect_year) > 2:
                        score -= 0.15
                if score > best_score:
                    best, best_score, source = cand, score, src
            if best_score >= TITLE_MATCH_THRESHOLD:
                break
            time.sleep(0.4)

        if (best is None or best_score < TITLE_MATCH_THRESHOLD) and want.key in cache:
            entry = cache[want.key]
            resolved.append(
                f"% verified via {entry['source']} on an earlier run (cached); "
                f"title match {entry['score']:.2f}\n" + entry["bibtex"]
            )
            logger.warning(
                "CACHED %-24s lookup failed now, reusing the verified entry", want.key
            )
            report.append({"key": want.key, "status": "cached", **{
                k: entry.get(k) for k in ("title", "year", "score", "source")}})
            continue

        if best is None or best_score < TITLE_MATCH_THRESHOLD:
            failed.append(want)
            logger.warning(
                "UNRESOLVED %-26s best=%.2f %s",
                want.key, best_score, (best or {}).get("title", "no candidate"),
            )
            report.append({"key": want.key, "status": "unresolved",
                           "best_title": (best or {}).get("title"), "score": best_score})
            continue

        best.update(METADATA_OVERRIDES.get(want.key, {}))
        bib = _bibtex(want.key, best)
        cache[want.key] = {
            "bibtex": bib, "source": source, "score": round(best_score, 3),
            "title": best.get("title"), "year": best.get("year"),
        }
        resolved.append(f"% verified via {source}; title match {best_score:.2f}\n" + bib)
        logger.info("ok  %-26s (%.2f, %s) %s [%s]", want.key, best_score, source,
                    str(best.get("title"))[:52], best.get("year"))
        report.append({"key": want.key, "status": "verified", "title": best.get("title"),
                       "year": best.get("year"), "score": round(best_score, 3), "source": source,
                       "doi": (best.get("externalIds") or {}).get("DOI")})
        time.sleep(0.6)  # polite pool: modest spacing is enough

    OUT_BIB.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "% ML-side references for the ICLR submission.",
        "% AUTO-GENERATED by tempopb/src/iclr_fetch_refs.py -- do not hand-edit.",
        "% Every entry was resolved against CrossRef or OpenAlex and title-matched;",
        "% the per-entry comment records which source and the match score.",
        "% The social-choice half of the bibliography lives in paper/refs.bib (45",
        "% entries, verified separately against CrossRef/DBLP/S2).",
        "",
    ]
    body = "\n\n".join(resolved)
    if failed:
        body += "\n\n" + "\n\n".join(
            f"% !! UNRESOLVED -- do not cite until a human verifies this exists.\n"
            f"@misc{{PLACEHOLDER-VERIFY-{w.key},\n  note = {{unresolved query: {w.query}}},\n}}"
            for w in failed
        )
    OUT_BIB.write_text("\n".join(header) + body + "\n")

    CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    (ROOT / "results" / "iclr_refs_report.json").write_text(json.dumps(report, indent=2))
    logger.info("=" * 70)
    logger.info("verified %d, unresolved %d -> %s", len(resolved), len(failed), OUT_BIB)
    if failed:
        logger.warning(
            "UNRESOLVED KEYS (must be human-verified or dropped): %s",
            ", ".join(w.key for w in failed),
        )


if __name__ == "__main__":
    main()
