"""Claim-ledger regressions for the ICLR main paper.

These tests protect the evidence boundary approved for Task 6: Warsaw results
are in-domain evidence, fresh-city changes are directional observations, and
the preregistered external decision is a falsification.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT.parent / "iclr_paper" / "tex" / "main.tex"
APPENDIX_TEX = MAIN_TEX.parent / "appendix.tex"
REPRODUCE_MD = MAIN_TEX.parent.parent / "REPRODUCE.md"
if not REPRODUCE_MD.is_file():
    REPRODUCE_MD = ROOT / "REPRODUCE.md"
FROZEN_INSTANCE_LISTS = (
    ROOT / "results" / "frozen_instances.txt",
    ROOT / "results" / "iclr_multicity_instances.txt",
    ROOT / "results" / "iclr_crosscity_instances.txt",
)
CROSSCITY_SOURCE = ROOT / "src" / "iclr_crosscity.py"

APPROVED_CITATION_KEYS = {
    "aziz2015jr",
    "baychkov2026mixed",
    "elkind2023temporal",
    "fairstein2024learningpb",
    "lackner2020perpetual",
    "maly2023equality",
    "maruo2026mechanism",
    "peters2020welfarism",
    "skalse2022defining",
    "stolicki2020pabulib",
    "thach2026llmrule",
}


def _environment(tex: str, name: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(?P<body>.*?)\\end\{{{re.escape(name)}\}}",
        tex,
        re.DOTALL,
    )
    assert match is not None, f"missing {name} environment"
    return match.group("body").strip()


def _section(tex: str, title: str) -> str:
    match = re.search(
        rf"\\section\{{{re.escape(title)}\}}(?P<body>.*?)(?=\\section(?:\*?)\{{)",
        tex,
        re.DOTALL,
    )
    assert match is not None, f"missing section {title}"
    return match.group("body").strip()


def _section_by_label(tex: str, label: str) -> str:
    match = re.search(
        rf"\\label\{{{re.escape(label)}\}}(?P<body>.*?)(?=\\section(?:\*?)\{{)",
        tex,
        re.DOTALL,
    )
    assert match is not None, f"missing section label {label}"
    return match.group("body").strip()


def _paragraph(scope: str, title: str) -> str:
    match = re.search(
        rf"\\paragraph\{{{re.escape(title)}\}}(?P<body>.*?)(?="
        r"\\(?:paragraph|(?:sub)*section)\*?\{|\\end\{document\}|\Z)",
        scope,
        re.DOTALL,
    )
    assert match is not None, f"missing paragraph {title}"
    return match.group("body").strip()


def _paragraph_units(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def _sentence_units(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return re.split(r"(?<=[.!?])\s+(?=(?:[A-Z]|\\))", normalized)


def _figure_with_asset(tex: str, asset: str) -> str:
    for figure in re.findall(
        r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.DOTALL
    ):
        if "{" + asset + "}" in figure:
            return figure
    raise AssertionError(f"missing figure for {asset}")


def _environment_with_label(tex: str, name: str, label: str) -> str:
    for environment in re.findall(
        rf"\\begin\{{{re.escape(name)}\}}.*?\\end\{{{re.escape(name)}\}}",
        tex,
        re.DOTALL,
    ):
        if rf"\label{{{label}}}" in environment:
            return environment
    raise AssertionError(f"missing {name} environment with label {label}")


def _listed_filenames(path: Path) -> set[str]:
    return {
        line.split("\t", 1)[0]
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    }


def test_main_paper_names_the_falsification_and_projection() -> None:
    tex = MAIN_TEX.read_text()
    required = (
        "projected support sets",
        r"\ExternalPrimaryWins",
        r"\ExternalPrimaryTies",
        r"\ExternalPrimaryLosses",
        "directionally",
        "seed-sensitive",
    )
    for phrase in required:
        assert phrase in tex


def test_native_replay_reproduction_stages_the_unbundled_voter_files() -> None:
    """A fresh archive must stage native ballots before cross-city replay."""
    guide = REPRODUCE_MD.read_text()
    replay_block = re.search(
        r"Rebuild the post-hoc native replay.*?```bash(?P<body>.*?)```",
        guide,
        re.DOTALL,
    )
    assert replay_block is not None
    commands = replay_block.group("body")
    assert re.search(
        r"python\s+src/iclr_crosscity\.py\s+--stage-from\s+data/pb(?:\s|$)",
        commands,
    )
    assert commands.index("iclr_crosscity.py") < commands.index(
        "iclr_crosscity_signatures.py"
    )


def test_age_lookup_reproduction_builds_its_history_free_prerequisite() -> None:
    guide = REPRODUCE_MD.read_text()
    audit_block = re.search(
        r"The age lookup is a separate post-lock audit.*?```bash(?P<body>.*?)```",
        guide,
        re.DOTALL,
    )
    assert audit_block is not None
    commands = audit_block.group("body")
    assert "src/iclr_history_free_audit.py" in commands
    assert commands.index("iclr_history_free_audit.py") < commands.index(
        "iclr_static_demographic_audit.py"
    )
    assert "history_free_seed42_g30.json" in guide


def test_reproduction_commands_enforce_the_committed_lockfile() -> None:
    guide = REPRODUCE_MD.read_text()
    normalized = " ".join(guide.split())
    assert "uv sync --locked" in guide
    assert "PYTHONPATH=src uv run pytest -q tests/test_iclr_*.py" in guide
    assert re.search(r"(?m)^uv sync$", guide) is None
    assert "uv run --with pytest" not in guide
    assert "post-lock plotting-only dependency amendment" in normalized
    assert "no pre-existing package or version changed" in normalized
    assert "iclr_lock_amendments.json" in guide


def test_public_manifest_count_matches_all_three_frozen_corpus_scopes() -> None:
    """The public provenance count must equal the builder's deduplicated union."""
    scopes = [_listed_filenames(path) for path in FROZEN_INSTANCE_LISTS]
    union_count = len(set().union(*scopes))
    main = MAIN_TEX.read_text()
    guide = REPRODUCE_MD.read_text()
    assert f"A {union_count}-file integrity manifest" in main
    assert f"all {union_count} unique files" in guide
    assert f"All {union_count} files" in guide
    for scope_count, scope_name in zip(
        map(len, scopes),
        ("primary", "multi-city", "native cross-city"),
        strict=True,
    ):
        assert f"{scope_count}-file {scope_name}" in guide


def test_both_frontier_captions_name_the_exact_four_rule_band() -> None:
    """The visual band excludes other tested hand-designed policies."""
    main_table = _environment_with_label(MAIN_TEX.read_text(), "table", "tab:main")
    appendix_figure = _figure_with_asset(APPENDIX_TEX.read_text(), "fig_frontier.pdf")
    for caption_scope in (main_table, appendix_figure):
        normalized = " ".join(caption_scope.split())
        for marker in (
            "four-rule",
            "greedy approvals/cost",
            r"\MES{}",
            r"\RES{}",
            "0.5",
            "1.0",
        ):
            assert marker in normalized
        assert "every tested hand-designed rule" not in normalized


def test_abstract_preserves_the_evidence_map_without_a_sentence_template() -> None:
    abstract = _environment(MAIN_TEX.read_text(), "abstract")
    required = (
        r"\EvalSeriesN",
        r"\ExternalSeriesN",
        r"\ExternalDecision",
        "Warsaw",
        "in-domain",
        "projected support sets",
        "exclusion",
    )
    for item in required:
        assert item in abstract
    assert "mechanically classified" not in abstract
    assert "external confirmation" not in abstract.lower()
    units = _paragraph_units(abstract)
    assert any(
        "Warsaw" in unit and "in-domain" in unit
        for unit in units
    )
    assert any(
        r"\ExternalDecision" in unit
        and any(
            phrase in unit.lower()
            for phrase in (
                "does not confirm external transfer",
                "do not confirm external transfer",
            )
        )
        for unit in units
    )


def test_introduction_ends_with_exactly_three_contributions() -> None:
    introduction = _section(MAIN_TEX.read_text(), "Introduction")
    contributions = introduction[introduction.index(r"\paragraph{Contributions.}") :]
    itemize = _environment(contributions, "itemize")
    assert contributions.rstrip().endswith(r"\end{itemize}")
    assert itemize.count(r"\item") == 3
    assert r"\textbf{" not in itemize


def test_main_paper_preserves_formal_and_controlled_evidence() -> None:
    tex = MAIN_TEX.read_text()
    for required in (
        r"\begin{definition}[Attribution and deficit]",
        r"\label{sec:containment}",
        r"\begin{proposition}[Representative-support degeneracy]",
        r"\begin{proposition}[Support floor]",
        "frozen-score intervention",
        r"\label{sec:identification}",
    ):
        assert required in tex


def test_identification_limits_scope_nonidentifiability_and_strategic_response() -> None:
    limits = _paragraph(
        _section(MAIN_TEX.read_text(), "Limitations"),
        "Identification and deployment limits.",
    )
    sentences = _sentence_units(limits)
    assert any(
        "weight" in sentence.lower()
        and "not identifiable" in sentence.lower()
        and "coefficient" in sentence.lower()
        for sentence in sentences
    )
    assert any(
        "strategic response" in sentence.lower()
        and any(
            boundary in sentence.lower()
            for boundary in (
                "cannot measure",
                "cannot infer",
                "does not measure",
                "does not infer",
            )
        )
        for sentence in sentences
    )


def test_frozen_score_scope_keeps_inputs_fixed_but_allows_trajectory_divergence() -> None:
    schematic = _figure_with_asset(MAIN_TEX.read_text(), "fig_schematic.pdf")
    sentences = _sentence_units(schematic)
    fixed_or_equivalent = ("fixed", "fixes", "same", "unchanged")
    for phrase in ("score weights", "feature definitions"):
        assert any(
            phrase in sentence.lower()
            and any(marker in sentence.lower() for marker in fixed_or_equivalent)
            for sentence in sentences
        )
    assert any(
        "realized score trajectories" in sentence.lower()
        and any(
            divergence in sentence.lower()
            for divergence in (
                "may diverge",
                "can diverge",
                "could diverge",
                "allowed to diverge",
            )
        )
        for sentence in sentences
    )


def test_native_approval_null_evidence_keeps_its_interval_uncertainty() -> None:
    native_approval = _paragraph(
        _section_by_label(MAIN_TEX.read_text(), "sec:sequential-transfer"),
        "Native approval ballots, without projection.",
    )
    for item in (
        r"\CrossCityAllDiff",
        r"\CrossCityAllCILow",
        r"\CrossCityAllCIHigh",
        r"\CrossCityAllP",
        "neither city differs detectably from zero",
    ):
        assert item in native_approval


def test_native_approval_replay_is_not_retroactively_described_as_locked() -> None:
    """Only priority and projected-support studies had pre-opening locks."""
    main = MAIN_TEX.read_text()
    appendix = APPENDIX_TEX.read_text()
    native_approval = _paragraph(
        _section_by_label(main, "sec:sequential-transfer"),
        "Native approval ballots, without projection.",
    )
    assert "post-hoc" in native_approval
    assert "no-refit" in native_approval
    assert "two locked studies" in main.lower()
    assert "two locked studies" in appendix.lower()
    for overstatement in ("each external study", "three locked external"):
        assert overstatement not in main.lower()
        assert overstatement not in appendix.lower()


def test_native_replay_separates_unseen_inputs_from_overlapping_warmup_years() -> None:
    native_appendix = _paragraph(
        APPENDIX_TEX.read_text(),
        "Native approval ballots.",
    )
    scopes = (native_appendix, CROSSCITY_SOURCE.read_text())
    for scope in scopes:
        normalized = " ".join(scope.split())
        assert "none of these ballots, projects, or cities" in normalized
        assert "every scored edition postdates fitting" in normalized
        assert "none of these ballots, projects, cities or years" not in normalized
        assert "none of these ballots, projects, cities, or years" not in normalized


def test_external_transfer_section_reports_the_complete_gate_anatomy() -> None:
    section = _section(MAIN_TEX.read_text(), "External transfer tests")
    for phrase in (
        "learned project priority",
        "preregistered",
        "falsified",
        "projected support sets",
        r"\ExternalElectionN",
        r"\ExternalSeriesN",
        r"\ExternalScoreN",
        "three fixed seeds",
        "equal-city aggregation",
        r"\ExternalSeedOneMacro",
        r"\ExternalSeedTwoMacro",
        r"\ExternalSeedFortyTwoMacro",
        r"\ExternalPrimaryWins",
        r"\ExternalPrimaryTies",
        r"\ExternalPrimaryLosses",
        "city-level uncertainty",
        "Holm",
        "exclusion",
        "seed spread",
        "direction",
        "macro-effect",
        "welfare",
        "actuation",
    ):
        assert phrase in section


def test_external_caption_is_self_contained_and_directional() -> None:
    figure = _figure_with_asset(MAIN_TEX.read_text(), "fig_external.pdf")
    for phrase in (
        r"n=\ExternalSeriesN",
        "paired bootstrap",
        r"\ExternalKatowicePrimaryHolmP",
        r"\ExternalKrakowPrimaryHolmP",
        "negative CSD change favors",
        "positive exclusion change is adverse",
        "projected support sets",
    ):
        assert phrase in figure


def test_main_paper_has_only_three_scientific_figure_assets() -> None:
    """Main text carries the three figures the argument needs, in order.

    The argument runs setting, headline result, external outcome. The headline
    is the frozen-score contrast between coverage and the fairness metric, so
    that figure sits in the main text and the fitted-class frontier moved to the
    appendix, where its numbers already live in the frontier table. The appendix
    assertions keep both relocated figures from silently disappearing.
    """
    tex = MAIN_TEX.read_text()
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{(fig_[^}]+)\}", tex)
    assert figures == [
        "fig_schematic.pdf",
        "fig_coverage.pdf",
        "fig_external.pdf",
    ]
    appendix = (MAIN_TEX.parent / "appendix.tex").read_text()
    for asset, reference in (
        ("fig_hack.pdf", r"\ref{fig:hack}"),
        ("fig_frontier.pdf", r"\ref{fig:frontier}"),
    ):
        assert asset in appendix
        assert reference in tex


def test_rebuild_guide_regenerates_every_main_text_figure() -> None:
    """Paper rebuild instructions must regenerate, not merely reuse, main figures."""
    main = MAIN_TEX.read_text()
    guide = REPRODUCE_MD.read_text()
    assets = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{(fig_[^}]+)\}", main)
    for asset in assets:
        stem = Path(asset).stem
        assert f"src/iclr_{stem}.py" in guide


def test_reproducibility_statement_scopes_the_three_verification_scripts() -> None:
    tex = MAIN_TEX.read_text()
    assert "reference replay and the endowment/direct-score containments" in tex
    assert "the hand-designed rule contained in each policy class" not in tex


def test_limitations_and_conclusion_keep_the_evidence_boundaries() -> None:
    tex = MAIN_TEX.read_text()
    limitations = _section(tex, "Limitations")
    for phrase in (
        "support-set projection",
        "two fresh cities",
        r"\ExternalSeriesN",
        "sparse actuation",
        "seed sensitivity",
        "city-level uncertainty",
        "exclusion increases",
        "observational fitted-class comparison",
    ):
        assert phrase in limitations

    conclusion = " ".join(_section(tex, "Conclusion").split())
    for phrase in (
        "Warsaw",
            "sparse",
            "allocation",
            "city-level uncertainty",
            "post-hoc native replay shows no detectable difference",
        ):
        assert phrase in conclusion


def test_main_paper_avoids_forbidden_external_claims() -> None:
    tex = MAIN_TEX.read_text().lower()
    forbidden = (
        "robust learned-endowment improvement",
        "external confirmation succeeds",
        "significant in both cities",
        "safeguard-preserving transfer",
    )
    for phrase in forbidden:
        assert phrase not in tex


def test_related_work_separates_mechanism_learning_from_reward_hacking() -> None:
    section = " ".join(_section(MAIN_TEX.read_text(), "Related work").split())
    assert r"mechanism-informed learning \citep{maruo2026mechanism}" in section
    assert r"tests of reward hacking \citep{skalse2022defining}" in section
    assert "pre-selection payment control" in section
    assert "project-score parameters" in section
    assert "allocation kernel" in section
    assert "no priority for learned PB rules or nonuniform" in section
    assert r"\citep{maruo2026mechanism,skalse2022defining}" not in section


def test_main_paper_uses_only_the_preverified_citation_keys() -> None:
    tex = MAIN_TEX.read_text()
    observed = {
        key.strip()
        for citation in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex)
        for key in citation.split(",")
    }
    assert observed == APPROVED_CITATION_KEYS
