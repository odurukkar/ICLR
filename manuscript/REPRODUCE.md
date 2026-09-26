# September 25 submission source, revision 3

This directory is the reconciled submission copy. The older `iclr_paper/tex/`
sources and historical release archives are preserved, not overwritten.

Revision 3 adds three wording clarifications: the introduction's payment
guarantee applies during the payment phase, the external corpus sentence counts
scored elections separately from series, and the conclusion/limitations unchanged count
refers to native-approval series. All results and figure assets are unchanged.

Revision 2 corrects the RES/MES partition comparison in prose, removes an
unsupported explanation of the native-transfer null, limits completion-phase
attribution to uniform Equal Shares, and states the implemented uniform fallback
when all raw learned endowments are zero. The replay comparison also makes clear
that its contrast includes each rule's induced history, not balance depletion
alone. Numerical results, tables, figures,
citations, and scientific input artifacts are unchanged.

## Compile the manuscript

The source archive contains the `tex/` directory and this guide. With a standard
TeX Live installation:

```bash
cd tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The official ICLR 2027 style is unchanged. The bibliography and main figure
assets are the previously checked September 23 versions.

## Numerical evidence

In the research repository, the append-only reconstruction is
`tempopb/src/iclr_manuscript_reconstruct_v2.py`. Its saved outputs are under
`tempopb/results/iclr_primary_warsaw_v2/manuscript_reconstruction/`:

- `macro_records.json`: all 596 macro classifications and per-value provenance;
- `numbers_candidate.tex`: 592 definitions, omitting four unsupported runtimes;
- `coverage.md`: exact changes versus the earlier working manuscript;
- `manifest.json`: output digests.

The corrected primary evidence is in
`results/iclr_primary_warsaw_v2/evaluation/paper_consumers.json`; the later
contextual-versus-age contrast is in
`analysis-output/contextual-age-lookup-20260920/summary.json`. The reconstruction
authenticates 49 fits, 25 analysis components, and the protected original
release, then formats saved values. It does not rerun fitting, allocation,
bootstrap resampling, or hypothesis tests. It is a post-hoc reconciliation,
not a new holdout experiment.

The 213 external/non-primary macro values were independently checked against
their separate recorded lineages. The 17 synthetic-example macros retain the
archived construction. Corrected-run elapsed times are unavailable; no runtime
range is reported for the corrected fits. Nine primary literal tables were
checked against saved records, with three cell corrections in the demographic
partition table. Two saved age-control confidence intervals were also corrected.

All included numerical figures were checked against their respective saved
evidence at source precision. The original appendix trajectory and fitted-map
plots were omitted because saved corrected outputs do not contain the required
cohort ledgers or endowment arrays. Their original files remain unchanged.

## Scope of the source archive

This is a compilable manuscript source package, not a full experimental-code
or voter-data archive. The historical reproducibility ZIP predates this
reconciliation and must not be represented as the matching corrected-results
supplement. No experiments or external submissions were performed in this
finalization pass.
