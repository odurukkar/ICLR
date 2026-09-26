# Reproduction boundaries

## Repository integrity

`python3 scripts/check_release.py` checks this export against
`RELEASE_MANIFEST.json`. It detects missing or modified listed files, checks
Python syntax, and reports unlisted files without treating them as released
content. The manifest is an integrity inventory, not a digital signature.

## Exact saved-results authentication

The separate original artifact is:

`ICLR2027_Coverage_Is_Not_Fairness_REPRODUCIBILITY_2026-09-25_r3.zip`

SHA-256:
`20e9a4ceea8cfa72d6aa84ff2b5785e15262909645f84bb9b5ff60a25be7a2a1`

Obtain it from the paper's authors/submission materials and extract it to a
separate empty folder. From that extraction:

```sh
cd tempopb
uv sync --locked
PYTHONDONTWRITEBYTECODE=1 uv run --locked python src/verify_iclr_saved_supplement.py
```

Expected output: `status: pass`, 339 included files checked, 49 primary fits,
24 multi-city fits, 596 macro classifications and 592 final macro definitions.
This authenticates saved evidence without rerunning experiments. Raw ballot
bytes are not rechecked. Setup may download dependencies; the verifier itself
does not need network access.

Do not run that strict verifier against this cleaned repository: its historical
archives and original package manifest are deliberately absent. Original
verification and release-builder source files are preserved for inspection,
but builders need inputs from the full research workspace.

## Research and figure generators

The three self-contained main-figure scripts reproduce the saved plotted
values without allocation or statistical reruns. Their figure data and code
behavior are unchanged; only internal style-review comments were reworded.

The remaining `tempopb/src/` files include historical and exploratory analyses.
Their presence is not authorization to overwrite locked artifacts. Training,
evaluation and older number/figure generators may need raw data and their
original directory layout, and may write analysis outputs. Use a separate
working copy and inspect the relevant protocol before invoking them.

For the final paper, use the included `manuscript/tex/numbers.tex` and assets;
`iclr_paper/tex/numbers.tex` is a historical authentication input, not the final
manuscript. Build the manuscript in a working copy to preserve this inventory.
