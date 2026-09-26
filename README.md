# Coverage Is Not Fairness

Research materials for **Coverage Is Not Fairness: Auditing Learned Controls in
Participatory Budgeting**, September 25, 2026 manuscript revision 3.

The paper studies the difference between group spending metrics and voter
coverage, and audits learned controls in participatory budgeting. Negative and
null results, post-hoc corrections and failed external validation decisions
are included; this repository does not claim a successful generalization result.

## Contents

- `manuscript/paper.pdf`: current anonymous paper (29 pages, nine main-text pages).
- `manuscript/tex/`: matching LaTeX, bibliography and five figure PDFs.
- `tempopb/src/`: allocation rules, policy learning, analysis and verification code.
- `tempopb/tests/`: scientific unit and regression tests.
- `tempopb/results/` and `tempopb/analysis-output/`: saved evidence from the r3 release.
- `figures/{schematic,coverage,external}/`: self-contained final main-figure
  scripts, PDF/PNG exports and plotted values.
- `docs/`: data, provenance, reproduction and attribution information.
- `RELEASE_MANIFEST.json`: file sizes, SHA-256 digests and export provenance.

## Quick checks

From the repository root, check the exported file hashes without dependencies:

```sh
python3 scripts/check_release.py
```

For a small data-free regression suite, use Python 3.12 and the locked environment:

```sh
cd tempopb
uv sync --locked
PYTHONPATH=src uv run --locked pytest -q -p no:cacheprovider \
  tests/test_iclr_statistics.py tests/test_iclr_tie_breaking.py \
  tests/test_iclr_outcome_containment.py tests/test_iclr_saved_verification.py
```

This tests synthetic fixtures and integrity checks, not the paper's experiments.
Some other tests require omitted raw data or the original development layout;
the entire historical test suite is not advertised as data-free.

## Reproduce figures or compile the paper

From `tempopb/`, run one self-contained figure script:

```sh
uv run --locked python ../figures/coverage/figure.py
```

Replace `coverage` with `schematic` or `external` as needed. These commands
rewrite exports in that figure directory; run them in a working copy. Original
stored numbers and intervals are embedded in the scripts. Fonts can affect
rendering across systems. The frontier and construction generators remain in
`tempopb/src/`; see [reproduction boundaries](docs/REPRODUCIBILITY.md) before
running historical generators.

Compile the paper using the commands in [manuscript/REPRODUCE.md](manuscript/REPRODUCE.md).

## Data and release boundaries

Raw ballot files, credentials, local environments, agent instructions, editorial
reviews and working notes are not included. Scientific audits and correction
records are retained. Read [data and provenance](docs/DATA_AND_PROVENANCE.md)
before distributing the repository, and [NOTICE.md](NOTICE.md) for attribution
and the unresolved project-license choice.

The exact authenticated r3 supplement is a separate immutable artifact. This
clean repository export does not replace its historical verification chain.
