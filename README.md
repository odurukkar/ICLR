<div align="center">

# Coverage Is Not Fairness
### Auditing Learned Controls in Participatory Budgeting

**Research materials for an ICLR 2027 submission**

Manuscript revision 3 · September 25, 2026

[**Read the paper**](manuscript/paper.pdf) · [**Source code**](tempopb/src/) · [**Reproducibility**](docs/REPRODUCIBILITY.md) · [**Data & provenance**](docs/DATA_AND_PROVENANCE.md)

</div>

---

## Overview

Can a participatory budgeting allocation score perfectly on a group-spending fairness metric while leaving almost every voter without a funded project they approved? This paper shows that it can, and examines how learned controls interact with the allocation rules that turn scores into funded projects.

We study **cumulative share deficit (CSD)** alongside **voter coverage**, audit learned endowments and project priorities, and evaluate whether findings transfer beyond the training setting. The results support reporting spending-based fairness together with voter coverage, and evaluating learned interventions through the allocation changes they actually produce.

> **Submission materials.** This repository accompanies a paper prepared for submission to ICLR 2027. It includes the anonymous manuscript, research code, saved results, and figure assets. This designation does not indicate acceptance.

## Research highlights

- **Spending fairness and voter coverage can diverge.** A constructed allocation achieves zero CSD while excluding nearly all voters.
- **Allocation rules matter.** Replaying fixed score weights and feature definitions through different rules reveals changes in exclusion and welfare that a spending metric alone does not capture.
- **Simple baselines are informative.** Learned endowments improve CSD on held-out Warsaw series, but a post-lock age-table baseline has a lower mean.
- **External validation limits the claims.** The external studies do not confirm transfer. Negative and null results, post-hoc corrections, and failed validation decisions are retained in the released evidence.

## Paper and repository guide

| Resource | Description |
| --- | --- |
| [Paper PDF](manuscript/paper.pdf) | Current anonymous manuscript: 29 pages, including nine main-text pages |
| [LaTeX source](manuscript/tex/) | Matching manuscript, bibliography, style files, and figure PDFs |
| [Research code](tempopb/src/) | Allocation rules, policy learning, analysis, and verification |
| [Tests](tempopb/tests/) | Scientific unit and regression tests |
| [Saved results](tempopb/results/) · [Analysis outputs](tempopb/analysis-output/) | Evidence from the revision 3 release |
| [Figures](figures/) | Self-contained main-figure scripts, PDF/PNG exports, and plotted values |
| [Reproduction guide](docs/REPRODUCIBILITY.md) | Supported checks, historical authentication, and reproduction boundaries |
| [Release manifest](RELEASE_MANIFEST.json) | File sizes, SHA-256 digests, and export provenance |

## Getting started

### 1. Check repository integrity

From the repository root, verify the listed file hashes and Python syntax without installing dependencies:

```sh
python3 scripts/check_release.py
```

This checks the exported materials; it does not rerun the experiments.

### 2. Run the data-free regression checks

Use **Python 3.12** and **uv** with the locked environment:

```sh
cd tempopb
uv sync --locked
PYTHONPATH=src uv run --locked pytest -q -p no:cacheprovider \
  tests/test_iclr_statistics.py tests/test_iclr_tie_breaking.py \
  tests/test_iclr_outcome_containment.py tests/test_iclr_saved_verification.py
```

This subset checks synthetic fixtures and saved-result integrity. Other tests may require omitted raw data or the original development layout; the full historical suite is not data-free.

### 3. Reproduce a main figure

From `tempopb/`, run:

```sh
uv run --locked python ../figures/coverage/figure.py
```

Replace `coverage` with `schematic` or `external` to reproduce the other self-contained main figures. The scripts use stored numbers and intervals and rewrite their figure exports, so run them in a working copy. Fonts may affect rendering across systems.

For the frontier and construction generators, consult the [reproduction boundaries](docs/REPRODUCIBILITY.md) before running historical code.

### 4. Build the manuscript

Follow the [manuscript build instructions](manuscript/REPRODUCE.md). Use the final assets and `numbers.tex` in `manuscript/tex/`.

## Data availability and scope

Raw ballot files are not redistributed. The [data and provenance guide](docs/DATA_AND_PROVENANCE.md) documents their sources, the export omissions, and the limited public-source ballot excerpt retained in authentication records. [EXCLUDED_RAW_DATA.json](EXCLUDED_RAW_DATA.json) inventories the omitted raw ballot files.

The corrected primary and first multi-city studies are post-hoc replays. This release preserves scientific audits, correction records, and unsuccessful external validation outcomes; it does not claim successful generalization.

The original authenticated revision 3 supplement is a separate immutable artifact. This repository does not replace its historical verification chain. See the [reproduction guide](docs/REPRODUCIBILITY.md) for its digest and authentication instructions.

## Attribution and licensing

See [NOTICE.md](NOTICE.md) for source attribution and third-party notices. No project-wide software license has been selected; the bundled third-party notices apply to their respective files.
