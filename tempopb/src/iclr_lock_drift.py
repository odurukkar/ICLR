"""Report which locked files have changed since an external protocol lock.

The external locks pin every tracked file by digest, and the v1 lock's
inventory is the whole ``src`` tree. That is a strong integrity guarantee for
the decision it certifies, but it also means the lock reports a difference
whenever the manuscript pipeline is touched afterwards -- and a paper keeps
being written after a study closes.

This module separates the two questions the whole-tree check conflates:

1. Did anything that could change the recorded decision move? Corpus files,
   fitted weights, the corpus manifest, the lineage records, and the study
   modules themselves must stay byte-exact forever. Any difference there
   invalidates the study.
2. Did anything else move? Generators, figure scripts and the artifact builder
   can drift without touching a decision that is already computed and stored.
   That drift is disclosed, not silently accepted.

A decision-critical file may still need to change, as one did here. Such a
change is not waived: it is recorded in ``results/iclr_lock_amendments.json``
with the locked digest, the amended digest, the reason, and the verification
that the recorded decisions still hold. Drift matching a record is reported as
an amendment; drift matching nothing fails.

It reads locks; it never writes them, and it never modifies a locked file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Dict, List, Mapping, Sequence, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

#: Post-lock changes to decision-critical files, each carrying the evidence
#: that the recorded decisions still hold. Drift matching an entry here is
#: reported as an amendment; drift that matches nothing still fails.
AMENDMENTS_PATH = RESULTS / "iclr_lock_amendments.json"

#: Locks in lineage order, newest last.
LOCK_PATHS: Tuple[Path, ...] = (
    RESULTS / "iclr_multicity" / "protocol_lock.json",
    RESULTS / "iclr_external_validation" / "protocol_lock.json",
    RESULTS / "iclr_external_support_set" / "protocol_lock.json",
    RESULTS / "iclr_external_support_set_amendment" / "protocol_lock.json",
    RESULTS / "iclr_multicity_v2" / "protocol_lock.json",
)


class MissingConfiguredLocksError(FileNotFoundError):
    """Raised when one or more locks required by the audit are absent."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(Path(path) for path in paths)
        labels = []
        for path in self.paths:
            try:
                labels.append(path.relative_to(ROOT).as_posix())
            except ValueError:
                labels.append(path.as_posix())
        if len(labels) == 1:
            message = f"configured protocol lock is missing: {labels[0]}"
        else:
            message = "configured protocol locks are missing: " + ", ".join(labels)
        super().__init__(message)


#: Label prefixes whose bytes bear on a recorded decision.
DECISION_PREFIXES: Tuple[str, ...] = (
    "artifact/",
    "corpus/",
    "fit/",
    "split/",
    "original/",
    "environment/",
    "v1/",
    "v2/",
)

#: A later lock re-tracks an earlier lock's inventory under this prefix. The
#: entry means the same thing as the label it wraps, so it is classified the
#: same way rather than being treated as a decision input by its prefix alone.
_SNAPSHOT_PREFIX = "v2_snapshot/"

#: Historical, non-scientific source files intentionally omitted from the
#: minimal release. Warning-only handling is restricted to this reviewed set;
#: a new or misspelled tracked label fails closed until explicitly classified.
KNOWN_NONCRITICAL_MISSING_LABELS = frozenset(
    {
        "source/analysis_attributes.py",
        "source/analysis_breadth_neutral.py",
        "source/analysis_ceiling.py",
        "source/analysis_cohort_behavior.py",
        "source/analysis_district_res.py",
        "source/analyze_behavioral_baselines.py",
        "source/analyze_behavioral_calibration.py",
        "source/analyze_gate_families.py",
        "source/analyze_weighting_sensitivity.py",
        "source/audit_coverage.py",
        "source/behavioral_sim.py",
        "source/behavioral_validation.py",
        "source/build_confirmatory_colab.py",
        "source/build_family_arms_colab.py",
        "source/build_figmirror_candidates.py",
        "source/build_gate_matrix_colab.py",
        "source/gen_figures.py",
        "source/gen_gate_numbers.py",
        "source/gen_numbers.py",
        "source/gen_supp_figures.py",
        "source/gen_supplement.py",
        "source/iclr_toy_model.py",
        "source/pseudonymize_behavioral_csv.py",
        "source/theory_check.py",
        "source/verify_gate_matrix_bundles.py",
    }
)

#: Present historical tooling whose current bytes intentionally differ from a
#: whole-tree lock. A new or misspelled drift label fails closed until reviewed.
KNOWN_NONCRITICAL_DRIFT_LABELS = frozenset(
    {
        "source/build_iclr_artifact.py",
        "source/gen_iclr_appendix.py",
        "source/gen_iclr_figures.py",
        "source/gen_iclr_numbers.py",
        "source/iclr_crosscity.py",
        "source/iclr_fetch_refs.py",
        "source/iclr_fig_dynamics.py",
        "source/iclr_fig_forest.py",
        "source/iclr_fig_hack.py",
        "source/iclr_fig_schematic.py",
        "source/iclr_style.py",
        "test/amendment",
        "test/external_support_set",
        "test/test_iclr_multicity_protocol.py",
        "test/test_iclr_multicity_train.py",
    }
)

#: Individual source modules that implement evaluation or the study protocol.
DECISION_SOURCES = frozenset(
    {
        "source/amendment",
        "source/iclr_approval_semantics_v2.py",
        "source/iclr_env.py",
        "source/iclr_cmaes.py",
        "source/iclr_corpus.py",
        "source/iclr_policy.py",
        "source/iclr_outcome.py",
        "source/iclr_priority_mes.py",
        "source/iclr_stats.py",
        "source/iclr_train.py",
        "source/iclr_external_validation.py",
        "source/iclr_external_support_set.py",
        "source/iclr_external_support_set_amendment.py",
        "source/iclr_multicity_evaluate.py",
        "source/iclr_multicity_evaluate_v2.py",
        "source/iclr_multicity_train.py",
        "source/iclr_multicity_train_v2.py",
        "source/iclr_multicity_protocol.py",
        "source/iclr_multicity_protocol_v2.py",
        "source/run_experiments.py",
        "source/rules.py",
        "source/cohorts.py",
        "source/parse_pb.py",
    }
)


def is_decision_critical(label: str) -> bool:
    """True when a tracked label's bytes could change the recorded decision."""
    if label.startswith(_SNAPSHOT_PREFIX):
        return is_decision_critical(label[len(_SNAPSHOT_PREFIX):])
    return label.startswith(DECISION_PREFIXES) or label in DECISION_SOURCES


def is_known_noncritical_missing(label: str) -> bool:
    """True only for a reviewed historical label the minimal release omits."""
    if label.startswith(_SNAPSHOT_PREFIX):
        return is_known_noncritical_missing(label[len(_SNAPSHOT_PREFIX):])
    return label in KNOWN_NONCRITICAL_MISSING_LABELS


def is_known_noncritical_drift(label: str) -> bool:
    """True only for reviewed tooling drift disclosed by the paper."""
    if label.startswith(_SNAPSHOT_PREFIX):
        return is_known_noncritical_drift(label[len(_SNAPSHOT_PREFIX):])
    return label in KNOWN_NONCRITICAL_DRIFT_LABELS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: object) -> bool:
    """True only for a canonical lowercase SHA-256 hexadecimal digest."""

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_tracked_path(
    label: str,
    value: object,
    root: Path | None = None,
) -> Path:
    """Resolve a lock row without permitting reads outside the repository."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"tracked path is malformed for {label}: {value!r}")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise RuntimeError(f"tracked path is outside repository for {label}: {value!r}")
    repository_root = Path(ROOT if root is None else root).resolve()
    candidate = (repository_root / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(repository_root)
    except ValueError as exc:
        raise RuntimeError(
            f"tracked path escapes repository for {label}: {value!r}"
        ) from exc
    return candidate


def recorded_amendments(
    path: Path = None,
) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    """Map (tracked path, locked digest, amended digest) to its record."""
    source = Path(path) if path is not None else AMENDMENTS_PATH
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    out: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for entry in payload.get("amendments", []):
        if entry.get("affects_recorded_decision") is not False:
            continue
        out[
            (
                entry["path"],
                entry["locked_sha256"],
                entry["amended_sha256"],
            )
        ] = entry
    return out


def lock_drift(
    lock_path: Path,
    *,
    root: Path | None = None,
    amendments_path: Path | None = None,
) -> Dict[str, List[str]]:
    """Classify every tracked file in one lock as intact, drifted, or missing."""
    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    tracked: Mapping[str, Mapping[str, str]] = payload["tracked_files"]
    report: Dict[str, List[str]] = {
        "intact": [],
        "drifted": [],
        "missing": [],
        "decision_drifted": [],
        "amended": [],
    }
    amendments = recorded_amendments(amendments_path)
    for label, row in sorted(tracked.items()):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"tracked row is malformed for {label}: {row!r}")
        locked_sha256 = row.get("sha256")
        if not is_sha256(locked_sha256):
            raise RuntimeError(
                f"locked SHA-256 is malformed for {label}: {locked_sha256!r}"
            )
        path = _resolve_tracked_path(label, row.get("path"), root)
        if not path.is_file():
            report["missing"].append(label)
            continue
        current_sha256 = _sha256(path)
        if current_sha256 == locked_sha256:
            report["intact"].append(label)
            continue
        report["drifted"].append(label)
        if not is_decision_critical(label):
            continue
        record = amendments.get((row["path"], locked_sha256, current_sha256))
        if record is None:
            report["decision_drifted"].append(label)
        else:
            report["amended"].append(label)
    return report


def audit(
    lock_paths: Sequence[Path] | None = None,
    *,
    root: Path | None = None,
    amendments_path: Path | None = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Report drift for every configured lock, failing if any lock is absent."""
    configured = LOCK_PATHS if lock_paths is None else lock_paths
    paths = tuple(Path(path) for path in configured)
    missing = tuple(path for path in paths if not path.is_file())
    if missing:
        raise MissingConfiguredLocksError(missing)
    repository_root = Path(ROOT if root is None else root).resolve()
    reports: Dict[str, Dict[str, List[str]]] = {}
    for path in paths:
        try:
            label = Path(path).resolve().relative_to(repository_root).as_posix()
        except ValueError:
            label = Path(path).resolve().as_posix()
        reports[label] = lock_drift(
            path,
            root=repository_root,
            amendments_path=amendments_path,
        )
    return reports


def audit_failure_categories(report: Mapping[str, List[str]]) -> Dict[str, List[str]]:
    """Return the exact lock rows that fail the accepted classification policy."""

    decision_missing = [
        label for label in report["missing"] if is_decision_critical(label)
    ]
    unknown_missing = [
        label
        for label in report["missing"]
        if not is_decision_critical(label)
        and not is_known_noncritical_missing(label)
    ]
    unknown_noncritical_drifted = [
        label
        for label in report["drifted"]
        if not is_decision_critical(label)
        and not is_known_noncritical_drift(label)
    ]
    return {
        "decision_drifted": list(report["decision_drifted"]),
        "decision_missing": decision_missing,
        "unknown_missing": unknown_missing,
        "unknown_noncritical_drifted": unknown_noncritical_drifted,
    }


def assert_audit_reports_pass(
    reports: Mapping[str, Mapping[str, List[str]]],
) -> None:
    """Raise when any audited lock row lacks an accepted classification."""

    rejected = {
        lock: categories
        for lock, report in reports.items()
        if any((categories := audit_failure_categories(report)).values())
    }
    if rejected:
        details = "; ".join(
            f"{lock}: "
            + ", ".join(
                f"{category}={labels}"
                for category, labels in categories.items()
                if labels
            )
            for lock, categories in rejected.items()
        )
        raise RuntimeError(
            "a locked input changed or is missing without an accepted "
            f"classification; the study is not verifiable ({details})"
        )


def assert_audit_passes(
    lock_paths: Sequence[Path] | None = None,
    *,
    root: Path | None = None,
    amendments_path: Path | None = None,
) -> Dict[str, Dict[str, List[str]]]:
    """Audit configured locks and fail unless every row is accepted."""

    reports = audit(
        lock_paths,
        root=root,
        amendments_path=amendments_path,
    )
    assert_audit_reports_pass(reports)
    return reports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failed = False
    try:
        reports = audit()
    except MissingConfiguredLocksError as exc:
        raise SystemExit(str(exc)) from None
    for lock, report in reports.items():
        logger.info(
            "%s: %d intact, %d drifted, %d missing",
            lock,
            len(report["intact"]),
            len(report["drifted"]),
            len(report["missing"]),
        )
        for label in report["drifted"]:
            logger.info("  drifted: %s", label)
        for label in report["amended"]:
            logger.info("  amended (disclosed, decision unaffected): %s", label)
        failure_categories = audit_failure_categories(report)
        decision_missing = failure_categories["decision_missing"]
        known_noncritical_missing = [
            label
            for label in report["missing"]
            if not is_decision_critical(label)
            and is_known_noncritical_missing(label)
        ]
        unknown_missing = failure_categories["unknown_missing"]
        unknown_noncritical_drifted = failure_categories[
            "unknown_noncritical_drifted"
        ]
        for label in known_noncritical_missing:
            logger.warning(
                "  missing (non-decision-critical; disclosed): %s", label
            )
        if (
            failure_categories["decision_drifted"]
            or decision_missing
            or unknown_missing
            or unknown_noncritical_drifted
        ):
            failed = True
            for label in failure_categories["decision_drifted"]:
                logger.error("  DECISION-CRITICAL DRIFT: %s", label)
            for label in decision_missing:
                logger.error("  DECISION-CRITICAL MISSING: %s", label)
            for label in unknown_missing:
                logger.error("  UNCLASSIFIED TRACKED LABEL MISSING: %s", label)
            for label in unknown_noncritical_drifted:
                logger.error("  UNCLASSIFIED TRACKED LABEL DRIFTED: %s", label)
    if failed:
        try:
            assert_audit_reports_pass(reports)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
    logger.info(
        "every decision-critical locked input is byte-identical or covered by "
        "a disclosed, decision-preserving amendment"
    )


if __name__ == "__main__":
    main()
