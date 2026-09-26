"""Corpus indexing and frozen evaluation splits for the ICLR learned-policy work.

The index reproduces the filter chain used by ``run_experiments.py`` (approval
ballots, recorded winners, demographic coverage, minimum consecutive run) so
that any policy evaluated here is scored on exactly the corpus the replay
results in ``results/`` were computed on.

Splits are frozen to disk on first build and reloaded thereafter; a policy must
never see a split boundary move between runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from cohorts import historical_winners
from parse_pb import PBInstance, parse_pb_file
from run_experiments import (
    MAX_VOTERS,
    MIN_DEMO_COVERAGE,
    MIN_RUN,
    demo_coverage,
    longest_run_years,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "pb"
SPLIT_DIR = ROOT / "results" / "iclr_splits"

# Series excluded from the primary corpus by the existing analyses; kept here so
# the learned policy is scored on the same 19 series as the replay baselines.
PRIMARY_EXCLUDE = frozenset(
    {"Poland/Warszawa/CITYWIDE", "Poland/Warszawa/subunit Wawer"}
)

__all__ = [
    "CorpusConfig",
    "SeriesRef",
    "Split",
    "build_series_index",
    "load_series",
    "temporal_split",
    "leave_city_out",
    "leave_district_out",
    "save_split",
    "load_split",
]


@dataclass(frozen=True)
class CorpusConfig:
    """Filter chain defining the usable longitudinal corpus."""

    data_dir: Path = DATA_DIR
    min_run: int = MIN_RUN
    min_demo_coverage: float = MIN_DEMO_COVERAGE
    max_voters: int = MAX_VOTERS
    exclude: frozenset = PRIMARY_EXCLUDE


@dataclass(frozen=True)
class SeriesRef:
    """One (city, district) series: its consecutive run and the files backing it."""

    key: str
    years: Tuple[int, ...]
    paths: Tuple[Path, ...]

    @property
    def city(self) -> str:
        parts = self.key.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else self.key


@dataclass(frozen=True)
class Split:
    """A frozen train/test partition over series and years."""

    name: str
    train: Tuple[Tuple[str, Tuple[int, ...]], ...]
    test: Tuple[Tuple[str, Tuple[int, ...]], ...]
    note: str = ""

    def train_keys(self) -> List[str]:
        return [k for k, _ in self.train]

    def test_keys(self) -> List[str]:
        return [k for k, _ in self.test]


def build_series_index(cfg: Optional[CorpusConfig] = None) -> Dict[str, SeriesRef]:
    """Scan the .pb corpus and return usable series keyed by series key.

    Args:
        cfg: Filter chain; defaults to the primary-corpus configuration.

    Returns:
        Mapping from series key to its longest consecutive run.

    Raises:
        FileNotFoundError: When the data directory does not exist.
    """
    cfg = cfg or CorpusConfig()
    if not cfg.data_dir.is_dir():
        raise FileNotFoundError(f"PB data directory not found: {cfg.data_dir}")

    by_series: Dict[str, Dict[int, Path]] = {}
    for path in sorted(cfg.data_dir.glob("*.pb")):
        try:
            inst = parse_pb_file(path)
        except (OSError, ValueError, KeyError) as exc:
            logger.error("parse failed %s: %s", path.name, exc)
            continue
        if inst.vote_type != "approval" or inst.year is None:
            continue
        if not historical_winners(inst):
            continue
        if not inst.votes or len(inst.votes) > cfg.max_voters:
            continue
        if demo_coverage(inst) < cfg.min_demo_coverage:
            continue
        by_series.setdefault(inst.series_key(), {})[inst.year] = path

    index: Dict[str, SeriesRef] = {}
    for key, years_map in sorted(by_series.items()):
        if key in cfg.exclude:
            continue
        run = longest_run_years(list(years_map))
        if len(run) < cfg.min_run:
            continue
        index[key] = SeriesRef(
            key=key,
            years=tuple(run),
            paths=tuple(years_map[y] for y in run),
        )

    logger.info("usable series: %d (run >= %d)", len(index), cfg.min_run)
    return index


def load_series(ref: SeriesRef, years: Optional[Sequence[int]] = None) -> Dict[int, PBInstance]:
    """Parse the instances of one series, optionally restricted to given years."""
    wanted = set(years) if years is not None else set(ref.years)
    return {
        y: parse_pb_file(p)
        for y, p in zip(ref.years, ref.paths)
        if y in wanted
    }


def temporal_split(
    index: Dict[str, SeriesRef],
    train_through: int = 2022,
    min_train_years: int = 3,
    min_test_years: int = 1,
) -> Split:
    """Train on years <= train_through, test on later years of the same series.

    A policy is fit on early editions and scored on editions it has never seen,
    which is the honest analogue of deployment: a city adopts the rule now and
    runs it on next year's ballots.
    """
    train: List[Tuple[str, Tuple[int, ...]]] = []
    test: List[Tuple[str, Tuple[int, ...]]] = []
    for key, ref in sorted(index.items()):
        early = tuple(y for y in ref.years if y <= train_through)
        late = tuple(y for y in ref.years if y > train_through)
        if len(early) < min_train_years or len(late) < min_test_years:
            continue
        train.append((key, early))
        test.append((key, late))
    return Split(
        name=f"temporal_{train_through}",
        train=tuple(train),
        test=tuple(test),
        note=(
            f"train years <= {train_through}, test years > {train_through}; "
            f"series need >= {min_train_years} train and >= {min_test_years} test years"
        ),
    )


def leave_city_out(index: Dict[str, SeriesRef], held_out_city: str) -> Split:
    """Train on every other city, test on all series of the held-out city.

    With this corpus the only available held-out city is Lodz, which supplies
    the paper's cross-city transfer test.
    """
    train = tuple(
        (k, r.years) for k, r in sorted(index.items()) if r.city != held_out_city
    )
    test = tuple(
        (k, r.years) for k, r in sorted(index.items()) if r.city == held_out_city
    )
    if not test:
        raise ValueError(f"no series found for held-out city {held_out_city!r}")
    return Split(
        name=f"city_out_{held_out_city.replace('/', '_')}",
        train=train,
        test=test,
        note=f"held-out city: {held_out_city}; full year runs on both sides",
    )


def leave_district_out(index: Dict[str, SeriesRef], fold: int, n_folds: int = 5) -> Split:
    """Deterministic k-fold over series keys (districts), sorted for stability."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold {fold} outside [0, {n_folds})")
    keys = sorted(index)
    test_keys = {k for i, k in enumerate(keys) if i % n_folds == fold}
    train = tuple((k, index[k].years) for k in keys if k not in test_keys)
    test = tuple((k, index[k].years) for k in keys if k in test_keys)
    return Split(
        name=f"district_out_f{fold}of{n_folds}",
        train=train,
        test=test,
        note=f"deterministic stride-{n_folds} fold {fold} over sorted series keys",
    )


def save_split(split: Split, split_dir: Path = SPLIT_DIR) -> Path:
    """Freeze a split to JSON so evaluation boundaries never drift."""
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{split.name}.json"
    payload = {
        "name": split.name,
        "note": split.note,
        "train": [[k, list(ys)] for k, ys in split.train],
        "test": [[k, list(ys)] for k, ys in split.test],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("froze split %s -> %s", split.name, path)
    return path


def load_split(name: str, split_dir: Path = SPLIT_DIR) -> Split:
    """Reload a frozen split.

    Raises:
        FileNotFoundError: When the split has not been frozen yet.
    """
    path = split_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"split not frozen: {path}")
    payload = json.loads(path.read_text())
    return Split(
        name=payload["name"],
        train=tuple((k, tuple(ys)) for k, ys in payload["train"]),
        test=tuple((k, tuple(ys)) for k, ys in payload["test"]),
        note=payload.get("note", ""),
    )
