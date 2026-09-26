"""Append-only protocol-v2 normalization for set-valued approval ballots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from iclr_corpus import SeriesRef
from iclr_multicity_protocol import (
    PABULIB_COMMIT,
    load_canonical_index_from_manifest,
    multicity_manifest_retained_sha256,
)
from parse_pb import PBInstance, Vote, parse_pb_file


SEMANTICS_PROFILE = "approval-set-first-occurrence-v2"
_EXPECTED_PARENT_MANIFEST_RETAINED_SHA256 = (
    "2c2d727c42a78c57cf2de890d0816bf5267bb31b0066e070164866eb5cdf427a"
)
_EXPECTED_CORPUS_SEMANTIC_SHA256 = (
    "f7929c40b974db3752a30976a59f1bd79afc9e23a9a635b0d3c56cc649d76f37"
)
_EXPECTED_MULTICITY_COUNTS = {
    "series": 75,
    "files": 397,
    "ballots": 1_126_949,
    "raw_approval_tokens": 7_057_192,
    "canonical_approval_tokens": 7_057_191,
    "affected_files": 1,
    "affected_ballots": 1,
    "removed_tokens": 1,
    "empty_ballots": 0,
    "unknown_project_tokens": 0,
}
_EXPECTED_WOLA_ANOMALY = {
    "series": "Poland/Warszawa/Wola",
    "year": 2021,
    "source_id": "Poland_Warszawa_2022_Wola.pb",
    "vote_index_zero_based": 2976,
    "voter_id": "37051",
    "original_projects": ["147", "1040", "1040"],
    "canonical_projects": ["147", "1040"],
    "duplicate_projects": ["1040"],
}
_EXPECTED_WOLA_FILE_ROW = {
    "series": "Poland/Warszawa/Wola",
    "year": 2021,
    "name": "Poland_Warszawa_2022_Wola.pb",
    "raw_file_sha256": (
        "9054e7b28c97b18a990d75baaea5777942a7703774863944324567fbec60f61a"
    ),
    "raw_semantic_sha256": (
        "1750e09bc13b584ec1a0066acb8861652ad4e0bbd68b2799923a0008ff867be1"
    ),
    "v2_semantic_sha256": (
        "495b6b54e9049e62554f2f433b0cf14ee2fb7b2dd3279134a9fcfbafda37b312"
    ),
    "changed": True,
}


@dataclass(frozen=True)
class DuplicateApprovalAnomaly:
    """One ballot whose raw token sequence repeated an approved project."""

    source_id: str
    vote_index_zero_based: int
    voter_id: str
    original_projects: tuple[str, ...]
    canonical_projects: tuple[str, ...]
    duplicate_projects: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalizationResult:
    """A fresh instance produced without mutating protocol-v1 objects."""

    instance: PBInstance
    anomalies: tuple[DuplicateApprovalAnomaly, ...]


def canonicalize_instance(
    instance: PBInstance,
    *,
    source_id: str | None = None,
) -> CanonicalizationResult:
    """Copy one parsed instance and remove repeated approvals in source order."""

    if instance.vote_type.casefold() != "approval":
        raise ValueError("protocol v2 canonicalizes approval ballots only")
    stable_source_id = source_id or Path(instance.path).name
    votes = []
    anomalies = []
    for vote_index, vote in enumerate(instance.votes):
        seen: set[str] = set()
        canonical_projects = []
        duplicate_projects = []
        for project_id in vote.projects:
            if project_id in seen:
                duplicate_projects.append(project_id)
            else:
                seen.add(project_id)
                canonical_projects.append(project_id)
        canonical_tuple = tuple(canonical_projects)
        votes.append(
            Vote(
                vid=vote.vid,
                projects=canonical_tuple,
                age=vote.age,
                sex=vote.sex,
                neighborhood=vote.neighborhood,
            )
        )
        if duplicate_projects:
            anomalies.append(
                DuplicateApprovalAnomaly(
                    source_id=stable_source_id,
                    vote_index_zero_based=vote_index,
                    voter_id=vote.vid,
                    original_projects=tuple(vote.projects),
                    canonical_projects=canonical_tuple,
                    duplicate_projects=tuple(duplicate_projects),
                )
            )
    normalized = PBInstance(
        # Several v1 feature/supporter caches key only on ``PBInstance.path``.
        # A distinct logical path prevents a raw-v1 cache entry from being
        # reused for the normalized v2 ballot semantics in the same process.
        path=f"{instance.path}#{SEMANTICS_PROFILE}",
        meta=dict(instance.meta),
        projects=dict(instance.projects),
        votes=votes,
    )
    return CanonicalizationResult(
        instance=normalized,
        anomalies=tuple(anomalies),
    )


def parse_pb_file_v2(path: Path) -> PBInstance:
    """Parse raw bytes with v1, then apply protocol-v2 approval semantics."""

    parsed = parse_pb_file(Path(path))
    return canonicalize_instance(parsed, source_id=Path(path).name).instance


def load_series_v2(
    ref: SeriesRef,
    years: Sequence[int] | None = None,
) -> dict[int, PBInstance]:
    """Load a requested series view using only protocol-v2 instances."""

    wanted = set(years) if years is not None else set(ref.years)
    return {
        year: parse_pb_file_v2(path)
        for year, path in zip(ref.years, ref.paths)
        if year in wanted
    }


def _semantics_receipt_file_index(
    receipt: Mapping[str, object],
) -> dict[tuple[str, int, str], Mapping[str, object]]:
    """Index authenticated per-election semantics without weakening the receipt."""

    if receipt.get("semantics_profile") != SEMANTICS_PROFILE:
        raise RuntimeError("v2 semantics receipt profile differs")
    rows = receipt.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("v2 semantics receipt file inventory is malformed")
    indexed: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("v2 semantics receipt file row is malformed")
        try:
            identity = (
                str(row["series"]),
                int(row["year"]),
                str(row["name"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("v2 semantics receipt file identity is malformed") from exc
        if identity in indexed:
            raise RuntimeError(f"duplicate v2 semantics receipt row: {identity}")
        for field in ("raw_file_sha256", "v2_semantic_sha256"):
            if not _is_sha256(row.get(field)):
                raise RuntimeError(
                    f"v2 semantics receipt {field} is malformed for {identity}"
                )
        indexed[identity] = row
    return indexed


def validate_manifest_receipt_identities_v2(
    index: Mapping[str, SeriesRef],
    receipt: Mapping[str, object],
) -> None:
    """Bind every manifest series/year/file identity before selecting fit rows."""

    receipt_identities = set(_semantics_receipt_file_index(receipt))
    manifest_identities = {
        (ref.key, int(year), Path(path).name)
        for key, ref in index.items()
        for year, path in zip(ref.years, ref.paths)
        if key == ref.key
    }
    if len(manifest_identities) != sum(len(ref.years) for ref in index.values()):
        raise RuntimeError("v2 manifest identities are malformed or duplicated")
    if manifest_identities != receipt_identities:
        missing = sorted(receipt_identities - manifest_identities)[:3]
        unexpected = sorted(manifest_identities - receipt_identities)[:3]
        raise RuntimeError(
            "v2 manifest identities differ from semantics receipt: "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_series_authenticated_v2(
    ref: SeriesRef,
    semantics_receipt: Mapping[str, object],
    years: Sequence[int] | None = None,
) -> dict[int, PBInstance]:
    """Load v2 instances bound to the receipt's bytes and parsed semantics."""

    requested = tuple(ref.years if years is None else years)
    unknown_years = sorted(set(requested) - set(ref.years))
    if unknown_years:
        raise RuntimeError(f"requested years are absent from {ref.key}: {unknown_years}")
    receipt_rows = _semantics_receipt_file_index(semantics_receipt)
    paths_by_year = dict(zip(ref.years, ref.paths))
    loaded: dict[int, PBInstance] = {}
    for year in requested:
        path = Path(paths_by_year[year])
        identity = (ref.key, int(year), path.name)
        row = receipt_rows.get(identity)
        if row is None:
            raise RuntimeError(f"missing v2 semantics receipt row for {identity}")
        expected_raw_sha256 = str(row["raw_file_sha256"])
        before = _path_sha256(path)
        if before != expected_raw_sha256:
            raise RuntimeError(f"v2 raw file digest differs for {ref.key} {year}")
        raw_instance = parse_pb_file(path)
        normalized = canonicalize_instance(
            raw_instance,
            source_id=path.name,
        ).instance
        after = _path_sha256(path)
        if after != before or after != expected_raw_sha256:
            raise RuntimeError(f"v2 raw file changed during parse for {ref.key} {year}")
        if any(
            len(vote.projects) != len(set(vote.projects))
            for vote in normalized.votes
        ):
            raise RuntimeError(
                f"v2 canonicalization retained duplicate approvals for {ref.key} {year}"
            )
        observed_semantic_sha256 = canonical_instance_sha256(normalized)
        if observed_semantic_sha256 != row["v2_semantic_sha256"]:
            raise RuntimeError(
                f"v2 canonical semantic digest differs for {ref.key} {year}"
            )
        loaded[int(year)] = normalized
    return loaded


def canonical_instance_sha256(instance: PBInstance) -> str:
    """Hash parsed election semantics without a machine-specific path."""

    for mapping_key, project in instance.projects.items():
        if mapping_key != project.pid:
            raise ValueError(
                f"project mapping key {mapping_key!r} differs from pid {project.pid!r}"
            )
    payload = {
        "meta": instance.meta,
        "projects": [
            {
                "pid": project.pid,
                "cost": project.cost,
                "selected": project.selected,
                "name": project.name,
                "category": project.category,
                "target": project.target,
                "neighborhood": project.neighborhood,
            }
            for _, project in sorted(instance.projects.items())
        ],
        "votes": [
            {
                "vid": vote.vid,
                "projects": list(vote.projects),
                "age": vote.age,
                "sex": vote.sex,
                "neighborhood": vote.neighborhood,
            }
            for vote in instance.votes
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_authenticated_semantics_receipt(
    manifest_path: Path,
    data_dir: Path,
    *,
    enforce_expected_counts: bool = True,
) -> dict[str, object]:
    """Authenticate, normalize, and summarize every manifest-listed election."""

    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("corpus manifest must be a JSON object")
    index = load_canonical_index_from_manifest(
        manifest_path,
        data_dir,
        enforce_expected_counts=enforce_expected_counts,
    )
    manifest_rows = manifest.get("files")
    if not isinstance(manifest_rows, list):
        raise RuntimeError("corpus manifest files must be a list")
    by_election: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in manifest_rows:
        if not isinstance(row, dict):
            raise RuntimeError("corpus manifest contains a malformed file row")
        key = (str(row.get("series", "")), int(row.get("year", -1)))
        if key in by_election:
            raise RuntimeError(f"duplicate corpus election in manifest: {key}")
        by_election[key] = row

    files = []
    anomalies = []
    raw_tokens = 0
    canonical_tokens = 0
    ballots = 0
    empty_ballots = 0
    unknown_project_tokens = 0
    affected_files: set[str] = set()
    consumed: set[tuple[str, int]] = set()
    for series, ref in sorted(index.items()):
        for year, path in zip(ref.years, ref.paths):
            key = (series, int(year))
            row = by_election.get(key)
            if row is None:
                raise RuntimeError(f"manifest row missing for corpus election: {key}")
            consumed.add(key)
            source_id = str(row.get("name", ""))
            expected_raw_sha256 = str(row.get("sha256", ""))
            if Path(path).name != source_id:
                raise RuntimeError(f"manifest file name differs for corpus election: {key}")
            if _path_sha256(path) != expected_raw_sha256:
                raise RuntimeError(f"raw hash changed before parsing: {source_id}")
            raw_instance = parse_pb_file(path)
            result = canonicalize_instance(raw_instance, source_id=source_id)
            if _path_sha256(path) != expected_raw_sha256:
                raise RuntimeError(f"raw hash changed while parsing: {source_id}")

            ballots += len(raw_instance.votes)
            raw_tokens += sum(len(vote.projects) for vote in raw_instance.votes)
            canonical_tokens += sum(
                len(vote.projects) for vote in result.instance.votes
            )
            empty_ballots += sum(
                not vote.projects for vote in result.instance.votes
            )
            known_projects = set(raw_instance.projects)
            unknown_project_tokens += sum(
                project_id not in known_projects
                for vote in raw_instance.votes
                for project_id in vote.projects
            )
            raw_semantic_sha256 = canonical_instance_sha256(raw_instance)
            v2_semantic_sha256 = canonical_instance_sha256(result.instance)
            changed = raw_semantic_sha256 != v2_semantic_sha256
            if changed:
                affected_files.add(source_id)
            files.append(
                {
                    "series": series,
                    "year": int(year),
                    "name": source_id,
                    "raw_file_sha256": expected_raw_sha256,
                    "raw_semantic_sha256": raw_semantic_sha256,
                    "v2_semantic_sha256": v2_semantic_sha256,
                    "changed": changed,
                }
            )
            for anomaly in result.anomalies:
                anomalies.append(
                    {
                        "series": series,
                        "year": int(year),
                        **{
                            key: list(value) if isinstance(value, tuple) else value
                            for key, value in asdict(anomaly).items()
                        },
                    }
                )

    if consumed != set(by_election):
        missing = sorted(set(by_election) - consumed)
        raise RuntimeError(f"unconsumed corpus manifest elections: {missing[:5]}")
    counts = {
        "series": len(index),
        "files": len(files),
        "ballots": ballots,
        "raw_approval_tokens": raw_tokens,
        "canonical_approval_tokens": canonical_tokens,
        "affected_files": len(affected_files),
        "affected_ballots": len(anomalies),
        "removed_tokens": raw_tokens - canonical_tokens,
        "empty_ballots": empty_ballots,
        "unknown_project_tokens": unknown_project_tokens,
    }
    corpus_semantic_sha256 = _canonical_json_sha256(
        {
            "semantics_profile": SEMANTICS_PROFILE,
            "files": files,
        }
    )
    receipt = {
        "schema_version": 1,
        "semantics_profile": SEMANTICS_PROFILE,
        "parent_manifest_retained_sha256": multicity_manifest_retained_sha256(
            manifest
        ),
        "source_commit": manifest.get("source_commit"),
        "counts": counts,
        "files": files,
        "anomalies": anomalies,
        "corpus_semantic_sha256": corpus_semantic_sha256,
    }
    if enforce_expected_counts:
        validate_multicity_semantics_receipt(receipt)
    return receipt


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_json_equal(actual: object, expected: object) -> bool:
    """Compare parsed JSON values without bool/int/float aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(observed, wanted)
            for observed, wanted in zip(actual, expected)
        )
    return actual == expected


def validate_multicity_semantics_receipt(
    receipt: dict[str, object],
) -> dict[str, object]:
    """Require the exact authenticated corpus-v2 semantic transition."""

    expected_top_level = {
        "schema_version",
        "semantics_profile",
        "parent_manifest_retained_sha256",
        "source_commit",
        "counts",
        "files",
        "anomalies",
        "corpus_semantic_sha256",
    }
    if set(receipt) != expected_top_level:
        raise RuntimeError("semantics receipt fields differ from the v2 schema")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
    ):
        raise RuntimeError("semantics receipt schema version differs")
    if receipt.get("semantics_profile") != SEMANTICS_PROFILE:
        raise RuntimeError("semantics receipt profile differs")
    if receipt.get("source_commit") != PABULIB_COMMIT:
        raise RuntimeError("semantics receipt source commit differs")
    if (
        receipt.get("parent_manifest_retained_sha256")
        != _EXPECTED_PARENT_MANIFEST_RETAINED_SHA256
    ):
        raise RuntimeError("semantics receipt parent manifest differs")
    counts = receipt.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(_EXPECTED_MULTICITY_COUNTS)
        or any(type(value) is not int for value in counts.values())
        or counts != _EXPECTED_MULTICITY_COUNTS
    ):
        raise RuntimeError("semantics receipt corpus counts differ")
    if not _exact_json_equal(
        receipt.get("anomalies"),
        [_EXPECTED_WOLA_ANOMALY],
    ):
        raise RuntimeError("semantics receipt anomaly inventory differs")

    files = receipt.get("files")
    if not isinstance(files, list) or len(files) != 397:
        raise RuntimeError("semantics receipt file inventory differs")
    expected_file_fields = {
        "series",
        "year",
        "name",
        "raw_file_sha256",
        "raw_semantic_sha256",
        "v2_semantic_sha256",
        "changed",
    }
    identities = []
    changed_rows = []
    for row in files:
        if not isinstance(row, dict) or set(row) != expected_file_fields:
            raise RuntimeError("semantics receipt contains a malformed file row")
        if (
            type(row.get("series")) is not str
            or type(row.get("year")) is not int
            or type(row.get("name")) is not str
            or type(row.get("changed")) is not bool
        ):
            raise RuntimeError("semantics receipt contains a malformed file row")
        identity = (row["series"], row["year"], row["name"])
        identities.append(identity)
        for field in (
            "raw_file_sha256",
            "raw_semantic_sha256",
            "v2_semantic_sha256",
        ):
            if not _is_sha256(row.get(field)):
                raise RuntimeError(
                    f"semantics receipt has invalid {field} for {identity}"
                )
        changed = row.get("changed")
        if changed != (
            row["raw_semantic_sha256"] != row["v2_semantic_sha256"]
        ):
            raise RuntimeError(f"semantics change flag differs for {identity}")
        if changed:
            changed_rows.append(row)
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise RuntimeError("semantics receipt file identities are not canonical")
    if not _exact_json_equal(changed_rows, [_EXPECTED_WOLA_FILE_ROW]):
        raise RuntimeError("semantics receipt changed-file inventory differs")

    observed_corpus_digest = receipt.get("corpus_semantic_sha256")
    recomputed_corpus_digest = _canonical_json_sha256(
        {
            "semantics_profile": SEMANTICS_PROFILE,
            "files": files,
        }
    )
    if (
        observed_corpus_digest != recomputed_corpus_digest
        or observed_corpus_digest != _EXPECTED_CORPUS_SEMANTIC_SHA256
    ):
        raise RuntimeError("semantics receipt corpus digest differs")
    return receipt
