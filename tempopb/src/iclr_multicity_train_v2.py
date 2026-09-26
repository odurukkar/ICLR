"""Training entry point for the append-only multicity protocol v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

from iclr_approval_semantics_v2 import (
    SEMANTICS_PROFILE,
    load_series_authenticated_v2,
    validate_manifest_receipt_identities_v2,
    validate_multicity_semantics_receipt,
)
from iclr_corpus import SeriesRef, Split
from iclr_env import EnvConfig
from iclr_multicity_protocol import (
    PABULIB_COMMIT,
    load_canonical_index_from_manifest,
    load_frozen_split,
)
from iclr_multicity_protocol_v2 import (
    FIT_SOURCE_FILES_V2,
    LOCK_PROFILE_V2,
    ProtocolInputsSnapshotV2,
    ProtocolLockSnapshotV2,
    SPLIT_NAMES,
    assert_protocol_inputs_unchanged_v2,
    canonical_result_output_path_v2,
    canonical_result_root_v2,
    exact_json_equal_v2,
    load_protocol_inputs_snapshot_v2,
    protocol_config_v2,
    read_regular_bytes_artifact_v2,
    verify_multicity_protocol_lock_snapshot_v2,
    write_immutable_json_artifact_v2,
)
from iclr_multicity_train import (
    MultiCityTrainConfig,
    fit_one,
    fit_output_path,
)
from iclr_train import SeriesData


ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results" / "iclr_multicity_v2"
logger = logging.getLogger(__name__)


def write_immutable_fit_v2(path: Path, payload: Mapping[str, object]) -> str:
    """Install one fit through the protocol-v2 atomic no-follow writer."""

    return write_immutable_json_artifact_v2(path, payload)


def parse_train_args_v2(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, MultiCityTrainConfig]:
    """Parse only canonical protocol-v2 run coordinates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLIT_NAMES, required=True)
    parser.add_argument(
        "--arm",
        choices=("priority", "endowment", "outcome"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--sigma0", type=float, default=0.4)
    parser.add_argument("--bound", type=float, default=10.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    return args, MultiCityTrainConfig(
        split=args.split,
        arm=args.arm,
        seed=args.seed,
        generations=1 if args.smoke else args.generations,
        popsize=4 if args.smoke else args.popsize,
        sigma0=args.sigma0,
        bound=args.bound,
    )


def _resolved_under(path: Path, parent: Path, label: str) -> Path:
    """Resolve a path and reject symlink or component escapes from its root."""

    resolved = Path(path).resolve()
    boundary = Path(parent).resolve()
    try:
        resolved.relative_to(boundary)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes its canonical v2 directory: {resolved}") from exc
    return resolved


def load_training_series_data_v2(
    split: Split,
    index: Mapping[str, SeriesRef],
    semantics_receipt: Mapping[str, object],
) -> list[SeriesData]:
    """Parse only frozen fit years under the protocol-v2 ballot semantics."""
    rows = []
    for key, fit_years in split.train:
        if not fit_years:
            continue
        ref = index[key]
        instances = load_series_authenticated_v2(
            ref,
            semantics_receipt,
            years=fit_years,
        )
        missing = sorted(set(fit_years) - set(instances))
        if missing:
            raise RuntimeError(f"missing v2 fit years for {key}: {missing}")
        rows.append(
            SeriesData(
                ref=ref,
                train_years=tuple(fit_years),
                test_years=(),
                train_only=dict(instances),
                all_years=dict(instances),
            )
        )
    return rows


def authorize_fit_v2(
    config: MultiCityTrainConfig,
    *,
    smoke: bool,
    result_root: Path,
    repo_root: Path,
) -> ProtocolLockSnapshotV2 | None:
    """Require the exact protocol-v2 lock for every production fit."""

    if config.split not in SPLIT_NAMES:
        raise RuntimeError(f"fit does not name a canonical v2 split: {config.split!r}")
    if config.arm not in {"priority", "endowment", "outcome"}:
        raise RuntimeError(f"fit does not name a canonical v2 arm: {config.arm!r}")
    if type(config.seed) is not int:
        raise RuntimeError("fit seed must be an integer")
    if smoke:
        return None
    root = Path(repo_root)
    result = Path(result_root)
    lock_path = result / "protocol_lock.json"
    if not lock_path.is_file():
        raise RuntimeError(f"protocol-v2 lock is required before fitting: {lock_path}")
    snapshot = verify_multicity_protocol_lock_snapshot_v2(
        lock_path,
        root,
        result,
        root / "data" / "pb_multicity",
    )
    payload = snapshot.payload
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
        or payload.get("lock_profile") != LOCK_PROFILE_V2
        or not exact_json_equal_v2(payload.get("protocol"), protocol_config_v2())
    ):
        raise RuntimeError("v2 protocol lock profile differs")
    spec = {"split": config.split, "arm": config.arm, "seed": config.seed}
    inventory = payload.get("fit_inventory", [])
    if not isinstance(inventory, list) or not any(
        exact_json_equal_v2(spec, row) for row in inventory
    ):
        raise RuntimeError(f"fit is outside the locked v2 inventory: {spec}")
    protocol = payload["protocol"]
    expected = {
        "generations": protocol["generations"],
        "popsize": protocol["popsize"],
        "bound": protocol["coefficient_bound"],
        "sigma0": protocol["sigma0"],
        "welfare_penalty": protocol["objective"]["welfare_penalty"],
        "welfare_floor": protocol["objective"]["welfare_floor_ratio"],
    }
    observed = {
        "generations": config.generations,
        "popsize": config.popsize,
        "bound": config.bound,
        "sigma0": config.sigma0,
        "welfare_penalty": config.welfare_penalty,
        "welfare_floor": config.welfare_floor,
    }
    if not exact_json_equal_v2(observed, expected):
        raise RuntimeError(
            f"fit hyperparameters differ from protocol-v2 lock: {observed} != {expected}"
        )
    return snapshot


def _bound_digest(
    lock_payload: Mapping[str, object] | None,
    label: str,
    path: Path,
) -> str:
    content = read_regular_bytes_artifact_v2(
        Path(path),
        label=label,
    )
    assert content is not None
    observed = hashlib.sha256(content).hexdigest()
    if lock_payload is None:
        return observed
    tracked = lock_payload.get("tracked_files")
    row = tracked.get(label) if isinstance(tracked, dict) else None
    expected = row.get("sha256") if isinstance(row, dict) else None
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or expected != observed
    ):
        raise RuntimeError(f"locked v2 digest differs for {label}")
    return expected


def build_fit_provenance_v2(
    lock_payload: Mapping[str, object] | None,
    *,
    result_root: Path,
    repo_root: Path,
    split_name: str,
    input_snapshot: ProtocolInputsSnapshotV2 | None = None,
) -> dict[str, object]:
    """Bind a fit to v2 semantics, staged inputs, and its full source closure."""

    result = Path(result_root)
    root = Path(repo_root)
    manifest_path = result / "corpus_manifest.json"
    split_path = result / "splits" / f"{split_name}.json"
    receipt_path = result / "approval_semantics_receipt.json"
    receipt_bytes = read_regular_bytes_artifact_v2(
        receipt_path,
        label="approval semantics receipt",
    )
    assert receipt_bytes is not None
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("approval semantics receipt must be a JSON object")
    validate_multicity_semantics_receipt(receipt)
    if receipt.get("semantics_profile") != SEMANTICS_PROFILE:
        raise RuntimeError("approval semantics receipt profile differs")

    if input_snapshot is None:
        manifest_digest = _bound_digest(
            lock_payload,
            "artifact/corpus_manifest",
            manifest_path,
        )
        split_digest = _bound_digest(
            lock_payload,
            f"split/{split_name}",
            split_path,
        )
    else:
        if split_name not in input_snapshot.split_artifacts:
            raise RuntimeError(f"protocol-v2 snapshot is missing split {split_name}")
        manifest_digest = input_snapshot.manifest.sha256
        split_digest = input_snapshot.split_artifacts[split_name].sha256
        if lock_payload is not None:
            tracked = lock_payload.get("tracked_files")
            expected_snapshot_digests = {
                "artifact/corpus_manifest": manifest_digest,
                f"split/{split_name}": split_digest,
            }
            for label, observed in expected_snapshot_digests.items():
                row = tracked.get(label) if isinstance(tracked, Mapping) else None
                if not isinstance(row, Mapping) or row.get("sha256") != observed:
                    raise RuntimeError(f"locked v2 snapshot digest differs for {label}")
    receipt_digest = _bound_digest(
        lock_payload,
        "artifact/approval_semantics_receipt",
        receipt_path,
    )
    source_hashes = {
        name: _bound_digest(
            lock_payload,
            f"source/{name}",
            root / "src" / name,
        )
        for name in FIT_SOURCE_FILES_V2
    }
    corpus_semantic_sha256 = receipt.get("corpus_semantic_sha256")
    if (
        not isinstance(corpus_semantic_sha256, str)
        or len(corpus_semantic_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in corpus_semantic_sha256
        )
    ):
        raise RuntimeError("approval semantics receipt corpus digest differs")
    if read_regular_bytes_artifact_v2(
        receipt_path,
        label="approval semantics receipt",
    ) != receipt_bytes:
        raise RuntimeError("approval semantics receipt changed during provenance build")

    provenance: dict[str, object] = {
        "pabulib_commit": PABULIB_COMMIT,
        "corpus_manifest_sha256": manifest_digest,
        "split_sha256": split_digest,
        "source_sha256": source_hashes,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": receipt_digest,
        "corpus_semantic_sha256": corpus_semantic_sha256,
    }
    if lock_payload is not None:
        lock_path = result / "protocol_lock.json"
        lock_bytes = read_regular_bytes_artifact_v2(
            lock_path,
            label="protocol-v2 lock",
        )
        assert lock_bytes is not None
        if not exact_json_equal_v2(
            json.loads(lock_bytes.decode("utf-8")),
            lock_payload,
        ):
            raise RuntimeError("protocol-v2 lock payload changed before provenance build")
        provenance = {
            "protocol_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            **provenance,
        }
    return provenance


def assert_protocol_lock_unchanged_before_fit_write_v2(
    initial_snapshot: ProtocolLockSnapshotV2,
    *,
    repo_root: Path,
    result_root: Path,
    data_dir: Path,
) -> None:
    """Revalidate the full locked closure used for fit provenance."""

    lock_path = canonical_result_output_path_v2(
        result_root,
        Path("protocol_lock.json"),
    )
    final_snapshot = verify_multicity_protocol_lock_snapshot_v2(
        lock_path,
        repo_root,
        result_root,
        data_dir,
    )
    if (
        final_snapshot.path != initial_snapshot.path
        or final_snapshot.content != initial_snapshot.content
        or final_snapshot.sha256 != initial_snapshot.sha256
        or not exact_json_equal_v2(final_snapshot.payload, initial_snapshot.payload)
    ):
        raise RuntimeError("protocol-v2 locked closure changed before fit write")


def main(argv: Sequence[str] | None = None) -> None:
    """Fit one protocol-v2 training-only artifact on normalized ballots."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args, config = parse_train_args_v2(argv)
    lock_snapshot = authorize_fit_v2(
        config,
        smoke=args.smoke,
        result_root=RESULT_ROOT,
        repo_root=ROOT,
    )
    lock_payload = lock_snapshot.payload if lock_snapshot is not None else None
    result_root = canonical_result_root_v2(RESULT_ROOT)
    split_dir = _resolved_under(
        result_root / "splits",
        result_root,
        "split directory",
    )
    split_path = _resolved_under(
        split_dir / f"{config.split}.json",
        split_dir,
        "split path",
    )
    corpus_manifest = RESULT_ROOT / "corpus_manifest.json"
    semantics_receipt = RESULT_ROOT / "approval_semantics_receipt.json"
    if not all(
        path.is_file() for path in (split_path, corpus_manifest, semantics_receipt)
    ):
        raise RuntimeError(
            "frozen v2 corpus, split, and semantics receipt must exist before fitting"
        )
    input_snapshot = load_protocol_inputs_snapshot_v2(
        lock_payload,
        RESULT_ROOT,
        ROOT / "data" / "pb_multicity",
        (config.split,),
    )
    index = input_snapshot.index
    split = input_snapshot.splits[config.split]
    receipt_bytes = read_regular_bytes_artifact_v2(
        semantics_receipt,
        label="v2 semantics receipt",
    )
    assert receipt_bytes is not None
    receipt_payload = json.loads(receipt_bytes.decode("utf-8"))
    if not isinstance(receipt_payload, dict):
        raise RuntimeError("v2 semantics receipt must be a JSON object")
    validate_multicity_semantics_receipt(receipt_payload)
    validate_manifest_receipt_identities_v2(index, receipt_payload)
    data = load_training_series_data_v2(split, index, receipt_payload)
    if read_regular_bytes_artifact_v2(
        semantics_receipt,
        label="v2 semantics receipt",
    ) != receipt_bytes:
        raise RuntimeError("v2 semantics receipt changed during training load")
    payload = fit_one(config, data, EnvConfig())
    payload["schema_version"] = 2
    payload["provenance"] = build_fit_provenance_v2(
        lock_payload,
        result_root=RESULT_ROOT,
        repo_root=ROOT,
        split_name=config.split,
        input_snapshot=input_snapshot,
    )
    if args.smoke:
        output = canonical_result_output_path_v2(
            result_root,
            Path("smoke")
            / config.split
            / config.arm
            / f"seed-{config.seed}.json",
        )
    else:
        expected_output = fit_output_path(result_root, config)
        output = canonical_result_output_path_v2(
            result_root,
            expected_output.relative_to(result_root),
        )
    assert_protocol_inputs_unchanged_v2(input_snapshot)
    if read_regular_bytes_artifact_v2(
        semantics_receipt,
        label="v2 semantics receipt",
    ) != receipt_bytes:
        raise RuntimeError("v2 semantics receipt changed before fit write")
    if lock_snapshot is not None:
        provenance = payload.get("provenance")
        expected_lock_sha256 = (
            provenance.get("protocol_lock_sha256")
            if isinstance(provenance, Mapping)
            else None
        )
        if not isinstance(expected_lock_sha256, str):
            raise RuntimeError("fit provenance is missing the protocol-v2 lock digest")
        if expected_lock_sha256 != lock_snapshot.sha256:
            raise RuntimeError("fit provenance differs from the authenticated lock")
        assert_protocol_lock_unchanged_before_fit_write_v2(
            lock_snapshot,
            repo_root=ROOT,
            result_root=result_root,
            data_dir=ROOT / "data" / "pb_multicity",
        )
    digest = write_immutable_fit_v2(output, payload)
    logger.info("wrote protocol-v2 training-only fit %s", output)
    print(f"output={output}")
    print(f"sha256={digest}")
    print(f"best_loss={payload['result']['best_loss']:.8f}")


if __name__ == "__main__":
    main()
