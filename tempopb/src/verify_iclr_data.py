"""Verify local Pabulib files against the frozen ICLR corpus manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/pb"))
    parser.add_argument("--manifest", type=Path, default=Path("DATA_MANIFEST.tsv"))
    args = parser.parse_args()
    failures = []
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        path = args.data / row["filename"]
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif path.stat().st_size != int(row["bytes"]):
            failures.append(f"size mismatch: {path}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"hash mismatch: {path}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"DATA GATE PASSED: {len(rows)}/{len(rows)} files match size and SHA-256")


if __name__ == "__main__":
    main()
