"""Check the listed release bytes and Python syntax without running experiments."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root: Path) -> dict:
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    if manifest.get("schema") != "coverage-github-export-v1":
        raise ValueError("Unsupported release manifest")
    files = manifest["files"]
    if not files or "RELEASE_MANIFEST.json" in files:
        raise ValueError("Invalid release inventory")
    python_files = 0
    for name, record in files.items():
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != name or "\\" in name):
            raise ValueError("Unsafe manifest path")
        path = root
        for part in relative.parts:
            path /= part
            if path.is_symlink():
                raise ValueError(f"Symlink is not released content: {name}")
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed release file: {name}")
        if path.suffix == ".py":
            ast.parse(raw, filename=name)
            python_files += 1
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and p.relative_to(root).as_posix()
                   not in set(files) | {"RELEASE_MANIFEST.json"}
                   and not any(part in {".git", ".venv", "__pycache__", ".pytest_cache"}
                               for part in p.relative_to(root).parts))
    return {"status": "pass", "listed_files": len(files),
            "python_files_parsed": python_files, "unlisted_files": extra,
            "scope": "listed-file integrity and syntax, not experimental replication"}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
