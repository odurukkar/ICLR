"""Saved-package verifier guards, using only synthetic file inventories."""
from __future__ import annotations

import hashlib
import json

import pytest

import verify_iclr_saved_supplement as verifier


def fixture_package(root):
    raw = b"saved evidence\n"
    (root / "record.json").write_bytes(raw)
    payload = {"schema": "iclr-saved-results-supplement-v1", "files": {
        "record.json": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}}
    (root / "MANIFEST.json").write_text(json.dumps(payload))
    return payload


@pytest.mark.parametrize("name", ["", "/absolute", "../outside", "a/../../b", "a\\b", "a//b", "./a"])
def test_unsafe_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        verifier.safe_path(tmp_path, name)


def test_symlink_parent_rejected(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verifier.safe_path(tmp_path, "linked/file.json")


def test_intact_inventory(tmp_path):
    fixture_package(tmp_path)
    assert verifier.verify_inventory(tmp_path) == 1


@pytest.mark.parametrize("change", ["content", "missing", "extra", "size", "schema", "raw", "symlink", "bool_size"])
def test_inventory_rejects_changes(tmp_path, change):
    payload = fixture_package(tmp_path)
    if change == "content": (tmp_path / "record.json").write_bytes(b"altered")
    elif change == "missing": (tmp_path / "record.json").unlink()
    elif change == "extra": (tmp_path / "unexpected").write_bytes(b"x")
    elif change == "size": payload["files"]["record.json"]["bytes"] = 0
    elif change == "bool_size": payload["files"]["record.json"]["bytes"] = True
    elif change == "schema": payload["schema"] = "wrong"
    elif change == "raw":
        (tmp_path / "ballot.pb").write_bytes((tmp_path / "record.json").read_bytes())
        payload["files"]["ballot.pb"] = payload["files"]["record.json"]
    elif change == "symlink": (tmp_path / "link").symlink_to(tmp_path / "record.json")
    (tmp_path / "MANIFEST.json").write_text(json.dumps(payload))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        verifier.verify_inventory(tmp_path)


def test_import_cache_is_not_treated_as_evidence(tmp_path):
    fixture_package(tmp_path)
    cache = tmp_path / "__pycache__"; cache.mkdir()
    (cache / "module.pyc").write_bytes(b"local import cache")
    assert verifier.verify_inventory(tmp_path) == 1
    (cache / "hidden.pb").write_bytes(b"not allowed")
    with pytest.raises(ValueError): verifier.verify_inventory(tmp_path)


def test_uv_environment_is_not_a_packaged_artifact(tmp_path):
    fixture_package(tmp_path)
    env = tmp_path / "tempopb/.venv"; env.mkdir(parents=True)
    (env / "python").symlink_to("/usr/bin/python3")
    assert verifier.verify_inventory(tmp_path) == 1


@pytest.mark.parametrize("digest", [None, "no digest", "A" * 64, "0" * 64])
def test_hash_checks_fail_closed(tmp_path, digest):
    (tmp_path / "file").write_bytes(b"x")
    with pytest.raises(ValueError): verifier.check_hash(tmp_path, "file", digest)
