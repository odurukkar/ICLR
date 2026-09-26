"""Saved-supplement packaging tests; no fitting, replay, or statistical calls."""
import io
import json
from pathlib import Path
import stat
import warnings
import zipfile

import pytest

import build_iclr_saved_supplement as builder


@pytest.mark.parametrize("name", ("", ".", "..", "/absolute", "a/../b", "a/./b", "a//b",
                                  "a\\b", "C:/file", "a\x00b", "a/", "raw.pb", "RAW.PB"))
def test_unsafe_or_raw_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        builder.safe_name(name)


def test_unicode_names_and_omitted_raw_inventory_paths_are_supported() -> None:
    assert builder.safe_name("results/Łódź.json") == "results/Łódź.json"
    assert builder.safe_name("data/ballots.pb", allow_raw=True) == "data/ballots.pb"


def test_path_content_check_does_not_redact() -> None:
    content = ("/" + "Users/" + "private_person/file.csv").encode()
    with pytest.raises(ValueError, match="no redaction"):
        builder.check_content(content, "receipt.json")


def test_synthetic_fixture_allowance_is_file_and_prefix_specific() -> None:
    for member, prefix in builder.SYNTHETIC_PATH_FIXTURES.items():
        builder.check_content(prefix.encode(), member)
        builder.check_content(prefix.encode(), "tempopb/" + member)
        with pytest.raises(ValueError):
            builder.check_content(prefix.encode(), "results/receipt.json")
        with pytest.raises(ValueError):
            builder.check_content(("/" + "Users/other_person/file").encode(), member)


def test_exact_bytes_and_expected_digest(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    raw = b'{"value":1}\r\n'
    path.write_bytes(raw)
    inventory = builder.Inventory(tmp_path)
    assert inventory.file("data.json", builder.sha(raw)) == raw
    assert inventory.files["data.json"] == raw
    with pytest.raises(ValueError, match="digest mismatch"):
        inventory.file("data.json", "0" * 64)


def test_symlink_file_and_parent_are_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "data.json").write_text("{}")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    (tmp_path / "file.json").symlink_to(real / "data.json")
    inventory = builder.Inventory(tmp_path)
    for name in ("linked/data.json", "file.json"):
        with pytest.raises(ValueError, match="symlink"):
            inventory.file(name)


def test_conflicts_fail_except_documented_historical_replacements(tmp_path: Path) -> None:
    inventory = builder.Inventory(tmp_path)
    name = "tempopb/results/table.csv"
    inventory.add(name, b"old", builder.HISTORICAL)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        inventory.add(name, b"current", "authenticated binding")
    inventory.add(name, b"current", "authenticated binding", replace_historical=True)
    assert inventory.replacements == [{"path": name, "historical_sha256": builder.sha(b"old"),
                                        "current_sha256": builder.sha(b"current"),
                                        "authenticated_by": "authenticated binding"}]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        inventory.add(name, b"another", "another binding", replace_historical=True)
    inventory.add("tempopb/src/code.py", b"old", builder.HISTORICAL)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        inventory.add("tempopb/src/code.py", b"new", "current", replace_historical=True)


@pytest.mark.parametrize("case", ("duplicate", "symlink", "raw", "traversal"))
def test_unsafe_zip_members_fail(case: str) -> None:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as archive:
        if case == "duplicate":
            archive.writestr("x", b"a")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("x", b"b")
        elif case == "symlink":
            info = zipfile.ZipInfo("link")
            info.create_system, info.external_attr = 3, (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "destination")
        else:
            archive.writestr("data/raw.pb" if case == "raw" else "../escape", b"a")
    with pytest.raises(ValueError):
        builder.zip_members(raw.getvalue())


def test_stable_zip_is_sorted_and_contains_regular_files() -> None:
    files = {"z": b"last", "Łódź": b"unicode", "a": b"first"}
    first = builder.zip_bytes(files)
    assert builder.zip_bytes(dict(reversed(list(files.items())))) == first
    assert builder.zip_members(first) == files
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == sorted(files)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert all(stat.S_ISREG(info.external_attr >> 16) for info in archive.infolist())


def test_literal_pin_reader_does_not_execute_source() -> None:
    raw = b'raise RuntimeError("must not execute")\nPIN = "abc"\nRELATIVE = Path("output/file.zip")\n'
    assert builder.constants(raw, ("PIN", "RELATIVE")) == {"PIN": "abc", "RELATIVE": "output/file.zip"}
    with pytest.raises(ValueError, match="pins missing"):
        builder.constants(raw, ("MISSING",))


def test_import_closure_is_recursive_and_does_not_execute(tmp_path: Path) -> None:
    directory = tmp_path / "tempopb/src"
    directory.mkdir(parents=True)
    for name, source in {"entry": "import child\n", "child": "def unused():\n    from leaf import x\n",
                         "leaf": "raise RuntimeError('never execute')\nx = 1\n"}.items():
        (directory / f"{name}.py").write_text(source)
    inventory = builder.Inventory(tmp_path)
    inventory.file("tempopb/src/entry.py")
    builder.import_closure(inventory)
    assert set(inventory.files) == {f"tempopb/src/{name}.py" for name in ("entry", "child", "leaf")}


def test_default_build_is_dry_and_write_is_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "collect", lambda workspace: {"a": b"saved"})
    report = builder.build(tmp_path)
    output = tmp_path / builder.OUTPUT
    assert report["written"] is False and not output.exists()
    output.parent.mkdir(parents=True)
    result = builder.build(tmp_path, write=True)
    assert result["written"] is True and builder.sha(output.read_bytes()) == result["sha256"]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        builder.build(tmp_path, write=True)
    assert output.read_bytes() == original


def test_actual_saved_inventory_is_complete_and_pinned() -> None:
    files = builder.collect()
    manifest = json.loads(files["MANIFEST.json"])
    assert manifest["schema"] == "iclr-saved-results-supplement-v1"
    assert set(manifest["files"]) == set(files) - {"MANIFEST.json"}
    assert all(row == {"sha256": builder.sha(files[name]), "bytes": len(files[name])}
               for name, row in manifest["files"].items())
    assert builder.sha(files["manuscript/paper.pdf"]) == builder.FINAL_PDF_SHA
    assert "manuscript/REPRODUCE.md" in files and "manuscript/tex/main.tex" in files
    assert not any(Path(name).suffix.lower() == ".pb" for name in files)
    assert not any("/smoke/" in name or "/fit_logs/" in name for name in files if "/iclr_multicity_v2/" in name)
    omitted = json.loads(files["EXCLUDED_RAW_DATA.json"])
    assert omitted["schema"] == "omitted-public-ballot-files-v1"
    assert len(omitted["files"]) == 529
    assert "not rechecked" in omitted["disclosure"] and "pseudonymous" in omitted["disclosure"]
    for family in (builder.PRIMARY, builder.MULTICITY):
        lock = json.loads(files[family + "/protocol_lock.json"])
        for row in lock["tracked_files"].values():
            name = "tempopb/" + row["path"]
            if name.endswith(".pb"):
                assert omitted["files"][name]["sha256"] == row["sha256"]
            else:
                assert builder.sha(files[name]) == row["sha256"]
    assert len([name for name in files if name.startswith(builder.MULTICITY + "/fits/")]) == 24
    assert len([name for name in files if name.startswith(builder.PRIMARY + "/fits/")]) == 49
    assert "tempopb/src/build_iclr_saved_supplement.py" in files
    assert "tempopb/src/verify_iclr_saved_supplement.py" in files
    assert "tempopb/tests/test_iclr_saved_verification.py" in files
