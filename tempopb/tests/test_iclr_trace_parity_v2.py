"""Corrected parity input, fail-closed execution, and immutable report tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _adapter():
    assert importlib.util.find_spec("iclr_trace_parity_v2") is not None, "corrected parity adapter is missing"
    return importlib.import_module("iclr_trace_parity_v2")


def _tiny(*, year=2023):
    from parse_pb import PBInstance, Project, Vote
    return PBInstance(
        path=f"test://corrected-parity/{year}#approval-set-first-occurrence-v2",
        meta={"budget": "10", "vote_type": "approval", "date_begin": f"{year}-01-01"},
        projects={"a": Project("a", 10.0, None)},
        votes=[Vote("1", ("a",), age=70, sex="F"), Vote("2", ("a",), age=30, sex="M")],
    )


def test_corrected_election_is_deeply_immutable_and_has_separate_raw_semantic_hashes():
    adapter = _adapter()
    instance = _tiny()
    election = adapter.freeze_election("Poland/Test/A", 2023, Path("raw.pb"), "a" * 64, instance, "scored")
    assert election.raw_sha256 == "a" * 64
    assert election.semantic_sha256 != election.raw_sha256
    assert election.materialize().votes == tuple(instance.votes)
    instance.meta["budget"] = "999"
    instance.votes.clear()
    assert election.materialize().budget == 10.0
    assert len(election.materialize().votes) == 2
    with pytest.raises(FrozenInstanceError):
        election.year = 2000
    with pytest.raises(FrozenInstanceError):
        election.votes[0].age = 1


@pytest.mark.parametrize("mutation", ["schema", "fits", "profile", "structural", "timestamp", "extra"])
def test_receipt_validation_rejects_wrong_schema_or_binding(mutation):
    adapter = _adapter()
    receipt = json.loads((ROOT / "results/iclr_multicity_v2/heldout_opened.json").read_bytes())
    expected = copy.deepcopy(receipt)
    if mutation == "schema":
        receipt["schema_version"] = 1
    elif mutation == "fits":
        receipt["fit_sha256"].pop(next(iter(receipt["fit_sha256"])))
    elif mutation == "profile":
        receipt["semantics_profile"] = "legacy"
    elif mutation == "structural":
        receipt["structural_gates_sha256"] = "0" * 64
    elif mutation == "timestamp":
        receipt["opened_at_utc"] = "yesterday"
    else:
        receipt["extra"] = True
    with pytest.raises(RuntimeError, match="receipt"):
        adapter.validate_opening_receipt(
            receipt, lock_sha256=expected["lock_sha256"], fit_sha256=expected["fit_sha256"],
            semantics_sha256=expected["semantics_receipt_sha256"],
            corpus_semantic_sha256=expected["corpus_semantic_sha256"],
            structural_sha256=expected["structural_gates_sha256"],
        )


def test_real_corrected_authentication_and_all_stage_counts_without_solver():
    adapter = _adapter()
    context = adapter.load_authenticated_context(ROOT)
    assert len(context.elections) == 397
    assert len(context.fits) == 6
    provenance = json.loads(context.provenance_json)
    assert provenance["fit_inventory_count"] == 24
    assert provenance["classification"] == "corrected post-hoc replay"
    assert provenance["primary_delta_sha256"] == "f380ef80a1da1fecd8bb2b44dd4a46e09414827fc6e26524a927a4a6740ceeb8"
    assert len({(r.series, r.year) for r in context.elections}) == 397
    wola = next(r for r in context.elections if r.series == "Poland/Warszawa/Wola" and r.year == 2021)
    assert wola.votes[2976].projects == ("147", "1040")
    assert wola.raw_sha256 == "9054e7b28c97b18a990d75baaea5777942a7703774863944324567fbec60f61a"
    assert wola.semantic_sha256 == "495b6b54e9049e62554f2f433b0cf14ee2fb7b2dd3279134a9fcfbafda37b312"
    for stage, expected in (("warsaw-sentinels", (10, 6, 4)), ("representative-temporal-scored", (6, 6, 0)), ("temporal-scored", (446, 446, 0)), ("native", (794, 446, 348)), ("full", (3176, 1784, 1392))):
        assert adapter.stage_counts(context, stage) == expected
    adapter.assert_context_unchanged(context)
    unexecuted = adapter.failure_report("native", ValueError("authentication is not execution"))
    with pytest.raises(RuntimeError, match="execution"):
        adapter._validate_execution_report(unexecuted, context)
    with pytest.raises(FrozenInstanceError):
        context.context_sha256 = "0" * 64


def test_invalid_stage_is_rejected():
    adapter = _adapter()
    with pytest.raises(ValueError, match="stage"):
        adapter.selected_long_stages("nativ")
    assert adapter.selected_long_stages("") == ()
    assert adapter.selected_long_stages("all") == ("temporal-scored", "native", "full")


def _tiny_elections(adapter):
    return tuple(adapter.freeze_election("Poland/Test/A", year, Path(f"{year}.pb"), "a" * 64, _tiny(year=year), "scored") for year in (2023, 2024))


def test_tiny_real_three_kernel_execution_emits_progress_and_keeps_timing_separate():
    adapter = _adapter()
    assert hasattr(adapter, "_run_cases"), "fail-closed corrected execution is missing"
    progress = []
    stats, timing = adapter._run_cases(adapter._uniform_cases(_tiny_elections(adapter), "synthetic"), stage="synthetic", expected_counts=(4, 4, 0), progress=progress.append)
    assert stats["cases_attempted"] == stats["cases_compared"] == stats["cases_exact"] == 4
    assert stats["status"] == "pass" and stats["failure"] is None
    assert stats["completion_counts"] == {"false": 2, "true": 2}
    assert "elapsed_seconds" not in stats and timing["elapsed_seconds"] >= 0.0
    assert [row["case_index"] for row in progress if row["event"] == "case-start"] == [1, 2, 3, 4]
    assert progress[-1]["cases_compared"] == 4
    assert progress[-1]["eta_seconds"] == 0.0


@pytest.mark.parametrize("mode", ["mismatch", "exception"])
def test_first_failure_stops_execution_and_does_not_fabricate_a_three_path_verdict(monkeypatch, mode):
    adapter = _adapter()
    assert hasattr(adapter, "_run_cases"), "fail-closed corrected execution is missing"
    from iclr_trace_parity import ThreePathParity
    calls = []
    def compare(instance, endowments, *, completion, instance_sha256):
        calls.append(instance.path)
        if len(calls) == 2:
            if mode == "exception":
                raise ValueError("payer rows must have unique voter indices")
            return ThreePathParity(instance_sha256, completion, ("a",), (), ("a",))
        return ThreePathParity(instance_sha256, completion, ("a",), ("a",), ("a",))
    monkeypatch.setattr(adapter, "compare_three_paths", compare)
    stats, _ = adapter._run_cases(adapter._uniform_cases(_tiny_elections(adapter), "synthetic"), stage="synthetic", expected_counts=(4, 4, 0), progress=lambda row: None)
    assert len(calls) == stats["cases_attempted"] == 2
    assert stats["cases_exact"] == 1
    assert stats["status"] == ("mismatch" if mode == "mismatch" else "error")
    assert stats["cases_compared"] == (2 if mode == "mismatch" else 1)
    assert stats["failure"]["three_path_exact"] is (False if mode == "mismatch" else None)
    assert stats["failure"]["case_index"] == 2
    assert next(adapter._uniform_cases(_tiny_elections(adapter), "synthetic")).election.year == 2023


def test_fitted_rollout_resets_history_for_each_fit_series_and_completion(monkeypatch):
    adapter = _adapter()
    assert hasattr(adapter, "_fitted_cases"), "corrected fitted rollout is missing"
    elections = _tiny_elections(adapter)
    elections += tuple(replace(row, series="Poland/Test/B") for row in elections)
    fits = tuple(adapter.EndowmentFit(f"split/endowment/seed-{seed}", "split", seed, "b" * 64, (float(seed), 0., 0., 0., 0., 0.)) for seed in (1, 2))
    context = adapter.AuthenticatedContext(ROOT, elections, fits, (("split", (("Poland/Test/A", (2024,)), ("Poland/Test/B", (2024,)))),), (), b"{}", "")
    observed = []
    def policy(weights, *, scheme):
        assert scheme == "age_sex"
        def apply(instance, state):
            observed.append((weights[0], state.year_index, dict(state.entitlements)))
            return [5., 5.]
        return apply
    monkeypatch.setattr(adapter, "linear_policy", policy)
    stats, _ = adapter._run_cases(adapter._fitted_cases(context), stage="synthetic", expected_counts=(16, 8, 8), progress=lambda row: None)
    assert stats["status"] == "pass" and stats["cases_compared"] == 16
    assert [row[1] for row in observed] == [0, 1] * 8
    assert all(not row[2] for row in observed if row[1] == 0)
    assert all(row[2] for row in observed if row[1] == 1)


def test_next_policy_failure_is_attributed_to_the_next_election(monkeypatch):
    adapter = _adapter()
    elections = _tiny_elections(adapter)
    fit = adapter.EndowmentFit("split/endowment/seed-42", "split", 42, "b" * 64, (1., 0., 0., 0., 0., 0.))
    context = adapter.AuthenticatedContext(ROOT, elections, (fit,), (("split", (("Poland/Test/A", (2024,)),)),), (), b"{}", "")
    def policy(weights, *, scheme):
        def apply(instance, state):
            if state.year_index == 1:
                raise ValueError("second election policy failed")
            return [5., 5.]
        return apply
    monkeypatch.setattr(adapter, "linear_policy", policy)
    stats, _ = adapter._run_cases(adapter._fitted_cases(context), stage="synthetic", expected_counts=(4, 2, 2), progress=lambda row: None)
    assert stats["cases_compared"] == 1
    assert stats["cases_attempted"] == 2
    assert stats["failure"]["year"] == 2024 and stats["failure"]["case_index"] == 2
    assert stats["failure"]["three_path_exact"] is None


def test_count_shortfall_never_qualifies_as_pass():
    adapter = _adapter()
    assert hasattr(adapter, "_run_cases"), "corrected count gate is missing"
    stats, _ = adapter._run_cases(adapter._uniform_cases(_tiny_elections(adapter), "synthetic"), stage="synthetic", expected_counts=(6, 6, 0), progress=lambda row: None)
    assert stats["status"] == "error"
    assert stats["failure"]["kind"] == "count_mismatch"


def test_repeated_single_completion_mode_cannot_pass_total_count_gate():
    adapter = _adapter()
    def repeated_mode():
        for case in adapter._uniform_cases(_tiny_elections(adapter), "synthetic"):
            yield replace(case, completion=False)
    stats, _ = adapter._run_cases(repeated_mode(), stage="synthetic", expected_counts=(4, 4, 0), progress=lambda row: None)
    assert stats["status"] == "error"
    assert stats["failure"]["kind"] == "completion_count_mismatch"


def test_sealed_file_and_materialized_semantic_tampering_are_rejected(tmp_path):
    adapter = _adapter()
    path = tmp_path / "source.py"
    path.write_bytes(b"original")
    elections = _tiny_elections(adapter)
    context = adapter.AuthenticatedContext(tmp_path, elections, (), (), (("repo/source.py", hashlib.sha256(b"original").hexdigest()),), b"{}", "")
    context = replace(context, context_sha256=adapter._fingerprint(context))
    adapter.assert_context_unchanged(context)
    changed = replace(context, elections=(replace(elections[0], meta=(("budget", "99"),)), elections[1]))
    with pytest.raises(RuntimeError, match="semantics"):
        adapter.assert_context_unchanged(changed)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        adapter.assert_context_unchanged(context)


def test_immutable_failure_reports_keep_certificate_and_branch_claims_false(tmp_path):
    adapter = _adapter()
    assert hasattr(adapter, "publish_report"), "immutable corrected report publication is missing"
    report = adapter.failure_report("native", ValueError("bad input"))
    assert report["gate_passed"] is False and report["three_path_exact"] is None
    assert report["float_runtime_proved"] is False and report["cache_safe"] is False
    assert report["literal_branch_equivalence_claimed"] is False
    output = tmp_path / "gate_report.json"
    adapter.publish_report(output, report, {"elapsed_seconds": 1.0})
    assert json.loads(output.read_bytes()) == report
    assert json.loads((tmp_path / "gate_report.timing.json").read_bytes())["elapsed_seconds"] == 1.0
    with pytest.raises(RuntimeError, match="exist|conflict|immutable"):
        adapter.publish_report(output, report, {"elapsed_seconds": 2.0})
    assert json.loads((tmp_path / "gate_report.timing.json").read_bytes())["elapsed_seconds"] == 1.0


def test_fabricated_context_cannot_publish_a_passing_gate(tmp_path):
    adapter = _adapter()
    context = adapter.AuthenticatedContext(tmp_path, (), (), (), (), b"{}", "")
    context = replace(context, context_sha256=adapter._fingerprint(context))
    report = adapter.failure_report("native", ValueError("not actually run"))
    report.update(status="pass", gate_passed=True, context_sha256=context.context_sha256)
    with pytest.raises(RuntimeError, match="authentic|factory|execution|gate"):
        adapter.publish_report(tmp_path / "gate_report.json", report, {}, context)
    assert not (tmp_path / "gate_report.json").exists()
    assert not (tmp_path / "gate_report.timing.json").exists()


@pytest.mark.parametrize("suffix", [".json", ".timing.json"])
def test_either_report_conflict_prevents_all_writes(tmp_path, suffix):
    adapter = _adapter()
    assert hasattr(adapter, "publish_report"), "immutable corrected report publication is missing"
    conflict = tmp_path / ("gate_report" + suffix)
    conflict.write_bytes(b"existing")
    other = tmp_path / ("gate_report.timing.json" if suffix == ".json" else "gate_report.json")
    with pytest.raises(RuntimeError):
        adapter.publish_report(tmp_path / "gate_report.json", adapter.failure_report("native", ValueError("bad")), {})
    assert conflict.read_bytes() == b"existing" and not other.exists()


def test_cli_emits_flushed_checkpoints_and_nonzero_exit_for_failure(monkeypatch, tmp_path, capsys):
    adapter = _adapter()
    assert hasattr(adapter, "main"), "corrected stage CLI is missing"
    def run(stage, repo_root, progress):
        progress({"event": "case-start", "case_index": 1})
        return adapter.failure_report(stage, ValueError("synthetic failure")), {"elapsed_seconds": 0.0}, None
    monkeypatch.setattr(adapter, "run_authenticated_stage", run)
    output = tmp_path / "report.json"
    assert adapter.main(["--stage", "native", "--output", str(output)]) == 1
    assert json.loads(output.read_bytes())["gate_passed"] is False
    assert '"case_index": 1' in capsys.readouterr().out


def test_cli_existing_output_and_second_writer_stop_before_authentication(monkeypatch, tmp_path):
    adapter = _adapter()
    def unexpected(*args, **kwargs):
        raise AssertionError("authentication/solver work started before output preflight")
    monkeypatch.setattr(adapter, "run_authenticated_stage", unexpected)
    output = tmp_path / "gate_report.json"
    with adapter.report_writer(output):
        with pytest.raises(RuntimeError, match="single writer"):
            adapter.main(["--stage", "native", "--output", str(output)])
    output.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="already exists"):
        adapter.main(["--stage", "native", "--output", str(output)])
    assert output.read_bytes() == b"existing"


@pytest.mark.parametrize("stage", ("temporal-scored", "native", "full"))
def test_opt_in_long_corrected_stage(stage):
    adapter = _adapter()
    if stage not in adapter.selected_long_stages(os.environ.get("ICLR_TRACE_PARITY_V2_STAGE", "")):
        pytest.skip("explicit corrected parity long-stage opt-in required")
    report, _, _ = adapter.run_authenticated_stage(stage, ROOT, lambda row: print(json.dumps(row), flush=True))
    assert report["gate_passed"] is True
