"""Three-path parity gates for traced nonuniform Equal Shares."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from parse_pb import PBInstance, Project, Vote
from iclr_trace_parity import selected_long_stages


ROOT = Path(__file__).resolve().parent.parent
OPT_IN_STAGE = os.environ.get("ICLR_TRACE_PARITY_STAGE", "")
ENABLED_LONG_STAGES = selected_long_stages(OPT_IN_STAGE)


def _tiny_instance(*, budget: str = "10") -> PBInstance:
    return PBInstance(
        path="test://trace-parity/tiny",
        meta={"budget": budget, "date_begin": "2024-01-01"},
        projects={"a": Project("a", 10.0, None)},
        votes=[Vote("0", ("a",)), Vote("1", ("a",))],
    )


def test_common_wrapper_returns_an_immutable_exact_three_path_result() -> None:
    from iclr_trace_parity import compare_three_paths

    observed = compare_three_paths(
        _tiny_instance(),
        (5.0, 5.0),
        completion=False,
    )

    assert observed.reference == observed.trace == observed.production == ("a",)
    assert observed.exact is True
    with pytest.raises(FrozenInstanceError):
        observed.reference = ()


@pytest.mark.parametrize(
    ("endowments", "message"),
    (
        ((10.0,), "one endowment"),
        ((5.0, math.nan), "finite"),
        ((5.0, math.inf), "finite"),
        ((11.0, -1.0), "nonnegative"),
        ((4.0, 4.0), "municipal budget"),
    ),
)
def test_common_wrapper_rejects_malformed_endowments_before_any_solver(
    monkeypatch: pytest.MonkeyPatch,
    endowments: tuple[float, ...],
    message: str,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("a solver was called before common validation")

    monkeypatch.setattr(iclr_trace_parity, "mes_with_endowments", unexpected)
    monkeypatch.setattr(iclr_trace_parity, "trace_equal_shares", unexpected)
    monkeypatch.setattr(iclr_trace_parity, "fast_mes_with_endowments", unexpected)

    with pytest.raises(ValueError, match=message):
        iclr_trace_parity.compare_three_paths(
            _tiny_instance(), endowments, completion=False
        )


@pytest.mark.parametrize("budget", ("0", "-1", "nan", "inf"))
def test_nonempty_instance_requires_a_positive_finite_budget(budget: str) -> None:
    from iclr_trace_parity import compare_three_paths

    with pytest.raises(ValueError, match="positive and finite"):
        compare_three_paths(_tiny_instance(budget=budget), (5.0, 5.0), completion=False)


def test_budget_sum_uses_the_production_tolerance_boundary() -> None:
    from iclr_trace_parity import compare_three_paths

    # Production accepts absolute error <= 1e-6 * max(1, abs(budget)).
    within = (5.0, 5.0 - 5e-6)
    outside = (5.0, 5.0 - 2e-5)

    assert compare_three_paths(
        _tiny_instance(), within, completion=False
    ).exact
    with pytest.raises(ValueError, match="municipal budget"):
        compare_three_paths(_tiny_instance(), outside, completion=False)


def test_empty_instance_has_an_explicit_empty_endowment_boundary() -> None:
    from iclr_trace_parity import compare_three_paths

    inst = PBInstance(path="test://trace-parity/empty", meta={"budget": "0"})
    observed = compare_three_paths(inst, (), completion=True)

    assert observed.reference == observed.trace == observed.production == ()


@pytest.mark.parametrize("budget", ("1", "-1", "nan", "inf", "garbage"))
def test_invalid_empty_instance_is_rejected_before_any_solver(
    monkeypatch: pytest.MonkeyPatch,
    budget: str,
) -> None:
    import iclr_trace_parity

    calls: list[str] = []

    def unexpected(*args, **kwargs):
        calls.append("solver")
        raise AssertionError("solver called for an invalid empty instance")

    monkeypatch.setattr(iclr_trace_parity, "mes_with_endowments", unexpected)
    monkeypatch.setattr(iclr_trace_parity, "trace_equal_shares", unexpected)
    monkeypatch.setattr(iclr_trace_parity, "fast_mes_with_endowments", unexpected)
    inst = PBInstance(
        path="test://trace-parity/invalid-empty", meta={"budget": budget}
    )

    with pytest.raises(ValueError, match="empty instance.*zero budget"):
        iclr_trace_parity.compare_three_paths(inst, (), completion=False)
    assert calls == []


def test_authenticated_instance_digest_is_carried_without_rereading_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    expected = "a" * 64

    def unexpected(*args, **kwargs):
        raise AssertionError("authenticated compare recomputed the instance digest")

    monkeypatch.setattr(iclr_trace_parity, "_canonical_instance_sha256", unexpected)
    observed = iclr_trace_parity.compare_three_paths(
        _tiny_instance(),
        (5.0, 5.0),
        completion=False,
        instance_sha256=expected,
    )

    assert observed.instance_sha256 == expected


def test_invalid_carried_instance_digest_is_rejected_before_any_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("solver called with an invalid instance digest")

    monkeypatch.setattr(iclr_trace_parity, "mes_with_endowments", unexpected)
    monkeypatch.setattr(iclr_trace_parity, "trace_equal_shares", unexpected)
    monkeypatch.setattr(iclr_trace_parity, "fast_mes_with_endowments", unexpected)

    with pytest.raises(ValueError, match="instance SHA-256"):
        iclr_trace_parity.compare_three_paths(
            _tiny_instance(),
            (5.0, 5.0),
            completion=False,
            instance_sha256="not-a-digest",
        )


def test_endowments_are_materialized_exactly_once() -> None:
    from iclr_trace_parity import compare_three_paths

    seen: list[float] = []

    def values():
        for value in (5.0, 5.0):
            seen.append(value)
            yield value

    assert compare_three_paths(
        _tiny_instance(), values(), completion=False
    ).exact
    assert seen == [5.0, 5.0]


def test_witness_is_canonicalized_outcome_equivalent_not_literal_loop_order() -> None:
    from iclr_residual_actuation import trace_equal_shares
    from iclr_trace_parity import compare_three_paths

    # The reference visits ``list(active)``.  The witness deliberately visits
    # the same round candidate multiset in sorted order, then preserves the
    # selected (rho, cost, pid), payment state, completion order, and winners.
    inst = PBInstance(
        path="test://trace-parity/canonicalized-witness",
        meta={"budget": "5"},
        projects={
            "b": Project("b", 5.0, None),
            "a": Project("a", 5.0, None),
        },
        votes=[Vote("0", ("b", "a"))],
    )
    witness = trace_equal_shares(inst, (5.0,), completion=False)
    first_round = tuple(
        step.project
        for step in witness.steps
        if step.phase == "payment-candidate"
    )[:2]

    assert first_round == ("a", "b")
    assert compare_three_paths(inst, (5.0,), completion=False).exact
    assert witness.winners == ("a",)


def test_partial_diagnostics_are_unmistakably_not_a_hard_gate() -> None:
    from iclr_trace_parity import ParityCase, audit_partial_diagnostics

    report = audit_partial_diagnostics(
        (
            ParityCase(
                instance=_tiny_instance(),
                endowments=(5.0, 5.0),
                completion=False,
                context="unit/uniform",
                phase="scored",
            ),
        ),
        stage="synthetic",
    )

    assert report.cases == 1
    assert report.scored_cases == 1
    assert report.warmup_cases == 0
    assert report.mismatches == ()
    assert report.elapsed_seconds >= 0.0
    assert report.peak_rss_bytes > 0
    assert not hasattr(report, "traces")
    assert report.gate_eligible is False
    assert report.gate_passed is False


def test_case_audit_stops_on_first_mismatch_with_bounded_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity
    from iclr_trace_parity import (
        ParityCase,
        ParityMismatchError,
        audit_partial_diagnostics,
    )

    monkeypatch.setattr(
        iclr_trace_parity,
        "fast_mes_with_endowments",
        lambda *args, **kwargs: set(),
    )
    cases = tuple(
        ParityCase(
            instance=_tiny_instance(),
            endowments=(5.0, 5.0),
            completion=False,
            context=f"unit/{index}",
            phase="warmup",
        )
        for index in range(20)
    )

    with pytest.raises(ParityMismatchError, match="unit/0") as raised:
        audit_partial_diagnostics(cases, stage="synthetic")

    assert raised.value.report.cases == 1
    assert len(raised.value.report.mismatches) == 1
    assert raised.value.report.mismatches[0].production == ()


def test_hard_gate_apis_require_both_completion_modes_before_inputs() -> None:
    from iclr_trace_parity import (
        run_authenticated_frozen_endowment_fits,
        run_authenticated_uniform_stage,
    )

    with pytest.raises(ValueError, match="exactly.*False.*True"):
        run_authenticated_uniform_stage(
            "native", repo_root=ROOT, completion_modes=(False,)
        )
    with pytest.raises(ValueError, match="exactly.*False.*True"):
        run_authenticated_frozen_endowment_fits(
            repo_root=ROOT, completion_modes=(True,)
        )
    with pytest.raises(ValueError, match="exactly.*False.*True"):
        run_authenticated_uniform_stage(
            "native", repo_root=ROOT, completion_modes=(0, 1)
        )


def _valid_receipt() -> tuple[dict[str, object], dict[str, str]]:
    fit_hashes = {f"fits/fit-{index}.json": f"{index:064x}" for index in range(24)}
    return (
        {
            "schema_version": 1,
            "lock_sha256": "a" * 64,
            "fit_sha256": fit_hashes.copy(),
            "structural_gates_sha256": "b" * 64,
            "opened_at_utc": "2026-08-11T18:58:19.925183+00:00",
        },
        fit_hashes,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda row: row.update(schema_version=2), "schema"),
        (lambda row: row.update(lock_sha256="c" * 64), "lock"),
        (
            lambda row: row.update(structural_gates_sha256="d" * 64),
            "structural",
        ),
        (
            lambda row: row["fit_sha256"].update({"fits/fit-0.json": "f" * 64}),
            "fit",
        ),
        (lambda row: row.update(lock_sha25="typo"), "fields"),
    ),
)
def test_receipt_tamper_or_typo_is_rejected(mutation, message: str) -> None:
    from iclr_trace_parity import _verify_heldout_receipt

    receipt, fit_hashes = _valid_receipt()
    mutation(receipt)

    with pytest.raises(RuntimeError, match=message):
        _verify_heldout_receipt(
            receipt,
            lock_sha256="a" * 64,
            fit_sha256=fit_hashes,
            structural_gates_sha256="b" * 64,
        )


def test_receipt_requires_exactly_all_24_fit_hashes() -> None:
    from iclr_trace_parity import _verify_heldout_receipt

    receipt, fit_hashes = _valid_receipt()
    receipt["fit_sha256"].pop("fits/fit-23.json")

    with pytest.raises(RuntimeError, match="24.*fit"):
        _verify_heldout_receipt(
            receipt,
            lock_sha256="a" * 64,
            fit_sha256=fit_hashes,
            structural_gates_sha256="b" * 64,
        )


def test_file_snapshot_hashes_and_decodes_the_same_single_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    path = tmp_path / "fit.json"
    content = b'{"result":{"best_weights":[1,2,3]}}\n'
    path.write_bytes(content)
    original = Path.read_bytes
    calls = 0

    def counted(observed: Path) -> bytes:
        nonlocal calls
        if observed == path:
            calls += 1
        return original(observed)

    monkeypatch.setattr(Path, "read_bytes", counted)
    snapshot = iclr_trace_parity._read_file_snapshot(path)

    assert calls == 1
    assert snapshot.sha256 == hashlib.sha256(content).hexdigest()
    assert json.loads(snapshot.content.decode("utf-8")) == {
        "result": {"best_weights": [1, 2, 3]}
    }


def test_long_stage_selector_rejects_typos_and_all_expands_every_gate() -> None:
    from iclr_trace_parity import selected_long_stages

    assert selected_long_stages("") == ()
    assert selected_long_stages("all") == ("temporal-scored", "native", "full")
    with pytest.raises(ValueError, match="unknown ICLR_TRACE_PARITY_STAGE"):
        selected_long_stages("temproal-scored")


@pytest.fixture(scope="session")
def authenticated_inputs():
    from iclr_trace_parity import load_authenticated_parity_inputs

    return load_authenticated_parity_inputs(ROOT)


def test_authenticated_inputs_bind_lock_inventory_manifest_splits_and_native_files(
) -> None:
    from iclr_trace_parity import load_authenticated_parity_inputs

    observed = load_authenticated_parity_inputs(ROOT)

    assert observed.fit_inventory_count == 24
    assert len(observed.endowment_fits) == 6
    assert observed.series_count == 75
    assert observed.election_count == 397
    assert observed.manifest_sha256 == (
        "fa1fe4a407a6e0e7c8f5a1321dde38c65df93a32165c380866deb7128bcadf62"
    )
    assert observed.policy_source_sha256 == (
        "4e28c13efc690947e5fb2bf1730df3ea9b223973b23ffb37aa89c17144427348"
    )
    assert observed.heldout_receipt_sha256 == (
        "da4d9bedf568f496ae54c5433e99988edd62b20b5217ef27a9cb972d28da34ee"
    )
    assert len(observed.all_fit_sha256) == 24
    assert len(observed.authentication_fingerprint) == 64
    assert len(observed.source_sha256) == 8
    assert sum(
        len(by_year) for by_year in observed.instance_semantic_sha256.values()
    ) == 397
    assert {name for name, _ in observed.split_sha256} == {
        "temporal_2022",
        "city_out_Poland_Gdynia",
        "city_out_Poland_Warszawa",
        "city_out_Poland_Łódź",
    }
    assert all("smoke" not in fit.path.parts for fit in observed.endowment_fits)


def test_authenticated_loader_reads_each_fit_once_as_a_byte_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    original = iclr_trace_parity._read_file_snapshot
    counts: dict[Path, int] = {}

    def counted(path: Path):
        resolved = Path(path).resolve()
        if "fits" in resolved.parts:
            counts[resolved] = counts.get(resolved, 0) + 1
        return original(path)

    monkeypatch.setattr(iclr_trace_parity, "_read_file_snapshot", counted)
    observed = iclr_trace_parity.load_authenticated_parity_inputs(ROOT)

    assert len(counts) == 24
    assert set(counts.values()) == {1}
    receipt = json.loads(
        (ROOT / "results/iclr_multicity/heldout_opened.json").read_text(
            encoding="utf-8"
        )
    )
    assert dict(observed.all_fit_sha256) == receipt["fit_sha256"]


def test_authenticated_loader_rejects_coordinated_receipt_byte_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    original = iclr_trace_parity._read_file_snapshot

    def drift_receipt(path: Path):
        snapshot = original(path)
        if snapshot.path.name != "heldout_opened.json":
            return snapshot
        content = snapshot.content + b" "
        return replace(
            snapshot,
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
        )

    monkeypatch.setattr(iclr_trace_parity, "_read_file_snapshot", drift_receipt)

    with pytest.raises(RuntimeError, match="receipt SHA-256"):
        iclr_trace_parity.load_authenticated_parity_inputs(ROOT)


def test_staged_selectors_use_manifest_years_and_exact_prespecified_files(
    authenticated_inputs,
) -> None:
    from iclr_trace_parity import select_stage_elections

    sentinels = select_stage_elections(authenticated_inputs, "warsaw-sentinels")
    representative = select_stage_elections(
        authenticated_inputs, "representative-temporal-scored"
    )
    temporal = select_stage_elections(authenticated_inputs, "temporal-scored")
    native = select_stage_elections(authenticated_inputs, "native")

    assert {(row.year, row.path.name) for row in sentinels} == {
        (2019, "Poland_Warszawa_2020_Bielany.pb"),
        (2019, "Poland_Warszawa_2020_Targowek.pb"),
        (2025, "Poland_Warszawa_2026_Bemowo.pb"),
        (2025, "Poland_Warszawa_2026_Bielany.pb"),
        (2025, "Poland_Warszawa_2026_Targowek.pb"),
    }
    assert tuple(row.path.name for row in representative) == (
        "Poland_Gdynia_2023_Kamienna_Gora__large.pb",
        "Poland_Lodz_2024_Rokicie.pb",
        "Poland_Warszawa_2024_Mokotow.pb",
    )
    assert tuple(row.year for row in representative) == (2023, 2024, 2023)
    assert len(temporal) == 223
    assert all(row.phase == "scored" for row in temporal)
    assert len(native) == 397
    assert sum(row.phase == "scored" for row in native) == 223
    assert sum(row.phase == "warmup" for row in native) == 174


@pytest.mark.parametrize("stage", ("temporal-scored", "native"))
def test_stage_identity_sets_reject_same_phase_duplicate_replacements(
    authenticated_inputs,
    stage: str,
) -> None:
    from iclr_trace_parity import select_stage_elections

    elections = list(authenticated_inputs.elections)
    scored_indices = [
        index for index, row in enumerate(elections) if row.phase == "scored"
    ]
    elections[scored_indices[0]] = elections[scored_indices[1]]
    duplicate_replacement = replace(
        authenticated_inputs,
        elections=tuple(elections),
    )

    with pytest.raises(RuntimeError, match="election identities"):
        select_stage_elections(duplicate_replacement, stage)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("path", "path"),
        ("sha256", "digest"),
        ("phase", "phase"),
        ("instance", "instance"),
    ),
)
def test_stage_selector_validates_every_election_row_binding(
    authenticated_inputs,
    field: str,
    message: str,
) -> None:
    from iclr_trace_parity import select_stage_elections

    elections = list(authenticated_inputs.elections)
    target = elections[0]
    if field == "path":
        changed = replace(target, path=target.path.with_name("wrong.pb"))
    elif field == "sha256":
        changed = replace(target, sha256="f" * 64)
    elif field == "phase":
        changed = replace(
            target,
            phase="warmup" if target.phase == "scored" else "scored",
        )
    else:
        changed = replace(target, instance=elections[1].instance)
    elections[0] = changed
    drifted = replace(authenticated_inputs, elections=tuple(elections))

    with pytest.raises(RuntimeError, match=message):
        select_stage_elections(drifted, "native")


@pytest.mark.parametrize(
    ("stage", "filename"),
    (
        ("warsaw-sentinels", "Poland_Warszawa_2020_Bielany.pb"),
        (
            "representative-temporal-scored",
            "Poland_Gdynia_2023_Kamienna_Gora__large.pb",
        ),
        ("temporal-scored", "Poland_Gdynia_2023_Kamienna_Gora__large.pb"),
        ("native", "Poland_Gdynia_2023_Kamienna_Gora__large.pb"),
    ),
)
def test_named_uniform_stage_rejects_partial_source_counts(
    authenticated_inputs,
    stage: str,
    filename: str,
) -> None:
    from iclr_trace_parity import select_stage_elections

    partial = replace(
        authenticated_inputs,
        elections=tuple(
            row for row in authenticated_inputs.elections if row.path.name != filename
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="election identities|missing staged files|source counts differ",
    ):
        select_stage_elections(partial, stage)


def test_full_gate_rejects_fit_or_split_count_drift_before_any_policy(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("policy materialized before source count gates")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    missing_fit = replace(
        authenticated_inputs,
        endowment_fits=authenticated_inputs.endowment_fits[:-1],
    )
    with pytest.raises(RuntimeError, match="six.*fit contexts"):
        iclr_trace_parity.audit_frozen_endowment_fits(missing_fit)

    temporal = authenticated_inputs.splits["temporal_2022"]
    test_rows = list(temporal.test)
    key, years = test_rows[0]
    test_rows[0] = (key, years[:-1])
    drifted_splits = dict(authenticated_inputs.splits)
    drifted_splits["temporal_2022"] = replace(
        temporal, test=tuple(test_rows)
    )
    split_drift = replace(authenticated_inputs, splits=drifted_splits)
    with pytest.raises(
        RuntimeError,
        match="split.test identities|fit source counts differ",
    ):
        iclr_trace_parity.audit_frozen_endowment_fits(split_drift)


@pytest.mark.parametrize(
    ("split_name", "missing_series", "duplicate_series"),
    (
        (
            "city_out_Poland_Gdynia",
            "Poland/Gdynia/Babie Doły | large",
            "Poland/Gdynia/Chwarzno-Wiczlino | large",
        ),
        (
            "city_out_Poland_Warszawa",
            "Poland/Warszawa/Bemowo",
            "Poland/Warszawa/Białołęka",
        ),
        (
            "city_out_Poland_Łódź",
            "Poland/Łódź/Andrzejów",
            "Poland/Łódź/Bałuty Zachodnie",
        ),
        (
            "temporal_2022",
            "Poland/Gdynia/Babie Doły | large",
            "Poland/Gdynia/Chwarzno-Wiczlino | large",
        ),
    ),
)
def test_full_gate_rejects_same_size_split_test_identity_substitutions(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
    split_name: str,
    missing_series: str,
    duplicate_series: str,
) -> None:
    import iclr_trace_parity

    policy_calls: list[str] = []

    def unexpected(*args, **kwargs):
        policy_calls.append("policy")
        raise AssertionError("policy materialized before split identity gates")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    split = authenticated_inputs.splits[split_name]
    test_rows = list(split.test)
    missing_index = next(
        index for index, (series, _) in enumerate(test_rows) if series == missing_series
    )
    duplicate_row = next(row for row in test_rows if row[0] == duplicate_series)
    assert len(authenticated_inputs.index[missing_series].years) == len(
        authenticated_inputs.index[duplicate_series].years
    )
    assert len(test_rows[missing_index][1]) == len(duplicate_row[1])
    test_rows[missing_index] = duplicate_row
    drifted_splits = dict(authenticated_inputs.splits)
    drifted_splits[split_name] = replace(split, test=tuple(test_rows))
    drifted = replace(authenticated_inputs, splits=drifted_splits)

    with pytest.raises(RuntimeError, match=f"{split_name}.*identit"):
        iclr_trace_parity.audit_frozen_endowment_fits(drifted)
    assert policy_calls == []


@pytest.mark.parametrize("field", ("weights", "path", "sha256"))
def test_full_gate_rejects_selected_fit_record_drift_before_any_policy(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("policy materialized before selected-fit authentication")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    fits = list(authenticated_inputs.endowment_fits)
    fit = fits[0]
    if field == "weights":
        weights = list(fit.weights)
        weights[0] = math.nextafter(weights[0], math.inf)
        changed = replace(fit, weights=tuple(weights))
    elif field == "path":
        changed = replace(fit, path=fit.path.with_name("wrong.json"))
    else:
        changed = replace(fit, sha256="f" * 64)
    fits[0] = changed
    drifted = replace(authenticated_inputs, endowment_fits=tuple(fits))

    with pytest.raises(RuntimeError, match="authenticated|fit|fingerprint"):
        iclr_trace_parity.audit_frozen_endowment_fits(drifted)


def test_full_gate_rejects_reversed_authenticated_chronology_before_any_policy(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("policy materialized before chronology authentication")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    series = "Poland/Gdynia/Działki Leśne | large"
    ref = authenticated_inputs.index[series]
    assert tuple(year for year in ref.years if year >= 2023) == (2023,)
    index = dict(authenticated_inputs.index)
    index[series] = replace(
        ref,
        years=tuple(reversed(ref.years)),
        paths=tuple(reversed(ref.paths)),
    )
    drifted = replace(authenticated_inputs, index=index)

    with pytest.raises(RuntimeError, match="authenticated|chronolog|fingerprint"):
        iclr_trace_parity.audit_frozen_endowment_fits(drifted)


def test_full_gate_rejects_coordinated_instance_and_election_replacement(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("policy materialized before instance authentication")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    target = authenticated_inputs.elections[0]
    changed_instance = replace(
        target.instance,
        meta={**target.instance.meta, "budget": str(target.instance.budget + 1.0)},
    )
    instances = dict(authenticated_inputs.instances)
    by_year = dict(instances[target.series])
    by_year[target.year] = changed_instance
    instances[target.series] = by_year
    elections = list(authenticated_inputs.elections)
    elections[0] = replace(target, instance=changed_instance)
    drifted = replace(
        authenticated_inputs,
        instances=instances,
        elections=tuple(elections),
    )

    with pytest.raises(RuntimeError, match="authenticated|semantic|fingerprint"):
        iclr_trace_parity.audit_frozen_endowment_fits(drifted)


def test_full_gate_rejects_coordinated_path_and_digest_map_drift(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("policy materialized before map authentication")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    target = authenticated_inputs.elections[0]
    new_path = target.path.with_name("coordinated.pb")
    new_digest = "f" * 64
    changed_instance = replace(target.instance, path=str(new_path))

    index = dict(authenticated_inputs.index)
    ref = index[target.series]
    paths = list(ref.paths)
    paths[ref.years.index(target.year)] = new_path
    index[target.series] = replace(ref, paths=tuple(paths))

    instances = dict(authenticated_inputs.instances)
    by_year = dict(instances[target.series])
    by_year[target.year] = changed_instance
    instances[target.series] = by_year

    digests = dict(authenticated_inputs.manifest_instance_sha256)
    digest_by_year = dict(digests[target.series])
    digest_by_year[target.year] = new_digest
    digests[target.series] = digest_by_year

    elections = list(authenticated_inputs.elections)
    elections[0] = replace(
        target,
        path=new_path,
        sha256=new_digest,
        instance=changed_instance,
    )
    drifted = replace(
        authenticated_inputs,
        index=index,
        instances=instances,
        manifest_instance_sha256=digests,
        elections=tuple(elections),
    )

    with pytest.raises(RuntimeError, match="authenticated|fingerprint"):
        iclr_trace_parity.audit_frozen_endowment_fits(drifted)


@pytest.mark.parametrize(
    "field",
    (
        "lock_sha256",
        "heldout_receipt_sha256",
        "manifest_sha256",
        "policy_source_sha256",
        "split_sha256",
        "fit_inventory_count",
        "all_fit_sha256",
    ),
)
def test_full_gate_rejects_provenance_or_inventory_field_drift_before_policy(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    import iclr_trace_parity

    def unexpected(*args, **kwargs):
        raise AssertionError("policy materialized before provenance authentication")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    if field == "split_sha256":
        values = list(authenticated_inputs.split_sha256)
        values[0] = (values[0][0], "f" * 64)
        drifted = replace(authenticated_inputs, split_sha256=tuple(values))
    elif field == "fit_inventory_count":
        drifted = replace(authenticated_inputs, fit_inventory_count=23)
    elif field == "all_fit_sha256":
        values = list(authenticated_inputs.all_fit_sha256)
        values[0] = (values[0][0], "f" * 64)
        drifted = replace(authenticated_inputs, all_fit_sha256=tuple(values))
    else:
        drifted = replace(authenticated_inputs, **{field: "f" * 64})

    with pytest.raises(RuntimeError, match="authenticated|provenance|inventory"):
        iclr_trace_parity.audit_frozen_endowment_fits(drifted)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("path", "path"),
        ("sha256", "digest"),
        ("phase", "phase"),
        ("instance", "instance"),
    ),
)
def test_full_gate_rejects_every_election_row_drift_before_any_policy(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    message: str,
) -> None:
    import iclr_trace_parity

    policy_calls: list[str] = []

    def unexpected(*args, **kwargs):
        policy_calls.append("policy")
        raise AssertionError("policy materialized before election binding gates")

    monkeypatch.setattr(iclr_trace_parity, "linear_policy", unexpected)
    elections = list(authenticated_inputs.elections)
    target = elections[0]
    if field == "path":
        changed = replace(target, path=target.path.with_name("wrong.pb"))
    elif field == "sha256":
        changed = replace(target, sha256="f" * 64)
    elif field == "phase":
        changed = replace(
            target,
            phase="warmup" if target.phase == "scored" else "scored",
        )
    else:
        changed = replace(target, instance=elections[1].instance)
    elections[0] = changed
    drifted = replace(authenticated_inputs, elections=tuple(elections))

    with pytest.raises(RuntimeError, match=message):
        iclr_trace_parity.audit_frozen_endowment_fits(drifted)
    assert policy_calls == []


def test_uniform_diagnostic_report_carries_authenticated_receipt_digest(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def exact(instance, endowments, *, completion, instance_sha256=None):
        return iclr_trace_parity.ThreePathParity(
            instance_sha256=instance_sha256,
            completion=completion,
            reference=(),
            trace=(),
            production=(),
        )

    monkeypatch.setattr(iclr_trace_parity, "compare_three_paths", exact)
    report = iclr_trace_parity.audit_uniform_stage(
        authenticated_inputs, "warsaw-sentinels"
    )

    assert not report.gate_passed
    assert report.heldout_receipt_sha256 == (
        authenticated_inputs.heldout_receipt_sha256
    )


def test_caller_supplied_uniform_audit_is_never_a_hard_gate(
    authenticated_inputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def exact(instance, endowments, *, completion, instance_sha256=None):
        return iclr_trace_parity.ThreePathParity(
            instance_sha256=instance_sha256,
            completion=completion,
            reference=(),
            trace=(),
            production=(),
        )

    monkeypatch.setattr(iclr_trace_parity, "compare_three_paths", exact)
    report = iclr_trace_parity.audit_uniform_stage(
        authenticated_inputs, "warsaw-sentinels"
    )

    assert report.exact
    assert report.gate_eligible is False
    assert report.gate_passed is False


def test_internal_authenticated_uniform_wrapper_is_the_hard_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_trace_parity

    def exact(instance, endowments, *, completion, instance_sha256=None):
        return iclr_trace_parity.ThreePathParity(
            instance_sha256=instance_sha256,
            completion=completion,
            reference=(),
            trace=(),
            production=(),
        )

    monkeypatch.setattr(iclr_trace_parity, "compare_three_paths", exact)
    report = iclr_trace_parity.run_authenticated_uniform_stage(
        "warsaw-sentinels", repo_root=ROOT
    )

    assert report.gate_passed
    assert report.context_sha256
    assert report.fit_inventory_count == 24
    assert len(report.all_fit_sha256) == 24


def test_five_repaired_warsaw_sentinels_have_zero_mismatches_in_both_modes(
) -> None:
    from iclr_trace_parity import run_authenticated_uniform_stage

    report = run_authenticated_uniform_stage(
        "warsaw-sentinels", repo_root=ROOT
    )

    assert report.exact
    assert report.cases == 10
    assert report.completion_counts == ((False, 5), (True, 5))
    assert report.scored_cases == 6
    assert report.warmup_cases == 4
    assert report.manifest_sha256 == (
        "fa1fe4a407a6e0e7c8f5a1321dde38c65df93a32165c380866deb7128bcadf62"
    )
    assert report.heldout_receipt_sha256 == (
        "da4d9bedf568f496ae54c5433e99988edd62b20b5217ef27a9cb972d28da34ee"
    )
    assert report.gate_passed


def test_low_median_high_temporal_scored_files_have_zero_mismatches(
) -> None:
    from iclr_trace_parity import run_authenticated_uniform_stage

    report = run_authenticated_uniform_stage(
        "representative-temporal-scored", repo_root=ROOT
    )

    assert report.exact
    assert report.cases == 6
    assert report.completion_counts == ((False, 3), (True, 3))
    assert report.scored_cases == 6
    assert report.warmup_cases == 0
    assert report.gate_passed


@pytest.mark.skipif(
    "temporal-scored" not in ENABLED_LONG_STAGES,
    reason="set ICLR_TRACE_PARITY_STAGE=temporal-scored for the 223-file stage",
)
def test_all_223_temporal_scored_elections_have_zero_mismatches(
) -> None:
    from iclr_trace_parity import run_authenticated_uniform_stage

    report = run_authenticated_uniform_stage("temporal-scored", repo_root=ROOT)

    assert report.exact
    assert report.cases == 446
    assert report.completion_counts == ((False, 223), (True, 223))
    assert report.scored_cases == 446
    assert report.warmup_cases == 0
    assert report.gate_passed


@pytest.mark.skipif(
    "native" not in ENABLED_LONG_STAGES,
    reason="set ICLR_TRACE_PARITY_STAGE=native for the 397-file stage",
)
def test_all_397_native_elections_have_zero_uniform_mismatches(
) -> None:
    from iclr_trace_parity import run_authenticated_uniform_stage

    report = run_authenticated_uniform_stage("native", repo_root=ROOT)

    assert report.exact
    assert report.cases == 794
    assert report.completion_counts == ((False, 397), (True, 397))
    assert report.scored_cases == 446
    assert report.warmup_cases == 348
    assert report.gate_passed


@pytest.mark.skipif(
    "full" not in ENABLED_LONG_STAGES,
    reason="set ICLR_TRACE_PARITY_STAGE=full for the six-fit 3,176-case stage",
)
def test_six_frozen_fits_have_zero_split_correct_mismatches(
) -> None:
    from iclr_trace_parity import run_authenticated_frozen_endowment_fits

    report = run_authenticated_frozen_endowment_fits(repo_root=ROOT)

    assert report.exact
    assert report.cases == 3_176
    assert report.completion_counts == ((False, 1_588), (True, 1_588))
    assert report.scored_cases == 1_784
    assert report.warmup_cases == 1_392
    assert report.gate_passed
    assert len(report.fit_sha256) == 6
    assert dict(report.context_counts) == {
        "city_out_Poland_Gdynia/endowment/seed-42": 248,
        "city_out_Poland_Warszawa/endowment/seed-42": 258,
        "city_out_Poland_Łódź/endowment/seed-42": 288,
        "temporal_2022/endowment/seed-1": 794,
        "temporal_2022/endowment/seed-2": 794,
        "temporal_2022/endowment/seed-42": 794,
    }
