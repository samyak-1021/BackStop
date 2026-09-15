"""Tests for the recovery runtime.

The sweep produces the headline numbers; these tests pin down the *mechanisms*
that produce them, so a regression shows up as a named failure rather than as a
percentage quietly drifting in a report nobody re-reads.

The structure mirrors the claims:

* the saga unwinds in reverse, retries its compensations, and reports failures
* idempotency turns duplicate delivery into a single effect
* reconciliation recovers effects the agent lost track of entirely
* ordering decides whether an irreversible action becomes an unrecoverable one
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from backstop.eval.episode import run_episode
from backstop.policies.scripted import ScriptedPolicy
from backstop.runtime.engine import BASELINE, RUNTIME
from backstop.runtime.saga import SagaLog
from backstop.world.verifier import (
    DOUBLE_CHARGE,
    FALSE_NOTIFICATION,
    LEAKED_STOCK,
    MONEY_TAKEN_NOTHING_SHIPPED,
)

# A seed whose scenario is satisfiable, so failures are attributable to faults
# rather than to an impossible task.
GOOD = dict(impossible_rate=0.0)


# --- The saga ----------------------------------------------------------------


async def test_saga_unwinds_in_reverse_order() -> None:
    """Later effects depend on earlier ones, so they must be undone first."""
    undone: list[str] = []
    saga = SagaLog()
    for name in ("reserve", "authorize", "capture", "ship"):
        saga.record(name, lambda n=name: _append(undone, n))

    result = await saga.unwind()

    assert undone == ["ship", "capture", "authorize", "reserve"]
    assert result.clean is True
    assert result.failed == []


async def test_saga_retries_a_flaky_compensation() -> None:
    """Compensations run over the same broken network as everything else."""
    attempts = {"n": 0}

    async def flaky() -> None:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("compensation failed")

    saga = SagaLog()
    saga.record("flaky", flaky)
    result = await saga.unwind(attempts=3)

    assert attempts["n"] == 3
    assert result.clean is True


async def test_saga_reports_a_compensation_it_could_not_complete() -> None:
    """A failed unwind is a real orphan and must never be silently swallowed."""

    async def always_fails() -> None:
        raise RuntimeError("payment gateway unreachable")

    saga = SagaLog()
    saga.record("refund", always_fails)
    result = await saga.unwind(attempts=2)

    assert result.clean is False
    assert result.failed[0][0] == "refund"
    assert "unreachable" in result.failed[0][1]


async def test_saga_records_irreversible_effects_separately() -> None:
    """An effect with no compensation is tracked, not ignored."""
    saga = SagaLog()
    saga.record("notify", compensate=None)
    result = await saga.unwind()

    assert result.irreversible == ["notify"]
    # Irreversible is not the same as failed — nothing went wrong here, the
    # action simply cannot be undone.
    assert result.clean is True


async def _append(sink: list[str], name: str) -> None:
    sink.append(name)


# --- End-to-end behaviour ----------------------------------------------------


async def test_clean_world_needs_no_protections() -> None:
    """With no faults, baseline and runtime must be indistinguishable.

    If they differ here, the runtime is changing behaviour rather than just
    defending it, and every later comparison is contaminated.
    """
    policy = ScriptedPolicy()
    for seed in range(25):
        base = await run_episode(seed, policy, BASELINE, 0.0, **GOOD)
        safe = await run_episode(seed, policy, RUNTIME, 0.0, **GOOD)
        assert base.fulfilled == safe.fulfilled is True
        assert base.violations == [] and safe.violations == []


async def test_runtime_leaves_no_orphans_at_a_realistic_fault_rate() -> None:
    """The central claim, as an assertion rather than a number in a report."""
    policy = ScriptedPolicy()
    outcomes = [
        await run_episode(seed, policy, RUNTIME, 0.20, **GOOD) for seed in range(80)
    ]
    orphans = [o for o in outcomes if o.has_orphans]

    assert not orphans, [str(v) for o in orphans for v in o.violations]


async def test_baseline_does_leave_orphans_at_the_same_rate() -> None:
    """The comparison only means something if the baseline actually breaks."""
    policy = ScriptedPolicy()
    outcomes = [
        await run_episode(seed, policy, BASELINE, 0.20, **GOOD) for seed in range(80)
    ]
    kinds = {k for o in outcomes for k in o.orphan_kinds}

    assert any(o.has_orphans for o in outcomes)
    # The interesting harms, not just "something went wrong".
    assert kinds & {MONEY_TAKEN_NOTHING_SHIPPED, LEAKED_STOCK, DOUBLE_CHARGE}


async def test_idempotency_is_what_prevents_double_charges() -> None:
    """Isolate one protection against one harm.

    Retries *without* idempotency should be able to double-charge; adding
    idempotency should remove that specific violation.
    """
    policy = ScriptedPolicy()
    retries_only = replace(BASELINE, max_retries=3, respect_retry_after=True)
    with_keys = replace(retries_only, idempotency=True)

    unsafe = [
        await run_episode(s, policy, retries_only, 0.35, **GOOD) for s in range(60)
    ]
    safe = [await run_episode(s, policy, with_keys, 0.35, **GOOD) for s in range(60)]

    unsafe_charges = sum(DOUBLE_CHARGE in o.orphan_kinds for o in unsafe)
    safe_charges = sum(DOUBLE_CHARGE in o.orphan_kinds for o in safe)

    assert unsafe_charges > 0, "retrying without keys should double-charge someone"
    assert safe_charges == 0, "idempotency must eliminate double charges entirely"


async def test_reconciliation_recovers_effects_the_agent_lost_track_of() -> None:
    """Compensation alone is not enough when the agent's own log is wrong.

    A lost response means the effect happened but the agent never learned its
    handle. Without reading the world back, there is nothing to compensate.
    """
    policy = ScriptedPolicy()
    blind = replace(BASELINE, compensate_on_failure=True, reconcile=False)
    seeing = replace(blind, reconcile=True)

    blind_out = [await run_episode(s, policy, blind, 0.35, **GOOD) for s in range(60)]
    seeing_out = [await run_episode(s, policy, seeing, 0.35, **GOOD) for s in range(60)]

    blind_orphans = sum(o.has_orphans for o in blind_out)
    seeing_orphans = sum(o.has_orphans for o in seeing_out)

    assert seeing_orphans < blind_orphans, (
        f"reconciliation should reduce orphans: {blind_orphans} -> {seeing_orphans}"
    )


async def test_blind_unwinding_manufactures_false_notifications() -> None:
    """A saga that unwinds past an irreversible action makes things worse.

    This is the bug the point-of-no-return rule exists to prevent, kept as a
    test because it is deeply counter-intuitive: the episode had *succeeded* —
    goods shipped, customer notified, money taken — and only the final
    bookkeeping call failed. A runtime that unwinds at that point cancels a
    real shipment and turns a sent email into a permanent lie.
    """
    blind = replace(RUNTIME, respect_point_of_no_return=False)
    outcomes = [
        await run_episode(seed, ScriptedPolicy(), blind, 0.6, **GOOD)
        for seed in range(80)
    ]
    lies = sum(FALSE_NOTIFICATION in o.orphan_kinds for o in outcomes)
    assert lies > 0, "blind unwinding should produce false notifications"


async def test_respecting_the_point_of_no_return_reduces_the_damage() -> None:
    """The fix, measured against the bug above on identical seeds."""
    blind = replace(RUNTIME, respect_point_of_no_return=False)
    seeds = range(80)

    blind_out = [
        await run_episode(s, ScriptedPolicy(), blind, 0.6, **GOOD) for s in seeds
    ]
    aware_out = [
        await run_episode(s, ScriptedPolicy(), RUNTIME, 0.6, **GOOD) for s in seeds
    ]

    blind_lies = sum(FALSE_NOTIFICATION in o.orphan_kinds for o in blind_out)
    aware_lies = sum(FALSE_NOTIFICATION in o.orphan_kinds for o in aware_out)

    assert aware_lies < blind_lies, (
        f"point-of-no-return handling should cut false notifications: "
        f"{blind_lies} -> {aware_lies}"
    )


async def test_escalation_happens_instead_of_a_destructive_unwind() -> None:
    """When it cannot finish and must not unwind, it asks for a human.

    Escalation is the honest third outcome. A runtime with only "succeed" and
    "roll back" has to pick one of them even when both are wrong.
    """
    outcomes = [
        await run_episode(seed, ScriptedPolicy(), RUNTIME, 0.75, **GOOD)
        for seed in range(80)
    ]
    assert any(o.result.escalated for o in outcomes)
    assert any(o.result.rolled_forward for o in outcomes)


async def test_impossible_orders_fail_cleanly_not_messily() -> None:
    """Refusing an impossible task without breaking anything is a success."""
    policy = ScriptedPolicy()
    outcomes = [
        await run_episode(seed, policy, RUNTIME, 0.15, impossible_rate=1.0)
        for seed in range(40)
    ]

    assert all(not o.fulfilled for o in outcomes)
    assert all(o.clean_failure for o in outcomes), [
        str(v) for o in outcomes for v in o.violations
    ]
    assert all(o.correct for o in outcomes)


async def test_episodes_are_reproducible() -> None:
    """Same seed, same result — twice, byte for byte where it matters."""
    policy = ScriptedPolicy()
    for seed in (3, 17, 42):
        first = await run_episode(seed, policy, RUNTIME, 0.3)
        second = await run_episode(seed, policy, RUNTIME, 0.3)
        assert first.fulfilled == second.fulfilled
        assert first.orphan_kinds == second.orphan_kinds
        assert first.result.tool_calls == second.result.tool_calls
        assert first.result.retries == second.result.retries


async def test_the_step_budget_is_enforced() -> None:
    """A confused agent must be cut off rather than looping forever."""
    policy = ScriptedPolicy(give_up_after=10_000)  # never gives up on its own
    tight = replace(RUNTIME, step_budget=3, roll_forward_attempts=2)
    outcome = await run_episode(1, policy, tight, 0.9, **GOOD)

    # The budget bounds the policy-driven phase. Roll-forward draws on its own
    # small, separate allowance: once an irreversible action has happened,
    # abandoning a half-finished operation to save two tool calls is a worse
    # trade than spending them. The guarantee is that both are bounded.
    assert outcome.result.steps_used <= 3 + 2


async def test_roll_forward_cannot_run_unbounded() -> None:
    """Even the 'must finish' path has a ceiling."""
    policy = ScriptedPolicy(give_up_after=1)
    config = replace(RUNTIME, step_budget=8, roll_forward_attempts=2)
    for seed in range(15):
        outcome = await run_episode(seed, policy, config, 0.8, **GOOD)
        assert outcome.result.steps_used <= 8 + 2


@pytest.mark.parametrize("rate", [0.0, 0.1, 0.25, 0.5])
async def test_stock_is_always_conserved(rate: float) -> None:
    """Whatever happens, quantity is never created or destroyed.

    Checked across fault rates for both configs: the baseline is allowed to
    leak stock into 'reserved' forever, but it must never make units appear
    or vanish. A failure here means the world itself is broken, which would
    invalidate every other measurement.
    """
    policy = ScriptedPolicy()
    for config in (BASELINE, RUNTIME):
        for seed in range(20):
            outcome = await run_episode(seed, policy, config, rate)
            assert not any(
                v.kind == "stock_not_conserved" for v in outcome.violations
            ), [str(v) for v in outcome.violations]
