"""Tests for the LLM policy and, more importantly, for policy-independence.

Two separate claims are under test here.

**The seam works.** The policy renders an observation, calls an async model,
parses the reply, survives garbage, and drives a full episode to completion.
Everything except the HTTP call to a live endpoint is exercised.

**The runtime is safe regardless of the policy.** This is the one that matters.
If the no-orphan guarantee only holds because the deterministic policy happens
to be careful, it is not a property of the runtime and it would not survive a
real model. So the runtime is also run against a policy that behaves far worse
than any model would — random actions, wrong order, skipped steps — and the
invariant has to hold anyway.
"""

from __future__ import annotations

import pytest

from backstop.eval.episode import run_episode
from backstop.policies.base import ActionKind, Observation
from backstop.policies.llm import (
    LLMPolicy,
    PlanFollowingModel,
    RandomPolicy,
    StubModel,
    parse_action,
    render_observation,
)
from backstop.runtime.engine import BASELINE, RUNTIME

GOOD = dict(impossible_rate=0.0)


def observation(**kwargs) -> Observation:
    base = dict(
        order_id="ord_1",
        sku="widget-a",
        quantity=2,
        amount_cents=2598,
        steps_remaining=10,
    )
    return Observation(**{**base, **kwargs})


# --- Parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"action": "reserve", "reason": "first step"}',
        '  {"action":"reserve"}  ',
        'Sure thing!\n{"action": "reserve", "reason": "go"}\nHope that helps.',
        '```json\n{"action": "reserve"}\n```',
        '{"action": "RESERVE"}',
        '{"action": " reserve "}',
    ],
)
def test_parses_the_shapes_models_actually_emit(reply: str) -> None:
    """Models ignore "reply with only JSON". The parser must not."""
    action = parse_action(reply)
    assert action is not None
    assert action.kind is ActionKind.RESERVE


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "I'm not sure what to do here.",
        "{not json at all}",
        '{"action": 42}',
        '{"action": "teleport_the_order"}',
        '{"reason": "forgot the action field"}',
        "[]",
        '["reserve"]',
    ],
)
def test_unparseable_replies_return_none_rather_than_raising(reply: str) -> None:
    """A bad reply is an ordinary event, not an exception."""
    assert parse_action(reply) is None


def test_reason_is_captured_and_bounded() -> None:
    action = parse_action('{"action": "ship", "reason": "' + "x" * 500 + '"}')
    assert action is not None
    assert len(action.reason) <= 200


# --- Prompt rendering --------------------------------------------------------


def test_observation_mentions_what_is_done_and_what_is_held() -> None:
    rendered = render_observation(
        observation(
            completed=[ActionKind.RESERVE],
            handles={"reservation_id": "rsv_abc"},
        )
    )
    assert "reserve" in rendered
    assert "rsv_abc" in rendered
    assert "ord_1" in rendered


def test_only_recent_failures_are_included() -> None:
    """The failure log grows all episode; the prompt must not grow with it."""
    rendered = render_observation(
        observation(failures=[f"failure-{i}" for i in range(20)])
    )
    assert "failure-19" in rendered
    assert "failure-0" not in rendered


# --- Policy behaviour --------------------------------------------------------


async def test_policy_returns_the_models_action() -> None:
    policy = LLMPolicy(StubModel(['{"action": "authorize", "reason": "pay"}']))
    action = await policy.next_action(observation())
    assert action.kind is ActionKind.AUTHORIZE


async def test_policy_retries_an_unparseable_reply_then_gives_up() -> None:
    """Bounded retries — never an infinite loop on a stuck model."""
    model = StubModel(["not json"])
    policy = LLMPolicy(model, max_parse_failures=3)

    action = await policy.next_action(observation())

    assert action.kind is ActionKind.GIVE_UP
    assert model.calls == 3
    assert policy.unparseable == 3


async def test_policy_recovers_when_the_model_comes_good() -> None:
    model = StubModel(["garbage", '{"action": "reserve"}'])
    policy = LLMPolicy(model, max_parse_failures=5)

    action = await policy.next_action(observation())

    assert action.kind is ActionKind.RESERVE
    assert policy.unparseable == 1


async def test_a_model_outage_becomes_a_clean_give_up() -> None:
    """The agent's brain failing must not leave the world half-changed."""

    class DeadModel:
        name = "dead"

        async def complete(self, system: str, user: str) -> str:
            raise ConnectionError("endpoint unreachable")

    policy = LLMPolicy(DeadModel())
    action = await policy.next_action(observation())

    assert action.kind is ActionKind.GIVE_UP
    assert "model call failed" in action.reason


async def test_policy_stops_when_the_budget_is_gone_without_calling_the_model() -> None:
    model = StubModel(['{"action": "reserve"}'])
    policy = LLMPolicy(model)

    action = await policy.next_action(observation(steps_remaining=0))

    assert action.kind is ActionKind.GIVE_UP
    assert model.calls == 0, "no point paying for a call we cannot act on"


# --- The seam, driven through whole episodes ---------------------------------


async def test_the_llm_policy_can_fulfill_an_order() -> None:
    """The full path: prompt -> model -> parse -> act, for a real episode."""
    policy = LLMPolicy(PlanFollowingModel())
    outcome = await run_episode(5, policy, RUNTIME, 0.0, **GOOD)

    assert outcome.fulfilled is True
    assert outcome.violations == []


async def test_the_llm_policy_survives_an_intermittently_broken_model() -> None:
    """Every third reply is prose instead of JSON, and the order still ships."""
    policy = LLMPolicy(PlanFollowingModel(malformed_every=3), max_parse_failures=5)
    outcome = await run_episode(5, policy, RUNTIME, 0.0, **GOOD)

    assert outcome.fulfilled is True
    assert policy.unparseable > 0, "the broken-reply path should have been hit"


async def test_the_llm_policy_under_faults_leaves_no_orphans() -> None:
    policy = LLMPolicy(PlanFollowingModel())
    outcomes = [
        await run_episode(seed, policy, RUNTIME, 0.25, **GOOD) for seed in range(25)
    ]
    orphaned = [o for o in outcomes if o.has_orphans]
    assert not orphaned, [str(v) for o in orphaned for v in o.violations]


# --- Policy independence: the claim that actually matters --------------------


async def test_the_runtime_is_safe_even_against_a_random_policy() -> None:
    """The guarantee must belong to the runtime, not to a well-behaved agent.

    A random policy calls things in the wrong order, repeats steps, and skips
    prerequisites — worse than any model would. If the no-orphan property holds
    here, it holds for a model too, because the runtime is what enforces it.
    """
    outcomes = [
        await run_episode(seed, RandomPolicy(seed=seed), RUNTIME, 0.15, **GOOD)
        for seed in range(40)
    ]
    orphaned = [o for o in outcomes if o.has_orphans]

    assert not orphaned, [
        f"seed {o.seed}: " + "; ".join(str(v) for v in o.violations) for o in orphaned
    ]


async def test_a_random_policy_does_break_things_without_the_runtime() -> None:
    """Confirm the previous test is not passing vacuously.

    If the random policy left the world clean under the *baseline* too, then it
    simply never does anything dangerous and the test above proves nothing.
    """
    outcomes = [
        await run_episode(seed, RandomPolicy(seed=seed), BASELINE, 0.15, **GOOD)
        for seed in range(40)
    ]
    assert any(o.has_orphans for o in outcomes), (
        "the random policy should damage an unprotected world, "
        "otherwise the safety test above is vacuous"
    )


async def test_stock_is_conserved_even_under_a_random_policy() -> None:
    """The hardest invariant, against the worst agent."""
    for seed in range(30):
        outcome = await run_episode(seed, RandomPolicy(seed=seed), RUNTIME, 0.2)
        assert not any(
            v.kind == "stock_not_conserved" for v in outcome.violations
        ), [str(v) for v in outcome.violations]
