"""The deterministic reference policy — the experimental control.

This is not a placeholder for an LLM. It is a control, and having one is better
methodology than not: because it makes exactly the same decisions every time,
all the variance in a measurement comes from the *injected faults* and the
*runtime protections*, which is what the project is trying to isolate. Swapping
in a model adds a second, noisier source of variance; you want the clean
measurement first to compare it against.

The policy knows the correct order of operations and nothing else. It does not
retry (the runtime's job), does not compensate (the saga's job), and does not
know that faults exist. It gives up when it has been told a step failed too
many times, or when the step budget is nearly spent.

**Ordering is the one interesting decision.** ``notify`` is irreversible, so it
comes after everything that can still fail. Put it earlier — as the
``EagerPolicy`` below deliberately does — and a late failure leaves a customer
holding an email about a shipment that was subsequently cancelled, which
nothing in the system can retract.
"""

from __future__ import annotations

from backstop.policies.base import Action, ActionKind, Observation

# The safe order. Every failable, reversible step completes before the single
# irreversible one is attempted.
SAFE_ORDER: list[ActionKind] = [
    ActionKind.RESERVE,
    ActionKind.AUTHORIZE,
    ActionKind.CAPTURE,
    ActionKind.SHIP,
    ActionKind.NOTIFY,
    ActionKind.COMPLETE,
]

# The tempting-but-wrong order: tell the customer as soon as payment clears,
# before the goods are actually committed to a shipment. Used to demonstrate
# that ordering, not just retrying, is part of correctness.
EAGER_ORDER: list[ActionKind] = [
    ActionKind.RESERVE,
    ActionKind.AUTHORIZE,
    ActionKind.CAPTURE,
    ActionKind.NOTIFY,
    ActionKind.SHIP,
    ActionKind.COMPLETE,
]


class ScriptedPolicy:
    """Walk the plan in order, giving up when a step keeps failing."""

    def __init__(
        self,
        order: list[ActionKind] | None = None,
        give_up_after: int = 3,
        name: str = "scripted",
    ) -> None:
        self._order = list(order or SAFE_ORDER)
        self._give_up_after = give_up_after
        self.name = name

    def reset(self) -> None:
        """No per-episode state: the observation carries everything."""

    async def next_action(self, observation: Observation) -> Action:
        # Out of budget. Stopping deliberately is what lets the runtime unwind
        # while it still has the steps to do it.
        if observation.steps_remaining <= 0:
            return Action(ActionKind.GIVE_UP, reason="step budget exhausted")

        # Too many failures. The runtime has already retried at the transport
        # level; repeated failures here mean something is genuinely wrong, and
        # continuing just digs the hole deeper.
        if len(observation.failures) >= self._give_up_after:
            last = observation.failures[-1]
            return Action(
                ActionKind.GIVE_UP,
                reason=f"{len(observation.failures)} failures; last: {last}",
            )

        for step in self._order:
            if step not in observation.completed:
                return Action(step, reason="next step in plan")

        return Action(ActionKind.COMPLETE, reason="plan finished")


class EagerPolicy(ScriptedPolicy):
    """Identical, except it notifies the customer before shipping.

    Exists to isolate one claim: that *when* you perform an irreversible action
    changes how often you leave an unrecoverable orphan, independently of
    retries and compensation.
    """

    def __init__(self, give_up_after: int = 3) -> None:
        super().__init__(order=EAGER_ORDER, give_up_after=give_up_after, name="eager")
