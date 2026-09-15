"""What an agent *decides*, separated from how the runtime *executes*.

A policy answers one question: given what has happened so far, what should the
next action be? It never performs the action, never retries, never compensates.
That separation is what lets the same measurement harness compare a
deterministic control policy against an LLM without changing anything else.

The action vocabulary is deliberately small and matches the world's tools
one-for-one. A richer action space would make the LLM policy's job harder in
ways that have nothing to do with reliability, which is what is being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ActionKind(StrEnum):
    RESERVE = "reserve"
    AUTHORIZE = "authorize"
    CAPTURE = "capture"
    SHIP = "ship"
    NOTIFY = "notify"
    COMPLETE = "complete"
    # Deciding to stop is a first-class action, not the absence of one. A
    # policy that can only push forward can never fail cleanly, and clean
    # failure is half of what this project measures.
    GIVE_UP = "give_up"


@dataclass
class Action:
    """One thing the policy wants done."""

    kind: ActionKind
    reason: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """What the policy knows when choosing its next action."""

    order_id: str
    sku: str
    quantity: int
    amount_cents: int
    # Handles obtained so far: reservation_id, payment_id, shipment_id.
    handles: dict[str, str] = field(default_factory=dict)
    # Steps that have definitely completed.
    completed: list[ActionKind] = field(default_factory=list)
    # Human-readable log of what went wrong, newest last.
    failures: list[str] = field(default_factory=list)
    steps_used: int = 0
    steps_remaining: int = 0


class Policy(Protocol):
    """The agent's decision function.

    ``next_action`` is async because a real policy has to be able to *call*
    something — a model endpoint, a planner, a human. Making it synchronous
    would have quietly restricted this seam to policies that can decide without
    any I/O, which excludes every interesting one.
    """

    name: str

    async def next_action(self, observation: Observation) -> Action:
        """Choose the next action given everything known so far."""
        ...

    def reset(self) -> None:
        """Clear any per-episode state."""
        ...
