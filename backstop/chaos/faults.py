"""The fault model: what can go wrong between an agent and its tools.

Every fault here is something that happens to real services every day. The
project's claim is only as good as this list is honest, so each one names the
real-world thing it stands for.

The ordering below is roughly "how badly does this break a naive agent":

``TIMEOUT``
    The call never comes back. The agent cannot tell a slow success from a
    failure — which is the root of most correctness bugs in this space.

``RATE_LIMITED``
    HTTP 429 with a ``Retry-After``. Retryable, but only if you actually wait;
    hammering makes it worse.

``SERVER_ERROR``
    A 500. Retryable, and the honest signal that something upstream broke.

``TRUNCATED_BODY``
    A 200 with a body that got cut off mid-JSON. Looks like success until you
    parse it. Catches agents that assume ``status == 200`` means "done".

``SCHEMA_DRIFT``
    A 200 with a *valid* body whose field names changed — ``reservation_id``
    arrives as ``id``. The nastiest quiet failure: nothing errors, the agent
    just reads ``None`` and carries on with a missing handle.

``LOST_RESPONSE``
    **The important one.** The request is delivered and applied — stock moves,
    money moves — and the *response* is lost. The agent sees a timeout and, if
    it retries without an idempotency key, does it all again. This single
    fault is why ``Idempotency-Key`` exists, and it is what separates a system
    that merely retries from one that retries *safely*.

``SLOW``
    A successful but slow response. Not a failure — included so latency budgets
    and step caps get exercised, and so "everything is fine, just slower" is
    represented in the mix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FaultKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    TRUNCATED_BODY = "truncated_body"
    SCHEMA_DRIFT = "schema_drift"
    LOST_RESPONSE = "lost_response"
    SLOW = "slow"


# Relative likelihood of each fault once the injector has decided to fire.
# LOST_RESPONSE is weighted heavily on purpose: it is the fault the recovery
# runtime exists to survive, so an evaluation that rarely produces it would
# flatter any agent that ignores idempotency.
DEFAULT_MIX: dict[FaultKind, float] = {
    FaultKind.TIMEOUT: 0.20,
    FaultKind.RATE_LIMITED: 0.15,
    FaultKind.SERVER_ERROR: 0.20,
    FaultKind.TRUNCATED_BODY: 0.10,
    FaultKind.SCHEMA_DRIFT: 0.10,
    FaultKind.LOST_RESPONSE: 0.20,
    FaultKind.SLOW: 0.05,
}

# Faults that let the request reach the world before anything goes wrong.
# These are the ones where a retry can duplicate a side effect.
APPLIES_THE_EFFECT = frozenset({FaultKind.LOST_RESPONSE, FaultKind.SLOW})

# Faults a well-behaved client should retry. SCHEMA_DRIFT and TRUNCATED_BODY
# are excluded deliberately: retrying doesn't help if the service has genuinely
# changed its contract, and an agent that retries forever on them is wrong.
RETRYABLE = frozenset(
    {
        FaultKind.TIMEOUT,
        FaultKind.RATE_LIMITED,
        FaultKind.SERVER_ERROR,
        FaultKind.LOST_RESPONSE,
    }
)


@dataclass(frozen=True)
class FaultEvent:
    """One injected fault, recorded so a run can be explained afterwards."""

    kind: FaultKind
    endpoint: str
    attempt: int
    applied_to_world: bool

    def __str__(self) -> str:  # pragma: no cover - presentation only
        suffix = " (effect applied)" if self.applied_to_world else ""
        return f"{self.kind} on {self.endpoint} attempt {self.attempt}{suffix}"


class ToolFailure(Exception):
    """Raised to the caller when an injected fault makes a call fail.

    Carries the fault kind so the runtime can decide whether to retry, and so
    the report can attribute failures to specific fault types rather than to an
    undifferentiated "it broke".
    """

    def __init__(self, kind: FaultKind, endpoint: str, detail: str = "") -> None:
        self.kind = kind
        self.endpoint = endpoint
        self.detail = detail
        super().__init__(f"{kind} on {endpoint}{f': {detail}' if detail else ''}")

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE
