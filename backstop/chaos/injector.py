"""Deterministic fault injection in front of the world.

This sits between the agent and the tools and decides, per call, whether to
misbehave. Two properties matter more than anything else here:

**Determinism.** The same (seed, fault rate) produces the same faults in the
same order, every run. Without that, a failing episode is a ghost you cannot
reproduce, and any A/B between baseline and runtime is comparing two different
worlds. The RNG is seeded per *episode*, and each decision advances it, so the
fault sequence is a pure function of the seed.

**Faults are a property of the network, not of the world.** The world app never
learns that chaos exists. That keeps the world an honest implementation, and it
means ``LOST_RESPONSE`` can apply a real side effect and *then* throw away the
reply — which is exactly what a dropped connection does, and what no
in-process mock ever reproduces convincingly.

The injector wraps an httpx transport, so it works identically against an
in-process ASGI app (fast, for sweeps) and a real server over a socket.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field

import httpx

from backstop.chaos.faults import (
    DEFAULT_MIX,
    FaultEvent,
    FaultKind,
    ToolFailure,
)

# Field renames used by SCHEMA_DRIFT. Each is a plausible refactor someone
# might ship without thinking about their callers.
_DRIFT_RENAMES = {
    "reservation_id": "id",
    "payment_id": "id",
    "shipment_id": "id",
    "notification_id": "id",
    "state": "status",
}


@dataclass
class ChaosConfig:
    """How badly the network is behaving."""

    # Probability that any single call is hit by a fault. 0.0 is a clean run.
    fault_rate: float = 0.0
    seed: int = 0
    mix: dict[FaultKind, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    # Multiplier on injected sleeps. Tests set this to 0 so a sweep of
    # thousands of episodes doesn't spend its life asleep; the *decisions* are
    # unchanged, only the wall-clock cost.
    time_scale: float = 1.0
    retry_after_seconds: int = 1


class ChaosTransport(httpx.AsyncBaseTransport):
    """An httpx transport that injects faults before or after the real call."""

    def __init__(self, inner: httpx.AsyncBaseTransport, config: ChaosConfig) -> None:
        self._inner = inner
        self._config = config
        self._rng = random.Random(config.seed)
        self.events: list[FaultEvent] = []
        self.calls: int = 0

    # -- fault selection ----------------------------------------------------

    def _pick_fault(self) -> FaultKind | None:
        """Decide this call's fate. Always advances the RNG exactly twice.

        Draw the roll and the kind unconditionally, so the sequence of faults
        depends only on the seed and the number of calls — not on which branch
        was taken. Conditional draws would make the stream diverge between the
        baseline and the runtime and quietly ruin the comparison.
        """
        roll = self._rng.random()
        kinds = list(self._config.mix)
        weights = [self._config.mix[k] for k in kinds]
        chosen = self._rng.choices(kinds, weights=weights, k=1)[0]
        return chosen if roll < self._config.fault_rate else None

    async def _sleep(self, seconds: float) -> None:
        scaled = seconds * self._config.time_scale
        if scaled > 0:
            await asyncio.sleep(scaled)

    # -- transport ----------------------------------------------------------

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        endpoint = request.url.path
        fault = self._pick_fault()

        # Reads are left alone. Injecting on GETs would inflate the fault count
        # without testing anything interesting — the risk in this domain lives
        # entirely in the mutating calls.
        if fault is None or request.method == "GET":
            return await self._inner.handle_async_request(request)

        # Recorded once the outcome is known, not before. `fault in
        # APPLIES_THE_EFFECT` says "this fault kind lets the request through",
        # which is necessary but not sufficient: the world can still reject it
        # (409 on insufficient stock, say), in which case nothing was applied.
        # Marking those as applied taught the runtime to record a
        # possibly-applied effect for something that never happened — harmless
        # for reserve, but `notify` sets irreversible=True, so a rejected notify
        # would have falsely tripped the point-of-no-return rule.
        def record(applied: bool) -> None:
            self.events.append(
                FaultEvent(
                    kind=fault,
                    endpoint=endpoint,
                    attempt=self.calls,
                    applied_to_world=applied,
                )
            )

        # --- Faults that stop the request before it reaches the world ------

        if fault is FaultKind.TIMEOUT:
            record(applied=False)
            await self._sleep(0.05)
            raise ToolFailure(fault, endpoint, "request timed out")

        if fault is FaultKind.RATE_LIMITED:
            record(applied=False)
            return httpx.Response(
                429,
                headers={"Retry-After": str(self._config.retry_after_seconds)},
                json={"detail": "rate limited"},
                request=request,
            )

        if fault is FaultKind.SERVER_ERROR:
            record(applied=False)
            return httpx.Response(
                500, json={"detail": "internal error"}, request=request
            )

        # --- Faults that let the request through first ---------------------

        response = await self._inner.handle_async_request(request)
        # The request reached the world. Whether it *changed* anything is the
        # world's answer, not the fault's.
        record(applied=response.status_code < 400)

        if fault is FaultKind.SLOW:
            await self._sleep(0.05)
            return response

        if fault is FaultKind.LOST_RESPONSE:
            # The world has already applied the effect. The caller will see a
            # timeout and has no way to know the difference. Everything that
            # makes this survivable has to happen on the client side.
            await response.aread()
            raise ToolFailure(
                fault, endpoint, "connection dropped after the request was applied"
            )

        if fault is FaultKind.TRUNCATED_BODY:
            body = (await response.aread()).decode()
            return httpx.Response(
                response.status_code,
                content=body[: max(1, len(body) // 2)],
                headers={"content-type": "application/json"},
                request=request,
            )

        if fault is FaultKind.SCHEMA_DRIFT:
            body = await response.aread()
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:  # pragma: no cover - defensive
                return response
            if isinstance(payload, dict):
                payload = {
                    _DRIFT_RENAMES.get(k, k): v for k, v in payload.items()
                }
            return httpx.Response(
                response.status_code, json=payload, request=request
            )

        return response  # pragma: no cover - every kind is handled above
