"""The tool client: every protection the runtime has, as an independent switch.

There is one client rather than a "naive" and a "safe" one on purpose. If the
baseline were a separate implementation, any difference in the measured results
could be an accident of how the two were written. With one client and a config,
the A/B differs in exactly the flags that are set — and each flag can be turned
on alone, which turns the headline comparison into an ablation.

The protections, and what each defends against:

``idempotency``
    Sends a deterministic ``Idempotency-Key`` derived from (order, step). This
    is the only defence against ``LOST_RESPONSE``: without it a retry re-applies
    the effect and the customer is charged twice.

``max_retries`` / ``respect_retry_after``
    Retries retryable faults with jittered exponential backoff, waiting out a
    ``Retry-After`` when the server sent one. Non-retryable faults (schema
    drift, truncated bodies) are surfaced immediately — retrying a contract
    change is just a slower failure.

``validate_responses``
    Checks the response actually contains the field the caller needs, instead
    of trusting a 200. This is what catches ``SCHEMA_DRIFT``, which otherwise
    hands the agent ``None`` and lets it carry on with a missing handle.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import httpx

from backstop.chaos.faults import FaultKind, ToolFailure


@dataclass
class ToolConfig:
    """Which protections are enabled."""

    idempotency: bool = False
    max_retries: int = 0
    respect_retry_after: bool = False
    validate_responses: bool = False
    base_backoff: float = 0.05
    # Scales every sleep. A sweep sets this to 0 so backoff decisions are
    # exercised without spending real seconds on them.
    time_scale: float = 1.0


@dataclass
class ToolCall:
    """One attempt at a tool call, for the trace."""

    endpoint: str
    attempt: int
    outcome: str  # "ok" | fault kind | "http_<status>"


class ToolError(Exception):
    """A tool call that could not be completed, after any retries."""

    def __init__(self, endpoint: str, reason: str, kind: FaultKind | None = None):
        self.endpoint = endpoint
        self.reason = reason
        self.kind = kind
        super().__init__(f"{endpoint}: {reason}")


class ToolClient:
    """Calls the world's tools with a configurable amount of care."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        config: ToolConfig,
        order_id: str,
        rng: random.Random | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._order_id = order_id
        # Seeded from the episode so jitter is reproducible too. Unseeded
        # jitter would make a "deterministic" episode only mostly deterministic.
        self._rng = rng or random.Random(hash(order_id) & 0xFFFF)
        self.trace: list[ToolCall] = []
        self.retries: int = 0

    # -- helpers ------------------------------------------------------------

    def _headers(self, step: str) -> dict[str, str]:
        if not self._config.idempotency:
            return {}
        # Deterministic and stable across retries — a fresh UUID per attempt
        # would defeat the entire mechanism, which is the classic way teams
        # ship idempotency that doesn't work.
        return {"Idempotency-Key": f"{self._order_id}:{step}"}

    async def _sleep(self, seconds: float) -> None:
        scaled = seconds * self._config.time_scale
        if scaled > 0:
            await asyncio.sleep(scaled)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter.

        Jitter matters even here: without it, concurrent retries re-collide on
        the same schedule, which is how a transient blip becomes a thundering
        herd.
        """
        ceiling = self._config.base_backoff * (2**attempt)
        return self._rng.uniform(0, ceiling)

    # -- the call -----------------------------------------------------------

    async def call(
        self,
        method: str,
        path: str,
        step: str,
        *,
        json_body: dict | None = None,
        expect: str | None = None,
    ) -> dict[str, Any]:
        """Call a tool, applying whichever protections are enabled.

        ``expect`` names a field the response must contain. With
        ``validate_responses`` on, its absence is an error rather than a silent
        ``None`` handed back to the caller.
        """
        attempts = self._config.max_retries + 1
        last: ToolError | None = None

        for attempt in range(attempts):
            if attempt > 0:
                self.retries += 1
            try:
                response = await self._client.request(
                    method, path, json=json_body, headers=self._headers(step)
                )
            except ToolFailure as fault:
                self.trace.append(ToolCall(path, attempt, fault.kind))
                last = ToolError(path, fault.detail or str(fault), fault.kind)
                if not fault.retryable or attempt == attempts - 1:
                    break
                await self._sleep(self._backoff(attempt))
                continue
            except httpx.HTTPError as exc:  # pragma: no cover - defensive
                self.trace.append(ToolCall(path, attempt, "transport_error"))
                last = ToolError(path, str(exc))
                if attempt == attempts - 1:
                    break
                await self._sleep(self._backoff(attempt))
                continue

            # --- a response arrived; it may still be a fault ---------------

            if response.status_code == 429:
                self.trace.append(ToolCall(path, attempt, FaultKind.RATE_LIMITED))
                last = ToolError(path, "rate limited", FaultKind.RATE_LIMITED)
                if attempt == attempts - 1:
                    break
                wait = self._backoff(attempt)
                if self._config.respect_retry_after:
                    # Honour the server's instruction. Ignoring it and backing
                    # off on our own schedule is how a client turns a polite
                    # 429 into a sustained overload.
                    wait = max(wait, float(response.headers.get("Retry-After", 1)))
                await self._sleep(wait)
                continue

            if response.status_code >= 500:
                self.trace.append(ToolCall(path, attempt, FaultKind.SERVER_ERROR))
                last = ToolError(path, "server error", FaultKind.SERVER_ERROR)
                if attempt == attempts - 1:
                    break
                await self._sleep(self._backoff(attempt))
                continue

            if response.status_code >= 400:
                # A business failure (insufficient stock, already voided).
                # Never retried: the answer will not change, and an agent that
                # retries a 409 forever is the other classic failure mode.
                self.trace.append(
                    ToolCall(path, attempt, f"http_{response.status_code}")
                )
                detail = _safe_detail(response)
                raise ToolError(path, f"{response.status_code}: {detail}")

            try:
                payload = response.json()
            except Exception:
                # A truncated body. Retrying is pointless — the injected cut is
                # a property of this response, and a real truncation means the
                # connection died mid-stream, which retry alone won't fix.
                self.trace.append(ToolCall(path, attempt, FaultKind.TRUNCATED_BODY))
                raise ToolError(
                    path, "unparseable response body", FaultKind.TRUNCATED_BODY
                ) from None

            if (
                self._config.validate_responses
                and expect is not None
                and expect not in payload
            ):
                self.trace.append(ToolCall(path, attempt, FaultKind.SCHEMA_DRIFT))
                raise ToolError(
                    path,
                    f"response is missing '{expect}' (got {sorted(payload)})",
                    FaultKind.SCHEMA_DRIFT,
                )

            self.trace.append(ToolCall(path, attempt, "ok"))
            return payload

        raise last or ToolError(path, "exhausted retries")


def _safe_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text[:80]))
    except Exception:  # pragma: no cover - defensive
        return response.text[:80]
