"""An LLM policy, and the seam it plugs into.

The deterministic policy is the experimental control; this is the thing it is a
control *for*. Everything here — prompt construction, response parsing, handling
a model that returns garbage — is implemented and tested against a stub model.
The only piece that has never run is the HTTP call to a live endpoint, because
that needs an API key.

That boundary is deliberate and worth stating plainly: "you can swap in a model"
is a claim, and a claim you haven't implemented is just a hope. What is proven
here is that the seam accepts a real, fallible, async decision-maker and that
the runtime's safety properties survive one.

**The policy is assumed hostile.** A model can hallucinate an action, emit
invalid JSON, skip a step, or loop. None of that may be allowed to corrupt the
world — the runtime's guarantees have to hold regardless of how badly the policy
behaves, because you do not control the model. ``RandomPolicy`` below exists to
test exactly that, and it misbehaves far more than any real model would.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Protocol

from backstop.policies.base import Action, ActionKind, Observation

SYSTEM_PROMPT = """You are fulfilling customer orders by calling tools.

Complete the order in this sequence:
  1. reserve    - hold the stock
  2. authorize  - place a hold on the customer's funds
  3. capture    - take the payment
  4. ship       - create the shipment
  5. notify     - email the customer (IRREVERSIBLE - only after shipping)
  6. complete   - mark the order done

You may also answer "give_up" if the task cannot be completed.

Reply with ONLY a JSON object, no other text:
  {"action": "<one of: reserve, authorize, capture, ship, notify, complete, \
give_up>", "reason": "<short>"}"""


class ModelClient(Protocol):
    """Anything that can turn a prompt into text."""

    name: str

    async def complete(self, system: str, user: str) -> str:
        """Return the model's raw reply."""
        ...


# Accepts a bare object or one wrapped in prose/code fences, because models do
# both regardless of what the prompt asked for.
_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action(reply: str) -> Action | None:
    """Turn a model reply into an Action, or None if it cannot be understood.

    Returning None rather than raising is the whole point: an unparseable reply
    is an ordinary event, not an exceptional one, and the caller needs to count
    it and carry on rather than crash the episode.
    """
    if not reply or not reply.strip():
        return None

    match = _JSON_OBJECT.search(reply)
    if match is None:
        return None
    try:
        payload: Any = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    raw = payload.get("action")
    if not isinstance(raw, str):
        return None
    try:
        kind = ActionKind(raw.strip().lower())
    except ValueError:
        # A hallucinated action name. Not an error — just not a move we have.
        return None

    reason = payload.get("reason")
    return Action(kind, reason=str(reason)[:200] if reason else "")


def render_observation(observation: Observation) -> str:
    """Describe the current state to the model.

    Deliberately compact. Every token here is paid on every step of every
    episode, and a sweep runs thousands of episodes — a verbose prompt is a
    recurring cost, not a one-off one.
    """
    done = ", ".join(observation.completed) or "nothing yet"
    handles = ", ".join(f"{k}={v}" for k, v in observation.handles.items()) or "none"
    lines = [
        f"Order {observation.order_id}: {observation.quantity} x {observation.sku}",
        f"Amount: {observation.amount_cents} cents",
        f"Completed steps: {done}",
        f"Handles held: {handles}",
        f"Steps remaining: {observation.steps_remaining}",
    ]
    if observation.failures:
        # Only the most recent failures: the full history grows without bound
        # and the old entries stop being actionable.
        recent = observation.failures[-3:]
        lines.append("Recent failures:")
        lines.extend(f"  - {f}" for f in recent)
    lines.append("What is the next action?")
    return "\n".join(lines)


class LLMPolicy:
    """Decides the next action by asking a model.

    Two safety behaviours that matter more than the prompt:

    * A reply that cannot be parsed is *retried* a bounded number of times and
      then becomes a give-up, rather than an exception or an infinite loop.
    * A model failure (endpoint down, timeout) is treated the same way. The
      agent's brain being unavailable must not leave the world half-changed —
      the runtime still gets a clean decision to stop, and can then unwind.
    """

    def __init__(
        self,
        client: ModelClient,
        max_parse_failures: int = 3,
        name: str | None = None,
    ) -> None:
        self._client = client
        self._max_parse_failures = max_parse_failures
        self.name = name or f"llm:{client.name}"
        self._parse_failures = 0
        self.calls = 0
        self.unparseable = 0

    def reset(self) -> None:
        self._parse_failures = 0

    async def next_action(self, observation: Observation) -> Action:
        if observation.steps_remaining <= 0:
            return Action(ActionKind.GIVE_UP, reason="step budget exhausted")

        try:
            self.calls += 1
            reply = await self._client.complete(
                SYSTEM_PROMPT, render_observation(observation)
            )
        except Exception as exc:
            # The model is part of the system and can fail like any other
            # dependency. Stopping cleanly is the correct response.
            return Action(ActionKind.GIVE_UP, reason=f"model call failed: {exc}")

        action = parse_action(reply)
        if action is None:
            self.unparseable += 1
            self._parse_failures += 1
            if self._parse_failures >= self._max_parse_failures:
                return Action(
                    ActionKind.GIVE_UP,
                    reason=f"{self._parse_failures} unparseable model replies",
                )
            # Ask again. The observation is unchanged, so this is a genuine
            # retry of the same decision rather than a different question.
            return await self.next_action(observation)

        self._parse_failures = 0
        return action


class StubModel:
    """A scripted model, for testing the policy without an API key.

    Takes a list of raw replies and returns them in order, repeating the last
    one forever. Malformed entries are the interesting ones: they are how the
    parse-failure and give-up paths get exercised.
    """

    name = "stub"

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies) or ['{"action": "give_up"}']
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return reply


class PlanFollowingModel:
    """A stub that reads the observation and names the next unfinished step.

    Not a model in any meaningful sense — it is a way to drive the *real*
    LLMPolicy (prompt rendering, parsing, failure counting) through a full
    episode deterministically, so that path is exercised end to end rather than
    only in unit tests.
    """

    name = "plan-following"

    ORDER = ["reserve", "authorize", "capture", "ship", "notify", "complete"]

    def __init__(self, malformed_every: int = 0) -> None:
        # Emit garbage every N calls, to prove an episode survives a model that
        # intermittently produces nonsense.
        self._malformed_every = malformed_every
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self._malformed_every and self.calls % self._malformed_every == 0:
            return "Sure! I'll reserve the stock now." # no JSON at all

        completed = ""
        for line in user.splitlines():
            if line.startswith("Completed steps:"):
                completed = line.split(":", 1)[1]
        for step in self.ORDER:
            if step not in completed:
                return json.dumps({"action": step, "reason": "next in plan"})
        return json.dumps({"action": "complete", "reason": "done"})


class RandomPolicy:
    """A deliberately terrible policy — the adversary for the runtime.

    Picks actions at random, in the wrong order, repeating and skipping steps.
    No language model would behave this badly.

    It exists to test the claim that actually matters: **the runtime's safety
    properties do not depend on the policy being sensible.** If a random policy
    can be made to leave money taken with nothing shipped, then the guarantee
    was never about the runtime — it was about the deterministic policy being
    careful, and it would not survive contact with a real model.
    """

    name = "random"

    def __init__(self, seed: int = 0, give_up_after: int = 12) -> None:
        self._rng = random.Random(seed)
        self._give_up_after = give_up_after
        self._steps = 0

    def reset(self) -> None:
        self._steps = 0

    async def next_action(self, observation: Observation) -> Action:
        self._steps += 1
        if self._steps > self._give_up_after or observation.steps_remaining <= 0:
            return Action(ActionKind.GIVE_UP, reason="random policy stopping")
        choices = [k for k in ActionKind if k is not ActionKind.GIVE_UP]
        return Action(self._rng.choice(choices), reason="chosen at random")


class OpenAICompatibleClient:
    """Chat-completions client for any OpenAI-compatible endpoint.

    Covers OpenAI, Groq, Together, OpenRouter and a local llama.cpp or Ollama
    server — one wire format, many providers, no SDK.

    **This is the one piece in the project that has never been run against a
    live endpoint.** It needs an API key. The policy logic above is fully
    tested; this is roughly thirty lines of HTTP beneath it. Treat it as
    unverified until someone points it at a real server.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> None:
        self.name = model
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._timeout = timeout

    async def complete(self, system: str, user: str) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": self._temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
