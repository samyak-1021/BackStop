"""Running one episode: policy decides, runtime executes, verifier judges.

The loop is small on purpose. Everything interesting lives in the switches on
``RuntimeConfig``, because the experiment is "which of these protections
actually prevents which harm?" — and that question only has a clean answer if
each protection can be turned on by itself.

The two named configurations:

``BASELINE``
    What an agent looks like when someone wires an LLM to some tools and ships
    it. No idempotency, no retries, no compensation. It is not a straw man: it
    is the default behaviour of every framework unless you go out of your way.

``RUNTIME``
    Everything on. Idempotent calls, jittered retries that respect
    ``Retry-After``, response validation, and a saga that unwinds in reverse
    when the episode cannot be completed.

One rule the loop enforces regardless of config: **a lost response is treated as
possibly-applied.** When a call fails with a fault that may have reached the
world, the effect is registered in the saga anyway, because the alternative —
assuming failure means nothing happened — is exactly how orphans are created.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from backstop.chaos.faults import APPLIES_THE_EFFECT
from backstop.policies.base import Action, ActionKind, Observation, Policy
from backstop.runtime.saga import CompensationResult, SagaLog
from backstop.runtime.tools import ToolClient, ToolConfig, ToolError
from backstop.world.scenarios import Scenario


@dataclass
class RuntimeConfig:
    """Every reliability protection, independently switchable."""

    idempotency: bool = False
    max_retries: int = 0
    respect_retry_after: bool = False
    validate_responses: bool = False
    compensate_on_failure: bool = False
    # Reject actions whose prerequisites are not met, before they reach a tool.
    # Retries and compensation defend against the *tools* misbehaving; this
    # defends against the *agent* misbehaving, which is a different threat and
    # the one you actually have when a model is driving.
    enforce_preconditions: bool = False
    # Before unwinding, read back what the world says actually happened and
    # rebuild the saga from that. This is what turns "I think I reserved
    # something but the response was lost" into a compensable, named effect.
    reconcile: bool = False
    # How many actions an episode may attempt before it is cut off. A budget is
    # itself a reliability feature: without one, a confused agent loops until
    # something else kills it, and in a real deployment that is a cost incident.
    step_budget: int = 24
    compensation_attempts: int = 3
    # Attempts to push a nearly-finished episode over the line before giving
    # up, used when unwinding is no longer safe.
    roll_forward_attempts: int = 3
    # When False the saga unwinds unconditionally, including past an
    # irreversible action. Kept switchable because it is how the damage that
    # motivated the rule gets measured rather than merely asserted.
    respect_point_of_no_return: bool = True
    time_scale: float = 1.0

    def tool_config(self) -> ToolConfig:
        return ToolConfig(
            idempotency=self.idempotency,
            max_retries=self.max_retries,
            respect_retry_after=self.respect_retry_after,
            validate_responses=self.validate_responses,
            time_scale=self.time_scale,
        )


BASELINE = RuntimeConfig(
    idempotency=False,
    max_retries=0,
    respect_retry_after=False,
    validate_responses=False,
    compensate_on_failure=False,
    reconcile=False,
    enforce_preconditions=False,
)

RUNTIME = RuntimeConfig(
    idempotency=True,
    max_retries=3,
    respect_retry_after=True,
    validate_responses=True,
    compensate_on_failure=True,
    reconcile=True,
    enforce_preconditions=True,
)


@dataclass
class EpisodeResult:
    """Everything one episode produced, before the verifier judges it."""

    order_id: str
    policy: str
    claimed_success: bool
    gave_up: bool
    steps_used: int
    tool_calls: int
    retries: int
    failures: list[str] = field(default_factory=list)
    # Actions the policy proposed that the runtime refused to perform.
    blocked: list[str] = field(default_factory=list)
    compensation: CompensationResult | None = None
    effects_recorded: list[str] = field(default_factory=list)
    # The episode got past its irreversible step and could not be completed or
    # safely unwound. In a real deployment this is the page to a human.
    escalated: bool = False
    rolled_forward: bool = False


class EpisodeRunner:
    """Executes a policy against the world under a given runtime config."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        scenario: Scenario,
        policy: Policy,
        config: RuntimeConfig,
    ) -> None:
        self._scenario = scenario
        self._policy = policy
        self._config = config
        self._tools = ToolClient(client, config.tool_config(), scenario.order_id)
        self._saga = SagaLog()
        self._handles: dict[str, str] = {}
        self._completed: list[ActionKind] = []
        self._failures: list[str] = []
        self._blocked: list[str] = []

    # -- compensations ------------------------------------------------------

    def _release(self, reservation_id: str):
        async def _do() -> None:
            await self._tools.call(
                "DELETE",
                f"/inventory/reservations/{reservation_id}",
                step=f"release:{reservation_id}",
            )

        return _do

    def _void_or_refund(self, payment_id: str, captured: bool):
        async def _do() -> None:
            # Which compensation applies depends on how far the payment got.
            # Calling the wrong one is a 409, so the saga has to know.
            path = "/payments/refunds" if captured else "/payments/voids"
            await self._tools.call(
                "POST", path, step=f"undo-pay:{payment_id}",
                json_body={"payment_id": payment_id},
            )

        return _do

    def _cancel_shipment(self, shipment_id: str):
        async def _do() -> None:
            await self._tools.call(
                "DELETE",
                f"/shipping/shipments/{shipment_id}",
                step=f"cancel-ship:{shipment_id}",
            )

        return _do

    # -- preconditions ------------------------------------------------------

    def _blocked_because(self, action: Action) -> str | None:
        """Why this action must not be performed now, or None if it is fine.

        These are the domain's own invariants, enforced where the agent cannot
        route around them. Each one corresponds directly to a violation the
        verifier checks for, which is the point: a harm the verifier can detect
        should be a harm the runtime can refuse.

        Note what is *not* here. Reserving twice, or authorizing before
        reserving, is wasteful but harmless and reversible — the runtime does
        not police inefficiency, only damage.
        """
        captured = ActionKind.CAPTURE in self._completed
        shipped = ActionKind.SHIP in self._completed

        if action.kind is ActionKind.CAPTURE:
            if "payment_id" not in self._handles:
                return "cannot capture before authorizing"

        elif action.kind is ActionKind.SHIP:
            if "reservation_id" not in self._handles:
                return "cannot ship without a reservation"
            if not captured:
                # Prevents `shipped_without_payment`: goods must never leave
                # before the money is taken.
                return "cannot ship before the payment is captured"

        elif action.kind is ActionKind.NOTIFY:
            if not shipped:
                # Prevents `false_notification`, the irreversible harm. The
                # customer may only be told the order shipped once it has.
                return "cannot tell the customer it shipped before it has"

        elif action.kind is ActionKind.COMPLETE:
            missing = [
                step
                for step in (ActionKind.CAPTURE, ActionKind.SHIP, ActionKind.NOTIFY)
                if step not in self._completed
            ]
            if missing:
                # Prevents `incomplete_completion`: claiming success is not
                # success, and the runtime will not record the claim.
                return f"cannot complete without: {', '.join(missing)}"

        return None

    # -- actions ------------------------------------------------------------

    async def _perform(self, action: Action) -> None:
        """Execute one action and register its effect in the saga."""
        scenario = self._scenario

        if action.kind is ActionKind.RESERVE:
            payload = await self._tools.call(
                "POST", "/inventory/reservations", step="reserve",
                json_body={
                    "order_id": scenario.order_id,
                    "sku": scenario.sku,
                    "quantity": scenario.quantity,
                },
                expect="reservation_id",
            )
            reservation_id = payload["reservation_id"]
            self._handles["reservation_id"] = reservation_id
            self._saga.record("reserve", self._release(reservation_id))

        elif action.kind is ActionKind.AUTHORIZE:
            payload = await self._tools.call(
                "POST", "/payments/authorizations", step="authorize",
                json_body={
                    "order_id": scenario.order_id,
                    "amount_cents": scenario.amount_cents,
                },
                expect="payment_id",
            )
            payment_id = payload["payment_id"]
            self._handles["payment_id"] = payment_id
            self._saga.record(
                "authorize", self._void_or_refund(payment_id, captured=False)
            )

        elif action.kind is ActionKind.CAPTURE:
            payment_id = self._handles["payment_id"]
            await self._tools.call(
                "POST", "/payments/captures", step="capture",
                json_body={"payment_id": payment_id},
                # No `expect`: the runtime never reads a field off this
                # response, and validating one it does not consume turns a
                # drifted-but-successful capture into a failed step. Validate
                # what you use, not everything you receive.
            )
            # Replace the authorize compensation: once captured, voiding is a
            # 409 and only a refund will do.
            self._saga.record(
                "capture", self._void_or_refund(payment_id, captured=True)
            )

        elif action.kind is ActionKind.SHIP:
            payload = await self._tools.call(
                "POST", "/shipping/shipments", step="ship",
                json_body={
                    "order_id": scenario.order_id,
                    "reservation_id": self._handles["reservation_id"],
                },
                expect="shipment_id",
            )
            shipment_id = payload["shipment_id"]
            self._handles["shipment_id"] = shipment_id
            self._saga.record("ship", self._cancel_shipment(shipment_id))

        elif action.kind is ActionKind.NOTIFY:
            await self._tools.call(
                "POST", "/notifications", step="notify",
                json_body={
                    "order_id": scenario.order_id,
                    "template": "order_shipped",
                },
                # Likewise: nothing downstream needs the notification id.
            )
            # No compensation exists, and none ever will. This is the only
            # genuinely irreversible effect in the domain, and the flag is what
            # trips the point-of-no-return rule.
            self._saga.record("notify", compensate=None, irreversible=True)

        elif action.kind is ActionKind.COMPLETE:
            await self._tools.call(
                "POST", f"/orders/{scenario.order_id}/complete", step="complete"
            )

    # -- the loop -----------------------------------------------------------

    async def run(self) -> EpisodeResult:
        self._policy.reset()
        claimed, gave_up, steps = await self._drive(self._config.step_budget)

        compensation: CompensationResult | None = None
        escalated = False
        rolled_forward = False

        if not claimed and self._config.compensate_on_failure:
            if self._config.respect_point_of_no_return and (
                self._saga.past_point_of_no_return
            ):
                # Unwinding would destroy a world that may well be fine, and it
                # cannot undo what the customer has already been told. Finish
                # the job instead; failing that, escalate rather than tear down.
                claimed, _, extra = await self._drive(
                    self._config.roll_forward_attempts, allow_give_up=False
                )
                steps += extra
                rolled_forward = claimed
                escalated = not claimed
            else:
                if self._config.reconcile:
                    # Ask the world what really happened before undoing
                    # anything: the agent's own log is exactly what a lost
                    # response corrupts.
                    await self._reconcile()
                compensation = await self._saga.unwind(
                    attempts=self._config.compensation_attempts
                )
                await self._cancel_order()

        elif not claimed:
            # No compensation configured. The effects stay where they are —
            # the baseline's defining behaviour, and the source of most of its
            # orphans.
            await self._cancel_order()

        return EpisodeResult(
            order_id=self._scenario.order_id,
            policy=self._policy.name,
            claimed_success=claimed,
            gave_up=gave_up,
            steps_used=steps,
            tool_calls=len(self._tools.trace),
            retries=self._tools.retries,
            failures=list(self._failures),
            blocked=list(self._blocked),
            compensation=compensation,
            effects_recorded=[e.name for e in self._saga.effects],
            escalated=escalated,
            rolled_forward=rolled_forward,
        )

    async def _drive(
        self, budget: int, allow_give_up: bool = True
    ) -> tuple[bool, bool, int]:
        """Run the policy for up to ``budget`` actions.

        Returns ``(claimed, gave_up, steps_used)``.

        ``allow_give_up=False`` is roll-forward mode: the policy may still want
        to stop, but stopping is not an option once an irreversible action has
        been performed, so its GIVE_UP is overridden and the rest of the plan
        is pushed through instead.
        """
        steps = 0
        gave_up = False

        while steps < budget:
            observation = Observation(
                order_id=self._scenario.order_id,
                sku=self._scenario.sku,
                quantity=self._scenario.quantity,
                amount_cents=self._scenario.amount_cents,
                handles=dict(self._handles),
                completed=list(self._completed),
                # Roll-forward hides the failure history so the policy proposes
                # the next real step instead of re-deciding to quit on the
                # strength of what has already gone wrong.
                failures=list(self._failures) if allow_give_up else [],
                steps_used=steps,
                steps_remaining=budget - steps,
            )
            action = await self._policy.next_action(observation)
            steps += 1

            if action.kind is ActionKind.GIVE_UP:
                if allow_give_up:
                    gave_up = True
                    break
                action = Action(ActionKind.COMPLETE, reason="forced roll-forward")

            if self._config.enforce_preconditions:
                blocked = self._blocked_because(action)
                if blocked is not None:
                    # Refused before it reaches a tool, so nothing in the world
                    # changes. Reported to the policy as a failure so it can
                    # choose differently next time.
                    self._blocked.append(f"{action.kind}: {blocked}")
                    self._failures.append(f"blocked {action.kind}: {blocked}")
                    continue

            try:
                await self._perform(action)
                self._completed.append(action.kind)
                if action.kind is ActionKind.COMPLETE:
                    return True, gave_up, steps
            except ToolError as error:
                self._failures.append(str(error))
                # A fault that may have reached the world means the effect
                # might exist even though the call "failed". Registering it is
                # what prevents an orphan; assuming otherwise creates one.
                if error.kind in APPLIES_THE_EFFECT:
                    self._note_possible_effect(action)
            except KeyError as error:
                # A handle this action needed was never obtained — the
                # downstream symptom of schema drift that slipped through.
                self._failures.append(f"missing handle {error} for {action.kind}")

        return False, gave_up, steps

    def _note_possible_effect(self, action: Action) -> None:
        """Register a compensation for an effect that may have landed.

        We do not have the handle — the response was lost — so the
        compensation has to be discovered rather than remembered. For effects
        whose handle is unknown, the saga records the fact so the report can
        attribute the orphan even when nothing can be undone.
        """
        self._saga.record(
            f"{action.kind}(possibly-applied)",
            compensate=None,
            # Only a notification is genuinely beyond recall. For everything
            # else this is uncertainty, not irreversibility: reconciliation can
            # still find the effect and undo it.
            irreversible=action.kind is ActionKind.NOTIFY,
            detail={"uncertain": True},
        )

    async def _reconcile(self) -> None:
        """Rebuild the compensation plan from the world's authoritative state.

        Replaces the saga wholesale rather than adding to it. Anything still
        live in the world needs undoing, whether or not the agent knows it did
        it; anything already released or refunded needs nothing. Building the
        plan from observed state rather than remembered actions is what makes
        this robust to the agent having lost track entirely.
        """
        try:
            effects = await self._tools.call(
                "GET", f"/orders/{self._scenario.order_id}/effects", step="reconcile"
            )
        except ToolError:
            self._failures.append("reconciliation read failed")
            return

        rebuilt = SagaLog()

        for reservation in effects.get("reservations", []):
            if reservation.get("state") == "held":
                rebuilt.record(
                    "reserve(reconciled)",
                    self._release(reservation["reservation_id"]),
                    detail=reservation,
                )

        for payment in effects.get("payments", []):
            state = payment.get("state")
            if state == "authorized":
                rebuilt.record(
                    "authorize(reconciled)",
                    self._void_or_refund(payment["payment_id"], captured=False),
                    detail=payment,
                )
            elif state == "captured":
                rebuilt.record(
                    "capture(reconciled)",
                    self._void_or_refund(payment["payment_id"], captured=True),
                    detail=payment,
                )

        for shipment in effects.get("shipments", []):
            if shipment.get("state") == "scheduled":
                rebuilt.record(
                    "ship(reconciled)",
                    self._cancel_shipment(shipment["shipment_id"]),
                    detail=shipment,
                )

        for note in effects.get("notifications", []):
            # Recorded so the report sees it; nothing can undo it.
            rebuilt.record(
                "notify(reconciled)", compensate=None, irreversible=True, detail=note
            )

        self._saga = rebuilt

    async def _cancel_order(self) -> None:
        """Put the order in a terminal state so the verifier can judge it."""
        try:
            await self._tools.call(
                "POST", f"/orders/{self._scenario.order_id}/cancel", step="cancel"
            )
        except ToolError:
            # Even this can fail under chaos. The verifier treats a still-
            # pending order as non-terminal and will not flag leaked stock,
            # so a failure here is recorded but not fatal.
            self._failures.append("could not cancel the order")
