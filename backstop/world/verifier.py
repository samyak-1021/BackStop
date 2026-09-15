"""Ground truth: is the world consistent after an episode?

Agent benchmarks routinely score themselves on whether the agent *reported*
success. That measures the agent's self-belief, not its effect. Everything here
ignores the agent entirely and reads the database.

Two kinds of outcome, and the distinction matters more than the success rate:

**Task success** — the order actually got fulfilled end to end. A task can fail
for boring reasons (the tools were down, the agent gave up) and that is fine;
a correct system is allowed to fail a task.

**Orphans** — the world was left inconsistent. Money taken with nothing shipped.
Stock reserved forever. A customer told their order shipped when it didn't.
These are *not* allowed, ever, at any fault rate. A system that fails a task
cleanly is working; a system that fails a task and leaves an orphan is broken.

That split is the whole thesis. "Success dropped under load" is a performance
story. "We charged people and shipped nothing" is an incident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backstop.world.models import (
    Notification,
    Order,
    OrderState,
    Payment,
    PaymentState,
    Reservation,
    ReservationState,
    Shipment,
    ShipmentState,
    Sku,
)


class OrphanKind(str):
    """Marker type; the constants below are the vocabulary of a violation."""


# Each constant is one way the world can be left wrong. They are deliberately
# phrased as the harm, not the mechanism — that is how they read in a report.
MONEY_TAKEN_NOTHING_SHIPPED = "money_taken_nothing_shipped"
SHIPPED_WITHOUT_PAYMENT = "shipped_without_payment"
DOUBLE_CHARGE = "double_charge"
LEAKED_STOCK = "leaked_stock"
STOCK_NOT_CONSERVED = "stock_not_conserved"
FALSE_NOTIFICATION = "false_notification"
INCOMPLETE_COMPLETION = "incomplete_completion"
DANGLING_AUTHORIZATION = "dangling_authorization"


@dataclass
class Violation:
    """One inconsistency found in the world."""

    kind: str
    order_id: str | None
    detail: str

    def __str__(self) -> str:  # pragma: no cover - presentation only
        where = f" [{self.order_id}]" if self.order_id else ""
        return f"{self.kind}{where}: {self.detail}"


@dataclass
class VerificationResult:
    """The verdict on one episode."""

    fulfilled: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def has_orphans(self) -> bool:
        return bool(self.violations)

    @property
    def clean_failure(self) -> bool:
        """Didn't finish the job, but left nothing broken behind.

        This is the outcome a correct system produces when the world is too
        degraded to succeed. It is a *good* result, and counting it separately
        from a dirty failure is the point of the whole verifier.
        """
        return not self.fulfilled and not self.violations


async def verify_order(session: AsyncSession, order_id: str) -> VerificationResult:
    """Check every invariant for one order and decide whether it was fulfilled."""
    order = await session.get(Order, order_id)
    if order is None:
        return VerificationResult(
            fulfilled=False,
            violations=[Violation("missing_order", order_id, "no such order")],
        )

    violations: list[Violation] = []

    reservations = list(
        (
            await session.scalars(
                select(Reservation).where(Reservation.order_id == order_id)
            )
        ).all()
    )
    payments = list(
        (await session.scalars(select(Payment).where(Payment.order_id == order_id))).all()
    )
    shipments = list(
        (
            await session.scalars(select(Shipment).where(Shipment.order_id == order_id))
        ).all()
    )
    notifications = list(
        (
            await session.scalars(
                select(Notification).where(Notification.order_id == order_id)
            )
        ).all()
    )

    live_shipments = [s for s in shipments if s.state == ShipmentState.SCHEDULED]
    net_captured = sum(
        p.captured_cents - p.refunded_cents
        for p in payments
        if p.state in (PaymentState.CAPTURED, PaymentState.REFUNDED)
    )
    held = [r for r in reservations if r.state == ReservationState.HELD]
    terminal = order.state != OrderState.PENDING

    # --- Money -------------------------------------------------------------

    if net_captured > 0 and not live_shipments:
        violations.append(
            Violation(
                MONEY_TAKEN_NOTHING_SHIPPED,
                order_id,
                f"{net_captured}c held with no live shipment",
            )
        )

    if live_shipments and net_captured <= 0:
        violations.append(
            Violation(
                SHIPPED_WITHOUT_PAYMENT,
                order_id,
                f"{len(live_shipments)} shipment(s) but net captured {net_captured}c",
            )
        )

    for payment in payments:
        # The signature of a failed idempotency key: a retry that moved money
        # a second time.
        if payment.capture_count > 1:
            violations.append(
                Violation(
                    DOUBLE_CHARGE,
                    order_id,
                    f"payment {payment.id} captured {payment.capture_count} times "
                    f"({payment.captured_cents}c total)",
                )
            )
        if payment.captured_cents > payment.amount_cents:
            violations.append(
                Violation(
                    DOUBLE_CHARGE,
                    order_id,
                    f"payment {payment.id} captured {payment.captured_cents}c "
                    f"against a {payment.amount_cents}c authorization",
                )
            )
        # An authorization left open on a finished order keeps the customer's
        # funds on hold. Less severe than a capture, still an orphan.
        if terminal and payment.state == PaymentState.AUTHORIZED:
            violations.append(
                Violation(
                    DANGLING_AUTHORIZATION,
                    order_id,
                    f"payment {payment.id} still authorized on a {order.state} order",
                )
            )

    # --- Stock -------------------------------------------------------------

    if terminal and held:
        violations.append(
            Violation(
                LEAKED_STOCK,
                order_id,
                f"{len(held)} reservation(s) still held on a {order.state} order",
            )
        )

    # --- Truthfulness ------------------------------------------------------

    # The irreversible one. A shipping notification is a promise; if no
    # shipment exists, that promise was false and cannot be withdrawn.
    for note in notifications:
        if note.template == "order_shipped" and not live_shipments:
            violations.append(
                Violation(
                    FALSE_NOTIFICATION,
                    order_id,
                    "customer told the order shipped, but no live shipment exists "
                    "(irreversible)",
                )
            )

    # --- Completion --------------------------------------------------------

    fulfilled = (
        order.state == OrderState.COMPLETED
        and bool(live_shipments)
        and net_captured == order.amount_cents
        and any(n.template == "order_shipped" for n in notifications)
    )

    if order.state == OrderState.COMPLETED and not fulfilled:
        violations.append(
            Violation(
                INCOMPLETE_COMPLETION,
                order_id,
                "order marked completed without the full set of effects "
                f"(shipments={len(live_shipments)}, captured={net_captured}c of "
                f"{order.amount_cents}c, notified="
                f"{any(n.template == 'order_shipped' for n in notifications)})",
            )
        )

    return VerificationResult(fulfilled=fulfilled, violations=violations)


async def verify_stock_conservation(session: AsyncSession) -> list[Violation]:
    """Nothing is created or destroyed — only moved between the three buckets.

    A global check rather than a per-order one, because the interesting bug is
    a *leak*: quantity that left ``on_hand`` under a failed episode and never
    came back. Per-order checks can't see that.
    """
    violations: list[Violation] = []
    for sku in (await session.scalars(select(Sku))).all():
        total = sku.on_hand + sku.reserved + sku.shipped
        if total != sku.initial_qty:
            violations.append(
                Violation(
                    STOCK_NOT_CONSERVED,
                    None,
                    f"sku {sku.code}: on_hand {sku.on_hand} + reserved {sku.reserved}"
                    f" + shipped {sku.shipped} = {total}, expected {sku.initial_qty}",
                )
            )
    return violations


async def verify_world(
    session: AsyncSession, order_ids: list[str] | None = None
) -> VerificationResult:
    """Verify every order (or a given subset) plus global stock conservation.

    ``fulfilled`` here means *all* orders were fulfilled — the strict reading,
    because a partially fulfilled batch is not a success.
    """
    if order_ids is None:
        order_ids = [o.id for o in (await session.scalars(select(Order))).all()]

    violations: list[Violation] = []
    all_fulfilled = bool(order_ids)
    for order_id in order_ids:
        result = await verify_order(session, order_id)
        violations.extend(result.violations)
        all_fulfilled = all_fulfilled and result.fulfilled

    violations.extend(await verify_stock_conservation(session))
    return VerificationResult(fulfilled=all_fulfilled, violations=violations)
