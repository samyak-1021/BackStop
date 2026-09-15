"""The simulated world's data model.

This is the **ground truth** the whole project is measured against. Success is
never "the agent said it finished" — it is "these tables ended up in a
consistent state", checked by ``backstop.world.verifier``.

The domain is order fulfillment, chosen because its compensations are real and
unambiguous:

===================  =========================  ==========================
Step                 Side effect                Compensation
===================  =========================  ==========================
reserve inventory    stock moves to ``reserved``  release the reservation
authorize payment    funds held                 void the authorization
capture payment      funds taken                refund the capture
create shipment      goods leave the warehouse  cancel the shipment
notify customer      an email is sent           **none — irreversible**
===================  =========================  ==========================

That last row is the interesting one. Because a notification cannot be
un-sent, any runtime that wants to stay correct has to order its side effects
so irreversible ones happen *last* — after everything that could still fail.
A naive agent that notifies early can be left having told a customer their
order shipped when it did not, and there is no recovery from that.

Every mutating operation records an ``Idempotency`` row keyed by the caller's
``Idempotency-Key``. That is what makes the "call succeeded but the response
was lost" fault survivable: a replay returns the original result instead of
charging the customer twice.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, DateTime


class Base(DeclarativeBase):
    """Declarative base for every world table."""


def _now() -> datetime:
    return datetime.now(UTC)


class OrderState(StrEnum):
    """Where an order is in its lifecycle.

    ``PENDING`` is the only non-terminal state. The verifier treats anything
    else as "the episode is over, the world had better be consistent".
    """

    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReservationState(StrEnum):
    HELD = "held"
    CONSUMED = "consumed"  # turned into a shipment
    RELEASED = "released"  # compensated


class PaymentState(StrEnum):
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    VOIDED = "voided"  # compensated before capture
    REFUNDED = "refunded"  # compensated after capture


class ShipmentState(StrEnum):
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"


class Sku(Base):
    """A stockable item.

    Quantities are split three ways rather than kept as one number, so the
    verifier can assert conservation: nothing is ever created or destroyed,
    only moved between on-hand, reserved and shipped.
    """

    __tablename__ = "skus"

    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    unit_price_cents: Mapped[int] = mapped_column()
    initial_qty: Mapped[int] = mapped_column()
    on_hand: Mapped[int] = mapped_column()
    reserved: Mapped[int] = mapped_column(default=0)
    shipped: Mapped[int] = mapped_column(default=0)


class Order(Base):
    """One unit of work the agent is asked to complete."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sku_code: Mapped[str] = mapped_column(ForeignKey("skus.code"))
    quantity: Mapped[int] = mapped_column()
    amount_cents: Mapped[int] = mapped_column()
    customer_email: Mapped[str] = mapped_column(String(256))
    state: Mapped[str] = mapped_column(String(16), default=OrderState.PENDING)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    reservations: Mapped[list[Reservation]] = relationship(
        back_populates="order", lazy="selectin"
    )
    payments: Mapped[list[Payment]] = relationship(
        back_populates="order", lazy="selectin"
    )
    shipments: Mapped[list[Shipment]] = relationship(
        back_populates="order", lazy="selectin"
    )
    notifications: Mapped[list[Notification]] = relationship(
        back_populates="order", lazy="selectin"
    )


class Reservation(Base):
    """Stock held for an order, before it ships."""

    __tablename__ = "reservations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    sku_code: Mapped[str] = mapped_column(ForeignKey("skus.code"))
    quantity: Mapped[int] = mapped_column()
    state: Mapped[str] = mapped_column(String(16), default=ReservationState.HELD)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    order: Mapped[Order] = relationship(back_populates="reservations")


class Payment(Base):
    """An authorization, and whatever happened to it afterwards.

    Authorization and capture are one row rather than two because a capture
    only ever belongs to exactly one authorization — keeping them together is
    what lets the verifier spot a double capture as a single-row invariant
    instead of a join.
    """

    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    amount_cents: Mapped[int] = mapped_column()
    state: Mapped[str] = mapped_column(String(16), default=PaymentState.AUTHORIZED)
    # Counts how many times capture actually moved money. Must never exceed 1;
    # this is the counter that catches an idempotency failure under a retry.
    capture_count: Mapped[int] = mapped_column(default=0)
    captured_cents: Mapped[int] = mapped_column(default=0)
    refunded_cents: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    order: Mapped[Order] = relationship(back_populates="payments")


class Shipment(Base):
    """Goods scheduled to leave the warehouse."""

    __tablename__ = "shipments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    reservation_id: Mapped[str] = mapped_column(ForeignKey("reservations.id"))
    state: Mapped[str] = mapped_column(String(16), default=ShipmentState.SCHEDULED)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    order: Mapped[Order] = relationship(back_populates="shipments")


class Notification(Base):
    """An email that has been sent.

    There is deliberately no ``state`` column and no compensation endpoint.
    Once a row exists here, the customer has been told something. If that
    statement was false, the world is permanently wrong — which is the point
    this table exists to make.
    """

    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), index=True)
    recipient: Mapped[str] = mapped_column(String(256))
    template: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    order: Mapped[Order] = relationship(back_populates="notifications")


class Idempotency(Base):
    """A record of a mutating call that has already been applied.

    Keyed by (endpoint, key) so the same idempotency key used against two
    different endpoints doesn't collide. ``response`` stores the original
    reply verbatim, so a replay is indistinguishable from the first call.

    This table is the entire defence against the "succeeded but the response
    was lost" fault. An agent that omits the key gets no protection — which is
    exactly what the baseline does, and why it double-charges.
    """

    __tablename__ = "idempotency"
    __table_args__ = (UniqueConstraint("endpoint", "key", name="uq_idem"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    endpoint: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(128), index=True)
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditEvent(Base):
    """Every mutating call the world actually applied.

    Distinct from the agent's own log on purpose: this is what the world says
    happened, so a run can be reconstructed even when the agent's account of
    it is wrong. ``replayed`` marks a call that an idempotency key absorbed —
    counting those is how we measure how often duplicate delivery occurred.
    """

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    endpoint: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str | None] = mapped_column(String(128), default=None)
    replayed: Mapped[bool] = mapped_column(default=False)
    detail: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
