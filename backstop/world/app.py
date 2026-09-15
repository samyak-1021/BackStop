"""The simulated ops world as an HTTP API.

Each endpoint is one **tool** the agent can call. They are deliberately written
the way real internal services are — not the way a convenient test double is:

* Mutating calls honour an ``Idempotency-Key`` header. Replaying one returns the
  original response instead of applying the effect twice. An agent that omits
  the key gets no protection, which is precisely how the baseline double-charges.
* Every applied call is written to ``audit_events``, so a run can be
  reconstructed from what the world says happened rather than from what the
  agent claims.
* Failures are real HTTP status codes with structured bodies, not exceptions.

Nothing here knows about chaos or recovery. Fault injection happens in front of
this app (``backstop.chaos``), so the world stays an honest implementation and
the faults stay a property of the *network*, which is where they live in reality.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backstop.world.db import get_session, session_dependency
from backstop.world.models import (
    AuditEvent,
    Idempotency,
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

SessionDep = Annotated[AsyncSession, Depends(get_session)]
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key")]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def _replay(
    session: AsyncSession, endpoint: str, key: str | None
) -> dict[str, Any] | None:
    """Return the stored response for this idempotency key, if any."""
    if key is None:
        return None
    row = (
        await session.scalars(
            select(Idempotency).where(
                Idempotency.endpoint == endpoint, Idempotency.key == key
            )
        )
    ).first()
    return row.response if row else None


async def _record(
    session: AsyncSession,
    endpoint: str,
    key: str | None,
    response: dict[str, Any],
    order_id: str | None = None,
    detail: dict | None = None,
) -> None:
    """Persist the idempotency record and the audit event for an applied call."""
    if key is not None:
        session.add(Idempotency(endpoint=endpoint, key=key, response=response))
    session.add(
        AuditEvent(
            order_id=order_id,
            endpoint=endpoint,
            idempotency_key=key,
            replayed=False,
            detail=detail,
        )
    )


async def _note_replay(
    session: AsyncSession, endpoint: str, key: str | None, order_id: str | None
) -> None:
    """Record that a duplicate delivery was absorbed by an idempotency key.

    Counting these is how the report shows how often the "succeeded but the
    response was lost" fault actually fired — and therefore how much work the
    idempotency layer is really doing.
    """
    session.add(
        AuditEvent(
            order_id=order_id, endpoint=endpoint, idempotency_key=key, replayed=True
        )
    )
    await session.commit()


# --- Request bodies ----------------------------------------------------------


class ReserveRequest(BaseModel):
    order_id: str
    sku: str
    quantity: int = Field(ge=1)


class AuthorizeRequest(BaseModel):
    order_id: str
    amount_cents: int = Field(ge=1)


class CaptureRequest(BaseModel):
    payment_id: str


class RefundRequest(BaseModel):
    payment_id: str


class ShipRequest(BaseModel):
    order_id: str
    reservation_id: str


class NotifyRequest(BaseModel):
    order_id: str
    template: str


def create_app(session_factory=None) -> FastAPI:
    """Build the world app bound to one episode's database.

    ``session_factory`` is per-app rather than global so concurrent episodes
    cannot see each other's data. Without it a sweep silently runs every
    episode against whichever world was bound last.
    """
    app = FastAPI(title="Backstop World", summary="Simulated order-fulfillment ops")
    if session_factory is not None:
        app.dependency_overrides[get_session] = session_dependency(session_factory)

    # --- Reads -------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/orders/{order_id}")
    async def get_order(order_id: str, session: SessionDep) -> dict[str, Any]:
        order = await session.get(Order, order_id)
        if order is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
        return {
            "order_id": order.id,
            "sku": order.sku_code,
            "quantity": order.quantity,
            "amount_cents": order.amount_cents,
            "customer_email": order.customer_email,
            "state": order.state,
        }

    @app.get("/orders/{order_id}/effects")
    async def get_effects(order_id: str, session: SessionDep) -> dict[str, Any]:
        """Everything that has actually happened to this order.

        The reconciliation endpoint. After a lost response an agent genuinely
        does not know whether its call landed, and guessing is how orphans are
        made. Real systems resolve that by reading back the authoritative
        state, so the world exposes it.

        Reads are never fault-injected, which encodes a real and usually-true
        assumption: read paths are more available than write paths (caches,
        replicas, no locks). Where that assumption fails, reconciliation
        degrades and the runtime falls back on what it remembered.
        """
        order = await session.get(Order, order_id)
        if order is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")

        reservations = (
            await session.scalars(
                select(Reservation).where(Reservation.order_id == order_id)
            )
        ).all()
        payments = (
            await session.scalars(select(Payment).where(Payment.order_id == order_id))
        ).all()
        shipments = (
            await session.scalars(select(Shipment).where(Shipment.order_id == order_id))
        ).all()
        notifications = (
            await session.scalars(
                select(Notification).where(Notification.order_id == order_id)
            )
        ).all()

        return {
            "order_id": order_id,
            "state": order.state,
            "reservations": [
                {"reservation_id": r.id, "state": r.state, "quantity": r.quantity}
                for r in reservations
            ],
            "payments": [
                {
                    "payment_id": p.id,
                    "state": p.state,
                    "captured_cents": p.captured_cents,
                }
                for p in payments
            ],
            "shipments": [
                {"shipment_id": s.id, "state": s.state} for s in shipments
            ],
            "notifications": [
                {"notification_id": n.id, "template": n.template}
                for n in notifications
            ],
        }

    @app.get("/inventory/{sku}")
    async def get_stock(sku: str, session: SessionDep) -> dict[str, Any]:
        row = await session.get(Sku, sku)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such sku")
        return {
            "sku": row.code,
            "available": row.on_hand,
            "reserved": row.reserved,
            "unit_price_cents": row.unit_price_cents,
        }

    # --- Inventory ---------------------------------------------------------

    @app.post("/inventory/reservations", status_code=201)
    async def reserve(
        body: ReserveRequest,
        session: SessionDep,
        request: Request,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Hold stock for an order. Compensated by ``release``."""
        cached = await _replay(session, "reserve", idempotency_key)
        if cached is not None:
            await _note_replay(session, "reserve", idempotency_key, body.order_id)
            return cached

        sku = await session.get(Sku, body.sku)
        if sku is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such sku")
        if sku.on_hand < body.quantity:
            # A real business failure, not an injected fault. The agent has to
            # cope with this too, and it must not be retried forever.
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"insufficient stock: {sku.on_hand} on hand, {body.quantity} requested",
            )

        sku.on_hand -= body.quantity
        sku.reserved += body.quantity
        reservation = Reservation(
            id=_new_id("rsv"),
            order_id=body.order_id,
            sku_code=body.sku,
            quantity=body.quantity,
        )
        session.add(reservation)

        response = {"reservation_id": reservation.id, "state": ReservationState.HELD}
        await _record(
            session, "reserve", idempotency_key, response, body.order_id,
            {"sku": body.sku, "quantity": body.quantity},
        )
        await session.commit()
        return response

    @app.delete("/inventory/reservations/{reservation_id}")
    async def release(
        reservation_id: str,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Compensation for ``reserve``: give the stock back.

        Idempotent by construction — releasing an already-released reservation
        is a no-op rather than an error, because a compensation that can fail
        on retry is not much of a compensation.
        """
        reservation = await session.get(Reservation, reservation_id)
        if reservation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such reservation")

        if reservation.state == ReservationState.HELD:
            sku = await session.get(Sku, reservation.sku_code)
            sku.reserved -= reservation.quantity
            sku.on_hand += reservation.quantity
            reservation.state = ReservationState.RELEASED
            await _record(
                session, "release", idempotency_key,
                {"reservation_id": reservation_id, "state": ReservationState.RELEASED},
                reservation.order_id,
            )
            await session.commit()
        return {"reservation_id": reservation_id, "state": reservation.state}

    # --- Payments ----------------------------------------------------------

    @app.post("/payments/authorizations", status_code=201)
    async def authorize(
        body: AuthorizeRequest,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Place a hold on funds. Compensated by ``void``."""
        cached = await _replay(session, "authorize", idempotency_key)
        if cached is not None:
            await _note_replay(session, "authorize", idempotency_key, body.order_id)
            return cached

        payment = Payment(
            id=_new_id("pay"), order_id=body.order_id, amount_cents=body.amount_cents
        )
        session.add(payment)
        response = {
            "payment_id": payment.id,
            "state": PaymentState.AUTHORIZED,
            "amount_cents": body.amount_cents,
        }
        await _record(
            session, "authorize", idempotency_key, response, body.order_id,
            {"amount_cents": body.amount_cents},
        )
        await session.commit()
        return response

    @app.post("/payments/captures", status_code=201)
    async def capture(
        body: CaptureRequest,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Take the money.

        The single most dangerous call in the system, and the one the whole
        idempotency story exists for. Note what happens *without* a key: the
        replay check is skipped, ``capture_count`` increments again, and the
        customer is charged twice. That is not a bug in this endpoint — it is
        the honest behaviour of a payments API, and it is what the verifier
        catches as ``double_charge``.
        """
        cached = await _replay(session, "capture", idempotency_key)
        if cached is not None:
            payment = await session.get(Payment, body.payment_id)
            await _note_replay(
                session, "capture", idempotency_key,
                payment.order_id if payment else None,
            )
            return cached

        payment = await session.get(Payment, body.payment_id)
        if payment is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such payment")
        if payment.state == PaymentState.VOIDED:
            raise HTTPException(status.HTTP_409_CONFLICT, "authorization was voided")

        payment.state = PaymentState.CAPTURED
        payment.capture_count += 1
        payment.captured_cents += payment.amount_cents

        response = {
            "payment_id": payment.id,
            "state": PaymentState.CAPTURED,
            "captured_cents": payment.captured_cents,
        }
        await _record(
            session, "capture", idempotency_key, response, payment.order_id,
            {"amount_cents": payment.amount_cents},
        )
        await session.commit()
        return response

    @app.post("/payments/voids")
    async def void(
        body: RefundRequest,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Compensation for ``authorize`` (before capture). Idempotent."""
        payment = await session.get(Payment, body.payment_id)
        if payment is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such payment")
        if payment.state == PaymentState.CAPTURED:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "already captured — refund instead"
            )
        if payment.state == PaymentState.AUTHORIZED:
            payment.state = PaymentState.VOIDED
            await _record(
                session, "void", idempotency_key,
                {"payment_id": payment.id, "state": PaymentState.VOIDED},
                payment.order_id,
            )
            await session.commit()
        return {"payment_id": payment.id, "state": payment.state}

    @app.post("/payments/refunds")
    async def refund(
        body: RefundRequest,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Compensation for ``capture``: give the money back. Idempotent."""
        payment = await session.get(Payment, body.payment_id)
        if payment is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such payment")
        if payment.state != PaymentState.REFUNDED and payment.captured_cents > 0:
            payment.refunded_cents = payment.captured_cents
            payment.state = PaymentState.REFUNDED
            await _record(
                session, "refund", idempotency_key,
                {"payment_id": payment.id, "state": PaymentState.REFUNDED},
                payment.order_id,
            )
            await session.commit()
        return {"payment_id": payment.id, "state": payment.state}

    # --- Shipping ----------------------------------------------------------

    @app.post("/shipping/shipments", status_code=201)
    async def ship(
        body: ShipRequest,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Consume a reservation and schedule a shipment. Compensated by cancel."""
        cached = await _replay(session, "ship", idempotency_key)
        if cached is not None:
            await _note_replay(session, "ship", idempotency_key, body.order_id)
            return cached

        reservation = await session.get(Reservation, body.reservation_id)
        if reservation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such reservation")
        if reservation.state != ReservationState.HELD:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"reservation is {reservation.state}, not held",
            )

        sku = await session.get(Sku, reservation.sku_code)
        sku.reserved -= reservation.quantity
        sku.shipped += reservation.quantity
        reservation.state = ReservationState.CONSUMED

        shipment = Shipment(
            id=_new_id("shp"),
            order_id=body.order_id,
            reservation_id=reservation.id,
        )
        session.add(shipment)
        response = {"shipment_id": shipment.id, "state": ShipmentState.SCHEDULED}
        await _record(session, "ship", idempotency_key, response, body.order_id)
        await session.commit()
        return response

    @app.delete("/shipping/shipments/{shipment_id}")
    async def cancel_shipment(
        shipment_id: str,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Compensation for ``ship``: recall the goods and restore the stock."""
        shipment = await session.get(Shipment, shipment_id)
        if shipment is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such shipment")

        if shipment.state == ShipmentState.SCHEDULED:
            shipment.state = ShipmentState.CANCELLED
            reservation = await session.get(Reservation, shipment.reservation_id)
            sku = await session.get(Sku, reservation.sku_code)
            # The goods come back to the shelf, not back to "reserved" — the
            # reservation is gone for good once it has been consumed.
            sku.shipped -= reservation.quantity
            sku.on_hand += reservation.quantity
            reservation.state = ReservationState.RELEASED
            await _record(
                session, "cancel_shipment", idempotency_key,
                {"shipment_id": shipment_id, "state": ShipmentState.CANCELLED},
                shipment.order_id,
            )
            await session.commit()
        return {"shipment_id": shipment_id, "state": shipment.state}

    # --- Notifications (no compensation exists) ----------------------------

    @app.post("/notifications", status_code=201)
    async def notify(
        body: NotifyRequest,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Email the customer. **There is no endpoint to undo this.**

        Deliberately irreversible. Any runtime that wants to stay correct has
        to treat this as a point of no return and only reach it once every
        failable step has already succeeded.
        """
        cached = await _replay(session, "notify", idempotency_key)
        if cached is not None:
            await _note_replay(session, "notify", idempotency_key, body.order_id)
            return cached

        order = await session.get(Order, body.order_id)
        if order is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")

        note = Notification(
            id=_new_id("ntf"),
            order_id=body.order_id,
            recipient=order.customer_email,
            template=body.template,
        )
        session.add(note)
        response = {"notification_id": note.id, "sent_to": order.customer_email}
        await _record(
            session, "notify", idempotency_key, response, body.order_id,
            {"template": body.template},
        )
        await session.commit()
        return response

    # --- Order lifecycle ---------------------------------------------------

    @app.post("/orders/{order_id}/complete")
    async def complete(
        order_id: str,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Mark the order done. Records the agent's *claim*, not the truth.

        The world accepts this without checking anything — exactly like a real
        status field would. Whether the claim was justified is the verifier's
        job, and an unjustified one shows up as ``incomplete_completion``.
        """
        order = await session.get(Order, order_id)
        if order is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
        if order.state == OrderState.PENDING:
            order.state = OrderState.COMPLETED
            await _record(
                session, "complete", idempotency_key,
                {"order_id": order_id, "state": OrderState.COMPLETED}, order_id,
            )
            await session.commit()
        return {"order_id": order_id, "state": order.state}

    @app.post("/orders/{order_id}/cancel")
    async def cancel_order(
        order_id: str,
        session: SessionDep,
        idempotency_key: IdempotencyKey = None,
    ) -> dict[str, Any]:
        """Give up on the order. The honest terminal state after unwinding."""
        order = await session.get(Order, order_id)
        if order is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such order")
        if order.state == OrderState.PENDING:
            order.state = OrderState.CANCELLED
            await _record(
                session, "cancel_order", idempotency_key,
                {"order_id": order_id, "state": OrderState.CANCELLED}, order_id,
            )
            await session.commit()
        return {"order_id": order_id, "state": order.state}

    return app
