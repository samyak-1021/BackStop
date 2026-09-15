"""Tests for the world and, more importantly, for the verifier.

The verifier defines what "correct" means for this entire project. If it is
wrong, every number the project produces is wrong — so these tests deliberately
*construct* each kind of orphan and assert it gets caught. A verifier that has
only ever seen healthy worlds has not been tested at all.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backstop.world import verifier as V
from backstop.world.app import create_app
from backstop.world.db import world_session
from backstop.world.models import Sku
from backstop.world.scenarios import build_scenario, seed_world


@pytest_asyncio.fixture
async def world():
    """A fresh world seeded with a satisfiable order, plus an HTTP client."""
    async with world_session() as (_engine, factory):
        scenario = build_scenario(seed=1, impossible_rate=0.0)
        async with factory() as session:
            await seed_world(session, scenario)

        app = create_app(factory)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://world") as client:
            yield client, factory, scenario


async def fulfill(client, scenario, *, idempotency: bool = True) -> dict:
    """Run the happy path end to end, returning the ids it produced."""
    def key(step: str) -> dict:
        return {"Idempotency-Key": f"{scenario.order_id}:{step}"} if idempotency else {}

    r = await client.post(
        "/inventory/reservations",
        json={
            "order_id": scenario.order_id,
            "sku": scenario.sku,
            "quantity": scenario.quantity,
        },
        headers=key("reserve"),
    )
    reservation_id = r.json()["reservation_id"]

    r = await client.post(
        "/payments/authorizations",
        json={"order_id": scenario.order_id, "amount_cents": scenario.amount_cents},
        headers=key("authorize"),
    )
    payment_id = r.json()["payment_id"]

    await client.post(
        "/payments/captures", json={"payment_id": payment_id}, headers=key("capture")
    )
    r = await client.post(
        "/shipping/shipments",
        json={"order_id": scenario.order_id, "reservation_id": reservation_id},
        headers=key("ship"),
    )
    shipment_id = r.json()["shipment_id"]

    await client.post(
        "/notifications",
        json={"order_id": scenario.order_id, "template": "order_shipped"},
        headers=key("notify"),
    )
    await client.post(f"/orders/{scenario.order_id}/complete", headers=key("complete"))

    return {
        "reservation_id": reservation_id,
        "payment_id": payment_id,
        "shipment_id": shipment_id,
    }


# --- The happy path must verify clean ----------------------------------------


async def test_full_fulfillment_verifies_clean(world) -> None:
    client, factory, scenario = world
    await fulfill(client, scenario)

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)
        conservation = await V.verify_stock_conservation(session)

    assert result.fulfilled is True
    assert result.violations == [], [str(v) for v in result.violations]
    assert conservation == []


async def test_doing_nothing_is_a_clean_failure(world) -> None:
    """The baseline for 'gave up without breaking anything'."""
    client, factory, scenario = world
    await client.post(f"/orders/{scenario.order_id}/cancel")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert result.fulfilled is False
    assert result.clean_failure is True


# --- Each orphan kind must be detected ---------------------------------------


async def test_detects_money_taken_but_nothing_shipped(world) -> None:
    """The headline orphan: charged, then the episode died before shipping."""
    client, factory, scenario = world
    r = await client.post(
        "/payments/authorizations",
        json={"order_id": scenario.order_id, "amount_cents": scenario.amount_cents},
    )
    await client.post("/payments/captures", json={"payment_id": r.json()["payment_id"]})
    await client.post(f"/orders/{scenario.order_id}/cancel")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert result.has_orphans
    assert any(v.kind == V.MONEY_TAKEN_NOTHING_SHIPPED for v in result.violations)


async def test_detects_double_charge_when_capture_is_replayed_without_a_key(
    world,
) -> None:
    """Duplicate delivery with no idempotency key charges twice.

    This is not a contrived case — it is exactly what the "succeeded but the
    response was lost" fault produces against an agent that doesn't send keys.
    """
    client, factory, scenario = world
    ids = await fulfill(client, scenario)

    # Replay the capture with no key, as a naive retry would.
    await client.post("/payments/captures", json={"payment_id": ids["payment_id"]})

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert any(v.kind == V.DOUBLE_CHARGE for v in result.violations)


async def test_idempotency_key_prevents_the_double_charge(world) -> None:
    """The same replay, with a key, is absorbed — money moves exactly once."""
    client, factory, scenario = world
    ids = await fulfill(client, scenario)

    for _ in range(5):
        await client.post(
            "/payments/captures",
            json={"payment_id": ids["payment_id"]},
            headers={"Idempotency-Key": f"{scenario.order_id}:capture"},
        )

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert result.fulfilled is True
    assert result.violations == [], [str(v) for v in result.violations]


async def test_detects_leaked_stock(world) -> None:
    """A reservation still held on a terminal order is stock nobody can sell."""
    client, factory, scenario = world
    await client.post(
        "/inventory/reservations",
        json={
            "order_id": scenario.order_id,
            "sku": scenario.sku,
            "quantity": scenario.quantity,
        },
    )
    await client.post(f"/orders/{scenario.order_id}/cancel")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert any(v.kind == V.LEAKED_STOCK for v in result.violations)


async def test_releasing_the_reservation_clears_the_leak(world) -> None:
    client, factory, scenario = world
    r = await client.post(
        "/inventory/reservations",
        json={
            "order_id": scenario.order_id,
            "sku": scenario.sku,
            "quantity": scenario.quantity,
        },
    )
    await client.delete(f"/inventory/reservations/{r.json()['reservation_id']}")
    await client.post(f"/orders/{scenario.order_id}/cancel")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)
        conservation = await V.verify_stock_conservation(session)

    assert result.clean_failure is True
    assert conservation == []


async def test_detects_a_false_shipping_notification(world) -> None:
    """The irreversible orphan: told the customer something that wasn't true."""
    client, factory, scenario = world
    await client.post(
        "/notifications",
        json={"order_id": scenario.order_id, "template": "order_shipped"},
    )
    await client.post(f"/orders/{scenario.order_id}/cancel")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert any(v.kind == V.FALSE_NOTIFICATION for v in result.violations)


async def test_unwinding_cannot_repair_a_false_notification(world) -> None:
    """Compensating every reversible step still leaves the lie in place.

    This is the point of having an irreversible action in the domain: a runtime
    cannot recover its way out of having sent it, it can only avoid sending it
    too early.
    """
    client, factory, scenario = world
    ids = await fulfill(client, scenario)

    # Perfect unwind, in reverse order.
    await client.post("/payments/refunds", json={"payment_id": ids["payment_id"]})
    await client.delete(f"/shipping/shipments/{ids['shipment_id']}")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert any(v.kind == V.FALSE_NOTIFICATION for v in result.violations), (
        "a sent notification must remain a violation after a full unwind"
    )


async def test_detects_shipping_without_payment(world) -> None:
    client, factory, scenario = world
    r = await client.post(
        "/inventory/reservations",
        json={
            "order_id": scenario.order_id,
            "sku": scenario.sku,
            "quantity": scenario.quantity,
        },
    )
    await client.post(
        "/shipping/shipments",
        json={
            "order_id": scenario.order_id,
            "reservation_id": r.json()["reservation_id"],
        },
    )
    await client.post(f"/orders/{scenario.order_id}/complete")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert any(v.kind == V.SHIPPED_WITHOUT_PAYMENT for v in result.violations)


async def test_detects_a_dangling_authorization(world) -> None:
    """Funds left on hold on a finished order."""
    client, factory, scenario = world
    await client.post(
        "/payments/authorizations",
        json={"order_id": scenario.order_id, "amount_cents": scenario.amount_cents},
    )
    await client.post(f"/orders/{scenario.order_id}/cancel")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert any(v.kind == V.DANGLING_AUTHORIZATION for v in result.violations)


async def test_detects_completion_without_the_work(world) -> None:
    """Claiming success is not success."""
    client, factory, scenario = world
    await client.post(f"/orders/{scenario.order_id}/complete")

    async with factory() as session:
        result = await V.verify_order(session, scenario.order_id)

    assert result.fulfilled is False
    assert any(v.kind == V.INCOMPLETE_COMPLETION for v in result.violations)


# --- Stock conservation ------------------------------------------------------


async def test_stock_is_conserved_through_a_full_unwind(world) -> None:
    client, factory, scenario = world
    ids = await fulfill(client, scenario)
    await client.delete(f"/shipping/shipments/{ids['shipment_id']}")

    async with factory() as session:
        assert await V.verify_stock_conservation(session) == []
        sku = await session.get(Sku, scenario.sku)

    assert sku.on_hand == scenario.initial_stock
    assert sku.reserved == 0
    assert sku.shipped == 0


async def test_conservation_check_catches_a_corrupted_ledger(world) -> None:
    """Prove the conservation check can actually fail.

    A check that has never been seen to fail is indistinguishable from one
    that always returns an empty list.
    """
    client, factory, scenario = world
    async with factory() as session:
        sku = await session.get(Sku, scenario.sku)
        sku.on_hand -= 1  # quantity vanishes
        await session.commit()
        violations = await V.verify_stock_conservation(session)

    assert any(v.kind == V.STOCK_NOT_CONSERVED for v in violations)


# --- Business failures are not faults ----------------------------------------


async def test_insufficient_stock_is_a_409_not_a_crash(world) -> None:
    client, factory, scenario = world
    r = await client.post(
        "/inventory/reservations",
        json={
            "order_id": scenario.order_id,
            "sku": scenario.sku,
            "quantity": scenario.initial_stock + 5,
        },
    )
    assert r.status_code == 409
    assert "insufficient stock" in r.json()["detail"]


@pytest.mark.parametrize("seed", range(40))
def test_scenarios_are_deterministic(seed: int) -> None:
    assert build_scenario(seed) == build_scenario(seed)


def test_some_scenarios_are_impossible() -> None:
    """The generator must produce tasks that cannot succeed."""
    scenarios = [build_scenario(s) for s in range(400)]
    impossible = [s for s in scenarios if not s.satisfiable]
    assert 20 < len(impossible) < 120, f"{len(impossible)} impossible of 400"
    for scenario in impossible:
        assert scenario.initial_stock < scenario.quantity
