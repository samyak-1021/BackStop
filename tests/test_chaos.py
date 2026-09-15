"""Tests for the fault injector.

Two properties carry the whole project and are tested hardest:

1. **Determinism.** The same seed produces the same faults. Without it, the
   baseline-vs-runtime comparison is two different worlds and means nothing.
2. **LOST_RESPONSE really applies the effect.** If it didn't, the idempotency
   story would be theatre — the runtime would look like it was defending
   against something that never happens.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from backstop.chaos.faults import FaultKind, ToolFailure
from backstop.chaos.injector import ChaosConfig, ChaosTransport
from backstop.world.app import create_app
from backstop.world.db import world_session
from backstop.world.models import Payment, Reservation, Sku
from backstop.world.scenarios import build_scenario, seed_world

AMPLE_STOCK = 100


@pytest_asyncio.fixture
async def chaotic():
    """Factory: build a client whose transport injects faults.

    Stock is topped up to an amount no test here can exhaust. These tests are
    about *duplication* — the same call landing more than once — and a world
    that runs out of stock partway through starts returning genuine 409s, which
    is a different failure mode entirely and would silently cap the counts
    being asserted. (It did, on the first run: seed 7 ships only 2 units.)
    """
    async with world_session() as (_engine, factory):
        scenario = build_scenario(seed=7, impossible_rate=0.0)
        async with factory() as session:
            await seed_world(session, scenario)
            sku = await session.get(Sku, scenario.sku)
            sku.initial_qty = AMPLE_STOCK
            sku.on_hand = AMPLE_STOCK
            await session.commit()
        app = create_app(factory)

        def make(config: ChaosConfig):
            transport = ChaosTransport(ASGITransport(app=app), config)
            client = AsyncClient(transport=transport, base_url="http://world")
            return client, transport

        yield make, factory, scenario


async def reserve(client, scenario, key: str | None = None, quantity: int = 1):
    """Reserve one unit by default.

    Deliberately one unit rather than the order's full quantity: several of
    these tests retry the same call many times, and reserving the full amount
    each time would exhaust stock and start returning a genuine 409. That is a
    real business failure, not the duplication these tests are about, and it
    would mask the property under test.
    """
    return await client.post(
        "/inventory/reservations",
        json={
            "order_id": scenario.order_id,
            "sku": scenario.sku,
            "quantity": quantity,
        },
        headers={"Idempotency-Key": key} if key else {},
    )


# --- Determinism -------------------------------------------------------------


def test_same_seed_gives_the_same_fault_sequence() -> None:
    def sequence(seed: int) -> list[FaultKind | None]:
        transport = ChaosTransport(
            ASGITransport(app=create_app()),
            ChaosConfig(fault_rate=0.4, seed=seed, time_scale=0.0),
        )
        return [transport._pick_fault() for _ in range(200)]

    assert sequence(42) == sequence(42)
    assert sequence(42) != sequence(43)


def test_fault_rate_is_honoured() -> None:
    for rate in (0.0, 0.1, 0.35, 0.8, 1.0):
        transport = ChaosTransport(
            ASGITransport(app=create_app()),
            ChaosConfig(fault_rate=rate, seed=1, time_scale=0.0),
        )
        faults = [transport._pick_fault() for _ in range(5000)]
        observed = sum(f is not None for f in faults) / len(faults)
        assert abs(observed - rate) < 0.03, f"rate {rate}, observed {observed:.3f}"


def test_the_rng_advances_identically_whether_or_not_a_fault_fires() -> None:
    """The fault stream must not depend on which branch was taken.

    If ``_pick_fault`` drew the kind only when a fault fired, two runs with
    different fault rates would desynchronise and later calls would see
    completely different faults — making a sweep across rates incomparable.
    """
    quiet = ChaosTransport(
        ASGITransport(app=create_app()),
        ChaosConfig(fault_rate=0.0, seed=99, time_scale=0.0),
    )
    noisy = ChaosTransport(
        ASGITransport(app=create_app()),
        ChaosConfig(fault_rate=1.0, seed=99, time_scale=0.0),
    )
    for _ in range(100):
        quiet._pick_fault()
        noisy._pick_fault()
    # Same seed, same number of draws -> same RNG position.
    assert quiet._rng.random() == noisy._rng.random()


def test_zero_fault_rate_injects_nothing() -> None:
    transport = ChaosTransport(
        ASGITransport(app=create_app()),
        ChaosConfig(fault_rate=0.0, seed=5, time_scale=0.0),
    )
    assert all(transport._pick_fault() is None for _ in range(2000))


# --- The dangerous fault -----------------------------------------------------


async def test_lost_response_applies_the_effect_then_fails(chaotic) -> None:
    """The fault the whole runtime exists to survive."""
    make, factory, scenario = chaotic
    client, _ = make(
        ChaosConfig(
            fault_rate=1.0,
            seed=1,
            time_scale=0.0,
            mix={FaultKind.LOST_RESPONSE: 1.0},
        )
    )

    with pytest.raises(ToolFailure) as caught:
        await reserve(client, scenario)
    assert caught.value.kind is FaultKind.LOST_RESPONSE

    # The caller saw a failure, but the world moved anyway.
    async with factory() as session:
        count = await session.scalar(select(func.count()).select_from(Reservation))
    assert count == 1, "LOST_RESPONSE must apply the effect before dropping the reply"


async def test_retrying_a_lost_response_without_a_key_duplicates_the_effect(
    chaotic,
) -> None:
    """The naive failure mode, demonstrated end to end."""
    make, factory, scenario = chaotic
    client, _ = make(
        ChaosConfig(
            fault_rate=1.0, seed=1, time_scale=0.0, mix={FaultKind.LOST_RESPONSE: 1.0}
        )
    )

    for _ in range(3):
        with pytest.raises(ToolFailure):
            await reserve(client, scenario)  # no idempotency key

    async with factory() as session:
        count = await session.scalar(select(func.count()).select_from(Reservation))
    assert count == 3, "three retries without a key should reserve three times"


async def test_retrying_a_lost_response_with_a_key_applies_once(chaotic) -> None:
    """The same storm, survived."""
    make, factory, scenario = chaotic
    client, _ = make(
        ChaosConfig(
            fault_rate=1.0, seed=1, time_scale=0.0, mix={FaultKind.LOST_RESPONSE: 1.0}
        )
    )

    for _ in range(5):
        with pytest.raises(ToolFailure):
            await reserve(client, scenario, key="ord:reserve")

    async with factory() as session:
        count = await session.scalar(select(func.count()).select_from(Reservation))
    assert count == 1, "an idempotency key must collapse the retries into one effect"


async def test_lost_capture_without_a_key_double_charges(chaotic) -> None:
    """The money version of the same bug."""
    make, factory, scenario = chaotic
    clean, _ = make(ChaosConfig(fault_rate=0.0, seed=1, time_scale=0.0))
    r = await clean.post(
        "/payments/authorizations",
        json={"order_id": scenario.order_id, "amount_cents": scenario.amount_cents},
    )
    payment_id = r.json()["payment_id"]

    lossy, _ = make(
        ChaosConfig(
            fault_rate=1.0, seed=2, time_scale=0.0, mix={FaultKind.LOST_RESPONSE: 1.0}
        )
    )
    for _ in range(2):
        with pytest.raises(ToolFailure):
            await lossy.post("/payments/captures", json={"payment_id": payment_id})

    async with factory() as session:
        payment = await session.get(Payment, payment_id)
    assert payment.capture_count == 2, "the customer was charged twice"


# --- The other fault kinds ---------------------------------------------------


async def test_rate_limited_returns_429_with_retry_after(chaotic) -> None:
    make, _factory, scenario = chaotic
    client, _ = make(
        ChaosConfig(
            fault_rate=1.0, seed=1, time_scale=0.0, mix={FaultKind.RATE_LIMITED: 1.0}
        )
    )
    response = await reserve(client, scenario)
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1


async def test_server_error_returns_500(chaotic) -> None:
    make, _factory, scenario = chaotic
    client, _ = make(
        ChaosConfig(
            fault_rate=1.0, seed=1, time_scale=0.0, mix={FaultKind.SERVER_ERROR: 1.0}
        )
    )
    assert (await reserve(client, scenario)).status_code == 500


async def test_truncated_body_looks_like_success_but_will_not_parse(chaotic) -> None:
    """A 200 that breaks anything assuming status code means success."""
    make, _factory, scenario = chaotic
    client, _ = make(
        ChaosConfig(
            fault_rate=1.0, seed=1, time_scale=0.0, mix={FaultKind.TRUNCATED_BODY: 1.0}
        )
    )
    response = await reserve(client, scenario)
    assert response.status_code == 201
    # Specifically a JSON decode failure: the status line says success and the
    # body does not parse, which is exactly the trap this fault sets.
    with pytest.raises(json.JSONDecodeError):
        response.json()


async def test_schema_drift_parses_fine_but_loses_the_handle(chaotic) -> None:
    """The quietest fault: valid JSON, wrong field names, no error anywhere."""
    make, _factory, scenario = chaotic
    client, _ = make(
        ChaosConfig(
            fault_rate=1.0, seed=1, time_scale=0.0, mix={FaultKind.SCHEMA_DRIFT: 1.0}
        )
    )
    response = await reserve(client, scenario)
    body = response.json()
    assert response.status_code == 201
    assert "reservation_id" not in body
    assert "id" in body


async def test_reads_are_never_faulted(chaotic) -> None:
    """GETs are left alone; the risk in this domain is all in the writes."""
    make, _factory, scenario = chaotic
    client, _ = make(ChaosConfig(fault_rate=1.0, seed=1, time_scale=0.0))
    for _ in range(25):
        assert (await client.get(f"/orders/{scenario.order_id}")).status_code == 200


async def test_events_are_recorded_for_attribution(chaotic) -> None:
    """Every injected fault is logged so a report can attribute failures."""
    make, _factory, scenario = chaotic
    client, transport = make(
        ChaosConfig(
            fault_rate=1.0, seed=3, time_scale=0.0, mix={FaultKind.SERVER_ERROR: 1.0}
        )
    )
    await reserve(client, scenario)
    await reserve(client, scenario)
    assert len(transport.events) == 2
    assert all(e.kind is FaultKind.SERVER_ERROR for e in transport.events)
    assert all(not e.applied_to_world for e in transport.events)


def test_retryable_classification_is_sensible() -> None:
    """Contract faults must not be retried; transport faults must be."""
    assert ToolFailure(FaultKind.TIMEOUT, "/x").retryable is True
    assert ToolFailure(FaultKind.LOST_RESPONSE, "/x").retryable is True
    assert ToolFailure(FaultKind.RATE_LIMITED, "/x").retryable is True
    assert ToolFailure(FaultKind.SERVER_ERROR, "/x").retryable is True
    # Retrying these cannot help — the service's contract itself changed.
    assert ToolFailure(FaultKind.SCHEMA_DRIFT, "/x").retryable is False
    assert ToolFailure(FaultKind.TRUNCATED_BODY, "/x").retryable is False
