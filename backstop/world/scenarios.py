"""Seeding one episode's world.

A scenario is the starting state plus the order the agent is asked to fulfill.
Everything is derived from a seed so an episode is reproducible: same seed,
same world, same faults, same result. That reproducibility is what makes a
regression debuggable instead of a ghost.

Some scenarios are deliberately *impossible* — not enough stock to fulfill the
order. A correct system fails those cleanly, without leaving an orphan, and a
harness that only ever generates satisfiable tasks never finds out whether its
agent can give up properly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from backstop.world.models import Order, Sku

CATALOGUE = [
    ("widget-a", 1_299),
    ("widget-b", 2_450),
    ("gizmo-c", 899),
    ("doohickey-d", 15_000),
    ("thingamajig-e", 4_100),
]


@dataclass(frozen=True)
class Scenario:
    """One episode's task and the world it starts in."""

    order_id: str
    sku: str
    quantity: int
    amount_cents: int
    customer_email: str
    initial_stock: int
    # True when the catalogue cannot satisfy the order. The only correct
    # outcome is a clean failure, so success is *not* the target here.
    satisfiable: bool

    @property
    def goal(self) -> str:
        """The task as it would be handed to an agent."""
        return (
            f"Fulfill order {self.order_id}: {self.quantity} x {self.sku} "
            f"for {self.customer_email}, total {self.amount_cents} cents. "
            "Reserve stock, take payment, ship it, and tell the customer."
        )


def build_scenario(seed: int, impossible_rate: float = 0.15) -> Scenario:
    """Derive a scenario deterministically from ``seed``."""
    rng = random.Random(seed)
    sku, unit_price = rng.choice(CATALOGUE)
    quantity = rng.randint(1, 3)

    impossible = rng.random() < impossible_rate
    if impossible:
        # Strictly fewer units in stock than the order needs.
        initial_stock = rng.randint(0, quantity - 1)
    else:
        initial_stock = quantity + rng.randint(0, 20)

    return Scenario(
        order_id=f"ord_{seed:06d}",
        sku=sku,
        quantity=quantity,
        amount_cents=unit_price * quantity,
        customer_email=f"customer{seed % 997}@example.com",
        initial_stock=initial_stock,
        satisfiable=not impossible,
    )


async def seed_world(session: AsyncSession, scenario: Scenario) -> None:
    """Write the scenario's starting state into a fresh world."""
    for code, unit_price in CATALOGUE:
        stock = scenario.initial_stock if code == scenario.sku else 50
        session.add(
            Sku(
                code=code,
                unit_price_cents=unit_price,
                initial_qty=stock,
                on_hand=stock,
                reserved=0,
                shipped=0,
            )
        )
    session.add(
        Order(
            id=scenario.order_id,
            sku_code=scenario.sku,
            quantity=scenario.quantity,
            amount_cents=scenario.amount_cents,
            customer_email=scenario.customer_email,
        )
    )
    await session.commit()
