"""Database wiring for the world.

Each episode gets its **own** database. That is not a performance choice, it is
a correctness one: the verifier asserts global properties like stock
conservation, which only mean anything if no other episode is touching the same
tables. Isolated worlds also make an episode reproducible from its seed alone.

SQLite in memory is the default because a sweep runs thousands of episodes and
each needs a fresh world. A file-backed URL works identically when you want to
open one up afterwards and look at what happened.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from backstop.world.models import Base

# A placeholder dependency. It is never called directly — each app instance
# overrides it with a closure over *its own* session factory (see
# ``create_app``). An earlier version kept the factory in a module-level global,
# which worked perfectly until episodes ran concurrently and promptly began
# sharing one world. Per-app binding is what actually makes an episode isolated.


async def create_world_engine(url: str = "sqlite+aiosqlite:///:memory:") -> AsyncEngine:
    """Build an engine with the schema already created.

    ``StaticPool`` plus a shared connection is required for in-memory SQLite:
    without it every pooled connection gets its *own* blank database and the
    schema vanishes between calls.
    """
    engine = create_async_engine(
        url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool if ":memory:" in url else None,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


def bind(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build a session factory for one episode's engine."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:  # pragma: no cover
    """Placeholder dependency; every app overrides this with its own factory."""
    raise RuntimeError(
        "world database is not bound — pass a session factory to create_app()"
    )


def session_dependency(factory: async_sessionmaker[AsyncSession]):
    """Build the per-app replacement for ``get_session``."""

    async def _dependency() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    return _dependency


@contextlib.asynccontextmanager
async def world_session(
    url: str = "sqlite+aiosqlite:///:memory:",
) -> AsyncGenerator[tuple[AsyncEngine, async_sessionmaker[AsyncSession]], None]:
    """Create, bind and dispose one episode's world."""
    engine = await create_world_engine(url)
    factory = bind(engine)
    try:
        yield engine, factory
    finally:
        await engine.dispose()
