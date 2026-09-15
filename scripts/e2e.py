#!/usr/bin/env python3
"""End-to-end check against a *real* server over *real* sockets.

Every number this project publishes is produced in-process: the sweep drives the
world through an ASGI transport, which is fast and correct but is not the thing
you would deploy. This script closes that gap.

It boots the world as an actual uvicorn process against a file-backed SQLite
database, points the same chaos injector and the same runtime at
``http://127.0.0.1:...``, and then verifies the result by opening that database
directly — so the verdict comes from the server's own storage rather than from
anything the client believed.

What it would catch that the in-process suite cannot: a real socket timing out
differently from a raised exception, connection handling that only works with
ASGI's in-memory shortcut, and anything that depends on the app being freshly
constructed per call rather than served by a long-lived process.

    python scripts/e2e.py
    python scripts/e2e.py --episodes 12
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backstop.chaos.injector import ChaosConfig, ChaosTransport
from backstop.policies.scripted import ScriptedPolicy
from backstop.runtime.engine import BASELINE, RUNTIME, EpisodeRunner
from backstop.world.models import Base
from backstop.world.scenarios import build_scenario, seed_world
from backstop.world.verifier import verify_order, verify_stock_conservation

_passed = 0
_failed: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    global _passed
    if ok:
        _passed += 1
        print(f"  \033[32mPASS\033[0m  {label}")
    else:
        _failed.append(label)
        print(f"  \033[31mFAIL\033[0m  {label}{f' — {detail}' if detail else ''}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_for_health(base_url: str, timeout: float = 45.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                if (await client.get("/health")).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.3)
    return False


async def prepare_database(db_path: Path) -> None:
    """Create the schema for one episode's file-backed world."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def main(episodes: int) -> int:
    workdir = Path(tempfile.mkdtemp(prefix="backstop-e2e-"))
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    db_path = workdir / "world.db"

    await prepare_database(db_path)

    # The server process builds its own app bound to this database file. It
    # shares nothing with this process except the file on disk, which is the
    # point: the client has no in-memory shortcut to the world.
    bootstrap = workdir / "serve.py"
    bootstrap.write_text(
        "import os\n"
        "from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine\n"
        "from backstop.world.app import create_app\n"
        "engine = create_async_engine(os.environ['BACKSTOP_DB_URL'])\n"
        "factory = async_sessionmaker(engine, expire_on_commit=False)\n"
        "app = create_app(factory)\n"
    )

    env = {
        **os.environ,
        "BACKSTOP_DB_URL": f"sqlite+aiosqlite:///{db_path}",
        "PYTHONPATH": str(Path.cwd()) + os.pathsep + str(workdir),
    }
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "serve:app",
            "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning",
        ],
        env=env,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        section("Server")
        healthy = await wait_for_health(base_url)
        if not healthy:
            server.terminate()
            out = server.stdout.read().decode() if server.stdout else ""
            print(out[-2000:])
            raise SystemExit("the world never became healthy over HTTP")
        check("world serves /health over a real socket", True)

        engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        factory = async_sessionmaker(engine, expire_on_commit=False)

        # --- A clean episode over real HTTP -----------------------------

        section("Clean episode over real HTTP")
        scenario = build_scenario(seed=101, impossible_rate=0.0)
        async with factory() as session:
            await seed_world(session, scenario)

        async with httpx.AsyncClient(base_url=base_url, timeout=20.0) as client:
            runner = EpisodeRunner(client, scenario, ScriptedPolicy(), RUNTIME)
            result = await runner.run()

        async with factory() as session:
            verdict = await verify_order(session, scenario.order_id)

        check("episode claims success", result.claimed_success, str(result.failures))
        check("world agrees it was fulfilled", verdict.fulfilled)
        check(
            "no orphans",
            not verdict.violations,
            "; ".join(str(v) for v in verdict.violations),
        )

        # --- Degraded episodes over real HTTP ---------------------------

        section(f"Degraded episodes over real HTTP ({episodes} each, 30% faults)")
        for label, config in (("baseline", BASELINE), ("runtime", RUNTIME)):
            correct = 0
            orphaned = 0
            for i in range(episodes):
                seed = 200 + i
                scenario = build_scenario(seed, impossible_rate=0.0)
                # A fresh database file per episode keeps the global stock
                # invariant meaningful, exactly as the in-process harness does.
                episode_db = workdir / f"ep-{label}-{i}.db"
                await prepare_database(episode_db)
                ep_engine = create_async_engine(f"sqlite+aiosqlite:///{episode_db}")
                ep_factory = async_sessionmaker(ep_engine, expire_on_commit=False)
                async with ep_factory() as session:
                    await seed_world(session, scenario)

                ep_port = free_port()
                ep_server = subprocess.Popen(
                    [
                        sys.executable, "-m", "uvicorn", "serve:app",
                        "--host", "127.0.0.1", "--port", str(ep_port),
                        "--log-level", "warning",
                    ],
                    env={
                        **env,
                        "BACKSTOP_DB_URL": f"sqlite+aiosqlite:///{episode_db}",
                    },
                    cwd=workdir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                ep_url = f"http://127.0.0.1:{ep_port}"
                try:
                    if not await wait_for_health(ep_url):
                        raise SystemExit(f"episode server on {ep_port} never started")
                    transport = ChaosTransport(
                        httpx.AsyncHTTPTransport(),
                        # time_scale stays real here: this run is about proving
                        # the thing works over a socket, and real sleeps are
                        # part of that.
                        ChaosConfig(fault_rate=0.30, seed=seed, time_scale=0.05),
                    )
                    async with httpx.AsyncClient(
                        transport=transport, base_url=ep_url, timeout=20.0
                    ) as client:
                        runner = EpisodeRunner(client, scenario, ScriptedPolicy(), config)
                        await runner.run()

                    async with ep_factory() as session:
                        verdict = await verify_order(session, scenario.order_id)
                        violations = list(verdict.violations)
                        violations.extend(await verify_stock_conservation(session))
                    if verdict.fulfilled and not violations:
                        correct += 1
                    if violations:
                        orphaned += 1
                finally:
                    ep_server.terminate()
                    ep_server.wait(timeout=10)
                    await ep_engine.dispose()

            print(
                f"    {label:<9} correct={correct}/{episodes}  orphans={orphaned}"
            )
            if label == "runtime":
                check(
                    "runtime leaves no orphans over real HTTP",
                    orphaned == 0,
                    f"{orphaned} orphaned episodes",
                )
                check(
                    "runtime is correct on most episodes over real HTTP",
                    correct >= episodes - 1,
                    f"{correct}/{episodes}",
                )
            else:
                check(
                    "baseline still degrades over real HTTP",
                    correct < episodes,
                    "baseline unexpectedly perfect — is chaos reaching the socket?",
                )

        await engine.dispose()

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            server.kill()
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\n{'=' * 56}")
    if _failed:
        print(f"\033[31m{len(_failed)} FAILED\033[0m, {_passed} passed")
        for name in _failed:
            print(f"  - {name}")
        return 1
    print(f"\033[32mAll {_passed} end-to-end checks passed over real HTTP\033[0m")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.episodes)))
