"""One complete, isolated, reproducible episode.

An episode is: build a world from a seed, run a policy against it under a
runtime config with faults injected at a given rate, then ask the verifier what
state the world was left in.

Isolation is per-episode — its own in-memory database — so the verifier's global
checks (stock conservation) mean something and so episodes can later run
concurrently without interfering.

Reproducibility comes from the seed alone: the same seed yields the same
scenario, the same fault sequence, and the same jitter. A failing episode can be
re-run and debugged instead of shrugged at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from httpx import ASGITransport, AsyncClient

from backstop.chaos.injector import ChaosConfig, ChaosTransport
from backstop.policies.base import Policy
from backstop.runtime.engine import EpisodeResult, EpisodeRunner, RuntimeConfig
from backstop.world.app import create_app
from backstop.world.db import world_session
from backstop.world.scenarios import Scenario, build_scenario, seed_world
from backstop.world.verifier import Violation, verify_order, verify_stock_conservation


@dataclass
class EpisodeOutcome:
    """What an episode produced, judged against the world's final state."""

    seed: int
    scenario: Scenario
    result: EpisodeResult
    fulfilled: bool
    violations: list[Violation] = field(default_factory=list)
    faults_injected: int = 0

    @property
    def has_orphans(self) -> bool:
        return bool(self.violations)

    @property
    def clean_failure(self) -> bool:
        return not self.fulfilled and not self.violations

    @property
    def correct(self) -> bool:
        """Did the system do the right thing?

        Not the same as success. For a satisfiable order the right thing is to
        fulfill it; for an impossible one it is to fail without breaking
        anything. Scoring both as "success rate" would punish a system for
        correctly refusing an impossible task.
        """
        if self.scenario.satisfiable:
            return self.fulfilled and not self.violations
        return not self.fulfilled and not self.violations

    @property
    def orphan_kinds(self) -> set[str]:
        return {v.kind for v in self.violations}


async def run_episode(
    seed: int,
    policy: Policy,
    runtime: RuntimeConfig,
    fault_rate: float = 0.0,
    *,
    impossible_rate: float = 0.15,
    time_scale: float = 0.0,
) -> EpisodeOutcome:
    """Run one episode end to end and judge it.

    ``time_scale=0`` by default: injected latency and backoff sleeps are
    decisions we want exercised, not seconds we want to spend. A sweep of
    thousands of episodes would otherwise be dominated by deliberate waiting.
    """
    scenario = build_scenario(seed, impossible_rate=impossible_rate)

    async with world_session() as (_engine, factory):
        async with factory() as session:
            await seed_world(session, scenario)

        app = create_app(factory)
        chaos = ChaosTransport(
            ASGITransport(app=app),
            ChaosConfig(fault_rate=fault_rate, seed=seed, time_scale=time_scale),
        )
        runtime = _with_time_scale(runtime, time_scale)

        async with AsyncClient(transport=chaos, base_url="http://world") as client:
            runner = EpisodeRunner(client, scenario, policy, runtime)
            result = await runner.run()

        async with factory() as session:
            verdict = await verify_order(session, scenario.order_id)
            violations = list(verdict.violations)
            violations.extend(await verify_stock_conservation(session))

    return EpisodeOutcome(
        seed=seed,
        scenario=scenario,
        result=result,
        fulfilled=verdict.fulfilled,
        violations=violations,
        faults_injected=len(chaos.events),
    )


def _with_time_scale(config: RuntimeConfig, time_scale: float) -> RuntimeConfig:
    """Return a copy of ``config`` with the sweep's time scale applied."""
    from dataclasses import replace

    return replace(config, time_scale=time_scale)
