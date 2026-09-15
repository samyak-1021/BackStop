"""The measurement harness.

Four things get measured, and the second and third are the ones that make this
different from a generic agent benchmark:

**Correctness rate** — did the system do the right thing? For a satisfiable
order that means fulfilling it; for an impossible one it means declining
without breaking anything. Scoring both as "success" would punish a system for
correctly refusing an impossible task, which is a behaviour you want to reward.

**Orphan rate** — how often was the world left inconsistent? Money taken with
nothing shipped, stock leaked, a customer told something untrue. This is a
*correctness* property rather than a performance one: a system is allowed to
fail a task, it is not allowed to leave a mess.

**pass^k** — the fraction of task groups where *all* k attempts succeeded.
A system that succeeds 80% of the time independently passes pass^3 only about
half as often. Single-run success rates systematically flatter unreliable
systems, and reporting pass^k is the cheapest correction for that.

**Recovery cost** — extra tool calls and retries. Reliability that costs 10x
the calls is a different trade-off from reliability that costs 1.4x, and a
report that omits the price is selling something.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median

from backstop.eval.episode import EpisodeOutcome, run_episode
from backstop.policies.base import Policy
from backstop.runtime.engine import RuntimeConfig


@dataclass
class Metrics:
    """Aggregate results for one (config, fault rate) cell."""

    label: str
    fault_rate: float
    episodes: int
    correct: int
    fulfilled: int
    orphaned: int
    clean_failures: int
    faults_injected: int
    median_tool_calls: float
    median_retries: float
    orphan_kinds: dict[str, int] = field(default_factory=dict)
    # How episodes ended. Recorded because the interesting claims about the
    # runtime are about *which* ending it chose — rolling forward past an
    # irreversible step, or escalating rather than unwinding one — and a claim
    # that lives only in prose is a claim nobody can re-check.
    rolled_forward: int = 0
    escalated: int = 0
    gave_up: int = 0
    # Orders that could actually be fulfilled. The rest are impossible by
    # construction, and declining them is the correct answer — so any
    # correctness figure has to be read against this number.
    satisfiable: int = 0

    @property
    def correct_rate(self) -> float:
        return self.correct / self.episodes if self.episodes else 0.0

    @property
    def orphan_rate(self) -> float:
        return self.orphaned / self.episodes if self.episodes else 0.0

    def row(self) -> dict:
        data = asdict(self)
        data["correct_rate"] = round(self.correct_rate, 4)
        data["orphan_rate"] = round(self.orphan_rate, 4)
        return data


def summarise(label: str, fault_rate: float, outcomes: list[EpisodeOutcome]) -> Metrics:
    kinds: Counter[str] = Counter()
    for outcome in outcomes:
        kinds.update(outcome.orphan_kinds)

    return Metrics(
        label=label,
        fault_rate=fault_rate,
        episodes=len(outcomes),
        correct=sum(o.correct for o in outcomes),
        fulfilled=sum(o.fulfilled for o in outcomes),
        orphaned=sum(o.has_orphans for o in outcomes),
        clean_failures=sum(o.clean_failure for o in outcomes),
        faults_injected=sum(o.faults_injected for o in outcomes),
        median_tool_calls=median([o.result.tool_calls for o in outcomes] or [0]),
        median_retries=median([o.result.retries for o in outcomes] or [0]),
        orphan_kinds=dict(kinds),
        rolled_forward=sum(o.result.rolled_forward for o in outcomes),
        escalated=sum(o.result.escalated for o in outcomes),
        gave_up=sum(o.result.gave_up for o in outcomes),
        satisfiable=sum(o.scenario.satisfiable for o in outcomes),
    )


async def run_cell(
    label: str,
    policy: Policy,
    config: RuntimeConfig,
    fault_rate: float,
    seeds: range | list[int],
    concurrency: int = 16,
) -> Metrics:
    """Run every seed for one cell, a few episodes at a time.

    Episodes are fully isolated (their own world, their own RNG), so running
    them concurrently changes nothing about the results — only how long the
    sweep takes.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(seed: int) -> EpisodeOutcome:
        async with semaphore:
            return await run_episode(seed, policy, config, fault_rate)

    outcomes = await asyncio.gather(*(one(seed) for seed in seeds))
    return summarise(label, fault_rate, list(outcomes))


async def pass_at_k(
    policy: Policy,
    config: RuntimeConfig,
    fault_rate: float,
    groups: int,
    k: int,
    concurrency: int = 16,
) -> float:
    """Fraction of task groups where all ``k`` attempts were correct.

    Each attempt uses a different seed, so it faces a different fault sequence
    against the same kind of task — which is the point. Consistency under
    varying conditions is what "reliable" means; getting lucky once is not.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(seed: int) -> EpisodeOutcome:
        async with semaphore:
            return await run_episode(seed, policy, config, fault_rate)

    all_seeds = [(g, g * 1000 + i) for g in range(groups) for i in range(k)]
    outcomes = await asyncio.gather(*(one(seed) for _g, seed in all_seeds))

    by_group: dict[int, list[bool]] = {}
    for (group, _seed), outcome in zip(all_seeds, outcomes, strict=True):
        by_group.setdefault(group, []).append(outcome.correct)

    return sum(all(v) for v in by_group.values()) / groups if groups else 0.0


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
