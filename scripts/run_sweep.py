#!/usr/bin/env python3
"""Run the full measurement and write results.json.

    python scripts/run_sweep.py                 # default: 200 episodes per cell
    python scripts/run_sweep.py --episodes 500

Four experiments:

1. **The curve** — correctness and orphan rate vs fault rate, baseline against
   runtime.
2. **The ablation** — each protection alone, at two fault rates. Two rates
   because at 20% several configurations saturate at 100% and a saturated cell
   tells you nothing about which protection was load-bearing.
3. **The point of no return** — the same runtime with and without the rule that
   a saga may not unwind past an irreversible action. This is where the most
   surprising result lives: unwinding blindly is worse than not unwinding.
4. **pass^k** — consistency, because a single-run success rate flatters an
   unreliable system.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import replace
from pathlib import Path

from backstop.eval.sweep import Metrics, pass_at_k, run_cell, write_results
from backstop.policies.scripted import EagerPolicy, ScriptedPolicy
from backstop.runtime.engine import BASELINE, RUNTIME

FAULT_RATES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75]
HEADLINE_RATE = 0.20
STRESS_RATE = 0.60


def show(metrics: Metrics) -> None:
    kinds = sorted(metrics.orphan_kinds.items(), key=lambda kv: -kv[1])[:3]
    detail = "  ".join(f"{k}={v}" for k, v in kinds)
    print(
        f"  {metrics.label:<26} rate={metrics.fault_rate:<5} "
        f"correct={metrics.correct_rate:>6.1%}  "
        f"orphans={metrics.orphan_rate:>6.1%}  "
        f"calls={metrics.median_tool_calls:>4.0f}  "
        f"retries={metrics.median_retries:>3.0f}   {detail}"
    )


async def main(episodes: int) -> None:
    started = time.monotonic()
    seeds = range(episodes)
    policy = ScriptedPolicy()
    results: dict = {"episodes_per_cell": episodes, "fault_rates": FAULT_RATES}

    print(
        f"\n=== 1. Correctness and orphans vs fault rate "
        f"({episodes} episodes/cell) ==="
    )
    curve = []
    for rate in FAULT_RATES:
        for label, config in (("baseline", BASELINE), ("runtime", RUNTIME)):
            metrics = await run_cell(label, policy, config, rate, seeds)
            show(metrics)
            curve.append(metrics.row())
        print()
    results["curve"] = curve

    ablations = {
        "none (baseline)": BASELINE,
        "+ retries only": replace(BASELINE, max_retries=3, respect_retry_after=True),
        "+ idempotency only": replace(BASELINE, idempotency=True),
        "+ validation only": replace(BASELINE, validate_responses=True),
        "+ compensation only": replace(BASELINE, compensate_on_failure=True),
        "+ preconditions only": replace(BASELINE, enforce_preconditions=True),
        "+ compensation+reconcile": replace(
            BASELINE, compensate_on_failure=True, reconcile=True
        ),
        "retries + idempotency": replace(
            BASELINE, max_retries=3, respect_retry_after=True, idempotency=True
        ),
        "everything (runtime)": RUNTIME,
    }
    ablation_rows = []
    for rate in (HEADLINE_RATE, STRESS_RATE):
        print(f"=== 2. Ablation — one protection at a time, at fault rate {rate} ===")
        for label, config in ablations.items():
            metrics = await run_cell(label, policy, config, rate, seeds)
            show(metrics)
            ablation_rows.append(metrics.row())
        print()
    results["ablation"] = ablation_rows

    print("=== 3. Unwinding past the point of no return ===")
    blind = replace(RUNTIME, respect_point_of_no_return=False)
    pnr_rows = []
    for rate in (0.45, STRESS_RATE, 0.75):
        for label, config in (
            ("unwinds blindly", blind),
            ("respects the PONR", RUNTIME),
        ):
            metrics = await run_cell(label, policy, config, rate, seeds)
            show(metrics)
            pnr_rows.append(metrics.row())
        print()
    results["point_of_no_return"] = pnr_rows

    print("=== 3b. Ordering: irreversible action early vs late ===")
    ordering_rows = []
    for rate in (0.45, STRESS_RATE):
        for label, chosen in (
            ("notify last", ScriptedPolicy()),
            ("notify before ship", EagerPolicy()),
        ):
            metrics = await run_cell(label, chosen, RUNTIME, rate, seeds)
            show(metrics)
            ordering_rows.append(metrics.row())
        print()
    results["ordering"] = ordering_rows

    print("=== 4. pass^k — consistency across repeated attempts ===")
    groups = max(30, episodes // 5)
    passk_rows = []
    for k in (1, 3, 5):
        for label, config in (("baseline", BASELINE), ("runtime", RUNTIME)):
            score = await pass_at_k(policy, config, HEADLINE_RATE, groups, k)
            print(f"  {label:<10} pass^{k} = {score:>6.1%}  ({groups} groups)")
            passk_rows.append(
                {"label": label, "k": k, "groups": groups, "score": round(score, 4)}
            )
    results["pass_k"] = passk_rows

    elapsed = time.monotonic() - started
    results["elapsed_seconds"] = round(elapsed, 1)
    out = Path("results/results.json")
    write_results(out, results)
    print(f"\nWrote {out} in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main(args.episodes))
