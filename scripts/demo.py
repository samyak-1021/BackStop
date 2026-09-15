#!/usr/bin/env python3
"""Watch a single episode, step by step.

The sweep gives you percentages. This gives you one story: what the agent tried,
which calls the network broke, what the runtime did about it, and what the world
looked like afterwards.

    python scripts/demo.py                          # a clean run
    python scripts/demo.py --fault-rate 0.4         # a rough one
    python scripts/demo.py --fault-rate 0.4 --baseline
    python scripts/demo.py --seed 17 --compare      # both, same faults
    python scripts/demo.py --policy random          # a badly-behaved agent

``--compare`` is the useful one: identical seed means identical scenario and an
identical fault sequence, so any difference in the outcome is attributable to
the runtime and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio

from backstop.eval.episode import EpisodeOutcome, run_episode
from backstop.policies.llm import LLMPolicy, PlanFollowingModel, RandomPolicy
from backstop.policies.scripted import EagerPolicy, ScriptedPolicy
from backstop.runtime.engine import BASELINE, RUNTIME

DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
OFF = "\033[0m"


def build_policy(name: str, seed: int):
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "eager":
        return EagerPolicy()
    if name == "llm":
        return LLMPolicy(PlanFollowingModel())
    return ScriptedPolicy()


def show(outcome: EpisodeOutcome, label: str) -> None:
    scenario = outcome.scenario
    result = outcome.result

    print(f"\n{BOLD}{'=' * 64}{OFF}")
    print(f"{BOLD}{label}{OFF}")
    print(f"{BOLD}{'=' * 64}{OFF}")

    print(f"\n{BOLD}Task{OFF}")
    print(f"  {scenario.goal}")
    print(f"  {DIM}stock on hand: {scenario.initial_stock}   "
          f"satisfiable: {scenario.satisfiable}{OFF}")

    print(f"\n{BOLD}What the network did{OFF}")
    if outcome.faults_injected == 0:
        print(f"  {DIM}no faults injected{OFF}")
    else:
        print(f"  {outcome.faults_injected} faults injected")

    print(f"\n{BOLD}What the agent got through{OFF}")
    print(f"  completed steps : {', '.join(result.effects_recorded) or 'none'}")
    print(f"  tool calls      : {result.tool_calls}")
    print(f"  retries         : {result.retries}")
    print(f"  steps used      : {result.steps_used}")

    if result.blocked:
        print(f"\n{BOLD}Actions the runtime refused{OFF}  "
              f"{DIM}(preconditions — the agent proposed real harm){OFF}")
        for entry in result.blocked[:6]:
            print(f"  {YELLOW}blocked{OFF}  {entry}")
        if len(result.blocked) > 6:
            print(f"  {DIM}... and {len(result.blocked) - 6} more{OFF}")

    if result.failures:
        print(f"\n{BOLD}Failures the agent saw{OFF}")
        for failure in result.failures[:6]:
            print(f"  {RED}x{OFF} {failure}")
        if len(result.failures) > 6:
            print(f"  {DIM}... and {len(result.failures) - 6} more{OFF}")

    print(f"\n{BOLD}How it ended{OFF}")
    if result.claimed_success and result.rolled_forward:
        print(f"  {BLUE}rolled forward{OFF} — past the point of no return, "
              "so it finished rather than unwound")
    elif result.claimed_success:
        print(f"  {GREEN}claimed success{OFF}")
    elif result.escalated:
        print(f"  {YELLOW}escalated to a human{OFF} — could not finish, "
              "and unwinding would have destroyed a real shipment")
    elif result.gave_up:
        print("  gave up")
    else:
        print("  ran out of budget")

    if result.compensation:
        comp = result.compensation
        print(f"  unwound         : {', '.join(comp.compensated) or 'nothing'}")
        if comp.failed:
            print(f"  {RED}failed to undo  : "
                  f"{', '.join(n for n, _ in comp.failed)}{OFF}")
        if comp.irreversible:
            print(f"  {YELLOW}could not undo  : "
                  f"{', '.join(comp.irreversible)}{OFF}")

    print(f"\n{BOLD}What the world says{OFF}  {DIM}(read from the database){OFF}")
    if outcome.fulfilled:
        print(f"  {GREEN}order fulfilled{OFF}")
    else:
        print("  order not fulfilled")

    if outcome.violations:
        print(f"  {RED}{len(outcome.violations)} orphan(s) left behind:{OFF}")
        for violation in outcome.violations:
            print(f"    {RED}!{OFF} {violation}")
    else:
        print(f"  {GREEN}no orphans — the world is consistent{OFF}")

    verdict = (
        f"{GREEN}CORRECT{OFF}"
        if outcome.correct
        else (
            f"{YELLOW}CLEAN FAILURE{OFF}"
            if outcome.clean_failure
            else f"{RED}INCORRECT{OFF}"
        )
    )
    print(f"\n  verdict: {verdict}")


async def main(args: argparse.Namespace) -> None:
    if args.compare:
        for label, config in (
            ("BASELINE — no protections", BASELINE),
            ("RUNTIME — every protection on", RUNTIME),
        ):
            outcome = await run_episode(
                args.seed,
                build_policy(args.policy, args.seed),
                config,
                args.fault_rate,
                impossible_rate=0.0 if args.satisfiable else 0.15,
            )
            show(outcome, f"{label}   (seed {args.seed}, faults {args.fault_rate})")
        print(
            f"\n{DIM}Same seed means the same scenario and the same fault "
            f"sequence, so any difference above is the runtime.{OFF}\n"
        )
        return

    config = BASELINE if args.baseline else RUNTIME
    label = "BASELINE" if args.baseline else "RUNTIME"
    outcome = await run_episode(
        args.seed,
        build_policy(args.policy, args.seed),
        config,
        args.fault_rate,
        impossible_rate=0.0 if args.satisfiable else 0.15,
    )
    show(outcome, f"{label}   (seed {args.seed}, faults {args.fault_rate})")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fault-rate", type=float, default=0.0)
    parser.add_argument("--baseline", action="store_true", help="no protections")
    parser.add_argument("--compare", action="store_true", help="run both, same faults")
    parser.add_argument(
        "--policy",
        choices=["scripted", "eager", "random", "llm"],
        default="scripted",
    )
    parser.add_argument(
        "--satisfiable",
        action="store_true",
        default=True,
        help="only generate orders that can actually be fulfilled",
    )
    asyncio.run(main(parser.parse_args()))
