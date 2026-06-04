#!/usr/bin/env python3
"""Demo runner.

Always vs the rule-based alien. Must specify an agent with --agent or --example.

Usage:
    python3 demo.py --list                        # list all curated examples
    python3 demo.py --example role_all_escape     # run a curated example
    python3 demo.py --agent role --seed 42        # specific seed, role agents
    python3 demo.py --agent coop                  # random seed, coop agents
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys

# ---------------------------------------------------------------------------
# Curated examples — verified seeds that reliably produce specific outcomes.
# Each entry: (agent_class, human_count, mission_count, seed, description)
# All runs use the rule-based alien.
# ---------------------------------------------------------------------------
EXAMPLES: dict[str, tuple] = {
    # ── Random agents (baseline — mechanics disabled) ─────────────────────
    "random_timeout":     ("random", 3, 2, 0,   "random agents wander aimlessly — nobody escapes"),
    "random_all_caught":  ("random", 3, 2, 1,   "alien catches all 3 random agents"),
    "random_some_escape": ("random", 3, 2, 734, "1 random agent stumbles to the exit by chance"),

    # ── Rule-based humans, no coordination (3 agents, 2 missions) ────────
    # Note: rule agents either all escape or all get caught — partial escape is rare
    "rule_all_escape":    ("human",  3, 2, 32,  "3/3 uncoordinated humans escape after completing missions"),
    "rule_all_caught":    ("human",  3, 2, 0,   "alien catches all 3 uncoordinated humans"),
    "rule_timeout":       ("human",  3, 2, 7,   "episode times out — agents loop without completing missions"),

    # ── Role-based agents (WORKER / DECOY / EXPLORER) ────────────────────
    "role_all_escape":    ("role",   3, 2, 5,   "3/3 humans escape — Worker/Decoy/Explorer roles succeed"),
    "role_some_escape":   ("role",   3, 2, 9,   "partial escape — Decoy sacrificed to protect Workers"),
    "role_all_caught":    ("role",   3, 2, 0,   "alien catches all 3 before missions complete"),

    # ── Cooperative agents (shared belief map) ────────────────────────────
    # Note: coop all_escape is rare with this spawn — coop some_escape and timeout shown instead
    "coop_some_escape":   ("coop",   3, 2, 0,   "partial escape — shared map helps but alien intercepts one"),
    "coop_all_caught":    ("coop",   3, 2, 1,   "alien corners all 3 before exit opens"),
    "coop_timeout":       ("coop",   3, 2, 5,   "episode times out — team stalled by alien pressure"),

    # ── Omniscient agents (full map known from start, upper-bound) ────────
    "omni_all_escape":    ("omniscient", 3, 2, 0,  "3/3 escape — full map knowledge, optimal routing"),
    "omni_some_escape":   ("omniscient", 3, 2, 1,  "partial escape despite full knowledge"),
    "omni_all_caught":    ("omniscient", 3, 2, 2,  "alien wins even against omniscient team"),
}
# All examples: 3 humans, 2 missions, mission_steps=10 — matches evaluate_agents.py constants.


def list_examples() -> None:
    col_w = max(len(k) for k in EXAMPLES) + 2
    print("\nCurated examples  (run with --example <name>)\n")
    current_prefix = None
    for name, (_, _, _, seed, desc) in EXAMPLES.items():
        prefix = name.split("_")[0]
        if prefix != current_prefix:
            print(f"  {'─' * 60}")
            current_prefix = prefix
        print(f"  {name:<{col_w}}  seed={seed:<4}  {desc}")
    print(f"  {'─' * 60}\n")


def build_run_command(
    agent: str,
    human_count: int,
    mission_count: int,
    seed: int,
    output: str,
    style: str = "full",
    no_show: bool = True,
) -> list[str]:
    if agent == "random":
        cmd = [
            sys.executable, "run.py",
            "--human-class", "random",
            "--alien-class", "alien",
            "--human-count", str(human_count),
            "--mission-count", str(mission_count),
            "--seed", str(seed),
            "--output", output,
            "--style", style,
            "--knowledge", "off",
        ]
    else:
        cmd = [
            sys.executable, "run.py",
            "--human-class", agent,
            "--human-count", str(human_count),
            "--mission-count", str(mission_count),
            "--seed", str(seed),
            "--output", output,
            "--style", style,
            "--knowledge", "on",
        ]
    if no_show:
        cmd.append("--no-show")
    return cmd


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    print()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo runner — always vs rule alien. Use --list to browse curated examples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent", choices=["random", "human", "role", "coop", "omniscient"],
                        help="Agent type (required unless using --example or --list)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for map and agents (random if omitted)")
    parser.add_argument("--example", metavar="NAME",
                        help="Run a curated example by name (see --list)")
    parser.add_argument("--list", action="store_true",
                        help="Print all curated examples and exit")
    parser.add_argument("--run-all", action="store_true",
                        help="Run every curated example and save GIFs to output/")
    parser.add_argument("--output", default=None,
                        help="Output GIF path (auto-named if omitted)")
    parser.add_argument("--style", choices=["full", "world"], default="full",
                        help="'full' = world + knowledge panels, 'world' = world only (default: full)")
    parser.add_argument("--show", action="store_true",
                        help="Open the GIF in a preview window after rendering")
    args = parser.parse_args()

    if args.list:
        list_examples()
        return

    if args.run_all:
        print(f"\nRunning all {len(EXAMPLES)} curated examples...\n")
        for i, (key, (agent, hc, mc, seed, desc)) in enumerate(EXAMPLES.items(), 1):
            output = f"output/demo_{key}_seed{seed}.gif"
            print(f"[{i}/{len(EXAMPLES)}] {key}  —  {desc}")
            cmd = build_run_command(agent, hc, mc, seed, output,
                                    style=args.style, no_show=not args.show)
            run(cmd)
        print(f"\nDone. {len(EXAMPLES)} GIFs saved to output/")
        return

    if args.example is not None:
        key = args.example
        if key not in EXAMPLES:
            print(f"Unknown example '{key}'. Use --list to see available names.")
            sys.exit(1)
        agent, human_count, mission_count, seed, desc = EXAMPLES[key]
        output = args.output or f"output/demo_{key}_seed{seed}.gif"
        print(f"\n{'─' * 60}")
        print(f"Example : {key}")
        print(f"Outcome : {desc}")
        print(f"Agents  : {human_count}x {agent} vs rule alien  |  missions: {mission_count}  |  seed: {seed}")
        print(f"Output  : {output}")
        print(f"{'─' * 60}\n")
        cmd = build_run_command(agent, human_count, mission_count, seed, output,
                                style=args.style, no_show=not args.show)
        run(cmd)
        return

    if args.agent is None:
        parser.print_help()
        print("\nError: --agent is required (or use --example / --list).")
        sys.exit(1)

    agent = args.agent
    seed = args.seed if args.seed is not None else random.randint(0, 9999)
    # 3 humans, 2 missions — matches evaluate_agents.py
    human_count = 3
    mission_count = 2
    output = args.output or f"output/demo_{agent}_seed{seed}.gif"
    print(f"\n{'─' * 60}")
    print(f"Agent   : {agent} vs rule alien  ({human_count} humans, {mission_count} missions)")
    print(f"Seed    : {seed}")
    print(f"Output  : {output}")
    print(f"{'─' * 60}\n")
    cmd = build_run_command(agent, human_count, mission_count, seed, output,
                            style=args.style, no_show=not args.show)
    run(cmd)


if __name__ == "__main__":
    main()
