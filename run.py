#!/usr/bin/env python3
"""Unified simulation entry point.

Usage:
    python run.py --demo rule    # rule-based agents, full mechanics
    python run.py --demo random  # random agents, no mechanics
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from map_generator import MapGenerator, Tile
from simulation import GenericMapSimulation, build_agent_spec


def find_tile(grid: np.ndarray, tile: Tile) -> tuple[int, int]:
    ys, xs = np.where(grid == int(tile))
    return int(ys[0]), int(xs[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simulation with rule-based or random agents")
    parser.add_argument("--demo", choices=["random", "rule"], default="rule",
                        help="Agent type: 'rule' for rule-based, 'random' for random (default: rule)")
    parser.add_argument("--knowledge", choices=["off", "on"], default="on",
                        help="Show per-agent knowledge panels in the GIF (default: on)")
    parser.add_argument("--style", choices=["full", "world"], default="full",
                        help="'full' = world + knowledge panels, 'world' = world only (default: full)")
    parser.add_argument("--seed", type=int, default=42, help="Map and agent seed (default: 42)")
    parser.add_argument("--width", type=int, default=50, help="Map width (default: 50)")
    parser.add_argument("--height", type=int, default=35, help="Map height (default: 35)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Map alpha in [-1, 1] (default: 0.0)")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum simulation steps (default: 300)")
    parser.add_argument("--fps", type=int, default=12, help="Animation FPS (default: 12)")
    parser.add_argument("--output", type=str, default="output/simulation.gif", help="Output GIF path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    parser.add_argument("--no-render", action="store_true", help="Skip GIF rendering (useful for debug runs)")
    parser.add_argument("--human-view", type=int, default=6, help="Human observation radius (default: 6)")
    parser.add_argument("--alien-fov", type=int, default=6, help="Alien FOV radius (default: 6)")
    parser.add_argument("--noise-radius", type=int, default=2, help="Max cell offset for player noise (default: 2)")
    return parser.parse_args()


def build_agents(grid: np.ndarray, demo: str, seed: int):
    human_start = find_tile(grid, Tile.PLAYER_START)
    alien_start = find_tile(grid, Tile.ALIEN_START)

    if demo == "rule":
        from agents.alien import AlienAgent
        from agents.base import Direction
        from agents.human import HumanAgent

        human = HumanAgent(start_pos=human_start, start_dir=Direction.NORTH)
        alien = AlienAgent(grid=grid.copy(), start_pos=alien_start)
        return [
            build_agent_spec("human_1", "human", human),
            build_agent_spec("alien_1", "alien", alien),
        ]

    from agents.random_alien import RandomAlienAgent
    from agents.random_human import RandomHumanAgent

    human = RandomHumanAgent(grid=grid.copy(), pos=human_start, rng=random.Random(seed))
    alien = RandomAlienAgent(grid=grid.copy(), pos=alien_start, rng=random.Random(seed + 1))
    return [
        build_agent_spec("human_1", "human", human),
        build_agent_spec("alien_1", "alien", alien),
    ]


def main():
    args = parse_args()

    generator = MapGenerator(width=args.width, height=args.height, alpha=args.alpha, seed=args.seed)
    grid = generator.generate()

    agents = build_agents(grid, args.demo, args.seed)
    simulation = GenericMapSimulation(
        grid=grid.copy(),
        agents=agents,
        knowledge_mode=args.knowledge,
        default_human_view=args.human_view,
        default_alien_view=args.alien_fov,
        enable_mechanics=(args.demo == "rule"),
        noise_radius=args.noise_radius,
        seed=args.seed,
    )

    frames, outcome = simulation.run(max_steps=args.max_steps)
    print(f"Outcome: {outcome}  ({len(frames) - 1} steps)")

    if not args.no_render:
        simulation.render(
            frames=frames,
            outcome=outcome,
            output_path=args.output,
            fps=args.fps,
            show_window=not args.no_show,
            world_only=(args.style == "world"),
        )


if __name__ == "__main__":
    main()
