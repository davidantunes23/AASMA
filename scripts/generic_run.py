#!/usr/bin/env python3
"""Run the generic map simulator with interchangeable agent implementations."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generic_simulation import GenericMapSimulation, build_agent_spec
from map_generator import MapGenerator, Tile


def find_tile(grid: np.ndarray, tile: Tile) -> tuple[int, int]:
    ys, xs = np.where(grid == int(tile))
    return int(ys[0]), int(xs[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a generic simulation with optional per-agent knowledge panels")
    parser.add_argument("--demo", choices=["random", "rule"], default="random", help="Which built-in agent pair to use")
    parser.add_argument("--knowledge", choices=["off", "on"], default="on", help="Render only the world or include per-agent knowledge panels")
    parser.add_argument("--style", choices=["full", "world"], default="full", help="Rendering style")
    parser.add_argument("--seed", type=int, default=42, help="Map and agent seed")
    parser.add_argument("--width", type=int, default=50, help="Map width")
    parser.add_argument("--height", type=int, default=35, help="Map height")
    parser.add_argument("--alpha", type=float, default=0.0, help="Map alpha")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum number of simulation steps")
    parser.add_argument("--fps", type=int, default=12, help="Animation frames per second")
    parser.add_argument("--output", type=str, default="output/generic_simulation.gif", help="Output GIF path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    parser.add_argument("--human-view", type=int, default=6, help="Default human observation radius for knowledge snapshots")
    parser.add_argument("--alien-fov", type=int, default=6, help="Default alien FOV radius for knowledge snapshots")
    return parser.parse_args()


def build_demo_agents(grid: np.ndarray, demo: str, seed: int):
    human_start = find_tile(grid, Tile.PLAYER_START)
    alien_start = find_tile(grid, Tile.ALIEN_START)

    if demo == "rule":
        from agents.alien import AlienAgent
        from agents.human import Direction as HumanDirection
        from agents.human import HumanAgent

        human = HumanAgent(start_pos=human_start, start_dir=HumanDirection.NORTH)
        alien = AlienAgent(grid=grid.copy(), start_pos=(alien_start[1], alien_start[0]))
        return [
            build_agent_spec("human_1", "human", human),
            build_agent_spec("alien_1", "alien", alien),
        ]

    from random_agents.random_alien import RandomAlienAgent
    from random_agents.random_human import RandomHumanAgent

    human = RandomHumanAgent(grid=grid.copy(), pos=human_start, rng=random.Random(seed))
    alien = RandomAlienAgent(grid=grid.copy(), pos=(alien_start[1], alien_start[0]), rng=random.Random(seed + 1))
    return [
        build_agent_spec("human_1", "human", human),
        build_agent_spec("alien_1", "alien", alien),
    ]


def main():
    args = parse_args()
    generator = MapGenerator(width=args.width, height=args.height, alpha=args.alpha, seed=args.seed)
    grid = generator.generate()

    agents = build_demo_agents(grid, args.demo, args.seed)
    simulation = GenericMapSimulation(
        grid=grid.copy(),
        agents=agents,
        knowledge_mode=args.knowledge,
        default_human_view=args.human_view,
        default_alien_view=args.alien_fov,
    )
    frames, outcome = simulation.run(max_steps=args.max_steps)
    print(f"Outcome: {outcome}")

    simulation.render(
        frames=frames,
        outcome=outcome,
        output_path=args.output,
        fps=args.fps,
        show_window=not args.no_show,
        world_only=args.style == "world",
    )


if __name__ == "__main__":
    main()
