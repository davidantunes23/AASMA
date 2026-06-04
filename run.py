#!/usr/bin/env python3
"""Unified simulation entry point.

Usage:
    python run.py --demo rule    # rule-based agents, full mechanics
    python run.py --demo random  # random agents, no mechanics
"""
from __future__ import annotations

import argparse
import random
from collections import deque

import numpy as np

from map_generator import MapGenerator, Tile
from simulation import GenericMapSimulation, build_agent_spec


def find_tile(grid: np.ndarray, tile: Tile) -> tuple[int, int]:
    ys, xs = np.where(grid == int(tile))
    return int(ys[0]), int(xs[0])


def topology_distance(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    if start == goal:
        return 0
    frontier = deque([(start, 0)])
    visited = {start}
    H, W = grid.shape
    while frontier:
        (y, x), dist = frontier.popleft()
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            ny, nx = y + dy, x + dx
            nxt = (ny, nx)
            if not (0 <= ny < H and 0 <= nx < W) or nxt in visited:
                continue
            if grid[ny, nx] == int(Tile.WALL):
                continue
            if nxt == goal:
                return dist + 1
            visited.add(nxt)
            frontier.append((nxt, dist + 1))
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simulation with rule-based or random agents")
    parser.add_argument("--demo", choices=["random", "rule"], default="rule",
                        help="Agent type: 'rule' for rule-based, 'random' for random (default: rule)")
    parser.add_argument("--human-count", type=int, default=3, help="Number of human agents to create (default: 3)")
    parser.add_argument("--human-class", choices=["random", "human", "role", "coop", "omniscient"], default="human",
                        help="Human agent implementation: 'human' (rule), 'role' (role-aware), 'coop' (shared-map cooperative), 'omniscient' (upper-bound: full map/missions/exit pre-known), or 'random' (default: human)")
    parser.add_argument("--alien-count", type=int, default=1, help="Number of alien agents to create (default: 1)")
    parser.add_argument("--alien-class", choices=["random", "alien"], default="alien",
                        help="Alien agent implementation: 'alien' (rule) or 'random' (default: alien)")
    parser.add_argument("--knowledge", choices=["off", "on"], default="on",
                        help="Show per-agent knowledge panels in the GIF (default: on)")
    parser.add_argument("--style", choices=["full", "world"], default="full",
                        help="'full' = world + knowledge panels, 'world' = world only (default: full)")
    parser.add_argument("--seed", type=int, default=42, help="Map and agent seed (default: 42)")
    parser.add_argument("--width", type=int, default=60, help="Map width (default: 50)")
    parser.add_argument("--height", type=int, default=40, help="Map height (default: 35)")
    parser.add_argument("--alpha", type=float, default=0.0, help="Map alpha in [-1, 1] (default: 0.0)")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation steps (default: 2000)")
    parser.add_argument("--fps", type=int, default=12, help="Animation FPS (default: 12)")
    parser.add_argument("--output", type=str, default="output/simulation.gif", help="Output GIF path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a preview window")
    parser.add_argument("--no-render", action="store_true", help="Skip GIF rendering (useful for debug runs)")
    parser.add_argument("--debug-log", action="store_true",
                        help="Write role/mission reassignment debug log to output/logs/")
    parser.add_argument("--human-view", type=int, default=6, help="Human observation radius (default: 6)")
    parser.add_argument("--alien-fov", type=int, default=6, help="Alien FOV radius (default: 6)")
    parser.add_argument("--noise-radius", type=int, default=2, help="Max cell offset for player noise (default: 2)")
    parser.add_argument("--noise-prob", type=float, default=0.10, help="Probability of emitting noise each step (default: 0.10)")
    parser.add_argument("--mission-steps", type=int, default=10,
                        help="Steps required to complete each mission tile (default: 10)")
    parser.add_argument("--mission-count", type=int, default=2,
                        help="Number of mission tiles to place (default: 2)")
    parser.add_argument("--random-map", action="store_true",
                        help="Use a random seed for map/agent generation")
    parser.add_argument("--min-start-distance", type=int, default=0,
                        help="Minimum topology distance between player and alien spawn tiles (default: 0)")
    return parser.parse_args()


def choose_human_positions(
    grid: np.ndarray,
    anchor: tuple[int, int],
    count: int,
    avoid: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    avoid_set = set(avoid or set())
    candidates: list[tuple[int, int]] = []
    if anchor not in avoid_set:
        candidates.append(anchor)
    floors = [(int(y), int(x)) for y, x in np.argwhere(grid == int(Tile.FLOOR))]
    floors = [pos for pos in floors if pos not in avoid_set and pos != anchor]
    floors.sort(key=lambda pos: abs(pos[0] - anchor[0]) + abs(pos[1] - anchor[1]))
    candidates.extend(floors)
    if len(candidates) < count:
        candidates.extend([anchor] * (count - len(candidates)))
    return candidates[:count]


def build_agents(
    grid: np.ndarray,
    demo: str,
    seed: int,
    min_start_distance: int = 0,
    human_count: int = 1,
    human_class: str = "human",
    alien_count: int = 1,
    alien_class: str = "alien",
):
    human_start = find_tile(grid, Tile.PLAYER_START)
    alien_start = find_tile(grid, Tile.ALIEN_START)

    if min_start_distance > 0:
        current_distance = topology_distance(grid, human_start, alien_start)
        if current_distance < min_start_distance:
            floor_tiles = np.argwhere(grid == int(Tile.FLOOR))
            floor_candidates = [(int(y), int(x)) for y, x in floor_tiles]
            floor_candidates.sort(
                key=lambda pos: (-topology_distance(grid, human_start, pos), abs(pos[0] - human_start[0]) + abs(pos[1] - human_start[1]))
            )
            if floor_candidates:
                alien_start = floor_candidates[0]

    human_positions = choose_human_positions(grid, human_start, human_count, avoid={alien_start})

    specs: list = []

    # Helper to create human instances depending on chosen class
    def make_human_instance(ix: int):
        pos = human_positions[ix]
        if human_class == "random" or demo == "random":
            from agents.random_human import RandomHumanAgent
            return RandomHumanAgent(pos=pos, rng=random.Random(seed + ix))
        if human_class == "role":
            from agents.base import Direction
            from agents.role_human import RoleHumanAgent
            return RoleHumanAgent(start_pos=pos, start_dir=Direction.NORTH)
        if human_class == "coop":
            from agents.base import Direction
            from agents.coop_role_human import CoopRoleHumanAgent
            return CoopRoleHumanAgent(start_pos=pos, start_dir=Direction.NORTH)
        if human_class == "omniscient":
            from agents.base import Direction
            from agents.omniscient_human import OmniscientHumanAgent
            return OmniscientHumanAgent(grid=grid.copy(), start_pos=pos, start_dir=Direction.NORTH)
        # default: rule-based HumanAgent
        from agents.base import Direction
        from agents.rule_human import HumanAgent
        return HumanAgent(start_pos=pos, start_dir=Direction.NORTH)

    # Helper to create alien instances depending on chosen class
    def make_alien_instance(ix: int):
        if alien_class == "random" or demo == "random":
            from agents.random_alien import RandomAlienAgent
            return RandomAlienAgent(pos=alien_start, rng=random.Random(seed + 100 + ix))
        from agents.rule_alien import AlienAgent
        return AlienAgent(start_pos=alien_start)

    # Create requested human agents
    for i in range(human_count):
        inst = make_human_instance(i)
        specs.append(build_agent_spec(f"human_{i+1}", "human", inst))

    # Create requested alien agents
    for j in range(alien_count):
        inst = make_alien_instance(j)
        specs.append(build_agent_spec(f"alien_{j+1}", "alien", inst))

    return specs


def main():
    args = parse_args()
    if args.random_map:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"Random map seed: {args.seed}")

    generator = MapGenerator(
        width=args.width,
        height=args.height,
        alpha=args.alpha,
        seed=args.seed,
        mission_count=args.mission_count,
    )
    grid = generator.generate()

    player_start = find_tile(grid, Tile.PLAYER_START)
    alien_start = find_tile(grid, Tile.ALIEN_START)

    agents = build_agents(
        grid,
        args.demo,
        args.seed,
        min_start_distance=args.min_start_distance,
        human_count=args.human_count,
        human_class=args.human_class,
        alien_count=args.alien_count,
        alien_class=args.alien_class,
    )
    simulation = GenericMapSimulation(
        grid=grid.copy(),
        agents=agents,
        knowledge_mode=args.knowledge,
        default_human_view=args.human_view,
        default_alien_view=args.alien_fov,
        enable_mechanics=(args.demo == "rule"),
        p_noise=args.noise_prob,
        noise_radius=args.noise_radius,
        seed=args.seed,
        mission_steps=args.mission_steps,
        debug_log=args.debug_log,
    )
    simulation.mission_tile_values = {int(Tile.MISSION)}

    if args.human_class == "omniscient":
        simulation.omniscient_roles = True
        for pos in simulation._mission_positions():
            simulation.add_mission(pos)

    if args.debug_log:
        print(f"Role debug log: {simulation.debug_log_path}")

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
