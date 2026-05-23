#!/usr/bin/env python3
"""Demo: run simulations with 1, 2, and 3 human agents (rule-based alien).

Creates GIFs in `output/` showing each scenario. Uses `RoleHumanAgent` for humans
so roles can be set and observed. The first human is always the WORKER; for 2
agents the second is DECOY; for 3 agents the third is RUNNER.
"""
from __future__ import annotations

import os
import random
import argparse
import sys
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from map_generator import MapGenerator, Tile
from simulation import GenericMapSimulation, build_agent_spec
from agents.role_human import RoleHumanAgent
from agents.alien import AlienAgent
from agents.base import Direction

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1, 2, and 3 human-agent demos with a rule-based alien")
    parser.add_argument("--seed", type=int, default=42, help="Map seed (default: 42)")
    parser.add_argument("--min-start-distance", type=int, default=0,
                        help="Minimum topology distance between player and alien spawns (default: 0)")
    parser.add_argument("--missions", type=int, default=1,
                        help="Number of mission tiles to place on the map (default: 1)")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per demo run (default: 200)")
    parser.add_argument("--fps", type=int, default=10, help="GIF frames per second (default: 10)")
    return parser.parse_args()


def find_tiles(grid: np.ndarray, tile_value: int) -> List[Tuple[int, int]]:
    ys, xs = np.where(grid == int(tile_value))
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


def choose_human_positions(grid: np.ndarray, anchor: Tuple[int, int], n: int) -> List[Tuple[int, int]]:
    # choose n walkable FLOOR tiles (or HIDE) near the anchor
    floors = find_tiles(grid, Tile.FLOOR)
    # sort by manhattan distance to anchor
    floors.sort(key=lambda p: abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]))
    chosen = []
    for p in floors:
        if p == anchor:
            continue
        if p in chosen:
            continue
        chosen.append(p)
        if len(chosen) >= n:
            break
    # ensure anchor is first human
    return [anchor] + chosen[: max(0, n - 1)]


def pick_mission_tile(grid: np.ndarray, avoid: List[Tuple[int, int]]) -> Tuple[int, int] | None:
    floors = find_tiles(grid, Tile.FLOOR)
    for p in floors:
        if p not in avoid:
            return p
    return None


def place_mission_tiles(
    grid: np.ndarray,
    count: int,
    player_start: Tuple[int, int],
    alien_spawn: Tuple[int, int],
    avoid: set[Tuple[int, int]],
) -> list[Tuple[int, int]]:
    if count <= 0:
        return []

    candidates = [(int(pos[0]), int(pos[1])) for pos in np.argwhere(grid == int(Tile.FLOOR))]
    candidates = [pos for pos in candidates if pos not in avoid]
    candidates.sort(
        key=lambda pos: (
            -min(
                topology_distance(grid, player_start, pos),
                topology_distance(grid, alien_spawn, pos),
            ),
            pos[0],
            pos[1],
        )
    )

    chosen = candidates[:count]
    for pos in chosen:
        grid[pos] = int(Tile.MISSION)
    return chosen


def pick_alien_spawn(grid: np.ndarray, human_start: Tuple[int, int], alien_start: Tuple[int, int], min_distance: int) -> Tuple[int, int]:
    if min_distance <= 0:
        return alien_start

    current_distance = topology_distance(grid, human_start, alien_start)
    if current_distance >= min_distance:
        return alien_start

    floor_tiles = find_tiles(grid, Tile.FLOOR)
    floor_tiles.sort(
        key=lambda pos: (
            -topology_distance(grid, human_start, pos),
            abs(pos[0] - alien_start[0]) + abs(pos[1] - alien_start[1]),
        )
    )
    return floor_tiles[0] if floor_tiles else alien_start


def topology_distance(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    from collections import deque

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


def run_demo_for_count(count: int, seed: int = 42, min_start_distance: int = 0, max_steps: int = 200, fps: int = 10):
    print(f"Running demo with {count} human agent(s)")
    gen = MapGenerator(width=50, height=35, alpha=0.0, seed=seed)
    grid = gen.generate()

    player_start = find_tiles(grid, Tile.PLAYER_START)[0]
    alien_start = find_tiles(grid, Tile.ALIEN_START)[0]
    alien_spawn = pick_alien_spawn(grid, player_start, alien_start, min_start_distance)

    mission_positions = place_mission_tiles(
        grid,
        count=1,
        player_start=player_start,
        alien_spawn=alien_spawn,
        avoid={player_start, alien_start, alien_spawn},
    )
    if mission_positions:
        print(f"Placed mission tiles at: {mission_positions}")

    human_positions = choose_human_positions(grid, player_start, count)

    humans = []
    for i, pos in enumerate(human_positions):
        h = RoleHumanAgent(start_pos=pos, start_dir=Direction.NORTH, view_length=6)
        humans.append(h)

    alien = AlienAgent(grid=grid.copy(), start_pos=alien_spawn)

    # Build agent specs: humans then alien
    specs = []
    for i, h in enumerate(humans, start=1):
        specs.append(build_agent_spec(f"human_{i}", "human", h))
    specs.append(build_agent_spec("alien_1", "alien", alien))

    sim = GenericMapSimulation(
        grid=grid.copy(),
        agents=specs,
        knowledge_mode="on",
        default_human_view=6,
        default_alien_view=6,
        enable_mechanics=True,
        noise_radius=2,
        seed=seed,
    )
    sim.mission_tile_values = {int(Tile.MISSION)}

    # Set initial roles explicitly: first WORKER, second DECOY, third RUNNER
    role_map = {}
    if count >= 1:
        role_map["human_1"] = "WORKER"
    if count >= 2:
        role_map["human_2"] = "DECOY"
    if count >= 3:
        role_map["human_3"] = "RUNNER"
    sim.set_initial_roles(role_map)

    # Event-driven reassignment only; keep role-based behaviors enabled.
    sim.enable_role_based(True)

    frames, outcome = sim.run(max_steps=max_steps)
    print(f"Outcome: {outcome} (steps: {len(frames)-1})")

    out_path = os.path.join(OUT_DIR, f"demo_{count}_humans.gif")
    sim.render(frames=frames, outcome=outcome, output_path=out_path, fps=fps, show_window=False)
    print(f"Saved GIF: {out_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    for n in (1, 2, 3):
        run_demo_for_count(
            n,
            seed=args.seed,
            min_start_distance=args.min_start_distance,
            max_steps=args.max_steps,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
