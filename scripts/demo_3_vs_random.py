#!/usr/bin/env python3
"""Demo: 3 role humans vs a random alien (GIF output)."""
from __future__ import annotations

import argparse
import os
import random
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
from agents.random_alien import RandomAlienAgent
from agents.base import Direction

OUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3 humans vs random alien demo")
    parser.add_argument("--seed", type=int, default=50, help="Map seed (default: 50)")
    parser.add_argument("--min-start-distance", type=int, default=10,
                        help="Minimum topology distance between player and alien spawn (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps (default: 500)")
    parser.add_argument("--fps", type=int, default=20, help="GIF fps (default: 20)")
    parser.add_argument("--output-prefix", type=str, default="output/demo_3_humans_vs_random",
                        help="Output GIF prefix (default: output/demo_3_humans_vs_random)")
    return parser.parse_args()


def find_tiles(grid: np.ndarray, tile_value: int) -> List[Tuple[int, int]]:
    ys, xs = np.where(grid == int(tile_value))
    return [(int(y), int(x)) for y, x in zip(ys, xs)]


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


def choose_human_positions(grid: np.ndarray, anchor: Tuple[int, int], n: int) -> List[Tuple[int, int]]:
    floors = find_tiles(grid, Tile.FLOOR)
    floors.sort(key=lambda p: abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]))
    chosen: List[Tuple[int, int]] = [anchor]
    for p in floors:
        if p in chosen:
            continue
        chosen.append(p)
        if len(chosen) >= n:
            break
    return chosen


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


def run_demo(
    base_grid: np.ndarray,
    missions: int,
    seed: int,
    min_start_distance: int,
    max_steps: int,
    fps: int,
    output_path: str,
) -> None:
    grid = base_grid.copy()

    player_start = find_tiles(grid, Tile.PLAYER_START)[0]
    alien_start = find_tiles(grid, Tile.ALIEN_START)[0]
    alien_spawn = pick_alien_spawn(grid, player_start, alien_start, min_start_distance)

    mission_positions = place_mission_tiles(
        grid,
        count=missions,
        player_start=player_start,
        alien_spawn=alien_spawn,
        avoid={player_start, alien_start, alien_spawn},
    )
    if mission_positions:
        print(f"Placed mission tiles at: {mission_positions}")
    else:
        print("Placed mission tiles at: []")

    human_positions = choose_human_positions(grid, player_start, 3)
    humans = [RoleHumanAgent(start_pos=pos, start_dir=Direction.NORTH, view_length=6) for pos in human_positions]

    alien = RandomAlienAgent(grid=grid.copy(), pos=alien_spawn, rng=random.Random(seed + 1))

    specs = [build_agent_spec(f"human_{i+1}", "human", h) for i, h in enumerate(humans)]
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

    role_map = {"human_1": "WORKER", "human_2": "DECOY", "human_3": "RUNNER"}
    sim.set_initial_roles(role_map)
    sim.enable_role_based(True)

    frames, outcome = sim.run(max_steps=max_steps)
    print(f"Outcome: {outcome} (steps: {len(frames) - 1})")

    sim.render(frames=frames, outcome=outcome, output_path=output_path, fps=fps, show_window=False)
    print(f"Saved GIF: {output_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)

    gen = MapGenerator(width=50, height=35, alpha=0.0, seed=args.seed)
    base_grid = gen.generate()

    for missions in (1, 2, 0):
        print(f"\n--- Running demo with {missions} mission(s) ---")
        out_path = f"{args.output_prefix}_missions_{missions}.gif"
        run_demo(
            base_grid=base_grid,
            missions=missions,
            seed=args.seed,
            min_start_distance=args.min_start_distance,
            max_steps=args.max_steps,
            fps=args.fps,
            output_path=out_path,
        )


if __name__ == "__main__":
    main()
