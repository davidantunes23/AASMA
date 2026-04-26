#!/usr/bin/env python3
"""Run and visualize a full game simulation."""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np

try:
    from agents.alien import AlienAgent, Direction as AlienDirection
    from agents.human import Direction as HumanDirection, HumanAgent
    from game import Game
    from map_generator import MapGenerator, Tile
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from agents.alien import AlienAgent, Direction as AlienDirection
    from agents.human import Direction as HumanDirection, HumanAgent
    from game import Game
    from map_generator import MapGenerator, Tile


@dataclass
class FrameState:
    step: int
    human_pos: tuple[int, int]
    alien_pos: tuple[int, int]
    known_map: np.ndarray


def find_tile_pos(grid: np.ndarray, tile: Tile) -> tuple[int, int]:
    matches = np.argwhere(grid == int(tile))
    if len(matches) == 0:
        raise ValueError(f"Tile {tile.name} not found in map")
    y, x = matches[0]
    return int(y), int(x)


def run_simulation(
    game: Game,
    exit_pos: tuple[int, int],
    max_steps: int,
) -> tuple[list[FrameState], str]:
    frames: list[FrameState] = []
    idle_steps = 0
    max_idle_steps = 50

    for step in range(max_steps + 1):
        known = game.human_agent._known_map.copy()
        frames.append(
            FrameState(
                step=step,
                human_pos=game.human_pos,
                alien_pos=game.alien_pos,
                known_map=known,
            )
        )

        if game.human_pos == exit_pos:
            return frames, "human_reached_exit"
        if game.human_pos == game.alien_pos:
            return frames, "alien_caught_human"
        if step == max_steps:
            break

        previous_human_pos = game.human_pos
        game._step()

        if game.human_pos == previous_human_pos:
            idle_steps += 1
            if idle_steps >= max_idle_steps:
                return frames, "human_stuck"
        else:
            idle_steps = 0

    return frames, "max_steps_reached"


def build_colormaps() -> tuple[ListedColormap, ListedColormap]:
    world_colors = [
        "#101217",  # WALL
        "#D3D6DB",  # FLOOR
        "#4EA5D9",  # VENT
        "#5FBF8F",  # HIDE
        "#F2C14E",  # PLAYER_START
        "#E76F51",  # ALIEN_START
        "#7A77FF",  # EXIT
    ]
    known_colors = [
        "#101217",  # WALL
        "#D3D6DB",  # FLOOR
        "#4EA5D9",  # VENT
        "#5FBF8F",  # HIDE
        "#F2C14E",  # PLAYER_START
        "#E76F51",  # ALIEN_START
        "#7A77FF",  # EXIT
        "#2A2E36",  # UNKNOWN
    ]
    return ListedColormap(world_colors), ListedColormap(known_colors)


def visualize(
    grid: np.ndarray,
    frames: list[FrameState],
    outcome: str,
    output_path: str,
    fps: int,
    show_window: bool,
):
    world_cmap, known_cmap = build_colormaps()
    unknown_value = 7

    fig, (ax_world, ax_known) = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    fig.patch.set_facecolor("#0B0D12")

    world_img = ax_world.imshow(grid, cmap=world_cmap, vmin=0, vmax=6)
    _ = world_img

    initial_known = np.where(frames[0].known_map == Game.UNSEEN_TILE, unknown_value, frames[0].known_map)
    known_img = ax_known.imshow(initial_known, cmap=known_cmap, vmin=0, vmax=7)

    for ax, title in ((ax_world, "World"), (ax_known, "Human Knowledge")):
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#11151F")

    hy, hx = frames[0].human_pos
    ay, ax = frames[0].alien_pos
    human_world_marker = ax_world.scatter(
        [hx], [hy], s=95, c="#00D4FF", edgecolors="white", linewidths=0.7, marker="o", zorder=5
    )
    alien_world_marker = ax_world.scatter(
        [ax], [ay], s=95, c="#FF4D6D", edgecolors="white", linewidths=0.7, marker="X", zorder=5
    )
    human_known_marker = ax_known.scatter(
        [hx], [hy], s=95, c="#00D4FF", edgecolors="white", linewidths=0.7, marker="o", zorder=5
    )

    exit_y, exit_x = find_tile_pos(grid, Tile.EXIT)
    ax_world.scatter([exit_x], [exit_y], s=80, c="#7A77FF", marker="*", zorder=6)
    ax_known.scatter([exit_x], [exit_y], s=80, c="#7A77FF", marker="*", zorder=6)

    status_text = fig.suptitle("", color="white", fontsize=12)

    def update(frame_index: int):
        state = frames[frame_index]
        hy0, hx0 = state.human_pos
        ay0, ax0 = state.alien_pos

        human_world_marker.set_offsets([[hx0, hy0]])
        alien_world_marker.set_offsets([[ax0, ay0]])
        human_known_marker.set_offsets([[hx0, hy0]])

        known = np.where(state.known_map == Game.UNSEEN_TILE, unknown_value, state.known_map)
        known_img.set_data(known)

        status_text.set_text(
            f"Step {state.step}/{len(frames) - 1} | Outcome: {outcome} | Human(y,x)=({hy0},{hx0}) | Alien(y,x)=({ay0},{ax0})"
        )
        return human_world_marker, alien_world_marker, human_known_marker, known_img, status_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=max(1, int(1000 / max(1, fps))),
        blit=False,
        repeat=False,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=max(1, fps)))
    print(f"Saved animation -> {output_path}")

    if show_window:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        backend = matplotlib.get_backend().lower()
        if not has_display:
            print("Interactive display not detected; skipping window preview.")
            plt.close(fig)
            return
        if "agg" in backend:
            print(f"Matplotlib backend '{matplotlib.get_backend()}' is non-interactive; skipping window preview.")
            plt.close(fig)
            return
        plt.show()
        return

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate and visualize one game episode")
    parser.add_argument("--width", type=int, default=32, help="Map width")
    parser.add_argument("--height", type=int, default=20, help="Map height")
    parser.add_argument("--alpha", type=float, default=0.0, help="Map alpha in [-1.0, 1.0]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--random-map",
        action="store_true",
        help="Generate a different random map seed for this run",
    )
    parser.add_argument("--view-length", type=int, default=6, help="Human observation cone length")
    parser.add_argument("--max-steps", type=int, default=220, help="Maximum simulation steps")
    parser.add_argument("--fps", type=int, default=8, help="Animation frames per second")
    parser.add_argument(
        "--output",
        type=str,
        default="output/game_simulation.gif",
        help="Output path for animation file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib window (still saves animation)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    map_seed = args.seed
    if args.random_map:
        map_seed = random.randint(0, 2**31 - 1)
    print(f"Using map seed: {map_seed}")

    generator = MapGenerator(
        width=args.width,
        height=args.height,
        alpha=args.alpha,
        seed=map_seed,
    )
    grid = generator.generate()

    human_start = find_tile_pos(grid, Tile.PLAYER_START)
    alien_start = find_tile_pos(grid, Tile.ALIEN_START)
    exit_pos = find_tile_pos(grid, Tile.EXIT)

    human_agent = HumanAgent(start_pos=human_start, start_dir=HumanDirection.NORTH)
    alien_agent = AlienAgent(start_pos=alien_start, start_dir=AlienDirection.WEST)

    game = Game(
        map=grid.copy(),
        human_agent=human_agent,
        alien_agent=alien_agent,
        human_view_length=args.view_length,
    )

    frames, outcome = run_simulation(
        game=game,
        exit_pos=exit_pos,
        max_steps=args.max_steps,
    )
    print(f"Simulation finished in {len(frames) - 1} steps with outcome: {outcome}")

    visualize(
        grid=grid,
        frames=frames,
        outcome=outcome,
        output_path=args.output,
        fps=args.fps,
        show_window=not args.no_show,
    )


if __name__ == "__main__":
    main()