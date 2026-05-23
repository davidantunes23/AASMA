#!/usr/bin/env python3
"""Evaluate rule-based agents across alpha values and plot win rates."""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from agents.alien import AlienAgent
    from agents.human import Direction as HumanDirection
    from agents.human import HumanAgent
    from game import Game
    from map_generator import MapGenerator, Tile
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from agents.alien import AlienAgent
    from agents.human import Direction as HumanDirection
    from agents.human import HumanAgent
    from game import Game
    from map_generator import MapGenerator, Tile


@dataclass
class WinRates:
    human: float
    alien: float
    draws: float
    avg_steps: float


def find_tile_pos(grid: np.ndarray, tile: Tile) -> tuple[int, int]:
    matches = np.argwhere(grid == int(tile))
    if len(matches) == 0:
        raise ValueError(f"Tile {tile.name} not found in map")
    y, x = matches[0]
    return int(y), int(x)


def build_game(grid: np.ndarray, view_length: int) -> tuple[Game, tuple[int, int]]:
    human_start = find_tile_pos(grid, Tile.PLAYER_START)
    alien_start = find_tile_pos(grid, Tile.ALIEN_START)
    exit_pos = find_tile_pos(grid, Tile.EXIT)

    human_agent = HumanAgent(start_pos=human_start, start_dir=HumanDirection.NORTH)
    alien_agent = AlienAgent(grid=grid.copy(), start_pos=(alien_start[1], alien_start[0]))

    game = Game(
        map=grid.copy(),
        human_agent=human_agent,
        alien_agent=alien_agent,
        human_view_length=view_length,
    )
    return game, exit_pos


def run_episode(
    grid: np.ndarray,
    max_steps: int,
    view_length: int,
    idle_limit: int,
) -> tuple[str, int]:
    game, exit_pos = build_game(grid, view_length)
    idle_steps = 0
    step_count = 0

    for _ in range(max_steps):
        previous_human_pos = game.human_pos
        game._step()
        step_count += 1

        if game.human_pos == exit_pos:
            return "human_reached_exit", step_count
        if game.human_pos == game.alien_pos:
            return "alien_caught_human", step_count

        if game.human_pos == previous_human_pos:
            idle_steps += 1
            if idle_steps >= idle_limit:
                return "human_stuck", step_count
        else:
            idle_steps = 0

    return "max_steps_reached", step_count


def evaluate_alpha(
    alpha: float,
    episodes: int,
    seed: int,
    width: int,
    height: int,
    max_steps: int,
    view_length: int,
    idle_limit: int,
) -> WinRates:
    human_wins = 0
    alien_wins = 0
    draws = 0
    total_steps = 0
    rng = np.random.default_rng(seed)

    for episode in range(episodes):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        random.seed(episode_seed)
        np.random.seed(episode_seed)

        generator = MapGenerator(
            width=width,
            height=height,
            alpha=alpha,
            seed=episode_seed,
        )
        grid = generator.generate()
        outcome, steps = run_episode(
            grid=grid,
            max_steps=max_steps,
            view_length=view_length,
            idle_limit=idle_limit,
        )
        total_steps += steps

        if outcome == "human_reached_exit":
            human_wins += 1
        elif outcome == "alien_caught_human":
            alien_wins += 1
        else:
            draws += 1

    total = max(episodes, 1)
    return WinRates(
        human=human_wins / total,
        alien=alien_wins / total,
        draws=draws / total,
        avg_steps=total_steps / total,
    )


def parse_alpha_list(raw_values: list[str]) -> list[float]:
    if len(raw_values) == 1 and "," in raw_values[0]:
        raw_values = [part.strip() for part in raw_values[0].split(",") if part.strip()]
    return [float(value) for value in raw_values]


def plot_win_rates(
    alphas: list[float],
    human: list[float],
    alien: list[float],
    draws: list[float],
    avg_steps: list[float],
    output: str,
    show_window: bool,
):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(alphas))
    bar_width = 0.6

    ax.bar(x, human, width=bar_width, color="#2E86C1", label="Human wins")
    ax.bar(x, draws, width=bar_width, bottom=np.array(human), color="#8B00D5", label="Draws")
    ax.bar(x, alien, width=bar_width, bottom=(np.array(human) + np.array(draws)), color="#C0392B", label="Alien wins")

    ax.set_title("Rule-based agent outcomes vs alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{alpha:+.1f}" for alpha in alphas])
    ax.grid(True, axis="y", alpha=0.25)

    ax_steps = ax.twinx()
    ax_steps.plot(x, avg_steps, color="#F39C12", marker="D", linewidth=1.8, label="Avg steps")
    ax_steps.set_ylabel("avg steps")
    ax_steps.set_ylim(bottom=0.0)

    handles, labels = ax.get_legend_handles_labels()
    step_handles, step_labels = ax_steps.get_legend_handles_labels()
    ax.legend(handles + step_handles, labels + step_labels, loc="best")

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        print(f"Saved plot -> {output}")

    if show_window:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        backend = matplotlib.get_backend().lower()
        if not has_display or backend == "agg":
            plt.close(fig)
            return
        plt.show()
        return

    plt.close(fig)


def parse_map_sizes(raw_values: list[str], fallback: tuple[int, int]) -> list[tuple[int, int]]:
    if not raw_values:
        return [fallback]
    sizes: list[tuple[int, int]] = []
    for raw in raw_values:
        value = raw.strip().lower().replace(" ", "")
        if "x" not in value:
            raise ValueError(f"Invalid map size '{raw}'. Expected format WIDTHxHEIGHT.")
        width_str, height_str = value.split("x", 1)
        sizes.append((int(width_str), int(height_str)))
    return sizes


def output_for_size(output_path: str, width: int, height: int) -> str:
    if not output_path:
        return output_path
    path = Path(output_path)
    suffix = path.suffix or ".png"
    stem = path.stem if path.stem else "rule_agent_win_rates"
    sized_name = f"{stem}_{width}x{height}{suffix}"
    return str(path.with_name(sized_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based agents across alpha values")
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=["-1", "-0.5", "0", "0.5", "1"],
        help="Alpha values to evaluate (space or comma separated)",
    )
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per alpha")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--map-sizes",
        nargs="+",
        default=["30x20", "45x30", "60x40"],
        help="Map sizes to evaluate (e.g. 30x20 45x30 60x40)",
    )
    parser.add_argument("--map-width", type=int, default=60, help="Fallback width if --map-sizes is empty")
    parser.add_argument("--map-height", type=int, default=40, help="Fallback height if --map-sizes is empty")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--view-length", type=int, default=6)
    parser.add_argument("--idle-limit", type=int, default=50)
    parser.add_argument(
        "--output",
        type=str,
        default="output/rule_agent_win_rates.png",
        help="Output path for the plot",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    alphas = parse_alpha_list(args.alphas)
    map_sizes = parse_map_sizes(args.map_sizes, (args.map_width, args.map_height))

    human_rates: list[float] = []
    print("Evaluating rule-based agents...")
    for width, height in map_sizes:
        human_rates = []
        alien_rates = []
        draw_rates = []
        avg_steps = []

        print(f"\nMap size: {width}x{height}")
        for alpha in alphas:
            rates = evaluate_alpha(
                alpha=alpha,
                episodes=args.episodes,
                seed=args.seed,
                width=width,
                height=height,
                max_steps=args.max_steps,
                view_length=args.view_length,
                idle_limit=args.idle_limit,
            )
            human_rates.append(rates.human)
            alien_rates.append(rates.alien)
            draw_rates.append(rates.draws)
            avg_steps.append(rates.avg_steps)
            print(
                "alpha={:+.2f} | human_win_rate={:.2%} | alien_win_rate={:.2%} | draws={:.2%} | avg_steps={:.1f}".format(
                    alpha,
                    rates.human,
                    rates.alien,
                    rates.draws,
                    rates.avg_steps,
                )
            )

        plot_win_rates(
            alphas=alphas,
            human=human_rates,
            alien=alien_rates,
            draws=draw_rates,
            avg_steps=avg_steps,
            output=output_for_size(args.output, width, height),
            show_window=not args.no_show,
        )


if __name__ == "__main__":
    main()
