#!/usr/bin/env python3
"""Evaluate multiple human/alien matchups with fixed alpha and escape percentages."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from agents.rule_alien import AlienAgent
    from agents.rule_human import HumanAgent
    from agents.random_alien import RandomAlienAgent
    from agents.random_human import RandomHumanAgent
    from agents.role_human import RoleHumanAgent
    from agents.coop_role_human import CoopRoleHumanAgent
    from agents.omniscient_human import OmniscientHumanAgent
    from agents.base import Direction
    from simulation import GenericMapSimulation, build_agent_spec
    from map_generator import MapGenerator, Tile
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from agents.rule_alien import AlienAgent
    from agents.rule_human import HumanAgent
    from agents.random_alien import RandomAlienAgent
    from agents.random_human import RandomHumanAgent
    from agents.role_human import RoleHumanAgent
    from agents.coop_role_human import CoopRoleHumanAgent
    from agents.omniscient_human import OmniscientHumanAgent
    from agents.base import Direction
    from simulation import GenericMapSimulation, build_agent_spec
    from map_generator import MapGenerator, Tile


MISSION_COUNT = 2
MISSION_STEPS = 20
ROLE_TEAM_SIZE = 3
ALPHA = 1.0
DEFAULT_OUTPUT_DIR = "output/eval_pairs"


@dataclass
class EscapeStats:
    escaped_counts: list[int]
    escaped_percentages: list[float]
    avg_steps: float


@dataclass(frozen=True)
class HumanSpec:
    key: str
    label: str
    count: int
    role_based: bool
    omniscient: bool


@dataclass(frozen=True)
class AlienSpec:
    key: str
    label: str


HUMAN_SPECS = [
    HumanSpec("random", "random_human", ROLE_TEAM_SIZE, role_based=False, omniscient=False),
    HumanSpec("rule", "rule_human", ROLE_TEAM_SIZE, role_based=False, omniscient=False),
    HumanSpec("omniscient", "omniscient_human", ROLE_TEAM_SIZE, role_based=True, omniscient=True),
    HumanSpec("role", "role_human_3", ROLE_TEAM_SIZE, role_based=True, omniscient=False),
    HumanSpec("coop", "coop_role_human_3", ROLE_TEAM_SIZE, role_based=True, omniscient=False),
]

ALIEN_SPECS = [
    AlienSpec("random", "random_alien"),
    AlienSpec("rule", "rule_alien"),
]


def find_tile_pos(grid: np.ndarray, tile: Tile) -> tuple[int, int]:
    matches = np.argwhere(grid == int(tile))
    if len(matches) == 0:
        raise ValueError(f"Tile {tile.name} not found in map")
    y, x = matches[0]
    return int(y), int(x)


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


def reset_role_agent_ids() -> None:
    RoleHumanAgent._next_agent_id = 0


def build_humans(
    spec: HumanSpec,
    grid: np.ndarray,
    positions: list[tuple[int, int]],
    seed: int,
    view_length: int,
) -> list[object]:
    if spec.key in {"role", "coop", "omniscient"}:
        reset_role_agent_ids()

    if spec.key == "random":
        return [
            RandomHumanAgent(
                grid=grid.copy(),
                pos=positions[idx],
                view_length=view_length,
                rng=random.Random(seed + 10 + idx),
            )
            for idx in range(len(positions))
        ]
    if spec.key == "rule":
        return [
            HumanAgent(start_pos=pos, start_dir=Direction.NORTH, view_length=view_length)
            for pos in positions
        ]
    if spec.key == "omniscient":
        return [
            OmniscientHumanAgent(
                grid=grid.copy(),
                start_pos=pos,
                start_dir=Direction.NORTH,
                view_length=view_length,
            )
            for pos in positions
        ]
    if spec.key == "role":
        return [RoleHumanAgent(start_pos=pos, start_dir=Direction.NORTH, view_length=view_length) for pos in positions]
    if spec.key == "coop":
        return [CoopRoleHumanAgent(start_pos=pos, start_dir=Direction.NORTH, view_length=view_length) for pos in positions]

    raise ValueError(f"Unknown human model key: {spec.key}")


def build_alien(
    spec: AlienSpec,
    grid: np.ndarray,
    start_pos: tuple[int, int],
    seed: int,
    view_length: int,
) -> object:
    if spec.key == "random":
        return RandomAlienAgent(
            grid=grid.copy(),
            pos=start_pos,
            view_length=view_length,
            rng=random.Random(seed + 100),
        )
    if spec.key == "rule":
        return AlienAgent(grid=grid.copy(), start_pos=start_pos, view_length=view_length)
    raise ValueError(f"Unknown alien model key: {spec.key}")


def build_simulation(
    grid: np.ndarray,
    view_length: int,
    seed: int,
    human_spec: HumanSpec,
    alien_spec: AlienSpec,
) -> GenericMapSimulation:
    human_start = find_tile_pos(grid, Tile.PLAYER_START)
    alien_start = find_tile_pos(grid, Tile.ALIEN_START)
    human_positions = choose_human_positions(
        grid,
        human_start,
        human_spec.count,
        avoid={alien_start},
    )

    human_agents = build_humans(human_spec, grid, human_positions, seed, view_length)
    alien_agent = build_alien(alien_spec, grid, alien_start, seed, view_length)

    agents: list = []
    for idx, agent in enumerate(human_agents):
        agents.append(build_agent_spec(f"human_{idx + 1}", "human", agent))
    agents.append(build_agent_spec("alien_1", "alien", alien_agent))

    simulation = GenericMapSimulation(
        grid=grid.copy(),
        agents=agents,
        knowledge_mode="off",
        default_human_view=view_length,
        default_alien_view=view_length,
        enable_mechanics=True,
        noise_radius=2,
        seed=seed,
        mission_steps=MISSION_STEPS,
    )
    simulation.mission_tile_values = {int(Tile.MISSION)}
    simulation.enable_role_based(human_spec.role_based)

    if human_spec.omniscient:
        simulation.omniscient_roles = True
        for pos in simulation._mission_positions():
            simulation.add_mission(pos)

    return simulation


def run_episode(
    grid: np.ndarray,
    max_steps: int,
    view_length: int,
    idle_limit: int,
    seed: int,
    human_spec: HumanSpec,
    alien_spec: AlienSpec,
) -> tuple[int, int]:
    simulation = build_simulation(grid, view_length, seed, human_spec, alien_spec)
    frames, outcome = simulation.run(max_steps=max_steps)

    outcome_frame = next((frame for frame in frames if frame.outcome is not None), frames[-1])
    outcome_step = int(outcome_frame.step)

    idle_steps = 0
    idle_trigger_idx: int | None = None
    for idx in range(1, len(frames)):
        prev_positions = [agent.position for agent in frames[idx - 1].agents if agent.role == "human"]
        curr_positions = [agent.position for agent in frames[idx].agents if agent.role == "human"]
        if curr_positions == prev_positions:
            idle_steps += 1
            if idle_steps >= idle_limit:
                idle_trigger_idx = idx
                break
        else:
            idle_steps = 0

    if idle_limit > 0 and idle_trigger_idx is not None:
        frame = frames[idle_trigger_idx]
        escaped_count = len(frame.escaped_humans or [])
        return escaped_count, int(frame.step)

    escaped_count = len(outcome_frame.escaped_humans or [])
    return escaped_count, outcome_step


def evaluate_matchup(
    episode_seeds: list[int],
    width: int,
    height: int,
    max_steps: int,
    view_length: int,
    idle_limit: int,
    human_spec: HumanSpec,
    alien_spec: AlienSpec,
) -> EscapeStats:
    escaped_counts = [0, 0, 0, 0]
    total_steps = 0

    for episode_seed in episode_seeds:
        random.seed(episode_seed)
        np.random.seed(episode_seed)

        generator = MapGenerator(
            width=width,
            height=height,
            alpha=ALPHA,
            seed=episode_seed,
            mission_count=MISSION_COUNT,
        )
        grid = generator.generate()
        escaped, steps = run_episode(
            grid=grid,
            max_steps=max_steps,
            view_length=view_length,
            idle_limit=idle_limit,
            seed=episode_seed,
            human_spec=human_spec,
            alien_spec=alien_spec,
        )
        total_steps += steps
        escaped_idx = max(0, min(escaped, ROLE_TEAM_SIZE))
        escaped_counts[escaped_idx] += 1

    total = max(len(episode_seeds), 1)
    escaped_percentages = [count / total * 100.0 for count in escaped_counts]
    return EscapeStats(
        escaped_counts=escaped_counts,
        escaped_percentages=escaped_percentages,
        avg_steps=total_steps / total,
    )


def build_episode_seeds(base_seed: int, width: int, height: int, episodes: int) -> list[int]:
    mix_seed = base_seed ^ (width << 16) ^ height
    rng = np.random.default_rng(mix_seed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(episodes)]


def plot_escape_counts(
    escaped_counts: list[int],
    escaped_percentages: list[float],
    title: str,
    output: str,
    show_window: bool,
):
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(escaped_counts))
    bars = ax.bar(x, escaped_counts, color="#2E86C1")
    ax.set_title(title)
    ax.set_xlabel("humans escaped")
    ax.set_ylabel("episode count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(len(escaped_counts))])
    ax.grid(True, axis="y", alpha=0.25)

    for bar, pct in zip(bars, escaped_percentages):
        height = bar.get_height()
        if height == 0:
            y = 0.05
            va = "bottom"
        else:
            y = height + max(0.05, height * 0.02)
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{pct:.1f}%",
            ha="center",
            va=va,
            fontsize=9,
        )

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


def plot_rule_alien_comparison(
    labels: list[str],
    escaped_counts_by_label: dict[str, list[int]],
    output: str,
    show_window: bool,
):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    x = np.arange(len(labels))
    colors = ["#c0392b", "#f39c12", "#27ae60", "#2980b9"]
    bottoms = np.zeros(len(labels))
    for escaped_idx, color in enumerate(colors):
        values = [escaped_counts_by_label.get(label, [0, 0, 0, 0])[escaped_idx] for label in labels]
        ax.bar(x, values, bottom=bottoms, color=color, label=f"{escaped_idx} escaped")
        bottoms += np.array(values)
    ax.set_title("Human models vs rule-based alien")
    ax.set_xlabel("human model")
    ax.set_ylabel("episode count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")

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


def pair_output_dir(output_root: str, human_label: str, alien_label: str) -> str:
    return os.path.join(output_root, f"{human_label}_vs_{alien_label}")


def write_summary_header(writer: csv.writer) -> None:
    writer.writerow([
        "map_width",
        "map_height",
        "alpha",
        "escaped_0_pct",
        "escaped_1_pct",
        "escaped_2_pct",
        "escaped_3_pct",
        "avg_steps",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple human/alien matchups")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per matchup")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--map-sizes",
        nargs="+",
        default=["60x40"],
        help="Map sizes to evaluate (e.g. 30x20 45x30 60x40)",
    )
    parser.add_argument("--map-width", type=int, default=60, help="Fallback width if --map-sizes is empty")
    parser.add_argument("--map-height", type=int, default=40, help="Fallback height if --map-sizes is empty")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--view-length", type=int, default=6)
    parser.add_argument("--idle-limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-pair plots and CSV summaries",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a matplotlib window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_sizes = parse_map_sizes(args.map_sizes, (args.map_width, args.map_height))

    rule_alien_totals: dict[str, list[int]] = {
        spec.label: [0, 0, 0, 0]
        for spec in HUMAN_SPECS
    }

    print("Evaluating all human vs alien pairings...")
    for human_spec in HUMAN_SPECS:
        for alien_spec in ALIEN_SPECS:
            pair_dir = pair_output_dir(args.output_dir, human_spec.label, alien_spec.label)
            os.makedirs(pair_dir, exist_ok=True)
            pair_name = f"{human_spec.label} vs {alien_spec.label}"

            print(f"\n=== {pair_name} ===")
            csv_path = os.path.join(pair_dir, "summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                write_summary_header(writer)

                for width, height in map_sizes:
                    episode_seeds = build_episode_seeds(args.seed, width, height, args.episodes)

                    print(f"\nMap size: {width}x{height}")
                    stats = evaluate_matchup(
                        episode_seeds=episode_seeds,
                        width=width,
                        height=height,
                        max_steps=args.max_steps,
                        view_length=args.view_length,
                        idle_limit=args.idle_limit,
                        human_spec=human_spec,
                        alien_spec=alien_spec,
                    )
                    writer.writerow([
                        width,
                        height,
                        f"{ALPHA:.3f}",
                        f"{stats.escaped_percentages[0]:.2f}",
                        f"{stats.escaped_percentages[1]:.2f}",
                        f"{stats.escaped_percentages[2]:.2f}",
                        f"{stats.escaped_percentages[3]:.2f}",
                        f"{stats.avg_steps:.2f}",
                    ])
                    print(
                        "escaped_counts={} | escaped_percentages={} | avg_steps={:.1f}".format(
                            stats.escaped_counts,
                            [f"{v:.1f}%" for v in stats.escaped_percentages],
                            stats.avg_steps,
                        )
                    )

                    if alien_spec.key == "rule":
                        totals = rule_alien_totals[human_spec.label]
                        for idx in range(len(totals)):
                            totals[idx] += stats.escaped_counts[idx]

                    plot_path = os.path.join(pair_dir, f"escaped_counts_{width}x{height}.png")
                    title = f"{pair_name} (map {width}x{height}, alpha={ALPHA:+.1f})"
                    plot_escape_counts(
                        escaped_counts=stats.escaped_counts,
                        escaped_percentages=stats.escaped_percentages,
                        title=title,
                        output=plot_path,
                        show_window=not args.no_show,
                    )

    comparison_output = os.path.join(args.output_dir, "rule_alien_human_comparison.png")
    comparison_labels = [spec.label for spec in HUMAN_SPECS]
    plot_rule_alien_comparison(
        labels=comparison_labels,
        escaped_counts_by_label=rule_alien_totals,
        output=comparison_output,
        show_window=not args.no_show,
    )


if __name__ == "__main__":
    main()