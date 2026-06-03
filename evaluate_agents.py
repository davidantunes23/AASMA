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
    from agents.base import Direction
    from agents.coop_role_human import CoopRoleHumanAgent
    from agents.omniscient_human import OmniscientHumanAgent
    from agents.random_human import RandomHumanAgent
    from agents.role_human import RoleHumanAgent
    from agents.rule_alien import AlienAgent
    from agents.rule_human import HumanAgent
    from map_generator import MapGenerator, Tile
    from simulation import GenericMapSimulation, SimulationFrame, build_agent_spec
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from agents.base import Direction
    from agents.coop_role_human import CoopRoleHumanAgent
    from agents.omniscient_human import OmniscientHumanAgent
    from agents.random_human import RandomHumanAgent
    from agents.role_human import RoleHumanAgent
    from agents.rule_alien import AlienAgent
    from agents.rule_human import HumanAgent
    from map_generator import MapGenerator, Tile
    from simulation import GenericMapSimulation, SimulationFrame, build_agent_spec


MISSION_COUNT = 2
MISSION_STEPS = 10
HUMAN_COUNT = 3
ALPHA = 0.0
DEFAULT_OUTPUT_DIR = "output/eval"


@dataclass
class EscapeStats:
    escaped_counts: list[int]
    escaped_percentages: list[float]
    avg_steps: float


@dataclass
class EpisodeRun:
    frames: list[SimulationFrame]
    outcome: str
    outcome_step: int
    timeout: bool


@dataclass
class EpisodeMetrics:
    seed: int
    steps: int
    escaped_count: int
    captured_count: int
    outcome: str
    outcome_type: str
    escape_steps: dict[str, int | None]
    capture_steps: dict[str, int | None]
    active_counts: list[int]
    mission_counts: list[int]


@dataclass
class MatchupMetrics:
    escape_stats: EscapeStats
    episodes: list[EpisodeMetrics]
    human_labels: list[str]


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
    HumanSpec("random", "random_human", HUMAN_COUNT, role_based=False, omniscient=False),
    HumanSpec("rule", "rule_human", HUMAN_COUNT, role_based=False, omniscient=False),
    HumanSpec("role", "role_human_3", HUMAN_COUNT, role_based=True, omniscient=False),
    HumanSpec("coop", "coop_role_human_3", HUMAN_COUNT, role_based=True, omniscient=False),
    HumanSpec("omniscient", "omniscient_human", HUMAN_COUNT, role_based=True, omniscient=True),
]

ALIEN_SPECS = [
    AlienSpec("rule", "rule_alien"),
]


def find_tile(grid: np.ndarray, tile: Tile) -> tuple[int, int]:
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
    start_pos: tuple[int, int],
    seed: int,
    view_length: int,
) -> object:
    if spec.key == "rule":
        return AlienAgent(start_pos=start_pos, view_length=view_length)
    raise ValueError(f"Unknown alien model key: {spec.key}")


def build_simulation(
    grid: np.ndarray,
    view_length: int,
    seed: int,
    human_spec: HumanSpec,
    alien_spec: AlienSpec,
    p_noise: float = 0.10,
) -> GenericMapSimulation:
    human_start = find_tile(grid, Tile.PLAYER_START)
    alien_start = find_tile(grid, Tile.ALIEN_START)
    human_positions = choose_human_positions(
        grid,
        human_start,
        human_spec.count,
        avoid={alien_start},
    )

    human_agents = build_humans(human_spec, grid, human_positions, seed, view_length)
    alien_agent = build_alien(alien_spec, alien_start, seed, view_length)

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
        p_noise=p_noise,
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
    p_noise: float = 0.10,
) -> EpisodeRun:
    simulation = build_simulation(grid, view_length, seed, human_spec, alien_spec, p_noise=p_noise)
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
        return EpisodeRun(
            frames=frames[: idle_trigger_idx + 1],
            outcome="idle_timeout",
            outcome_step=int(frame.step),
            timeout=True,
        )

    timeout = outcome == "max_steps_reached"
    return EpisodeRun(
        frames=frames,
        outcome=outcome,
        outcome_step=outcome_step,
        timeout=timeout,
    )


def classify_outcome_type(
    outcome: str,
    escaped_count: int,
    captured_count: int,
    total_humans: int,
    timeout: bool,
) -> str:
    if timeout or outcome in {"max_steps_reached", "idle_timeout"}:
        return "timeout"
    if escaped_count >= total_humans:
        return "full_escape"
    if captured_count >= total_humans:
        return "full_capture"
    if escaped_count > 0:
        return "partial_escape"
    return "timeout"


def extract_episode_metrics(
    seed: int,
    run: EpisodeRun,
    total_humans: int,
    max_steps: int,
) -> EpisodeMetrics:
    frames = run.frames
    if not frames:
        return EpisodeMetrics(
            seed=seed,
            steps=0,
            escaped_count=0,
            captured_count=0,
            outcome=run.outcome,
            outcome_type="timeout",
            escape_steps={},
            capture_steps={},
            active_counts=[0] * (max_steps + 1),
            mission_counts=[0] * (max_steps + 1),
        )

    human_labels = [a.label for a in frames[0].agents if a.role == "human"]
    escape_steps = {label: None for label in human_labels}
    capture_steps = {label: None for label in human_labels}
    active_by_step: dict[int, int] = {}
    mission_by_step: dict[int, int] = {}

    for frame in frames:
        step = min(int(frame.step), max_steps)
        escaped = set(frame.escaped_humans or [])
        captured = set(frame.captured_humans or [])
        for label in escaped:
            if label in escape_steps and escape_steps[label] is None:
                escape_steps[label] = step
        for label in captured:
            if label in capture_steps and capture_steps[label] is None:
                capture_steps[label] = step
        active_by_step[step] = max(0, total_humans - len(escaped) - len(captured))
        mission_by_step[step] = len(frame.mission_done_events or [])

    active_counts = [0] * (max_steps + 1)
    mission_counts = [0] * (max_steps + 1)
    last_active = total_humans
    last_missions = 0
    for step in range(max_steps + 1):
        if step in active_by_step:
            last_active = active_by_step[step]
        active_counts[step] = last_active
        if step in mission_by_step:
            last_missions = mission_by_step[step]
        mission_counts[step] = last_missions

    final_frame = frames[-1]
    escaped_count = len(final_frame.escaped_humans or [])
    captured_count = len(final_frame.captured_humans or [])
    outcome_type = classify_outcome_type(
        run.outcome,
        escaped_count,
        captured_count,
        total_humans,
        run.timeout,
    )

    return EpisodeMetrics(
        seed=seed,
        steps=min(int(run.outcome_step), max_steps),
        escaped_count=escaped_count,
        captured_count=captured_count,
        outcome=run.outcome,
        outcome_type=outcome_type,
        escape_steps=escape_steps,
        capture_steps=capture_steps,
        active_counts=active_counts,
        mission_counts=mission_counts,
    )


def evaluate_matchup(
    episode_seeds: list[int],
    width: int,
    height: int,
    max_steps: int,
    view_length: int,
    idle_limit: int,
    human_spec: HumanSpec,
    alien_spec: AlienSpec,
    p_noise: float = 0.10,
    stop_on_timeout: bool = False,
) -> MatchupMetrics:
    escaped_counts = [0, 0, 0, 0]
    total_steps = 0
    episodes: list[EpisodeMetrics] = []
    human_labels: list[str] = []
    total_humans = human_spec.count

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
        run = run_episode(
            grid=grid,
            max_steps=max_steps,
            view_length=view_length,
            idle_limit=idle_limit,
            seed=episode_seed,
            human_spec=human_spec,
            alien_spec=alien_spec,
            p_noise=p_noise,
        )
        if stop_on_timeout and run.outcome == "max_steps_reached":
            print(f"\n[STOP] max_steps reached: seed={episode_seed}, map={width}x{height}, human={human_spec.key}")
            print(f"python run.py --human-class {human_spec.key} --human-count {human_spec.count} --seed {episode_seed} --width {width} --height {height} --max-steps {max_steps} --no-show")
            sys.exit(0)

        metrics = extract_episode_metrics(
            seed=episode_seed,
            run=run,
            total_humans=total_humans,
            max_steps=max_steps,
        )
        episodes.append(metrics)
        if not human_labels and run.frames:
            human_labels = [a.label for a in run.frames[0].agents if a.role == "human"]
        total_steps += metrics.steps
        escaped_idx = max(0, min(metrics.escaped_count, HUMAN_COUNT))
        escaped_counts[escaped_idx] += 1

    total = max(len(episode_seeds), 1)
    escaped_percentages = [count / total * 100.0 for count in escaped_counts]
    return MatchupMetrics(
        escape_stats=EscapeStats(
            escaped_counts=escaped_counts,
            escaped_percentages=escaped_percentages,
            avg_steps=total_steps / total,
        ),
        episodes=episodes,
        human_labels=human_labels,
    )


def build_episode_seeds(base_seed: int, width: int, height: int, episodes: int) -> list[int]:
    mix_seed = base_seed ^ (width << 16) ^ height
    rng = np.random.default_rng(mix_seed)
    return [int(rng.integers(0, 2**31 - 1)) for _ in range(episodes)]


def plot_survivor_count_comparison(
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


def plot_mission_completion_comparison(
    labels: list[str],
    mission_counts_by_label: dict[str, list[int]],
    output: str,
    show_window: bool,
):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))

    x = np.arange(len(labels))

    colors = [
        "#c0392b",  # 0 missions
        "#f39c12",  # 1 mission
        "#27ae60",  # 2 missions
    ]

    bottoms = np.zeros(len(labels))

    for completed, color in enumerate(colors):
        values = [
            mission_counts_by_label.get(label, [0, 0, 0])[completed]
            for label in labels
        ]

        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=color,
            label=f"{completed} missions",
        )

        bottoms += np.array(values)

    ax.set_title("Mission completion vs rule-based alien")
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
        has_display = bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
        )
        backend = matplotlib.get_backend().lower()
        if not has_display or backend == "agg":
            plt.close(fig)
            return
        plt.show()
        return

    plt.close(fig)

def plot_survival_curve(
    episodes: list[EpisodeMetrics],
    total_humans: int,
    max_steps: int,
    title: str,
    output: str,
    show_window: bool,
):
    if not episodes:
        return
    max_episode_step = max((ep.steps for ep in episodes), default=0)
    max_step = min(max_steps, max_episode_step)
    counts = np.array([ep.active_counts[: max_step + 1] for ep in episodes], dtype=float)
    mean = counts.mean(axis=0)
    p25 = np.percentile(counts, 25, axis=0)
    p75 = np.percentile(counts, 75, axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(max_step + 1)
    ax.plot(x, mean, color="#2E86C1", label="mean active")
    ax.fill_between(x, p25, p75, color="#2E86C1", alpha=0.2, label="25-75%")
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("humans active")
    ax.set_ylim(0, total_humans)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)

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


def plot_capture_escape_timeline(
    episodes: list[EpisodeMetrics],
    human_labels: list[str],
    title: str,
    output: str,
    show_window: bool,
):
    if not episodes:
        return
    max_episode_step = max((ep.steps for ep in episodes), default=0)
    fig_height = max(4.5, 0.25 * len(episodes) + 1.5)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    offsets = [
        (idx - (len(human_labels) - 1) / 2) * 0.12
        for idx in range(len(human_labels))
    ]

    for ep_idx, ep in enumerate(episodes):
        ax.hlines(ep_idx, 0, ep.steps, color="#d0d0d0", linewidth=0.6)
        for label, offset in zip(human_labels, offsets):
            y = ep_idx + offset
            if ep.escape_steps.get(label) is not None:
                ax.scatter(ep.escape_steps[label], y, marker="o", s=26, c="#27ae60")
            elif ep.capture_steps.get(label) is not None:
                ax.scatter(ep.capture_steps[label], y, marker="x", s=28, c="#c0392b")
            else:
                ax.scatter(ep.steps, y, marker="^", s=22, c="#7f8c8d")

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("episode index")
    ax.set_yticks(range(len(episodes)))
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0, max_episode_step)

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


def plot_steps_boxplot(
    labels: list[str],
    episode_steps_by_label: dict[str, list[int]],
    output: str,
    show_window: bool,
):
    filtered_labels = []
    filtered_data = []

    for label in labels:
        values = episode_steps_by_label.get(label, [])

        if len(values) == 0:
            continue

        filtered_labels.append(label)
        filtered_data.append(values)

    if not filtered_data:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.boxplot(
        filtered_data,
        labels=filtered_labels,
        showfliers=True,
    )

    ax.set_title("Episode duration vs rule-based alien")
    ax.set_xlabel("human model")
    ax.set_ylabel("steps")

    plt.setp(
        ax.get_xticklabels(),
        rotation=20,
        ha="right",
    )

    ax.grid(True, axis="y", alpha=0.25)

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        print(f"Saved plot -> {output}")

    if show_window:
        has_display = bool(
            os.environ.get("DISPLAY")
            or os.environ.get("WAYLAND_DISPLAY")
        )
        backend = matplotlib.get_backend().lower()

        if not has_display or backend == "agg":
            plt.close(fig)
            return

        plt.show()
        return

    plt.close(fig)

def plot_bar_comparison(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output: str,
    show_window: bool,
):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))

    x = np.arange(len(labels))
    ax.bar(x, values)

    ax.set_title(title)
    ax.set_xlabel("human model")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)

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


def plot_output_dir(output_root: str, plot_type: str, human_label: str, alien_label: str) -> str:
    return os.path.join(output_root, "plots", plot_type)


def csv_output_dir(output_root: str, csv_type: str, human_label: str, alien_label: str) -> str:
    return os.path.join(output_root, "csv")


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


def write_summary_table_header(writer: csv.writer) -> None:
    writer.writerow([
        "map_width",
        "map_height",
        "episodes",
        "avg_steps",
        "escaped_0_count",
        "escaped_1_count",
        "escaped_2_count",
        "escaped_3_count",
        "alien_captures",
        "timeout_rate_pct",
        "mission_completion_rate_pct",
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
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--view-length", type=int, default=6)
    parser.add_argument("--idle-limit", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for per-pair plots and CSV summaries",
    )
    parser.add_argument(
        "--humans",
        nargs="+",
        choices=["random", "omniscient", "rule", "role", "coop", "all"],
        default=["all"],
        help="Which human model(s) to evaluate (space-separated)",
    )
    parser.add_argument(
        "--aliens",
        nargs="+",
        choices=["rule", "all"],
        default=["all"],
        help="Which alien model(s) to evaluate (space-separated)",
    )
    parser.add_argument("--noise-prob", type=float, default=0.10,
                        help="Probability of emitting noise each step (default: 0.10)")
    parser.add_argument("--stop-on-timeout", action="store_true",
                        help="Print seed and exit on first max-steps episode")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a matplotlib window for each plot (default: save only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_sizes = parse_map_sizes(args.map_sizes, (args.map_width, args.map_height))
    if "all" in args.aliens:
        selected_aliens = ALIEN_SPECS
    else:
        selected_aliens = [spec for spec in ALIEN_SPECS if spec.key in args.aliens]

    if "all" in args.humans:
        selected_humans = HUMAN_SPECS
    else:
        selected_humans = [spec for spec in HUMAN_SPECS if spec.key in args.humans]

    rule_alien_totals: dict[str, list[int]] = {
        spec.label: [0, 0, 0, 0]
        for spec in HUMAN_SPECS
    }

    rule_alien_mission_totals: dict[str, list[int]] = {
        spec.label: [0, 0, 0]  # completed 0,1,2 missions
        for spec in HUMAN_SPECS
    }

    rule_alien_episode_steps: dict[str, list[int]] = {
        spec.label: []
        for spec in HUMAN_SPECS
    }

    rule_alien_steps_per_mission: dict[str, list[float]] = {
        spec.label: []
        for spec in HUMAN_SPECS
    }

    print("Evaluating human vs alien pairings")
    for human_spec in selected_humans:
        for alien_spec in selected_aliens:
            pair_name = f"{human_spec.label} vs {alien_spec.label}"

            print(f"\n=== {pair_name} ===")
            csv_dir = csv_output_dir(args.output_dir, "summary", human_spec.label, alien_spec.label)
            os.makedirs(csv_dir, exist_ok=True)
            pair_slug = f"{human_spec.label}_vs_{alien_spec.label}"
            csv_path = os.path.join(csv_dir, f"summary_{pair_slug}.csv")
            summary_table_path = os.path.join(csv_dir, f"summary_table_{pair_slug}.csv")
            with (
                open(csv_path, "w", newline="", encoding="utf-8") as handle,
                open(summary_table_path, "w", newline="", encoding="utf-8") as table_handle,
            ):
                writer = csv.writer(handle)
                table_writer = csv.writer(table_handle)
                write_summary_header(writer)
                write_summary_table_header(table_writer)

                for width, height in map_sizes:
                    episode_seeds = build_episode_seeds(args.seed, width, height, args.episodes)

                    print(f"\nMap size: {width}x{height}")
                    matchup = evaluate_matchup(
                        episode_seeds=episode_seeds,
                        width=width,
                        height=height,
                        max_steps=args.max_steps,
                        view_length=args.view_length,
                        idle_limit=args.idle_limit,
                        human_spec=human_spec,
                        alien_spec=alien_spec,
                        p_noise=args.noise_prob,
                        stop_on_timeout=args.stop_on_timeout,
                    )
                    stats = matchup.escape_stats
                    episodes = matchup.episodes
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

                    escaped_counts = stats.escaped_counts
                    timeout_count = sum(1 for ep in episodes if ep.outcome_type == "timeout")
                    timeout_rate = timeout_count / max(len(episodes), 1) * 100.0
                    mission_completed = [
                        (ep.mission_counts[-1] if ep.mission_counts else 0)
                        for ep in episodes
                    ]
                    if MISSION_COUNT > 0 and episodes:
                        mission_completion_rate = (
                            sum(mission_completed) / (len(episodes) * MISSION_COUNT) * 100.0
                        )
                    else:
                        mission_completion_rate = 0.0
                    alien_captures = sum(ep.captured_count for ep in episodes)
                    
                    table_writer.writerow([
                        width,
                        height,
                        len(episodes),
                        f"{stats.avg_steps:.2f}",
                        escaped_counts[0] if len(escaped_counts) > 0 else 0,
                        escaped_counts[1] if len(escaped_counts) > 1 else 0,
                        escaped_counts[2] if len(escaped_counts) > 2 else 0,
                        escaped_counts[3] if len(escaped_counts) > 3 else 0,
                        alien_captures,
                        f"{timeout_rate:.2f}",
                        f"{mission_completion_rate:.2f}",
                    ])

                    if alien_spec.key == "rule":
                        totals = rule_alien_totals[human_spec.label]
                        for idx in range(len(totals)):
                            totals[idx] += stats.escaped_counts[idx]

                        rule_alien_episode_steps[human_spec.label].extend(
                            ep.steps for ep in episodes
                        )

                        mission_histogram = rule_alien_mission_totals[human_spec.label]

                        for ep in episodes:
                            completed = ep.mission_counts[-1] if ep.mission_counts else 0
                            completed = max(0, min(completed, MISSION_COUNT))
                            mission_histogram[completed] += 1

                        completed_missions = sum(mission_completed)
                        if completed_missions > 0:
                            rule_alien_steps_per_mission[human_spec.label].append(
                                sum(ep.steps for ep in episodes) / completed_missions
                            )

                    title = f"{pair_name} (map {width}x{height}, alpha={ALPHA:+.1f})"
                    survival_dir = plot_output_dir(
                        args.output_dir,
                        "survival_curve",
                        human_spec.label,
                        alien_spec.label,
                    )
                    os.makedirs(survival_dir, exist_ok=True)
                    plot_survival_curve(
                        episodes=episodes,
                        total_humans=human_spec.count,
                        max_steps=args.max_steps,
                        title=f"Survival curve - {title}",
                        output=os.path.join(
                            survival_dir,
                            f"survival_curve_{human_spec.label}_vs_{alien_spec.label}_{width}x{height}.png",
                        ),
                        show_window=args.show,
                    )
                    timeline_dir = plot_output_dir(
                        args.output_dir,
                        "capture_escape_timeline",
                        human_spec.label,
                        alien_spec.label,
                    )
                    os.makedirs(timeline_dir, exist_ok=True)
                    plot_capture_escape_timeline(
                        episodes=episodes,
                        human_labels=matchup.human_labels,
                        title=f"Capture/escape timeline - {title}",
                        output=os.path.join(
                            timeline_dir,
                            f"capture_escape_timeline_{human_spec.label}_vs_{alien_spec.label}_{width}x{height}.png",
                        ),
                        show_window=args.show,
                    )

    if any(spec.key == "rule" for spec in selected_aliens):
        comparison_dir = os.path.join(args.output_dir, "plots", "rule_alien_comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        comparison_labels = [spec.label for spec in selected_humans]

        comparison_output = os.path.join(comparison_dir, "survivor_counts.png")
        plot_survivor_count_comparison(
            labels=comparison_labels,
            escaped_counts_by_label=rule_alien_totals,
            output=comparison_output,
            show_window=args.show,
        )

        comparison_output = os.path.join(comparison_dir, "episode_steps_boxplot.png")

        plot_steps_boxplot(
            labels=comparison_labels,
            episode_steps_by_label=rule_alien_episode_steps,
            output=comparison_output,
            show_window=args.show,
        )

        comparison_output = os.path.join(comparison_dir, "mission_completion_counts.png")

        print("\nMission completion statistics:")

        for label in comparison_labels:
            counts = rule_alien_mission_totals[label]

            total_episodes = sum(counts)

            if total_episodes == 0:
                continue

            percentages = [
                100.0 * count / total_episodes
                for count in counts
            ]

            values = rule_alien_steps_per_mission.get(label, [])

            avg_steps_per_mission = (
                np.mean(values)
                if values
                else 0.0
            )

            print(
                f"{label}: "
                f"0 missions={percentages[0]:.1f}% "
                f"1 mission={percentages[1]:.1f}% "
                f"2 missions={percentages[2]:.1f}% | "
                f"avg steps/mission={avg_steps_per_mission:.1f}"
            )

        plot_mission_completion_comparison(
            labels=comparison_labels,
            mission_counts_by_label=rule_alien_mission_totals,
            output=comparison_output,
            show_window=args.show,
        )

        comparison_output = os.path.join(comparison_dir, "average_steps_per_mission.png")
        plot_bar_comparison(
            labels=comparison_labels,
            values=[
                np.mean(rule_alien_steps_per_mission[label])
                if rule_alien_steps_per_mission[label]
                else 0.0
                for label in comparison_labels
            ],
            title="Average steps per completed mission",
            ylabel="steps / mission",
            output=comparison_output,
            show_window=args.show,
        )


if __name__ == "__main__":
    main()