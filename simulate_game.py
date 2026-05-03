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
    from agents.alien import AlienAgent, compute_fov
    from agents.human import Direction as HumanDirection, HumanAgent
    from game import Game
    from map_generator import MapGenerator, Tile
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from agents.alien import AlienAgent, compute_fov
    from agents.human import Direction as HumanDirection, HumanAgent
    from game import Game
    from map_generator import MapGenerator, Tile


@dataclass
class FrameState:
    step: int
    human_pos: tuple[int, int]
    human_hidden: bool
    alien_pos: tuple[int, int]
    known_map: np.ndarray
    alien_belief: np.ndarray
    human_sees_alien: bool = False
    alien_sees_human: bool = False
    radar_threat: str | None = None  # CRITICAL, CLOSE, NEAR, FAR
    radar_dist: int | None = None    # Actual Manhattan distance
    noise_ripple_pos: tuple[int, int] | None = None  # Position of heard noise
    alien_heard_pos: tuple[int, int] | None = None  # Where alien heard the sound
    alien_pursuing: bool = False  # Is alien actively pursuing sound


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
        alien_knowledge = game.alien_agent.knowledge.get_copy()
        human_hidden = game.human_agent._is_hidden()
        human_obs = game._human_cone_observation()
        alien_pos_yx = game.alien_pos
        human_sees_alien = False
        if 0 <= alien_pos_yx[0] < human_obs.shape[0] and 0 <= alien_pos_yx[1] < human_obs.shape[1]:
            human_sees_alien = human_obs[alien_pos_yx] != Game.UNSEEN_TILE
        alien_pos_xy = (alien_pos_yx[1], alien_pos_yx[0])
        human_pos_xy = (game.human_pos[1], game.human_pos[0])
        alien_fov = compute_fov(game.alien_agent.grid, alien_pos_xy, game.alien_agent.fov_radius)
        alien_sees_human = human_pos_xy in alien_fov and not human_hidden
        
        # Capture radar and noise info
        radar_threat = getattr(game, 'last_radar_threat', None)
        radar_dist = getattr(game, 'last_radar_dist', None)
        noise_ripple = game.last_noise_ripple
        
        # Capture alien clues (where alien heard sound)
        alien_heard = game.alien_agent.last_heard_pos
        alien_pursuing = alien_heard is not None and game.alien_agent.steps_since_heard <= 5
        
        frames.append(
            FrameState(
                step=step,
                human_pos=game.human_pos,
                human_hidden=human_hidden,
                alien_pos=game.alien_pos,
                known_map=known,
                alien_belief=alien_knowledge,
                human_sees_alien=human_sees_alien,
                alien_sees_human=alien_sees_human,
                radar_threat=radar_threat,
                radar_dist=radar_dist,
                noise_ripple_pos=noise_ripple,
                alien_heard_pos=alien_heard,
                alien_pursuing=alien_pursuing,
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
        "#1a1a2e",  # WALL (deep navy)
        "#2e2e4a",  # FLOOR (dark slate)
        "#9b59b6",  # VENT (purple)
        "#27ae60",  # HIDE (green)
        "#2980b9",  # PLAYER_START (blue)
        "#c0392b",  # ALIEN_START (red)
        "#f39c12",  # EXIT (amber)
    ]
    known_colors = [
        "#1a1a2e",  # WALL (deep navy)
        "#2e2e4a",  # FLOOR (dark slate)
        "#9b59b6",  # VENT (purple)
        "#27ae60",  # HIDE (green)
        "#2980b9",  # PLAYER_START (blue)
        "#c0392b",  # ALIEN_START (red)
        "#f39c12",  # EXIT (amber)
        "#1d1f26",  # UNKNOWN (very dark slate)
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
    player_seen_value = 8  # Use 8 for player seen in alien knowledge

    fig, (ax_world, ax_known, ax_belief) = plt.subplots(1, 3, figsize=(16, 5), dpi=120)
    fig.patch.set_facecolor("#000000")

    world_img = ax_world.imshow(grid, cmap=world_cmap, vmin=0, vmax=6)
    _ = world_img

    initial_known = np.where(frames[0].known_map == Game.UNSEEN_TILE, unknown_value, frames[0].known_map)
    known_img = ax_known.imshow(initial_known, cmap=known_cmap, vmin=0, vmax=7)

    # Alien knowledge map - convert UNKNOWN(-1) and PLAYER_SEEN(-2) to display values
    initial_alien_known = frames[0].alien_belief.copy()
    initial_alien_known = np.where(initial_alien_known == -1, unknown_value, initial_alien_known)  # UNKNOWN
    initial_alien_known = np.where(initial_alien_known == -2, player_seen_value, initial_alien_known)  # PLAYER_SEEN
    belief_img = ax_belief.imshow(initial_alien_known, cmap=known_cmap, vmin=0, vmax=7)

    for ax, title in ((ax_world, "World"), (ax_known, "Human Knowledge"), (ax_belief, "Alien Knowledge")):
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0f0f1e")

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
    alien_known_marker = ax_known.scatter(
        [ax], [ay], s=95, c="#FF4D6D", edgecolors="white", linewidths=0.7, marker="X", zorder=5
    )
    
    # Markers on belief map: alien position and player
    alien_belief_marker = ax_belief.scatter(
        [ax], [ay], s=95, c="#FF4D6D", edgecolors="white", linewidths=0.7, marker="X", zorder=5
    )
    human_belief_marker = ax_belief.scatter(
        [hx], [hy], s=95, c="#00D4FF", edgecolors="white", linewidths=0.7, marker="o", zorder=5
    )

    hidden_world_ring = ax_world.scatter(
        [hx], [hy], s=180, facecolors="none", edgecolors="#7CFF6B", linewidths=1.8, marker="o", zorder=4
    )
    hidden_known_ring = ax_known.scatter(
        [hx], [hy], s=180, facecolors="none", edgecolors="#7CFF6B", linewidths=1.8, marker="o", zorder=4
    )

    exit_y, exit_x = find_tile_pos(grid, Tile.EXIT)
    ax_world.scatter([exit_x], [exit_y], s=80, c="#7A77FF", marker="*", zorder=6)
    ax_known.scatter([exit_x], [exit_y], s=80, c="#7A77FF", marker="*", zorder=6)
    ax_belief.scatter([exit_x], [exit_y], s=80, c="#7A77FF", marker="*", zorder=6)

    # === RADAR THREAT INDICATOR ===
    # Display threat level color circle around player
    radar_threat_circle = plt.Circle((hx, hy), 0.6, fill=True, zorder=4)
    ax_world.add_patch(radar_threat_circle)
    
    # === NOISE RIPPLE VISUALIZATION ===
    # Show expanding ripple when sound is heard
    noise_ripple_circles = [
        plt.Circle((hx, hy), 0.3, fill=False, edgecolor='yellow', linewidth=1.5, zorder=3, linestyle='--'),
        plt.Circle((hx, hy), 0.6, fill=False, edgecolor='yellow', linewidth=1.2, zorder=3, linestyle='--', alpha=0.6),
        plt.Circle((hx, hy), 0.9, fill=False, edgecolor='yellow', linewidth=0.8, zorder=3, linestyle='--', alpha=0.3),
    ]
    for circle in noise_ripple_circles:
        ax_known.add_patch(circle)
    
    # === ALIEN HEARD LOCATION VISUALIZATION ===
    # Red dashed circle showing where alien heard the sound
    alien_heard_circle = plt.Circle((hx, hy), 0.5, fill=False, edgecolor='#FF0000', linewidth=2.0, zorder=4, linestyle=':', alpha=0.8)
    ax_belief.add_patch(alien_heard_circle)
    alien_heard_marker = ax_belief.scatter([hx], [hy], s=60, c='#FF0000', edgecolors='white', linewidths=0.5, marker='*', zorder=5, alpha=0.7)
    total_steps = len(frames) - 1
    status_text = fig.suptitle("", color="white", fontsize=12, x=0.02, ha="left", fontfamily="monospace")

    def update(frame_index: int):
        state = frames[frame_index]
        hy0, hx0 = state.human_pos
        ay0, ax0 = state.alien_pos

        human_world_marker.set_offsets([[hx0, hy0]])
        alien_world_marker.set_offsets([[ax0, ay0]])
        human_known_marker.set_offsets([[hx0, hy0]])
        alien_known_marker.set_offsets([[ax0, ay0]])
        alien_belief_marker.set_offsets([[ax0, ay0]])
        human_belief_marker.set_offsets([[hx0, hy0]])
        hidden_world_ring.set_offsets([[hx0, hy0]])
        hidden_known_ring.set_offsets([[hx0, hy0]])

        # === UPDATE RADAR THREAT INDICATOR COLOR ===
        threat_colors = {
            "CRITICAL": "#FF0000",  # Red
            "CLOSE": "#FF8800",     # Orange
            "NEAR": "#FFFF00",      # Yellow
            "FAR": "#00FF00",       # Green
            None: "#888888",        # Gray (no radar signal)
        }
        threat_color = threat_colors.get(state.radar_threat, "#888888")
        radar_threat_circle.set_center((hx0, hy0))
        radar_threat_circle.set_facecolor(threat_color)
        radar_threat_circle.set_alpha(0.5)

        if state.human_hidden:
            human_world_marker.set_facecolor("#7CFF6B")
            human_known_marker.set_facecolor("#7CFF6B")
            hidden_world_ring.set_visible(True)
            hidden_known_ring.set_visible(True)
        else:
            human_world_marker.set_facecolor("#00D4FF")
            human_known_marker.set_facecolor("#00D4FF")
            hidden_world_ring.set_visible(False)
            hidden_known_ring.set_visible(False)

        alien_known_marker.set_visible(state.human_sees_alien)
        human_belief_marker.set_visible(state.alien_sees_human)
        
        # === UPDATE NOISE RIPPLE POSITION AND VISIBILITY ===
        if state.noise_ripple_pos is not None:
            ny, nx = state.noise_ripple_pos
            for i, circle in enumerate(noise_ripple_circles):
                circle.set_center((nx, ny))
                circle.set_visible(True)
        else:
            # Hide ripples if no noise
            for circle in noise_ripple_circles:
                circle.set_visible(False)
        
        # === UPDATE ALIEN HEARD LOCATION ===
        if state.alien_heard_pos is not None:
            ahx, ahy = state.alien_heard_pos
            alien_heard_circle.set_center((ahx, ahy))
            alien_heard_marker.set_offsets([[ahx, ahy]])
            alien_heard_circle.set_visible(True)
            alien_heard_marker.set_visible(True)
            # Red color if actively pursuing, dimmed if old
            alpha = 0.9 if state.alien_pursuing else 0.4
            alien_heard_circle.set_alpha(alpha)
            alien_heard_marker.set_alpha(alpha)
        else:
            alien_heard_circle.set_visible(False)
            alien_heard_marker.set_visible(False)

        known = np.where(state.known_map == Game.UNSEEN_TILE, unknown_value, state.known_map)
        known_img.set_data(known)
        
        # Update alien knowledge map display
        alien_known = state.alien_belief.copy()
        alien_known = np.where(alien_known == -1, unknown_value, alien_known)  # UNKNOWN
        alien_known = np.where(alien_known == -2, player_seen_value, alien_known)  # PLAYER_SEEN
        belief_img.set_data(alien_known)

        status_text.set_text(
            "Step {step:4d}/{total:4d} | Outcome: {outcome:<18} | "
            "Human(y,x)=({hy:3d},{hx:3d}) | Alien(y,x)=({ay:3d},{ax:3d}) | "
            "Radar: {radar:<8} | Sound: {sound:<3} | AlienSound: {asound:<3} | Hidden: {hidden:<3}".format(
                step=state.step,
                total=total_steps,
                outcome=outcome,
                hy=hy0,
                hx=hx0,
                ay=ay0,
                ax=ax0,
                radar=state.radar_threat or "NONE",
                sound="YES" if state.noise_ripple_pos else "NO",
                asound="YES" if state.alien_pursuing else "NO",
                hidden="YES" if state.human_hidden else "NO",
            )
        )
        return human_world_marker, alien_world_marker, human_known_marker, alien_known_marker, alien_belief_marker, human_belief_marker, hidden_world_ring, hidden_known_ring, known_img, belief_img, status_text, radar_threat_circle, *noise_ripple_circles, alien_heard_circle, alien_heard_marker

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


def visualize_world_only(
    grid: np.ndarray,
    frames: list[FrameState],
    outcome: str,
    output_path: str,
    fps: int,
    show_window: bool,
):
    """Simple visualization: just the world map with player and alien positions."""
    world_cmap, _ = build_colormaps()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
    fig.patch.set_facecolor("#000000")

    world_img = ax.imshow(grid, cmap=world_cmap, vmin=0, vmax=6)
    ax.set_title("Game World", color="white", fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0f0f1e")

    hy, hx = frames[0].human_pos
    ay, ax_pos = frames[0].alien_pos
    
    human_marker = ax.scatter(
        [hx], [hy], s=120, c="#00D4FF", edgecolors="white", linewidths=1.0, marker="o", zorder=5
    )
    alien_marker = ax.scatter(
        [ax_pos], [ay], s=120, c="#FF4D6D", edgecolors="white", linewidths=1.0, marker="X", zorder=5
    )
    
    # Hidden indicator ring
    hidden_ring = ax.scatter(
        [hx], [hy], s=200, facecolors="none", edgecolors="#7CFF6B", linewidths=2.0, marker="o", zorder=4
    )
    
    # Exit marker
    exit_y, exit_x = find_tile_pos(grid, Tile.EXIT)
    ax.scatter([exit_x], [exit_y], s=100, c="#f39c12", marker="*", zorder=6)

    total_steps = len(frames) - 1
    status_text = fig.suptitle("", color="white", fontsize=12, x=0.02, ha="left", fontfamily="monospace")

    def update(frame_index: int):
        state = frames[frame_index]
        hy0, hx0 = state.human_pos
        ay0, ax0 = state.alien_pos

        human_marker.set_offsets([[hx0, hy0]])
        alien_marker.set_offsets([[ax0, ay0]])
        hidden_ring.set_offsets([[hx0, hy0]])
        
        if state.human_hidden:
            human_marker.set_facecolor("#7CFF6B")
            hidden_ring.set_visible(True)
        else:
            human_marker.set_facecolor("#00D4FF")
            hidden_ring.set_visible(False)
        
        status_text.set_text(
            "Step {step:4d}/{total:4d} | Outcome: {outcome:<18} | "
            "Player(y,x)=({hy:3d},{hx:3d}) | Alien(y,x)=({ay:3d},{ax:3d}) | Hidden: {hidden:<3}".format(
                step=state.step,
                total=total_steps,
                outcome=outcome,
                hy=hy0,
                hx=hx0,
                ay=ay0,
                ax=ax0,
                hidden="YES" if state.human_hidden else "NO",
            )
        )
        return human_marker, alien_marker, hidden_ring, status_text

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
    print(f"Saved world-only animation -> {output_path}")

    if show_window:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        backend = matplotlib.get_backend().lower()
        if not has_display or "agg" in backend:
            plt.close(fig)
            return
        plt.show()
        return

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate and visualize one game episode")
    parser.add_argument("--width", type=int, default=50, help="Map width")
    parser.add_argument("--height", type=int, default=35, help="Map height")
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
    parser.add_argument(
        "--style",
        type=str,
        default="full",
        choices=["full", "world"],
        help="Visualization style: 'full' (3 maps) or 'world' (world only)",
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

    # Convert from (y, x) to (x, y) for alien agent
    human_agent = HumanAgent(start_pos=human_start, start_dir=HumanDirection.NORTH)
    alien_agent = AlienAgent(grid=grid.copy(), start_pos=(alien_start[1], alien_start[0]))

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

    # Determine output path based on style
    output_path = args.output
    
    if args.style == "world":
        # World-only visualization
        world_output = output_path.replace(".gif", "_world.gif")
        visualize_world_only(
            grid=grid,
            frames=frames,
            outcome=outcome,
            output_path=world_output,
            fps=args.fps,
            show_window=not args.no_show,
        )
    else:
        # Full 3-map visualization (default)
        visualize(
            grid=grid,
            frames=frames,
            outcome=outcome,
            output_path=output_path,
            fps=args.fps,
            show_window=not args.no_show,
        )


if __name__ == "__main__":
    main()