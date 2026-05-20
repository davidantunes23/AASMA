from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence
import inspect
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

try:
    from map_generator import Tile
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from map_generator import Tile


KnowledgeMode = Literal["off", "on"]
AgentRole = Literal["human", "alien", "other"]

UNKNOWN_TILE = -1
PLAYER_SEEN_TILE = -2
RADAR_PING_TILE = -3
NOISE_RIPPLE_TILE = -4


@dataclass
class AgentSpec:
    """Describes one agent instance in a generic simulation."""

    label: str
    role: AgentRole
    agent: object
    view_radius: Optional[int] = None


@dataclass
class AgentFrame:
    label: str
    role: AgentRole
    position: tuple[int, int]
    hidden: bool = False
    knowledge_map: Optional[np.ndarray] = None


@dataclass
class SimulationFrame:
    step: int
    outcome: str | None
    agents: list[AgentFrame]


@dataclass
class GenericKnowledge:
    """Per-agent map knowledge used only for visualization.

    This is independent from the agent implementation. Any agent can run without
    exposing a knowledge API, while the simulation still records what the agent
    could see at each step.
    """

    grid_shape: tuple[int, int]
    known_map: np.ndarray = field(init=False)

    def __post_init__(self):
        self.known_map = np.full(self.grid_shape, UNKNOWN_TILE, dtype=np.int16)

    def update_from_observation(self, observation: np.ndarray):
        visible_mask = observation != UNKNOWN_TILE
        self.known_map[visible_mask] = observation[visible_mask]

    def get_copy(self) -> np.ndarray:
        return self.known_map.copy()


class GenericMapSimulation:
    """Generic map simulation that supports heterogeneous agent implementations.

    The simulator does not require a shared base class. It adapts to agents that
    expose either `_act(...)` or `step(...)`, supports multiple humans/aliens,
    and can record world-only or per-agent knowledge views.
    """

    WORLD_COLORS = [
        "#1a1a2e",  # WALL
        "#2e2e4a",  # FLOOR
        "#9b59b6",  # VENT
        "#27ae60",  # HIDE
        "#2980b9",  # PLAYER_START
        "#c0392b",  # ALIEN_START
        "#f39c12",  # EXIT
    ]
    KNOWLEDGE_COLORS = [
        "#1a1a2e",  # WALL
        "#2e2e4a",  # FLOOR
        "#9b59b6",  # VENT
        "#27ae60",  # HIDE
        "#2980b9",  # PLAYER_START
        "#c0392b",  # ALIEN_START
        "#f39c12",  # EXIT
        "#1d1f26",  # UNKNOWN
    ]

    HUMAN_MARKERS = ["o", "s", "^", "D", "P", "h"]
    ALIEN_MARKERS = ["X", "v", "<", ">", "*", "8"]

    def __init__(
        self,
        grid: np.ndarray,
        agents: Sequence[AgentSpec],
        knowledge_mode: KnowledgeMode = "off",
        default_human_view: int = 6,
        default_alien_view: int = 6,
    ):
        self.grid = grid
        self.agents = list(agents)
        self.knowledge_mode = knowledge_mode
        self.default_human_view = default_human_view
        self.default_alien_view = default_alien_view
        self.knowledge: Dict[str, GenericKnowledge] = {}

        if self.knowledge_mode == "on":
            for spec in self.agents:
                self.knowledge[spec.label] = GenericKnowledge(self.grid.shape)

    @staticmethod
    def _position_mode(spec: AgentSpec) -> Literal["yx", "xy"]:
        if spec.role == "alien":
            return "xy"
        return "yx"

    @classmethod
    def _get_position(cls, spec: AgentSpec) -> tuple[int, int]:
        agent = spec.agent
        mode = cls._position_mode(spec)
        if mode == "xy":
            x, y = getattr(agent, "pos")
            return int(y), int(x)
        y, x = getattr(agent, "pos") if hasattr(agent, "pos") else getattr(agent, "position")
        return int(y), int(x)

    @classmethod
    def _set_position(cls, spec: AgentSpec, position_yx: tuple[int, int]):
        agent = spec.agent
        mode = cls._position_mode(spec)
        if mode == "xy":
            y, x = position_yx
            setattr(agent, "pos", (int(x), int(y)))
        else:
            if hasattr(agent, "pos"):
                setattr(agent, "pos", (int(position_yx[0]), int(position_yx[1])))
            else:
                setattr(agent, "position", (int(position_yx[0]), int(position_yx[1])))

    @staticmethod
    def _direction_name(direction: object) -> str:
        return getattr(direction, "name", str(direction))

    @staticmethod
    def _action_name(action: object) -> str:
        return getattr(action, "name", str(action))

    @staticmethod
    def _in_bounds(grid: np.ndarray, position: tuple[int, int]) -> bool:
        y, x = position
        return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]

    @staticmethod
    def _step_from_direction(position: tuple[int, int], direction: object) -> tuple[int, int]:
        y, x = position
        name = GenericMapSimulation._direction_name(direction)
        if name == "NORTH":
            return (y - 1, x)
        if name == "EAST":
            return (y, x + 1)
        if name == "SOUTH":
            return (y + 1, x)
        if name == "WEST":
            return (y, x - 1)
        return position

    @staticmethod
    def _nearest_target(source: tuple[int, int], targets: Iterable[tuple[int, int]]) -> tuple[int, int] | None:
        candidates = list(targets)
        if not candidates:
            return None
        candidates.sort(key=lambda pos: abs(pos[0] - source[0]) + abs(pos[1] - source[1]))
        return candidates[0]

    def _view_radius_for(self, spec: AgentSpec) -> int:
        if spec.view_radius is not None:
            return spec.view_radius
        if spec.role == "human":
            return getattr(spec.agent, "view_length", self.default_human_view)
        if spec.role == "alien":
            return getattr(spec.agent, "fov_radius", self.default_alien_view)
        return self.default_human_view

    def _partial_observation(self, position: tuple[int, int], radius: int) -> np.ndarray:
        obs = np.full(self.grid.shape, UNKNOWN_TILE, dtype=np.int16)
        y, x = position
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) > radius:
                    continue
                ny, nx = y + dy, x + dx
                if self._in_bounds(self.grid, (ny, nx)):
                    obs[ny, nx] = int(self.grid[ny, nx])
        return obs

    def _update_knowledge(self, label: str, position: tuple[int, int], radius: int):
        if self.knowledge_mode != "on":
            return
        observation = self._partial_observation(position, radius)
        self.knowledge[label].update_from_observation(observation)

    def _is_hidden_position(self, position: tuple[int, int]) -> bool:
        return self._in_bounds(self.grid, position) and self.grid[position] == int(Tile.HIDE)

    @staticmethod
    def _call_with_signature(method, *candidates):
        signature = inspect.signature(method)
        parameter_count = len(signature.parameters)
        if parameter_count == 0:
            return method()
        return method(*candidates[:parameter_count])

    def _call_action_method(self, agent: object, obs: np.ndarray):
        method = getattr(agent, "_act", None)
        if method is None:
            method = getattr(agent, "act", None)
        if method is None:
            return None

        signature = inspect.signature(method)
        parameter_count = len(signature.parameters)
        if parameter_count == 1:
            return method(obs)
        if parameter_count == 2:
            return method(obs, None)
        return method(obs, None, None)

    def _call_step_method(
        self,
        spec: AgentSpec,
        target_yx: tuple[int, int],
        step_num: int,
    ):
        agent = spec.agent
        method = getattr(agent, "step", None)
        if method is None:
            return None

        mode = self._position_mode(spec)
        if mode == "xy":
            target = (target_yx[1], target_yx[0])
        else:
            target = target_yx

        signature = inspect.signature(method)
        parameter_count = len(signature.parameters)
        if parameter_count == 0:
            return method()
        if parameter_count == 1:
            return method(target)
        if parameter_count == 2:
            return method(target, target)
        return method(target, target, step_num)

    def _apply_action_result(self, agent: object, current_pos_yx: tuple[int, int], result) -> tuple[int, int]:
        if result is None:
            return current_pos_yx

        if isinstance(result, tuple) and len(result) == 2 and all(isinstance(v, (int, np.integer)) for v in result):
            return (int(result[0]), int(result[1]))

        if isinstance(result, (list, tuple)) and len(result) >= 2:
            action, direction = result[0], result[1]
            action_name = self._action_name(action)
            if action_name == "WAIT":
                return current_pos_yx
            if action_name == "WALK":
                next_pos = self._step_from_direction(current_pos_yx, direction)
                if self._in_bounds(self.grid, next_pos) and self.grid[next_pos] != int(Tile.WALL):
                    return next_pos
        return current_pos_yx

    def _snapshot_agents(self) -> list[AgentFrame]:
        snapshot: list[AgentFrame] = []
        for spec in self.agents:
            position = self._get_position(spec)
            hidden = bool(getattr(spec.agent, "hidden", False)) or (spec.role == "human" and self._is_hidden_position(position))
            knowledge_map = None
            if self.knowledge_mode == "on":
                knowledge_map = self.knowledge[spec.label].get_copy()
            snapshot.append(
                AgentFrame(
                    label=spec.label,
                    role=spec.role,
                    position=position,
                    hidden=hidden,
                    knowledge_map=knowledge_map,
                )
            )
        return snapshot

    def _is_human(self, spec: AgentSpec) -> bool:
        return spec.role == "human"

    def _is_alien(self, spec: AgentSpec) -> bool:
        return spec.role == "alien"

    def _exit_position(self) -> tuple[int, int] | None:
        matches = np.argwhere(self.grid == int(Tile.EXIT))
        if len(matches) == 0:
            return None
        y, x = matches[0]
        return int(y), int(x)

    def _has_collision(self, agents: list[AgentFrame]) -> bool:
        humans = [agent.position for agent in agents if agent.role == "human"]
        aliens = [agent.position for agent in agents if agent.role == "alien"]
        return any(human == alien for human in humans for alien in aliens)

    def run(self, max_steps: int = 200) -> tuple[list[SimulationFrame], str]:
        frames: list[SimulationFrame] = []
        exit_pos = self._exit_position()

        for step in range(max_steps + 1):
            current_agents = self._snapshot_agents()
            outcome = None
            if exit_pos is not None:
                for agent in current_agents:
                    if agent.role == "human" and agent.position == exit_pos:
                        outcome = f"human_reached_exit:{agent.label}"
                        break
            if outcome is None and self._has_collision(current_agents):
                outcome = "alien_caught_human"

            frames.append(SimulationFrame(step=step, outcome=outcome, agents=current_agents))
            if outcome is not None or step == max_steps:
                return frames, outcome or "max_steps_reached"

            # Update each agent in order using nearest opposing agent as its target.
            for spec in self.agents:
                current_position = self._get_position(spec)
                opposing_positions = [
                    self._get_position(other)
                    for other in self.agents
                    if other.label != spec.label
                    and other.role != spec.role
                    and not (other.role == "human" and self._is_hidden_position(self._get_position(other)))
                ]
                nearest_target = self._nearest_target(current_position, opposing_positions) or current_position
                radius = self._view_radius_for(spec)
                self._update_knowledge(spec.label, current_position, radius)
                observation = self._partial_observation(current_position, radius)

                step_method = getattr(spec.agent, "step", None)
                action_method = getattr(spec.agent, "_act", None) or getattr(spec.agent, "act", None)

                if callable(step_method):
                    new_pos = self._call_step_method(spec, nearest_target, step)
                    if isinstance(new_pos, tuple) and len(new_pos) == 2:
                        if self._position_mode(spec) == "xy":
                            # Step methods in this repo return (x, y) for alien-style agents.
                            self._set_position(spec, (int(new_pos[1]), int(new_pos[0])))
                        else:
                            self._set_position(spec, (int(new_pos[0]), int(new_pos[1])))
                    else:
                        # Some step methods mutate their own position and return it; sync from object.
                        self._set_position(spec, self._get_position(spec))
                elif callable(action_method):
                    result = self._call_action_method(spec.agent, observation)
                    new_pos = self._apply_action_result(spec.agent, current_position, result)
                    self._set_position(spec, new_pos)
                else:
                    # If the agent exposes neither interface, keep it stationary.
                    self._set_position(spec, current_position)

                if spec.role == "human":
                    setattr(spec.agent, "hidden", self._is_hidden_position(self._get_position(spec)))

        return frames, "max_steps_reached"

    def render(
        self,
        frames: list[SimulationFrame],
        outcome: str,
        output_path: str,
        fps: int = 8,
        show_window: bool = False,
        world_only: bool = False,
    ):
        world_cmap = ListedColormap(self.WORLD_COLORS)
        knowledge_cmap = ListedColormap(self.KNOWLEDGE_COLORS)
        unknown_value = 7

        has_single_human_alien = len(frames[0].agents) == 2 and {agent.role for agent in frames[0].agents} == {"human", "alien"}

        if world_only or self.knowledge_mode == "off":
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
            fig.patch.set_facecolor("#000000")
            ax.imshow(self.grid, cmap=world_cmap, vmin=0, vmax=6)
            ax.set_title("Game World", color="white", fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#0f0f1e")

            marker_artists = []
            hidden_rings = []
            color_map = self._role_colors()
            for index, agent in enumerate(frames[0].agents):
                color = color_map[agent.role][index % len(color_map[agent.role])]
                marker = self._role_marker(agent.role, index)
                y, x = agent.position
                artist = ax.scatter([x], [y], s=120, c=color, edgecolors="white", linewidths=1.0, marker=marker, zorder=5)
                marker_artists.append(artist)
                if agent.role == "human":
                    hidden_rings.append(
                        ax.scatter([x], [y], s=210, facecolors="none", edgecolors="#7CFF6B", linewidths=2.0, marker="o", zorder=4)
                    )

            exit_pos = self._exit_position()
            if exit_pos is not None:
                exit_y, exit_x = exit_pos
                ax.scatter([exit_x], [exit_y], s=100, c="#f39c12", marker="*", zorder=6)

            status_text = fig.suptitle("", color="white", fontsize=12, x=0.02, ha="left", fontfamily="monospace")

            def update(frame_index: int):
                state = frames[frame_index]
                human_ring_idx = 0
                for artist, agent in zip(marker_artists, state.agents):
                    y, x = agent.position
                    artist.set_offsets([[x, y]])
                    if agent.hidden:
                        artist.set_alpha(0.55)
                        if agent.role == "human":
                            hidden_rings[human_ring_idx].set_offsets([[x, y]])
                            hidden_rings[human_ring_idx].set_visible(True)
                            human_ring_idx += 1
                    else:
                        artist.set_alpha(1.0)
                        if agent.role == "human":
                            hidden_rings[human_ring_idx].set_visible(False)
                            human_ring_idx += 1
                status_text.set_text(f"Step {state.step:4d}/{len(frames)-1:4d} | Outcome: {outcome}")
                return [*marker_artists, *hidden_rings, status_text]

            animation = FuncAnimation(fig, update, frames=len(frames), interval=max(1, int(1000 / max(1, fps))), blit=False, repeat=False)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            animation.save(output_path, writer=PillowWriter(fps=max(1, fps)))
            print(f"Saved animation -> {output_path}")
            if show_window:
                self._maybe_show(fig)
            else:
                plt.close(fig)
            return

        # Full mode: world + one knowledge panel per agent.
        n_panels = 1 + len(frames[0].agents)
        if has_single_human_alien:
            fig_size = (16, 5)
        else:
            fig_size = (max(16, 5 * n_panels), 5)

        fig, axes = plt.subplots(1, n_panels, figsize=fig_size, dpi=120)
        fig.patch.set_facecolor("#000000")
        if n_panels == 1:
            axes = [axes]

        world_ax = axes[0]
        world_ax.imshow(self.grid, cmap=world_cmap, vmin=0, vmax=6)
        world_ax.set_title("World", color="white", fontsize=12, fontweight="bold")
        world_ax.set_xticks([])
        world_ax.set_yticks([])
        world_ax.set_facecolor("#0f0f1e")

        exit_pos = self._exit_position()
        if exit_pos is not None:
            exit_y, exit_x = exit_pos
            world_ax.scatter([exit_x], [exit_y], s=100, c="#f39c12", marker="*", zorder=6)

        color_map = self._role_colors()
        world_markers = []
        hidden_rings = []
        for index, agent in enumerate(frames[0].agents):
            color = color_map[agent.role][index % len(color_map[agent.role])]
            marker = self._role_marker(agent.role, index)
            y, x = agent.position
            world_markers.append(
                world_ax.scatter([x], [y], s=120, c=color, edgecolors="white", linewidths=1.0, marker=marker, zorder=5)
            )
            if agent.role == "human":
                hidden_rings.append(
                    world_ax.scatter([x], [y], s=210, facecolors="none", edgecolors="#7CFF6B", linewidths=2.0, marker="o", zorder=4)
                )

        knowledge_axes = axes[1:]
        knowledge_images = []
        knowledge_markers = []
        for axis, agent in zip(knowledge_axes, frames[0].agents):
            axis.set_title(f"{agent.label} ({agent.role})", color="white", fontsize=11, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#0f0f1e")
            if agent.knowledge_map is None:
                initial = np.full(self.grid.shape, unknown_value, dtype=np.int16)
            else:
                initial = np.where(agent.knowledge_map == UNKNOWN_TILE, unknown_value, agent.knowledge_map)
            image = axis.imshow(initial, cmap=knowledge_cmap, vmin=0, vmax=7)
            knowledge_images.append(image)
            y, x = agent.position
            if agent.role == "alien":
                marker_style = "X"
                marker_color = "#FF4D6D"
            else:
                marker_style = "o"
                marker_color = "#00D4FF"
            knowledge_markers.append(
                axis.scatter([x], [y], s=95, c=marker_color, edgecolors="white", linewidths=0.7, marker=marker_style, zorder=5)
            )

        status_text = fig.suptitle("", color="white", fontsize=12, x=0.02, ha="left", fontfamily="monospace")

        def update(frame_index: int):
            state = frames[frame_index]
            human_ring_idx = 0
            for artist, agent in zip(world_markers, state.agents):
                y, x = agent.position
                artist.set_offsets([[x, y]])
                artist.set_alpha(0.55 if agent.hidden else 1.0)
                if agent.role == "human":
                    hidden_rings[human_ring_idx].set_offsets([[x, y]])
                    hidden_rings[human_ring_idx].set_visible(agent.hidden)
                    human_ring_idx += 1

            for image, marker, agent in zip(knowledge_images, knowledge_markers, state.agents):
                if agent.knowledge_map is None:
                    data = np.full(self.grid.shape, unknown_value, dtype=np.int16)
                else:
                    data = np.where(agent.knowledge_map == UNKNOWN_TILE, unknown_value, agent.knowledge_map)
                image.set_data(data)
                y, x = agent.position
                marker.set_offsets([[x, y]])

            status_text.set_text(f"Step {state.step:4d}/{len(frames)-1:4d} | Outcome: {outcome}")
            return [*world_markers, *hidden_rings, *knowledge_images, *knowledge_markers, status_text]

        animation = FuncAnimation(fig, update, frames=len(frames), interval=max(1, int(1000 / max(1, fps))), blit=False, repeat=False)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        animation.save(output_path, writer=PillowWriter(fps=max(1, fps)))
        print(f"Saved animation -> {output_path}")
        if show_window:
            self._maybe_show(fig)
        else:
            plt.close(fig)

    @staticmethod
    def _role_colors() -> dict[AgentRole, list[str]]:
        return {
            "human": ["#00D4FF", "#4DD6FF", "#6FE7FF", "#8BE9FD"],
            "alien": ["#FF4D6D", "#FF7A59", "#FF8E72", "#FF6B6B"],
            "other": ["#A0A0A0", "#C0C0C0"],
        }

    @staticmethod
    def _role_marker(role: AgentRole, index: int) -> str:
        if role == "alien":
            return GenericMapSimulation.ALIEN_MARKERS[index % len(GenericMapSimulation.ALIEN_MARKERS)]
        if role == "human":
            return GenericMapSimulation.HUMAN_MARKERS[index % len(GenericMapSimulation.HUMAN_MARKERS)]
        return "o"

    @staticmethod
    def _maybe_show(fig):
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        backend = matplotlib.get_backend().lower()
        if not has_display or backend == "agg":
            plt.close(fig)
            return
        plt.show()


def build_agent_spec(label: str, role: AgentRole, agent: object, view_radius: Optional[int] = None) -> AgentSpec:
    return AgentSpec(label=label, role=role, agent=agent, view_radius=view_radius)
