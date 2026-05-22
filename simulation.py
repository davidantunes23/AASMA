from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from agents.base import Direction, cone_fov
from map_generator import Tile

KnowledgeMode = Literal["off", "on"]
AgentRole = Literal["human", "alien", "other"]

UNKNOWN_TILE = -1
PLAYER_SEEN_TILE = -2
RADAR_PING_TILE = -3
NOISE_RIPPLE_TILE = -4

RADAR_BANDS = {
    "CRITICAL": (0, 7),
    "CLOSE": (8, 12),
    "NEAR": (13, 18),
    "FAR": (19, float("inf")),
}


@dataclass
class AgentSpec:
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
    mode: str = ""  # human: EXPLORING/FLEEING/HIDING; alien: SEARCH/INVESTIGATE/HUNT
    fov: frozenset = field(default_factory=frozenset)         # (y,x) cells visible this step
    visible_opponent: Optional[tuple[int, int]] = None        # opponent pos if inside FOV and not hidden


@dataclass
class SimulationFrame:
    step: int
    outcome: str | None
    agents: list[AgentFrame]
    exit_unlocked: bool = True
    noise_ripple_pos: tuple[int, int] | None = None
    radar_threat: str | None = None
    alien_heard_pos: tuple[int, int] | None = None  # (y, x) where alien heard a sound


THREAT_COLORS = {
    "CRITICAL": "#FF0000",
    "CLOSE":    "#FF8800",
    "NEAR":     "#FFFF00",
    "FAR":      "#00FF00",
}


@dataclass
class GenericKnowledge:
    grid_shape: tuple[int, int]
    known_map: np.ndarray = field(init=False)

    def __post_init__(self):
        self.known_map = np.full(self.grid_shape, UNKNOWN_TILE, dtype=np.int16)

    def update_from_observation(self, observation: np.ndarray):
        # Only store valid tile IDs (0–7). Marker values like RADAR_PING (-3)
        # must not be written here — they'd render as out-of-range colormap indices.
        visible_mask = observation >= 0
        self.known_map[visible_mask] = observation[visible_mask]

    def get_copy(self) -> np.ndarray:
        return self.known_map.copy()


@dataclass
class Mission:
    tile_pos: tuple[int, int]
    steps_remaining: int

    def decrement(self):
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

class MissionManager:
    def __init__(self, missions: list[Mission]):
        self.missions: dict[tuple[int, int], Mission] = {
            mission.tile_pos: mission for mission in missions
        }

    def update(self, human_pos: tuple[int, int]):
        curr_mission_done = False
        mission = self.missions.get(human_pos)
        if mission is None:
            return False
        mission.decrement()
        if mission.steps_remaining <= 0:
            self.missions.pop(human_pos, None)
            return True
        return False
    
    def active_missions(self) -> list[Mission]:
        return list(self.missions.values())
    
    def exit_unlocked(self) -> bool:
        return len(self.missions) == 0


class GenericMapSimulation:
    """Generic map simulation supporting heterogeneous agent implementations.

    When enable_mechanics=True (default), adds the full game mechanics that
    rule-based agents rely on: topology-aware radar, probabilistic noise/sound
    generation, and directional cone observation for action-based agents.
    Random agents that ignore these signals work with enable_mechanics=False.
    """

    WORLD_COLORS = [
        "#1a1a2e",  # WALL
        "#2e2e4a",  # FLOOR
        "#9b59b6",  # VENT
        "#27ae60",  # HIDE
        "#2980b9",  # PLAYER_START
        "#c0392b",  # ALIEN_START
        "#f39c12",  # EXIT
        "#1abc9c",  # MISSION
    ]
    KNOWLEDGE_COLORS = [
        "#1a1a2e",  # WALL
        "#2e2e4a",  # FLOOR
        "#9b59b6",  # VENT
        "#27ae60",  # HIDE
        "#2980b9",  # PLAYER_START
        "#c0392b",  # ALIEN_START
        "#f39c12",  # EXIT
        "#1abc9c",  # MISSION
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
        enable_mechanics: bool = True,
        p_noise: float = 0.1,
        noise_radius: int = 2,
        radar_interval: int = 5,
        seed: int = 0,
        mission_steps: int | None = 3,
        mission_manager: MissionManager | None = None,
    ):
        self.grid = grid
        self.agents = list(agents)
        self.knowledge_mode = knowledge_mode
        self.default_human_view = default_human_view
        self.default_alien_view = default_alien_view
        self.enable_mechanics = enable_mechanics
        self.p_noise = p_noise
        self.noise_radius = noise_radius
        self.radar_interval = radar_interval
        self._rng = np.random.default_rng(seed)  # seeded RNG — no global state pollution

        # Radar state (used when enable_mechanics=True)
        self.steps_since_radar = 0
        self.radar_active_for = 0
        self.last_radar_threat: str | None = None
        self.last_radar_dist: int | None = None
        self.last_noise_ripple: tuple[int, int] | None = None
        self.noise_ripple_age = 0

        self.mission_manager = mission_manager
        if self.mission_manager is None and mission_steps is not None and mission_steps > 0:
            mission_positions = self._mission_positions()
            if mission_positions:
                self.mission_manager = MissionManager(
                    [Mission(position, mission_steps) for position in mission_positions]
                )
        if self.mission_manager is not None:
            for spec in self.agents:
                if spec.role == "human":
                    setattr(spec.agent, "mission_manager", self.mission_manager)

        if self.knowledge_mode == "on":
            self.knowledge: Dict[str, GenericKnowledge] = {
                spec.label: GenericKnowledge(self.grid.shape) for spec in self.agents
            }
        else:
            self.knowledge = {}

    # ── Position helpers ──────────────────────────────────────────────────────

    @classmethod
    def _get_position(cls, spec: AgentSpec) -> tuple[int, int]:
        raw = getattr(spec.agent, "pos")
        return (int(raw[0]), int(raw[1]))

    @classmethod
    def _set_position(cls, spec: AgentSpec, position_yx: tuple[int, int]):
        setattr(spec.agent, "pos", (int(position_yx[0]), int(position_yx[1])))

    # ── Observation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _in_bounds(grid: np.ndarray, position: tuple[int, int]) -> bool:
        y, x = position
        return 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]

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
        return getattr(spec.agent, "view_length", self.default_human_view)

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

    # ── Game mechanics (radar, noise, cone observation) ───────────────────────

    def _topology_distance(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        if start == goal:
            return 0
        frontier = deque([(start, 0)])
        visited = {start}
        while frontier:
            (y, x), dist = frontier.popleft()
            for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                ny, nx = y + dy, x + dx
                nxt = (ny, nx)
                if not self._in_bounds(self.grid, nxt) or nxt in visited:
                    continue
                if self.grid[ny, nx] == int(Tile.WALL):
                    continue
                if nxt == goal:
                    return dist + 1
                visited.add(nxt)
                frontier.append((nxt, dist + 1))
        sy, sx = start
        gy, gx = goal
        return abs(sy - gy) + abs(sx - gx)

    def _update_radar(self, human_pos_yx: tuple[int, int], alien_pos_yx: tuple[int, int]):
        self.steps_since_radar += 1
        if self.steps_since_radar >= self.radar_interval:
            self.steps_since_radar = 0
            dist = self._topology_distance(human_pos_yx, alien_pos_yx)
            for threat_level, (min_d, max_d) in RADAR_BANDS.items():
                if min_d <= dist <= max_d:
                    self.last_radar_threat = threat_level
                    self.last_radar_dist = dist
                    self.radar_active_for = 2
                    break
        else:
            if self.radar_active_for > 0:
                self.radar_active_for -= 1
            else:
                self.last_radar_threat = None
                self.last_radar_dist = None

    def _cone_observation(
        self,
        position: tuple[int, int],
        direction: Direction,
        view_length: int,
    ) -> np.ndarray:
        obs = np.full(self.grid.shape, UNKNOWN_TILE, dtype=np.int16)
        for vy, vx in cone_fov(self.grid, position, direction, view_length):
            obs[vy, vx] = int(self.grid[vy, vx])
        if self.last_radar_threat is not None:
            hy, hx = position
            obs[hy, hx] = RADAR_PING_TILE
        return obs

    def _generate_noise(
        self,
        human_pos_yx: tuple[int, int],
        human_hidden: bool,
    ) -> tuple[int, int]:
        hy, hx = human_pos_yx
        heard_yx = (hy, hx)
        if not human_hidden and self._rng.random() < self.p_noise:
            off_y = int(self._rng.integers(-self.noise_radius, self.noise_radius + 1))
            off_x = int(self._rng.integers(-self.noise_radius, self.noise_radius + 1))
            ny = max(0, min(hy + off_y, self.grid.shape[0] - 1))
            nx = max(0, min(hx + off_x, self.grid.shape[1] - 1))
            heard_yx = (ny, nx)
            self.last_noise_ripple = (hy, hx)
            self.noise_ripple_age = 0
        if self.last_noise_ripple is not None:
            self.noise_ripple_age += 1
            if self.noise_ripple_age > 2:
                self.last_noise_ripple = None
        return heard_yx

    # ── Agent helpers ─────────────────────────────────────────────────────────

    def _update_knowledge(
        self,
        label: str,
        position: tuple[int, int],
        radius: int,
        direction: Direction | None = None,
    ):
        if self.knowledge_mode != "on":
            return
        if self.enable_mechanics and direction is not None:
            observation = self._cone_observation(position, direction, radius)
        else:
            observation = self._partial_observation(position, radius)
        self.knowledge[label].update_from_observation(observation)

    def _is_hidden_position(self, position: tuple[int, int]) -> bool:
        return self._in_bounds(self.grid, position) and self.grid[position] == int(Tile.HIDE)

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def _snapshot_agents(self) -> list[AgentFrame]:
        snapshot: list[AgentFrame] = []
        for spec in self.agents:
            position = self._get_position(spec)
            hidden = bool(getattr(spec.agent, "hidden", False)) or (
                spec.role == "human" and self._is_hidden_position(position)
            )
            knowledge_map = self.knowledge[spec.label].get_copy() if self.knowledge_mode == "on" else None

            if spec.role == "alien":
                state_attr = getattr(spec.agent, "state", None)
                mode = state_attr.name if state_attr is not None else ""
            elif spec.role == "human":
                if hidden:
                    mode = "HIDING"
                elif getattr(spec.agent, "_known_exit", None) is not None:
                    mode = "→ EXIT"
                else:
                    mode = "EXPLORING"
            else:
                mode = ""

            # Compute cone FOV for this agent
            direction = getattr(spec.agent, "direction", None)
            view_len = getattr(spec.agent, "view_length", self.default_human_view)
            if direction is not None and self.enable_mechanics:
                fov: frozenset = frozenset(cone_fov(self.grid, position, direction, view_len))
            else:
                fov = frozenset()

            # Find the nearest visible opponent (in FOV, not hiding)
            visible_opponent = None
            for other in self.agents:
                if other.role != spec.role:
                    opp_pos = self._get_position(other)
                    opp_hidden = (bool(getattr(other.agent, "hidden", False))
                                  or self._is_hidden_position(opp_pos))
                    if opp_pos in fov and not opp_hidden:
                        visible_opponent = opp_pos
                        break

            snapshot.append(AgentFrame(label=spec.label, role=spec.role, position=position,
                                       hidden=hidden, knowledge_map=knowledge_map, mode=mode,
                                       fov=fov, visible_opponent=visible_opponent))
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

    def _mission_positions(self) -> list[tuple[int, int]]:
        matches = np.argwhere(self.grid == int(Tile.MISSION))
        return [(int(y), int(x)) for y, x in matches]

    def _has_collision(self, agents: list[AgentFrame]) -> bool:
        humans = [a.position for a in agents if a.role == "human"]
        aliens = [a.position for a in agents if a.role == "alien"]
        return any(h == al for h in humans for al in aliens)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, max_steps: int = 200) -> tuple[list[SimulationFrame], str]:
        frames: list[SimulationFrame] = []
        exit_pos = self._exit_position()

        for step in range(max_steps + 1):
            current_agents = self._snapshot_agents()
            outcome = None
            exit_unlocked = self.mission_manager.exit_unlocked() if self.mission_manager else True
            if exit_pos is not None:
                if exit_unlocked:
                    for agent in current_agents:
                        if agent.role == "human" and agent.position == exit_pos:
                            outcome = f"human_reached_exit:{agent.label}"
                            break
            if outcome is None and self._has_collision(current_agents):
                outcome = "alien_caught_human"

            alien_heard = None
            if self.enable_mechanics:
                for spec in self.agents:
                    if spec.role == "alien":
                        alien_heard = getattr(spec.agent, "last_heard_pos", None)
                        break

            frames.append(SimulationFrame(
                step=step,
                outcome=outcome,
                agents=current_agents,
                exit_unlocked=exit_unlocked,
                noise_ripple_pos=self.last_noise_ripple if self.enable_mechanics else None,
                radar_threat=self.last_radar_threat if self.enable_mechanics else None,
                alien_heard_pos=alien_heard,
            ))
            if outcome is not None or step == max_steps:
                return frames, outcome or "max_steps_reached"

            # Update radar before agents act so humans can react to current threat level
            if self.enable_mechanics:
                human_specs = [s for s in self.agents if s.role == "human"]
                alien_specs = [s for s in self.agents if s.role == "alien"]
                if human_specs and alien_specs:
                    self._update_radar(
                        self._get_position(human_specs[0]),
                        self._get_position(alien_specs[0]),
                    )

            for spec in self.agents:
                current_position = self._get_position(spec)
                # Aliens always receive the actual human position — hiding is handled internally
                # by AlienAgent.step() via player_hiding detection. Excluding the hidden human
                # here would cause nearest_target to fall back to the alien's own position,
                # making the alien think it sees itself as the player.
                if spec.role == "alien":
                    opposing_positions = [
                        self._get_position(other)
                        for other in self.agents
                        if other.label != spec.label and other.role != spec.role
                    ]
                else:
                    opposing_positions = [
                        self._get_position(other)
                        for other in self.agents
                        if other.label != spec.label
                        and other.role != spec.role
                        and not (other.role == "human" and self._is_hidden_position(self._get_position(other)))
                    ]
                nearest_target = self._nearest_target(current_position, opposing_positions) or current_position
                radius = self._view_radius_for(spec)
                direction = getattr(spec.agent, "direction", None)
                self._update_knowledge(spec.label, current_position, radius, direction)

                # Build observation (cone for agents with a direction, circular otherwise)
                if self.enable_mechanics and direction is not None:
                    obs = self._cone_observation(current_position, direction, radius)
                else:
                    obs = self._partial_observation(current_position, radius)

                # Pass observation to agents that accept it (observe() is a no-op for random agents)
                if hasattr(spec.agent, "observe"):
                    if spec.role == "human":
                        spec.agent.observe(obs, self.last_radar_threat, self.last_radar_dist)
                    else:
                        spec.agent.observe(obs)

                # Compute heard_pos (y, x) for alien step calls
                if self.enable_mechanics and spec.role == "alien":
                    human_specs = [s for s in self.agents if s.role == "human"]
                    if human_specs:
                        h_pos_yx = self._get_position(human_specs[0])
                        h_hidden = bool(getattr(human_specs[0].agent, "hidden", False))
                        heard_yx = self._generate_noise(h_pos_yx, h_hidden)
                    else:
                        heard_yx = nearest_target
                else:
                    heard_yx = nearest_target

                # Uniform step call — all agents return new (y, x) position
                new_pos = spec.agent.step(nearest_target, heard_yx, step)
                self._set_position(spec, (int(new_pos[0]), int(new_pos[1])))

                if spec.role == "human":
                    setattr(spec.agent, "hidden", self._is_hidden_position(self._get_position(spec)))

        return frames, "max_steps_reached"

    # ── Render ────────────────────────────────────────────────────────────────

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
        unknown_value = 8

        has_single_pair = len(frames[0].agents) == 2 and {a.role for a in frames[0].agents} == {"human", "alien"}

        if world_only or self.knowledge_mode == "off":
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
            fig.patch.set_facecolor("#000000")
            ax.imshow(self.grid, cmap=world_cmap, vmin=0, vmax=7)
            ax.autoscale(False)  # prevent large Circle patches from shrinking the map
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
                        ax.scatter([x], [y], s=210, facecolors="none", edgecolors="#7CFF6B",
                                   linewidths=2.0, marker="o", zorder=4)
                    )

            exit_pos = self._exit_position()
            exit_marker = None
            if exit_pos is not None:
                ey, ex = exit_pos
                exit_marker = ax.scatter([ex], [ey], s=100, c="#f39c12", marker="*", zorder=6)

            # Radar threat rings — one concentric outline ring per threat band,
            # always faintly visible; the active band is highlighted.
            # Radii match the outer edge of each RADAR_BANDS distance range.
            _THREAT_RING_CFG = [
                (7,  "#FF3333", "CRITICAL"),
                (12, "#FF8800", "CLOSE"),
                (18, "#FFEE00", "NEAR"),
                (25, "#44FF88", "FAR"),
            ]
            h0y, h0x = next((a.position for a in frames[0].agents if a.role == "human"), (0, 0))
            threat_rings = []
            for radius, color, _ in _THREAT_RING_CFG:
                ring = plt.Circle((h0x, h0y), radius, fill=False, edgecolor=color,
                                  linewidth=0.8, linestyle="--", alpha=0.08, zorder=3)
                ax.add_patch(ring)
                threat_rings.append(ring)
            _threat_order = {"CRITICAL": 0, "CLOSE": 1, "NEAR": 2, "FAR": 3}

            # Noise ripple — 3 concentric dashed yellow rings
            ripple_circles = [
                plt.Circle((h0x, h0y), 0.35 * (i + 1), fill=False, edgecolor="yellow",
                            linewidth=max(0.5, 1.5 - i * 0.4), linestyle="--",
                            alpha=[0.7, 0.4, 0.2][i], zorder=3)
                for i in range(3)
            ]
            for c in ripple_circles:
                c.set_visible(False)
                ax.add_patch(c)
            # Alien heard position marker
            alien_heard_marker = ax.scatter([h0x], [h0y], s=70, c="#FF4444", marker="*", zorder=6, alpha=0.8)
            alien_heard_marker.set_visible(False)

            status_text = fig.suptitle("", color="white", fontsize=11, x=0.02, ha="left", fontfamily="monospace")

            def update_world(frame_index: int):
                state = frames[frame_index]
                human_ring_idx = 0
                for artist, agent in zip(marker_artists, state.agents):
                    y, x = agent.position
                    artist.set_offsets([[x, y]])
                    artist.set_alpha(0.55 if agent.hidden else 1.0)
                    if agent.role == "human":
                        hidden_rings[human_ring_idx].set_offsets([[x, y]])
                        hidden_rings[human_ring_idx].set_visible(agent.hidden)
                        human_ring_idx += 1

                # Radar threat rings — recentre on current human pos, highlight active band
                humans = [a for a in state.agents if a.role == "human"]
                active_idx = _threat_order.get(state.radar_threat, -1)
                if humans:
                    hy, hx = humans[0].position
                    for i, ring in enumerate(threat_rings):
                        ring.set_center((hx, hy))
                        if i == active_idx:
                            ring.set_alpha(0.5)
                            ring.set_linewidth(2.0)
                        else:
                            ring.set_alpha(0.08)
                            ring.set_linewidth(0.8)

                # Noise ripple
                if state.noise_ripple_pos is not None:
                    ny, nx = state.noise_ripple_pos
                    for c in ripple_circles:
                        c.set_center((nx, ny))
                        c.set_visible(True)
                else:
                    for c in ripple_circles:
                        c.set_visible(False)

                # Alien heard position
                if state.alien_heard_pos is not None:
                    ahy, ahx = state.alien_heard_pos
                    alien_heard_marker.set_offsets([[ahx, ahy]])
                    alien_heard_marker.set_visible(True)
                else:
                    alien_heard_marker.set_visible(False)

                if exit_marker is not None:
                    exit_color = "#f39c12" if state.exit_unlocked else "#7f8c8d"
                    exit_alpha = 1.0 if state.exit_unlocked else 0.55
                    exit_marker.set_color(exit_color)
                    exit_marker.set_alpha(exit_alpha)

                human_mode = next((a.mode for a in state.agents if a.role == "human"), "")
                alien_mode = next((a.mode for a in state.agents if a.role == "alien"), "")
                exit_state = "UNLOCKED" if state.exit_unlocked else "LOCKED"
                status_text.set_text(
                    f"Step {state.step:4d}/{len(frames)-1:4d}"
                    f"  |  Human: {human_mode:<10}"
                    f"  Alien: {alien_mode:<11}"
                    f"  Exit: {exit_state:<9}"
                    f"  |  {outcome}"
                )
                artists = [*marker_artists, *hidden_rings, *threat_rings, *ripple_circles,
                           alien_heard_marker, status_text]
                if exit_marker is not None:
                    artists.append(exit_marker)
                return artists

            animation = FuncAnimation(fig, update_world, frames=len(frames),
                                      interval=max(1, int(1000 / max(1, fps))), blit=False, repeat=False)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            animation.save(output_path, writer=PillowWriter(fps=max(1, fps)))
            print(f"Saved animation -> {output_path}")
            if show_window:
                self._maybe_show(fig)
            else:
                plt.close(fig)
            return

        # Full mode: world + one knowledge panel per agent
        n_panels = 1 + len(frames[0].agents)
        fig_size = (16, 5) if has_single_pair else (max(16, 5 * n_panels), 5)
        fig, axes = plt.subplots(1, n_panels, figsize=fig_size, dpi=120)
        fig.patch.set_facecolor("#000000")
        if n_panels == 1:
            axes = [axes]

        world_ax = axes[0]
        world_ax.imshow(self.grid, cmap=world_cmap, vmin=0, vmax=7)
        world_ax.autoscale(False)  # prevent large Circle patches from shrinking the map
        world_ax.set_title("World", color="white", fontsize=12, fontweight="bold")
        world_ax.set_xticks([])
        world_ax.set_yticks([])
        world_ax.set_facecolor("#0f0f1e")

        exit_pos = self._exit_position()
        exit_marker = None
        if exit_pos is not None:
            ey, ex = exit_pos
            exit_marker = world_ax.scatter([ex], [ey], s=100, c="#f39c12", marker="*", zorder=6)

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
                    world_ax.scatter([x], [y], s=210, facecolors="none", edgecolors="#7CFF6B",
                                     linewidths=2.0, marker="o", zorder=4)
                )

        knowledge_axes = axes[1:]
        knowledge_images = []
        knowledge_markers = []
        fov_overlays = []     # semi-transparent FOV highlight per panel
        opp_markers = []      # visible-opponent marker per panel
        H_grid, W_grid = self.grid.shape
        for axis, agent in zip(knowledge_axes, frames[0].agents):
            axis.set_title(f"{agent.label} ({agent.role})", color="white", fontsize=11, fontweight="bold")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#0f0f1e")
            if agent.knowledge_map is None:
                initial = np.full(self.grid.shape, unknown_value, dtype=np.int16)
            else:
                initial = np.where(agent.knowledge_map < 0, unknown_value, agent.knowledge_map)
            image = axis.imshow(initial, cmap=knowledge_cmap, vmin=0, vmax=8)
            knowledge_images.append(image)
            y, x = agent.position
            marker_style = "X" if agent.role == "alien" else "o"
            marker_color = "#FF4D6D" if agent.role == "alien" else "#00D4FF"
            knowledge_markers.append(
                axis.scatter([x], [y], s=95, c=marker_color, edgecolors="white", linewidths=0.7,
                             marker=marker_style, zorder=5)
            )
            # FOV overlay: RGBA image, cells in FOV get a faint tint
            fov_rgba = np.zeros((H_grid, W_grid, 4), dtype=np.float32)
            fov_img = axis.imshow(fov_rgba, zorder=3)
            fov_overlays.append((fov_img, agent.role))
            # Visible-opponent marker (opponent's style, shown when in FOV)
            opp_style = "o" if agent.role == "alien" else "X"
            opp_color = "#00D4FF" if agent.role == "alien" else "#FF4D6D"
            opp_m = axis.scatter([x], [y], s=95, c=opp_color, edgecolors="white",
                                 linewidths=1.2, marker=opp_style, zorder=6)
            opp_m.set_visible(False)
            opp_markers.append(opp_m)

        # Radar threat rings on world panel
        _THREAT_RING_CFG = [
            (7,  "#FF3333", "CRITICAL"),
            (12, "#FF8800", "CLOSE"),
            (18, "#FFEE00", "NEAR"),
            (25, "#44FF88", "FAR"),
        ]
        h0y, h0x = next((a.position for a in frames[0].agents if a.role == "human"), (0, 0))
        threat_rings = []
        for radius, color, _ in _THREAT_RING_CFG:
            ring = plt.Circle((h0x, h0y), radius, fill=False, edgecolor=color,
                              linewidth=0.8, linestyle="--", alpha=0.08, zorder=3)
            world_ax.add_patch(ring)
            threat_rings.append(ring)
        _threat_order = {"CRITICAL": 0, "CLOSE": 1, "NEAR": 2, "FAR": 3}
        # Noise ripple rings on world panel
        ripple_circles = [
            plt.Circle((h0x, h0y), 0.35 * (i + 1), fill=False, edgecolor="yellow",
                        linewidth=max(0.5, 1.5 - i * 0.4), linestyle="--",
                        alpha=[0.7, 0.4, 0.2][i], zorder=3)
            for i in range(3)
        ]
        for c in ripple_circles:
            c.set_visible(False)
            world_ax.add_patch(c)
        # Alien heard position marker — shown on alien knowledge panel
        alien_knowledge_axes = [
            ax for ax, agent in zip(knowledge_axes, frames[0].agents) if agent.role == "alien"
        ]
        alien_heard_markers = []
        for alien_ax in alien_knowledge_axes:
            m = alien_ax.scatter([h0x], [h0y], s=70, c="#FF4444", marker="*", zorder=6, alpha=0.85)
            m.set_visible(False)
            alien_heard_markers.append(m)

        status_text = fig.suptitle("", color="white", fontsize=11, x=0.02, ha="left", fontfamily="monospace")

        def update_full(frame_index: int):
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
            for image, marker, (fov_img, role), opp_m, agent in zip(
                knowledge_images, knowledge_markers, fov_overlays, opp_markers, state.agents
            ):
                if agent.knowledge_map is None:
                    data = np.full(self.grid.shape, unknown_value, dtype=np.int16)
                else:
                    data = np.where(agent.knowledge_map < 0, unknown_value, agent.knowledge_map)
                image.set_data(data)
                y, x = agent.position
                marker.set_offsets([[x, y]])

                # FOV overlay — faint tint on currently visible cells
                fov_rgba = np.zeros((H_grid, W_grid, 4), dtype=np.float32)
                tint = (1.0, 0.3, 0.4, 0.18) if role == "alien" else (0.0, 0.83, 1.0, 0.18)
                for vy, vx in agent.fov:
                    fov_rgba[vy, vx] = tint
                fov_img.set_data(fov_rgba)

                # Visible-opponent marker
                if agent.visible_opponent is not None:
                    oy, ox = agent.visible_opponent
                    opp_m.set_offsets([[ox, oy]])
                    opp_m.set_visible(True)
                else:
                    opp_m.set_visible(False)

            # Radar threat rings
            humans = [a for a in state.agents if a.role == "human"]
            active_idx = _threat_order.get(state.radar_threat, -1)
            if humans:
                hy, hx = humans[0].position
                for i, ring in enumerate(threat_rings):
                    ring.set_center((hx, hy))
                    if i == active_idx:
                        ring.set_alpha(0.5)
                        ring.set_linewidth(2.0)
                    else:
                        ring.set_alpha(0.08)
                        ring.set_linewidth(0.8)

            # Noise ripple
            if state.noise_ripple_pos is not None:
                ny, nx = state.noise_ripple_pos
                for c in ripple_circles:
                    c.set_center((nx, ny))
                    c.set_visible(True)
            else:
                for c in ripple_circles:
                    c.set_visible(False)

            # Alien heard position on alien knowledge panels
            for m in alien_heard_markers:
                if state.alien_heard_pos is not None:
                    ahy, ahx = state.alien_heard_pos
                    m.set_offsets([[ahx, ahy]])
                    m.set_visible(True)
                else:
                    m.set_visible(False)

            if exit_marker is not None:
                exit_color = "#f39c12" if state.exit_unlocked else "#7f8c8d"
                exit_alpha = 1.0 if state.exit_unlocked else 0.55
                exit_marker.set_color(exit_color)
                exit_marker.set_alpha(exit_alpha)

            human_mode = next((a.mode for a in state.agents if a.role == "human"), "")
            alien_mode = next((a.mode for a in state.agents if a.role == "alien"), "")
            exit_state = "UNLOCKED" if state.exit_unlocked else "LOCKED"
            status_text.set_text(
                f"Step {state.step:4d}/{len(frames)-1:4d}"
                f"  |  Human: {human_mode:<10}"
                f"  Alien: {alien_mode:<11}"
                f"  Exit: {exit_state:<9}"
                f"  |  {outcome}"
            )
            fov_imgs = [img for img, _ in fov_overlays]
            artists = [*world_markers, *hidden_rings, *threat_rings, *ripple_circles,
                       *alien_heard_markers, *knowledge_images, *knowledge_markers,
                       *fov_imgs, *opp_markers, status_text]
            if exit_marker is not None:
                artists.append(exit_marker)
            return artists

        animation = FuncAnimation(fig, update_full, frames=len(frames),
                                  interval=max(1, int(1000 / max(1, fps))), blit=False, repeat=False)
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
