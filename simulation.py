from __future__ import annotations

import os
import tempfile
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Optional, Sequence
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle

from agents.base import Direction, cone_fov, TeamRole
from agents.alien import AlienState
from agents.coord_bus import CoordMessage, CoordType
from agents.role_manager import (
    assign_worker_greedy,
    assign_decoy_farthest,
    assign_runner_greedy,
    clear_roles,
)
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
    escaped: bool = False
    team_role: Optional[TeamRole] = None
    knowledge_map: Optional[np.ndarray] = None
    mode: str = ""  # human: EXPLORING/FLEEING/HIDING; alien: SEARCH/INVESTIGATE/HUNT
    fov: frozenset = field(default_factory=frozenset)         # (y,x) cells visible this step
    visible_opponent: Optional[tuple[int, int]] = None        # opponent pos if inside FOV and not hidden
    agent_id: Optional[int] = None


@dataclass
class SimulationFrame:
    step: int
    outcome: str | None
    agents: list[AgentFrame]
    grid_snapshot: np.ndarray | None = None
    exit_unlocked: bool = True
    noise_ripple_pos: tuple[int, int] | None = None
    noise_deliberate: bool | None = None
    radar_threat: str | None = None
    alien_heard_pos: tuple[int, int] | None = None  # (y, x) where alien heard a sound
    shared_missions: list[tuple[int, int]] | None = None
    shared_exit: tuple[int, int] | None = None
    captured_humans: list[str] | None = None
    escaped_humans: list[str] | None = None
    mission_done_events: list[tuple[tuple[int, int], int]] | None = None
    agent_roles: dict[int, str] | None = None


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

        # Role / mission bookkeeping for dynamic reassignment
        self.known_missions: set[tuple[int, int]] = set()
        self.pending_missions: set[tuple[int, int]] = set()
        # periodic reassignment removed — use event-driven reassignment only
        self.reassign_interval: int | None = None
        # Role-based configuration toggle: when False, the simulation will not
        # perform automatic role reassignment; roles may still be set manually
        # via `set_initial_roles` or `allocate_roles_by_counts`.
        self.role_based: bool = True
        # Mission tile values to auto-detect (set of integer tile ids).
        # If empty, automatic mission detection is disabled. Example: {7}
        self.mission_tile_values: set[int] = set()

        # Mission counts for agent UI/logic
        self.total_missions: int = 0

        # Radar state (used when enable_mechanics=True)
        self.steps_since_radar = 0
        self.radar_active_for = 0
        self.last_radar_threat: str | None = None
        self.last_radar_dist: int | None = None
        self.last_noise_ripple: tuple[int, int] | None = None
        self.noise_ripple_age = 0
        # Whether the last_noise_ripple was a deliberate loud noise
        # (set by an agent) or a stochastic ambient noise event.
        self.last_noise_deliberate: bool = False

        # Track captured humans for role reassignment filtering
        self.captured_humans: set[str] = set()
        self.escaped_humans: set[str] = set()

        # Shared-coordinates bus (role-aware agents only)
        self.shared_mission_coords: set[tuple[int, int]] = set()
        self.shared_exit_coord: tuple[int, int] | None = None
        self.completed_mission_events: list[tuple[tuple[int, int], int]] = []
        self._active_frame: SimulationFrame | None = None
        self._last_role_by_agent_id: dict[int, str] = {}
        # Enforce a minimum dwell time on mission tiles. Default minimum is 20 steps.
        self.mission_steps_required: int = max(20, int(mission_steps) if mission_steps is not None else 20)
        self._mission_dwell_progress: dict[tuple[int, int], int] = {}

        # Debug log for role/mission reassignment decisions.
        self.debug_log_path = Path(tempfile.gettempdir()) / f"aasma_role_debug_{os.getpid()}_{uuid4().hex}.log"
        self.debug_log_path.write_text("", encoding="utf-8")
        self._debug_log(f"init: role_based={self.role_based}, knowledge_mode={self.knowledge_mode}, debug_log={self.debug_log_path}")
        self._debug_log(f"mission_steps_required={self.mission_steps_required}")

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

        # Assign stable agent_id values if role-aware agents expose the field.
        # Treat negative values as the unset sentinel (agents may default to -1).
        human_idx = 0
        for spec in self.agents:
            if spec.role == "human" and hasattr(spec.agent, "agent_id"):
                if getattr(spec.agent, "agent_id", -1) < 0:
                    setattr(spec.agent, "agent_id", human_idx)
                human_idx += 1
        self._debug_log(
            "agent_ids: " + ", ".join(
                f"{spec.label}=id{getattr(spec.agent, 'agent_id', '?')}:role={getattr(spec.agent, 'team_role', None)}"
                for spec in self.agents if spec.role == "human"
            )
        )

    def _debug_log(self, message: str) -> None:
        try:
            with self.debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(message.rstrip() + "\n")
        except Exception:
            pass

    @staticmethod
    def _role_name(role: TeamRole | None) -> str:
        if role is None:
            return "NONE"
        try:
            return role.name
        except Exception:
            return str(role)

    def _track_role_changes(self, agents: list[AgentFrame], step_tag: str) -> dict[int, str]:
        current_roles: dict[int, str] = {}
        for agent in agents:
            if agent.role != "human" or agent.agent_id is None:
                continue
            current_role = self._role_name(agent.team_role)
            current_roles[int(agent.agent_id)] = current_role
            previous_role = self._last_role_by_agent_id.get(int(agent.agent_id))
            if previous_role != current_role:
                self._debug_log(
                    f"role_change: step={step_tag}, id={int(agent.agent_id)}, label={agent.label}, {previous_role or 'NONE'} -> {current_role}"
                )
        self._last_role_by_agent_id = current_roles.copy()
        return current_roles

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
        elif self.radar_active_for > 0:
            self.radar_active_for -= 1
        else:
            self.last_radar_threat = None
            self.last_radar_dist = None

    def _radar_threat_for_distance(self, dist: int) -> str | None:
        for threat_level, (min_d, max_d) in RADAR_BANDS.items():
            if min_d <= dist <= max_d:
                return threat_level
        return None

    def _cone_observation(
        self,
        position: tuple[int, int],
        direction: Direction,
        view_length: int,
        radar_threat: str | None = None,
    ) -> np.ndarray:
        obs = np.full(self.grid.shape, UNKNOWN_TILE, dtype=np.int16)
        for vy, vx in cone_fov(self.grid, position, direction, view_length):
            obs[vy, vx] = int(self.grid[vy, vx])
        if radar_threat is not None:
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
            if self.noise_ripple_age > 9:
                self.last_noise_ripple = None
                self.last_noise_deliberate = False
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
            observation = self._cone_observation(position, direction, radius, None)
        else:
            observation = self._partial_observation(position, radius)
        self.knowledge[label].update_from_observation(observation)

    # ── Mission / role reassignment helpers ─────────────────────────────────

    def add_mission(self, position: tuple[int, int]) -> None:
        """Register a newly discovered mission and trigger immediate reassignment."""
        if position not in self.known_missions:
            self.known_missions.add(position)
            self.pending_missions.add(position)
            self._debug_log(
                f"add_mission: pos={position}, known={sorted(self.known_missions)}, pending={sorted(self.pending_missions)}"
            )
            self._maybe_reassign_roles(new_mission=True)

    def complete_mission(self, position: tuple[int, int]) -> None:
        """Mark a mission complete and trigger reassignment."""
        self.known_missions.discard(position)
        self.pending_missions.discard(position)
        self._debug_log(
            f"complete_mission: pos={position}, known={sorted(self.known_missions)}, pending={sorted(self.pending_missions)}"
        )
        self._maybe_reassign_roles(mission_completed=True)

    def _locked_worker_labels(self) -> set[str]:
        """Return labels for WORKERs that should not be reassigned.

        A worker is considered locked if: it currently has `team_role==WORKER` and
        either is located on a mission tile or exposes `mission_progress > 0`.
        """
        locked: set[str] = set()
        for s in self.agents:
            if s.role != "human":
                continue
            if getattr(s, "label", "") in self.captured_humans:
                continue
            agent = getattr(s, "agent", None)
            if agent is None:
                continue
            if getattr(agent, "team_role", None) == TeamRole.WORKER:
                # Check mission progress attribute if present
                progress = getattr(agent, "mission_progress", 0)
                pos = self._get_position(s)
                on_mission_tile = pos in self.known_missions
                if progress > 0 or on_mission_tile:
                    locked.add(getattr(s, "label", ""))
        return locked

    def _maybe_reassign_roles(self, new_mission: bool = False, mission_completed: bool = False, worker_died: bool = False, exit_opened: bool = False) -> None:
        """Reassign roles according to policy.

        Only reassign among free agents (not locked). If no free agents, keep roles and queue missions.
        This function is event-driven: it should be called on mission discovery/completion
        or other major events. Periodic reassignment has been disabled.
        """
        # Respect role-based toggle
        if not self.role_based:
            self._debug_log(
                f"reassign_skip: role_based=False trigger={{new_mission={new_mission}, mission_completed={mission_completed}, worker_died={worker_died}, exit_opened={exit_opened}}}"
            )
            return

        # If there are no pending missions and no explicit trigger, skip
        if not self.pending_missions and not (new_mission or mission_completed or worker_died or exit_opened):
            self._debug_log(
                f"reassign_skip: no_pending_and_no_trigger trigger={{new_mission={new_mission}, mission_completed={mission_completed}, worker_died={worker_died}, exit_opened={exit_opened}}}"
            )
            return

        locked = self._locked_worker_labels()
        human_specs = [
            s for s in self.agents
            if s.role == "human" and getattr(s, "label", "") not in self.captured_humans
        ]
        assignable_specs = [s for s in human_specs if getattr(s, "label", "") not in locked]
        current_workers = [s for s in human_specs if getattr(s.agent, "team_role", None) == TeamRole.WORKER]
        current_decoys = [s for s in human_specs if getattr(s.agent, "team_role", None) == TeamRole.DECOY]
        current_runners = [s for s in human_specs if getattr(s.agent, "team_role", None) == TeamRole.RUNNER]

        self._debug_log(
            "reassign_start: "
            f"trigger={{new_mission={new_mission}, mission_completed={mission_completed}, worker_died={worker_died}, exit_opened={exit_opened}}}, "
            f"known={sorted(self.known_missions)}, pending={sorted(self.pending_missions)}, "
            f"locked={sorted(locked)}, assignable={[s.label for s in assignable_specs]}, "
            f"workers={[s.label for s in current_workers]}, decoys={[s.label for s in current_decoys]}, runners={[s.label for s in current_runners]}, "
            f"roles_before={[(s.label, getattr(s.agent, 'team_role', None)) for s in human_specs]}"
        )

        missions = list(self.pending_missions) if self.pending_missions else list(self.known_missions)
        if not missions:
            self._debug_log("reassign_skip: no_missions_to_assign")
            return

        # Reassign all non-locked roles from scratch, but preserve locked workers.
        self._debug_log(
            f"clear_roles: before={[(s.label, getattr(s.agent, 'team_role', None)) for s in assignable_specs]}"
        )
        clear_roles(assignable_specs)
        self._debug_log(
            f"clear_roles: after={[(s.label, getattr(s.agent, 'team_role', None)) for s in assignable_specs]}"
        )

        target_worker_count = min(len(missions), len(human_specs))
        worker_label = None
        assigned_workers = len(locked)
        remaining_specs = list(assignable_specs)
        while assigned_workers < target_worker_count and remaining_specs:
            chosen_label = assign_worker_greedy(remaining_specs, missions)
            self._debug_log(f"assign_worker_greedy: chosen={chosen_label}")
            if chosen_label is None:
                break
            worker_label = chosen_label
            assigned_workers += 1
            promoted = next((s for s in remaining_specs if getattr(s, "label", None) == chosen_label), None)
            if promoted is not None:
                pos = getattr(promoted.agent, "pos", None)
                if pos is not None:
                    nearest = self._nearest_target(pos, missions)
                    if nearest is not None and nearest in self.pending_missions:
                        self.pending_missions.discard(nearest)
            remaining_specs = [s for s in remaining_specs if getattr(s, "label", None) != chosen_label]

        decoy_label = assign_decoy_farthest(remaining_specs, missions) if remaining_specs else None
        self._debug_log(f"assign_decoy_farthest: chosen={decoy_label}")
        remaining_specs = [s for s in remaining_specs if getattr(s, "label", None) != decoy_label]

        exit_pos = self._exit_position()
        runner_label = assign_runner_greedy(remaining_specs, exit_pos, None) if remaining_specs else None
        self._debug_log(f"assign_runner_greedy: chosen={runner_label}, exit_pos={exit_pos}")

        self._debug_log(
            f"reassign_done: roles_after={[(s.label, getattr(s.agent, 'team_role', None)) for s in human_specs]}"
        )

    # ── Role configuration API ──────────────────────────────────────────────
    def enable_role_based(self, enabled: bool) -> None:
        """Enable or disable automatic role reassignment."""
        self.role_based = bool(enabled)

    def set_initial_roles(self, role_map: dict[str, str | TeamRole]) -> None:
        """Set explicit roles for agents by label.

        `role_map` maps agent label -> role name (e.g. 'WORKER', 'DECOY', 'RUNNER')
        or a `TeamRole` value. Only human agents are affected.
        """
        for spec in self.agents:
            if spec.role != "human":
                continue
            label = getattr(spec, "label", None)
            if label in role_map:
                val = role_map[label]
                if isinstance(val, TeamRole):
                    setattr(spec.agent, "team_role", val)
                else:
                    try:
                        tr = TeamRole[val]
                        setattr(spec.agent, "team_role", tr)
                    except Exception:
                        # ignore invalid names
                        pass

    def allocate_roles_by_counts(self, counts: dict[str, int]) -> None:
        """Allocate roles among human agents by counts.

        `counts` keys can be 'WORKER', 'DECOY', 'RUNNER' (case-sensitive).
        Assignment is greedy: it assigns roles to available human agents in
        the order they appear in `self.agents` until counts are satisfied.
        This is a deterministic, user-controlled allocation useful for
        experiments where you fix the team composition.
        """
        # Clear previous roles for humans
        human_specs = [s for s in self.agents if s.role == "human"]
        for s in human_specs:
            try:
                setattr(s.agent, "team_role", TeamRole.NONE)
            except Exception:
                pass

        # Use greedy role_manager to pick best agents for each role count
        remaining = list(human_specs)
        missions = list(self.known_missions) if self.known_missions else []
        exit_pos = self._exit_position()

        # assign WORKERs greedily
        for _ in range(counts.get("WORKER", 0)):
            if not remaining:
                break
            chosen_label = assign_worker_greedy(remaining, missions)
            if not chosen_label:
                break
            # remove chosen from remaining
            remaining = [s for s in remaining if getattr(s, "label", None) != chosen_label]

        # assign DECOYs greedily
        for _ in range(counts.get("DECOY", 0)):
            if not remaining:
                break
            chosen_label = assign_decoy_farthest(remaining, missions)
            if not chosen_label:
                break
            remaining = [s for s in remaining if getattr(s, "label", None) != chosen_label]

        # assign RUNNERs greedily
        for _ in range(counts.get("RUNNER", 0)):
            if not remaining:
                break
            chosen_label = assign_runner_greedy(remaining, exit_pos, None)
            if not chosen_label:
                break
            remaining = [s for s in remaining if getattr(s, "label", None) != chosen_label]

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
            escaped = spec.label in self.escaped_humans
            if escaped:
                hidden = True
            knowledge_map = self.knowledge[spec.label].get_copy() if self.knowledge_mode == "on" else None

            if spec.role == "alien":
                state_attr = getattr(spec.agent, "state", None)
                mode = state_attr.name if state_attr is not None else ""
            elif spec.role == "human":
                if escaped:
                    mode = "ESCAPED"
                elif hidden:
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
                                       hidden=hidden, escaped=escaped, team_role=getattr(spec.agent, "team_role", None),
                                       knowledge_map=knowledge_map, mode=mode,
                                       fov=fov, visible_opponent=visible_opponent,
                                       agent_id=getattr(spec.agent, "agent_id", None)))
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

    def _missions_remaining(self) -> int:
        if not self.mission_tile_values:
            return 0
        count = 0
        for mv in self.mission_tile_values:
            count += int(np.count_nonzero(self.grid == int(mv)))
        return count

    def _update_exit_open_state(self) -> bool:
        exit_open = self._missions_remaining() == 0
        for spec in self.agents:
            if spec.role == "human":
                try:
                    setattr(spec.agent, "exit_open", exit_open)
                except Exception:
                    pass
        return exit_open

    def _complete_mission_at(self, position: tuple[int, int]) -> bool:
        if not self.mission_tile_values:
            return False
        if self.grid[position] not in {int(v) for v in self.mission_tile_values}:
            return False

        progress = self._mission_dwell_progress.get(position, 0) + 1
        self._mission_dwell_progress[position] = progress
        self._debug_log(
            f"mission_progress: pos={position}, progress={progress}/{self.mission_steps_required}"
        )
        if progress < self.mission_steps_required:
            return False

        completer_id = None
        for spec in self.agents:
            if spec.role != "human":
                continue
            if self._get_position(spec) == position:
                completer_id = getattr(spec.agent, "agent_id", None)
                break
        self._debug_log(f"mission_complete_start: pos={position}, completer_id={completer_id}")
        self.grid[position] = int(Tile.FLOOR)
        if self.knowledge_mode == "on":
            for km in self.knowledge.values():
                km.known_map[position] = int(Tile.FLOOR)
        for spec in self.agents:
            if spec.role != "human":
                continue
            agent = getattr(spec, "agent", None)
            if agent is not None and hasattr(agent, "remove_mission"):
                try:
                    agent.remove_mission(position)
                except Exception:
                    pass
            if agent is not None and hasattr(agent, "receive_coords"):
                try:
                    agent.receive_coords([
                        CoordMessage(CoordType.MISSION_DONE, position, sender_id=-1)
                    ])
                except Exception:
                    pass
        self.known_missions.discard(position)
        self.pending_missions.discard(position)
        self.shared_mission_coords.discard(position)
        self._mission_dwell_progress.pop(position, None)
        if completer_id is None:
            completer_id = -1
        event = (position, int(completer_id))
        self.completed_mission_events.append(event)
        self._debug_log(
            f"mission_complete_done: pos={position}, completer_id={completer_id}, completed_events={self.completed_mission_events}"
        )
        if self._active_frame is not None:
            if self._active_frame.mission_done_events is None:
                self._active_frame.mission_done_events = []
            self._active_frame.mission_done_events.append(event)
        self._maybe_reassign_roles(mission_completed=True)
        return True

    def _captured_human_labels(self, agents: list[AgentFrame], captured_labels: set[str]) -> set[str]:
        alien_positions = {a.position for a in agents if a.role == "alien"}
        if not alien_positions:
            return set()
        newly_captured = set()
        for a in agents:
            if a.role != "human" or a.label in captured_labels:
                continue
            if a.position in alien_positions:
                newly_captured.add(a.label)
        return newly_captured

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, max_steps: int = 200) -> tuple[list[SimulationFrame], str]:
        frames: list[SimulationFrame] = []
        exit_pos = self._exit_position()
        captured_humans: set[str] = set()
        escaped_humans: set[str] = set()
        self.captured_humans = captured_humans
        self.escaped_humans = escaped_humans
        all_human_labels = [s.label for s in self.agents if s.role == "human"]
        if self.total_missions == 0:
            self.total_missions = self._missions_remaining()

        def all_resolved() -> bool:
            return bool(all_human_labels) and len(captured_humans | escaped_humans) >= len(all_human_labels)

        def all_captured() -> bool:
            return bool(all_human_labels) and len(captured_humans) >= len(all_human_labels)

        def escape_outcome() -> str | None:
            if not escaped_humans:
                return None
            return "human_escaped_all" if len(escaped_humans) >= len(all_human_labels) else "human_escaped_some_captured"

        for step in range(max_steps + 1):
            exit_open = self._update_exit_open_state()
            current_agents = self._snapshot_agents()
            outcome = None
            if all_resolved():
                outcome = escape_outcome()
                if outcome is None:
                    outcome = "alien_caught_all_humans" if len(all_human_labels) > 1 else "alien_caught_human"
            if outcome is None:
                outcome = "alien_caught_all_humans" if all_captured() and len(all_human_labels) > 1 else "alien_caught_human" if all_captured() else None

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
                grid_snapshot=self.grid.copy(),
                exit_unlocked=exit_open,
                noise_ripple_pos=self.last_noise_ripple if self.enable_mechanics else None,
                noise_deliberate=self.last_noise_deliberate if self.enable_mechanics else None,
                radar_threat=self.last_radar_threat if self.enable_mechanics else None,
                alien_heard_pos=alien_heard,
                shared_missions=sorted(self.shared_mission_coords),
                shared_exit=self.shared_exit_coord,
                captured_humans=sorted(captured_humans),
                escaped_humans=sorted(escaped_humans),
                mission_done_events=list(self.completed_mission_events),
                agent_roles=self._track_role_changes(current_agents, f"{step}:pre"),
            ))
            self._active_frame = frames[-1]
            if outcome is not None or step == max_steps:
                self._active_frame = None
                return frames, outcome or "max_steps_reached"

            # Update radar before agents act so humans can react to current threat level
            radar_threat_by_label: dict[str, str | None] = {}
            radar_dist_by_label: dict[str, int | None] = {}
            if self.enable_mechanics:
                human_specs = [s for s in self.agents if s.role == "human" and s.label not in captured_humans and s.label not in escaped_humans]
                alien_specs = [s for s in self.agents if s.role == "alien"]
                if human_specs and alien_specs:
                    alien_pos = self._get_position(alien_specs[0])
                    self._update_radar(
                        self._get_position(human_specs[0]),
                        alien_pos,
                    )
                    if self.radar_active_for > 0:
                        for hs in human_specs:
                            dist = self._topology_distance(self._get_position(hs), alien_pos)
                            radar_threat_by_label[hs.label] = self._radar_threat_for_distance(dist)
                            radar_dist_by_label[hs.label] = dist
                if len(human_specs) == 1:
                    only_label = human_specs[0].label
                    self.last_radar_threat = radar_threat_by_label.get(only_label)
                    self.last_radar_dist = radar_dist_by_label.get(only_label)
                else:
                    self.last_radar_threat = None
                    self.last_radar_dist = None

            # First pass: build observations and call observe() for all agents
            current_positions = {s.label: self._get_position(s) for s in self.agents}
            # Call observe/update knowledge
            for spec in self.agents:
                if spec.role == "human" and (spec.label in captured_humans or spec.label in escaped_humans):
                    continue
                current_position = current_positions[spec.label]
                radius = self._view_radius_for(spec)
                direction = getattr(spec.agent, "direction", None)
                if spec.role == "human":
                    try:
                        setattr(spec.agent, "missions_total", self.total_missions)
                        setattr(spec.agent, "missions_remaining", self._missions_remaining())
                    except Exception:
                        pass
                self._update_knowledge(spec.label, current_position, radius, direction)

                if self.enable_mechanics and direction is not None:
                    radar_threat = radar_threat_by_label.get(spec.label) if spec.role == "human" else None
                    obs = self._cone_observation(current_position, direction, radius, radar_threat)
                else:
                    obs = self._partial_observation(current_position, radius)

                if hasattr(spec.agent, "observe"):
                    # Automatic mission detection: if this observation contains
                    # any of the configured mission tile values, register them.
                    if spec.role == "human" and self.mission_tile_values:
                        for mv in self.mission_tile_values:
                            ys, xs = np.where(obs == mv)
                            for y, x in zip(ys, xs):
                                self.add_mission((int(y), int(x)))

                    if spec.role == "human":
                        try:
                            setattr(spec.agent, "exit_open", exit_open)
                        except Exception:
                            pass
                        spec.agent.observe(
                            obs,
                            radar_threat_by_label.get(spec.label),
                            radar_dist_by_label.get(spec.label),
                        )
                    else:
                        spec.agent.observe(obs)

            # Reassign roles immediately if triggered by events (e.g. add_mission).
            # Periodic reassignment tick removed; use event-driven reassignment only.

            # Coord-bus relay: collect shared coords from role-aware humans and broadcast.
            if self.role_based:
                outbox = []
                for spec in self.agents:
                    if spec.role != "human" or spec.label in captured_humans or spec.label in escaped_humans:
                        continue
                    agent = getattr(spec, "agent", None)
                    if agent is not None and hasattr(agent, "flush_outbox"):
                        outbox.extend(agent.flush_outbox())
                if outbox:
                    for msg in outbox:
                        if getattr(msg, "coord_type", None) is None:
                            continue
                        if str(getattr(msg, "coord_type", "")) == "CoordType.EXIT":
                            self.shared_exit_coord = getattr(msg, "pos", None) or self.shared_exit_coord
                        elif str(getattr(msg, "coord_type", "")) == "CoordType.MISSION":
                            pos = getattr(msg, "pos", None)
                            if pos is not None:
                                self.shared_mission_coords.add(pos)
                    for spec in self.agents:
                        if spec.role != "human" or spec.label in captured_humans or spec.label in escaped_humans:
                            continue
                        agent = getattr(spec, "agent", None)
                        if agent is None or not hasattr(agent, "receive_coords"):
                            continue
                        agent_id = getattr(agent, "agent_id", None)
                        to_deliver = [m for m in outbox if getattr(m, "sender_id", None) != agent_id]
                        if to_deliver:
                            agent.receive_coords(to_deliver)

            # Second pass: humans act first so they can set made_loud_noise etc.
            alien_positions = [self._get_position(s) for s in self.agents if s.role == "alien"]
            mission_positions_touched: set[tuple[int, int]] = set()
            for spec in [s for s in self.agents if s.role == "human" and s.label not in captured_humans and s.label not in escaped_humans]:
                try:
                    setattr(spec.agent, "exit_open", self._missions_remaining() == 0)
                except Exception:
                    pass
                current_position = self._get_position(spec)
                opposing_positions = [pos for pos in alien_positions]
                nearest_target = self._nearest_target(current_position, opposing_positions) or current_position
                # humans generally ignore heard_pos; pass None
                new_pos = spec.agent.step(nearest_target, None, step)
                self._set_position(spec, (int(new_pos[0]), int(new_pos[1])))

                if exit_open and exit_pos is not None and self._get_position(spec) == exit_pos:
                    escaped_humans.add(spec.label)
                    try:
                        setattr(spec.agent, "escaped", True)
                    except Exception:
                        pass

                # Mission dwell is counted once per occupied mission tile each step.
                mission_positions_touched.add(self._get_position(spec))

            for mission_pos in mission_positions_touched:
                self._complete_mission_at(mission_pos)

            # Mark any humans who collided with aliens after human movement.
            post_human_agents = self._snapshot_agents()
            newly_captured = self._captured_human_labels(post_human_agents, captured_humans | escaped_humans)
            if newly_captured:
                captured_humans.update(newly_captured)
                self._maybe_reassign_roles(worker_died=True)
            else:
                captured_humans.update(newly_captured)

            # Compute heard position for aliens (consider deliberate loud noises or DECOY response to WORKER threat)
            human_specs = [s for s in self.agents if s.role == "human" and s.label not in captured_humans and s.label not in escaped_humans]
            loud_source = None
            if human_specs:
                # Respect agent-level loud-noise intent. Agents are responsible
                # for deciding when to emit deliberate loud signals (e.g.
                # DECOY behavior). Simulation only consumes the `made_loud_noise`
                # flag set by agents and does not synthesize audible events
                # from team-level heuristics.
                for hs in human_specs:
                    agent_obj = getattr(hs, "agent", None)
                    if agent_obj is not None and getattr(agent_obj, "made_loud_noise", False):
                        loud_source = hs
                        break

            # Now set heard_yx for aliens
            if loud_source is not None:
                h_pos_yx = self._get_position(loud_source)
                heard_yx_for_aliens = h_pos_yx
                self.last_noise_ripple = h_pos_yx
                self.noise_ripple_age = 0
                self.last_noise_deliberate = True
                try:
                    setattr(loud_source.agent, "made_loud_noise", False)
                except Exception:
                    pass
            else:
                heard_yx_for_aliens = None
                noise_sources: list[tuple[tuple[int, int], tuple[int, int]]] = []
                for hs in human_specs:
                    agent_obj = getattr(hs, "agent", None)
                    if agent_obj is None or bool(getattr(agent_obj, "hidden", False)):
                        continue
                    if self._rng.random() < self.p_noise:
                        h_pos_yx = self._get_position(hs)
                        off_y = int(self._rng.integers(-self.noise_radius, self.noise_radius + 1))
                        off_x = int(self._rng.integers(-self.noise_radius, self.noise_radius + 1))
                        ny = max(0, min(h_pos_yx[0] + off_y, self.grid.shape[0] - 1))
                        nx = max(0, min(h_pos_yx[1] + off_x, self.grid.shape[1] - 1))
                        noise_sources.append((h_pos_yx, (ny, nx)))

                if noise_sources:
                    idx = int(self._rng.integers(len(noise_sources)))
                    h_pos_yx, heard_yx_for_aliens = noise_sources[idx]
                    self.last_noise_ripple = h_pos_yx
                    self.noise_ripple_age = 0
                    self.last_noise_deliberate = False
                elif self.last_noise_ripple is not None:
                    self.noise_ripple_age += 1
                    if self.noise_ripple_age > 9:
                        self.last_noise_ripple = None
                        self.last_noise_deliberate = False

            # Third pass: aliens act using heard_yx_for_aliens and updated human positions
            for spec in [s for s in self.agents if s.role == "alien"]:
                current_position = self._get_position(spec)
                opposing_positions = [
                    self._get_position(other)
                    for other in self.agents
                    if other.label != spec.label
                    and other.role != spec.role
                    and other.label not in captured_humans
                    and other.label not in escaped_humans
                ]
                nearest_target = self._nearest_target(current_position, opposing_positions) or current_position
                # Pass the same heard position to all aliens (None if no humans)
                new_pos = spec.agent.step(nearest_target, heard_yx_for_aliens, step)
                self._set_position(spec, (int(new_pos[0]), int(new_pos[1])))

            # Mark any humans who collided with aliens after alien movement.
            post_alien_agents = self._snapshot_agents()
            newly_captured = self._captured_human_labels(post_alien_agents, captured_humans)
            if newly_captured:
                captured_humans.update(newly_captured)
                # If there are still humans remaining, clear alien pursuit
                # state so it resumes searching instead of camping.
                if len(captured_humans) < len(all_human_labels):
                    for spec in [s for s in self.agents if s.role == "alien"]:
                        agent = getattr(spec, "agent", None)
                        if agent is None:
                            continue
                        try:
                            agent.last_known_pos = None
                            agent.last_heard_pos = None
                            agent.steps_since_heard = 999
                            agent.player_known_hiding = False
                            agent.path = []
                            agent.state = AlienState.SEARCH
                        except Exception:
                            pass
            else:
                captured_humans.update(newly_captured)

            if all_resolved():
                outcome = escape_outcome()
                if outcome is None:
                    # If nobody escaped, fall back to the capture outcomes.
                    outcome = "alien_caught_all_humans" if len(all_human_labels) > 1 else "alien_caught_human"
                final_agents = post_alien_agents
                alien_heard = None
                if self.enable_mechanics:
                    for spec in self.agents:
                        if spec.role == "alien":
                            alien_heard = getattr(spec.agent, "last_heard_pos", None)
                            break
                frames.append(SimulationFrame(
                    step=step + 1,
                    outcome=outcome,
                    agents=final_agents,
                    grid_snapshot=self.grid.copy(),
                    noise_ripple_pos=self.last_noise_ripple if self.enable_mechanics else None,
                    radar_threat=self.last_radar_threat if self.enable_mechanics else None,
                    alien_heard_pos=alien_heard,
                    shared_missions=sorted(self.shared_mission_coords),
                    shared_exit=self.shared_exit_coord,
                    captured_humans=sorted(captured_humans),
                    escaped_humans=sorted(escaped_humans),
                    mission_done_events=list(self.completed_mission_events),
                    agent_roles=self._track_role_changes(final_agents, f"{step+1}:caught"),
                ))
                self._active_frame = None
                return frames, outcome

            # Update hidden flag for humans after movement
            for spec in [s for s in self.agents if s.role == "human" and s.label not in captured_humans and s.label not in escaped_humans]:
                setattr(spec.agent, "hidden", self._is_hidden_position(self._get_position(spec)))

            # Refresh exit state after any mission completions during this step.
            exit_open = self._update_exit_open_state()

        self._active_frame = None
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
        # Role-aware humans may start with team_role=NONE until missions are known.
        # Use agent_id as the stable signal that role labels should be rendered.
        has_role_agents = any(
            a.role == "human" and a.agent_id is not None
            for a in frames[0].agents
        )

        if world_only or self.knowledge_mode == "off":
            def _set_scatter_marker(artist, marker: str) -> None:
                ms = MarkerStyle(marker)
                path = ms.get_path().transformed(ms.get_transform())
                artist.set_paths([path])

            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)
            fig.patch.set_facecolor("#000000")
            initial_grid = frames[0].grid_snapshot if frames[0].grid_snapshot is not None else self.grid
            world_image = ax.imshow(initial_grid, cmap=world_cmap, vmin=0, vmax=7)
            ax.autoscale(False)  # prevent large Circle patches from shrinking the map
            ax.set_title("Game World", color="white", fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#0f0f1e")

            marker_artists = []
            hidden_rings = []
            color_map = self._role_colors()
            label_texts = []
            for index, agent in enumerate(frames[0].agents):
                color = color_map[agent.role][index % len(color_map[agent.role])]
                marker = self._role_marker(agent, index)
                y, x = agent.position
                artist = ax.scatter([x], [y], s=120, c=color, edgecolors="white", linewidths=1.0, marker=marker, zorder=5)
                marker_artists.append(artist)
                # Add id:ROLE text for human agents when roles are in use
                if has_role_agents and agent.role == "human":
                    aid = agent.agent_id if getattr(agent, "agent_id", None) is not None else "?"
                    role_name = agent.team_role.name if agent.team_role is not None else "NONE"
                    txt = ax.text(x + 0.3, y, f"id{aid}: {role_name}", color="white", fontsize=8, zorder=7)
                    label_texts.append(txt)
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
            if len([a for a in frames[0].agents if a.role == "human"]) == 1:
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
            legend_text = None
            shared_text = None
            if has_role_agents:
                legend_text = fig.text(0.985, 0.92, self._role_legend_text(),
                                       color="white", fontsize=10, ha="right",
                                       va="top", fontfamily="monospace")
                shared_text = fig.text(0.985, 0.65, "", color="white", fontsize=9,
                                       ha="right", va="top", fontfamily="monospace")

            def update_world(frame_index: int):
                state = frames[frame_index]
                if state.grid_snapshot is not None:
                    world_image.set_data(state.grid_snapshot)
                captured = set(state.captured_humans or [])
                escaped = set(state.escaped_humans or [])
                resolved = captured | escaped
                human_ring_idx = 0
                for artist, agent in zip(marker_artists, state.agents):
                    if agent.role == "human" and agent.label in resolved:
                        artist.set_visible(False)
                        if human_ring_idx < len(hidden_rings):
                            hidden_rings[human_ring_idx].set_visible(False)
                        human_ring_idx += 1
                        continue
                    y, x = agent.position
                    _set_scatter_marker(artist, self._role_marker(agent, human_ring_idx if agent.role == "human" else 0))
                    artist.set_visible(True)
                    artist.set_offsets([[x, y]])
                    artist.set_alpha(0.55 if agent.hidden else 1.0)
                    if agent.role == "human":
                        hidden_rings[human_ring_idx].set_offsets([[x, y]])
                        hidden_rings[human_ring_idx].set_visible(agent.hidden)
                        human_ring_idx += 1
                # Update label texts positions and content — pair texts with human agents only
                if has_role_agents:
                    human_agents = [a for a in state.agents if a.role == "human"]
                    for txt, agent in zip(label_texts, human_agents):
                        aid = agent.agent_id if getattr(agent, "agent_id", None) is not None else "?"
                        role_name = state.agent_roles.get(int(aid), self._role_name(agent.team_role)) if state.agent_roles else self._role_name(agent.team_role)
                        txt.set_position((agent.position[1] + 0.3, agent.position[0]))
                        txt.set_text(f"id{aid}: {role_name}")
                        txt.set_visible(agent.label not in resolved)

                # Radar threat rings — recentre on current human pos, highlight active band
                humans = [a for a in state.agents if a.role == "human"]
                active_idx = _threat_order.get(state.radar_threat, -1)
                if len(humans) == 1:
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
                    color = "#00FFFF" if state.noise_deliberate else "yellow"
                    for c in ripple_circles:
                        c.set_edgecolor(color)
                        c.set_center((nx, ny))
                        c.set_visible(True)
                else:
                    for c in ripple_circles:
                        c.set_visible(False)

                # Alien heard position
                if state.alien_heard_pos is not None:
                    ahy, ahx = state.alien_heard_pos
                    color = "#00FFFF" if state.noise_deliberate else "#FF4444"
                    alien_heard_marker.set_color(color)
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
                    f"  |  {self._format_human_roles(state)}"
                    f"  Alien: {alien_mode:<11}"
                    f"  Exit: {exit_state:<9}"
                    f"  |  {outcome}"
                )
                if shared_text is not None:
                    shared_text.set_text(self._format_shared_coords(state))
                artists = [*marker_artists, *hidden_rings, *threat_rings, *ripple_circles,
                           alien_heard_marker, status_text]
                if legend_text is not None:
                    artists.append(legend_text)
                if shared_text is not None:
                    artists.append(shared_text)
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
        def _set_scatter_marker(artist, marker: str) -> None:
            ms = MarkerStyle(marker)
            path = ms.get_path().transformed(ms.get_transform())
            artist.set_paths([path])

        n_panels = 1 + len(frames[0].agents)
        fig_size = (16, 5) if has_single_pair else (max(16, 5 * n_panels), 5)
        fig, axes = plt.subplots(1, n_panels, figsize=fig_size, dpi=120)
        fig.patch.set_facecolor("#000000")
        if n_panels == 1:
            axes = [axes]

        world_ax = axes[0]
        initial_grid = frames[0].grid_snapshot if frames[0].grid_snapshot is not None else self.grid
        world_image = world_ax.imshow(initial_grid, cmap=world_cmap, vmin=0, vmax=7)
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
            marker = self._role_marker(agent, index)
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
            # Show agent label with id and team role when available
            if agent.role == "human":
                aid = agent.agent_id if getattr(agent, "agent_id", None) is not None else "?"
                axis.set_title(f"{agent.label} (id{aid})", color="white", fontsize=11, fontweight="bold")
            else:
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
        if len([a for a in frames[0].agents if a.role == "human"]) == 1:
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
        legend_text = None
        shared_text = None
        if has_role_agents:
            legend_text = fig.text(0.985, 0.92, self._role_legend_text(),
                                   color="white", fontsize=10, ha="right",
                                   va="top", fontfamily="monospace")
            shared_text = fig.text(0.985, 0.65, "", color="white", fontsize=9,
                                   ha="right", va="top", fontfamily="monospace")

        def update_full(frame_index: int):
            state = frames[frame_index]
            if state.grid_snapshot is not None:
                world_image.set_data(state.grid_snapshot)
            human_ring_idx = 0
            captured = set(state.captured_humans or [])
            escaped = set(state.escaped_humans or [])
            resolved = captured | escaped
            for artist, agent in zip(world_markers, state.agents):
                if agent.role == "human" and agent.label in resolved:
                    artist.set_visible(False)
                    if human_ring_idx < len(hidden_rings):
                        hidden_rings[human_ring_idx].set_visible(False)
                    human_ring_idx += 1
                    continue
                y, x = agent.position
                _set_scatter_marker(artist, self._role_marker(agent, human_ring_idx if agent.role == "human" else 0))
                artist.set_visible(True)
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
                if agent.role == "human" and agent.label in resolved:
                    marker.set_visible(False)
                    opp_m.set_visible(False)
                    continue
                y, x = agent.position
                _set_scatter_marker(marker, "X" if agent.role == "alien" else self._role_marker(agent, 0))
                marker.set_visible(True)
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
            if len(humans) == 1:
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
                color = "#00FFFF" if state.noise_deliberate else "yellow"
                for c in ripple_circles:
                    c.set_edgecolor(color)
                    c.set_center((nx, ny))
                    c.set_visible(True)
            else:
                for c in ripple_circles:
                    c.set_visible(False)

            # Alien heard position on alien knowledge panels
            for m in alien_heard_markers:
                if state.alien_heard_pos is not None:
                    ahy, ahx = state.alien_heard_pos
                    color = "#00FFFF" if state.noise_deliberate else "#FF4444"
                    m.set_color(color)
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
                f"  |  {self._format_human_roles(state)}"
                f"  Alien: {alien_mode:<11}"
                f"  Exit: {exit_state:<9}"
                f"  |  {outcome}"
            )
            if shared_text is not None:
                shared_text.set_text(self._format_shared_coords(state))
            fov_imgs = [img for img, _ in fov_overlays]
            artists = [*world_markers, *hidden_rings, *threat_rings, *ripple_circles,
                       *alien_heard_markers, *knowledge_images, *knowledge_markers,
                       *fov_imgs, *opp_markers, status_text]
            if legend_text is not None:
                artists.append(legend_text)
            if shared_text is not None:
                artists.append(shared_text)
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
    def _role_marker(agent: AgentFrame, index: int) -> str:
        if agent.role == "alien":
            # Use cross for aliens to match alien view marker for the first alien
            return "X"

        if agent.role == "human" and agent.team_role not in (None, TeamRole.NONE):
            role_markers = {
                TeamRole.WORKER: "o",
                TeamRole.DECOY: "D",
                TeamRole.RUNNER: "^",
                TeamRole.NONE: GenericMapSimulation.HUMAN_MARKERS[index % len(GenericMapSimulation.HUMAN_MARKERS)],
            }
            return role_markers.get(agent.team_role, GenericMapSimulation.HUMAN_MARKERS[index % len(GenericMapSimulation.HUMAN_MARKERS)])

        return GenericMapSimulation.HUMAN_MARKERS[index % len(GenericMapSimulation.HUMAN_MARKERS)]

    @staticmethod
    def _role_legend_text() -> str:
        return "ROLE ICONS\nWORKER: o\nDECOY: D\nRUNNER: ^"

    @staticmethod
    def _format_shared_coords(state: SimulationFrame) -> str:
        lines = ["SHARED COORDS"]
        exit_pos = state.shared_exit
        if exit_pos is not None:
            lines.append(f"EXIT: ({exit_pos[0]}, {exit_pos[1]})")
        else:
            lines.append("EXIT: --")
        missions = state.shared_missions or []
        if missions:
            shown = missions[:6]
            lines.append("MISSIONS:")
            for pos in shown:
                lines.append(f"- ({pos[0]}, {pos[1]})")
            if len(missions) > len(shown):
                lines.append(f"... +{len(missions) - len(shown)}")
        else:
            lines.append("MISSIONS: --")
        role_items = state.agent_roles or {}
        if role_items:
            lines.append("ROLES:")
            for agent_id, role_name in sorted(role_items.items()):
                lines.append(f"- id{agent_id}: {role_name}")
        done_events = state.mission_done_events or []
        if done_events:
            lines.append("DONE:")
            shown_done = done_events[-4:]
            for pos, agent_id in shown_done:
                lines.append(f"- ({pos[0]}, {pos[1]}) by id{agent_id}")
            if len(done_events) > len(shown_done):
                lines.append(f"... +{len(done_events) - len(shown_done)}")
        else:
            lines.append("DONE: --")
        return "\n".join(lines)

    @staticmethod
    def _format_human_roles(state: SimulationFrame) -> str:
        role_items = state.agent_roles or {}
        if not role_items:
            return "Humans: --"
        label_by_id: dict[int, str] = {}
        for agent in state.agents:
            if agent.role == "human" and agent.agent_id is not None:
                label_by_id[int(agent.agent_id)] = agent.label
        parts = []
        for agent_id in sorted(role_items):
            label = label_by_id.get(agent_id, f"human_{agent_id + 1}")
            parts.append(f"{label}=id{agent_id}")
        return "Humans: " + ", ".join(parts)

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
