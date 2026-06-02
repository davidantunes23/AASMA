"""Rule-based human agent (single-agent, no team coordination).

Uses BFS navigation over a partial map built from cone observations.
Priority stack each step:
  1. Stay hidden while threat ≥ CLOSE
  2. Seek hiding spot when threatened and exit is far
  3. Navigate to nearest known mission, then to exit once open
  4. Explore unknown frontiers

This agent does not coordinate with teammates. Use RoleHumanAgent for
multi-agent scenarios with explicit role assignment.
"""
from collections import deque

import numpy as np
from enum import Enum

from agents.base import BaseHumanAgent, Direction

# Action enum kept for interface compatibility with the RL training pipeline.
class Action(Enum):
    WAIT = 0
    WALK = 1
    LOUD_NOISE = 2
from map_generator import Tile


class HumanAgent(BaseHumanAgent):
    """Rule-based human agent. Uses BFS navigation and radar-reactive hiding.

    All positions are (y, x) / (row, col). The simulation calls observe()
    once per step before step() so the agent can update its internal map.
    """

    # Special sentinel values embedded in the observation array by the simulation.
    UNKNOWN      = -1   # cell not yet seen by this agent's cone FOV
    ALIEN        = -2   # alien position marker (injected when alien is visible)
    RADAR_PING   = -3   # radar proximity alert tile
    NOISE_RIPPLE = -4   # sound ripple marker (ignored for map building)

    def __init__(self, start_pos: tuple[int, int], start_dir: Direction = Direction.NORTH, view_length: int = 6):
        self.pos = start_pos
        self.direction = start_dir
        self.view_length = view_length
        self.hidden: bool = False             # True when standing on a HIDE tile
        self.exit_open: bool = False          # set by simulation once all missions complete
        self.last_radar_threat: str | None = None  # most recent radar band (CRITICAL/CLOSE/NEAR/FAR)
        self.last_radar_dist: int | None = None    # topology distance to alien at last radar tick
        self._known_map: np.ndarray | None = None  # tile map built from cone observations
        self._known_exit: tuple[int, int] | None = None    # exit position once seen
        self._known_missions: set[tuple[int, int]] = set() # mission tiles seen and not yet completed
        self._completed_missions: set[tuple[int, int]] = set()  # missions already finished
        self._current_objective: tuple[int, int] | None = None  # current BFS navigation target
        self.mission_manager = None           # reserved for external mission management hooks
        self._observed_aliens: set[tuple[int, int]] = set()  # alien positions seen this step

    # ── Public interface ──────────────────────────────────────────────────────

    def observe(
        self,
        obs: np.ndarray,
        radar_threat: str | None = None,
        radar_dist: int | None = None,
    ) -> None:
        if radar_threat is not None:
            self.last_radar_threat = radar_threat
            self.last_radar_dist = radar_dist
        self._init_memory(obs)
        self._integrate_observation(obs)

    def step(
        self,
        _player_pos: tuple[int, int],
        _heard_pos: tuple[int, int] | None = None,
        _step_num: int = 0,
    ) -> tuple[int, int]:
        """Return new (y, x) position. observe() must be called first each step."""
        if self.exit_open and self._tile_at(self.pos) == int(Tile.EXIT):
            return self.pos  # already on open exit — stay and let simulation register escape

        # PRIORITY 1: Stay hidden while threat is high.
        if self.hidden:
            if not self._should_keep_hiding():
                self.hidden = False
            else:
                return self.pos

        # PRIORITY 2: Hide when threatened and no nearby exit.
        if self._should_hide_now():
            spot = self._get_closest_hiding_spot()
            if spot is not None:
                nxt = self._step_toward_target(spot)
                if nxt is not None:
                    if nxt != self.pos:
                        self.pos = nxt
                    else:
                        self.hidden = True
                    self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                    return self.pos

        # PRIORITY 3: Navigate to current objective (mission tile or exit).
        self._current_objective = self._select_objective()
        if self._current_objective is not None and self._current_objective in self._known_missions:
            if self.pos == self._current_objective:
                completed = self._advance_current_mission()
                self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                if not completed:
                    return self.pos  # dwell here; simulation tracks progress
                self._current_objective = None

        if self._current_objective is None:
            self._current_objective = self._select_objective()

        if self._current_objective is not None:
            nxt = self._step_toward_target(self._current_objective)
            if nxt is not None and nxt != self.pos:
                self.pos = nxt
                self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                return self.pos

        # PRIORITY 4: Explore unknown frontiers.
        nxt = self._adjacent_unknown_step()
        if nxt is None:
            nxt = self._next_step_to_nearest_floor_frontier()
        if nxt is None:
            nxt = self._next_step_to_nearest_frontier()
        if nxt is None:
            nxt = self._best_local_move()

        if self._is_observed_alien(nxt):
            nxt = None

        if nxt is not None and nxt != self.pos:
            self.direction = self._direction_from_step(self.pos, nxt)
            self.pos = nxt

        self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
        return self.pos

    def reset(self, start_pos: tuple[int, int] | None = None) -> None:
        if start_pos is not None:
            self.pos = start_pos
        self._known_map = None
        self._known_exit = None
        self._known_missions = set()
        self._completed_missions = set()
        self._current_objective = None
        self._observed_aliens = set()
        self.hidden = False
        self.last_radar_threat = None
        self.last_radar_dist = None

    def remove_mission(self, position: tuple[int, int]) -> None:
        """Called by the simulation when a mission tile is completed."""
        pos = (int(position[0]), int(position[1]))
        self._known_missions.discard(pos)
        self._completed_missions.add(pos)
        if self._current_objective == pos:
            self._current_objective = None

    # ── Observation integration ───────────────────────────────────────────────

    def _init_memory(self, obs: np.ndarray):
        """Allocate the known map on the first observation of a new episode."""
        if self._known_map is not None and self._known_map.shape == obs.shape:
            return
        self._known_map = np.full(obs.shape, self.UNKNOWN, dtype=np.int16)
        self._known_exit = None
        self._known_missions = set()
        self._completed_missions = set()
        self._current_objective = None
        self._observed_aliens = set()

    def _integrate_observation(self, obs: np.ndarray):
        """Copy visible tile IDs into the known map; detect exit, missions, and aliens."""
        radar_active = np.any(obs == self.RADAR_PING)
        # Only copy real tile IDs — exclude the special sentinel markers.
        visible_mask = (
            (obs != self.UNKNOWN)
            & (obs != self.ALIEN)
            & (obs != self.RADAR_PING)
            & (obs != self.NOISE_RIPPLE)
        )
        self._known_map[visible_mask] = obs[visible_mask]
        self.hidden = self._tile_at(self.pos) == int(Tile.HIDE)

        # A radar ping means the alien is within topology distance — treat
        # the agent's own cell as a rough alien sighting for avoidance.
        if radar_active:
            self._observed_aliens = {self.pos}
        else:
            self._observed_aliens = set()

        ey, ex = np.where(self._known_map == int(Tile.EXIT))
        if len(ey) > 0:
            self._known_exit = (int(ey[0]), int(ex[0]))

        my, mx = np.where(obs == int(Tile.MISSION))
        if len(my) > 0:
            seen = {(int(y), int(x)) for y, x in zip(my, mx)}
            self._known_missions |= seen

        # Remove any mission tile that was visible this step but is no longer
        # marked as MISSION — it was completed while the agent was watching.
        if self._known_missions:
            vy, vx = np.where(visible_mask)
            visible_positions = {(int(y), int(x)) for y, x in zip(vy, vx)}
            stale = {
                pos for pos in self._known_missions
                if pos in visible_positions and obs[pos] != int(Tile.MISSION)
            }
            if stale:
                self._known_missions -= stale
        self._known_missions -= self._completed_missions

    # ── Navigation ────────────────────────────────────────────────────────────

    def _step_toward_target(self, target: tuple[int, int]) -> tuple[int, int] | None:
        """BFS next step toward a specific cell."""
        nxt = self._bfs_next_step(lambda pos: pos == target)
        if nxt is not None and nxt != self.pos:
            self.direction = self._direction_from_step(self.pos, nxt)
        return nxt

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        """BFS next step toward the nearest FLOOR cell adjacent to unknown space."""
        return self._bfs_next_step(self._is_floor_frontier)

    def _select_objective(self) -> tuple[int, int] | None:
        """Choose the current navigation target: nearest mission, then exit."""
        if self._known_missions:
            return min(
                self._known_missions,
                key=lambda pos: abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1]),
            )
        if self._exit_unlocked():
            return self._known_exit
        return None

    def _exit_unlocked(self) -> bool:
        return bool(self.exit_open)

    def _advance_current_mission(self) -> bool:
        """Dwell logic stub — mission progress is tracked by the simulation, not the agent."""
        if self._current_objective is None:
            return False
        return False  # always return False so the agent keeps dwelling; simulation fires completion

    def _adjacent_unknown_step(self) -> tuple[int, int] | None:
        """Prefer stepping directly into an unknown cell if one is reachable.
        Tie-breaks by turn cost to avoid zig-zagging.
        """
        y, x = self.pos
        candidates: list[tuple[tuple[int, int], Direction]] = []
        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            candidate = (ny, nx)
            if not self._in_bounds(ny, nx):
                continue
            if candidate in self._observed_aliens:
                continue
            if self._known_map[ny, nx] != self.UNKNOWN:
                continue
            candidates.append((candidate, direction))
        if not candidates:
            return None
        candidates.sort(key=lambda item: self._turn_cost(self.direction, item[1]))
        return candidates[0][0]

    def _next_step_to_nearest_frontier(self) -> tuple[int, int] | None:
        """BFS next step toward the nearest traversable frontier (any tile type)."""
        return self._bfs_next_step(self._is_frontier)

    def _bfs_next_step(self, is_target) -> tuple[int, int] | None:
        """Generic BFS that returns the first step on the shortest path to any
        cell satisfying ``is_target``. Returns None if no reachable target exists.
        """
        start = self.pos
        if not self._in_bounds(*start):
            return None
        frontier = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            current = frontier.popleft()
            if current != start and is_target(current):
                return self._first_step_from_path(current, parents)
            for neighbor, _ in self._walkable_neighbors(current):
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                frontier.append(neighbor)
        return None

    def _first_step_from_path(
        self,
        target: tuple[int, int],
        parents: dict[tuple[int, int], tuple[int, int] | None],
    ) -> tuple[int, int] | None:
        """Walk the BFS parent map back to find the first step from self.pos."""
        current = target
        while parents[current] is not None and parents[current] != self.pos:
            current = parents[current]
        if parents[current] is None:
            return None
        return current

    def _best_local_move(self) -> tuple[int, int] | None:
        """Fallback: pick the walkable neighbour with the most unknown neighbours,
        preferring frontier cells and minimising turn cost.
        """
        neighbors = self._walkable_neighbors(self.pos)
        if not neighbors:
            return None
        frontier_neighbors = [item for item in neighbors if self._is_frontier(item[0])]
        candidates = frontier_neighbors if frontier_neighbors else neighbors
        candidates.sort(key=lambda item: (
            self._turn_cost(self.direction, item[1]),
            -self._unknown_neighbor_count(item[0]),
        ))
        return candidates[0][0]

    def _walkable_neighbors(self, position: tuple[int, int]) -> list[tuple[tuple[int, int], Direction]]:
        """Return all known-traversable neighbours of ``position``, excluding alien cells."""
        y, x = position
        neighbors: list[tuple[tuple[int, int], Direction]] = []
        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            candidate = (ny, nx)
            if not self._in_bounds(ny, nx):
                continue
            if self._is_observed_alien(candidate):
                continue
            if not self._is_traversable_known(candidate):
                continue
            neighbors.append((candidate, direction))
        return neighbors

    def _get_closest_hiding_spot(self) -> tuple[int, int] | None:
        """BFS to the nearest HIDE tile reachable through the known map."""
        hiding_spots = self._get_known_hiding_spots()
        if not hiding_spots:
            return None
        start = self.pos
        if not self._in_bounds(*start):
            return None
        frontier = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            current = frontier.popleft()
            if current in hiding_spots:
                return current
            for neighbor, _ in self._walkable_neighbors(current):
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                frontier.append(neighbor)
        return None

    # ── Decision helpers ──────────────────────────────────────────────────────

    def _should_keep_hiding(self) -> bool:
        """Stay hidden until the radar threat drops below CLOSE."""
        return self.last_radar_threat in {"CRITICAL", "CLOSE"}

    def _should_hide_now(self) -> bool:
        """Decide whether to seek a hiding spot this step.

        Exception: if the exit is known and very close (≤15 cells away) a CLOSE
        threat is worth ignoring — the agent can reach the exit before the alien.
        """
        if self.last_radar_threat is None:
            return False
        if self.last_radar_threat == "CRITICAL":
            return True
        if self.last_radar_threat == "CLOSE":
            if self._known_exit is not None:
                dist = abs(self._known_exit[0] - self.pos[0]) + abs(self._known_exit[1] - self.pos[1])
                if dist <= 15:
                    return False
            return True
        return False

    # ── Map query helpers ─────────────────────────────────────────────────────

    def _get_known_hiding_spots(self) -> list[tuple[int, int]]:
        if self._known_map is None:
            return []
        hy, hx = np.where(self._known_map == int(Tile.HIDE))
        return [(int(y), int(x)) for y, x in zip(hy, hx)]

    def _is_floor_frontier(self, position: tuple[int, int]) -> bool:
        if self._tile_at(position) != int(Tile.FLOOR):
            return False
        return self._has_unknown_neighbor(position)

    def _is_frontier(self, position: tuple[int, int]) -> bool:
        if not self._is_traversable_known(position):
            return False
        return self._has_unknown_neighbor(position)

    def _has_unknown_neighbor(self, position: tuple[int, int]) -> bool:
        y, x = position
        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            if self._in_bounds(ny, nx) and self._known_map[ny, nx] == self.UNKNOWN:
                return True
        return False

    def _unknown_neighbor_count(self, position: tuple[int, int]) -> int:
        y, x = position
        count = 0
        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            if self._in_bounds(ny, nx) and self._known_map[ny, nx] == self.UNKNOWN:
                count += 1
        return count

    def _is_traversable_known(self, position: tuple[int, int]) -> bool:
        """A cell is traversable if it is known and not a wall, alien, or locked exit."""
        tile = self._tile_at(position)
        if tile == int(Tile.EXIT) and not self.exit_open:
            return False
        return tile not in (self.UNKNOWN, self.ALIEN, int(Tile.WALL))

    def _is_observed_alien(self, position: tuple[int, int] | None) -> bool:
        if position is None:
            return False
        return position in self._observed_aliens

    def _in_bounds(self, y: int, x: int) -> bool:
        if self._known_map is None:
            return False
        return 0 <= y < self._known_map.shape[0] and 0 <= x < self._known_map.shape[1]

    def _tile_at(self, position: tuple[int, int]) -> int:
        return int(self._known_map[position])

    # ── Direction helpers ─────────────────────────────────────────────────────

    def _direction_from_step(self, start: tuple[int, int], end: tuple[int, int]) -> Direction:
        dy = end[0] - start[0]
        dx = end[1] - start[1]
        if dy == -1 and dx == 0:
            return Direction.NORTH
        if dy == 0 and dx == 1:
            return Direction.EAST
        if dy == 1 and dx == 0:
            return Direction.SOUTH
        if dy == 0 and dx == -1:
            return Direction.WEST
        return self.direction

    def _direction_delta(self, direction: Direction) -> tuple[int, int]:
        if direction == Direction.NORTH:
            return (-1, 0)
        if direction == Direction.EAST:
            return (0, 1)
        if direction == Direction.SOUTH:
            return (1, 0)
        return (0, -1)

    def _turn_cost(self, current: Direction, candidate: Direction) -> int:
        """Penalise turns: 0 = straight, 1 = left/right, 2 = U-turn."""
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        delta = (order.index(candidate) - order.index(current)) % 4
        if delta == 0:
            return 0
        if delta == 2:
            return 2
        return 1
