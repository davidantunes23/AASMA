from collections import deque

import numpy as np
from enum import Enum

from agents.base import BaseHumanAgent, Direction


class Action(Enum):
    WAIT = 0
    WALK = 1
    LOUD_NOISE = 2
from map_generator import Tile


class HumanAgent(BaseHumanAgent):
    """Rule-based human agent. Uses BFS navigation, radar-reactive hiding.

    All positions are (y, x) / (row, col). The simulation calls observe()
    once per step before step() so the agent can update its internal map.
    """

    UNKNOWN = -1
    ALIEN = -2
    RADAR_PING = -3
    NOISE_RIPPLE = -4

    def __init__(self, start_pos: tuple[int, int], start_dir: Direction = Direction.NORTH, view_length: int = 6):
        self.pos = start_pos
        self.direction = start_dir
        self.view_length = view_length
        self.hidden: bool = False
        self.exit_open: bool = False
        self.last_radar_threat: str | None = None
        self.last_radar_dist: int | None = None
        self._known_map: np.ndarray | None = None
        self._known_exit: tuple[int, int] | None = None
        self._known_missions: set[tuple[int, int]] = set()
        self._completed_missions: set[tuple[int, int]] = set()
        self._current_objective: tuple[int, int] | None = None
        self.mission_manager = None
        self._observed_aliens: set[tuple[int, int]] = set()

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
            return self.pos

        # PRIORITY 1: Stay hidden while threat is high
        if self.hidden:
            if not self._should_keep_hiding():
                self.hidden = False
            else:
                return self.pos

        # PRIORITY 2: Handle current objective (mission or exit)
        self._current_objective = self._select_objective()
        if self._current_objective is not None and self._current_objective in self._known_missions:
            if self.pos == self._current_objective:
                self._advance_current_mission()
                self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                return self.pos

        if self._current_objective is not None:
            nxt = self._step_toward_target(self._current_objective)
            if nxt is not None and nxt != self.pos:
                self.pos = nxt
                self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                return self.pos

        # PRIORITY 3: Hide when threatened and no nearby exit
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

        # PRIORITY 4: Explore
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

    # ── Observation integration ───────────────────────────────────────────────

    def _init_memory(self, obs: np.ndarray):
        if self._known_map is not None and self._known_map.shape == obs.shape:
            return
        self._known_map = np.full(obs.shape, self.UNKNOWN, dtype=np.int16)
        self._known_exit = None
        self._known_missions = set()
        self._completed_missions = set()
        self._current_objective = None
        self._observed_aliens = set()

    def _integrate_observation(self, obs: np.ndarray):
        radar_active = np.any(obs == self.RADAR_PING)
        visible_mask = (
            (obs != self.UNKNOWN)
            & (obs != self.ALIEN)
            & (obs != self.RADAR_PING)
            & (obs != self.NOISE_RIPPLE)
        )
        self._known_map[visible_mask] = obs[visible_mask]
        self.hidden = self._tile_at(self.pos) == int(Tile.HIDE)
        if radar_active:
            self._observed_aliens = {self.pos}
        else:
            self._observed_aliens = set()
        ey, ex = np.where(self._known_map == int(Tile.EXIT))
        if len(ey) > 0:
            self._known_exit = (int(ey[0]), int(ex[0]))
        my, mx = np.where(self._known_map == int(Tile.MISSION))
        if len(my) > 0:
            seen = {(int(y), int(x)) for y, x in zip(my, mx)}
            self._known_missions |= seen
            self._known_missions -= self._completed_missions

    # ── Navigation ────────────────────────────────────────────────────────────

    def _step_toward_target(self, target: tuple[int, int]) -> tuple[int, int] | None:
        nxt = self._bfs_next_step(lambda pos: pos == target)
        if nxt is not None and nxt != self.pos:
            self.direction = self._direction_from_step(self.pos, nxt)
        return nxt

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        return self._bfs_next_step(self._is_floor_frontier)

    def _select_objective(self) -> tuple[int, int] | None:
        if self._known_missions:
            return min(
                self._known_missions,
                key=lambda pos: abs(pos[0] - self.pos[0]) + abs(pos[1] - self.pos[1]),
            )
        if self._exit_unlocked():
            return self._known_exit
        return None

    def _exit_unlocked(self) -> bool:
        if self.mission_manager is None:
            return True
        return self.mission_manager.exit_unlocked()

    def _advance_current_mission(self) -> None:
        if self._current_objective is None:
            return
        if self.mission_manager is not None:
            completed = self.mission_manager.update(self._current_objective)
        else:
            completed = True
        if completed:
            self._completed_missions.add(self._current_objective)
            self._known_missions.discard(self._current_objective)

    def _adjacent_unknown_step(self) -> tuple[int, int] | None:
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
        return self._bfs_next_step(self._is_frontier)

    def _bfs_next_step(self, is_target) -> tuple[int, int] | None:
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
        current = target
        while parents[current] is not None and parents[current] != self.pos:
            current = parents[current]
        if parents[current] is None:
            return None
        return current

    def _best_local_move(self) -> tuple[int, int] | None:
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
        return self.last_radar_threat in {"CRITICAL", "CLOSE"}

    def _should_hide_now(self) -> bool:
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
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        delta = (order.index(candidate) - order.index(current)) % 4
        if delta == 0:
            return 0
        if delta == 2:
            return 2
        return 1
