from enum import Enum
from collections import deque
from pathlib import Path
import sys

import numpy as np

try:
    from map_generator import Tile
except ModuleNotFoundError as exc:
    if exc.name != "map_generator":
        raise
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    from map_generator import Tile

class Direction(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4

class Action(Enum):
    WALK = 1


class HumanAgent:
    UNKNOWN = -1
    ALIEN = -2

    def __init__(self, start_pos: tuple, start_dir: Direction):
        self.position = start_pos
        self.direction = start_dir
        self._known_map: np.ndarray | None = None
        self._known_exit: tuple[int, int] | None = None
        self._observed_aliens: set[tuple[int, int]] = set()

    def _act(self, obs) -> tuple:
        self._init_memory(obs)
        self._integrate_observation(obs)

        next_position = None
        if self._known_exit is not None:
            next_position = self._bfs_next_step(lambda pos: pos == self._known_exit)
        if next_position is None:
            next_position = self._adjacent_unknown_step()
        if next_position is None:
            next_position = self._next_step_to_nearest_floor_frontier()
        if next_position is None:
            next_position = self._next_step_to_nearest_frontier()
        if next_position is None:
            next_position = self._best_local_move()

        if self._is_observed_alien(next_position):
            next_position = None

        if next_position is not None and next_position != self.position:
            self.direction = self._direction_from_step(self.position, next_position)

        return (Action.WALK, self.direction)

    def _get_direction(self) -> Direction:
        return self.direction

    def _init_memory(self, obs: np.ndarray):
        if self._known_map is not None and self._known_map.shape == obs.shape:
            return
        self._known_map = np.full(obs.shape, self.UNKNOWN, dtype=np.int16)
        self._known_exit = None
        self._observed_aliens = set()

    def _integrate_observation(self, obs: np.ndarray):
        visible_mask = (obs != self.UNKNOWN) & (obs != self.ALIEN)
        self._known_map[visible_mask] = obs[visible_mask]

        ay, ax = np.where(obs == self.ALIEN)
        self._observed_aliens = {(int(y), int(x)) for y, x in zip(ay, ax)}

        ey, ex = np.where(self._known_map == int(Tile.EXIT))
        if len(ey) > 0:
            self._known_exit = (int(ey[0]), int(ex[0]))

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        return self._bfs_next_step(self._is_floor_frontier)

    def _adjacent_unknown_step(self) -> tuple[int, int] | None:
        y, x = self.position
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
        start = self.position
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
        while parents[current] is not None and parents[current] != self.position:
            current = parents[current]

        if parents[current] is None:
            return None
        return current

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

    def _best_local_move(self) -> tuple[int, int] | None:
        neighbors = self._walkable_neighbors(self.position)
        if not neighbors:
            return None

        frontier_neighbors = [item for item in neighbors if self._is_frontier(item[0])]
        candidates = frontier_neighbors if frontier_neighbors else neighbors

        candidates.sort(
            key=lambda item: (
                self._turn_cost(self.direction, item[1]),
                -self._unknown_neighbor_count(item[0]),
            )
        )
        return candidates[0][0]

    def _unknown_neighbor_count(self, position: tuple[int, int]) -> int:
        y, x = position
        count = 0
        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            if self._in_bounds(ny, nx) and self._known_map[ny, nx] == self.UNKNOWN:
                count += 1
        return count

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

    def _direction_from_step(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> Direction:
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
        ci = order.index(current)
        ni = order.index(candidate)
        delta = (ni - ci) % 4

        if delta == 0:
            return 0
        if delta == 2:
            return 2
        return 1

    def _in_bounds(self, y: int, x: int) -> bool:
        if self._known_map is None:
            return False
        return 0 <= y < self._known_map.shape[0] and 0 <= x < self._known_map.shape[1]

    def _tile_at(self, position: tuple[int, int]) -> int:
        return int(self._known_map[position])

    def _is_traversable_known(self, position: tuple[int, int]) -> bool:
        tile = self._tile_at(position)
        if tile in (self.UNKNOWN, self.ALIEN, int(Tile.WALL)):
            return False
        return True

    def _is_observed_alien(self, position: tuple[int, int] | None) -> bool:
        if position is None:
            return False
        return position in self._observed_aliens

    