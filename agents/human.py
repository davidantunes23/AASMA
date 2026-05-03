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
    WAIT = 2


class HumanAgent:
    # Observation markers for cells the agent has not yet seen
    UNKNOWN = -1
    # Marker for cells containing observed alien positions
    ALIEN = -2
    # Radar marker: indicates alien proximity but not exact location
    RADAR_PING = -3
    # Noise ripple marker: visualizes sound event propagation
    NOISE_RIPPLE = -4
    # Marker for the player when concealed in a hiding spot
    HIDDEN = -5

    def __init__(self, start_pos: tuple, start_dir: Direction):
        # Current position in the map grid
        self.position = start_pos
        # Current facing direction
        self.direction = start_dir
        # Accumulated knowledge of the map from observations
        self._known_map: np.ndarray | None = None
        # Cached position of exit once discovered (None until found)
        self._known_exit: tuple[int, int] | None = None
        # Set of alien positions detected via radar or observation
        self._observed_aliens: set[tuple[int, int]] = set()
        # Whether the human is currently hidden in a hiding spot
        self.hidden: bool = False
        # Last radar threat level: CRITICAL (0-5 cells), CLOSE (6-10), NEAR (11-16), FAR (17+)
        self.last_radar_threat: str | None = None
        # Distance to alien from last radar ping
        self.last_radar_dist: int | None = None

    def _act(self, obs, radar_threat: str | None = None, radar_dist: int | None = None) -> tuple:
        # Update radar state with latest threat assessment
        if radar_threat is not None:
            self.last_radar_threat = radar_threat
            self.last_radar_dist = radar_dist
        # Build and refresh internal map knowledge from observation
        self._init_memory(obs)
        self._integrate_observation(obs)

        # PRIORITY 1: If hidden, remain hidden until threat subsides
        # Hidden state persists through multiple steps if threat remains high
        if self.hidden:
            if not self._should_keep_hiding(radar_threat):
                self.hidden = False
            else:
                return (Action.WAIT, self.direction)

        # PRIORITY 2: Exit pursuit is the primary goal once location is known
        # Navigate directly to exit without delay when it is visible
        if self._known_exit is not None:
            next_position = self._step_toward_target(self._known_exit)
            if next_position is not None:
                return (Action.WALK, self.direction)

        # PRIORITY 3: Enter hiding when threat level reaches CRITICAL (0-5 cells) or CLOSE (6-10) with no nearby exit
        # Seeking hiding spots is the third priority when exit is unavailable
        if self._should_hide_now(radar_threat):
            hiding_spot = self._get_closest_hiding_spot()
            if hiding_spot is not None:
                next_position = self._step_toward_target(hiding_spot)
                if next_position is not None:
                    if next_position != self.position:
                        # Move toward hiding spot
                        return (Action.WALK, self.direction)
                    else:
                        # Arrived at hiding spot, enter hidden state
                        self.hidden = True
                        return (Action.WAIT, self.direction)

        # PRIORITY 4: Exploration - map unknown territory using multiple strategies
        # First try to move to adjacent unknown cells to explore efficiently
        next_position = self._adjacent_unknown_step()
        # Then search for the nearest frontier of known/unknown boundary on FLOOR tiles
        if next_position is None:
            next_position = self._next_step_to_nearest_floor_frontier()
        # Expand search to any traversable frontier if no floor frontier found
        if next_position is None:
            next_position = self._next_step_to_nearest_frontier()
        # Fall back to best local move: prefer frontier cells with least rotation
        if next_position is None:
            next_position = self._best_local_move()

        # Safety check: do not move into observed alien positions
        if self._is_observed_alien(next_position):
            next_position = None

        # Update direction to face target if moving
        if next_position is not None and next_position != self.position:
            self.direction = self._direction_from_step(self.position, next_position)

        return (Action.WALK, self.direction)

    def _step_toward_target(self, target: tuple[int, int]) -> tuple[int, int] | None:
        # Use BFS to find next step toward target; updates direction to face it
        # Returns None if target is unreachable or current position is out of bounds
        next_position = self._bfs_next_step(lambda pos: pos == target)
        if next_position is not None and next_position != self.position:
            self.direction = self._direction_from_step(self.position, next_position)
        return next_position

    def _get_direction(self) -> Direction:
        return self.direction

    def _init_memory(self, obs: np.ndarray):
        # Initialize or reset map knowledge when map dimensions change
        # First observation triggers creation of full UNKNOWN grid
        if self._known_map is not None and self._known_map.shape == obs.shape:
            return
        self._known_map = np.full(obs.shape, self.UNKNOWN, dtype=np.int16)
        self._known_exit = None
        self._observed_aliens = set()

    def _integrate_observation(self, obs: np.ndarray):
        # Check if radar is currently active in observation
        radar_active = np.any(obs == self.RADAR_PING)
        
        # Record all visible world tiles (floors, walls, exits, hiding spots) in memory
        # Exclude markers (UNKNOWN, ALIEN, RADAR_PING, NOISE_RIPPLE) from storage
        visible_mask = (obs != self.UNKNOWN) & (obs != self.ALIEN) & (obs != self.RADAR_PING) & (obs != self.NOISE_RIPPLE)
        self._known_map[visible_mask] = obs[visible_mask]

        # Update hidden state by checking if current tile is a hiding spot
        # This state will be used by game.py to suppress noise generation
        self.hidden = self._tile_at(self.position) == int(Tile.HIDE)

        # When radar is active, mark current position as alien detection uncertainty zone
        # When radar is inactive, clear observed alien set (exact position no longer known)
        if radar_active:
            self._observed_aliens = {self.position}
        else:
            self._observed_aliens = set()

        # Record exit location in memory once discovered for quick reference
        ey, ex = np.where(self._known_map == int(Tile.EXIT))
        if len(ey) > 0:
            self._known_exit = (int(ey[0]), int(ex[0]))

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        return self._bfs_next_step(self._is_floor_frontier)

    def _adjacent_unknown_step(self) -> tuple[int, int] | None:
        # Collect all adjacent cells that are unknown and safe to explore
        y, x = self.position
        candidates: list[tuple[tuple[int, int], Direction]] = []

        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            candidate = (ny, nx)

            # Skip out-of-bounds cells
            if not self._in_bounds(ny, nx):
                continue
            # Skip cells with observed aliens
            if candidate in self._observed_aliens:
                continue
            # Only add truly unknown cells
            if self._known_map[ny, nx] != self.UNKNOWN:
                continue

            candidates.append((candidate, direction))

        if not candidates:
            return None

        # Sort by rotation cost to minimize unnecessary turning
        candidates.sort(key=lambda item: self._turn_cost(self.direction, item[1]))
        return candidates[0][0]

    def _next_step_to_nearest_frontier(self) -> tuple[int, int] | None:
        return self._bfs_next_step(self._is_frontier)

    def _bfs_next_step(self, is_target) -> tuple[int, int] | None:
        # Breadth-first search to find nearest cell matching predicate (is_target)
        # Returns the first step from current position toward the target
        start = self.position
        if not self._in_bounds(*start):
            return None

        # BFS explores all reachable cells, finding closest match by distance
        frontier = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while frontier:
            current = frontier.popleft()

            # Check if this cell matches target criteria (skip start position)
            if current != start and is_target(current):
                return self._first_step_from_path(current, parents)

            # Expand frontier to all walkable neighbors
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
        # Floor frontier: specifically FLOOR tiles adjacent to unknown territory
        # Preferred over general frontier for more structured exploration
        if self._tile_at(position) != int(Tile.FLOOR):
            return False
        return self._has_unknown_neighbor(position)

    def _is_frontier(self, position: tuple[int, int]) -> bool:
        # Frontier cells are traversable and adjacent to unknown territory
        if not self._is_traversable_known(position):
            return False
        return self._has_unknown_neighbor(position)

    def _has_unknown_neighbor(self, position: tuple[int, int]) -> bool:
        # Check if any adjacent cell is marked UNKNOWN (unexplored)
        y, x = position
        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            if self._in_bounds(ny, nx) and self._known_map[ny, nx] == self.UNKNOWN:
                return True
        return False

    def _best_local_move(self) -> tuple[int, int] | None:
        # Choose best adjacent move when no frontier cells are reachable
        neighbors = self._walkable_neighbors(self.position)
        if not neighbors:
            return None

        # Prioritize frontier neighbors (cells adjacent to unknown territory)
        frontier_neighbors = [item for item in neighbors if self._is_frontier(item[0])]
        candidates = frontier_neighbors if frontier_neighbors else neighbors

        # Sort by: (1) rotation cost to minimize turning, (2) unknown neighbors to maximize exploration
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
        # Return all adjacent cells that are safe and known to be traversable
        # Each tuple contains (position, direction_to_reach_it)
        y, x = position
        neighbors: list[tuple[tuple[int, int], Direction]] = []

        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            dy, dx = self._direction_delta(direction)
            ny, nx = y + dy, x + dx
            candidate = (ny, nx)

            # Skip out-of-bounds cells
            if not self._in_bounds(ny, nx):
                continue
            # Skip cells with observed aliens
            if self._is_observed_alien(candidate):
                continue
            # Skip non-traversable cells (walls, unknown, etc.)
            if not self._is_traversable_known(candidate):
                continue

            neighbors.append((candidate, direction))

        return neighbors

    def _direction_from_step(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> Direction:
        # Convert a step (start -> end) into its corresponding direction
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
        # If step is invalid, keep current direction
        return self.direction

    def _direction_delta(self, direction: Direction) -> tuple[int, int]:
        # Return (dy, dx) for a given direction
        if direction == Direction.NORTH:
            return (-1, 0)
        if direction == Direction.EAST:
            return (0, 1)
        if direction == Direction.SOUTH:
            return (1, 0)
        return (0, -1)

    def _turn_cost(self, current: Direction, candidate: Direction) -> int:
        # Calculate rotational cost: 0 for no turn, 1 for 90°, 2 for 180°
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        ci = order.index(current)
        ni = order.index(candidate)
        delta = (ni - ci) % 4

        # No turn needed
        if delta == 0:
            return 0
        # 180° turn (highest cost)
        if delta == 2:
            return 2
        # 90° turn (low cost)
        return 1

    def _in_bounds(self, y: int, x: int) -> bool:
        # Check if position is within known map boundaries
        if self._known_map is None:
            return False
        return 0 <= y < self._known_map.shape[0] and 0 <= x < self._known_map.shape[1]

    def _tile_at(self, position: tuple[int, int]) -> int:
        # Retrieve tile ID at a given position from known map
        return int(self._known_map[position])

    def _is_traversable_known(self, position: tuple[int, int]) -> bool:
        # Check if a known tile is passable (not wall, not alien, not unexplored)
        tile = self._tile_at(position)
        if tile in (self.UNKNOWN, self.ALIEN, int(Tile.WALL)):
            return False
        return True

    def _is_observed_alien(self, position: tuple[int, int] | None) -> bool:
        # Check if position contains a detected alien (from radar or prior observation)
        if position is None:
            return False
        return position in self._observed_aliens

    def _is_hidden(self) -> bool:
        # Query current hidden state
        return self.hidden

    def _should_hide_now(self, radar_threat: str | None) -> bool:
        # Decide whether to begin seeking a hiding spot based on threat level
        if radar_threat is None:
            return False

        # Always hide if threat is CRITICAL (alien within 5 cells)
        if radar_threat == "CRITICAL":
            return True

        # Hide on CLOSE threat (6-10 cells away) unless exit is very near (within 15 cells)
        if radar_threat == "CLOSE":
            if self._known_exit is not None:
                dist_to_exit = abs(self._known_exit[0] - self.position[0]) + abs(self._known_exit[1] - self.position[1])
                # If exit is close, run for it instead of hiding
                if dist_to_exit <= 15:
                    return False
            return True

        return False

    def _should_keep_hiding(self, radar_threat: str | None) -> bool:
        # Determine if current threat level justifies remaining hidden
        # Use current radar threat if available, otherwise fall back to last known threat
        effective_threat = radar_threat if radar_threat is not None else self.last_radar_threat
        # Stay hidden only for CRITICAL and CLOSE threat levels
        return effective_threat in {"CRITICAL", "CLOSE"}

    def _get_known_hiding_spots(self) -> list[tuple[int, int]]:
        # Return all hiding spot tiles discovered in known map
        if self._known_map is None:
            return []
        
        hiding_spots = []
        hy, hx = np.where(self._known_map == int(Tile.HIDE))
        for y, x in zip(hy, hx):
            hiding_spots.append((int(y), int(x)))
        return hiding_spots
    
    def _get_closest_hiding_spot(self) -> tuple[int, int] | None:
        # Find the nearest reachable hiding spot using BFS distance
        hiding_spots = self._get_known_hiding_spots()
        if not hiding_spots:
            return None

        # BFS ensures we return a spot that is actually reachable via walkable path
        start = self.position
        if not self._in_bounds(*start):
            return None
        
        # BFS explores from current position until a hiding spot is found
        frontier = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        
        while frontier:
            current = frontier.popleft()

            # Return first hiding spot encountered (guaranteed closest)
            if current in hiding_spots:
                return current
            
            # Expand to all walkable neighbors
            for neighbor, _ in self._walkable_neighbors(current):
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                frontier.append(neighbor)
        
        return None

    