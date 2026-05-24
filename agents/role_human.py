from collections import deque
from dataclasses import dataclass  # [CHANGE] imported for CoordMessage
from enum import Enum, auto        # [CHANGE] imported for CoordType

import numpy as np

from agents.base import BaseHumanAgent, Direction, TeamRole
from map_generator import Tile


# ── Coord communication types ─────────────────────────────────────────────────
# [CHANGE] Added CoordType and CoordMessage here (inline, no separate file needed).
# Justification: keeps the module self-contained; role manager only needs to import
# this one file to drive the message bus.

class CoordType(Enum):
    MISSION = auto()   # a new mission tile was discovered
    EXIT    = auto()   # the exit tile was discovered


@dataclass
class CoordMessage:
    coord_type: CoordType
    pos: tuple[int, int]
    sender_id: int   # agent index — used to skip re-delivering to the sender


class RoleHumanAgent(BaseHumanAgent):
    """Role-aware human agent. Behaves like `HumanAgent` but exposes
    `team_role` and respects masking operators (e.g., WORKER disables hiding).
    """

    UNKNOWN      = -1
    ALIEN        = -2
    RADAR_PING   = -3
    NOISE_RIPPLE = -4

    def __init__(
        self,
        start_pos: tuple[int, int],
        start_dir: Direction = Direction.NORTH,
        view_length: int = 6,
    ):
        self.pos          = start_pos
        self.direction    = start_dir
        self.view_length  = view_length
        self.hidden: bool = False
        self.exit_open: bool = False
        self.team_role: TeamRole | None = TeamRole.NONE
        self.last_radar_threat: str | None = None
        self.last_radar_dist:   int | None = None
        self._known_map:  np.ndarray | None       = None
        self._known_exit: tuple[int, int] | None  = None
        self._observed_aliens: set[tuple[int, int]] = set()
        self.made_loud_noise: bool = False

        # [CHANGE] agent_id: must be set by role manager after construction.
        # Justification: needed so outbox messages carry the sender's identity,
        # allowing the bus to skip re-delivering a message to its own sender.
        self.agent_id: int = 0

        # [CHANGE] mission_positions now stays as the live list consumed by
        # _nearest_mission(), but is populated BOTH externally (role manager
        # injection, unchanged) AND via receive_coords() for new discoveries.
        # No behaviour change — existing callers still work.
        self.mission_positions: list[tuple[int, int]] = []

        # [CHANGE] _known_mission_coords: a set mirror of mission_positions used
        # for O(1) duplicate checks in receive_coords() and _integrate_observation().
        # Justification: list.count / list.__contains__ is O(n); set is O(1).
        self._known_mission_coords: set[tuple[int, int]] = set()

        # [CHANGE] _outbox: collects CoordMessages produced during observe().
        # The role manager calls flush_outbox() each step and fans messages out.
        # Justification: decouples discovery from delivery; agents never talk
        # directly to each other, preserving the existing single-agent interface.
        self._outbox: list[CoordMessage] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def observe(
        self,
        obs: np.ndarray,
        radar_threat: str | None = None,
        radar_dist:   int | None = None,
    ) -> None:
        if radar_threat is not None:
            self.last_radar_threat = radar_threat
            self.last_radar_dist   = radar_dist
        self._init_memory(obs)
        self._integrate_observation(obs)

    def step(
        self,
        _player_pos: tuple[int, int],
        _heard_pos:  tuple[int, int] | None = None,
        _step_num:   int = 0,
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

        if self.team_role == TeamRole.DECOY:
            return self._decoy_step()
        if self.team_role == TeamRole.RUNNER:
            return self._runner_step()
        if self.team_role == TeamRole.WORKER:
            return self._worker_step()

        # PRIORITY 2: Run to exit once known
        if self.exit_open and self._known_exit is not None:
            nxt = self._step_toward_target(self._known_exit)
            if nxt is not None and nxt != self.pos:
                self.pos    = nxt
                self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                return self.pos

        # PRIORITY 3: Hide when threatened
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
            self.pos       = nxt

        self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
        return self.pos

    def reset(self, start_pos: tuple[int, int] | None = None) -> None:
        if start_pos is not None:
            self.pos = start_pos
        self._known_map           = None
        self._known_exit          = None
        self._observed_aliens     = set()
        self.hidden               = False
        self.last_radar_threat    = None
        self.last_radar_dist      = None
        # [CHANGE] also reset coord-bus state on episode reset.
        # Justification: stale outbox messages or mission sets from a previous
        # episode would pollute the new one.
        self._outbox.clear()
        self.mission_positions        = []
        self._known_mission_coords    = set()

    # ── Coord message bus ─────────────────────────────────────────────────────

    # [CHANGE] flush_outbox: called by the role manager after every observe()
    # to collect messages this agent produced and broadcast them to others.
    # Justification: pull-based delivery keeps agents passive; the role manager
    # drives all coordination, consistent with the existing design.
    def flush_outbox(self) -> list[CoordMessage]:
        msgs = self._outbox.copy()
        self._outbox.clear()
        return msgs

    # [CHANGE] receive_coords: injects discovered coords from other agents.
    # Each coord type is handled separately:
    #   MISSION → appended to mission_positions + written into _known_map so
    #             BFS can path to the tile even before the agent sees it.
    #   EXIT    → sets _known_exit + writes tile into _known_map for the same
    #             reason. Only accepted if we don't already know the exit
    #             (first-write-wins; our own observation always takes precedence
    #             because _integrate_observation runs before receive_coords).
    def receive_coords(self, messages: list[CoordMessage]) -> None:
        for msg in messages:
            if msg.coord_type == CoordType.MISSION:
                if msg.pos not in self._known_mission_coords:
                    self._known_mission_coords.add(msg.pos)
                    self.mission_positions.append(msg.pos)
                    if self._known_map is not None and self._in_bounds(*msg.pos):
                        if self._known_map[msg.pos] == self.UNKNOWN:
                            self._known_map[msg.pos] = int(Tile.MISSION)

            elif msg.coord_type == CoordType.EXIT:
                if self._known_exit is None:
                    self._known_exit = msg.pos
                    if self._known_map is not None and self._in_bounds(*msg.pos):
                        self._known_map[msg.pos] = int(Tile.EXIT)

    # [CHANGE] remove_mission: called by the role manager when MissionManager
    # marks a tile complete. Keeps mission_positions and the set in sync so the
    # WORKER and DECOY stop targeting or fleeing from a finished tile.
    def remove_mission(self, pos: tuple[int, int]) -> None:
        self._known_mission_coords.discard(pos)
        if pos in self.mission_positions:
            self.mission_positions.remove(pos)

    # ── Observation integration ───────────────────────────────────────────────

    def _init_memory(self, obs: np.ndarray) -> None:
        if self._known_map is not None and self._known_map.shape == obs.shape:
            return
        self._known_map       = np.full(obs.shape, self.UNKNOWN, dtype=np.int16)
        self._known_exit      = None
        self._observed_aliens = set()

    def _integrate_observation(self, obs: np.ndarray) -> None:
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

        # [CHANGE] Exit discovery now also queues a CoordMessage the first time.
        # Justification: previously _known_exit was silently updated; now the
        # discovery is broadcast so the RUNNER can share it with WORKER/DECOY
        # (and vice-versa) without a shared map.
        ey, ex = np.where(self._known_map == int(Tile.EXIT))
        if len(ey) > 0:
            found_exit = (int(ey[0]), int(ex[0]))
            if self._known_exit is None:
                # First sighting — emit broadcast
                self._outbox.append(CoordMessage(
                    coord_type=CoordType.EXIT,
                    pos=found_exit,
                    sender_id=self.agent_id,
                ))
            self._known_exit = found_exit

        # [CHANGE] Mission discovery: scan the *fresh* obs (not full known_map)
        # so we only emit a message when the tile enters the FOV for the first
        # time. Using obs instead of _known_map prevents re-broadcasting tiles
        # that were written in by receive_coords() in a prior step.
        # Justification: avoids duplicate messages flooding the outbox every step.
        my, mx = np.where(obs == int(Tile.MISSION))
        for y, x in zip(my.tolist(), mx.tolist()):
            pos = (int(y), int(x))
            if pos not in self._known_mission_coords:
                self._known_mission_coords.add(pos)
                self.mission_positions.append(pos)
                self._outbox.append(CoordMessage(
                    coord_type=CoordType.MISSION,
                    pos=pos,
                    sender_id=self.agent_id,
                ))

    # ── Navigation ────────────────────────────────────────────────────────────

    def _step_toward_target(self, target: tuple[int, int]) -> tuple[int, int] | None:
        nxt = self._bfs_next_step(lambda pos: pos == target)
        if nxt is not None and nxt != self.pos:
            self.direction = self._direction_from_step(self.pos, nxt)
        return nxt

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        return self._bfs_next_step(self._is_floor_frontier)

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
        # [CHANGE] converted hiding_spots to a set for O(1) membership check
        # inside the BFS loop. Justification: original used a list, so `current
        # in hiding_spots` was O(n) per BFS node — costly on large maps.
        hiding_set = set(hiding_spots)
        frontier = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            current = frontier.popleft()
            if current in hiding_set:
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
        # WORKER role disables hiding (masking operator)
        if self.team_role == TeamRole.WORKER:
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

    # ── WORKER helpers ────────────────────────────────────────────────────────

    def _worker_step(self) -> tuple[int, int]:
        target = self._nearest_mission()
        if target is not None and self.pos != target:
            nxt = self._step_toward_target(target)
            if nxt is not None and nxt != self.pos:
                self.direction = self._direction_from_step(self.pos, nxt)
                self.pos       = nxt
                self.hidden    = False
                return self.pos
        if target == self.pos:
            return self.pos
        # No mission known: explore
        nxt = (
            self._adjacent_unknown_step()
            or self._next_step_to_nearest_frontier()
            or self._best_local_move()
        )
        if nxt and nxt != self.pos:
            self.direction = self._direction_from_step(self.pos, nxt)
            self.pos       = nxt
        self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
        return self.pos

    def _nearest_mission(self) -> tuple[int, int] | None:
        if not self.mission_positions:
            return None
        return min(
            self.mission_positions,
            key=lambda m: abs(m[0] - self.pos[0]) + abs(m[1] - self.pos[1]),
        )

    # ── DECOY helpers ─────────────────────────────────────────────────────────

    def _decoy_step(self) -> tuple[int, int]:
        # CRITICAL: alien is adjacent — pure survival, no noise.
        # Justification: noise here just confirms a position the alien
        # already knows. Only action is to flee to the nearest hide spot.
        if self.last_radar_threat == "CRITICAL":
            self.made_loud_noise = False
            spot = self._get_closest_hiding_spot()
            if spot is not None:
                nxt = self._step_toward_target(spot)
                if nxt is not None and nxt != self.pos:
                    self.direction = self._direction_from_step(self.pos, nxt)
                    self.pos       = nxt
                    self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                    return self.pos
            return self.pos

        # CLOSE: alien is nearby — hide silently, never make noise.
        # Justification: too dangerous to bait; reaching cover is the only
        # priority. No noise even if already hidden — hidden means silent.
        if self.last_radar_threat == "CLOSE":
            self.made_loud_noise = False
            if self.hidden:
                # Already in cover — stay put and wait it out.
                return self.pos
            spot = self._get_closest_hiding_spot()
            if spot is not None:
                nxt = self._step_toward_target(spot)
                if nxt is not None and nxt != self.pos:
                    self.direction = self._direction_from_step(self.pos, nxt)
                    self.pos       = nxt
                    self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                    return self.pos
            return self.pos

        # NEAR: alien is approaching but Decoy is still safe — optimal bait window.
        # Justification: alien is close enough to hear the noise and be drawn
        # toward the Decoy's area (away from missions), but far enough that the
        # Decoy can reposition before it arrives.
        # Noise only allowed when NOT hidden — hidden agents cannot make noise.
        if self.last_radar_threat == "NEAR":
            self.made_loud_noise = not self.hidden
            # Reposition to farthest-from-missions tile regardless of noise,
            # so the alien arrives at an empty spot (decoy-and-dodge pattern).
            far_tile = self._farthest_from_missions()
            if far_tile is not None:
                nxt = self._step_toward_target(far_tile)
                if nxt is not None and nxt != self.pos:
                    self.direction = self._direction_from_step(self.pos, nxt)
                    self.pos       = nxt
                    self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                    return self.pos
            return self.pos

        # FAR or no threat — reposition silently to attraction zone.
        # Justification: use the quiet window to get into a good structural
        # position so the next NEAR trigger happens as far from missions as
        # possible, maximising the distraction value of the next noise call.
        self.made_loud_noise = False
        far_tile = self._farthest_from_missions()
        if far_tile is not None and far_tile != self.pos:
            nxt = self._step_toward_target(far_tile)
            if nxt is not None and nxt != self.pos:
                self.direction = self._direction_from_step(self.pos, nxt)
                self.pos       = nxt
                self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                return self.pos

        # Fallback: explore outward to expand known map and find better
        # farthest-from-missions candidates in unexplored regions.
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
            self.pos       = nxt

        self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
        return self.pos

    def _farthest_from_missions(self) -> tuple[int, int] | None:
        if not self.mission_positions or self._known_map is None:
            return None
        H, W = self._known_map.shape
        best       = None
        best_score = -1
        for y in range(H):
            for x in range(W):
                if not self._is_traversable_known((y, x)):
                    continue
                min_d = min(
                    abs(y - my) + abs(x - mx)
                    for my, mx in self.mission_positions
                )
                if min_d > best_score:
                    best_score = min_d
                    best       = (y, x)
        return best

    # ── RUNNER helpers ────────────────────────────────────────────────────────

    def _runner_step(self) -> tuple[int, int]:
        # PRIORITY 1: Exit open and known → escape if safe
        if self.exit_open and self._known_exit is not None:
            if self.last_radar_threat is None or self.last_radar_threat == "FAR":
                nxt = self._step_toward_target(self._known_exit)
                if nxt is not None and nxt != self.pos:
                    self.direction = self._direction_from_step(self.pos, nxt)
                    self.pos       = nxt
                    self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                    return self.pos
                return self.pos  # already at exit or blocked
            # Threat too high: hide and wait
            if self.hidden:
                if self._should_keep_hiding():
                    return self.pos
                else:
                    self.hidden = False
            spot = self._get_closest_hiding_spot()
            if spot is not None:
                nxt = self._step_toward_target(spot)
                if nxt is not None and nxt != self.pos:
                    self.direction = self._direction_from_step(self.pos, nxt)
                    self.pos       = nxt
                    self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                    return self.pos
            return self.pos

        # PRIORITY 2: Exit known but locked → stage near it
        if self._known_exit is not None:
            if self.last_radar_threat in {"CRITICAL", "CLOSE"}:
                spot = self._get_closest_hiding_spot()
                if spot is not None:
                    nxt = self._step_toward_target(spot)
                    if nxt is not None and nxt != self.pos:
                        self.direction = self._direction_from_step(self.pos, nxt)
                        self.pos       = nxt
                        self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                        return self.pos
                return self.pos
            nxt = self._step_toward_exit_area()
            if nxt is not None and nxt != self.pos:
                self.direction = self._direction_from_step(self.pos, nxt)
                self.pos       = nxt
            self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
            return self.pos

        # PRIORITY 3: Exit not found → explore to find it
        # [CHANGE] RUNNER exploration now prefers floor frontiers over generic
        # frontiers first, then falls back to generic, then best local.
        # Justification: floor frontiers are more likely to lead into open rooms
        # where the exit tile is placed, giving the RUNNER a better exploration
        # strategy than random frontier walking.
        nxt = self._adjacent_unknown_step()
        if nxt is None:
            nxt = self._next_step_to_nearest_floor_frontier()
        if nxt is None:
            nxt = self._next_step_to_nearest_frontier()
        if nxt is None:
            nxt = self._best_local_move()
        if nxt is not None and nxt != self.pos:
            self.direction = self._direction_from_step(self.pos, nxt)
            self.pos       = nxt
        self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
        return self.pos

    def _step_toward_exit_area(self) -> tuple[int, int] | None:
        """Move toward a cell adjacent to the exit — not onto it while locked."""
        if self._known_exit is None:
            return None
        ey, ex = self._known_exit
        staging = []
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            ny, nx = ey + dy, ex + dx
            if self._in_bounds(ny, nx) and self._is_traversable_known((ny, nx)):
                staging.append((ny, nx))
        if not staging:
            return None
        staging.sort(key=lambda p: abs(p[0] - self.pos[0]) + abs(p[1] - self.pos[1]))
        target = staging[0]
        if self.pos == target:
            return self.pos
        return self._step_toward_target(target)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _is_observed_alien(self, position: tuple[int, int] | None) -> bool:
        if position is None:
            return False
        return position in self._observed_aliens

    def _in_bounds(self, y: int, x: int) -> bool:
        if self._known_map is None:
            return False
        return 0 <= y < self._known_map.shape[0] and 0 <= x < self._known_map.shape[1]

    def _tile_at(self, position: tuple[int, int]) -> int:
        if self._known_map is None:
            return self.UNKNOWN
        return int(self._known_map[position])

    # ── Direction helpers ─────────────────────────────────────────────────────

    def _direction_from_step(self, start: tuple[int, int], end: tuple[int, int]) -> Direction:
        dy = end[0] - start[0]
        dx = end[1] - start[1]
        if dy == -1 and dx == 0:
            return Direction.NORTH
        if dy == 0  and dx == 1:
            return Direction.EAST
        if dy == 1  and dx == 0:
            return Direction.SOUTH
        if dy == 0  and dx == -1:
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
