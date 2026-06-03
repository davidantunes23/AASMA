"""Role-aware human agent.

Each agent is assigned one of three roles by the role manager:
  WORKER  — navigates to uncompleted mission tiles and dwells on them.
  DECOY   — repositions away from missions and emits deliberate noise to draw
             the alien away from workers.
  RUNNER  — stages near the exit and escapes as soon as it opens; never
             completes missions itself.

All roles share the same survival priority stack (hide when threatened) and
converge to exit-seeking once all missions are done.
"""
from collections import deque

import numpy as np

from agents.base import BaseHumanAgent, Direction, TeamRole
from agents.coord_bus import CoordMessage, CoordType
from map_generator import Tile


class RoleHumanAgent(BaseHumanAgent):
    """Role-aware human agent built on top of BFS navigation and partial map knowledge.

    The simulation calls ``observe()`` then ``step()`` each turn.
    Role assignment is done externally by the role manager via ``agent.team_role``.
    """

    _next_agent_id = 0  # class-level counter so each instance gets a unique ID

    # Special sentinel values embedded in the observation array by the simulation.
    UNKNOWN      = -1   # cell not yet seen by this agent's cone FOV
    ALIEN        = -2   # alien position marker (injected when alien is visible)
    RADAR_PING   = -3   # radar proximity alert tile
    NOISE_RIPPLE = -4   # sound ripple marker (ignored for map building)

    # Decoy noise burst parameters.
    LOUD_NOISE_DURATION_STEPS = 10   # how many steps a loud-noise burst lasts
    LOUD_NOISE_COOLDOWN_STEPS = 20   # minimum quiet steps between bursts

    def __init__(
        self,
        start_pos: tuple[int, int],
        start_dir: Direction = Direction.NORTH,
        view_length: int = 6,
    ):
        super().__init__(pos=start_pos, direction=start_dir, view_length=view_length)
        self.agent_id: int = RoleHumanAgent._next_agent_id
        RoleHumanAgent._next_agent_id += 1

        self.team_role: TeamRole | None = TeamRole.RUNNER  # default until role manager assigns
        self.last_radar_threat: str | None = None  # most recent radar band (CRITICAL/CLOSE/NEAR/FAR)
        self.last_radar_dist:   int | None = None  # topology distance to alien at last radar tick
        self._known_map:  np.ndarray | None       = None   # tile map built from cone observations
        self._known_exit: tuple[int, int] | None  = None   # exit position once seen or received
        self._observed_aliens: set[tuple[int, int]] = set()  # alien positions seen this step
        self.made_loud_noise: bool = False         # True this step if decoy is emitting a signal
        self.loud_noise_pos: tuple[int, int] | None = None  # position of the ongoing noise burst
        self._loud_noise_steps_left: int = 0       # remaining steps in current burst
        self._loud_noise_cooldown: int = (
            self.LOUD_NOISE_COOLDOWN_STEPS if self.team_role == TeamRole.DECOY else 0
        )

        # Mission counters kept in sync by the simulation via attribute writes.
        self.missions_total: int = 0       # total missions placed on the map this episode
        self.missions_remaining: int = 0   # missions not yet completed

        # Known mission positions, updated from cone observations and teammate messages.
        self.mission_positions: list[tuple[int, int]] = []

        # Missions currently being worked by a teammate (avoids double-assignment).
        self.active_mission_positions: set[tuple[int, int]] = set()

        # Mission tile this worker is currently heading toward.
        self.current_mission: tuple[int, int] | None = None

        # Fast lookup to avoid broadcasting the same mission tile twice.
        self._known_mission_coords: set[tuple[int, int]] = set()

        # Persistent mission discovery memory — not cleared when a mission completes,
        # so the runner can tell whether all missions have been found yet.
        self._seen_mission_coords: set[tuple[int, int]] = set()

        # Outbound coord messages queued during observe(), sent via flush_outbox().
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
            return self.pos  # already on open exit — stay and let simulation register escape

        # PRIORITY 1: Stay hidden while threat is high.
        if self.hidden:
            if not self._should_keep_hiding():
                self.hidden = False
            else:
                return self.pos

        # PRIORITY 2: Hide when threatened (shared survival logic, overrides all roles).
        # Silence the decoy noise burst before hiding so the alien isn't baited
        # toward cover the decoy is about to enter.
        if self._should_hide_now():
            if self.team_role == TeamRole.DECOY:
                self._update_decoy_loud_noise(False)
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

        # PRIORITY 3: If all missions are done, prioritize exit/search regardless of role.
        if self.missions_remaining == 0:
            if self._known_exit is not None:
                nxt = self._step_toward_target(self._known_exit)
                if nxt is not None and nxt != self.pos:
                    self.pos = nxt
                    self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                    return self.pos
            nxt = self._next_step_to_nearest_frontier() or self._best_local_move()
            if nxt is not None and nxt != self.pos:
                self.direction = self._direction_from_step(self.pos, nxt)
                self.pos = nxt
            self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
            return self.pos

        # PRIORITY 4: Delegate to role-specific behaviour.
        if self.team_role == TeamRole.DECOY:
            return self._decoy_step()
        if self.team_role == TeamRole.RUNNER:
            return self._runner_step()
        if self.team_role == TeamRole.WORKER:
            return self._worker_step()

        # Fallback (no role assigned): run to exit if open, otherwise explore.
        if self.exit_open and self._known_exit is not None:
            nxt = self._step_toward_target(self._known_exit)
            if nxt is not None and nxt != self.pos:
                self.pos    = nxt
                self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
                return self.pos

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
        self._outbox.clear()
        self._nav_path            = []
        self._nav_target          = None
        self._steps_on_path       = 0
        self.mission_positions        = []
        self._known_mission_coords    = set()
        self._seen_mission_coords     = set()
        self.active_mission_positions = set()
        self.current_mission          = None
        self.loud_noise_pos           = None
        self._loud_noise_steps_left   = 0
        self._loud_noise_cooldown     = 0

    # ── Coord message bus ─────────────────────────────────────────────────────

    def flush_outbox(self) -> list[CoordMessage]:
        """Return and clear messages queued during the latest observe() call."""
        msgs = self._outbox.copy()
        self._outbox.clear()
        return msgs

    def receive_coords(self, messages: list[CoordMessage]) -> None:
        """Merge coordinates received from teammates via the simulation relay."""
        for msg in messages:
            if msg.coord_type == CoordType.MISSION:
                if msg.pos not in self._known_mission_coords:
                    self._known_mission_coords.add(msg.pos)
                    self._seen_mission_coords.add(msg.pos)
                    self.mission_positions.append(msg.pos)
                    if self._known_map is not None and self._in_bounds(*msg.pos):
                        if self._known_map[msg.pos] == self.UNKNOWN:
                            self._known_map[msg.pos] = int(Tile.MISSION)

            elif msg.coord_type == CoordType.MISSION_ACTIVE:
                self.active_mission_positions.add(msg.pos)

            elif msg.coord_type == CoordType.MISSION_DONE:
                self.remove_mission(msg.pos)

            elif msg.coord_type == CoordType.EXIT:
                if self._known_exit is None:
                    self._known_exit = msg.pos
                    if self._known_map is not None and self._in_bounds(*msg.pos):
                        self._known_map[msg.pos] = int(Tile.EXIT)

    def remove_mission(self, pos: tuple[int, int]) -> None:
        """Drop a completed mission from all tracking structures."""
        self._known_mission_coords.discard(pos)
        self.active_mission_positions.discard(pos)
        if self.current_mission == pos:
            self.current_mission = None
        if pos in self.mission_positions:
            self.mission_positions.remove(pos)

    # ── Observation integration ───────────────────────────────────────────────

    def _init_memory(self, obs: np.ndarray) -> None:
        """Allocate the known map on the first observation of a new episode."""
        if self._known_map is not None and self._known_map.shape == obs.shape:
            return
        self._known_map       = np.full(obs.shape, self.UNKNOWN, dtype=np.int16)
        self._known_exit      = None
        self._observed_aliens = set()
        self._nav_path        = []
        self._nav_target      = None
        self._steps_on_path   = 0

    def _integrate_observation(self, obs: np.ndarray) -> None:
        """Copy visible tile IDs into the known map and broadcast new discoveries."""
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

        # Broadcast the exit the first time it enters the FOV.
        ey, ex = np.where(self._known_map == int(Tile.EXIT))
        if len(ey) > 0:
            found_exit = (int(ey[0]), int(ex[0]))
            if self._known_exit is None:
                self._outbox.append(CoordMessage(
                    coord_type=CoordType.EXIT,
                    pos=found_exit,
                    sender_id=self.agent_id,
                ))
            self._known_exit = found_exit

        # Broadcast each mission tile the first time it is observed.
        my, mx = np.where(obs == int(Tile.MISSION))
        for y, x in zip(my.tolist(), mx.tolist()):
            pos = (int(y), int(x))
            if pos not in self._known_mission_coords:
                self._known_mission_coords.add(pos)
                self._seen_mission_coords.add(pos)
                self.mission_positions.append(pos)
                self._outbox.append(CoordMessage(
                    coord_type=CoordType.MISSION,
                    pos=pos,
                    sender_id=self.agent_id,
                ))

    # ── Navigation ────────────────────────────────────────────────────────────

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        """BFS next step toward the nearest FLOOR cell adjacent to unknown space."""
        return self._bfs_next_step(self._is_floor_frontier)

    def _adjacent_unknown_step(self) -> tuple[int, int] | None:
        """Prefer stepping directly into an unknown cell if one is reachable.
        Tie-breaks by turn cost so the agent doesn't zig-zag unnecessarily.
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
        preferring frontier cells and minimising turn cost to avoid oscillation.
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

    # ── WORKER helpers ────────────────────────────────────────────────────────

    def _worker_step(self) -> tuple[int, int]:
        """Navigate to the nearest unclaimed mission tile and dwell on it."""
        target = self._nearest_mission()
        if target is not None and self.pos != target:
            nxt = self._step_toward_target(target)
            if nxt is not None and nxt != self.pos:
                self.direction = self._direction_from_step(self.pos, nxt)
                self.pos       = nxt
                self.hidden    = False
                return self.pos
        if target == self.pos:
            return self.pos  # dwelling on the mission tile — simulation tracks progress

        # No reachable mission: explore to potentially discover new ones.
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
        """Return this worker's assigned mission, or the nearest uncontested one."""
        if self.current_mission is not None:
            return self.current_mission
        # Exclude missions already being serviced by a teammate.
        available = [m for m in self.mission_positions if m not in self.active_mission_positions]
        if not available:
            return None
        return min(
            available,
            key=lambda m: abs(m[0] - self.pos[0]) + abs(m[1] - self.pos[1]),
        )

    # ── DECOY helpers ─────────────────────────────────────────────────────────

    def _update_decoy_loud_noise(self, can_start_noise: bool) -> bool:
        """Advance the loud-noise state machine. Returns True if a new burst started."""
        started_now = False
        if self._loud_noise_cooldown > 0:
            self._loud_noise_cooldown -= 1

        # Cannot start noise from inside a hide tile — that would reveal the hiding spot.
        if self._tile_at(self.pos) == int(Tile.HIDE):
            can_start_noise = False

        if self._loud_noise_steps_left == 0 and self._loud_noise_cooldown == 0 and can_start_noise:
            self._loud_noise_steps_left = self.LOUD_NOISE_DURATION_STEPS
            started_now = True
            self.loud_noise_pos = self.pos

        if self._loud_noise_steps_left > 0:
            self.made_loud_noise = True
            self._loud_noise_steps_left -= 1
            if self._loud_noise_steps_left == 0:
                self._loud_noise_cooldown = self.LOUD_NOISE_COOLDOWN_STEPS
        else:
            self.made_loud_noise = False
            self.loud_noise_pos = None
        return started_now

    def _decoy_step(self) -> tuple[int, int]:
        # CRITICAL and CLOSE are handled by PRIORITY 2 in step() — noise is
        # silenced there before seeking cover. _decoy_step() is only reached
        # when the threat is NEAR, FAR, or None.

        # NEAR: optimal bait window — alien is close enough to hear and be drawn
        # away from missions but far enough that the decoy can reposition first.
        if self.last_radar_threat == "NEAR":
            if self._update_decoy_loud_noise(not self.hidden):
                return self.pos
            # Move to farthest-from-missions tile so the alien arrives at empty space.
            far_tile = self._farthest_from_missions()
            if far_tile is not None:
                nxt = self._step_toward_target(far_tile)
                if nxt is not None and nxt != self.pos:
                    self.direction = self._direction_from_step(self.pos, nxt)
                    self.pos       = nxt
            self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
            return self.pos

        # FAR or no threat — preemptively reposition and optionally emit noise.
        if self.last_radar_threat == "FAR":
            if self._update_decoy_loud_noise(not self.hidden):
                return self.pos
        far_tile = self._farthest_from_missions()
        if far_tile is not None and far_tile != self.pos:
            nxt = self._step_toward_target(far_tile)
            if nxt is not None and nxt != self.pos:
                self.direction = self._direction_from_step(self.pos, nxt)
                self.pos       = nxt
            self.hidden    = bool(self._tile_at(self.pos) == int(Tile.HIDE))
            return self.pos

        # Fallback: explore to discover better far-from-missions candidates.
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
        """Find the known traversable cell with the greatest min-distance to any mission.
        HIDE tiles are excluded — the decoy should be visible to attract the alien.
        """
        if not self.mission_positions or self._known_map is None:
            return None
        H, W = self._known_map.shape
        best       = None
        best_score = -1
        for y in range(H):
            for x in range(W):
                if not self._is_traversable_known((y, x)):
                    continue
                if self._tile_at((y, x)) == int(Tile.HIDE):
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
        """Stage near the exit and escape as soon as it opens; explore until then."""
        # Determine whether there are still undiscovered missions on the map.
        # The runner avoids staging at the exit while missions are unknown.
        if self.missions_total > 0:
            missing_missions = (
                not self.exit_open
                and len(self._seen_mission_coords) < self.missions_total
            )
        else:
            # Fallback when total mission count is unavailable from the simulation.
            missing_missions = (
                not self.exit_open
                and self.missions_remaining > 0
                and self.missions_remaining > len(self._known_mission_coords)
            )

        # PRIORITY 1: Exit known but locked and all missions accounted for →
        # stage in the cell adjacent to the exit ready to sprint through.
        if self._known_exit is not None and not missing_missions:
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
            if nxt is None or nxt == self.pos:
                # Path to exit area unknown — explore to fill the gap.
                nxt = (
                    self._adjacent_unknown_step()
                    or self._next_step_to_nearest_floor_frontier()
                    or self._next_step_to_nearest_frontier()
                    or self._best_local_move()
                )
            if nxt is not None and nxt != self.pos:
                self.direction = self._direction_from_step(self.pos, nxt)
                self.pos       = nxt
            self.hidden = bool(self._tile_at(self.pos) == int(Tile.HIDE))
            return self.pos

        # Default: explore until exit and all missions are known.
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
        """Penalise turns: 0 = straight, 1 = left/right, 2 = U-turn."""
        order = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        delta = (order.index(candidate) - order.index(current)) % 4
        if delta == 0:
            return 0
        if delta == 2:
            return 2
        return 1
