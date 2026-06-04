"""Rule-based alien agent with a three-state FSM, A* pathfinding, and vent teleportation.

States:
  SEARCH      — explores unknown frontiers; falls back to patrol waypoints when fully explored.
  INVESTIGATE — moves to the last known or last heard player position.
  HUNT        — pursues a visible player at double speed (2 cells/step).

Transitions:
  Any state   → HUNT        when player is visible.
  HUNT        → HUNT        when player ducks into a HIDE tile (alien was already chasing).
  HUNT        → INVESTIGATE when player disappears and a recent sound cue exists.
  HUNT        → SEARCH      when player disappears and no recent sound.
  INVESTIGATE → SEARCH      when the alien reaches the last known position and the area
                             is ≥70% explored, or the sound cue expires.

Map knowledge is built incrementally from cone observations delivered by the simulation,
identical to how human agents work. No full map is available at construction time.
"""
import heapq
from enum import Enum, auto
from typing import Optional

import numpy as np

from agents.base import (
    BaseAlienAgent,
    Direction,
    direction_from_delta,
)
from map_generator import Tile

UNKNOWN     = -1   # cell not yet observed
PLAYER_SEEN = -2   # special marker written to knowledge map when a player was spotted here

# Tiles the alien can normally walk on (HIDE excluded — aliens cannot enter hiding spots).
PASSABLE_ALIEN      = {int(Tile.FLOOR), int(Tile.VENT), int(Tile.PLAYER_START), int(Tile.ALIEN_START), int(Tile.EXIT), int(Tile.MISSION)}
# Extended passable set used only when the alien confirmed the player is hiding in a spot.
PASSABLE_ALIEN_RUSH = {int(Tile.FLOOR), int(Tile.VENT), int(Tile.HIDE), int(Tile.PLAYER_START), int(Tile.ALIEN_START), int(Tile.EXIT), int(Tile.MISSION)}
# Tiles a player can walk on (used for vent teleport savings estimation).
PASSABLE_PLAYER     = {int(Tile.FLOOR), int(Tile.VENT), int(Tile.HIDE), int(Tile.PLAYER_START), int(Tile.ALIEN_START), int(Tile.EXIT), int(Tile.MISSION)}

# Cardinal directions as (dy, dx) offsets on a (y, x) grid.
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class AlienState(Enum):
    SEARCH      = auto()
    INVESTIGATE = auto()
    HUNT        = auto()


# Movement speed (cells per step) in each state.
SPEED = {
    AlienState.SEARCH:      1,
    AlienState.INVESTIGATE: 1,
    AlienState.HUNT:        2,  # double speed makes the alien hard to outrun
}

# Vent teleportation is only considered when the sound source is far away and
# the shortcut saves a meaningful number of steps.
VENT_ROUTE_MIN_SOUND_DISTANCE = 8   # minimum direct distance before vents are evaluated
VENT_ROUTE_MIN_SAVINGS        = 4   # shortcut must save at least this many steps
VENT_TELEPORT_COST            = 0   # teleport itself is instant (no extra step cost)


def heuristic(a, b):
    """Manhattan distance between two (y, x) positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal, passable):
    """A* pathfinding on a 2-D grid. All positions are (y, x). Returns path or []."""
    if start == goal:
        return [start]
    H, W = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g = {start: 0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        cy, cx = cur
        for dy, dx in DIRS:
            ny, nx = cy + dy, cx + dx
            nb = (ny, nx)
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if grid[ny, nx] not in passable:
                continue
            ng = g[cur] + 1
            if ng < g.get(nb, float("inf")):
                came_from[nb] = cur
                g[nb] = ng
                heapq.heappush(open_set, (ng + heuristic(nb, goal), nb))
    return []


class KnowledgeMap:
    """Incremental map of observed tiles and player sightings. All positions (y, x).

    Starts fully unknown; cells are filled in as the alien's cone FOV sweeps over them.
    """

    def __init__(self, shape: tuple[int, int]):
        H, W = shape
        self.knowledge = np.full((H, W), UNKNOWN, dtype=np.int16)  # observed tile IDs
        self.seen_vents: set[tuple[int, int]] = set()               # vent positions discovered so far

    def update_from_observation(self, visible_cells, grid_map, player_pos, player_visible, player_hiding):
        """Write observed tile IDs into the knowledge array and register any vents seen."""
        for vy, vx in visible_cells:  # (y, x) tuples
            if 0 <= vy < grid_map.shape[0] and 0 <= vx < grid_map.shape[1]:
                self.knowledge[vy, vx] = int(grid_map[vy, vx])
                if grid_map[vy, vx] == int(Tile.VENT):
                    self.seen_vents.add((vy, vx))
        if player_visible and not player_hiding and player_pos:
            py, px = player_pos  # (y, x)
            if 0 <= py < grid_map.shape[0] and 0 <= px < grid_map.shape[1]:
                self.knowledge[py, px] = PLAYER_SEEN  # mark last known player cell

    def get_unknown_frontier(self):
        """Return observed passable cells that border at least one unknown cell.
        Used to drive SEARCH exploration toward unexplored regions.
        """
        H, W = self.knowledge.shape
        candidates = []
        for y in range(H):
            for x in range(W):
                if self.knowledge[y, x] not in (UNKNOWN, PLAYER_SEEN) and self.knowledge[y, x] in PASSABLE_ALIEN:
                    for dy, dx in DIRS:
                        ny, nx = y + dy, x + dx
                        if 0 <= nx < W and 0 <= ny < H:
                            if self.knowledge[ny, nx] == UNKNOWN:
                                candidates.append((y, x))
                                break
        return candidates

    def get_seen_vents(self):
        return list(self.seen_vents)

    def get_previously_seen_player_area(self):
        """Return all cells marked PLAYER_SEEN — used as SEARCH fallback targets."""
        H, W = self.knowledge.shape
        player_areas = []
        for y in range(H):
            for x in range(W):
                if self.knowledge[y, x] == PLAYER_SEEN:
                    player_areas.append((y, x))
        return player_areas

    def get_copy(self):
        return self.knowledge.copy()


def build_waypoints(map_array: np.ndarray, n=6, seed=0):
    """Select n well-spaced patrol waypoints from observed passable cells.

    Points are chosen greedily so each is at least map_width/3 apart (Manhattan),
    ensuring the alien patrols different map regions rather than clustering.
    Called lazily once the alien has fully explored the known map.
    """
    rng = np.random.default_rng(seed)
    H, W = map_array.shape
    cands = [(y, x) for y in range(H) for x in range(W) if map_array[y, x] in PASSABLE_ALIEN]
    if not cands:
        return []
    cands = np.array(cands)
    rng.shuffle(cands)
    chosen = [tuple(cands[0])]
    for pt in cands[1:]:
        pt = tuple(pt)
        if all(heuristic(pt, c) >= W // 3 for c in chosen):
            chosen.append(pt)
        if len(chosen) >= n:
            break
    return chosen


class AlienAgent(BaseAlienAgent):
    """Rule-based alien with FSM, A* pathfinding, and vent routing.

    All positions use (y, x) / (row, col) convention.
    Uses directional cone FOV — can only see what is in front of it.
    Knowledge of the map is built incrementally from cone observations, same as humans.
    """

    def __init__(
        self,
        start_pos: tuple[int, int],
        view_length: int = 6,
        replan_every: int = 3,
    ) -> None:
        super().__init__(pos=start_pos, direction=Direction.SOUTH, view_length=view_length)
        self.start_pos = start_pos    # spawn position used by reset()
        self.replan_every = replan_every  # steps between A* replans (HUNT always replans)
        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, start_pos=None):
        self.pos               = start_pos or self.start_pos
        self.direction         = Direction.SOUTH
        self.state             = AlienState.SEARCH
        self.knowledge         = None  # initialised lazily on first observe() call
        self.last_known_pos    = None
        self.player_known_hiding = False
        self.last_heard_pos    = None
        self.steps_since_heard = 0
        self.path              = []
        self.waypoints         = []  # built lazily from observed map when first needed
        self.wp_idx            = 0
        self.steps_no_replan   = 0
        self.steps_in_state    = 0
        self.history           = []
        self.vent_teleport_used  = False
        # self._rl_action_override = None  # reserved for RL training — not used in rule-based play
        self._player_seen      = False
        self._player_hiding    = False
        self._player_pos       = None

    # ── Observation ───────────────────────────────────────────────────────────

    def observe(self, obs: np.ndarray, opponent_positions=None) -> None:
        """Ingest cone observation and update knowledge map.

        obs: full-grid-shaped array with UNKNOWN=-1 outside FOV, true tile IDs inside.
        opponent_positions: list of (pos, hidden) tuples for alive human agents, used
            to determine whether a visible cell contains a player or an empty HIDE tile.
        """
        if self.knowledge is None:
            self.knowledge = KnowledgeMap(obs.shape)
        visible = {(int(y), int(x)) for y, x in zip(*np.where(obs != UNKNOWN))}

        self._player_seen    = False
        self._player_hiding  = False
        self._player_pos     = None

        if opponent_positions:
            # Process nearest opponent first so the alien always reacts to the
            # closest threat rather than an arbitrary one.
            by_distance = sorted(opponent_positions, key=lambda op: heuristic(self.pos, op[0]))
            for pos, _hidden in by_distance:
                py, px = pos
                if (py, px) in visible:
                    if obs[py, px] == int(Tile.HIDE):
                        self._player_hiding = True   # player is inside a hide tile
                    else:
                        self._player_seen = True
                    self._player_pos = pos
                    break  # stop at nearest visible opponent

        self.knowledge.update_from_observation(
            visible, obs, self._player_pos,
            self._player_seen, self._player_hiding,
        )

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, player_pos: tuple, heard_pos: tuple = None, step_num: int = 0) -> tuple:
        """Execute one step. player_pos and heard_pos are (y, x)."""
        if heard_pos is None:
            heard_pos = player_pos

        # A sound cue arrives when heard_pos differs from the true player position
        # (the simulation jitters the position by noise_radius cells).
        sound_detected = heard_pos != player_pos
        if sound_detected:
            self.last_heard_pos      = heard_pos
            self.steps_since_heard   = 0
            self.path                = []
            self.steps_no_replan     = self.replan_every  # force replan toward sound
        else:
            self.steps_since_heard += 1
            if self.steps_since_heard > 5:
                self.last_heard_pos = None  # sound cue has gone stale

        # Visual perception state set by observe(), called by the simulation before step().
        player_seen   = self._player_seen
        player_hiding = self._player_hiding
        # observe() used pre-movement positions; refresh with the simulation's nearest-target
        # so pursuit is accurate on the same tick the player was spotted.
        if (player_seen or player_hiding) and player_pos is not None:
            self._player_pos = player_pos

        # STATE TRANSITION
        prev = self.state
        self._transition(player_seen, player_hiding, self._player_pos)
        if self.state != prev:
            self.steps_in_state = 0
            self.path = []
        else:
            self.steps_in_state += 1

        # VENT TELEPORTATION — only attempted when actively pursuing a sound cue
        self.vent_teleport_used = False
        if self.last_heard_pos is not None and self.steps_since_heard <= 5:
            target_vent = self._evaluate_vent_teleport(self.last_heard_pos)
            if target_vent:
                self._teleport_to_vent(target_vent)

        # MOVEMENT — HUNT moves 2 cells per step; all other states move 1
        steps = SPEED[self.state]
        for _ in range(steps):
            # RL training hook (not active in rule-based play):
            # if self._rl_action_override is not None:
            #     dy, dx = self._rl_action_override
            #     y, x = self.pos
            #     ny, nx = y + dy, x + dx
            #     if (0 <= ny < H and 0 <= nx < W
            #             and self.knowledge.knowledge[ny, nx] in PASSABLE_ALIEN):
            #         new_pos = (ny, nx)
            #         if new_pos != self.pos:
            #             self.direction = direction_from_delta(dy, dx)
            #         self.pos = new_pos
            # else:
            self.pos = self._move_one()

        self.steps_no_replan += 1
        self.history.append({
            "step":              step_num,
            "state":             self.state.name,
            "pos":               self.pos,
            "direction":         self.direction.name,
            "player_seen":       player_seen,
            "player_hiding":     player_hiding,
            "speed":             steps,
            "dist_to_player":    heuristic(self.pos, player_pos),
            "heard_pos":         heard_pos if sound_detected else None,
            "pursuing_sound":    self.last_heard_pos is not None,
            "vent_teleport_used": self.vent_teleport_used,
        })
        return self.pos

    # ── State machine ─────────────────────────────────────────────────────────

    def _get_explored_ratio(self, center: tuple, radius: int) -> float:
        """Return the fraction of cells within a square radius that have been observed."""
        cy, cx = center
        explored = 0
        total    = 0
        H, W = self.knowledge.knowledge.shape
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    total += 1
                    if self.knowledge.knowledge[ny, nx] != UNKNOWN:
                        explored += 1
        return explored / total if total > 0 else 0.0

    def _chase_passable(self) -> set:
        """Return the passable tile set for movement toward a known target.
        Switches to RUSH (includes HIDE) only when the player was confirmed hiding.
        """
        if self.player_known_hiding and self.state in (AlienState.HUNT, AlienState.INVESTIGATE):
            return PASSABLE_ALIEN_RUSH
        return PASSABLE_ALIEN

    def _transition(self, player_seen: bool, player_hiding: bool, player_pos: tuple):
        if player_seen:
            self.last_known_pos      = player_pos
            self.player_known_hiding = False
            self.state               = AlienState.HUNT
        elif player_hiding and self.state == AlienState.HUNT:
            # Player ducked into a hide spot while the alien was already chasing them.
            # Only rush in if the alien was already in HUNT — a wandering alien that
            # merely sees a HIDE tile has no reason to know anyone is inside.
            self.last_known_pos      = player_pos
            if not self.player_known_hiding:
                # First time we learn the player is hiding — force a replan toward the hide tile.
                self.path            = []
                self.steps_no_replan = self.replan_every
            self.player_known_hiding = True
            # state stays HUNT
        elif self.state == AlienState.HUNT:
            self.player_known_hiding = False
            if self.last_heard_pos is None or self.steps_since_heard > 5:
                self.state = AlienState.SEARCH
            else:
                self.state = AlienState.INVESTIGATE
        elif self.state == AlienState.INVESTIGATE:
            if self.last_known_pos and heuristic(self.pos, self.last_known_pos) <= 1:
                # Arrived at last known position — if the area is well explored,
                # the player is clearly not nearby; give up and resume SEARCH.
                if self._get_explored_ratio(self.last_known_pos, 4) > 0.7:
                    self.state = AlienState.SEARCH
            elif self.last_heard_pos is None or self.steps_since_heard > 5:
                self.state = AlienState.SEARCH

    # ── Vent routing ──────────────────────────────────────────────────────────

    def _best_seen_vent_route_for_sound(self, sound_pos: tuple) -> Optional[tuple]:
        """Return the entry vent for the best vent shortcut to sound_pos, or None.

        Evaluates whether walking to a nearby vent, teleporting, then walking to the
        sound source is faster than the direct path by at least VENT_ROUTE_MIN_SAVINGS.
        """
        seen_vents = self.knowledge.get_seen_vents()
        if len(seen_vents) < 2:
            return None
        km = self.knowledge.knowledge
        direct_path = astar(km, self.pos, sound_pos, PASSABLE_ALIEN)
        direct_dist = len(direct_path) - 1 if direct_path else float("inf")
        if direct_dist < VENT_ROUTE_MIN_SOUND_DISTANCE:
            return None  # sound is already close — not worth the detour
        start_vent = min(seen_vents, key=lambda v: heuristic(self.pos, v))
        dest_vent  = min(seen_vents, key=lambda v: heuristic(v, sound_pos))
        if start_vent == dest_vent:
            return None
        entry_path = astar(km, self.pos, start_vent, PASSABLE_ALIEN)
        entry_dist = len(entry_path) - 1 if entry_path else float("inf")
        exit_path  = astar(km, dest_vent, sound_pos, PASSABLE_ALIEN)
        exit_dist  = len(exit_path) - 1 if exit_path else float("inf")
        if exit_dist == float("inf"):
            return None  # can't reach sound from exit vent
        if direct_dist != float("inf"):
            if direct_dist - (entry_dist + VENT_TELEPORT_COST + exit_dist) < VENT_ROUTE_MIN_SAVINGS:
                return None
        return start_vent

    def _evaluate_vent_teleport(self, sound_pos: tuple) -> Optional[tuple]:
        """Return the best destination vent if teleporting from the current vent is worthwhile.

        Only callable when already standing on a vent — unlike _best_seen_vent_route_for_sound,
        this skips the walk-to-entry-vent cost because the alien is already there.
        """
        py, px = self.pos
        if self.knowledge.knowledge[py, px] != int(Tile.VENT):
            return None  # not on a vent
        seen_vents = self.knowledge.get_seen_vents()
        if len(seen_vents) < 2:
            return None
        km = self.knowledge.knowledge
        direct_path = astar(km, self.pos, sound_pos, PASSABLE_ALIEN)
        direct_dist = len(direct_path) - 1 if direct_path else float("inf")
        if direct_dist < VENT_ROUTE_MIN_SOUND_DISTANCE:
            return None
        best_vent = min(seen_vents, key=lambda v: heuristic(v, sound_pos))
        if best_vent == self.pos:
            return None
        exit_path = astar(km, best_vent, sound_pos, PASSABLE_ALIEN)
        exit_dist = len(exit_path) - 1 if exit_path else float("inf")
        if exit_dist == float("inf"):
            return None
        if direct_dist != float("inf"):
            if direct_dist - (VENT_TELEPORT_COST + exit_dist) < VENT_ROUTE_MIN_SAVINGS:
                return None
        return best_vent

    def _teleport_to_vent(self, target_vent: tuple):
        self.pos                = target_vent
        self.vent_teleport_used = True
        self.path               = []
        self.steps_no_replan    = self.replan_every  # force replan from new position

    # ── Movement and path planning ────────────────────────────────────────────

    def _move_one(self) -> tuple:
        # Always replan in HUNT so the 2-cell-per-step movement never overshoots
        # the player using a stale path from the previous step.
        if not self.path or self.steps_no_replan >= self.replan_every or self.state == AlienState.HUNT:
            self.path            = self._plan_path()
            self.steps_no_replan = 0

        new_pos = self.pos

        if self.path:
            self.path.pop(0)
            if self.path:
                new_pos = self.path[0]

        if new_pos == self.pos:
            # Path exhausted or blocked — take a greedy step toward the current goal.
            if self.state in (AlienState.HUNT, AlienState.INVESTIGATE):
                fallback_goal = self.last_known_pos
            else:
                fallback_goal = self.last_heard_pos
            greedy = self._greedy_step_toward(fallback_goal, self._chase_passable())
            if greedy is not None:
                new_pos = greedy

        if new_pos != self.pos:
            self.direction = direction_from_delta(new_pos[0] - self.pos[0], new_pos[1] - self.pos[1])
        return new_pos

    def _greedy_step_toward(self, goal: Optional[tuple], passable: set | None = None) -> Optional[tuple]:
        """Single-step greedy move: pick the known-passable neighbour closest to goal."""
        if goal is None:
            return None
        if passable is None:
            passable = PASSABLE_ALIEN
        best_step = None
        best_dist = heuristic(self.pos, goal)
        H, W = self.knowledge.knowledge.shape
        y, x = self.pos
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if self.knowledge.knowledge[ny, nx] not in passable:
                continue
            candidate      = (ny, nx)
            candidate_dist = heuristic(candidate, goal)
            if candidate_dist < best_dist:
                best_dist = candidate_dist
                best_step = candidate
        return best_step

    def _plan_path(self) -> list:
        """Compute an A* path to the current state's goal using the observed map."""
        km   = self.knowledge.knowledge
        goal = self.pos  # default: stay put (A* will return empty path)

        if self.state == AlienState.HUNT:
            path = astar(km, self.pos, self.last_known_pos, self._chase_passable())
            if not path:
                # last_known_pos not yet in the known map — head to the nearest
                # frontier that is closest to where the player was last seen.
                frontiers = [f for f in self.knowledge.get_unknown_frontier() if f != self.pos]
                if frontiers:
                    goal = min(frontiers, key=lambda f: heuristic(f, self.last_known_pos))
                    path = astar(km, self.pos, goal, PASSABLE_ALIEN)
            elif len(path) <= 1 and self.player_known_hiding:
                # Alien has arrived at the hide tile but can't advance further (already there
                # or path exhausted). Give up the rush and resume SEARCH to avoid freezing.
                self.player_known_hiding = False
                self.state = AlienState.SEARCH
                frontiers = [f for f in self.knowledge.get_unknown_frontier() if f != self.pos]
                if frontiers:
                    goal = min(frontiers, key=lambda f: heuristic(self.pos, f))
                    path = astar(km, self.pos, goal, PASSABLE_ALIEN)
            return path

        elif self.last_heard_pos is not None and self.steps_since_heard <= 5:
            # Active sound cue — route toward it, optionally via a vent shortcut.
            py, px         = self.pos
            current_is_vent = km[py, px] == int(Tile.VENT)
            best_sound_vent = self._best_seen_vent_route_for_sound(self.last_heard_pos)

            sound_y, sound_x = self.last_heard_pos
            if self.knowledge.knowledge[sound_y, sound_x] == UNKNOWN:
                # Sound position not yet in known map — navigate to the nearest
                # known frontier that is closest to the sound source.
                frontiers   = self.knowledge.get_unknown_frontier()
                target_goal = (min(frontiers, key=lambda f: heuristic(f, self.last_heard_pos))
                               if frontiers else self.last_heard_pos)
            else:
                target_goal = self.last_heard_pos

            if not current_is_vent and best_sound_vent is not None:
                goal = best_sound_vent  # head to vent entry for shortcut
            else:
                goal = target_goal

        elif self.state == AlienState.INVESTIGATE:
            if self.last_known_pos is None:
                goal = self.pos
            elif km[self.last_known_pos[0], self.last_known_pos[1]] == int(Tile.HIDE):
                if self.player_known_hiding:
                    # Confirmed: player was seen entering — rush in through HIDE tiles.
                    return astar(km, self.pos, self.last_known_pos, PASSABLE_ALIEN_RUSH)
                else:
                    # Suspected: last known pos is a hide tile but player wasn't seen
                    # entering — patrol adjacent cells instead of freezing at the door.
                    hy, hx = self.last_known_pos
                    H, W   = km.shape
                    adjacent = [
                        (hy + dy, hx + dx) for dy, dx in DIRS
                        if (0 <= hy + dy < H
                            and 0 <= hx + dx < W
                            and km[hy + dy, hx + dx] in PASSABLE_ALIEN)
                    ]
                    goal = min(adjacent, key=lambda p: heuristic(self.pos, p)) if adjacent else self.pos
            else:
                goal = self.last_known_pos

        else:  # SEARCH
            # Exclude current position: A*(pos, pos) produces no movement.
            frontiers = [f for f in self.knowledge.get_unknown_frontier() if f != self.pos]
            if frontiers:
                goal = min(frontiers, key=lambda f: heuristic(self.pos, f))
            elif self.last_heard_pos is not None and self.steps_since_heard <= 10:
                goal = self.last_heard_pos
            else:
                prev_areas = self.knowledge.get_previously_seen_player_area()
                if prev_areas:
                    goal = min(prev_areas, key=lambda p: heuristic(self.pos, p))
                else:
                    # Map fully explored — build patrol waypoints from observed tiles
                    # and cycle through them to keep the alien moving.
                    if not self.waypoints:
                        self.waypoints = build_waypoints(self.knowledge.knowledge)
                    if self.waypoints:
                        goal = self.waypoints[self.wp_idx % len(self.waypoints)]
                        self.wp_idx += 1

        return astar(km, self.pos, goal, PASSABLE_ALIEN)
