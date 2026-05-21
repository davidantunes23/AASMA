import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from agents.base import (
    BaseAlienAgent,
    Direction,
    cone_fov,
    direction_from_delta,
)

# Tile type identifiers (must match map_generator.Tile enum)
WALL = 0
FLOOR = 1
VENT = 2
HIDE = 3
PLAYER_START = 4
ALIEN_START = 5
EXIT = 6
UNKNOWN = -1
PLAYER_SEEN = -2

PASSABLE_ALIEN = {FLOOR, VENT, PLAYER_START, ALIEN_START, EXIT}
PASSABLE_ALIEN_RUSH = {FLOOR, VENT, HIDE, PLAYER_START, ALIEN_START, EXIT}  # used when entering a confirmed hiding spot
PASSABLE_PLAYER = {FLOOR, VENT, HIDE, PLAYER_START, ALIEN_START, EXIT}

# Cardinal directions as (dy, dx) applied to (y, x) — right, left, down, up
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


class AlienState(Enum):
    SEARCH = auto()
    INVESTIGATE = auto()
    HUNT = auto()


SPEED = {
    AlienState.SEARCH: 1,
    AlienState.INVESTIGATE: 1,
    AlienState.HUNT: 2,
}

VENT_ROUTE_MIN_SOUND_DISTANCE = 8
VENT_ROUTE_MIN_SAVINGS = 4
VENT_TELEPORT_COST = 0


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal, passable):
    """A* pathfinding. All positions are (y, x). Returns path or []."""
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


class BeliefMap:
    """Bayesian probability map over player position. All positions (y, x)."""

    def __init__(self, grid):
        self.grid = grid
        H, W = grid.shape
        self.belief = np.zeros((H, W), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                if grid[y, x] in PASSABLE_PLAYER:
                    self.belief[y, x] = 1.0
        self._norm()

    def _norm(self):
        t = self.belief.sum()
        if t > 1e-12:
            self.belief /= t

    def diffuse(self, stay=0.5):
        H, W = self.grid.shape
        out = np.zeros_like(self.belief)
        move = (1 - stay) / 4
        for y in range(H):
            for x in range(W):
                p = self.belief[y, x]
                if p == 0:
                    continue
                out[y, x] += p * stay
                for dy, dx in DIRS:
                    ny, nx = y + dy, x + dx
                    if 0 <= nx < W and 0 <= ny < H and self.grid[ny, nx] in PASSABLE_PLAYER:
                        out[ny, nx] += p * move
                    else:
                        out[y, x] += p * move
        self.belief = out
        self._norm()

    def observe(self, visible, player_visible, player_pos=None, player_hiding=False):
        if player_visible and not player_hiding and player_pos:
            self.belief[:] = 0.0
            self.belief[player_pos[0], player_pos[1]] = 1.0  # player_pos is (y, x)
        else:
            for vy, vx in visible:  # visible contains (y, x) tuples
                if self.grid[vy, vx] != HIDE:
                    self.belief[vy, vx] = 0.0
        self._norm()

    def peak(self):
        if self.belief.max() < 1e-12:
            return None
        r, c = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return (int(r), int(c))  # (y, x)


class KnowledgeMap:
    """Tracks observed tiles and player sightings. All positions (y, x)."""

    def __init__(self, grid):
        self.grid = grid
        H, W = grid.shape
        self.knowledge = np.full((H, W), UNKNOWN, dtype=np.int16)
        self.seen_vents: set[tuple[int, int]] = set()

    def update_from_observation(self, visible_cells, grid_map, player_pos, player_visible, player_hiding):
        for vy, vx in visible_cells:  # (y, x) tuples
            if 0 <= vy < grid_map.shape[0] and 0 <= vx < grid_map.shape[1]:
                self.knowledge[vy, vx] = int(grid_map[vy, vx])
                if grid_map[vy, vx] == VENT:
                    self.seen_vents.add((vy, vx))
        if player_visible and not player_hiding and player_pos:
            py, px = player_pos  # (y, x)
            if 0 <= py < grid_map.shape[0] and 0 <= px < grid_map.shape[1]:
                self.knowledge[py, px] = PLAYER_SEEN

    def get_unknown_frontier(self):
        H, W = self.knowledge.shape
        candidates = []
        for y in range(H):
            for x in range(W):
                if self.knowledge[y, x] not in (UNKNOWN, PLAYER_SEEN) and self.grid[y, x] in PASSABLE_ALIEN:
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
        H, W = self.knowledge.shape
        player_areas = []
        for y in range(H):
            for x in range(W):
                if self.knowledge[y, x] == PLAYER_SEEN:
                    player_areas.append((y, x))
        return player_areas

    def get_copy(self):
        return self.knowledge.copy()


def build_waypoints(grid, n=6, seed=0):
    """Create n well-spaced patrol waypoints. Returns list of (y, x) positions."""
    rng = np.random.default_rng(seed)
    H, W = grid.shape
    cands = [(y, x) for y in range(H) for x in range(W) if grid[y, x] in PASSABLE_ALIEN]
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


@dataclass
class AlienAgent(BaseAlienAgent):
    """Rule-based alien with FSM, A* pathfinding, belief map, and vent routing.

    All positions use (y, x) / (row, col) convention.
    Uses directional cone FOV — can only see what is in front of it.
    """

    grid: np.ndarray
    start_pos: tuple[int, int]
    view_length: int = 6
    replan_every: int = 3

    pos: tuple[int, int] = field(init=False)
    direction: Direction = field(init=False)
    hidden: bool = field(default=False, init=False)
    state: AlienState = field(init=False)
    knowledge: KnowledgeMap = field(init=False)
    belief: BeliefMap = field(init=False)
    last_known_pos: Optional[tuple] = field(init=False)
    player_known_hiding: bool = field(default=False, init=False)
    last_heard_pos: Optional[tuple] = field(init=False)
    steps_since_heard: int = field(init=False)
    path: list = field(init=False)
    waypoints: list = field(init=False)
    wp_idx: int = field(init=False)
    steps_no_replan: int = field(init=False)
    steps_in_state: int = field(init=False)
    history: list = field(init=False)
    _rl_action_override: Optional[tuple] = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self, start_pos=None):
        self.pos = start_pos or self.start_pos
        self.direction = Direction.SOUTH
        self.state = AlienState.SEARCH
        self.knowledge = KnowledgeMap(self.grid)
        self.belief = BeliefMap(self.grid)
        self.last_known_pos = None
        self.player_known_hiding = False
        self.last_heard_pos = None
        self.steps_since_heard = 0
        self.path = []
        self.waypoints = build_waypoints(self.grid)
        self.wp_idx = 0
        self.steps_no_replan = 0
        self.steps_in_state = 0
        self.history = []
        self.vent_teleport_used = False
        self._rl_action_override = None

    def step(self, player_pos: tuple, heard_pos: tuple = None, step_num: int = 0) -> tuple:
        """Execute one step. player_pos and heard_pos are (y, x)."""
        if heard_pos is None:
            heard_pos = player_pos

        sound_detected = heard_pos != player_pos
        if sound_detected:
            self.last_heard_pos = heard_pos
            self.steps_since_heard = 0
            self.path = []
            self.steps_no_replan = self.replan_every
        else:
            self.steps_since_heard += 1
            if self.steps_since_heard > 5:
                self.last_heard_pos = None

        # STEP 1: VISUAL PERCEPTION — cone FOV based on current direction
        fov = cone_fov(self.grid, self.pos, self.direction, self.view_length)
        py, px = player_pos
        in_fov = (py, px) in fov
        player_hiding = in_fov and self.grid[py, px] == HIDE
        player_seen = in_fov and not player_hiding

        # STEP 2: UPDATE KNOWLEDGE MAP
        self.knowledge.update_from_observation(fov, self.grid, player_pos, player_seen, player_hiding)

        # STEP 3: UPDATE BELIEF MAP WITH AUDITORY EVIDENCE
        if sound_detected:
            hy, hx = heard_pos
            if 0 <= hy < self.grid.shape[0] and 0 <= hx < self.grid.shape[1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = hy + dy, hx + dx
                        if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]:
                            if self.grid[ny, nx] in PASSABLE_PLAYER:
                                self.belief.belief[ny, nx] += 0.1
        self.belief._norm()

        # STEP 4: STATE TRANSITION
        prev = self.state
        self._transition(player_seen, player_hiding, player_pos)
        print(
            f"[alien s={step_num:3d}] pos={self.pos} dir={self.direction.name:<5} "
            f"state={self.state.name:<11} in_fov={in_fov} "
            f"seen={player_seen} hiding={player_hiding} known_hiding={self.player_known_hiding} "
            f"path_len={len(self.path)}"
        )
        if self.state != prev:
            self.steps_in_state = 0
            self.path = []
        else:
            self.steps_in_state += 1

        # STEP 4.5: VENT TELEPORTATION
        self.vent_teleport_used = False
        if self.last_heard_pos is not None and self.steps_since_heard <= 5:
            target_vent = self._evaluate_vent_teleport(self.last_heard_pos)
            if target_vent:
                self._teleport_to_vent(target_vent)

        # STEP 5: MOVEMENT
        steps = SPEED[self.state]
        for _ in range(steps):
            if self._rl_action_override is not None:
                dy, dx = self._rl_action_override
                y, x = self.pos
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]
                        and self.grid[ny, nx] in PASSABLE_ALIEN):
                    new_pos = (ny, nx)
                    if new_pos != self.pos:
                        self.direction = direction_from_delta(dy, dx)
                    self.pos = new_pos
            else:
                self.pos = self._move_one(player_pos, player_seen)

        self.steps_no_replan += 1
        self.history.append({
            "step": step_num,
            "state": self.state.name,
            "pos": self.pos,
            "direction": self.direction.name,
            "player_seen": player_seen,
            "player_hiding": player_hiding,
            "speed": steps,
            "dist_to_player": heuristic(self.pos, player_pos),
            "heard_pos": heard_pos if sound_detected else None,
            "pursuing_sound": self.last_heard_pos is not None,
            "vent_teleport_used": self.vent_teleport_used,
        })
        return self.pos

    def _get_explored_ratio(self, center: tuple, radius: int) -> float:
        cy, cx = center
        explored = 0
        total = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]:
                    if self.grid[ny, nx] in PASSABLE_ALIEN:
                        total += 1
                        if self.knowledge.knowledge[ny, nx] != UNKNOWN:
                            explored += 1
        return explored / total if total > 0 else 0.0

    def _chase_passable(self) -> set:
        """Return the passable tile set for movement toward a known target."""
        if self.player_known_hiding and self.state in (AlienState.HUNT, AlienState.INVESTIGATE):
            return PASSABLE_ALIEN_RUSH
        return PASSABLE_ALIEN

    def _transition(self, player_seen: bool, player_hiding: bool, player_pos: tuple):
        if player_seen:
            self.last_known_pos = player_pos
            self.player_known_hiding = False
            self.state = AlienState.HUNT
        elif player_hiding:
            # Alien sees player inside a hiding spot — record location and rush in
            self.last_known_pos = player_pos
            self.player_known_hiding = True
            self.path = []                        # must replan: passable set changes to include HIDE
            self.steps_no_replan = self.replan_every
            self.state = AlienState.HUNT
        elif self.state == AlienState.HUNT:
            self.player_known_hiding = False
            if self.last_heard_pos is None or self.steps_since_heard > 5:
                self.state = AlienState.SEARCH
            else:
                self.state = AlienState.INVESTIGATE
        elif self.state == AlienState.INVESTIGATE:
            if self.last_known_pos and heuristic(self.pos, self.last_known_pos) <= 1:
                if self._get_explored_ratio(self.last_known_pos, 4) > 0.7:
                    self.state = AlienState.SEARCH
            elif self.last_heard_pos is None or self.steps_since_heard > 5:
                self.state = AlienState.SEARCH

    def _best_seen_vent_route_for_sound(self, sound_pos: tuple) -> Optional[tuple]:
        seen_vents = self.knowledge.get_seen_vents()
        if len(seen_vents) < 2:
            return None
        direct_dist = heuristic(self.pos, sound_pos)
        if direct_dist < VENT_ROUTE_MIN_SOUND_DISTANCE:
            return None
        start_vent = min(seen_vents, key=lambda v: heuristic(self.pos, v))
        dest_vent = min(seen_vents, key=lambda v: heuristic(v, sound_pos))
        if start_vent == dest_vent:
            return None
        dist_via_vent = heuristic(self.pos, start_vent) + VENT_TELEPORT_COST + heuristic(dest_vent, sound_pos)
        if direct_dist - dist_via_vent < VENT_ROUTE_MIN_SAVINGS:
            return None
        return start_vent

    def _evaluate_vent_teleport(self, sound_pos: tuple) -> Optional[tuple]:
        if self.grid[self.pos[0], self.pos[1]] != VENT:  # pos is (y, x)
            return None
        seen_vents = self.knowledge.get_seen_vents()
        if len(seen_vents) < 2:
            return None
        direct_dist = heuristic(self.pos, sound_pos)
        if direct_dist < VENT_ROUTE_MIN_SOUND_DISTANCE:
            return None
        best_vent = min(seen_vents, key=lambda v: heuristic(v, sound_pos))
        if best_vent == self.pos:
            return None
        if direct_dist - (VENT_TELEPORT_COST + heuristic(best_vent, sound_pos)) < VENT_ROUTE_MIN_SAVINGS:
            return None
        return best_vent

    def _teleport_to_vent(self, target_vent: tuple):
        self.pos = target_vent
        self.vent_teleport_used = True
        self.path = []
        self.steps_no_replan = self.replan_every

    def _move_one(self, player_pos: tuple, player_seen: bool) -> tuple:
        if not self.path or self.steps_no_replan >= self.replan_every:
            self.path = self._plan_path(player_pos, player_seen)
            self.steps_no_replan = 0

        new_pos = self.pos

        if self.path:
            self.path.pop(0)
            if self.path:
                new_pos = self.path[0]

        if new_pos == self.pos:
            # Fallback: greedy step toward current objective
            if self.state == AlienState.HUNT:
                fallback_goal = player_pos
            elif self.state == AlienState.INVESTIGATE:
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
        if goal is None:
            return None
        if passable is None:
            passable = PASSABLE_ALIEN
        best_step = None
        best_dist = heuristic(self.pos, goal)
        y, x = self.pos
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if not (0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]):
                continue
            if self.grid[ny, nx] not in passable:
                continue
            candidate = (ny, nx)
            candidate_dist = heuristic(candidate, goal)
            if candidate_dist < best_dist:
                best_dist = candidate_dist
                best_step = candidate
        return best_step

    def _plan_path(self, player_pos: tuple, player_seen: bool) -> list:
        goal = self.pos

        if self.state == AlienState.HUNT:
            return astar(self.grid, self.pos, player_pos, self._chase_passable())

        elif self.last_heard_pos is not None and self.steps_since_heard <= 5:
            current_is_vent = self.grid[self.pos[0], self.pos[1]] == VENT
            best_sound_vent = self._best_seen_vent_route_for_sound(self.last_heard_pos)

            sound_y, sound_x = self.last_heard_pos
            if self.knowledge.knowledge[sound_y, sound_x] == UNKNOWN:
                frontiers = self.knowledge.get_unknown_frontier()
                target_goal = (min(frontiers, key=lambda f: heuristic(f, self.last_heard_pos))
                               if frontiers else self.last_heard_pos)
            else:
                target_goal = self.last_heard_pos

            if not current_is_vent and best_sound_vent is not None:
                goal = best_sound_vent
            else:
                goal = target_goal

        elif self.state == AlienState.INVESTIGATE:
            if self.last_known_pos is None:
                goal = self.pos
            elif self.grid[self.last_known_pos[0], self.last_known_pos[1]] == HIDE:
                if self.player_known_hiding:
                    # Confirmed: player was seen entering — rush in
                    return astar(self.grid, self.pos, self.last_known_pos, PASSABLE_ALIEN_RUSH)
                else:
                    # Suspected: patrol adjacent passable cells instead of freezing
                    hy, hx = self.last_known_pos
                    adjacent = [
                        (hy + dy, hx + dx) for dy, dx in DIRS
                        if (0 <= hy + dy < self.grid.shape[0]
                            and 0 <= hx + dx < self.grid.shape[1]
                            and self.grid[hy + dy, hx + dx] in PASSABLE_ALIEN)
                    ]
                    goal = min(adjacent, key=lambda p: heuristic(self.pos, p)) if adjacent else self.pos
            else:
                goal = self.last_known_pos

        else:  # SEARCH
            frontiers = self.knowledge.get_unknown_frontier()
            if frontiers:
                goal = min(frontiers, key=lambda f: heuristic(self.pos, f))
            elif self.last_heard_pos is not None and self.steps_since_heard <= 10:
                goal = self.last_heard_pos
            else:
                prev_areas = self.knowledge.get_previously_seen_player_area()
                if prev_areas:
                    goal = min(prev_areas, key=lambda p: heuristic(self.pos, p))
                elif self.waypoints:
                    goal = self.waypoints[self.wp_idx % len(self.waypoints)]
                    self.wp_idx += 1

        return astar(self.grid, self.pos, goal, PASSABLE_ALIEN)
