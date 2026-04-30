#!/usr/bin/env python3
"""
alien_agent.py  —  Alien with limited FOV, knowledge map, and adaptive speed.

States:
    SEARCH      → has gaps in knowledge: explore unknown areas
    INVESTIGATE → has last known pos: A* toward it
    HUNT        → player visible: A* replanning, moves 2 cells per step

Speed burst:
    - SEARCH / INVESTIGATE: 1 cell per step
    - HUNT (player visible): 2 cells per step (alien sees player and charges)
    
Knowledge Map:
    - Similar to human knowledge map
    - Tracks observed tiles vs unknown
    - Cannot make probabilistic assumptions about player movement
"""

import heapq
import numpy as np
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

# ── Tile constants ─────────────────────────────────────────────────────────────
WALL         = 0
FLOOR        = 1
VENT         = 2
HIDE         = 3
PLAYER_START = 4
ALIEN_START  = 5
EXIT         = 6
UNKNOWN      = -1
PLAYER_SEEN  = -2

PASSABLE_ALIEN  = {FLOOR, VENT, HIDE, PLAYER_START, ALIEN_START, EXIT}

DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# ── State ──────────────────────────────────────────────────────────────────────
class AlienState(Enum):
    SEARCH      = auto()
    INVESTIGATE = auto()
    HUNT        = auto()

SPEED = {
    AlienState.SEARCH:      1,
    AlienState.INVESTIGATE: 1,
    AlienState.HUNT:        2,
}

# ── A* ─────────────────────────────────────────────────────────────────────────
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, passable):
    """Returns full path [(x,y), ...] from start to goal, or [] if unreachable."""
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
        cx, cy = cur
        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            nb = (nx, ny)
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

# ── FOV (ray-cast, HIDE and WALL block LOS) ────────────────────────────────────
def compute_fov(grid, origin, radius):
    ox, oy = origin
    H, W = grid.shape
    visible = {origin}
    LOS_BLOCKERS = {WALL, HIDE}
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            tx, ty = ox + dx, oy + dy
            if not (0 <= tx < W and 0 <= ty < H):
                continue
            if abs(dx) + abs(dy) > radius:
                continue
            steps = max(abs(dx), abs(dy), 1)
            blocked = False
            for s in range(1, steps + 1):
                ix = round(ox + dx * s / steps)
                iy = round(oy + dy * s / steps)
                if not (0 <= ix < W and 0 <= iy < H):
                    blocked = True; break
                cell = (ix, iy)
                visible.add(cell)
                if grid[iy, ix] in LOS_BLOCKERS and cell != (tx, ty):
                    blocked = True; break
            if not blocked:
                visible.add((tx, ty))
    return visible

# ── Belief map ─────────────────────────────────────────────────────────────────
class BeliefMap:
    """
    Bayesian probability grid over player position.
    - Uniform prior over all player-walkable cells.
    - diffuse(): spread probability (player may have moved).
    - observe(): zero out visible cells with no player; collapse if player seen.
    """
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
                for dx, dy in DIRS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and self.grid[ny, nx] in PASSABLE_PLAYER:
                        out[ny, nx] += p * move
                    else:
                        out[y, x] += p * move
        self.belief = out
        self._norm()

    def observe(self, visible, player_visible, player_pos=None, player_hiding=False):
        if player_visible and not player_hiding and player_pos:
            self.belief[:] = 0.0
            self.belief[player_pos[1], player_pos[0]] = 1.0
        else:
            for (vx, vy) in visible:
                if self.grid[vy, vx] != HIDE:
                    self.belief[vy, vx] = 0.0
        self._norm()

    def peak(self):
        if self.belief.max() < 1e-12:
            return None
        r, c = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return (int(c), int(r))


# ── Knowledge map (similar to human knowledge) ─────────────────────────────────
class KnowledgeMap:
    """
    Tracks what the alien has observed.
    - Similar to human's knowledge map
    - Stores observed tiles vs unknown
    - Marks player when visible
    """
    def __init__(self, grid):
        self.grid = grid
        H, W = grid.shape
        self.knowledge = np.full((H, W), UNKNOWN, dtype=np.int16)

    def update_from_observation(self, visible_cells, grid_map, player_pos, player_visible, player_hiding):
        """Update knowledge from visible cells."""
        for (vx, vy) in visible_cells:
            if 0 <= vy < grid_map.shape[0] and 0 <= vx < grid_map.shape[1]:
                self.knowledge[vy, vx] = int(grid_map[vy, vx])
        
        # Mark player if visible
        if player_visible and not player_hiding and player_pos:
            px, py = player_pos
            if 0 <= py < grid_map.shape[0] and 0 <= px < grid_map.shape[1]:
                self.knowledge[py, px] = PLAYER_SEEN

    def get_unknown_frontier(self):
        """Find cells that are passable but have unknown neighbors."""
        H, W = self.knowledge.shape
        candidates = []
        for y in range(H):
            for x in range(W):
                # Cell must be known and passable
                if self.knowledge[y, x] not in (UNKNOWN, PLAYER_SEEN) and \
                   self.grid[y, x] in PASSABLE_ALIEN:
                    # Check if it has unknown neighbors
                    for dx, dy in DIRS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            if self.knowledge[ny, nx] == UNKNOWN:
                                candidates.append((x, y))
                                break
        return candidates

    def get_copy(self):
        """Return a copy of the knowledge map."""
        return self.knowledge.copy()


# ── Patrol waypoints ────────────────────────────────────────────────────────────
def build_waypoints(grid, n=6, seed=0):
    rng = np.random.default_rng(seed)
    H, W = grid.shape
    cands = [(x, y) for y in range(H) for x in range(W)
             if grid[y, x] in PASSABLE_ALIEN]
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


# ── Alien Agent ────────────────────────────────────────────────────────────────
@dataclass
class AlienAgent:
    """
    Behaviour Tree alien with:
      - Limited FOV (ray-cast, HIDE blocks LOS)
      - Knowledge map tracking observations (like human knowledge)
      - 3 states: SEARCH / INVESTIGATE / HUNT
      - Speed burst: 1 cell normally, 2 cells when hunting (player visible)
      - Cannot assume player won't revisit known areas

    Parameters
    ----------
    grid        : np.ndarray map
    start_pos   : (x, y)
    fov_radius  : Manhattan radius of vision
    replan_every: replan A* every N steps
    """
    grid:        np.ndarray
    start_pos:   tuple
    fov_radius:  int = 6
    replan_every: int = 3

    pos:              tuple      = field(init=False)
    state:            AlienState = field(init=False)
    knowledge:        KnowledgeMap = field(init=False)
    last_known_pos:   Optional[tuple] = field(init=False)
    path:             list       = field(init=False)
    waypoints:        list       = field(init=False)
    wp_idx:           int        = field(init=False)
    steps_no_replan:  int        = field(init=False)
    steps_in_state:   int        = field(init=False)
    history:          list       = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self, start_pos=None):
        self.pos            = start_pos or self.start_pos
        self.state          = AlienState.SEARCH
        self.knowledge      = KnowledgeMap(self.grid)
        self.last_known_pos = None
        self.path           = []
        self.waypoints      = build_waypoints(self.grid)
        self.wp_idx         = 0
        self.steps_no_replan = 0
        self.steps_in_state  = 0
        self.history         = []

    # ── Main step ──────────────────────────────────────────────────────────────
    def step(self, player_pos: tuple, step_num: int = 0) -> tuple:
        """
        Observe → update knowledge → transition → move (1 or 2 cells).
        Returns new alien (x, y).
        """
        # 1. Sense
        fov           = compute_fov(self.grid, self.pos, self.fov_radius)
        px, py        = player_pos
        in_fov        = (px, py) in fov
        player_hiding = in_fov and self.grid[py, px] == HIDE
        player_seen   = in_fov and not player_hiding

        # 2. Update knowledge map from observation
        self.knowledge.update_from_observation(fov, self.grid, player_pos, player_seen, player_hiding)

        # 3. Behaviour tree transition
        prev = self.state
        self._transition(player_seen, player_pos)
        if self.state != prev:
            self.steps_in_state = 0
            self.path = []
        else:
            self.steps_in_state += 1

        # 4. Move — speed depends on state
        steps = SPEED[self.state]
        for _ in range(steps):
            self.pos = self._move_one(player_pos, player_seen)

        self.steps_no_replan += 1
        self.history.append({
            "step":           step_num,
            "state":          self.state.name,
            "pos":            self.pos,
            "player_seen":    player_seen,
            "player_hiding":  player_hiding,
            "speed":          steps,
            "dist_to_player": heuristic(self.pos, player_pos),
        })
        return self.pos

    # ── Behaviour tree ─────────────────────────────────────────────────────────
    def _transition(self, player_seen: bool, player_pos: tuple):
        """State machine: SEARCH ↔ INVESTIGATE ↔ HUNT"""
        if player_seen:
            self.last_known_pos = player_pos
            self.state = AlienState.HUNT
        elif self.state == AlienState.HUNT:
            # Just lost sight → investigate last known pos
            self.state = AlienState.INVESTIGATE
        elif self.state == AlienState.INVESTIGATE:
            # Reached last known pos with no sighting → give up, search
            if self.last_known_pos and heuristic(self.pos, self.last_known_pos) <= 1:
                self.state = AlienState.SEARCH
        # else stay in current state

    # ── Single movement ────────────────────────────────────────────────────────
    def _move_one(self, player_pos: tuple, player_seen: bool) -> tuple:
        """Make one movement step based on current state."""
        if not self.path or self.steps_no_replan >= self.replan_every:
            self.path = self._plan_path(player_pos, player_seen)
            self.steps_no_replan = 0

        if self.path:
            self.path.pop(0)  # consume current position
            if self.path:
                return self.path[0]

        # No path found or path exhausted: stay
        return self.pos

    def _plan_path(self, player_pos: tuple, player_seen: bool) -> list:
        """Plan A* path based on state and knowledge map."""
        if self.state == AlienState.HUNT:
            goal = player_pos
        elif self.state == AlienState.INVESTIGATE:
            goal = self.last_known_pos or self.pos
        else:  # SEARCH
            # Explore unknown frontiers (edges of known areas)
            frontiers = self.knowledge.get_unknown_frontier()
            if frontiers:
                # Pick closest frontier
                goal = min(frontiers, key=lambda f: heuristic(self.pos, f))
            else:
                # Fallback to next waypoint
                if self.waypoints:
                    goal = self.waypoints[self.wp_idx % len(self.waypoints)]
                    self.wp_idx += 1
                else:
                    goal = self.pos

        return astar(self.grid, self.pos, goal, PASSABLE_ALIEN)