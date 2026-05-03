import heapq
import numpy as np
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

# Tile type identifiers (must match map_generator.Tile enum)
WALL         = 0  # Impassable barrier
FLOOR        = 1  # Traversable corridor
VENT         = 2  # Traversable vent (teleport mechanism)
HIDE         = 3  # Hiding spot (blocks alien FOV, walkable by player)
PLAYER_START = 4  # Starting position for player
ALIEN_START  = 5  # Starting position for alien
EXIT         = 6  # Goal location for player
# Knowledge map markers
UNKNOWN      = -1  # Never observed before
PLAYER_SEEN  = -2  # Position where player was previously observed

# Movement passability sets
PASSABLE_ALIEN  = {FLOOR, VENT, PLAYER_START, ALIEN_START, EXIT}  # Alien can walk through these
PASSABLE_PLAYER = {FLOOR, VENT, HIDE, PLAYER_START, ALIEN_START, EXIT}  # Player can walk through these

# Cardinal directions: right, left, down, up
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Alien behavior states
class AlienState(Enum):
    # SEARCH: explore areas with gaps in knowledge map
    SEARCH      = auto()
    # INVESTIGATE: player was seen here; return to check again
    INVESTIGATE = auto()
    # HUNT: player currently visible in FOV; active pursuit
    HUNT        = auto()

# Movement speed per state (cells per step)
SPEED = {
    AlienState.SEARCH:      1,  # Cautious exploration pace
    AlienState.INVESTIGATE: 1,  # Careful investigation pace
    AlienState.HUNT:        2,  # Speed burst when player visible
}

# Vent teleportation thresholds
VENT_ROUTE_MIN_SOUND_DISTANCE = 8  # Only teleport if sound is 8+ cells away
VENT_ROUTE_MIN_SAVINGS = 4         # Vent route must save 4+ steps vs direct path
VENT_TELEPORT_COST = 0             # Step cost for a vent teleport (0 = instant)

# Pathfinding utilities
def heuristic(a, b):
    # Manhattan distance for A* heuristic (admissible and consistent)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, passable):
    # A* pathfinding with Manhattan distance heuristic
    # Returns full path [(x,y), ...] from start to goal, or [] if unreachable
    # Only traverses tiles in passable set
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

# Field of view computation using ray-casting
def compute_fov(grid, origin, radius):
    # Ray-cast FOV within Manhattan radius
    # HIDE spots and WALL tiles block line-of-sight
    # Returns set of visible (x, y) positions from origin
    ox, oy = origin
    H, W = grid.shape
    visible = {origin}  # Origin is always visible
    LOS_BLOCKERS = {WALL, HIDE}  # These tiles block vision
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

# Bayesian probability map for player location estimation
class BeliefMap:
    # Maintains probabilistic distribution over player position
    # Initialized with uniform prior over all player-passable cells
    # Updated by auditory evidence (heard noise) and visual observation
    # Supports diffusion (player movement model) and observation updates
    def __init__(self, grid):
        # Initialize uniform prior over all player-passable locations
        self.grid = grid
        H, W = grid.shape
        self.belief = np.zeros((H, W), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                if grid[y, x] in PASSABLE_PLAYER:
                    self.belief[y, x] = 1.0
        self._norm()

    def _norm(self):
        # Normalize belief to probability distribution (sum to 1.0)
        t = self.belief.sum()
        if t > 1e-12:
            self.belief /= t

    def diffuse(self, stay=0.5):
        # Spread belief: assume player has 50% chance to stay, 50% to move to adjacent cell
        # This models the alien's uncertainty about where player went
        H, W = self.grid.shape
        out = np.zeros_like(self.belief)
        move = (1 - stay) / 4  # Equal probability to each of 4 directions
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
        # Update belief based on visual observation
        # If player is visible and not hiding: certainty (collapse to single cell)
        # Otherwise: zero out visible non-HIDE cells (player can't be there)
        if player_visible and not player_hiding and player_pos:
            self.belief[:] = 0.0
            self.belief[player_pos[1], player_pos[0]] = 1.0
        else:
            # Eliminate cells that are visible and not hiding spots
            for (vx, vy) in visible:
                if self.grid[vy, vx] != HIDE:
                    self.belief[vy, vx] = 0.0
        self._norm()

    def peak(self):
        # Return the most probable player position from belief distribution
        # Returns None if all belief is zero
        if self.belief.max() < 1e-12:
            return None
        r, c = np.unravel_index(np.argmax(self.belief), self.belief.shape)
        return (int(c), int(r))


# Knowledge map: tracks observed tiles vs unexplored areas
class KnowledgeMap:
    # Like the human agent's map, records what the alien has observed
    # Initially all UNKNOWN; gets populated as FOV reveals tiles
    # Marks PLAYER_SEEN at locations where player was previously observed
    def __init__(self, grid):
        # Initialize knowledge map with all cells as UNKNOWN
        self.grid = grid
        H, W = grid.shape
        self.knowledge = np.full((H, W), UNKNOWN, dtype=np.int16)
        self.seen_vents = set()  # Track observed vent positions

    def update_from_observation(self, visible_cells, grid_map, player_pos, player_visible, player_hiding):
        # Record all tiles from FOV into knowledge map
        # Mark PLAYER_SEEN where player was observed (if visible and not hiding)
        for (vx, vy) in visible_cells:
            if 0 <= vy < grid_map.shape[0] and 0 <= vx < grid_map.shape[1]:
                self.knowledge[vy, vx] = int(grid_map[vy, vx])
                if grid_map[vy, vx] == VENT:
                    self.seen_vents.add((vx, vy))
        
        # Mark player if visible
        if player_visible and not player_hiding and player_pos:
            px, py = player_pos
            if 0 <= py < grid_map.shape[0] and 0 <= px < grid_map.shape[1]:
                self.knowledge[py, px] = PLAYER_SEEN

    def get_unknown_frontier(self):
        # Return list of known-passable cells adjacent to UNKNOWN territory
        # These are priority exploration targets in SEARCH state
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

    def get_seen_vents(self):
        # Return list of (x, y) positions of all observed vents
        # These are potential teleport targets for strategic routing
        return list(self.seen_vents)

    def get_previously_seen_player_area(self):
        # Return list of (x, y) positions where player was previously observed
        # In SEARCH state, re-patrol these areas assuming player might return
        H, W = self.knowledge.shape
        player_areas = []
        for y in range(H):
            for x in range(W):
                if self.knowledge[y, x] == PLAYER_SEEN:
                    player_areas.append((x, y))
        return player_areas

    def get_copy(self):
        # Return independent copy of knowledge state
        # Useful for backup/rollback scenarios
        return self.knowledge.copy()


# Generate patrol route waypoints for SEARCH state
def build_waypoints(grid, n=6, seed=0):
    # Create n well-spaced patrol points across passable terrain
    # Uses greedy spacing: each waypoint at least W/3 cells away from others
    # Provides fallback patrol route when no other objectives are available
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


# Main alien agent implementation
@dataclass
class AlienAgent:
    # Opponent agent with adaptive behavior and strategic vent usage
    # 
    # Perception:
    #   - Visual FOV: ray-cast with HIDE/WALL blocking line-of-sight
    #   - Auditory: hears noise at uncertain position from game events
    #   - Knowledge: tracks observed tiles vs unexplored areas
    #   - Belief: probabilistic distribution over player location
    #
    # Behavior:
    #   - SEARCH (exploration): unknown frontiers, waypoints, previous sightings
    #   - INVESTIGATE (targeted): revisit last known position
    #   - HUNT (pursuit): player visible in FOV, move 2 cells/step
    #
    # Tactics:
    #   - Vent teleportation: strategically jump vents for long-distance pursuit
    #   - Sound pursuit: prioritize navigation toward heard noises
    #   - Fallback movement: greedy stepping when A* pathfinding fails
    #   - Exploration: use knowledge map to focus search on unknown frontiers
    #
    # Parameters:
    #   grid: map array with tile values
    #   start_pos: (x, y) starting position
    #   fov_radius: Manhattan distance of vision
    #   replan_every: replans A* path every N steps (default 3)
    grid:        np.ndarray
    start_pos:   tuple
    fov_radius:  int = 6
    replan_every: int = 3

    pos:              tuple      = field(init=False)
    state:            AlienState = field(init=False)
    knowledge:        KnowledgeMap = field(init=False)
    belief:           BeliefMap = field(init=False)  # Bayesian belief over player position
    last_known_pos:   Optional[tuple] = field(init=False)
    last_heard_pos:   Optional[tuple] = field(init=False)  # Last position where alien heard noise
    steps_since_heard: int = field(init=False)  # Track how old the heard evidence is
    path:             list       = field(init=False)
    waypoints:        list       = field(init=False)
    wp_idx:           int        = field(init=False)
    steps_no_replan:  int        = field(init=False)
    steps_in_state:   int        = field(init=False)
    history:          list       = field(init=False)

    def __post_init__(self):
        # Initialize state after dataclass construction
        self.reset()

    def reset(self, start_pos=None):
        # Reset agent to initial state for new game/episode
        self.pos            = start_pos or self.start_pos
        self.state          = AlienState.SEARCH  # Start by searching unknown areas
        self.knowledge      = KnowledgeMap(self.grid)  # Empty knowledge of map
        self.belief         = BeliefMap(self.grid)  # Uniform prior over player location
        self.last_known_pos = None  # No player sighting yet
        self.last_heard_pos = None  # No sound heard yet
        self.steps_since_heard = 0  # Time since last auditory evidence
        self.path           = []  # No current A* path
        self.waypoints      = build_waypoints(self.grid)  # Patrol route for SEARCH
        self.wp_idx         = 0  # Current waypoint index
        self.steps_no_replan = 0  # Steps since last A* replan
        self.steps_in_state  = 0  # Steps in current state
        self.history         = []  # Step-by-step record for analysis
        self.vent_teleport_used = False  # Flag if teleported this step

    def step(self, player_pos: tuple, heard_pos: tuple = None, step_num: int = 0) -> tuple:
        # Execute one step: perceive → update knowledge → state transition → move
        #
        # Args:
        #   player_pos: Exact player position (used for FOV visual check)
        #   heard_pos: Uncertain position of heard noise (auditory evidence)
        #   step_num: Current game step number
        #
        # Returns:
        #   New alien position after movement(s)
        # Distinguish between exact player position (for FOV) and heard noise position (for auditory tracking)
        if heard_pos is None:
            heard_pos = player_pos
        
        # AUDITORY EVIDENCE TRACKING
        # Detect when a noise is heard distinct from exact player position
        sound_detected = heard_pos != player_pos
        if sound_detected:
            # New sound heard: update tracking and force replanning toward sound
            self.last_heard_pos = heard_pos
            self.steps_since_heard = 0
            self.path = []  # Discard current path to replan toward sound
            self.steps_no_replan = self.replan_every  # Force immediate replanning
        else:
            # No new sound: age existing evidence
            self.steps_since_heard += 1
            if self.steps_since_heard > 5:
                self.last_heard_pos = None  # Forget sounds older than 5 steps
        
        # STEP 1: VISUAL PERCEPTION
        # Compute FOV from current position and check if player is visible
        fov           = compute_fov(self.grid, self.pos, self.fov_radius)
        px, py        = player_pos
        in_fov        = (px, py) in fov
        player_hiding = in_fov and self.grid[py, px] == HIDE  # Player in HIDE tile not visible
        player_seen   = in_fov and not player_hiding  # Visible only if in FOV and not hiding

        # STEP 2: UPDATE KNOWLEDGE MAP
        # Record all observed tiles from FOV into knowledge map
        self.knowledge.update_from_observation(fov, self.grid, player_pos, player_seen, player_hiding)
        
        # STEP 3: UPDATE BELIEF MAP WITH AUDITORY EVIDENCE
        # When noise is heard, boost belief probability around heard location
        if sound_detected:
            hx, hy = heard_pos
            if 0 <= hx < self.grid.shape[1] and 0 <= hy < self.grid.shape[0]:
                # Spread probability in 3x3 grid around heard location
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = hx + dx, hy + dy
                        if 0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]:
                            if self.grid[ny, nx] in PASSABLE_PLAYER:
                                self.belief.belief[ny, nx] += 0.1  # Increase belief at heard area
        
        # Normalize belief back to probability distribution
        self.belief._norm()

        # STEP 4: STATE TRANSITION
        # Check if behavior state should change based on observations
        prev = self.state
        self._transition(player_seen, player_pos)
        if self.state != prev:
            self.steps_in_state = 0
            self.path = []  # Discard old path when state changes
        else:
            self.steps_in_state += 1

        # STEP 4.5: VENT TELEPORTATION
        # Check if strategic teleport through vent is beneficial
        self.vent_teleport_used = False
        if self.last_heard_pos is not None and self.steps_since_heard <= 5:
            target_vent = self._evaluate_vent_teleport(self.last_heard_pos)
            if target_vent:
                self._teleport_to_vent(target_vent)  # Immediate teleport if beneficial

        # STEP 5: MOVEMENT
        # Move 1 cell (SEARCH/INVESTIGATE) or 2 cells (HUNT) based on state
        steps = SPEED[self.state]
        for _ in range(steps):
            self.pos = self._move_one(player_pos, player_seen)

        # Track replanning counter
        self.steps_no_replan += 1
        # Record step for analysis and debugging
        self.history.append({
            "step":              step_num,
            "state":             self.state.name,
            "pos":               self.pos,
            "player_seen":       player_seen,
            "player_hiding":     player_hiding,
            "speed":             steps,
            "dist_to_player":    heuristic(self.pos, player_pos),
            "heard_pos":         heard_pos if sound_detected else None,
            "pursuing_sound":    self.last_heard_pos is not None,
            "vent_teleport_used": self.vent_teleport_used,
        })
        return self.pos

    def _get_explored_ratio(self, center: tuple, radius: int) -> float:
        # Calculate exploration completeness: ratio of known cells to total passable cells
        # Within given radius from center
        # Returns 0.0 (all unknown) to 1.0 (fully explored)
        cx, cy = center
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

    def _transition(self, player_seen: bool, player_pos: tuple):
        # STATE MACHINE: SEARCH ↔ INVESTIGATE ↔ HUNT
        # Transitions:
        #   - Player visible → HUNT
        #   - Lost sight in HUNT → SEARCH unless there is recent sound evidence
        #   - Investigated area thoroughly → back to SEARCH
        #
        # Uses knowledge map (70% exploration threshold) to decide when to abandon investigation
        if player_seen:
            # Player visible: switch to aggressive pursuit
            self.last_known_pos = player_pos
            self.state = AlienState.HUNT
        elif self.state == AlienState.HUNT:
            # Lost sight of player: if no recent sound, resume exploration immediately
            if self.last_heard_pos is None or self.steps_since_heard > 5:
                self.state = AlienState.SEARCH
            else:
                # Recent sound still exists, so investigate the last known area
                self.state = AlienState.INVESTIGATE
        elif self.state == AlienState.INVESTIGATE:
            # Check if investigation is complete
            if self.last_known_pos and heuristic(self.pos, self.last_known_pos) <= 1:
                # At last known position: check if area is thoroughly explored
                explored_ratio = self._get_explored_ratio(self.last_known_pos, 4)
                # If 70%+ of surrounding area is mapped and player not found, return to general search
                if explored_ratio > 0.7:
                    self.state = AlienState.SEARCH
            elif self.last_heard_pos is None or self.steps_since_heard > 5:
                # No fresh evidence remains, so stop investigating and go back to exploration
                self.state = AlienState.SEARCH

    def _best_seen_vent_route_for_sound(self, sound_pos: tuple) -> Optional[tuple]:
        # Find if any observed vent provides strategic advantage for reaching heard sound
        # Returns vent (x,y) if beneficial, else None
        # Gates: sound must be 8+ cells away, vent route must save 4+ steps
        seen_vents = self.knowledge.get_seen_vents()
        if len(seen_vents) < 2:
            return None

        # Only consider vent routing if sound is far away (8+ cells)
        direct_dist = heuristic(self.pos, sound_pos)
        if direct_dist < VENT_ROUTE_MIN_SOUND_DISTANCE:
            return None

        start_vent = min(seen_vents, key=lambda v: heuristic(self.pos, v))
        dest_vent = min(seen_vents, key=lambda v: heuristic(v, sound_pos))
        if start_vent == dest_vent:
            return None

        dist_via_vent = heuristic(self.pos, start_vent) + VENT_TELEPORT_COST + heuristic(dest_vent, sound_pos)
        savings = direct_dist - dist_via_vent
        if savings < VENT_ROUTE_MIN_SAVINGS:
            return None

        return start_vent

    def _evaluate_vent_teleport(self, sound_pos: tuple) -> Optional[tuple]:
        # Decide if immediate vent teleportation is beneficial
        # Precondition: must currently be standing on a vent
        # Returns target vent if teleport is worthwhile, else None
        #
        # Logic:
        #   - Find all observed vents
        #   - Calculate: dist_via_vent = dist(current_vent, other_vent) + dist(other_vent, sound)
        #   - Compare vs direct distance: dist(current_vent, sound)
        #   - If any vent saves 4+ steps (VENT_ROUTE_MIN_SAVINGS), teleport there
        #   - If not on a vent now: return None (walk to vent first, then teleport later)
        # Can only teleport if currently on a vent
        if self.grid[self.pos[1], self.pos[0]] != VENT:
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

        dist_via_vent = VENT_TELEPORT_COST + heuristic(best_vent, sound_pos)
        savings = direct_dist - dist_via_vent
        if savings < VENT_ROUTE_MIN_SAVINGS:
            return None

        return best_vent

    def _teleport_to_vent(self, target_vent: tuple):
        # Execute teleportation to target vent
        # Updates position, marks teleport as used this step, discards current path
        # Move to target vent instantly
        self.pos = target_vent
        self.vent_teleport_used = True  # Mark that teleport was used
        self.path = []  # Discard path and replan from new position
        self.steps_no_replan = self.replan_every  # Force immediate replanning

    def _move_one(self, player_pos: tuple, player_seen: bool) -> tuple:
        # Execute one step of movement (1 cell)
        # Replan path if depleted or replan interval elapsed
        # Fall back to greedy stepping if pathfinding fails
        # Check if path needs replanning
        if not self.path or self.steps_no_replan >= self.replan_every:
            self.path = self._plan_path(player_pos, player_seen)
            self.steps_no_replan = 0

        # Follow current path if available
        if self.path:
            self.path.pop(0)  # Consume starting position
            if self.path:
                return self.path[0]  # Return next step

        # FALLBACK: No path or path exhausted, move greedily toward current objective
        if self.state == AlienState.HUNT:
            fallback_goal = player_pos
        elif self.state == AlienState.INVESTIGATE:
            fallback_goal = self.last_known_pos
        else:  # SEARCH
            fallback_goal = self.last_heard_pos
        
        greedy = self._greedy_step_toward(fallback_goal)
        if greedy is not None:
            return greedy

        # No valid move: stay in place
        return self.pos

    def _greedy_step_toward(self, goal: Optional[tuple]) -> Optional[tuple]:
        # Return one step toward goal that reduces Manhattan distance
        # Uses as fallback when A* pathfinding returns empty path
        if goal is None:
            return None

        best_step = None
        best_dist = heuristic(self.pos, goal)
        x, y = self.pos
        # Try all four cardinal directions
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            # Check bounds
            if not (0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]):
                continue
            # Check passability
            if self.grid[ny, nx] not in PASSABLE_ALIEN:
                continue
            candidate = (nx, ny)
            candidate_dist = heuristic(candidate, goal)
            # Pick step that reduces distance most
            if candidate_dist < best_dist:
                best_dist = candidate_dist
                best_step = candidate

        return best_step

    def _plan_path(self, player_pos: tuple, player_seen: bool) -> list:
        # Compute A* path based on current state and knowledge
        # Priority hierarchy determines goal based on available information
        # DEFAULT: stay put
        goal = self.pos
        
        # PRIORITY 1: HUNT - Player is visible
        if self.state == AlienState.HUNT:
            goal = player_pos
        
        # PRIORITY 2: SOUND PURSUIT - Recent auditory evidence
        elif self.last_heard_pos is not None and self.steps_since_heard <= 5:
            current_is_vent = self.grid[self.pos[1], self.pos[0]] == VENT
            best_sound_vent = self._best_seen_vent_route_for_sound(self.last_heard_pos)
            
            if not current_is_vent:
                # Not on vent: move to beneficial vent or directly to sound
                if best_sound_vent is not None:
                    goal = best_sound_vent  # Intermediate goal: reach vent
                else:
                    goal = self.last_heard_pos  # Direct pursuit of sound
            else:
                # Already on vent: pursue sound directly (teleport will happen separately)
                goal = self.last_heard_pos
        
        # PRIORITY 3: INVESTIGATE - Check last known player position
        elif self.state == AlienState.INVESTIGATE:
            goal = self.last_known_pos or self.pos
        
        # PRIORITY 4: SEARCH - Explore unknown areas
        else:  # SEARCH state
            # Sub-priorities for exploration:
            # 4.1: Unknown frontiers (edges of explored territory)
            frontiers = self.knowledge.get_unknown_frontier()
            if frontiers:
                goal = min(frontiers, key=lambda f: heuristic(self.pos, f))
            else:
                # 4.2: Recent sound location (within 10 steps)
                if self.last_heard_pos is not None and self.steps_since_heard <= 10:
                    goal = self.last_heard_pos
                else:
                    # 4.3: Areas where player was previously seen (player might return)
                    prev_player_areas = self.knowledge.get_previously_seen_player_area()
                    if prev_player_areas:
                        goal = min(prev_player_areas, key=lambda p: heuristic(self.pos, p))
                    # 4.4: Follow patrol waypoints
                    elif self.waypoints:
                        goal = self.waypoints[self.wp_idx % len(self.waypoints)]
                        self.wp_idx += 1
                    # 4.5: No objectives available, stay put
                    else:
                        goal = self.pos

        # Compute and return A* path to goal
        return astar(self.grid, self.pos, goal, PASSABLE_ALIEN)