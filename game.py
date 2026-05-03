from collections import deque

import numpy as np

from agents.alien import AlienAgent
from agents.human import Action, Direction, HumanAgent
from map_generator import Tile


class Game:
    UNSEEN_TILE = -1
    RADAR_PING = -3      # Radar distance indicator
    NOISE_RIPPLE = -4    # Noise event ripple (visual marker)

    # Radar threat levels based on topology-aware distance
    # Critical is intentionally wider because the alien accelerates when it sees the player
    RADAR_BANDS = {
        "CRITICAL": (0, 7),
        "CLOSE": (8, 12),
        "NEAR": (13, 18),
        "FAR": (19, float('inf')),
    }

    def __init__(
        self,
        map: np.ndarray,
        human_agent: HumanAgent,
        alien_agent: AlienAgent,
        human_view_length: int = 6,
        p_noise: float = 0.1,
        radar_interval: int = 5,
    ):
        self.map = map
        self.human_agent = human_agent
        self.alien_agent = alien_agent
        self.human_pos = human_agent.position
        self.alien_pos = alien_agent.pos
        self.human_view_length = max(0, human_view_length)
        self.step_num = 0
        
        # Noise and radar parameters
        self.p_noise = p_noise          # Probability of creating noise per step [0.05-0.15]
        self.radar_interval = radar_interval  # Radar ping interval in steps (default 5)
        self.steps_since_radar = 0      # Counter for radar pings
        self.radar_active_for = 0       # How many more steps the radar remains active (persistence)
        self.last_radar_threat = None   # Last radar threat level (CRITICAL/CLOSE/NEAR/FAR)
        self.last_radar_dist = None     # Last radar distance measurement
        self.last_heard_pos = None      # Last position where alien heard noise
        self.last_noise_ripple = None   # Position of last noise ripple (for visualization)
        self.noise_ripple_age = 0       # Age of noise ripple (fades over time)
        
        human_agent._init_memory(self._human_cone_observation())

    def _topology_distance(self, start: tuple[int, int], goal: tuple[int, int]) -> int:
        if start == goal:
            return 0

        frontier = deque([(start, 0)])
        visited = {start}

        while frontier:
            (y, x), dist = frontier.popleft()
            for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                ny, nx = y + dy, x + dx
                next_pos = (ny, nx)
                if not self._in_bounds(ny, nx):
                    continue
                if next_pos in visited:
                    continue
                if self.map[ny, nx] == Tile.WALL:
                    continue
                if next_pos == goal:
                    return dist + 1
                visited.add(next_pos)
                frontier.append((next_pos, dist + 1))

        human_y, human_x = start
        alien_y, alien_x = goal
        return abs(human_y - alien_y) + abs(human_x - alien_x)

    def _step(self):
        # === UPDATE RADAR FIRST (before player acts) ===
        # So player can make decisions based on current radar state
        self.steps_since_radar += 1
        if self.steps_since_radar >= self.radar_interval:
            self.steps_since_radar = 0
            # Player receives a radar ping with threat level based on topology-aware distance
            dist = self._topology_distance(self.human_pos, self.alien_pos)
            for threat_level, (min_d, max_d) in self.RADAR_BANDS.items():
                if min_d <= dist <= max_d:
                    # Store radar info and set persistence (radar is active for 2-3 steps)
                    self.last_radar_threat = threat_level
                    self.last_radar_dist = dist
                    self.radar_active_for = 2  # Radar persists for 2 more steps after ping
                    break
        else:
            # Age out radar persistence
            if self.radar_active_for > 0:
                self.radar_active_for -= 1
                # Keep radar active during persistence
            else:
                # Radar completely cleared
                self.last_radar_threat = None
                self.last_radar_dist = None
        
        # === PLAYER ACTION (with current radar state) ===
        human_action = self.human_agent._act(self._human_cone_observation(), self.last_radar_threat, self.last_radar_dist)

        match human_action[0]:
            case Action.WALK:
                self.human_pos = self._walk(self.human_pos, human_action[1])
                self.human_agent.position = self.human_pos
            case Action.WAIT:
                self.human_agent.position = self.human_pos

        # Convert human_pos from (y, x) to (x, y) for alien agent, then convert result back
        human_x, human_y = self.human_pos[1], self.human_pos[0]
        
        # === NOISE GENERATION (Auditory Evidence) ===
        # Player has p_noise probability of creating a noise event each step
        # BUT: player produces no sound if hiding
        heard_pos = (human_x, human_y)  # Default: exact position
        if not self.human_agent.hidden and np.random.random() < self.p_noise:
            # Noise occurs! Add uncertainty
            noise_offset_x = np.random.randint(-4, 5)  # -5 to +5
            noise_offset_y = np.random.randint(-4, 5)
            heard_x = max(0, min(human_x + noise_offset_x, self.map.shape[1] - 1))
            heard_y = max(0, min(human_y + noise_offset_y, self.map.shape[0] - 1))
            heard_pos = (heard_x, heard_y)
            self.last_heard_pos = heard_pos
            self.last_noise_ripple = (human_y, human_x)  # Store ripple center in (y,x)
            self.noise_ripple_age = 0
        
        # Age the noise ripple
        if self.last_noise_ripple is not None:
            self.noise_ripple_age += 1
            if self.noise_ripple_age > 2:  # Ripple fades after 2 steps
                self.last_noise_ripple = None
        
        # Pass heard position to alien (noisy auditory evidence)
        # But alien also gets exact visual observation from FOV
        alien_x, alien_y = self.alien_agent.step((human_x, human_y), heard_pos, self.step_num)
        self.alien_pos = (alien_y, alien_x)  # Convert back to (y, x)
        self.step_num += 1

    def _walk(self, position: tuple[int, int], direction: Direction) -> tuple[int, int]:
        new_position = position
        match direction:
            case Direction.NORTH:
                new_position = (position[0] - 1, position[1])
            case Direction.EAST:
                new_position = (position[0], position[1] + 1)
            case Direction.SOUTH:
                new_position = (position[0] + 1, position[1])
            case Direction.WEST:
                new_position = (position[0], position[1] - 1)
        if self._in_bounds(*new_position) and self.map[new_position] != Tile.WALL:
            return new_position
        return position

    def _in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.map.shape[0] and 0 <= x < self.map.shape[1]

    def _human_cone_observation(self) -> np.ndarray:
        obs = np.full(self.map.shape, self.UNSEEN_TILE, dtype=np.int16)

        hy, hx = self.human_pos
        if not self._in_bounds(hy, hx):
            return obs

        look_direction = self.human_agent._get_direction()
        obs[hy, hx] = int(self.map[hy, hx])

        for depth in range(1, self.human_view_length + 1):
            for lateral in range(-depth, depth + 1):
                ty, tx = self._cone_target(hy, hx, look_direction, depth, lateral)
                if not self._in_bounds(ty, tx):
                    continue
                if self._has_line_of_sight((hy, hx), (ty, tx)):
                    obs[ty, tx] = int(self.map[ty, tx])
        
        # === NOISE RIPPLE VISUALIZATION ===
        # Show yellow ripple around player's true position when noise occurs
        if self.last_noise_ripple is not None:
            ny, nx = self.last_noise_ripple
            # Draw ripple in a small radius around noise center
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ry, rx = ny + dy, nx + dx
                    if self._in_bounds(ry, rx) and obs[ry, rx] != self.UNSEEN_TILE:
                        # Optionally show ripple (for debugging/visualization)
                        # obs[ry, rx] = self.NOISE_RIPPLE  # Uncomment for ripple visualization
                        pass

        # === RADAR INDICATOR AT PLAYER POSITION ===
        # When radar pings (every N steps), show indicator at player's position
        # This tells the player "alien detected nearby" but not exact location
        if self.last_radar_threat is not None:
            obs[hy, hx] = self.RADAR_PING  # Override with radar ping indicator

        return obs

    def _cone_target(
        self,
        oy: int,
        ox: int,
        direction,
        depth: int,
        lateral: int,
    ) -> tuple[int, int]:
        if direction == Direction.NORTH:
            return (oy - depth, ox + lateral)
        if direction == Direction.EAST:
            return (oy + lateral, ox + depth)
        if direction == Direction.SOUTH:
            return (oy + depth, ox + lateral)
        if direction == Direction.WEST:
            return (oy + lateral, ox - depth)
        return (oy, ox)

    def _has_line_of_sight(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        cells = self._bresenham_cells(start, end)
        for y, x in cells[1:-1]:
            if self.map[y, x] == Tile.WALL:
                return False
        return True

    def _bresenham_cells(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        y0, x0 = start
        y1, x1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cells: list[tuple[int, int]] = []
        while True:
            cells.append((y0, x0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells
