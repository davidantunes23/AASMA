import numpy as np
from agents.alien import AlienAgent
from agents.human import HumanAgent, Action, Direction
from map_generator import Tile


class Game:
    UNSEEN_TILE = -1

    def __init__(
        self,
        map: np.ndarray,
        human_agent: HumanAgent,
        alien_agent: AlienAgent,
        human_view_length: int = 6,
    ):
        self.map = map
        self.human_agent = human_agent
        self.alien_agent = alien_agent
        self.human_pos = human_agent.position
        self.alien_pos = alien_agent.pos
        self.human_view_length = max(0, human_view_length)
        self.step_num = 0
        human_agent._init_memory(self._human_cone_observation())

    def _step(self):       
        human_action = self.human_agent._act(self._human_cone_observation())

        match human_action[0]:
            case Action.WALK:
                self.human_pos = self._walk(self.human_pos, human_action[1])
                self.human_agent.position = self.human_pos

        # Convert human_pos from (y, x) to (x, y) for alien agent, then convert result back
        human_x, human_y = self.human_pos[1], self.human_pos[0]
        alien_x, alien_y = self.alien_agent.step((human_x, human_y), self.step_num)
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

        ay, ax = self.alien_pos
        if self._in_bounds(ay, ax) and obs[ay, ax] != self.UNSEEN_TILE:
            obs[ay, ax] = HumanAgent.ALIEN

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
