import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.base import BaseHumanAgent, Direction, direction_from_delta

PASSABLE_HUMAN = {1, 2, 3, 4, 5, 6}


@dataclass
class RandomHumanAgent(BaseHumanAgent):
    """Random-walking human baseline. All positions are (y, x)."""

    grid: np.ndarray
    pos: tuple
    view_length: int = 6
    rng: random.Random = field(default_factory=random.Random, repr=False)
    direction: Direction = field(default=Direction.NORTH, init=False)
    hidden: bool = field(default=False, init=False)
    exit_open: bool = field(default=False, init=False)

    def reset(self, start_pos: Optional[tuple] = None):
        if start_pos is not None:
            self.pos = start_pos

    def step(self, _player_pos: tuple, _heard_pos: tuple = None, _step_num: int = 0) -> tuple:
        if self.exit_open and self.grid[self.pos] == 6:
            return self.pos

        y, x = self.pos
        height, width = self.grid.shape
        neighbours = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and self.grid[ny, nx] in PASSABLE_HUMAN:
                if self.grid[ny, nx] == 6 and not self.exit_open:
                    continue
                neighbours.append((ny, nx))

        old_pos = self.pos
        if self.rng.choice(["walk", "wait"]) == "walk" and neighbours:
            self.pos = self.rng.choice(neighbours)

        if self.pos != old_pos:
            self.direction = direction_from_delta(self.pos[0] - old_pos[0], self.pos[1] - old_pos[1])
        self.hidden = bool(self.grid[self.pos] == 3)
        return self.pos
