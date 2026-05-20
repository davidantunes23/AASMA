import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


# Simple random-moving human baseline for evaluation
PASSABLE_HUMAN = {1, 2, 3, 4, 5, 6}


class Direction(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class Action(Enum):
    WALK = 1
    WAIT = 2


@dataclass
class RandomHumanAgent:
    # Minimal human agent that moves randomly among passable neighbours.

    grid: np.ndarray  # The world grid, used for determining valid moves.
    pos: Tuple[int, int]  # Current position (y, x) of the human.
    rng: random.Random = field(default_factory=random.Random, repr=False)

    # Reset position (e.g. at start of new episode)
    def reset(self, start_pos: Optional[Tuple[int, int]] = None):
        if start_pos is not None:
            self.pos = start_pos

    # Main method to determine next move based on current position and grid knowledge.
    def step(self, player_pos: tuple, heard_pos: tuple = None, step_num: int = 0) -> tuple:
        y, x = self.pos
        height, width = self.grid.shape
        neighbours = []

        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and self.grid[ny, nx] in PASSABLE_HUMAN:
                neighbours.append((ny, nx))

        actions = [Action.WALK, Action.WAIT]
        choice = self.rng.choice(actions)
        if choice == Action.WALK and neighbours:
            self.pos = self.rng.choice(neighbours)
        self.hidden = bool(self.grid[self.pos] == 3)
        return self.pos
