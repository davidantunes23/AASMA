"""Random-walking human agent used as a lower-bound baseline for evaluation."""
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.base import BaseHumanAgent, Direction, direction_from_delta

# Tile IDs a human can walk on )
PASSABLE_HUMAN = {1, 2, 3, 4, 5, 6}


@dataclass
class RandomHumanAgent(BaseHumanAgent):
    """Human that moves randomly each step, with no strategy or coordination.

    Each turn it picks uniformly between walking to a random adjacent cell
    and waiting. The exit tile is blocked until ``exit_open`` is set by the
    simulation (i.e. all missions completed). Has no vent teleportation —
    unlike the alien, humans are not assumed to know the vent network.
    Used as a lower-bound baseline; its escape rate reflects pure luck.
    """

    grid: np.ndarray           # static map used for tile lookups and bounds checks
    pos: tuple                 # current (y, x) position
    view_length: int = 6       # FOV radius read by the simulation to build observations
    rng: random.Random = field(default_factory=random.Random, repr=False)  # per-agent RNG
    direction: Direction = field(default=Direction.NORTH, init=False)  # current facing direction
    hidden: bool = field(default=False, init=False)      # True when standing on a HIDE tile
    exit_open: bool = field(default=False, init=False)   # set by simulation once all missions done

    def reset(self, start_pos: Optional[tuple] = None):
        if start_pos is not None:
            self.pos = start_pos

    def step(self, _player_pos: tuple, _heard_pos: tuple = None, _step_num: int = 0) -> tuple:
        # Stay on the exit once reached — simulation will register the escape
        if self.exit_open and self.grid[self.pos] == 6:
            return self.pos

        y, x = self.pos
        height, width = self.grid.shape

        # Collect all passable adjacent cells, excluding the locked exit
        neighbours = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and self.grid[ny, nx] in PASSABLE_HUMAN:
                if self.grid[ny, nx] == 6 and not self.exit_open:
                    continue  # treat locked exit as impassable
                neighbours.append((ny, nx))

        old_pos = self.pos
        if self.rng.choice(["walk", "wait"]) == "walk" and neighbours:
            self.pos = self.rng.choice(neighbours)

        # Update facing direction and hidden state after moving
        if self.pos != old_pos:
            self.direction = direction_from_delta(self.pos[0] - old_pos[0], self.pos[1] - old_pos[1])
        self.hidden = bool(self.grid[self.pos] == 3)  # HIDE=3
        return self.pos
