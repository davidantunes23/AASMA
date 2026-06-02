"""Random-walking alien agent used as a baseline for evaluation."""
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.base import BaseAlienAgent, Direction, direction_from_delta

# Tile IDs the alien can walk on (excludes WALL=0, HIDE=3)
PASSABLE_ALIEN = {1, 2, 4, 5, 6}


@dataclass
class RandomAlienAgent(BaseAlienAgent):
    """Alien that moves randomly each step.

    Each turn it picks uniformly among: walk to a random adjacent cell, wait,
    or teleport to a previously seen vent (if currently standing on one).
    Used as a lower-bound baseline — no pursuit, no memory of players.
    """

    _grid: np.ndarray          # physical map — tile lookups and bounds checks only
    pos: tuple                 # current (y, x) position
    view_length: int = 6       # FOV radius used for vent discovery and read by the simulation
    rng: random.Random = field(default_factory=random.Random, repr=False)  # per-agent RNG
    direction: Direction = field(default=Direction.SOUTH, init=False)  # current facing direction
    hidden: bool = field(default=False, init=False)      # aliens never hide; kept for interface compatibility
    seen_vents: set = field(default_factory=set, init=False, repr=False)  # vent positions discovered so far

    def reset(self, start_pos: Optional[tuple] = None):
        if start_pos is not None:
            self.pos = start_pos

    def step(self, player_pos: tuple, heard_pos: tuple = None, step_num: int = 0) -> tuple:
        y, x = self.pos

        # Discover any vent tiles within view range (Manhattan distance)
        for dy in range(-self.view_length, self.view_length + 1):
            for dx in range(-self.view_length, self.view_length + 1):
                if abs(dx) + abs(dy) > self.view_length:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self._grid.shape[0] and 0 <= nx < self._grid.shape[1]:
                    if self._grid[ny, nx] == 2:  # VENT
                        self.seen_vents.add((ny, nx))

        # Collect walkable neighbours
        H, W = self._grid.shape
        neighbours = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and self._grid[ny, nx] in PASSABLE_ALIEN:
                neighbours.append((ny, nx))

        # Vent teleport is only available when standing on a vent with others known
        actions = ["walk", "wait"]
        other_vents = []
        if self._grid[y, x] == 2 and len(self.seen_vents) > 1:
            other_vents = [v for v in self.seen_vents if v != (y, x)]
            if other_vents:
                actions.append("vent")

        old_pos = self.pos
        choice = self.rng.choice(actions)
        if choice == "vent" and other_vents:
            self.pos = self.rng.choice(other_vents)
        elif choice == "walk" and neighbours:
            self.pos = self.rng.choice(neighbours)

        if self.pos != old_pos:
            self.direction = direction_from_delta(self.pos[0] - old_pos[0], self.pos[1] - old_pos[1])
        return self.pos
