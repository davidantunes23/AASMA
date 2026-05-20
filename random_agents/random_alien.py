import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


# Simple random-moving alien baseline for evaluation
PASSABLE_ALIEN = {1, 2, 4, 5, 6}


class Direction(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class Action(Enum):
    WALK = 1
    WAIT = 2
    VENT = 3


@dataclass
class RandomAlienAgent:
    # Minimal alien agent that moves randomly among passable neighbours.

    grid: np.ndarray # The world grid, used for determining valid moves and vents.
    pos: Tuple[int, int] # Current position (x, y) of the alien.
    fov_radius: int = 6 # Radius for "seeing" vents in the grid.
    rng: random.Random = field(default_factory=random.Random, repr=False) # Random number generator for reproducibility.
    seen_vents: set = field(default_factory=set, init=False, repr=False) # Set of vent positions seen so far.

    # Reset position (e.g. at start of new episode)
    def reset(self, start_pos: Optional[Tuple[int, int]] = None):
        if start_pos is not None:
            self.pos = start_pos

    # Main method to determine next action based on current position and grid knowledge.
    def step(self, player_pos: tuple, heard_pos: tuple = None, step_num: int = 0) -> tuple:
        

        x, y = self.pos
        # Update knowledge of vents within FOV
        for dy in range(-self.fov_radius, self.fov_radius + 1):
            for dx in range(-self.fov_radius, self.fov_radius + 1):
                if abs(dx) + abs(dy) > self.fov_radius: 
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]:
                    if self.grid[ny, nx] == 2:  # VENT tile 
                        self.seen_vents.add((nx, ny)) 
        
        # Determine valid neighbouring positions for walking
        H, W = self.grid.shape
        neighbours = []

        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and self.grid[ny, nx] in PASSABLE_ALIEN:
                neighbours.append((nx, ny))

        # Build available actions. Include VENT only when on a vent and another vent is known.
        actions = [Action.WALK, Action.WAIT]
        other_vents = []
        if self.grid[y, x] == 2 and len(self.seen_vents) > 1:
            other_vents = [v for v in self.seen_vents if v != (x, y)]
            if other_vents:
                actions.append(Action.VENT)

        # Choose an action uniformly among available actions
        choice = self.rng.choice(actions)
        if choice == Action.VENT and other_vents:
            self.pos = self.rng.choice(other_vents)
            return self.pos
        if choice == Action.WALK:
            if neighbours:
                self.pos = self.rng.choice(neighbours)
            return self.pos

        # WAIT: do nothing
        return self.pos