"""Random-walking alien agent used as a baseline for evaluation."""
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.base import BaseAlienAgent, Direction, direction_from_delta

UNKNOWN = -1

# Tile IDs the alien can walk on (excludes WALL=0, HIDE=3)
PASSABLE_ALIEN = {1, 2, 4, 5, 6}


@dataclass
class RandomAlienAgent(BaseAlienAgent):
    """Alien that moves randomly each step.

    Each turn it picks uniformly among: walk to a random adjacent cell, wait,
    or teleport to a previously seen vent (if currently standing on one).
    Knowledge of the map is built incrementally from cone observations — the agent
    can only move to tiles it has already seen.
    Used as a lower-bound baseline — no pursuit, no memory of players.
    """

    pos: tuple                   # current (y, x) position
    view_length: int = 6         # FOV radius read by the simulation to build observations
    rng: random.Random = field(default_factory=random.Random, repr=False)  # per-agent RNG
    direction: Direction = field(default=Direction.SOUTH, init=False)  # current facing direction
    hidden: bool = field(default=False, init=False)      # aliens never hide; kept for interface compatibility
    _known_map: Optional[np.ndarray] = field(default=None, init=False, repr=False)  # observed tile map
    seen_vents: set = field(default_factory=set, init=False, repr=False)  # vent positions discovered so far

    def reset(self, start_pos: Optional[tuple] = None):
        if start_pos is not None:
            self.pos = start_pos
        self._known_map = None
        self.seen_vents = set()

    def observe(self, obs: np.ndarray, _opponent_positions=None) -> None:
        """Ingest cone observation, updating the known map and vent registry."""
        if self._known_map is None:
            self._known_map = np.full(obs.shape, UNKNOWN, dtype=np.int16)
        visible_mask = obs >= 0
        self._known_map[visible_mask] = obs[visible_mask]
        vy, vx = np.where(obs == 2)  # VENT=2
        for y, x in zip(vy.tolist(), vx.tolist()):
            self.seen_vents.add((int(y), int(x)))

    def step(self, _player_pos: tuple, _heard_pos: tuple = None, _step_num: int = 0) -> tuple:
        if self._known_map is None:
            return self.pos  # no observation yet — stay put

        y, x = self.pos
        H, W = self._known_map.shape

        # Collect walkable neighbours from observed map
        neighbours = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and self._known_map[ny, nx] in PASSABLE_ALIEN:
                neighbours.append((ny, nx))

        # Vent teleport is only available when standing on a known vent with others known
        actions = ["walk", "wait"]
        other_vents = []
        if self._known_map[y, x] == 2 and len(self.seen_vents) > 1:
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
