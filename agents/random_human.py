"""Random-walking human agent used as a lower-bound baseline for evaluation."""
import random
from typing import Optional

import numpy as np

from agents.base import BaseHumanAgent, Direction, direction_from_delta

UNKNOWN = -1

# Tile IDs a human can walk on
PASSABLE_HUMAN = {1, 2, 3, 4, 5, 6, 7}


class RandomHumanAgent(BaseHumanAgent):
    """Human that moves randomly each step, with no strategy or coordination.

    Each turn it picks uniformly between walking to a random adjacent cell
    and waiting. The exit tile is blocked until ``exit_open`` is set by the
    simulation (i.e. all missions completed).
    Knowledge of the map is built incrementally from cone observations —
    the agent can only move to tiles it has already seen.
    Used as a lower-bound baseline; its escape rate reflects pure luck.
    """

    def __init__(self, pos: tuple, view_length: int = 6, rng: Optional[random.Random] = None):
        super().__init__(pos=pos, direction=Direction.NORTH, view_length=view_length)
        self.rng = rng if rng is not None else random.Random()  # per-agent RNG
        self._known_map: Optional[np.ndarray] = None  # tile map built from cone observations
        self._seen_vents: set = set()                 # vent positions discovered so far

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self, start_pos: Optional[tuple] = None) -> None:
        if start_pos is not None:
            self.pos = start_pos
        self._known_map = None
        self._seen_vents = set()

    # ── Observation ───────────────────────────────────────────────────────────

    def observe(self, obs: np.ndarray, _radar_threat=None, _radar_dist=None) -> None:
        """Ingest cone observation, updating the known map and vent registry."""
        if self._known_map is None:
            self._known_map = np.full(obs.shape, UNKNOWN, dtype=np.int16)
        visible_mask = obs >= 0
        self._known_map[visible_mask] = obs[visible_mask]
        vy, vx = np.where(obs == 2)  # VENT=2
        for y, x in zip(vy.tolist(), vx.tolist()):
            self._seen_vents.add((int(y), int(x)))

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, _human_pos: tuple, _heard_pos: tuple = None, _step_num: int = 0) -> tuple:
        if self._known_map is None:
            return self.pos  # no observation yet — stay put

        # Stay on the exit once reached — simulation will register the escape
        if self.exit_open and self._known_map[self.pos] == 6:
            return self.pos

        y, x = self.pos
        H, W = self._known_map.shape

        # Collect passable adjacent cells from observed map, excluding the locked exit
        neighbours = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and self._known_map[ny, nx] in PASSABLE_HUMAN:
                if self._known_map[ny, nx] == 6 and not self.exit_open:
                    continue  # treat locked exit as impassable
                neighbours.append((ny, nx))

        old_pos = self.pos
        if self.rng.choice(["walk", "wait"]) == "walk" and neighbours:
            self.pos = self.rng.choice(neighbours)

        if self.pos != old_pos:
            self.direction = direction_from_delta(self.pos[0] - old_pos[0], self.pos[1] - old_pos[1])
        self.hidden = bool(self._known_map[self.pos] == 3)  # HIDE=3
        return self.pos
