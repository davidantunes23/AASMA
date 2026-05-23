from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from enum import IntEnum

import numpy as np


class Direction(Enum):
    NORTH = 1
    EAST  = 2
    SOUTH = 3
    WEST  = 4


def cone_fov(
    grid: np.ndarray,
    origin_yx: tuple[int, int],
    direction: Direction,
    view_length: int,
) -> set[tuple[int, int]]:
    """Return set of (y, x) positions visible from origin_yx in a forward cone."""
    H, W = grid.shape
    BLOCKERS = {0, 3}  # WALL=0, HIDE=3
    oy, ox = origin_yx
    visible: set[tuple[int, int]] = {origin_yx}
    for depth in range(1, view_length + 1):
        for lateral in range(-depth, depth + 1):
            ty, tx = _cone_target(oy, ox, direction, depth, lateral)
            if not (0 <= ty < H and 0 <= tx < W):
                continue
            if _has_los(grid, (oy, ox), (ty, tx), BLOCKERS):
                visible.add((ty, tx))
    return visible


def _cone_target(oy: int, ox: int, direction: Direction, depth: int, lateral: int) -> tuple[int, int]:
    if direction == Direction.NORTH:
        return (oy - depth, ox + lateral)
    if direction == Direction.EAST:
        return (oy + lateral, ox + depth)
    if direction == Direction.SOUTH:
        return (oy + depth, ox + lateral)
    return (oy + lateral, ox - depth)  # WEST


def _has_los(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], blockers: set) -> bool:
    for y, x in _bresenham(start, end)[1:-1]:
        if grid[y, x] in blockers:
            return False
    return True


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y0, x0 = start
    y1, x1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
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


def direction_from_delta(dy: int, dx: int) -> Direction:
    """Derive facing direction from a (dy, dx) movement delta."""
    if dy == -1:
        return Direction.NORTH
    if dy == 1:
        return Direction.SOUTH
    if dx == 1:
        return Direction.EAST
    return Direction.WEST


class BaseAgent(ABC):
    """Common interface for all agents. All positions are (y, x) / (row, col)."""

    pos: tuple[int, int]
    direction: Direction
    view_length: int

    @abstractmethod
    def step(
        self,
        player_pos: tuple[int, int],
        heard_pos: tuple[int, int] | None = None,
        step_num: int = 0,
    ) -> tuple[int, int]:
        """Execute one step. Returns new (y, x) position."""
        ...

    def reset(self, start_pos: tuple[int, int] | None = None) -> None:
        if start_pos is not None:
            self.pos = start_pos


class BaseAlienAgent(BaseAgent):
    """Alien-role agents. pos=(y,x), directional cone FOV. Aliens cannot hide."""

    grid: np.ndarray


class BaseHumanAgent(BaseAgent):
    """Human-role agents. pos=(y,x), directional cone FOV, can hide.

    The simulation calls observe() once per step before calling step(), giving
    the agent its current observation and radar state.
    """

    hidden: bool = False
    exit_open: bool = False
    # Team role (WORKER / RUNNER / DECOY / etc.). May be None when unassigned.
    team_role: "TeamRole | None" = None

    def observe(
        self,
        obs: np.ndarray,
        radar_threat: str | None = None,
        radar_dist: int | None = None,
    ) -> None:
        """Ingest this step's observation. No-op by default (random agents)."""
        pass


class TeamRole(IntEnum):
    NONE = 0
    WORKER = 1
    RUNNER = 2
    DECOY = 3
    SCOUT = 4
