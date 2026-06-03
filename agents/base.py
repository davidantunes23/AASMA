"""Base classes and shared utilities for all agents.

Coordinate convention: all positions are (y, x) / (row, col) throughout.

Exports:
    Direction, TeamRole          - enums used by every agent
    cone_fov()                   - directional FOV with wall-blocked line-of-sight
    direction_from_delta()       - derive Direction from a movement delta
    BaseAgent, BaseAlienAgent, BaseHumanAgent  - abstract agent hierarchy
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, IntEnum

import numpy as np


class Direction(Enum):
    NORTH = 1 
    EAST  = 2
    SOUTH = 3
    WEST  = 4


class TeamRole(IntEnum):
    NONE   = 0
    WORKER = 1  # completes mission tiles
    RUNNER = 2  # stages near exit, escapes when safe
    DECOY  = 3  # draws alien away from the team


# ── Field-of-view ─────────────────────────────────────────────────────────────

def cone_fov(
    grid: np.ndarray,
    origin_yx: tuple[int, int],
    direction: Direction,
    view_length: int,
) -> set[tuple[int, int]]:
    """Return all (y, x) cells visible from *origin_yx* in a forward cone.
    
    The cone spans *view_length* rows deep and widens by one cell per row.
    Walls (tile 0) and hide spots (tile 3) block line-of-sight.
    """
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
    """True iff no blocker tile lies on the Bresenham line between start and end."""
    for y, x in _bresenham(start, end)[1:-1]:  # skip endpoints
        if grid[y, x] in blockers:
            return False
    return True


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Return all integer cells on the line from start to end (inclusive)."""
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
    """Derive the facing Direction from a single-step (dy, dx) delta."""
    if dy == -1:
        return Direction.NORTH
    if dy == 1:
        return Direction.SOUTH
    if dx == 1:
        return Direction.EAST
    return Direction.WEST


# ── Agent base classes ────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for every agent. All positions are (y, x) / (row, col)."""

    def __init__(self, pos: tuple[int, int], direction: Direction, view_length: int = 6) -> None:
        self.pos = pos                  # current (row, col) position
        self.direction = direction      # facing direction, determines FOV cone orientation
        self.view_length = view_length  # how many cells deep the FOV cone reaches
        self.hidden = False             # True when standing on a HIDE tile; read by simulation on all agents

    @abstractmethod
    def step(
        self,
        player_pos: tuple[int, int],
        heard_pos: tuple[int, int] | None = None,
        step_num: int = 0,
    ) -> tuple[int, int]:
        """Advance one simulation step. Returns the agent's new (y, x) position."""
        ...

    def observe(
        self,
        obs: np.ndarray,
        **kwargs,
    ) -> None:
        """Ingest this step's observation array. No-op by default."""

    def reset(self, start_pos: tuple[int, int] | None = None) -> None:
        if start_pos is not None:
            self.pos = start_pos


class BaseAlienAgent(BaseAgent):
    """Alien-role agents. Build map knowledge incrementally from cone observations, same as human agents."""


class BaseHumanAgent(BaseAgent):
    """Human-role agents. Can hide; receives radar and observation updates each step.

    The simulation calls ``observe()`` before ``step()`` every turn so the agent
    can update its internal knowledge map and radar state first.
    """

    def __init__(self, pos: tuple[int, int], direction: Direction = Direction.NORTH, view_length: int = 6) -> None:
        super().__init__(pos, direction, view_length)
        self.exit_open: bool = False           # set by simulation once all missions are completed
        self.team_role: "TeamRole | None" = None  # assigned role (WORKER/RUNNER/DECOY); None if uncoordinated
