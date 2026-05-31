from __future__ import annotations

import numpy as np

from agents.base import Direction, TeamRole
from agents.role_human import RoleHumanAgent
from map_generator import Tile


class OmniscientHumanAgent(RoleHumanAgent):
    """Upper-bound human agent: full map, all missions, and exit pre-known.

    Alien position is NOT included — the agent still reacts to radar threats.
    No RUNNER role: exit is always known so exploration-for-exit is obsolete.
    """

    def __init__(
        self,
        grid: np.ndarray,
        start_pos: tuple[int, int],
        start_dir: Direction = Direction.NORTH,
        view_length: int = 6,
    ) -> None:
        super().__init__(start_pos, start_dir, view_length)

        # Store grid for reset().
        self._grid = grid.copy()

        # Override initial role — simulation will assign optimally on first add_mission.
        self.team_role = TeamRole.NONE

        self._apply_omniscience(grid)

    def _apply_omniscience(self, grid: np.ndarray) -> None:
        self._known_map = grid.copy().astype(np.int16)

        ey, ex = np.where(grid == int(Tile.EXIT))
        self._known_exit = (int(ey[0]), int(ex[0])) if len(ey) > 0 else None

        self.mission_positions = []
        self._known_mission_coords = set()
        my, mx = np.where(grid == int(Tile.MISSION))
        for y, x in zip(my.tolist(), mx.tolist()):
            pos = (int(y), int(x))
            self._known_mission_coords.add(pos)
            self.mission_positions.append(pos)

    # ── observe: radar only, no map learning ─────────────────────────────────

    def observe(
        self,
        obs: np.ndarray,
        radar_threat: str | None = None,
        radar_dist: int | None = None,
    ) -> None:
        if radar_threat is not None:
            self.last_radar_threat = radar_threat
            self.last_radar_dist = radar_dist
        self.hidden = self._tile_at(self.pos) == int(Tile.HIDE)
        radar_active = np.any(obs == self.RADAR_PING)
        self._observed_aliens = {self.pos} if radar_active else set()

    # ── remove_mission: also update _known_map ────────────────────────────────

    def remove_mission(self, pos: tuple[int, int]) -> None:
        super().remove_mission(pos)
        if self._known_map is not None and self._in_bounds(*pos):
            self._known_map[pos] = int(Tile.FLOOR)

    # ── reset: restore full omniscience ──────────────────────────────────────

    def reset(self, start_pos: tuple[int, int] | None = None) -> None:
        super().reset(start_pos)
        self._apply_omniscience(self._grid)
        self.team_role = TeamRole.NONE
