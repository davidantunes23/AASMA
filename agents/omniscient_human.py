"""Omniscient human agent — upper-bound performance baseline.

At construction the full map grid, all exit coordinates, and all mission
positions are pre-loaded into the agent's knowledge. This removes the
exploration phase entirely: agents navigate directly to missions and the
exit from step one.

Alien position is NOT pre-loaded — the agent still reacts to radar threats
exactly like RoleHumanAgent. This makes it a fair upper bound: it shows
how much performance is limited by map exploration vs. alien avoidance.

Used to benchmark how close coordinated (role/coop) agents get to the
theoretical best a human team could achieve with perfect map knowledge.
"""
from __future__ import annotations

import numpy as np

from agents.base import Direction, TeamRole
from agents.role_human import RoleHumanAgent
from map_generator import Tile


class OmniscientHumanAgent(RoleHumanAgent):
    """Upper-bound human agent: full map, all missions, and exit pre-known at start."""

    def __init__(
        self,
        grid: np.ndarray,
        start_pos: tuple[int, int],
        start_dir: Direction = Direction.NORTH,
        view_length: int = 6,
    ) -> None:
        super().__init__(start_pos, start_dir, view_length)
        self._grid = grid.copy()       # kept so reset() can restore omniscience each episode
        self.team_role = TeamRole.NONE # defer role assignment until simulation registers missions
        self._apply_omniscience(grid)

    def _apply_omniscience(self, grid: np.ndarray) -> None:
        """Pre-load the full map, exit position, and all mission positions."""
        self._known_map = grid.copy().astype(np.int16)  # full map visible from step one

        ey, ex = np.where(grid == int(Tile.EXIT))
        self._known_exit = (int(ey[0]), int(ex[0])) if len(ey) > 0 else None

        # Populate mission tracking structures so role assignment and navigation
        # work correctly from step one without waiting for cone observations.
        self.mission_positions = []
        self._known_mission_coords = set()
        self._seen_mission_coords = set()
        my, mx = np.where(grid == int(Tile.MISSION))
        for y, x in zip(my.tolist(), mx.tolist()):
            pos = (int(y), int(x))
            self._known_mission_coords.add(pos)
            self._seen_mission_coords.add(pos)
            self.mission_positions.append(pos)

    # ── observe: radar only, no map learning ─────────────────────────────────

    def observe(
        self,
        obs: np.ndarray,
        radar_threat: str | None = None,
        radar_dist: int | None = None,
    ) -> None:
        """Skip map integration — map is already fully known. Only update radar and alien sighting."""
        if radar_threat is not None:
            self.last_radar_threat = radar_threat
            self.last_radar_dist = radar_dist
        self.hidden = self._tile_at(self.pos) == int(Tile.HIDE)
        radar_active = np.any(obs == self.RADAR_PING)  # radar ping means alien is nearby
        self._observed_aliens = {self.pos} if radar_active else set()  # treat own cell as danger zone

    # ── remove_mission: also clear the tile from the known map ───────────────

    def remove_mission(self, pos: tuple[int, int]) -> None:
        """Mark the completed mission tile as FLOOR so navigation doesn't target it again."""
        super().remove_mission(pos)
        if self._known_map is not None and self._in_bounds(*pos):
            self._known_map[pos] = int(Tile.FLOOR)  # rewrite so A*/BFS won't route back here

    # ── reset: restore full omniscience for the next episode ─────────────────

    def reset(self, start_pos: tuple[int, int] | None = None) -> None:
        super().reset(start_pos)
        self._apply_omniscience(self._grid)
        self.team_role = TeamRole.NONE
