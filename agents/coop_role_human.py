from __future__ import annotations

from collections import deque

from agents.base import Direction
from agents.role_human import RoleHumanAgent
from agents.coord_bus import CoordType, CoordMessage
from agents.shared_belief import SharedBeliefMap


class CoopRoleHumanAgent(RoleHumanAgent):
    """RoleHumanAgent extended with a shared belief map.

    When multiple CoopRoleHumanAgents run together the simulation wires them
    to the same SharedBeliefMap instance before the first observe() call.
    This means every cone observation written by any agent is immediately
    reflected in every other agent's _known_map — no extra messages needed.

    On top of passive map sharing, each agent registers its current
    exploration frontier target so teammates prefer cells nobody else is
    already heading toward, naturally partitioning the map.

    When only one agent is present (or shared_map is never set) the class
    degrades gracefully to RoleHumanAgent behaviour.
    """

    def __init__(
        self,
        start_pos: tuple[int, int],
        start_dir: Direction = Direction.NORTH,
        view_length: int = 6,
    ) -> None:
        super().__init__(start_pos, start_dir, view_length)
        self.shared_map: SharedBeliefMap | None = None

    # ── Memory initialisation ─────────────────────────────────────────────────

    def _init_memory(self, obs) -> None:
        if (
            self.shared_map is not None
            and self.shared_map.known_map.shape == obs.shape
        ):
            if self._known_map is not self.shared_map.known_map:
                prev_known_exit = self._known_exit
                # First time: alias _known_map to the shared array.
                # All subsequent writes via _integrate_observation go directly
                # into the shared array without any extra copying.
                self._known_map = self.shared_map.known_map
                # Preserve already-known exit when switching map backing.
                self._known_exit = prev_known_exit
                self._observed_aliens = set()
            return
        # Fallback (solo agent or shape mismatch): own independent map.
        super()._init_memory(obs)

    # ── Observation ───────────────────────────────────────────────────────────

    def _integrate_observation(self, obs) -> None:
        # Parent writes visible tiles into self._known_map (which IS the shared
        # array when wired up) and queues EXIT/MISSION coord-bus messages.
        super()._integrate_observation(obs)
        if self.shared_map is not None:
            self.shared_map.set_position(self.agent_id, self.pos)

    def receive_coords(self, messages: list[CoordMessage]) -> None:
        # Accept discovery messages so coop agents learn missions and exits
        # via teammate broadcasts like other RoleHumanAgents.
        super().receive_coords(messages)

        # Mirror mission-active claims into the shared map so teammates avoid
        # exploring mission tiles that are already being worked on.
        if self.shared_map is None:
            return
        for msg in messages:
            try:
                if msg.coord_type == CoordType.MISSION_ACTIVE:
                    # register the mission as our current target so frontiers
                    # and exploration avoid it when prefer_unclaimed=True
                    self.shared_map.set_target(self.agent_id, msg.pos)
                elif msg.coord_type == CoordType.MISSION_DONE:
                    # clear any target pointing to the completed mission
                    cur = getattr(self.shared_map, "_targets", {}).get(self.agent_id)
                    if cur == msg.pos:
                        self.shared_map.set_target(self.agent_id, None)
            except Exception:
                pass

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        _player_pos: tuple[int, int],
        _heard_pos: tuple[int, int] | None = None,
        _step_num: int = 0,
    ) -> tuple[int, int]:
        result = super().step(_player_pos, _heard_pos, _step_num)
        if self.shared_map is not None:
            self.shared_map.set_position(self.agent_id, result)
        return result

    # ── Cooperative frontier BFS ──────────────────────────────────────────────

    def _next_step_to_nearest_frontier(self) -> tuple[int, int] | None:
        return self._coop_bfs_frontier(self._is_frontier)

    def _next_step_to_nearest_floor_frontier(self) -> tuple[int, int] | None:
        return self._coop_bfs_frontier(self._is_floor_frontier)

    def _coop_bfs_frontier(self, is_frontier) -> tuple[int, int] | None:
        """BFS to nearest frontier; prefers cells unclaimed by teammates.

        Pass 1 — skip frontiers already targeted by another agent.
        Pass 2 (fallback) — accept any frontier (prevents deadlock when all
                            reachable frontiers are claimed by teammates).

        Registers the chosen destination in shared_map so teammates can see it.
        """
        if self.shared_map is None:
            return self._bfs_next_step(is_frontier)

        # --- Pass 1: nearest unclaimed frontier ---
        first_step, dest = self._bfs_frontier_with_dest(
            is_frontier,
            prefer_unclaimed=True,
        )
        if first_step is not None:
            self.shared_map.set_target(self.agent_id, dest)
            return first_step

        # --- Pass 2: any frontier (ignoring claims) ---
        first_step, dest = self._bfs_frontier_with_dest(
            is_frontier,
            prefer_unclaimed=False,
        )
        self.shared_map.set_target(self.agent_id, dest)
        return first_step

    def _bfs_frontier_with_dest(
        self,
        is_frontier,
        prefer_unclaimed: bool,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        """BFS that returns (first_step, destination).

        When prefer_unclaimed is True, only stops at frontiers not targeted by
        another agent. Returns (None, None) if no matching cell is reachable.
        """
        start = self.pos
        if not self._in_bounds(*start):
            return None, None

        queue = deque([start])
        parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            current = queue.popleft()
            if current != start and is_frontier(current):
                if prefer_unclaimed and self.shared_map is not None:
                    if self.shared_map.is_targeted_by_other(current, self.agent_id):
                        # Keep searching; don't stop here.
                        for neighbor, _ in self._walkable_neighbors(current):
                            if neighbor not in parents:
                                parents[neighbor] = current
                                queue.append(neighbor)
                        continue
                first_step = self._first_step_from_path(current, parents)
                return first_step, current
            for neighbor, _ in self._walkable_neighbors(current):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)

        return None, None
