from __future__ import annotations

from typing import Iterable, Sequence

from agents.base import TeamRole


def assign_worker_greedy(agent_specs: Sequence[object], mission_positions: Iterable[tuple[int, int]]):
    """Assign `TeamRole.WORKER` to the human agent closest to any mission.

    - `agent_specs` is a sequence of objects with attributes `role`, `label`, and `agent`.
    - `mission_positions` is an iterable of (y, x) mission tile coordinates.

    The function does NOT mutate agent state; it only selects the best
    candidate and returns its label. The caller is responsible for committing
    the role change and any mission claim side-effects.
    """
    missions = list(mission_positions)
    # Clear existing worker assignments
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

    # Choose the human closest to any mission using Manhattan distance.

    if not missions or not human_specs:
        return None

    # Greedy: select human with minimal Manhattan distance to nearest mission
    best = None
    best_dist = None
    for s in human_specs:
        pos = getattr(s.agent, "pos", None)
        if pos is None:
            continue
        min_d = min(abs(pos[0] - m[0]) + abs(pos[1] - m[1]) for m in missions)
        if best is None or min_d < best_dist:
            best = s
            best_dist = min_d

    if best is None:
        return None

    return getattr(best, "label", None)


def clear_roles(agent_specs: Sequence[object]):
    """Clear team roles on all human agents."""
    for s in agent_specs:
        if getattr(s, "role", None) == "human":
            try:
                # Reset role to NONE for human agents.
                setattr(s.agent, "team_role", TeamRole.NONE)
            except Exception:
                pass


def assign_decoy_farthest(agent_specs: Sequence[object], mission_positions: Iterable[tuple[int, int]]):
    """Assign `TeamRole.DECOY` to the human agent farthest from mission positions.

    Returns the label of the agent assigned, or None. Note: this function
    does not mutate `agent.team_role`.
    """
    missions = list(mission_positions)
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

    if not missions or not human_specs:
        return None
    
    for s in human_specs:
        if hasattr(s.agent, "mission_positions"):
            # keep mission_positions updated for heuristics; this is a benign
            # convenience mutation used by agents' local scoring.
            setattr(s.agent, "mission_positions", missions)

    # Score agents by distance to missions; choose the one farthest away.

    best = None
    best_score = -1
    for s in human_specs:
        pos = getattr(s.agent, "pos", None)
        if pos is None:
            continue
        # score = min distance to missions (we choose the agent with max of this)
        min_d = min(abs(pos[0] - m[0]) + abs(pos[1] - m[1]) for m in missions)
        if min_d > best_score:
            best = s
            best_score = min_d

    if best is None:
        return None

    return getattr(best, "label", None)


def assign_workers_omniscient(
    agent_specs: Sequence[object],
    mission_positions: Iterable[tuple[int, int]],
) -> list[tuple[object, tuple[int, int]]]:
    """Return an optimal (agent_spec, mission) pairing via greedy min-cost matching.

    Each iteration picks the globally closest (agent, mission) pair and removes
    both from the remaining pools so no two workers are paired with the same
    mission. Does NOT mutate agent state — the caller commits roles and mission
    assignments after inspecting the returned pairs.

    Returns a list of (spec, mission_pos) pairs in assignment order.
    """
    missions = list(mission_positions)
    available_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

    if not missions or not available_specs:
        return []

    n_workers = min(len(missions), len(available_specs))
    available_missions = list(missions)
    pairs: list[tuple[object, tuple[int, int]]] = []

    for _ in range(n_workers):
        if not available_specs or not available_missions:
            break
        best_spec = None
        best_mission = None
        best_dist = float("inf")
        for s in available_specs:
            pos = getattr(s.agent, "pos", None)
            if pos is None:
                continue
            for m in available_missions:
                d = abs(pos[0] - m[0]) + abs(pos[1] - m[1])
                if d < best_dist:
                    best_dist = d
                    best_spec = s
                    best_mission = m
        if best_spec is None:
            break
        pairs.append((best_spec, best_mission))
        available_specs.remove(best_spec)
        available_missions.remove(best_mission)

    return pairs


def assign_runner_residual(agent_specs: Sequence[object]) -> str | None:
    """Assign RUNNER to the first available agent (residual after Worker+Decoy).

    This ignores exit/alien proximity: any leftover agent becomes the Runner,
    making the role assignment independent of spawn geometry.
    """
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]
    if not human_specs:
        return None
    return getattr(human_specs[0], "label", None)


