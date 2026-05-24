"""Simple team-role manager utilities for human agents.

This module implements a minimal, greedy assignment function that picks the
closest human agent to any mission tile and assigns it the WORKER role. The
implementation is intentionally lightweight and designed to be called from the
simulation loop when the set of visible mission tiles changes.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from agents.base import TeamRole


def assign_worker_greedy(agent_specs: Sequence[object], mission_positions: Iterable[tuple[int, int]]):
    """Assign `TeamRole.WORKER` to the human agent closest to any mission.

    - `agent_specs` is a sequence of objects with attributes `role`, `label`, and `agent`.
    - `mission_positions` is an iterable of (y, x) mission tile coordinates.

    The function mutates `spec.agent.team_role` for human agents. Returns the
    label of the agent promoted to WORKER, or None if no assignment was made.
    """
    missions = list(mission_positions)
    # Clear existing worker assignments
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

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

    try:
        setattr(best.agent, "team_role", TeamRole.WORKER)
    except Exception:
        return None

    return getattr(best, "label", None)


def clear_roles(agent_specs: Sequence[object]):
    """Clear team roles on all human agents."""
    for s in agent_specs:
        if getattr(s, "role", None) == "human":
            try:
                setattr(s.agent, "team_role", TeamRole.NONE)
            except Exception:
                pass


def assign_decoy_farthest(agent_specs: Sequence[object], mission_positions: Iterable[tuple[int, int]]):
    """Assign `TeamRole.DECOY` to the human agent farthest from mission positions.

    Also sets `agent.mission_positions` for role-aware agents so they can
    compute farthest tiles locally.
    Returns the label of the agent assigned, or None.
    """
    missions = list(mission_positions)
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]
    for s in human_specs:
        if hasattr(s.agent, "mission_positions"):
            setattr(s.agent, "mission_positions", missions)
    if not missions or not human_specs:
        return None

    if not missions or not human_specs:
        return None

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

    try:
        setattr(best.agent, "team_role", TeamRole.DECOY)
        if hasattr(best.agent, "mission_positions"):
            setattr(best.agent, "mission_positions", missions)
    except Exception:
        return None

    return getattr(best, "label", None)


def assign_runner_greedy(agent_specs: Sequence[object], exit_position: tuple[int, int] | None, alien_position: tuple[int, int] | None = None):
    """Assign `TeamRole.RUNNER` to the human agent best positioned to reach the exit.

    Scoring favors closeness to exit and distance from the alien. Returns the
    label of the chosen agent, or None.
    """
    human_specs = [s for s in agent_specs if getattr(s, "role", None) == "human"]

    if exit_position is None or not human_specs:
        return None

    best = None
    best_score = None
    for s in human_specs:
        pos = getattr(s.agent, "pos", None)
        if pos is None:
            continue
        dist_exit = abs(pos[0] - exit_position[0]) + abs(pos[1] - exit_position[1])
        dist_alien = float('inf') if alien_position is None else (abs(pos[0] - alien_position[0]) + abs(pos[1] - alien_position[1]))
        # score: prefer small dist_exit and large dist_alien
        score = -dist_exit + dist_alien
        if best is None or score > best_score:
            best = s
            best_score = score

    if best is None:
        return None

    try:
        setattr(best.agent, "team_role", TeamRole.RUNNER)
    except Exception:
        return None

    return getattr(best, "label", None)
