Roles & Coordination — Summary for AASMA
=======================================

This document explains the team-role system, how loud-noise works, when reassignment happens, and the code-level changes made to the repository to support these behaviours.

1) High-level role definitions
------------------------------
- WORKER
  - Primary mission executor: the agent that stands on a mission tile and completes the mission.
  - Should be conserved while making progress — we avoid pulling a worker off an active mission.

- DECOY
  - Distraction / protection specialist. Its purpose is to draw alien attention away from the WORKER.
  - Does not panic and emit loud noise based on its own radar; instead it hides when personally threatened and the simulation triggers loud noise from a DECOY only when a WORKER is threatened.

- RUNNER
  - Escape / survival role. Prioritises reaching the exit and staying safe. Will be promoted to WORKER only when free and needed.

2) Loud-noise / auditory mechanics
----------------------------------
- deliberate loud noise flag
  - Role-aware agents (implemented in `agents/role_human.py`) expose `made_loud_noise` (boolean). The flag represents an intentional, high-fidelity sound event.
  - The simulation (see `simulation.py`) checks for this flag during the step and treats that agent's exact tile as the alien `heard_pos` for investigation.

- ambient noise
  - The simulation still produces probabilistic ambient noise using `_generate_noise()` (jittered offsets and visual ripple).

- trigger rule (updated)
  - DECOYs no longer emit loud noise in response to their own radar. Instead the simulation triggers a DECOY loud-noise when a WORKER is threatened (CRITICAL or CLOSE).
  - If any human explicitly sets `made_loud_noise`, the simulation will use that exact source immediately.

3) Reassignment policy (what triggers role reallocation)
-------------------------------------------------------
- Immediate reassign triggers
  - New mission discovery (when a mission changes from unknown to known)
  - Mission completed
  - Worker death / unreachability
  - Exit opens (if relevant)
- Locking & safe assignments
  - Any WORKER currently on a mission tile or exposing a `mission_progress > 0` attribute is considered locked and will not be reassigned during automatic role reallocation. This prevents wasting in-progress work.
  - When a new mission is discovered, the sim re-runs assignment among the free (non-locked) agents and will promote the best available agent to WORKER and pair a DECOY if possible.

4) New / updated APIs and fields (what changed in code)
------------------------------------------------------
- `agents/role_human.py`
  - Added `team_role` support and `made_loud_noise` indicator (role-aware agent implementation).
  - DECOY behaviour changed: hides on personal threat, does not itself set `made_loud_noise` anymore.

- `simulation.py` (major changes)
  - New mission bookkeeping and APIs:
    - `add_mission(position)` — register a newly discovered mission and trigger immediate reassignment.
    - `complete_mission(position)` — mark mission complete and trigger reassignment.
  - Automatic mission detection support:
    - `mission_tile_values` (set[int]): configure which observed tile IDs count as missions. When a human observes those tile values the sim auto-calls `add_mission()` for the observed coordinates.
  - Role-based toggle and configuration APIs:
    - `enable_role_based(enabled: bool)` — enable/disable automatic reassignment.
    - `set_initial_roles(role_map)` — set explicit roles by agent label.
    - `allocate_roles_by_counts(counts)` — allocate a composition of roles among agents; now greedy (uses `agents/role_manager.py` scoring to pick best agents).
  - Reassignment logic:
    - `_maybe_reassign_roles()` now runs on immediate events (e.g., `add_mission`). It reassigns only among free (non-locked) agents and preserves locked WORKERs.
  - Three-phase simulation loop (observation → human actions → compute sound → alien actions) to ensure observations are integrated, roles reassigned, humans can emit, and aliens react to emits in the same timestep.

- `agents/role_manager.py`
  - Greedy assigners (WORKER, DECOY, RUNNER) are used by the sim for automatic and greedy manual assignments.

- `training/envs.py`
  - Action mapping extended for `LOUD_NOISE` so RL agents can map actions to intentional loud-signal semantics. (Note: you must wire the RL agent action handler to set `made_loud_noise` / call simulation APIs when using learned agents.)

5) Files changed in this work
-----------------------------
- `agents/role_human.py` — added role-aware agent behaviour and `made_loud_noise` flag.
- `agents/role_manager.py` — assignment helpers for WORKER / DECOY / RUNNER.
- `simulation.py` — mission detection, role toggles, immediate reassignment, DECOY→WORKER threat wiring, and run-loop refactor.
- `training/envs.py` — action space mapping updated to include `LOUD_NOISE` (training mapping).

6) How to adopt in your workflows
---------------------------------
- To enable automatic mission detection and immediate reassignment:
  - Set `sim.mission_tile_values = {<your mission tile id(s)>}` (for example, `{7}` if your mission tile has id 7).
  - Keep `sim.role_based = True` (default) or call `sim.enable_role_based(True)`.
  - Ensure mission discovery is represented as those tile values appearing in human observations (or call `sim.add_mission((y,x))` manually when appropriate).

- To control roles manually or for experiments:
  - Disable automatic reassignment: `sim.enable_role_based(False)`
  - Assign roles explicitly: `sim.set_initial_roles({'human_0': 'WORKER', 'human_1': 'DECOY'})` or use counts:
    - `sim.allocate_roles_by_counts({'WORKER':1,'DECOY':1,'RUNNER':1})` — this now picks agents greedily based on position / mission geometry.

7) Implementation notes & future ideas
-------------------------------------
- `mission_progress` is an optional attribute for agents — if present it is used to lock an active WORKER. If your human agents use a different progress metric or attribute name, adapt `_locked_worker_specs()` accordingly.
- RL wiring: to let learned agents trigger loud noise, the environment / agent bridge should set `agent.made_loud_noise = True` (or call a sim API) whenever the agent chooses the `LOUD_NOISE` action.
- Logging / metrics: you may want to add a small event log when `add_mission`/`complete_mission`/`_maybe_reassign_roles` run to profile reassign frequency and evaluate strategies.

If you want, I can:
- Add a small demo script (`scripts/demo_roles.py`) showing a WORKER on mission A, discovery of mission B, immediate reassignment promoting a RUNNER to WORKER and pairing a DECOY, with visualized noise ripple.
- Add constructor-time `mission_tile_values` argument to `GenericMapSimulation` for convenience.

---
Created in repo: ROLES_README.md
