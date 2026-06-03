# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dependencies

```bash
pip install numpy matplotlib
# For RL training (currently broken — see Training section):
pip install stable-baselines3
```

## Common Commands

```bash
# Run role-based simulation with 3 humans and 2 missions (default)
python run.py --human-class role --seed 42

# Debug run — skips GIF rendering, prints per-step alien state
python run.py --human-class role --seed 42 --no-render

# Cooperative shared-map agents
python run.py --human-class coop --seed 42 --no-show

# Omniscient upper-bound humans (full map/missions/exit pre-known)
python run.py --human-class omniscient --seed 42 --no-show

# Single rule-based human, no missions (classic escape)
python run.py --human-class human --human-count 1 --mission-count 0 --no-show

# Random baseline agents (mechanics disabled)
python run.py --demo random --seed 42 --no-show

# World-only view (no knowledge panels)
python run.py --human-class role --style world --no-show --seed 42

# Alien-favoured map (more vents), player-favoured (more hiding spots)
python run.py --alpha 0.8 --seed 42 --no-show
python run.py --alpha -0.5 --seed 42 --no-show

# Ensure minimum spawn distance between human and alien
python run.py --min-start-distance 20 --seed 42 --no-show

# Print map to terminal
python map_generator.py 42

# Save map PNG visualizations
python map_generator.py 42 --visualize

# Multi-episode evaluation (rule alien only, all human models)
python evaluate_agents.py --episodes 10 --no-show

# Verification command (use after any change)
python run.py --human-class human --human-count 1 --mission-count 0 --no-show --output output/verify.gif && echo "OK"
```

Key `run.py` flags: `--seed`, `--width`, `--height`, `--alpha`, `--max-steps` (default: 300), `--fps`, `--noise-radius`, `--noise-prob` (default: 0.10), `--human-view`, `--alien-fov`, `--knowledge {on,off}`, `--style {full,world}`, `--no-render`, `--no-show`, `--random-map`, `--min-start-distance`, `--human-count` (default: 3), `--human-class {human,role,coop,omniscient,random}`, `--alien-count`, `--alien-class {alien,random}`, `--mission-count` (default: 2), `--mission-steps`.

## Architecture

### Coordinate convention

**All positions are `(y, x)` / `(row, col)`** throughout agents, simulation, and pathfinding.

### Tile IDs

```plaintext
0=WALL  1=FLOOR  2=VENT  3=HIDE  4=PLAYER_START  5=ALIEN_START  6=EXIT  7=MISSION
```

`PASSABLE_ALIEN = {1,2,4,5,6}` — aliens cannot enter HIDE tiles normally.
`PASSABLE_ALIEN_RUSH = {1,2,3,4,5,6}` — used when alien confirmed the player is hiding there.

### Agent hierarchy (`agents/`)

```plaintext
BaseAgent (ABC)                  agents/base.py            pos:(y,x), direction, view_length, step(), reset()
├── BaseAlienAgent                                          + grid
│   ├── AlienAgent               agents/rule_alien.py      FSM + A* + BeliefMap + KnowledgeMap
│   └── RandomAlienAgent         agents/random_alien.py
└── BaseHumanAgent                                          + hidden, observe()
    ├── HumanAgent               agents/rule_human.py      BFS navigation, radar-reactive, no coordination
    ├── RoleHumanAgent           agents/role_human.py      role-aware, coord bus, WORKER/DECOY/RUNNER
    │   ├── CoopRoleHumanAgent   agents/coop_role_human.py shared belief map + role coordination
    │   └── OmniscientHumanAgent agents/omniscient_human.py full map/missions/exit pre-known at start
    └── RandomHumanAgent         agents/random_human.py
```

`Direction`, `cone_fov()`, `TeamRole`, and `direction_from_delta()` live in `agents/base.py`.

**Interface contract**: every agent exposes `step(player_pos, heard_pos, step_num) → (y,x)`. Human agents additionally have `observe(obs, radar_threat, radar_dist)` called before `step()` each turn.

**Adding a new agent**: subclass `BaseAlienAgent` or `BaseHumanAgent`, implement `step()` (and `observe()` for humans), then wrap with `build_agent_spec(label, role, agent)` from `simulation.py`.

### AlienAgent state machine (`agents/rule_alien.py`)

- **SEARCH**: explores unknown frontiers, falls back to patrol waypoints
- **INVESTIGATE**: moves to last known / last heard position
- **HUNT** (speed=2): pursues visible player; `player_known_hiding=True` unlocks `PASSABLE_ALIEN_RUSH`

Transition to HUNT-with-hiding only fires if the alien was **already in HUNT** when it sees `player_hiding=True`. A wandering alien cannot detect a player inside a hide spot.

`_move_one()` always replans in HUNT to prevent the 2-cell overshoot ("tunneling") bug. Vent teleportation triggers only when savings exceed `VENT_ROUTE_MIN_SAVINGS = 4` steps and sound distance exceeds `VENT_ROUTE_MIN_SOUND_DISTANCE = 8`.

### Role-based coordination (`agents/role_human.py`, `agents/role_manager.py`, `agents/coord_bus.py`)

`RoleHumanAgent` extends `BaseHumanAgent` with team coordination:

- **TeamRole** enum (in `agents/base.py`): `WORKER`, `DECOY`, `RUNNER`, `NONE`
  - **WORKER**: navigates to nearest uncompleted mission tile
  - **DECOY**: repositions to draw alien away from missions; emits `made_loud_noise` when threat is NEAR/FAR
  - **RUNNER**: stages near exit and escapes when threat drops to FAR; never completes missions
- **Coord bus** (`agents/coord_bus.py`): `CoordMessage` with `CoordType.MISSION`, `MISSION_DONE`, or `EXIT`. Role-aware agents call `flush_outbox()` / `receive_coords()` each step; the simulation relays messages between teammates.
- **Greedy assignment** (`agents/role_manager.py`): `assign_worker_greedy`, `assign_decoy_farthest`, `assign_runner_greedy` mutate `agent.team_role` on matching agent specs. Assignment is event-driven (mission discovered/completed, worker captured, exit unlocked).

All roles share the survival priority: hiding overrides role-specific tasks when radar threat is CRITICAL or CLOSE. After all missions complete, all roles converge to exit-seeking.

### Cooperative shared-map agents (`agents/coop_role_human.py`, `agents/shared_belief.py`)

`CoopRoleHumanAgent` extends `RoleHumanAgent` with a `SharedBeliefMap`. All agents in a team alias their `_known_map` to the same `SharedBeliefMap.known_map` array so any cone observation is immediately visible to teammates. Agents also register navigation targets so the frontier BFS prefers cells not already claimed by a teammate. Degrades gracefully to solo `RoleHumanAgent` when used alone.

### Omniscient agents (`agents/omniscient_human.py`)

`OmniscientHumanAgent` extends `RoleHumanAgent`. At construction the full grid, all exit coordinates, and all mission positions are pre-loaded into the agent's knowledge. Alien position is **not** included — the agent still reacts to radar threats. Used as an upper-bound performance baseline.

### Mission system (`simulation.py`)

Missions are tile ID `7` placed on the map at runtime (controlled by `--mission-count`, default 2). A human must accumulate `--mission-steps` steps on a mission tile to complete it (default 10); steps do not need to be consecutive — progress persists if the agent leaves and returns. The exit is **locked** until all missions are completed. The simulation tracks dwell progress in `_mission_dwell_progress` and notifies role-aware agents via coord messages when a mission completes.

### Simulation loop (`simulation.py`)

`GenericMapSimulation.run()` processes agents in list order (humans first, then aliens) each step:

1. Update radar
2. For each agent: build cone observation → call `observe()` (humans) → call `step()`
3. Relay coord messages between role-aware human agents
4. Update mission dwell progress; fire `MISSION_DONE` events and trigger role reassignment
5. Alien `nearest_target` always uses the human's **actual** position — the alien's own `cone_fov` + `player_hiding` flag handles visibility correctly

Key mechanics (active when `enable_mechanics=True`; disabled for `--demo random`):

- **Radar**: topology-aware BFS distance → CRITICAL/CLOSE/NEAR/FAR every `radar_interval` steps
- **Noise**: player emits a jittered sound position each step with `p_noise` probability (default 0.10, configurable via `--noise-prob`); suppressed when hiding; DECOY agents can also set `made_loud_noise=True` for deliberate signals forwarded to the alien as an exact position
- **Cone FOV**: both agent types use `cone_fov()` — directional, wall/hide-blocked LoS via Bresenham

`GenericMapSimulation` accepts `debug_log=False`; pass `True` to write role/mission reassignment decisions to `output/logs/` (disabled by default — no files or directories are created otherwise).

### Visualization (`simulation.py` render)

`SimulationFrame` captures per-step: agent positions, team roles, `fov`, `visible_opponent`, radar threat, noise ripple (with `noise_deliberate` flag), shared mission/exit coords, mission completion events.

Render produces a multi-panel GIF: **World** panel + per-agent knowledge panels. `--style world` suppresses knowledge panels. Role labels (WORKER/DECOY/RUNNER) are shown on agent markers in the world panel.

### Map generator (`map_generator.py`)

`MapGenerator(width, height, alpha, seed)` — `alpha ∈ [-1, 1]`: negative = more HIDE tiles, positive = more VENT tiles. Pre-generated maps cached as JSON in `maps/`.

### Evaluation (`evaluate_agents.py`)

Evaluates all human models against the rule-based alien across multiple episodes. Human models: `random`, `rule`, `role`, `coop`, `omniscient`. Output goes to `output/eval/` by default.

Plots produced per matchup: `survival_curve` (mean active humans over time), `capture_escape_timeline` (per-episode outcome scatter). A `rule_alien_human_comparison` stacked bar chart aggregates all human models. CSV summaries with escaped counts, avg steps, timeout rate, and mission completion rate are written alongside plots.

**Outcome classification**: `full_escape`, `partial_escape`, `full_capture`, `timeout`. Note that `timeout` covers both `max_steps_reached` and `idle_timeout` (triggered when all agents are stationary for `--idle-limit` steps, default 50) — idle timeouts can occur well before `max_steps`.

Key `evaluate_agents.py` flags: `--episodes` (default: 30), `--seed`, `--map-sizes` (default: `60x40`), `--max-steps` (default: 2000), `--view-length`, `--idle-limit`, `--output-dir`, `--humans {random,rule,role,coop,omniscient,all}`, `--noise-prob`.

### Training (`training/`)

**Partially broken.** `training/envs.py` (`BaseAETEnv`, `AlienEnv`, `PlayerEnv`) imports `from game import Game` — a module that no longer exists — so the gymnasium environments do not run. The obs/reward logic in `training/obs_rewards.py` is intact and importable.

The staged training design (`train_staged.py`) targets a 4-phase PPO curriculum:

1. Human vs rule-based alien until >20% escape rate
2. Alien vs rule-based human until >30% catch rate
3. Both vs historical checkpoint pools
4. Full AET co-training

The observation space is a 128-float vector; action space is 6 discrete actions (WAIT + 4 walks + LOUD_NOISE).
