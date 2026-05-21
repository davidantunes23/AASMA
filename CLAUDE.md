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
# Run rule-based simulation (produces output/simulation.gif)
python run.py --demo rule --seed 42

# Debug run — skips GIF rendering, prints per-step alien state
python run.py --demo rule --seed 42 --no-render

# Random baseline agents
python run.py --demo random --seed 42 --no-show

# World-only view (no knowledge panels)
python run.py --demo rule --style world --no-show --seed 42

# Alien-favoured map (more vents), player-favoured (more hiding spots)
python run.py --alpha 0.8 --seed 42 --no-show
python run.py --alpha -0.5 --seed 42 --no-show

# Print map to terminal
python map_generator.py 42

# Save map PNG visualizations
python map_generator.py 42 --visualize

# Verification command (use after any change)
python run.py --demo rule --no-show --output output/verify.gif && echo "OK"
```

Key `run.py` flags: `--seed`, `--width`, `--height`, `--alpha`, `--max-steps`, `--fps`, `--noise-radius`, `--human-view`, `--alien-fov`, `--knowledge {on,off}`, `--style {full,world}`, `--no-render`, `--no-show`.

## Architecture

### Coordinate convention

**All positions are `(y, x)` / `(row, col)`** throughout. This applies to agents, the simulation, and all internal pathfinding. The simulation no longer needs coordinate conversion between agent types.

### Tile IDs

```plaintext
0=WALL  1=FLOOR  2=VENT  3=HIDE  4=PLAYER_START  5=ALIEN_START  6=EXIT
```

`PASSABLE_ALIEN = {1,2,4,5,6}` — aliens cannot enter HIDE tiles normally.
`PASSABLE_ALIEN_RUSH = {1,2,3,4,5,6}` — used when alien confirmed the player is hiding there.

### Agent hierarchy (`agents/`)

```plaintext
BaseAgent (ABC)           agents/base.py   pos:(y,x), direction, view_length, step(), reset()
├── BaseAlienAgent                         + grid
│   ├── AlienAgent        agents/alien.py  FSM + A* + BeliefMap + KnowledgeMap
│   └── RandomAlienAgent  agents/random_alien.py
└── BaseHumanAgent                         + hidden, observe()
    ├── HumanAgent        agents/human.py  BFS navigation, radar-reactive
    └── RandomHumanAgent  agents/random_human.py
```

`Direction`, `cone_fov()`, and `direction_from_delta()` live in `agents/base.py` and are shared by all agents.

**Interface contract**: every agent exposes `step(player_pos, heard_pos, step_num) → (y,x)`. Human agents additionally have `observe(obs, radar_threat, radar_dist)` which the simulation calls before `step()` each turn.

**FOV**: both agents use the same directional forward `cone_fov()` — wall/hide-blocked via Bresenham line-of-sight. There is no FOV asymmetry between them.

**Adding a new agent**: subclass `BaseAlienAgent` or `BaseHumanAgent`, implement `step()` (and `observe()` for humans), then wrap with `build_agent_spec(label, role, agent)` from `simulation.py`.

### AlienAgent state machine (`agents/alien.py`)

- **SEARCH**: explores unknown frontiers, falls back to patrol waypoints
- **INVESTIGATE**: moves to last known / last heard position
- **HUNT** (speed=2): pursues visible player; `player_known_hiding=True` unlocks `PASSABLE_ALIEN_RUSH` so it can enter hiding spots

Transition to HUNT-with-hiding only fires if the alien was **already in HUNT** when it sees `player_hiding=True`. A wandering alien cannot detect a player already inside a hide spot.

`_move_one()` always replans when in HUNT to prevent the 2-cell overshoot ("tunneling") bug in narrow corridors.

Vent teleportation triggers only when the savings exceed `VENT_ROUTE_MIN_SAVINGS = 4` steps and the heard-position distance exceeds `VENT_ROUTE_MIN_SOUND_DISTANCE = 8`.

### Simulation loop (`simulation.py`)

`GenericMapSimulation.run()` processes agents in list order (human first, then alien) each step:

1. Update radar
2. For each agent: build cone observation → call `observe()` (humans) → call `step()`
3. For the alien specifically: `nearest_target` always uses the human's **actual** position (not censored when hiding) — the alien's own `cone_fov` + `player_hiding` flag handles visibility correctly

Key mechanics (active when `enable_mechanics=True`; disabled automatically for `--demo random`):

- **Radar**: topology-aware BFS distance → CRITICAL/CLOSE/NEAR/FAR ping every `radar_interval` steps
- **Noise**: player emits a jittered sound position each step with probability `p_noise`; suppressed when hiding; offset controlled by `noise_radius`
- **Cone FOV**: both agents use `cone_fov()` — directional, wall/hide-blocked LoS via Bresenham

### Visualization (`simulation.py` render)

`SimulationFrame` captures per-step: agent positions, `fov` (frozenset), `visible_opponent`, radar threat, noise ripple position, alien heard position.

Render produces a 3-panel GIF: **World** | **human_1 knowledge** | **alien_1 knowledge**. Knowledge panels show: explored tiles (cone-LoS accurate), FOV overlay (faint tint), visible-opponent marker, agent's own position marker.

`GenericKnowledge.update_from_observation()` only stores values `>= 0` to prevent radar/noise marker values (negative) from leaking into the visual colormap.

### Map generator (`map_generator.py`)

`MapGenerator(width, height, alpha, seed)` — `alpha ∈ [-1, 1]` shifts balance: negative = more HIDE tiles, positive = more VENT tiles. Visualization helpers (`visualise_map`, `run_demo`, etc.) are defined in the same file. The `__main__` block supports `--visualize` to save PNGs. Pre-generated maps are cached as JSON in `maps/`.

### Training (`training/`)

**Broken — do not use.** All training files (`train_aet.py`, `train_staged.py`, etc.) import from `training/envs.py`, which in turn does `from game import Game` — a module that no longer exists. The RL observation space (128-float vector, 5 discrete actions) and `stable-baselines3` PPO setup are defined in `training/envs.py` for reference, but none of it runs.
