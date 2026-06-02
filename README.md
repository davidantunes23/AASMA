# AASMA — Asymmetric Agents in a Hide-and-Seek Game

An Alien Isolation-inspired asymmetric pursuit-evasion game with procedurally generated maps, multi-agent coordination, and a flexible simulation framework for studying cooperative escape strategies.

---

## The Game

Three human agents compete against one alien on a partially-observable grid map:

| Agent            | Goal                                          | Abilities                                  |
|------------------|-----------------------------------------------|--------------------------------------------|
| **Humans** (blue)| Complete all missions, then reach the exit    | Walk, wait, hide in hiding spots           |
| **Alien** (red)  | Catch all humans before they escape           | Walk, wait, teleport through vents (faster)|

**Key mechanics:**

- **Missions** — humans must stand on mission tiles for 20 consecutive steps each; the exit is locked until all missions are complete.
- **Partial observation** — humans see a directional forward cone; the alien has a wider omnidirectional FOV blocked by walls and hiding spots.
- **Radar** — every few steps each human receives a threat-level ping (CRITICAL / CLOSE / NEAR / FAR) based on path distance to the alien.
- **Noise** — each step a human has a chance to emit a sound at a jittered position; the alien hears this and uses it to track humans.
- **Hiding spots** — a human inside a hiding spot breaks line-of-sight and suppresses sound emission; the alien cannot enter unless it already saw the human hide there.
- **Vents** — the alien can teleport between vents when it would save significant distance (≥4 steps savings, sound distance ≥8).

---

## Quick Start

```bash
# Clone and install dependencies
git clone <repo>
cd AASMA
pip install numpy matplotlib

# Role-based team of 3 humans vs rule-based alien (2 missions)
python run.py --human-class role --no-show

# Shared-map cooperative team
python run.py --human-class coop --no-show

# Omniscient upper-bound (full map/missions/exit pre-known)
python run.py --human-class omniscient --no-show

# Random baseline agents
python run.py --demo random --no-show

# Run without producing a GIF (faster, for debugging)
python run.py --human-class role --no-render
```

Output is saved to `output/simulation.gif` by default.

---

## Directory Structure

```
AASMA/
├── agents/
│   ├── base.py               — BaseAgent, BaseHumanAgent, BaseAlienAgent, Direction, TeamRole
│   ├── rule_human.py         — rule-based human (BFS, hiding, radar-reactive, no coordination)
│   ├── role_human.py         — role-aware human (WORKER / DECOY / RUNNER + coord bus)
│   ├── coop_role_human.py    — cooperative human (shared belief map + role coordination)
│   ├── omniscient_human.py   — upper-bound human (full map/missions/exit pre-known)
│   ├── random_human.py       — random human baseline
│   ├── rule_alien.py         — rule-based alien (A*, belief map, FSM, vent routing)
│   ├── random_alien.py       — random alien baseline
│   ├── role_manager.py       — greedy role assignment helpers (WORKER/DECOY/RUNNER)
│   ├── coord_bus.py          — inter-agent coordinate message bus
│   └── shared_belief.py      — shared belief map for cooperative agents
├── map_generator.py          — procedural map generation + PNG visualization
├── simulation.py             — simulation engine (game loop, rendering, GIF output)
├── run.py                    — unified CLI entry point
├── evaluate_agents.py        — multi-episode evaluation across all human/alien pairings
├── training/                 — RL training pipeline (partially broken — see below)
├── maps/                     — cached map JSON files
└── output/                   — generated GIFs, PNGs, and evaluation plots
```

---

## Running Simulations

All simulation options go through `run.py`. Defaults are **3 humans**, **2 missions**, **rule-based alien**.

```
python run.py [options]

--demo {rule,random}     Agent pair preset: 'rule' enables mechanics, 'random' disables them (default: rule)
--human-class {human,role,coop,omniscient,random}
                         Human agent implementation (default: human)
--alien-class {alien,random}
                         Alien agent implementation (default: alien)
--human-count N          Number of human agents (default: 3)
--alien-count N          Number of alien agents (default: 1)
--mission-count N        Number of mission tiles (default: 2)
--mission-steps N        Steps to dwell on each mission tile (minimum 20, default: 20)
--seed N                 Map and agent seed (default: 42)
--width N                Map width in cells (default: 50)
--height N               Map height in cells (default: 35)
--alpha F                Map bias: negative = more hides, positive = more vents (default: 0.0)
--max-steps N            Episode length cap (default: 300)
--fps N                  GIF frames per second (default: 12)
--output PATH            Output GIF path (default: output/simulation.gif)
--style {full,world}     full = world + knowledge panels; world = world only (default: full)
--knowledge {on,off}     Show per-agent knowledge panels in the GIF (default: on)
--no-show                Skip interactive preview window
--no-render              Skip GIF rendering (useful for debug/headless runs)
--random-map             Use a random seed for map generation
--min-start-distance N   Minimum BFS distance between human and alien spawn (default: 0)
--human-view N           Human observation radius (default: 6)
--alien-fov N            Alien FOV radius (default: 6)
--noise-radius N         Max cell offset for ambient noise (default: 2)
```

### Agent configurations

```bash
# Rule-based single human, no missions (classic escape)
python run.py --human-class human --human-count 1 --mission-count 0 --no-show

# Random agents (no mechanics — exact positions, no radar/noise)
python run.py --demo random --no-show

# Role-based team: WORKER completes missions, DECOY distracts, RUNNER escapes
python run.py --human-class role --no-show

# Cooperative team: shared belief map so all agents explore together
python run.py --human-class coop --no-show

# Omniscient upper-bound: full map + all missions + exit pre-known
python run.py --human-class omniscient --no-show

# World view only (no per-agent knowledge panels)
python run.py --human-class role --style world --no-show

# Alien-favoured map (more vents)
python run.py --human-class role --alpha 0.8 --no-show

# Player-favoured map (more hiding spots), longer episode
python run.py --human-class role --alpha -0.5 --max-steps 500 --no-show

# Ensure a minimum safe starting distance
python run.py --human-class role --min-start-distance 20 --no-show

# Verification run (quick sanity check after changes)
python run.py --human-class human --human-count 1 --mission-count 0 --no-show --output output/verify.gif && echo "OK"
```

---

## Evaluation

`evaluate_agents.py` runs all human/alien pairings across multiple episodes and produces bar charts and CSV summaries.

```bash
# Quick evaluation (10 episodes per pairing)
python evaluate_agents.py --episodes 10 --no-show

# Full evaluation (default 30 episodes, saves plots to output/eval_pairs/)
python evaluate_agents.py --no-show

# Custom map sizes
python evaluate_agents.py --map-sizes 30x20 45x30 60x40 --no-show
```

**Pairings evaluated:**

- Human models: `random_human`, `rule_human`, `omniscient_human`, `role_human_3`, `coop_role_human_3`
- Alien models: `random_alien`, `rule_alien`

**Outputs** (in `output/eval_pairs/`):

- Per-pairing bar chart: distribution of how many humans escaped (0/1/2/3)
- Per-pairing `summary.csv` with escaped percentages and average steps
- `rule_alien_human_comparison.png`: stacked bar chart comparing all human models vs the rule alien

**Key flags:**

```text
--episodes N         Episodes per matchup (default: 30)
--seed N             Base random seed (default: 42)
--map-sizes W1xH1 …  Map sizes to test (default: 60x40)
--max-steps N        Per-episode step cap (default: 1000)
--view-length N      Observation radius for all agents (default: 6)
--idle-limit N       Steps of no movement before forcing episode end (default: 50)
--output-dir PATH    Output directory (default: output/eval_pairs)
--no-show            Do not open matplotlib windows
```

---

## Map Generator

Maps are procedurally generated with a single `alpha` parameter:

| alpha | Effect                                          |
|-------|-------------------------------------------------|
| `< 0` | Player-favoured: more hiding spots, fewer vents |
| `= 0` | Balanced                                        |
| `> 0` | Alien-favoured: more vents, fewer hiding spots  |

**Tile types:**

| Tile         | ID | Description                                    |
|--------------|----|------------------------------------------------|
| WALL         | 0  | Impassable by all                              |
| FLOOR        | 1  | Passable by all                                |
| VENT         | 2  | Alien teleport network                         |
| HIDE         | 3  | Human hiding spot; blocks alien FOV            |
| PLAYER_START | 4  | Human spawn point                              |
| ALIEN_START  | 5  | Alien spawn point                              |
| EXIT         | 6  | Human goal (locked until all missions done)    |
| MISSION      | 7  | Mission tile (dwell for 20 steps to complete)  |

```bash
# Print map to terminal
python map_generator.py 42

# Save PNG visualizations to output/
python map_generator.py 42 --visualize
```

---

## Agent Architecture

### Hierarchy

```plaintext
BaseAgent (ABC)                  agents/base.py         pos:(y,x), direction, step(), reset()
├── BaseAlienAgent                                       + grid
│   ├── AlienAgent               agents/rule_alien.py   FSM (SEARCH/INVESTIGATE/HUNT) + A* + BeliefMap
│   └── RandomAlienAgent         agents/random_alien.py
└── BaseHumanAgent                                       + hidden, observe()
    ├── HumanAgent               agents/rule_human.py   BFS navigation, radar-reactive, no coordination
    ├── RoleHumanAgent           agents/role_human.py   role-aware (WORKER/DECOY/RUNNER) + coord bus
    │   ├── CoopRoleHumanAgent   agents/coop_role_human.py  shared belief map + role coordination
    │   └── OmniscientHumanAgent agents/omniscient_human.py full map/missions/exit pre-known
    └── RandomHumanAgent         agents/random_human.py
```

All agents implement `step(player_pos, heard_pos, step_num) → (y,x)`. Human agents additionally expose `observe(obs, radar_threat, radar_dist)` called before `step()` each turn.

### Human agent comparison

| Agent                 | Map knowledge | Coordination           | Roles               | Use case                   |
|-----------------------|---------------|------------------------|---------------------|----------------------------|
| `RandomHumanAgent`    | None          | None                   | None                | Baseline / plumbing tests  |
| `HumanAgent`          | Own FOV       | None                   | None                | Single-agent behaviour     |
| `RoleHumanAgent`      | Own FOV       | Coord bus (sparse)     | WORKER/DECOY/RUNNER | Team strategy experiments  |
| `CoopRoleHumanAgent`  | Shared map    | Coord bus + shared map | WORKER/DECOY/RUNNER | Cooperative exploration    |
| `OmniscientHumanAgent`| Full map      | Coord bus              | WORKER/DECOY/RUNNER | Upper-bound performance    |

### Role semantics (RoleHumanAgent and subclasses)

- **WORKER** — navigates to nearest uncompleted mission tile and dwells until done.
- **DECOY** — repositions to draw the alien away from active missions; emits deliberate loud noise when threat is NEAR/FAR.
- **RUNNER** — stages near the exit; escapes as soon as threat drops to FAR; never completes missions.

All roles share a **survival priority**: hiding overrides role tasks when radar threat is CRITICAL or CLOSE. Once all missions complete, all roles converge to exit-seeking.

Role assignment is **event-driven**: missions are discovered/completed, workers are caught, exits unlock. `role_manager.py` provides `assign_worker_greedy`, `assign_decoy_farthest`, `assign_runner_greedy`.

### Coordination mechanisms

**Coord bus** (`agents/coord_bus.py`): `CoordMessage` with `CoordType.MISSION`, `MISSION_DONE`, or `EXIT`. Agents call `flush_outbox()` / `receive_coords()` each step; the simulation relays messages between teammates. Shares sparse coordinate events — not full world state.

**Shared belief map** (`agents/shared_belief.py`): used by `CoopRoleHumanAgent`. All agents in a team alias their `_known_map` to the same `SharedBeliefMap.known_map` array, so any cone observation is immediately visible to teammates. Agents also register navigation targets to avoid redundant exploration.

### Alien agent (FSM)

- **SEARCH**: explores unknown frontiers, falls back to patrol waypoints.
- **INVESTIGATE**: moves to last known / last heard position.
- **HUNT** (speed=2): pursues visible human; unlocks `PASSABLE_ALIEN_RUSH` (enters HIDE tiles) only if the alien was already in HUNT when it saw the human hide.

Vent teleportation triggers only when path savings exceed 4 steps and sound distance exceeds 8.

---

## Interpreting outputs

**GIF panels** (default `--style full`):

- **World panel**: map with all agents, FOV cones, radar threat rings, noise ripple, role labels (WORKER/DECOY/RUNNER), and mission/exit markers.
- **Per-agent knowledge panels**: what each human or alien believes about the map (explored vs unknown tiles, known mission/exit coords).

**Outcome messages** printed to stdout:

- `ESCAPED` — all surviving humans reached the exit.
- `CAUGHT` — all humans were caught before escaping.
- `TIMEOUT` — episode reached `--max-steps` with neither outcome.

**Evaluation plots** (from `evaluate_agents.py`):

- Bar height = number of episodes where exactly N humans escaped.
- Colour coding in the comparison chart: red=0 escaped, orange=1, green=2, blue=3.

---

## Training (partially broken)

`training/envs.py` imports `from game import Game` — a module that no longer exists — so the gymnasium environments do not run. The obs/reward logic in `training/obs_rewards.py` is intact and importable.

The planned staged training (`train_staged.py`) uses a 4-phase PPO curriculum:

1. Human vs rule-based alien until >20% escape rate
2. Alien vs rule-based human until >30% catch rate
3. Both vs historical checkpoint pools
4. Full AET co-training

Observation space: 128-float vector. Action space: 6 discrete actions (WAIT + 4 walks + LOUD_NOISE).
