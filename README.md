# AASMA — Asymmetric Agents in a Hide-and-Seek Game

An Alien Isolation-inspired asymmetric pursuit-evasion game with procedurally generated maps, rule-based and random agent baselines, and a flexible simulation framework.

---

## The Game

Two agents compete on a partially-observable grid map:

| Agent             | Goal                                | Abilities                          |
|-------------------|-------------------------------------|------------------------------------|
| **Human** (blue)  | Reach the exit without being caught | Walk, wait, hide in hiding spots   |
| **Alien** (red)   | Catch the human before it escapes   | Walk, wait, teleport through vents |

**Key mechanics:**

- **Partial observation** — the human sees a directional forward cone; the alien has an omnidirectional FOV blocked by walls and hiding spots.
- **Radar** — every few steps the human receives a threat-level ping (CRITICAL / CLOSE / NEAR / FAR) based on path distance to the alien.
- **Noise** — each step the human has a chance to emit a sound at a jittered position; the alien hears this and uses it to track the human.
- **Hiding spots** — the human can enter a hiding spot to break line-of-sight and suppress sound emission.
- **Vents** — the alien can teleport between observed vents when it would save significant distance.

---

## Quick Start

```bash
# Clone and install dependencies
git clone <repo>
cd AASMA
pip install numpy matplotlib

# Run a rule-based game (produces a GIF)
python scripts/run.py --demo rule --no-show

# Run with random agents
python scripts/run.py --demo random --no-show

# Show the world map only (no knowledge panels)
python scripts/run.py --demo rule --style world --no-show
```

Output is saved to `output/simulation.gif` by default.

---

## Directory Structure

```
AASMA/
├── agents/
│   ├── alien.py          — rule-based alien (A*, belief map, FSM, vent routing)
│   ├── human.py          — rule-based human (BFS, hiding, radar-reactive)
│   ├── role_human.py     — role-aware human (WORKER / DECOY / RUNNER support)
│   ├── role_manager.py   — greedy role assignment helpers (WORKER/DECOY/RUNNER)
│   ├── random_alien.py   — random alien baseline (vent teleport + walk)
│   └── random_human.py   — random human baseline (simple random walk)
├── map_generator.py      — procedural map generation + PNG visualization
├── simulation.py         — simulation engine (game loop, rendering, GIF output)
├── scripts/
│   └── run.py            — unified CLI entry point
├── training/             — RL training pipeline (WIP)
├── maps/                 — saved map JSON files
└── output/               — generated GIFs and PNGs
```

---

## Running Simulations

All simulation options go through `scripts/run.py`:

```
python scripts/run.py [options]

--demo {rule,random}     Agent pair to use (default: rule)
--style {full,world}     full = world + knowledge panels; world = world only (default: full)
--knowledge {on,off}     Show per-agent knowledge panels (default: on)
--seed N                 Map and agent seed (default: 42)
--width N                Map width in cells (default: 50)
--height N               Map height in cells (default: 35)
--alpha F                Map bias: negative = more hides, positive = more vents (default: 0.0)
--max-steps N            Episode length cap (default: 300)
--fps N                  GIF frames per second (default: 12)
--output PATH            Output GIF path (default: output/simulation.gif)
--no-show                Skip interactive preview window
--human-view N           Human observation radius (default: 6)
--alien-fov N            Alien FOV radius (default: 6)
```

**Examples:**

```bash
# Alien-favoured map, world view only
python scripts/run.py --alpha 0.8 --style world --no-show

# Player-favoured map, longer episode
python scripts/run.py --alpha -0.5 --max-steps 500 --seed 7 --no-show

# Random agents, no knowledge panels
python scripts/run.py --demo random --knowledge off --no-show
```

---

## Map Generator

Maps are procedurally generated with a single `alpha` parameter that shifts the balance:

| alpha | Effect                                          |
|-------|-------------------------------------------------|
| `< 0` | Player-favoured: more hiding spots, fewer vents |
| `= 0` | Balanced                                        |
| `> 0` | Alien-favoured: more vents, fewer hiding spots  |

**Tile types:**

| Tile          | Value | Description                         |
|---------------|-------|-------------------------------------|
| WALL          | 0     | Impassable                          |
| FLOOR         | 1     | Passable by both                    |
| VENT          | 2     | Alien teleport network              |
| HIDE          | 3     | Human hiding spot, blocks alien FOV |
| PLAYER_START  | 4     | Human spawn                         |
| ALIEN_START   | 5     | Alien spawn                         |
| EXIT          | 6     | Human goal                          |

**CLI usage:**

```bash
# Print map to terminal
python map_generator.py [seed] [--width N] [--height N]

# Save PNG visualizations to output/
python map_generator.py [seed] --visualize
```

---

## Agents

There are three principal agent families in the codebase — each useful for different experiments and demonstrations:

- **Random agents** (`agents/random_human.py`, `agents/random_alien.py`) — very simple baselines that choose random valid moves (and the random alien optionally vents). They are role-unaware and do not participate in coordination or emit deliberate loud-noise. Useful as low-skill baselines and for testing the simulation plumbing.

- **Rule-based agents (no coordination)** (`agents/human.py`, `agents/alien.py`) — the original game-play agents. The human implements a priority-driven BFS/steering policy (hide, explore, go-to-exit) and responds to radar/noise signals; the alien runs an FSM (SEARCH / INVESTIGATE / HUNT) with belief-tracking and vent routing. These agents operate independently and do not use inter-agent coord messages or team roles.

- **Role-based coordination (first version)** (`agents/role_human.py` + `agents/role_manager.py` + `simulation.py` support) — role-aware humans expose `team_role`, `flush_outbox()` and `receive_coords()` for a simple coord-bus. Agents can publish `CoordMessage` for `EXIT` and `MISSION`; the simulation relays messages to teammates and maintains light shared state (`shared_mission_coords`, `shared_exit_coord`). Role reassignment is event-driven (missions discovered/completed, worker captured, exit opened) and uses greedy assignment helpers. Important: this coordination design shares sparse coordinate messages and assigned roles, not full per-agent knowledge maps — agents keep their own memory and observations.

Overview of differences

- **Information sharing:** Random and rule-based (no coordination) agents have no teammate messaging. Role-based coordination shares only coordinate messages (not a full shared world view).
- **Role semantics:** Only role-based agents expose `team_role` (WORKER/DECOY/RUNNER) and the sim performs event-driven reassignment; rule-based agents have no team roles.
- **Noise / deliberate signals:** All mechanics (radar, ambient noise) exist for rule-based and role-based agents when `enable_mechanics=True`. Role-based agents can also set `made_loud_noise` for deliberate signals that the sim forwards to aliens as an exact heard position.
- **Use cases:** Use random agents for baselines and plumbing tests; use rule-based (no coordination) for single-agent behaviour and classic evaluations; use role-based coordination when experimenting with simple team strategies, mission allocation, and coordinated distraction/defence behaviours.

See `ROLES_README.md` for a developer-oriented explanation of the current coordination implementation and APIs.

---

## Simulation Engine (`simulation.py`)
`GenericMapSimulation` runs the game loop and renders output GIFs. It supports agents that expose either a `step(player_pos, heard_pos, step_num)` or `_act(obs, radar_threat, radar_dist)` interface.

**Key parameters:**

```python
GenericMapSimulation(
        grid,
        agents,                  # list of AgentSpec
        knowledge_mode="on",   # record per-agent knowledge maps for rendering
        enable_mechanics=True,   # enables radar, noise, cone observation
        p_noise=0.1,             # probability of stochastic sound events per step
        radar_interval=5,        # steps between radar pings
)
```

When `enable_mechanics=False` (commonly used with random agents), radar/noise/cone mechanics are disabled and agents receive exact observations.


**Coordination & roles

Basic role support exists for role-aware human agents (`WORKER`, `DECOY`, `RUNNER`). Role-aware agents expose coordination hooks (outbox/receive) and may set deliberate loud-noise signals; the simulation performs event-driven reassignment and maintains shared mission/exit coordinates.

For a full developer-oriented description, implementation notes, and API examples, see `ROLES_README.md`.
Notes & experimental tips

- Role-based reassignment is deterministic given the sim seed and current agent positions; this is helpful for reproducible experiments.
- If you want fixed roles for ablation studies, call `sim.enable_role_based(False)` and use `sim.set_initial_roles(...)`.
- Random agents (`RandomHumanAgent`, `RandomAlienAgent`) are role-unaware and will not interact with the coord-bus even if mixed into the sim.

The above replaces the earlier brief summary; see `ROLES_README.md` for an expanded developer-oriented description and implementation notes.
