# AASMA: Asymmetric Multi-Agent Reinforcement Learning

Training co-evolving policies for asymmetric partially-observable pursuit-evasion games.

## Quick Start

### Full Training (4-8 Hours)

```bash
cd /Users/mariaramos/Desktop/AASMA-1

# Activate environment
source .venv/bin/activate

# Stage 1-3: Staged training with opponent pools (4-8 hours)
python training/train_staged.py \
  --phase1-max-steps 100000 \
  --phase2-max-steps 100000 \
  --phase3-steps 50000 \
  --eval-episodes 20 \
  --pool-size 10 \
  --checkpoint-interval 5000 \
  --output-dir models

# Stage 4: ADA+ER co-training (2-3 hours)
python training/train_aet_with_ada_er.py \
  --rounds 50 \
  --round-steps 10000 \
  --eval-interval 2 \
  --eval-episodes 20 \
  --window-size 200 \
  --output-dir models

# View results
cat models/aet_metrics.jsonl | python -m json.tool | tail -50
```

### Quick Mode (1 Hour - Demo/Testing)

```bash
cd /Users/mariaramos/Desktop/AASMA-1
source .venv/bin/activate

# Stage 1-3: Quick version (~50 minutes)
python training/train_staged.py \
  --phase1-max-steps 20000 \
  --phase2-max-steps 20000 \
  --phase3-steps 10000 \
  --eval-episodes 5 \
  --pool-size 3 \
  --checkpoint-interval 5000 \
  --output-dir models

# Stage 4: Quick version (~10 minutes)
python training/train_aet_with_ada_er.py \
  --rounds 5 \
  --round-steps 2000 \
  --eval-interval 1 \
  --eval-episodes 5 \
  --window-size 50 \
  --output-dir models

# View results
cat models/aet_metrics.jsonl | python -m json.tool | tail -20
```

**Trade-offs in Quick Mode:**
- Fewer training steps (20% of full) → lower final performance
- Fewer eval episodes (5 vs 20) → noisier metrics
- Smaller opponent pool (3 vs 10) → less curriculum diversity
- Fewer ADA rounds (5 vs 50) → may not converge fully

**When to use:**
- Testing code changes
- Debugging
- Validating pipeline works on your hardware
- Development iteration (not final results)

| Parameter | Quick Mode | Full Mode | Impact |
|-----------|-----------|-----------|--------|
| Phase 1 steps | 20k | 100k | 5x faster |
| Phase 2 steps | 20k | 100k | 5x faster |
| Phase 3 steps | 10k | 50k | 5x faster |
| Eval episodes | 5 | 20 | Noisier metrics |
| Pool size | 3 | 10 | Less curriculum |
| ADA rounds | 5 | 50 | May not converge |
| **Total time** | **~1 hour** | **4-8 hours** | GPU saves 50% |

---

## Table of Contents

1. [Quick Start](#quick-start)
   - [Full Training (4-8 Hours)](#full-training-4-8-hours)
   - [Quick Mode (1 Hour Demo)](#quick-mode-1-hour---demotest)
2. [Architecture Overview](#architecture-overview)
3. [Design Decisions](#design-decisions)
4. [Training Pipeline](#training-pipeline)
5. [Running Each Stage](#running-each-stage)
6. [Metrics Philosophy](#metrics-philosophy)
7. [Files Explanation](#files-explanation)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### The Game

**Asymmetric Hide-and-Seek** (60×40 grid):

- **Human (blue)**: Escapes to exit, can hide, has limited vision
- **Alien (red)**: Pursues human, no hiding, hears noises, larger vision when pursuing
- **Both**: Partially observable, discrete actions (4 directions + wait), 500-step episodes

### Training System

**4-Stage Pipeline:**

```
Stage 0 (implicit)          Stage 1              Stage 2             Stage 3          Stage 4
├─ Fixed map                ├─ Human only        ├─ Alien only        ├─ Both train    ├─ Both with
├─ Reward calibration       ├─ vs rule-based     ├─ vs rule-based     ├─ vs opponent   ├─ ADA allocation
├─ Observation validation   │  alien             │  human             │  pools          ├─ Environment
│                           │                    │                    ├─ Historical    │  randomization
│                           ├─ Early stop:       ├─ Early stop:       │  pools (pFSP)  │  (ER)
│                           │  >20% escape       │  >30% catch        ├─ Early stop:   │
│                           │                    │                    │  >25% both      └─ Goal:
│                           └─ Output:           └─ Output:           │  imbalance      50-50
│                              player_stage1_*     alien_stage2_*     │  <0.2           balance

                                                                       └─ Output:
                                                                          player/alien_stage3_*
```

### Key Components

| File | Purpose | Lines |
|------|---------|-------|
| `training/train_staged.py` | Stages 1-3 pipeline | 400+ |
| `training/train_aet_with_ada_er.py` | Stage 4 with ADA+ER | 560+ |
| `training/envs.py` | Gymnasium wrappers | 350+ |
| `training/obs_rewards.py` | Asymmetric reward functions | 200+ |
| `training/metrics.py` | Evaluation metrics (separate from rewards) | 300+ |
| `training/environment_randomization.py` | Difficulty adjustment | 300+ |
| `agents/alien.py` | Rule-based alien baseline | - |
| `agents/human.py` | Rule-based human baseline | - |
| `game.py` | Core game logic | - |
| `map_generator.py` | Procedural map generation | - |

---

## Design Decisions

### 1. Separate Networks Per Role

**Decision**: Human and Alien have completely independent PPO models

**Why**:
- Asymmetric observations (human sees cones, alien sees belief maps)
- Asymmetric action spaces (human can hide, alien cannot)
- Asymmetric objectives (escape vs catch)
- Prevents catastrophic interference if combined

**Impact**: 2x network memory but stable learning per role

### 2. Staged Training Instead of Fresh Co-Training

**Problem**: If you start with both agents fresh, co-adaptation spirals occur:
- Alien learns to exploit one escape pattern
- Human learns to exploit that exploit
- System never stabilizes

**Solution**: Progressive curriculum

1. **Stage 1**: Human learns vs fixed rule-based (1 opponent, simple target)
2. **Stage 2**: Alien learns vs fixed rule-based (1 opponent, simple target)
3. **Stage 3**: Both train vs pools of opponents (mix of rule-based + learned)
4. **Stage 4**: Full co-training with adaptive balancing

**Impact**: Stable convergence, win rates reach competitive equilibrium (45-55% each)

### 3. Historical Opponent Pools (pFSP)

**Problem**: Every training round, both agents update immediately, creating fresh-vs-fresh co-adaptation

**Solution**: Opponent pool mechanism

```python
# When training human, alien opponent is sampled from:
pool = [
    75% chance: latest alien checkpoint (learning harder)
    25% chance: random older checkpoint (curriculum diversity)
]
```

**Impact**: 
- Prevents co-adaptation spirals
- Provides curriculum-like difficulty progression
- Keeps both agents improving without runaway dominance

### 4. Asymmetric Reward Shaping

**Problem**: Same reward for both agents won't work (asymmetric game)

**Solution**: Role-specific rewards

**Alien Rewards** (focused on pursuit):
```
Terminal:
  +20 : catch human
  -20 : human escapes
  -0.5: timeout

Shaping (order of magnitude < terminal):
  +0.3 : reduce distance when hearing human (pursuit)
  +0.02: explore new cell (first 50% of episode)
  -0.01: per step cost (prevents idle loops)
```

**Human Rewards** (focused on escape):
```
Terminal:
  +20 : reach exit
  -20 : caught
  0   : timeout (not penalized for survival)

Shaping (order of magnitude < terminal):
  +0.05: move closer to exit (when exit known)
  +0.5 : hide one-time (encourages hiding strategy)
  -0.2 : danger penalty (encourages reactive hiding)
  +0.01: explore (first 50%)
  -0.01: per step cost
```

**Why asymmetric?**
- Alien must be rewarded for pursuit signals (distance reduction when hearing)
- Human must be rewarded for strategic hiding (one-time bonus prevents farming)
- Terminal rewards always dominate (prevents reward hacking)

### 5. Separate Metrics from Rewards

**Problem**: High reward doesn't guarantee objective success (reward hacking)

**Solution**: Evaluate on actual objectives, not training reward

**Training Metrics** (for optimizer):
- Role-specific shaped rewards
- Immediate feedback to PPO

**Evaluation Metrics** (for validation):
- **Win rate**: Did human escape? (yes/no)
- **Catch rate**: Did alien catch? (yes/no)
- **Episode length**: How many steps?
- **Path efficiency**: How many unique cells visited?
- **Idle fraction**: How many consecutive idle steps?

**Monitoring**:
```python
# This shows if reward hacking is happening:
reward_alien = 150.0  # Training reward high
win_rate_alien = 0.2  # But only 20% catch rate?
# → Something wrong with reward function
```

**Impact**: Detects reward hacking early, validates learning is real

### 6. Adaptive Data Adjustment (ADA)

**Problem**: In Phase 3-4, one agent may dominate completely

**Solution**: Allocate training steps inversely to win rate

```python
# If alien is at 35% win rate and human at 65%:
alien_steps = base_steps * 1.5    # 50% more steps (weaker side)
human_steps = base_steps * 0.5    # 50% fewer steps (stronger side)

# Total always = 2 * base_steps, just redistributed
```

**Impact**: Automatic rebalancing toward competitive equilibrium

### 7. Environment Randomization (ER)

**Problem**: Map geometry can permanently favor one side

**Solution**: Adjust difficulty when imbalance persists

```python
if imbalance > 0.3 for 20+ consecutive episodes:
    if alien_wr > human_wr:
        HELP_HUMAN: add hiding spots, reduce alien vision
    else:
        HELP_ALIEN: remove hiding spots, increase alien vision
    
    Revert when imbalance recovers
```

**Impact**: 
- Prevents permanent map bias
- Accelerates rebalancing
- Temporary, not permanent change

### 8. PPO Hyperparameters (Fixed)

```python
gamma = 0.99              # Long-term rewards matter
learning_rate = 5e-5      # Stable, prevents instability
ent_coef = 0.01           # Entropy prevents mode collapse
n_steps = 400             # Rollout buffer size
batch_size = 64           # Mini-batch for update
```

**Why stable-baselines3?**
- Proven in this domain
- Off-policy friendly (compatible with opponent sampling)
- Consistent behavior across runs

---

## Training Pipeline

### Prerequisites

```bash
pip install stable-baselines3[extra] gymnasium numpy
```

### Stage 1-3: Staged Training (Sequential)

```bash
python training/train_staged.py \
  --phase1-max-steps 100000 \
  --phase2-max-steps 100000 \
  --phase3-steps 50000 \
  --eval-episodes 20 \
  --pool-size 10 \
  --checkpoint-interval 5000 \
  --output-dir models
```

**What happens:**

1. **Phase 1** (Human vs rule-based):
   - Trains for up to 100k steps
   - Evaluates every 5k steps
   - Stops early if escape rate > 20%
   - Saves: `models/player_stage1_final.zip`

2. **Phase 2** (Alien vs rule-based):
   - Trains for up to 100k steps
   - Evaluates every 5k steps
   - Stops early if catch rate > 30%
   - Saves: `models/alien_stage2_final.zip`

3. **Phase 3** (Both vs opponent pools):
   - Trains for 50k steps total
   - Opponent pool initialized with rule-based + stage 2/1 checkpoints
   - Saves checkpoint every 5k steps
   - Saves: `models/player_stage3_final.zip`, `models/alien_stage3_final.zip`
   - Outputs: `models/aet_metrics.jsonl` (evaluation history)

**Success Criteria**:
- Phase 1: escape_rate ≥ 20%
- Phase 2: catch_rate ≥ 30%
- Phase 3: both win_rates ≥ 25%, imbalance < 0.2

### Stage 4: ADA+ER Co-Training

```bash
python training/train_aet_with_ada_er.py \
  --rounds 50 \
  --round-steps 10000 \
  --eval-interval 2 \
  --eval-episodes 20 \
  --window-size 200 \
  --output-dir models
```

**What happens:**

1. **Initialization**:
   - Loads Stage 3 final checkpoints
   - Creates opponent pools (both ~75% latest, ~25% history)
   - Initializes ADA tracker (200-episode rolling window)
   - Initializes ER randomizer (threshold=0.3, persistence=20)

2. **Per Round** (50 total):
   - Compute rolling win rates from last 200 episodes
   - Compute imbalance: `|alien_wr - human_wr|`
   - **ADA allocation**: Weaker agent gets up to 2x steps
   - **ER check**: If imbalance > 0.3 for 20+ eps, adjust map difficulty
   - Train human for `alien_steps` step against sampled alien
   - Train alien for `human_steps` steps against sampled human
   - Evaluate both (20 episodes each)
   - Save metrics and checkpoints

3. **Outputs**:
   - `models/human_round_NNN.zip` (checkpoints per round)
   - `models/alien_round_NNN.zip`
   - `models/aet_ada_er_metrics.jsonl` (full history)

**Success Criteria**:
- Imbalance converges: 0.3+ → <0.05
- Final win rates: both ~50%
- No crashes, monotonic improvement

---

## Running Each Stage

### Stage 1-3: Complete Pipeline

**Command**:
```bash
cd /Users/mariaramos/Desktop/AASMA-1
python training/train_staged.py \
  --phase1-max-steps 100000 \
  --phase2-max-steps 100000 \
  --phase3-steps 50000 \
  --eval-episodes 20 \
  --pool-size 10 \
  --checkpoint-interval 5000 \
  --output-dir models
```

**Flags**:
- `--phase1-max-steps`: Max steps for human training (default: 100000)
- `--phase2-max-steps`: Max steps for alien training (default: 100000)
- `--phase3-steps`: Total steps for both-vs-pools (default: 50000)
- `--eval-episodes`: Episodes per evaluation (default: 20)
- `--pool-size`: Max historical checkpoints (default: 10)
- `--checkpoint-interval`: Save checkpoint every N steps (default: 5000)
- `--output-dir`: Where to save models (default: models/)

**Expected output**:
```
PHASE 1: Train HUMAN against rule-based alien
├─ Training progress (every 5k steps)
├─ Win rate: 0.05 → 0.12 → 0.18 → 0.22 ✓
├─ Early stop: 22% > 20%
└─ Saved: player_stage1_final.zip

PHASE 2: Train ALIEN against rule-based human
├─ Training progress (every 5k steps)
├─ Catch rate: 0.08 → 0.15 → 0.28 → 0.32 ✓
├─ Early stop: 32% > 30%
└─ Saved: alien_stage2_final.zip

PHASE 3: Train BOTH vs opponent pools
├─ Round 1: Pool sampling 75% latest
├─ Round 2-10: Both agents training
├─ Final imbalance: 0.15 ✓
└─ Saved: player_stage3_final.zip, alien_stage3_final.zip, checkpoint_*.zip
```

**Time**: 4-8 hours (CPU: 8h, GPU: 1-2h)

---

### Stage 4: ADA+ER Co-Training

**Command** (requires Stage 3 success):
```bash
python training/train_aet_with_ada_er.py \
  --rounds 50 \
  --round-steps 10000 \
  --eval-interval 2 \
  --eval-episodes 20 \
  --window-size 200 \
  --output-dir models
```

**Flags**:
- `--rounds`: Number of training rounds (default: 50)
- `--round-steps`: Total steps per round (default: 10000)
- `--eval-interval`: Evaluate every N rounds (default: 2)
- `--eval-episodes`: Episodes per evaluation (default: 20)
- `--window-size`: Rolling win-rate window (default: 200)
- `--pool-size`: Historical pool size (default: 10)
- `--output-dir`: Output directory (default: models/)

**Expected output**:
```
AET Round 1
├─ Win rates: Alien=35%, Human=65%, Imbalance=30%
├─ ADA allocation: Alien=15k steps, Human=5k steps
├─ ER check: Imbalance 30% > 0.3 threshold? NO
└─ Training complete

AET Round 2-5
├─ Win rates improving
├─ Imbalance: 30% → 25% → 20% → 15% (decreasing ✓)
├─ ADA allocation adapts each round
└─ ER activates when needed

...

AET Round 50
├─ Win rates: Alien=48%, Human=52%, Imbalance=4%
├─ ADA: Equal allocation
├─ ER: Inactive (balanced)
└─ Convergence achieved ✓
```

**Time**: 2-3 hours

---

## Metrics Philosophy

### Why Separate?

**Training Reward** (for optimizer):
- Role-specific, shaped for learning
- Immediate feedback loop
- Can be hand-tuned per phase

**Evaluation Metric** (for validation):
- Objective success: did agent achieve goal?
- Detects reward hacking: high reward ≠ high win rate
- Platform for ablations: what actually helped?

### Key Metrics

| Metric | Meaning | Target |
|--------|---------|--------|
| **Win Rate** | % episodes where agent succeeded | Alien: >30%, Human: >20%, Stage 3: >25% both |
| **Imbalance** | \|alien_wr - human_wr\| | Stage 3: <0.2, Stage 4: <0.05 |
| **Episode Length** | Steps to termination | Alien: 100-200, Human: 150-250 |
| **Path Efficiency** | Unique cells / total steps | >40% (good exploration) |
| **Idle Fraction** | Consecutive still steps / total | <15% (not stuck) |
| **Staleness Score** | HIGH_IDLE + LOW_EXPLORATION + ORBITING | <0.3 (healthy policy) |

### Reading Results

```bash
# Get latest metrics
tail -5 models/aet_metrics.jsonl | python -m json.tool
```

**Example output**:
```json
{
  "round": 25,
  "alien_wr": 0.48,
  "human_wr": 0.52,
  "imbalance": 0.04,
  "alien_avg_episode_length": 145,
  "human_avg_episode_length": 190,
  "path_efficiency_human": 0.52,
  "idle_fraction_human": 0.08,
  "staleness_score": 0.15,
  "ada_enabled": true,
  "er_mode": "NORMAL"
}
```

**Interpretation**:
- Imbalance 0.04 ✓ (converged)
- Both win rates ~50% ✓ (competitive)
- Path efficiency 52% ✓ (exploring)
- Idle fraction 8% ✓ (not stuck)
- Staleness score 0.15 ✓ (healthy)

---

## Files Explanation

### Core Training

#### `training/train_staged.py`
Implements Stages 1-3: progressive training with early stopping

**Key Classes**:
- `HistoricalOpponentPool`: Manages opponent checkpoints (75% latest, 25% history)
- `train_phase_1()`: Human vs rule-based alien
- `train_phase_2()`: Alien vs rule-based human  
- `train_phase_3()`: Both vs opponent pools

**Outputs**: Stage 3 checkpoints for Stage 4 initialization

#### `training/train_aet_with_ada_er.py`
Implements Stage 4: full AET with ADA+ER

**Key Classes**:
- `RolePipeline`: Encapsulates agent (network, buffer, checkpoints)
- `RollingWinRateTracker`: Tracks 200-episode window, computes imbalance
- `EnvironmentRandomizerAdvanced`: Triggers ER when imbalance persists
- `compute_ada_allocation()`: Allocates steps inversely to win rate

**Outputs**: Stage 4 checkpoints, detailed metrics with ADA/ER logs

### Environment & Rewards

#### `training/envs.py`
Gymnasium environment wrappers for both agents

**Key Classes**:
- `BaseAETEnv`: Core wrapper (5 actions: WAIT + 4 directions)
- `AlienEnv(BaseAETEnv)`: Alien-controlled environment
- `PlayerEnv(BaseAETEnv)`: Human-controlled environment

**Features**:
- Reward logging per episode (debug component contributions)
- Observation: 128-float Box (role-specific encoding)
- Discrete action space: 5 (WAIT + 4 cardinal walks)

#### `training/obs_rewards.py`
Asymmetric reward computation

**Key Functions**:
- `compute_alien_reward()`: Terminal (±20), pursuit (+0.3), exploration, cost (-0.01)
- `compute_player_reward()`: Terminal (±20), exit progress (+0.05), hide bonus (+0.5), danger (-0.2)
- `get_alien_obs()`: 128-float observation (belief map, hearing, position)
- `get_player_obs()`: 128-float observation (radar threat, exit distance, known map)

**Design**: Asymmetric shaping < terminal rewards (prevents hacking)

### Metrics & Validation

#### `training/metrics.py`
Separates training rewards from evaluation metrics

**Key Classes**:
- `EpisodeMetrics`: Per-episode stats (win, length, efficiency, idle)
- `EvaluationWindow`: Aggregates N episodes into summary
- `MetricsTracker`: Central repository for all windows
- `detect_staleness()`: Identifies HIGH_IDLE, LOW_EXPLORATION, ORBITING

**Outputs**: JSONL with full evaluation history

#### `training/environment_randomization.py`
Adjusts map difficulty to rebalance learning

**Key Classes**:
- `EnvironmentRandomizerAdvanced`: Tracks imbalance history, triggers ER
- `MapRandomizer`: Modifies tiles (add/remove hiding spots, adjust vision)
- `BalanceMode` enum: NORMAL, HELP_HUMAN, HELP_ALIEN

**Trigger**: If imbalance > 0.3 for 20+ consecutive episodes

### Game Core

#### `game.py`
Core game logic (pursuit, hiding, sensing)

#### `map_generator.py`
Procedural map generation with alpha (wall density) control

#### `agents/alien.py` & `agents/human.py`
Rule-based baselines for opponent pools

### Utilities

#### `simulate_warmup.py`
Visualizes episodes from trained checkpoints

```bash
python training/simulate_warmup.py \
  --alien-model models/alien_stage3_final.zip \
  --player-model models/player_stage3_final.zip \
  --output output/visualization.gif
```

---

## Troubleshooting

### Phase 1: Human win_rate stuck at 0%

**Symptom**: Human can't escape even against rule-based alien

**Diagnosis**:
1. Check if exit detection works:
   ```bash
   python -c "from game import Game; from map_generator import *; m=MapGenerator(60,40).generate(); print(f'Exit at: {[(y,x) for y,x in enumerate(m) if m[y,x]==int(Tile.EXIT)]}')"
   ```

2. Check reward magnitude:
   ```bash
   grep "exit_progress" training/obs_rewards.py
   ```

**Fix**: Increase exit progress reward (+0.05 → +0.10)

### Phase 2: Alien win_rate stuck at 0%

**Symptom**: Alien can't catch even against rule-based human

**Diagnosis**:
1. Check if hearing works:
   ```bash
   grep "steps_since_heard" training/obs_rewards.py
   ```

2. Check if pursuit distance is computed correctly:
   ```bash
   grep "_curr_dist" training/envs.py
   ```

**Fix**: Increase pursuit reward (+0.3 → +0.5) or check hearing range

### Phase 3: Imbalance stays high (>0.3)

**Symptom**: One agent dominates, pools not helping

**Diagnosis**:
1. Check pool sampling:
   ```bash
   grep "latest_prob" training/train_staged.py  # Should be 0.75
   ```

2. Check checkpoint frequency:
   - `--checkpoint-interval` should be 5000-10000 (not too rare)

**Fix**: 
- Increase `--pool-size` (10 → 20)
- Decrease checkpoint interval (5000 → 2000)

### Stage 4: ADA not rebalancing

**Symptom**: Win rates stuck at (30%, 70%), ADA allocation not changing

**Diagnosis**:
```bash
grep "alien_steps\|human_steps" training/train_aet_with_ada_er.py
```

**Fix**: Check `compute_ada_allocation()` formula is correct

### Stage 4: ER never activates

**Symptom**: ER_mode always "NORMAL", imbalance not improving

**Diagnosis**:
1. Check threshold:
   ```python
   randomizer = EnvironmentRandomizerAdvanced(enable_threshold=0.3)
   ```

2. Check persistence window:
   ```python
   randomizer = EnvironmentRandomizerAdvanced(persistence_window=20)
   ```

**Fix**: Lower threshold (0.3 → 0.2) or reduce persistence (20 → 10)

### General: Training won't start

**Error**: `ModuleNotFoundError: No module named 'stable_baselines3'`

**Fix**:
```bash
pip install stable-baselines3[extra] gymnasium numpy
```

### Training slow

**CPU vs GPU**:
- CPU: 8-16 hours total
- GPU: 1-3 hours total

**Speed up**:
- Use GPU if available
- Reduce `--eval-episodes` (20 → 10)
- Reduce `--round-steps` (10000 → 5000)

---

## Expected Outcomes

### After Stage 1-3

Human Policy:
- Win rate: 25-40%
- Episode length: 150-250 steps
- Path efficiency: 40-60%
- Robust vs rule-based and learned opponents

Alien Policy:
- Win rate: 25-40%
- Episode length: 100-200 steps
- Path efficiency: 40-60%
- Robust vs rule-based and learned opponents

### After Stage 4 (50 rounds)

Competitive Equilibrium:
- Alien win rate: 45-55%
- Human win rate: 45-55%
- Imbalance: < 5%
- Convergence: ~20-30 rounds with ADA

Properties:
- Both policies stable (low staleness)
- Diverse gameplay (path efficiency >40%)
- Adaptive to opponent changes (ER/ADA working)

---

## Advanced Topics

### Ablation Studies

To isolate which components help:

1. **No ADA**: Set `compute_ada_allocation()` to return `(base_steps, base_steps)`
2. **No ER**: Set `should_apply_er()` to always return `False`
3. **No Pools**: Set pool sampling to always use latest checkpoint
4. **No Shaping**: Set reward shaping coefficients to 0

Compare final imbalance and convergence speed between variants.

### pFSP Enhancement

Current: Uniform sampling from history
Proposed: Weight by skill-boundary distance (pFSP)

```python
# Future: Sample checkpoints nearest to current agents' skill level
# Instead of: random uniform
```

### Multi-Map Training

Rotate map seed every N rounds to test generalization:

```python
if round % 5 == 0:
    seed = random.randint(0, 10000)
    fixed_map = make_fixed_map(seed=seed)
```

---

## References

- Stable Baselines 3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
- pFSP (Historical Opponent Pools): OpenAI Five

---

## Contact

For issues or questions about the training pipeline, refer to `ABLATION_CHECKLIST.md` for concrete diagnostics or review the specific training file mentioned in the error message.
