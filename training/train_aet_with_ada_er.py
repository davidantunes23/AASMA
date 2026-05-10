#!/usr/bin/env python3
"""
Advanced AET training with Adaptive Data Adjustment (ADA), Environment Randomization (ER),
and simple prioritized fictitious self-play (pFSP)-style opponent selection.

Implements:
1. Separate training pipelines per role (distinct networks, buffers, learning rates)
2. Adaptive Data Adjustment: allocate more updates to weaker agent based on rolling win rates
3. Environment Randomization: adjust map difficulty to rebalance when one side dominates
4. Historical opponent pools for stability
5. pFSP-inspired opponent prioritization based on per-checkpoint win rates
"""

import os
import sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import json
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from map_generator import MapGenerator, Tile
from training.envs import AlienEnv, PlayerEnv
from training.train_staged import HistoricalOpponentPool

from training.environment_randomization import (
    EnvironmentRandomizerAdvanced,
    BalanceMode,
    ERScenario,
    create_balanced_er_config,
)


# ────────────────────────── Pipelines & tracking ───────────────────────────────

@dataclass
class RolePipeline:
    """Encapsulates training pipeline for one role (human or alien)."""
    role: str  # "human" or "alien"
    model: PPO
    opponent_pool: HistoricalOpponentPool
    checkpoint_dir: str
    learning_rate: float = 5e-5
    gamma: float = 0.99
    ent_coef: float = 0.01
    n_steps: int = 400
    batch_size: int = 64

    # Training state
    total_steps: int = 0
    total_episodes: int = 0
    update_count: int = 0

    # Performance tracking
    win_history: deque = field(default_factory=lambda: deque(maxlen=200))
    recent_outcomes: Dict[str, int] = field(default_factory=lambda: {
        "win": 0, "loss": 0, "timeout": 0
    })


@dataclass
class RollingWinRateTracker:
    """Tracks rolling win rates and computes imbalance."""
    window_size: int = 200  # Keep last N episodes
    alien_wins: deque = field(default_factory=deque)
    human_wins: deque = field(default_factory=deque)
    episode_count: int = 0

    def record_outcome(self, outcome: str):
        """Record episode outcome."""
        self.episode_count += 1

        if outcome == "alien_caught_human":
            self.alien_wins.append(True)
            self.human_wins.append(False)
        elif outcome == "human_reached_exit":
            self.alien_wins.append(False)
            self.human_wins.append(True)
        else:  # timeout
            self.alien_wins.append(False)
            self.human_wins.append(False)

        # Trim to window
        if len(self.alien_wins) > self.window_size:
            self.alien_wins.popleft()
            self.human_wins.popleft()

    def get_win_rates(self) -> Tuple[float, float]:
        """Return (alien_win_rate, human_win_rate)."""
        if not self.alien_wins:
            return 0.5, 0.5

        alien_wr = sum(self.alien_wins) / len(self.alien_wins)
        human_wr = sum(self.human_wins) / len(self.human_wins)
        return alien_wr, human_wr

    def get_imbalance(self) -> float:
        """Return imbalance metric: abs(alien_wr - human_wr)."""
        alien_wr, human_wr = self.get_win_rates()
        return abs(alien_wr - human_wr)


# ───────────────────────────── Evaluation helper ───────────────────────────────

def run_episode_collect_stats(
    role: str,
    model: PPO,
    opponent_model: Optional[PPO],
    fixed_map: np.ndarray,
    max_steps: int = 500,
) -> Dict:
    """Run single evaluation episode on the fixed map and collect outcome."""
    if role == "human":
        env = PlayerEnv(fixed_map, max_steps=max_steps, opponent_model=opponent_model)
    else:
        env = AlienEnv(fixed_map, max_steps=max_steps, opponent_model=opponent_model)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    episode_steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        episode_steps += 1
        done = terminated or truncated

    outcome = info.get("outcome", "ongoing")

    return {
        "outcome": outcome,
        "total_reward": total_reward,
        "steps": episode_steps,
        "role": role,
    }


# ───────────────────────── pFSP-style opponent sampling ───────────────────────

def pfs_sample_opponent(
    pool: HistoricalOpponentPool,
    rng: np.random.Generator,
    default_latest_prob: float = 0.75,
) -> Optional[str]:
    """
    pFSP-inspired sampling from HistoricalOpponentPool.

    If no per-checkpoint win_rate is available, fall back to the pool's
    built-in latest/history rule. Once win_rate is known, prioritize
    opponents whose win_rate is closest to 0.5 (hardest).
    """
    if not pool.checkpoints:
        return None

    # If only one checkpoint, just return it
    if len(pool.checkpoints) == 1:
        return pool.checkpoints[0].path

    # Collect win rates; if many are None, we fallback to simple logic
    win_rates = [c.win_rate for c in pool.checkpoints]
    any_stats = any(wr is not None for wr in win_rates)

    if not any_stats:
        # Fallback: replicate old behaviour (latest vs history)
        if rng.random() < default_latest_prob:
            return pool.checkpoints[-1].path
        else:
            return rng.choice(pool.checkpoints[:-1]).path

    # Build priorities: opponents near win_rate==0.5 get highest weight
    priorities = []
    for wr in win_rates:
        if wr is None:
            # Unknown difficulty; give low but non-zero weight
            priorities.append(0.1)
        else:
            # Difficulty highest around 0.5; use Gaussian-like bump
            diff = abs(wr - 0.5)
            # Smaller diff -> larger weight; k controls sharpness
            k = 10.0
            priorities.append(float(np.exp(-k * diff)))

    priorities = np.asarray(priorities, dtype=np.float64)
    if priorities.sum() <= 0:
        # Degenerate; fallback again
        if rng.random() < default_latest_prob:
            return pool.checkpoints[-1].path
        else:
            return rng.choice(pool.checkpoints[:-1]).path

    probs = priorities / priorities.sum()
    idx = int(rng.choice(len(pool.checkpoints), p=probs))
    return pool.checkpoints[idx].path


# ────────────────────── ADA allocation (unchanged) ────────────────────────────

def compute_ada_allocation(
    alien_wr: float,
    human_wr: float,
    base_steps: int,
) -> Tuple[int, int]:
    """Compute ADA step allocation based on win rate imbalance.

    More steps to weaker side. Returns (alien_steps, human_steps).
    """
    imbalance = abs(alien_wr - human_wr)

    if imbalance < 0.05:
        # Balanced: equal allocation
        return base_steps, base_steps

    # Allocate extra steps to weaker side
    if alien_wr < human_wr:
        # Alien is weaker, give it more
        alien_scale = 1.0 + imbalance * 2.0  # Up to 2x multiplier
        human_scale = max(0.5, 1.0 - imbalance)
    else:
        # Human is weaker
        human_scale = 1.0 + imbalance * 2.0
        alien_scale = max(0.5, 1.0 - imbalance)

    total_scale = alien_scale + human_scale
    alien_steps = int(base_steps * (alien_scale / total_scale))
    human_steps = int(base_steps * (human_scale / total_scale))

    return alien_steps, human_steps


# ─────────────────────── One AET training round ───────────────────────────────

def train_aet_round(
    human_pipeline: RolePipeline,
    alien_pipeline: RolePipeline,
    fixed_map: np.ndarray,
    fixed_metadata: Dict,
    tracker: RollingWinRateTracker,
    randomizer: EnvironmentRandomizerAdvanced,
    args,
) -> Dict:
    """Run one full AET training round with both agents."""

    print(f"\n{'='*70}")
    print(f"AET Round {human_pipeline.update_count + 1}")
    print(f"{'='*70}")

    # Get current win rates and compute allocation
    alien_wr, human_wr = tracker.get_win_rates()
    imbalance = tracker.get_imbalance()

    print(
        f"Current win rates: Alien={alien_wr:.2%}, "
        f"Human={human_wr:.2%}, Imbalance={imbalance:.2%}"
    )

    # Update ER with both win rates (signed imbalance handled inside)
    current_mode = randomizer.update_imbalance(alien_wr, human_wr)

    # High-level ER config for logging / optional use
    er_cfg = create_balanced_er_config(randomizer, alien_wr, human_wr, apply_now=False)

    if randomizer.should_apply_er():
        print(
            f"⚠️  ER ACTIVE (candidate): {current_mode.value} "
            f"(strength={randomizer.adjustment_strength:.2f})"
        )

    # Compute ADA allocation
    alien_steps, human_steps = compute_ada_allocation(
        alien_wr, human_wr, args.round_steps
    )
    print(
        f"ADA allocation: Alien={alien_steps:6d} steps, "
        f"Human={human_steps:6d} steps"
    )

    # ── Decide map & ER scenario for this round ───────────────────────────────
    training_map = fixed_map.copy()
    training_metadata = fixed_metadata.copy()
    episode_mods = {
        "alien_vision": 1.0,
        "human_vision": 1.0,
        "alien_speed": 1.0,
        "human_speed": 1.0,
    }

    er_applied = False
    scenario: ERScenario | None = None

    if randomizer.should_apply_er():
        # Sample scenario and regenerate map
        scenario = randomizer.sample_scenario()
        training_map, training_metadata = randomizer.randomize_map(
            fixed_map,
            fixed_metadata,
            scenario,
            seed=args.seed + human_pipeline.update_count,
        )
        mods = randomizer.get_episode_modifiers(scenario)
        episode_mods.update(
            {
                "alien_vision": mods["alien_vision"],
                "human_vision": mods["human_vision"],
                "alien_speed": mods["alien_speed"],
                "human_speed": mods["human_speed"],
            }
        )
        er_applied = True
        print(
            f"    ER applied: mode={randomizer.current_mode.value}, "
            f"alpha_shift={scenario.alpha_shift:.2f}, "
            f"target_pe_scale={scenario.target_player_exit_scale:.2f}, "
            f"target_ap_scale={scenario.target_alien_player_scale:.2f}"
        )
    else:
        print("    ER not applied this round (using fixed map).")

    # ── Train human ────────────────────────────────────────────────────────────
    print(f"\n[Human] Training for {human_steps} steps...")
    # Training sampling: keep simple latest+history; pFSP is mainly used in eval
    sampled_alien_path = alien_pipeline.opponent_pool.sample_opponent()
    if sampled_alien_path and os.path.exists(sampled_alien_path):
        opponent_model = PPO.load(sampled_alien_path)
    else:
        opponent_model = None

    human_env = DummyVecEnv(
        [
            lambda: PlayerEnv(
                training_map,
                max_steps=args.max_steps,
                opponent_model=opponent_model,
            )
        ]
    )
    human_pipeline.model.set_env(human_env)
    human_pipeline.model.learn(total_timesteps=human_steps, reset_num_timesteps=False)
    human_pipeline.total_steps += human_steps
    human_pipeline.update_count += 1

    # ── Train alien ────────────────────────────────────────────────────────────
    print(f"\n[Alien] Training for {alien_steps} steps...")
    sampled_human_path = human_pipeline.opponent_pool.sample_opponent()
    if sampled_human_path and os.path.exists(sampled_human_path):
        opponent_model = PPO.load(sampled_human_path)
    else:
        opponent_model = None

    alien_env = DummyVecEnv(
        [
            lambda: AlienEnv(
                training_map,
                max_steps=args.max_steps,
                opponent_model=opponent_model,
            )
        ]
    )
    alien_pipeline.model.set_env(alien_env)
    alien_pipeline.model.learn(total_timesteps=alien_steps, reset_num_timesteps=False)
    alien_pipeline.total_steps += alien_steps
    alien_pipeline.update_count += 1

    # ── Periodic evaluation on fixed map (pFSP updates happen here) ───────────
    if human_pipeline.update_count % args.eval_interval == 0:
        print(f"\n[Evaluation] Running {args.eval_episodes} episodes per side...")

        # RNG for pFSP sampling
        pfsp_rng = np.random.default_rng(seed=42 + human_pipeline.update_count)

        for _ in range(args.eval_episodes):
            # Sample alien opponent using pFSP from alien pool
            sampled_alien = None
            sampled_alien_path = pfs_sample_opponent(
                alien_pipeline.opponent_pool, pfsp_rng
            )
            if sampled_alien_path and os.path.exists(sampled_alien_path):
                sampled_alien = PPO.load(sampled_alien_path)

            # Sample human opponent using pFSP from human pool
            sampled_human = None
            sampled_human_path = pfs_sample_opponent(
                human_pipeline.opponent_pool, pfsp_rng
            )
            if sampled_human_path and os.path.exists(sampled_human_path):
                sampled_human = PPO.load(sampled_human_path)

            # Run episodes (on original fixed map, not randomized)
            human_result = run_episode_collect_stats(
                "human",
                human_pipeline.model,
                sampled_alien,
                fixed_map,
                args.max_steps,
            )
            alien_result = run_episode_collect_stats(
                "alien",
                alien_pipeline.model,
                sampled_human,
                fixed_map,
                args.max_steps,
            )

            # Track global outcomes
            tracker.record_outcome(human_result["outcome"])
            tracker.record_outcome(alien_result["outcome"])

            # Update pipelines (for simple win/loss counting)
            if human_result["outcome"] == "human_reached_exit":
                human_pipeline.recent_outcomes["win"] += 1
            elif human_result["outcome"] == "alien_caught_human":
                human_pipeline.recent_outcomes["loss"] += 1
            else:
                human_pipeline.recent_outcomes["timeout"] += 1

            if alien_result["outcome"] == "alien_caught_human":
                alien_pipeline.recent_outcomes["win"] += 1
            elif alien_result["outcome"] == "human_reached_exit":
                alien_pipeline.recent_outcomes["loss"] += 1
            else:
                alien_pipeline.recent_outcomes["timeout"] += 1

            # OPTIONAL: hook here to update per-checkpoint win stats in the pools.
            # Since HistoricalOpponentPool.update_win_rate expects an index,
            # and we only have paths here, you could extend the pool to provide
            # a mapping from path -> index and update win_rate accordingly.

        # Save checkpoints
        human_checkpoint = os.path.join(
            human_pipeline.checkpoint_dir,
            f"human_round_{human_pipeline.update_count:03d}.zip",
        )
        alien_checkpoint = os.path.join(
            alien_pipeline.checkpoint_dir,
            f"alien_round_{alien_pipeline.update_count:03d}.zip",
        )
        human_pipeline.model.save(human_checkpoint)
        alien_pipeline.model.save(alien_checkpoint)

        human_pipeline.opponent_pool.add_checkpoint(
            human_checkpoint, human_pipeline.update_count
        )
        alien_pipeline.opponent_pool.add_checkpoint(
            alien_checkpoint, alien_pipeline.update_count
        )

        # Print results
        alien_wr, human_wr = tracker.get_win_rates()
        print(
            f"  Human: {human_pipeline.recent_outcomes} | "
            f"Pool size: {len(human_pipeline.opponent_pool.checkpoints)}"
        )
        print(
            f"  Alien: {alien_pipeline.recent_outcomes} | "
            f"Pool size: {len(alien_pipeline.opponent_pool.checkpoints)}"
        )
        print(f"  Current rates: Alien={alien_wr:.2%}, Human={human_wr:.2%}")

    # Metrics for logging
    alien_wr, human_wr = tracker.get_win_rates()
    return {
        "round": human_pipeline.update_count,
        "human_steps": human_steps,
        "alien_steps": alien_steps,
        "alien_wr": float(alien_wr),
        "human_wr": float(human_wr),
        "imbalance": float(imbalance),
        "er_active": bool(er_applied),
        "er_mode": randomizer.current_mode.value if er_applied else "none",
        "er_strength": float(randomizer.adjustment_strength) if er_applied else 0.0,
    }


# ────────────────────────────── Main entrypoint ────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate single fixed reference map & metadata (alpha=0.0 baseline)
    gen = MapGenerator(width=60, height=40, alpha=0.0, seed=args.seed)
    fixed_map = gen.generate()
    fixed_metadata = gen.metadata

    print("=" * 70)
    print(
        "AET Training with Adaptive Data Adjustment (ADA) "
        "& Environment Randomization (ER) + pFSP opponent sampling"
    )
    print("=" * 70)

    # Load Phase 3 checkpoints as initializers
    human_model = PPO.load(
        os.path.join(args.output_dir, "player_stage3_final.zip"),
        env=DummyVecEnv([lambda: PlayerEnv(fixed_map, max_steps=args.max_steps)]),
    )
    alien_model = PPO.load(
        os.path.join(args.output_dir, "alien_stage3_final.zip"),
        env=DummyVecEnv([lambda: AlienEnv(fixed_map, max_steps=args.max_steps)]),
    )

    # Create pipelines
    human_pipeline = RolePipeline(
        role="human",
        model=human_model,
        opponent_pool=HistoricalOpponentPool(max_pool_size=args.pool_size),
        checkpoint_dir=args.output_dir,
    )
    alien_pipeline = RolePipeline(
        role="alien",
        model=alien_model,
        opponent_pool=HistoricalOpponentPool(max_pool_size=args.pool_size),
        checkpoint_dir=args.output_dir,
    )

    # Initialize pools with stage3 checkpoints
    human_pipeline.opponent_pool.add_checkpoint(
        os.path.join(args.output_dir, "player_stage3_final.zip"), 0
    )
    alien_pipeline.opponent_pool.add_checkpoint(
        os.path.join(args.output_dir, "alien_stage3_final.zip"), 0
    )

    # Trackers
    tracker = RollingWinRateTracker(window_size=args.window_size)
    randomizer = EnvironmentRandomizerAdvanced(
        enable_threshold=0.3,
        persistence_window=20,
    )

    # Training loop
    metrics = []
    for _ in range(args.rounds):
        round_metrics = train_aet_round(
            human_pipeline,
            alien_pipeline,
            fixed_map,
            fixed_metadata,
            tracker,
            randomizer,
            args,
        )
        metrics.append(round_metrics)

        # Save metrics
        metrics_file = os.path.join(args.output_dir, "aet_ada_er_metrics.jsonl")
        with open(metrics_file, "a") as f:
            f.write(json.dumps(round_metrics) + "\n")

    print("\n" + "=" * 70)
    print("AET Training Complete")
    print("=" * 70)
    print(f"Final models saved to: {args.output_dir}/")
    print(
        f"Metrics saved to: "
        f"{os.path.join(args.output_dir, 'aet_ada_er_metrics.jsonl')}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AET training with ADA, ER, and pFSP-style opponent sampling"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--round-steps", type=int, default=10000)
    parser.add_argument(
        "--eval-interval", type=int, default=2, help="Evaluate every N rounds"
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="models_hard")
    parser.add_argument(
        "--pool-size", type=int, default=10, help="Max historical opponent pool size"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Rolling win-rate window for imbalance tracking",
    )

    args = parser.parse_args()
    main(args)