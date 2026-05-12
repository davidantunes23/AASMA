"""Staged training for asymmetric AET setup.

Avoids instability from fresh-vs-fresh co-adaptation in asymmetric partially-observed games.

Phase 1: Train human PPO against fixed rule-based alien until >20% escape rate
Phase 2: Train alien PPO against fixed rule-based human until >30% catch rate  
Phase 3: Train both against mix of rule-based + frozen learned opponents with historical pools
Phase 4: Full AET-style co-training with opponent pools (later)
"""
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from map_generator import MapGenerator
from training.envs import AlienEnv, PlayerEnv


@dataclass
class CheckpointInfo:
    """Metadata for a saved checkpoint."""
    path: str
    update_num: int
    win_rate: Optional[float] = None
    total_episodes: int = 0
    catch_count: int = 0
    escape_count: int = 0


class HistoricalOpponentPool:
    """Manages a pool of historical opponent checkpoints.
    
    Simple version uses 70-80% latest + 20-30% historical uniform sampling.
    Infrastructure ready for pFSP-style skill-boundary sampling.
    """
    
    def __init__(self, max_pool_size: int = 10, latest_prob: float = 0.75):
        self.max_pool_size = max_pool_size
        self.latest_prob = latest_prob  # Probability of sampling latest
        self.checkpoints: List[CheckpointInfo] = []
        self.rng = random.Random()
    
    def add_checkpoint(self, path: str, update_num: int, win_rate: Optional[float] = None):
        """Add checkpoint to pool, evicting oldest if at capacity."""
        checkpoint = CheckpointInfo(path=path, update_num=update_num, win_rate=win_rate)
        self.checkpoints.append(checkpoint)
        
        # Keep only most recent max_pool_size checkpoints
        if len(self.checkpoints) > self.max_pool_size:
            self.checkpoints = self.checkpoints[-self.max_pool_size:]
    
    def sample_opponent(self) -> Optional[str]:
        """Sample opponent: 70-80% latest, 20-30% historical uniform.
        
        Returns: path to checkpoint, or None for rule-based opponent
        """
        if not self.checkpoints:
            return None
        
        # Sample from latest or history
        if self.rng.random() < self.latest_prob:
            return self.checkpoints[-1].path  # Latest
        else:
            # Uniform from history (excluding latest)
            if len(self.checkpoints) > 1:
                return self.rng.choice(self.checkpoints[:-1]).path
            else:
                return self.checkpoints[-1].path
    
    def update_win_rate(self, checkpoint_idx: int, outcome: str, is_win: bool):
        """Update checkpoint statistics (for future pFSP)."""
        if 0 <= checkpoint_idx < len(self.checkpoints):
            info = self.checkpoints[checkpoint_idx]
            info.total_episodes += 1
            if is_win:
                if outcome == "alien_caught_human":
                    info.catch_count += 1
                else:
                    info.escape_count += 1
            info.win_rate = (info.catch_count + info.escape_count) / info.total_episodes
    
    def get_pool_stats(self) -> Dict:
        """Return pool statistics for logging."""
        if not self.checkpoints:
            return {"size": 0}
        
        return {
            "size": len(self.checkpoints),
            "latest_update": self.checkpoints[-1].update_num,
            "oldest_update": self.checkpoints[0].update_num,
            "latest_win_rate": self.checkpoints[-1].win_rate,
        }


def make_fixed_map(seed=42, width=40, height=30, alpha=-0.1):
    """Create a fixed training map. 
    
    Default 40x30 (alpha=-0.1) is easier for baseline rule-based agents to escape.
    Full 60x40 (alpha=0.0) should be used AFTER learning basics.
    """
    gen = MapGenerator(width=width, height=height, alpha=alpha, seed=seed)
    grid = gen.generate()
    return grid


def evaluate_agent(role: str, model, opponent_model, fixed_map: np.ndarray,
                   episodes: int = 20, max_steps: int = 500, seed: int = 42,
                   alien_freeze_steps: int = 0):
    """Evaluate agent win rate against an opponent.

    Returns: win_count, total_episodes, outcome_counts
    """
    wins = 0
    outcomes = {"alien_caught_human": 0, "human_reached_exit": 0, "max_steps_reached": 0}

    for episode in range(episodes):
        if role == "alien":
            env = AlienEnv(fixed_map, max_steps=max_steps, opponent_model=opponent_model)
        else:
            env = PlayerEnv(fixed_map, max_steps=max_steps, opponent_model=opponent_model,
                            alien_freeze_steps=alien_freeze_steps)
        
        obs, _ = env.reset(seed=seed + episode)
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            outcome = info.get("outcome", "ongoing")
        
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        if (role == "alien" and outcome == "alien_caught_human") or \
           (role == "human" and outcome == "human_reached_exit"):
            wins += 1
    
    win_rate = wins / episodes
    return win_rate, outcomes


def train_phase_1(fixed_map: np.ndarray, args):
    """Phase 1: Train human against rule-based alien until >20% escape rate."""
    print("\n" + "="*60)
    print("PHASE 1: Train HUMAN against rule-based alien")
    print("="*60)
    print("Target: >20% escape rate")
    print(f"Max steps per phase: {args.phase1_max_steps}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment (opponent_model=None means rule-based alien)
    freeze = args.alien_freeze_steps
    player_env = DummyVecEnv([lambda: PlayerEnv(fixed_map, max_steps=args.max_steps,
                                                 opponent_model=None,
                                                 alien_freeze_steps=freeze)])
    
    # Create or load model
    player_model = PPO("MlpPolicy", env=player_env, verbose=1,
                       gamma=0.99, learning_rate=5e-5,
                       ent_coef=0.01, n_steps=400, batch_size=80,
                       device="cpu")
    
    # Training loop with periodic evaluation
    total_steps = 0
    best_win_rate = 0.0
    eval_interval = 5000
    
    while total_steps < args.phase1_max_steps:
        player_model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        total_steps += eval_interval
        
        # Evaluate
        win_rate, outcomes = evaluate_agent(
            "human", player_model, None, fixed_map,
            episodes=args.eval_episodes, max_steps=args.max_steps,
            alien_freeze_steps=args.alien_freeze_steps,
        )
        
        print(f"\n[Phase 1] Steps: {total_steps:7d} | Win Rate: {win_rate:.2%} | Outcomes: {outcomes}")
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            player_model.save(os.path.join(args.output_dir, "player_stage1_best.zip"))
        
        # Stop if threshold reached
        if win_rate >= 0.20:
            print(f"\n✓ Phase 1 COMPLETE: Reached {win_rate:.2%} escape rate (threshold: 20%)")
            player_model.save(os.path.join(args.output_dir, "player_stage1_final.zip"))
            return player_model, total_steps
    
    print(f"\n⚠ Phase 1 TIMEOUT: Best win rate {best_win_rate:.2%} (target: 20%)")
    player_model.save(os.path.join(args.output_dir, "player_stage1_final.zip"))
    return player_model, total_steps


def train_phase_2(fixed_map: np.ndarray, args):
    """Phase 2: Train alien against rule-based human until >30% catch rate."""
    print("\n" + "="*60)
    print("PHASE 2: Train ALIEN against rule-based human")
    print("="*60)
    print("Target: >30% catch rate")
    print(f"Max steps per phase: {args.phase2_max_steps}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create environment (opponent_model=None means rule-based human)
    alien_env = DummyVecEnv([lambda: AlienEnv(fixed_map, max_steps=args.max_steps,
                                              opponent_model=None)])
    
    # Create or load model
    alien_model = PPO("MlpPolicy", env=alien_env, verbose=1,
                      gamma=0.99, learning_rate=5e-5,
                      ent_coef=0.01, n_steps=400, batch_size=80,
                      device="cpu")
    
    # Training loop with periodic evaluation
    total_steps = 0
    best_win_rate = 0.0
    eval_interval = 5000
    
    while total_steps < args.phase2_max_steps:
        alien_model.learn(total_timesteps=eval_interval, reset_num_timesteps=False)
        total_steps += eval_interval
        
        # Evaluate
        win_rate, outcomes = evaluate_agent(
            "alien", alien_model, None, fixed_map,
            episodes=args.eval_episodes, max_steps=args.max_steps
        )
        
        print(f"\n[Phase 2] Steps: {total_steps:7d} | Win Rate: {win_rate:.2%} | Outcomes: {outcomes}")
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            alien_model.save(os.path.join(args.output_dir, "alien_stage2_best.zip"))
        
        # Stop if threshold reached
        if win_rate >= 0.30:
            print(f"\n✓ Phase 2 COMPLETE: Reached {win_rate:.2%} catch rate (threshold: 30%)")
            alien_model.save(os.path.join(args.output_dir, "alien_stage2_final.zip"))
            return alien_model, total_steps
    
    print(f"\n⚠ Phase 2 TIMEOUT: Best win rate {best_win_rate:.2%} (target: 30%)")
    alien_model.save(os.path.join(args.output_dir, "alien_stage2_final.zip"))
    return alien_model, total_steps


def train_phase_3(fixed_map: np.ndarray, player_model, alien_model, args):
    """Phase 3: Train both against mix of rule-based + frozen learned opponents.
    
    Maintains historical pools of checkpoints:
    - Human opponent pool: mix of frozen alien checkpoints + rule-based
    - Alien opponent pool: mix of frozen human checkpoints + rule-based
    
    Sampling: 75% latest learned opponent, 25% random from history (initially)
    """
    print("\n" + "="*60)
    print("PHASE 3: Train BOTH with historical opponent pools")
    print("="*60)
    print("Human: vs mix of frozen alien checkpoints + rule-based")
    print("Alien: vs mix of frozen human checkpoints + rule-based")
    print("Pool sampling: 75% latest, 25% history")
    print(f"Duration: {args.phase3_steps} steps each")
    print()
    
    # Create opponent pools
    alien_opponent_pool = HistoricalOpponentPool(
        max_pool_size=args.pool_size, latest_prob=0.75
    )
    human_opponent_pool = HistoricalOpponentPool(
        max_pool_size=args.pool_size, latest_prob=0.75
    )
    
    # Add Phase 2 trained models as initial pool members
    alien_opponent_pool.add_checkpoint(
        os.path.join(args.output_dir, "alien_stage2_final.zip"), 
        update_num=0, 
        win_rate=None
    )
    human_opponent_pool.add_checkpoint(
        os.path.join(args.output_dir, "player_stage1_final.zip"),
        update_num=0,
        win_rate=None
    )
    
    # Phase 3a: Train human against opponent pool
    print("[Phase 3a] Training HUMAN against alien opponent pool...")
    checkpoint_interval = args.checkpoint_interval
    total_steps = 0
    update_count = 0
    
    while total_steps < args.phase3_steps // 2:
        # Sample opponent for next batch of training
        sampled_opponent_path = alien_opponent_pool.sample_opponent()
        
        if sampled_opponent_path and os.path.exists(sampled_opponent_path):
            sampled_opponent = PPO.load(sampled_opponent_path)
        else:
            sampled_opponent = None  # Use rule-based
        
        # Create environment with sampled (frozen) opponent
        player_env = DummyVecEnv([
            lambda opponent=sampled_opponent: PlayerEnv(
                fixed_map, max_steps=args.max_steps, opponent_model=opponent
            )
        ])
        player_model.set_env(player_env)
        
        # Train for checkpoint interval
        player_model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        total_steps += checkpoint_interval
        update_count += 1
        
        # Save checkpoint periodically
        if update_count % 2 == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"player_stage3_checkpoint_{update_count:03d}.zip"
            )
            player_model.save(checkpoint_path)
            human_opponent_pool.add_checkpoint(checkpoint_path, update_num=update_count)
            print(f"  [Human] Saved checkpoint {update_count} | Pool size: {len(human_opponent_pool.checkpoints)}")
    
    player_model.save(os.path.join(args.output_dir, "player_stage3_final.zip"))
    print(f"[Phase 3a] COMPLETE: {total_steps} steps, {len(human_opponent_pool.checkpoints)} human checkpoints")
    
    # Phase 3b: Train alien against opponent pool
    print("\n[Phase 3b] Training ALIEN against human opponent pool...")
    total_steps = 0
    update_count = 0
    
    while total_steps < args.phase3_steps // 2:
        # Sample opponent for next batch of training
        sampled_opponent_path = human_opponent_pool.sample_opponent()
        
        if sampled_opponent_path and os.path.exists(sampled_opponent_path):
            sampled_opponent = PPO.load(sampled_opponent_path)
        else:
            sampled_opponent = None  # Use rule-based
        
        # Create environment with sampled (frozen) opponent
        alien_env = DummyVecEnv([
            lambda opponent=sampled_opponent: AlienEnv(
                fixed_map, max_steps=args.max_steps, opponent_model=opponent
            )
        ])
        alien_model.set_env(alien_env)
        
        # Train for checkpoint interval
        alien_model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        total_steps += checkpoint_interval
        update_count += 1
        
        # Save checkpoint periodically
        if update_count % 2 == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"alien_stage3_checkpoint_{update_count:03d}.zip"
            )
            alien_model.save(checkpoint_path)
            alien_opponent_pool.add_checkpoint(checkpoint_path, update_num=update_count)
            print(f"  [Alien] Saved checkpoint {update_count} | Pool size: {len(alien_opponent_pool.checkpoints)}")
    
    alien_model.save(os.path.join(args.output_dir, "alien_stage3_final.zip"))
    print(f"[Phase 3b] COMPLETE: {total_steps} steps, {len(alien_opponent_pool.checkpoints)} alien checkpoints")
    
    # Final evaluation
    print("\n" + "-"*60)
    print("Phase 3 - Final Evaluation (vs latest learned opponent):")
    print("-"*60)
    
    # Evaluate against latest opponent in pool
    latest_alien = PPO.load(alien_opponent_pool.checkpoints[-1].path) if alien_opponent_pool.checkpoints else None
    player_wr, player_outcomes = evaluate_agent(
        "human", player_model, latest_alien, fixed_map,
        episodes=args.eval_episodes, max_steps=args.max_steps
    )
    print(f"Human vs latest alien: {player_wr:.2%} escape rate | {player_outcomes}")
    
    latest_human = PPO.load(human_opponent_pool.checkpoints[-1].path) if human_opponent_pool.checkpoints else None
    alien_wr, alien_outcomes = evaluate_agent(
        "alien", alien_model, latest_human, fixed_map,
        episodes=args.eval_episodes, max_steps=args.max_steps
    )
    print(f"Alien vs latest human: {alien_wr:.2%} catch rate | {alien_outcomes}")
    
    # Log pool statistics
    print("\n" + "-"*60)
    print("Historical Pools:")
    print("-"*60)
    print(f"Human opponent pool: {json.dumps(human_opponent_pool.get_pool_stats(), indent=2)}")
    print(f"Alien opponent pool: {json.dumps(alien_opponent_pool.get_pool_stats(), indent=2)}")
    
    return player_model, alien_model, human_opponent_pool, alien_opponent_pool


def main(args):
    fixed_map = make_fixed_map(seed=args.seed, width=args.map_width, height=args.map_height, alpha=args.map_alpha)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    config = {
        "seed": args.seed,
        "map_width": args.map_width,
        "map_height": args.map_height,
        "map_alpha": args.map_alpha,
        "max_steps": args.max_steps,
        "phase1_max_steps": args.phase1_max_steps,
        "phase2_max_steps": args.phase2_max_steps,
        "phase3_steps": args.phase3_steps,
        "eval_episodes": args.eval_episodes,
        "pool_size": args.pool_size,
        "checkpoint_interval": args.checkpoint_interval,
    }
    with open(os.path.join(args.output_dir, "staged_training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Phase 1: Train human
    player_model, phase1_steps = train_phase_1(fixed_map, args)
    
    # Phase 2: Train alien
    alien_model, phase2_steps = train_phase_2(fixed_map, args)
    
    # Phase 3: Train both against mixed opponents with historical pools
    player_model, alien_model, human_opponent_pool, alien_opponent_pool = train_phase_3(
        fixed_map, player_model, alien_model, args
    )
    
    print("\n" + "="*60)
    print("STAGED TRAINING COMPLETE")
    print("="*60)
    print(f"Phase 1 (human):  {phase1_steps:7d} steps")
    print(f"Phase 2 (alien):  {phase2_steps:7d} steps")
    print(f"Phase 3 (both):   {args.phase3_steps:7d} steps")
    print(f"\nModels saved to: {args.output_dir}/")
    print(f"Human opponent pool size: {len(human_opponent_pool.checkpoints)}")
    print(f"Alien opponent pool size: {len(alien_opponent_pool.checkpoints)}")
    print("\nNext step: Run train_aet_loop.py for full AET co-training")
    print("  Use stage3 checkpoints or pools as initializers for co-training")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Staged training to avoid co-adaptation instability with historical opponent pools"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for map generation")
    parser.add_argument("--map-width", type=int, default=40,
                        help="Map width (default 40 for easier rule-based learning)")
    parser.add_argument("--map-height", type=int, default=30,
                        help="Map height (default 30 for easier rule-based learning)")
    parser.add_argument("--map-alpha", type=float, default=-0.1,
                        help="Map difficulty alpha (default -0.1 for fewer walls, use 0.0 for full difficulty)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--phase1-max-steps", type=int, default=100000,
                        help="Max training steps for Phase 1 (human)")
    parser.add_argument("--phase2-max-steps", type=int, default=100000,
                        help="Max training steps for Phase 2 (alien)")
    parser.add_argument("--phase3-steps", type=int, default=50000,
                        help="Training steps for Phase 3 (both)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Episodes to evaluate per checkpoint")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--pool-size", type=int, default=10,
                        help="Max size of historical opponent pool")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--alien-freeze-steps", type=int, default=150,
                        help="Phase 1: alien stays frozen for this many steps per episode (curriculum)")
    
    args = parser.parse_args()
    main(args)
