"""
Evaluation Metrics Module - Separate from Training Rewards

Tracks: win rate, catch rate, escape rate, episode length, path efficiency.
NOT mean reward (reward is for learning, metrics are for validation).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode evaluation."""
    role: str  # "human" or "alien"
    outcome: str  # "human_reached_exit", "alien_caught_human", "timeout"
    episode_length: int
    unique_cells_visited: int
    total_cells_available: int
    consecutive_idle_steps: int
    max_consecutive_idle: int
    episode_reward: float  # For correlation analysis only
    opponent_role: str
    opponent_type: str  # "rule_based", "learned", "checkpoint_NNN"
    
    def __post_init__(self):
        # Compute derived metrics
        self.path_efficiency = (
            self.unique_cells_visited / self.total_cells_available
            if self.total_cells_available > 0 else 0.0
        )
        self.idle_fraction = (
            self.consecutive_idle_steps / self.episode_length
            if self.episode_length > 0 else 0.0
        )


@dataclass
class EvaluationWindow:
    """Aggregated metrics over N episodes (one evaluation window)."""
    window_id: int
    update_num: int
    role: str
    opponent_type: str
    num_episodes: int = 0
    
    # Win condition outcomes
    wins: int = 0  # Successes for this role
    losses: int = 0  # Failures for this role
    timeouts: int = 0  # Timeout (no win/loss)
    
    # Aggregate statistics
    avg_episode_length: float = 0.0
    max_episode_length: int = 0
    min_episode_length: int = 999999
    
    avg_unique_cells: float = 0.0
    max_unique_cells: int = 0
    
    avg_path_efficiency: float = 0.0
    avg_idle_fraction: float = 0.0
    avg_max_consecutive_idle: float = 0.0
    
    avg_episode_reward: float = 0.0  # For correlation tracking
    
    # Episode records for post-hoc analysis
    episodes: List[EpisodeMetrics] = field(default_factory=list)
    
    def compute_aggregates(self):
        """Compute aggregate statistics from episode list."""
        if not self.episodes:
            return
        
        self.num_episodes = len(self.episodes)
        
        # Count outcomes
        self.wins = sum(1 for e in self.episodes if self._is_win(e))
        self.losses = sum(1 for e in self.episodes if self._is_loss(e))
        self.timeouts = sum(1 for e in self.episodes if e.outcome == "timeout")
        
        # Aggregate episode metrics
        self.avg_episode_length = np.mean([e.episode_length for e in self.episodes])
        self.max_episode_length = max([e.episode_length for e in self.episodes])
        self.min_episode_length = min([e.episode_length for e in self.episodes])
        
        self.avg_unique_cells = np.mean([e.unique_cells_visited for e in self.episodes])
        self.max_unique_cells = max([e.unique_cells_visited for e in self.episodes])
        
        self.avg_path_efficiency = np.mean([e.path_efficiency for e in self.episodes])
        self.avg_idle_fraction = np.mean([e.idle_fraction for e in self.episodes])
        self.avg_max_consecutive_idle = np.mean([e.max_consecutive_idle for e in self.episodes])
        
        self.avg_episode_reward = np.mean([e.episode_reward for e in self.episodes])
    
    def _is_win(self, episode: EpisodeMetrics) -> bool:
        """Check if episode is a win for this role."""
        if self.role == "human":
            return episode.outcome == "human_reached_exit"
        else:  # alien
            return episode.outcome == "alien_caught_human"
    
    def _is_loss(self, episode: EpisodeMetrics) -> bool:
        """Check if episode is a loss for this role."""
        return not self._is_win(episode) and episode.outcome != "timeout"
    
    def get_win_rate(self) -> float:
        """Win rate for this role."""
        if self.num_episodes == 0:
            return 0.0
        return self.wins / self.num_episodes
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "window_id": self.window_id,
            "update_num": self.update_num,
            "role": self.role,
            "opponent_type": self.opponent_type,
            "num_episodes": self.num_episodes,
            "win_rate": self.get_win_rate(),
            "wins": self.wins,
            "losses": self.losses,
            "timeouts": self.timeouts,
            "avg_episode_length": float(self.avg_episode_length),
            "max_episode_length": self.max_episode_length,
            "min_episode_length": self.min_episode_length,
            "avg_unique_cells": float(self.avg_unique_cells),
            "max_unique_cells": self.max_unique_cells,
            "avg_path_efficiency": float(self.avg_path_efficiency),
            "avg_idle_fraction": float(self.avg_idle_fraction),
            "avg_max_consecutive_idle": float(self.avg_max_consecutive_idle),
            "avg_episode_reward": float(self.avg_episode_reward),
        }


class MetricsTracker:
    """Central tracker for evaluation metrics across training."""
    
    def __init__(self):
        self.windows: List[EvaluationWindow] = []
        self.window_counter = 0
        self.update_counter = 0
    
    def create_window(self, role: str, opponent_type: str) -> EvaluationWindow:
        """Create new evaluation window."""
        window = EvaluationWindow(
            window_id=self.window_counter,
            update_num=self.update_counter,
            role=role,
            opponent_type=opponent_type,
        )
        self.window_counter += 1
        return window
    
    def record_window(self, window: EvaluationWindow):
        """Record completed window."""
        window.compute_aggregates()
        self.windows.append(window)
    
    def update(self):
        """Increment update counter."""
        self.update_counter += 1
    
    def get_latest_human_wr(self) -> float:
        """Get latest human win rate against any opponent."""
        human_windows = [w for w in self.windows if w.role == "human"]
        if not human_windows:
            return 0.5
        return human_windows[-1].get_win_rate()
    
    def get_latest_alien_wr(self) -> float:
        """Get latest alien win rate against any opponent."""
        alien_windows = [w for w in self.windows if w.role == "alien"]
        if not alien_windows:
            return 0.5
        return alien_windows[-1].get_win_rate()
    
    def get_imbalance(self) -> float:
        """Get current imbalance."""
        return abs(self.get_latest_alien_wr() - self.get_latest_human_wr())
    
    def save_to_jsonl(self, filepath: str):
        """Save all windows to JSONL."""
        with open(filepath, "w") as f:
            for window in self.windows:
                f.write(json.dumps(window.to_dict()) + "\n")
    
    def print_summary(self):
        """Print human-readable summary."""
        if not self.windows:
            print("No evaluation windows recorded.")
            return
        
        print("\n" + "="*70)
        print("EVALUATION METRICS SUMMARY")
        print("="*70)
        
        # Group by role
        human_windows = [w for w in self.windows if w.role == "human"]
        alien_windows = [w for w in self.windows if w.role == "alien"]
        
        if human_windows:
            print("\n[HUMAN] Latest window:")
            latest_human = human_windows[-1]
            print(f"  vs {latest_human.opponent_type}: {latest_human.get_win_rate():.1%} win rate")
            print(f"    Episodes: {latest_human.num_episodes} | Length: {latest_human.avg_episode_length:.0f}±{latest_human.max_episode_length-latest_human.min_episode_length:.0f}")
            print(f"    Path eff: {latest_human.avg_path_efficiency:.1%} | Idle: {latest_human.avg_idle_fraction:.1%}")
            print(f"    Unique cells: {latest_human.avg_unique_cells:.0f}/{latest_human.max_unique_cells}")
        
        if alien_windows:
            print("\n[ALIEN] Latest window:")
            latest_alien = alien_windows[-1]
            print(f"  vs {latest_alien.opponent_type}: {latest_alien.get_win_rate():.1%} win rate")
            print(f"    Episodes: {latest_alien.num_episodes} | Length: {latest_alien.avg_episode_length:.0f}±{latest_alien.max_episode_length-latest_alien.min_episode_length:.0f}")
            print(f"    Path eff: {latest_alien.avg_path_efficiency:.1%} | Idle: {latest_alien.avg_idle_fraction:.1%}")
            print(f"    Unique cells: {latest_alien.avg_unique_cells:.0f}/{latest_alien.max_unique_cells}")
        
        print("\n" + "="*70)


def detect_staleness(window: EvaluationWindow) -> Dict[str, any]:
    """Detect if agent is stalling or stuck in failure modes."""
    
    issues = []
    
    # High idle fraction indicates indecision or stuck loops
    if window.avg_idle_fraction > 0.2:
        issues.append(f"HIGH_IDLE: {window.avg_idle_fraction:.1%} of steps idle (potential indecisive loops)")
    
    # Low path efficiency indicates limited exploration
    if window.avg_path_efficiency < 0.15:
        issues.append(f"LOW_EXPLORATION: only {window.avg_path_efficiency:.1%} unique cells visited")
    
    # Short episodes despite not winning could indicate giving up
    if window.avg_episode_length < 100 and window.get_win_rate() < 0.3:
        issues.append(f"SHORT_EPISODES: {window.avg_episode_length:.0f} avg length with <30% win rate (giving up?)")
    
    # Very long episodes with low idle could indicate orbiting
    if window.avg_episode_length > 450 and window.avg_idle_fraction < 0.05:
        issues.append(f"POSSIBLE_ORBITING: {window.avg_episode_length:.0f} avg length with low idle (looping path?)")
    
    return {
        "has_issues": len(issues) > 0,
        "issue_list": issues,
        "staleness_score": window.avg_idle_fraction + (1.0 - window.avg_path_efficiency) * 0.5,
    }
