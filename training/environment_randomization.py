"""Environment Randomization (ER) utilities for AET training.

Implements difficulty adjustment mechanisms to rebalance training when one side becomes too strong.
Includes spawn point randomization, vision/movement adjustments, and obstacle placement.
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class BalanceMode(Enum):
    """Environment randomization modes."""
    NORMAL = "normal"           # No adjustment
    HELP_HUMAN = "help_human"   # Weaken alien, strengthen human escapes
    HELP_ALIEN = "help_alien"   # Strengthen alien, weaken human escapes


@dataclass
class SpawnAdjustment:
    """Configuration for spawn point adjustments."""
    human_spawn_offset: Tuple[int, int] = (0, 0)  # Offset from original
    alien_spawn_offset: Tuple[int, int] = (0, 0)
    human_to_exit_dist: float = 1.0               # Multiplier for distance
    alien_to_human_dist: float = 1.0              # Multiplier for distance


class EnvironmentRandomizerAdvanced:
    """Advanced environment randomization with concrete difficulty adjustments."""
    
    def __init__(self, enable_threshold: float = 0.3, persistence_window: int = 20):
        """Initialize randomizer.
        
        Args:
            enable_threshold: Trigger ER if imbalance > this (e.g., 0.3 = 30%)
            persistence_window: Number of episodes imbalance must persist
        """
        self.enable_threshold = enable_threshold
        self.persistence_window = persistence_window
        self.imbalance_history = []
        self.current_mode = BalanceMode.NORMAL
        self.adjustment_strength = 0.0  # 0.0-1.0, how much to adjust
    
    def update_imbalance(self, imbalance: float) -> BalanceMode:
        """Update imbalance tracking and decide ER mode."""
        self.imbalance_history.append(imbalance)
        
        if len(self.imbalance_history) > self.persistence_window:
            self.imbalance_history.pop(0)
        
        # Check persistence
        if len(self.imbalance_history) >= self.persistence_window:
            mean_imbalance = np.mean(self.imbalance_history)
            
            if mean_imbalance > self.enable_threshold:
                self.adjustment_strength = min(1.0, mean_imbalance * 2.0)
            else:
                self.adjustment_strength = 0.0
                self.current_mode = BalanceMode.NORMAL
        
        return self.current_mode
    
    def get_spawn_adjustment(self, mode: BalanceMode, alien_wr: float, 
                            human_wr: float, strength: float = 0.5) -> SpawnAdjustment:
        """Compute spawn adjustments based on mode and win rate difference.
        
        Args:
            mode: Current ER mode
            alien_wr: Alien win rate
            human_wr: Human win rate
            strength: Adjustment strength (0.0-1.0)
        
        Returns:
            SpawnAdjustment with spawn offsets and distance multipliers
        """
        if mode == BalanceMode.NORMAL:
            return SpawnAdjustment()
        
        # Compute adjustment magnitude
        wr_diff = alien_wr - human_wr  # Positive means alien winning
        magnitude = abs(wr_diff) * strength
        
        if mode == BalanceMode.HELP_HUMAN:
            # Make it easier for human to escape
            # - Spawn human farther from alien (reduce pursuit difficulty)
            # - Spawn human closer to exit (reduce escape distance)
            # - Reduce visibility
            return SpawnAdjustment(
                human_spawn_offset=(int(5 * magnitude), int(5 * magnitude)),
                alien_spawn_offset=(-int(3 * magnitude), -int(3 * magnitude)),
                human_to_exit_dist=max(0.3, 1.0 - 0.5 * magnitude),  # Min 30% distance
                alien_to_human_dist=1.0 + 0.3 * magnitude,            # Further away
            )
        else:  # HELP_ALIEN
            # Make it easier for alien to catch
            # - Spawn alien closer to human (increase pursuit advantage)
            # - Spawn human farther from exit (increase escape difficulty)
            # - Increase visibility
            return SpawnAdjustment(
                human_spawn_offset=(-int(5 * magnitude), -int(5 * magnitude)),
                alien_spawn_offset=(int(3 * magnitude), int(3 * magnitude)),
                human_to_exit_dist=min(1.5, 1.0 + 0.5 * magnitude),   # Max 150% distance
                alien_to_human_dist=max(0.7, 1.0 - 0.3 * magnitude),  # Closer together
            )
    
    def get_vision_adjustment(self, mode: BalanceMode, strength: float = 0.5) -> Dict[str, float]:
        """Get vision range adjustments for each agent.
        
        Args:
            mode: Current ER mode
            strength: Adjustment strength (0.0-1.0)
        
        Returns:
            Dict with "alien_vision" and "human_vision" multipliers (1.0 = no change)
        """
        if mode == BalanceMode.NORMAL:
            return {"alien_vision": 1.0, "human_vision": 1.0}
        
        if mode == BalanceMode.HELP_HUMAN:
            # Reduce alien vision, keep human normal
            return {
                "alien_vision": max(0.6, 1.0 - 0.3 * strength),
                "human_vision": 1.0,
            }
        else:  # HELP_ALIEN
            # Increase alien vision, keep human normal
            return {
                "alien_vision": 1.0 + 0.4 * strength,
                "human_vision": 1.0,
            }
    
    def get_movement_adjustment(self, mode: BalanceMode, strength: float = 0.5) -> Dict[str, float]:
        """Get movement speed adjustments.
        
        Args:
            mode: Current ER mode
            strength: Adjustment strength (0.0-1.0)
        
        Returns:
            Dict with "alien_speed" and "human_speed" multipliers (1.0 = no change)
        """
        if mode == BalanceMode.NORMAL:
            return {"alien_speed": 1.0, "human_speed": 1.0}
        
        if mode == BalanceMode.HELP_HUMAN:
            # Slow alien, speed up human slightly
            return {
                "alien_speed": max(0.8, 1.0 - 0.2 * strength),
                "human_speed": 1.0 + 0.1 * strength,
            }
        else:  # HELP_ALIEN
            # Speed up alien, slow human slightly
            return {
                "alien_speed": 1.0 + 0.2 * strength,
                "human_speed": max(0.9, 1.0 - 0.1 * strength),
            }
    
    def should_apply_er(self) -> bool:
        """Check if ER should be applied this episode."""
        if len(self.imbalance_history) < self.persistence_window:
            return False
        
        mean_imbalance = np.mean(self.imbalance_history)
        return mean_imbalance > self.enable_threshold


class MapRandomizer:
    """Randomizes map difficulty by modifying tile properties."""
    
    def __init__(self):
        self.original_map = None
        self.current_map = None
    
    def randomize_for_mode(self, map_array: np.ndarray, mode: BalanceMode, 
                          strength: float = 0.5) -> np.ndarray:
        """Randomize map difficulty for ER mode.
        
        Tile types:
        - 0: EMPTY
        - 1: WALL
        - 2: HIDE (safe for human)
        
        Args:
            map_array: Original map array (H, W)
            mode: ER mode
            strength: How much to adjust (0.0-1.0)
        
        Returns:
            Modified map array
        """
        randomized = map_array.copy()
        
        if mode == BalanceMode.NORMAL:
            return randomized
        
        if mode == BalanceMode.HELP_HUMAN:
            # Increase hiding spots, reduce alien pressure
            return self._add_hiding_spots(randomized, strength)
        else:  # HELP_ALIEN
            # Remove hiding spots, increase alien pressure
            return self._reduce_hiding_spots(randomized, strength)
    
    def _add_hiding_spots(self, map_array: np.ndarray, strength: float) -> np.ndarray:
        """Add safe hiding tiles to help human.
        
        Strategy: Convert some empty tiles to HIDE tiles (value 2).
        """
        randomized = map_array.copy()
        empty_mask = randomized == 0  # EMPTY tiles
        empty_indices = np.argwhere(empty_mask)
        
        if len(empty_indices) == 0:
            return randomized
        
        # Select N random empty tiles to convert to hiding spots
        num_to_convert = max(1, int(len(empty_indices) * strength * 0.1))
        selected_indices = empty_indices[
            np.random.choice(len(empty_indices), min(num_to_convert, len(empty_indices)), replace=False)
        ]
        
        for idx in selected_indices:
            randomized[idx[0], idx[1]] = 2  # Convert to HIDE
        
        return randomized
    
    def _reduce_hiding_spots(self, map_array: np.ndarray, strength: float) -> np.ndarray:
        """Remove safe hiding tiles to help alien.
        
        Strategy: Convert some HIDE tiles back to EMPTY tiles (value 0).
        """
        randomized = map_array.copy()
        hide_mask = randomized == 2  # HIDE tiles
        hide_indices = np.argwhere(hide_mask)
        
        if len(hide_indices) == 0:
            return randomized
        
        # Select N random hiding tiles to convert back to empty
        num_to_convert = max(1, int(len(hide_indices) * strength * 0.2))
        selected_indices = hide_indices[
            np.random.choice(len(hide_indices), min(num_to_convert, len(hide_indices)), replace=False)
        ]
        
        for idx in selected_indices:
            randomized[idx[0], idx[1]] = 0  # Convert back to EMPTY
        
        return randomized


def create_balanced_er_config(alien_wr: float, human_wr: float) -> Dict:
    """Create a complete ER configuration based on current win rates.
    
    Returns:
        Dict with er_enabled, mode, spawn_adjustment, vision, movement, map_randomization
    """
    imbalance = abs(alien_wr - human_wr)
    
    randomizer = EnvironmentRandomizerAdvanced()
    randomizer.adjustment_strength = min(1.0, imbalance * 2.0)
    
    if imbalance < 0.15:
        mode = BalanceMode.NORMAL
        strength = 0.0
    elif alien_wr > human_wr:
        mode = BalanceMode.HELP_HUMAN
        strength = randomizer.adjustment_strength
    else:
        mode = BalanceMode.HELP_ALIEN
        strength = randomizer.adjustment_strength
    
    return {
        "er_enabled": imbalance > 0.3,
        "mode": mode.value,
        "strength": strength,
        "spawn_adjustment": randomizer.get_spawn_adjustment(mode, alien_wr, human_wr, strength),
        "vision_adjustment": randomizer.get_vision_adjustment(mode, strength),
        "movement_adjustment": randomizer.get_movement_adjustment(mode, strength),
    }
