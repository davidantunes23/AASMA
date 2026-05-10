"""
environment_randomization.py

Environment Randomization (ER) for AET, implemented via map_generator.

When persistent win-rate imbalance is detected, this module samples an ER
scenario that either:
- strengthens the weaker side via more favourable maps (alpha & distances), or
- puts the stronger side into disadvantageous situations.

This closely follows the paper's ER idea: scenario-level environment
randomization to rebalance and enrich training, without changing the game rules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from map_generator import MapGenerator


class BalanceMode(Enum):
    NORMAL = "normal"
    HELP_HUMAN = "help_human"
    HELP_ALIEN = "help_alien"


@dataclass
class ERScenario:
    """
    Encodes one ER scenario for an episode.

    alpha_shift:
        How much to shift the base alpha (negative -> more human-favoured).
    target_player_exit_scale:
        Scale factor applied to baseline dist_player_exit to derive a target.
    target_alien_player_scale:
        Scale factor applied to baseline dist_alien_player to derive a target.
    early_weaker_side_boost / strong_side_disadvantage:
        Booleans that can be used to apply transient speed/vision tweaks in
        the Game/agents (optional).
    duration_steps:
        How long transient buffs/nerfs should last at episode start.
    """
    alpha_shift: float = 0.0
    target_player_exit_scale: float = 1.0
    target_alien_player_scale: float = 1.0
    early_weaker_side_boost: bool = False
    strong_side_disadvantage: bool = False
    duration_steps: int = 0


class EnvironmentRandomizerAdvanced:
    """
    Advanced ER that uses:
    - persistent signed imbalance (alien_wr - human_wr),
    - scenario sampling (pure help, strong-side disadvantage, early boost),
    - map regeneration via MapGenerator (alpha & distances).

    Usage:
        er = EnvironmentRandomizerAdvanced()
        mode = er.update_imbalance(alien_wr, human_wr)
        if er.should_apply_er():
            scenario = er.sample_scenario()
            training_map, training_meta = er.randomize_map(
                base_map, base_metadata, scenario, seed=episode_seed
            )
            mods = er.get_episode_modifiers(scenario)
        else:
            training_map, training_meta = base_map, base_metadata
            mods = default_mods
    """

    def __init__(
        self,
        enable_threshold: float = 0.3,
        persistence_window: int = 20,
        episode_apply_prob: float = 0.95,
        seed: Optional[int] = None,
    ):
        """
        Args:
            enable_threshold:
                Minimum absolute mean imbalance to trigger ER. E.g. 0.3 means
                ER activates only if one side is ahead by ~30 percentage points.
            persistence_window:
                How many recent rounds to average over; ER only considers
                activating once this many results are available.
            episode_apply_prob:
                Probability to apply ER in an episode once a mode != NORMAL is
                active; allows occasional non-ER episodes for diversity.
        """
        self.enable_threshold = enable_threshold
        self.persistence_window = persistence_window
        self.episode_apply_prob = episode_apply_prob

        # Signed imbalance history: alien_wr - human_wr
        self.imbalance_history: list[float] = []
        self.current_mode: BalanceMode = BalanceMode.NORMAL
        self.adjustment_strength: float = 0.0  # in [0, 1]
        self.rng = np.random.default_rng(seed)

    # ── Imbalance tracking ─────────────────────────────────────────────────────
    def update_imbalance(self, alien_wr: float, human_wr: float) -> BalanceMode:
        """
        Update internal imbalance statistics and set current_mode/strength.

        Returns:
            BalanceMode for the *next* episodes.
        """
        signed_imbalance = float(alien_wr - human_wr)
        self.imbalance_history.append(signed_imbalance)
        if len(self.imbalance_history) > self.persistence_window:
            self.imbalance_history.pop(0)

        # Not enough history yet → no ER.
        if len(self.imbalance_history) < self.persistence_window:
            self.current_mode = BalanceMode.NORMAL
            self.adjustment_strength = 0.0
            return self.current_mode

        mean_imbalance = float(np.mean(self.imbalance_history))
        abs_mean = abs(mean_imbalance)

        if abs_mean <= self.enable_threshold:
            # Balanced enough; disable ER.
            self.current_mode = BalanceMode.NORMAL
            self.adjustment_strength = 0.0
        else:
            # Decide who to help: if alien_wr > human_wr => mean_imbalance > 0
            self.current_mode = (
                BalanceMode.HELP_HUMAN if mean_imbalance > 0 else BalanceMode.HELP_ALIEN
            )
            # Map imbalance beyond threshold to [0,1] strength.
            self.adjustment_strength = float(
                np.clip(
                    (abs_mean - self.enable_threshold)
                    / max(1e-6, 1.0 - self.enable_threshold),
                    0.0,
                    1.0,
                )
            )

        return self.current_mode

    def should_apply_er(self) -> bool:
        """Check if ER should be applied this episode."""
        if self.current_mode == BalanceMode.NORMAL:
            return False
        return bool(self.rng.random() < self.episode_apply_prob)

    # ── Scenario sampling ──────────────────────────────────────────────────────
    def sample_scenario(self) -> ERScenario:
        """
        Sample an ERScenario given the current mode and adjustment_strength.

        Rough mix:
            ~45%: pure weaker-side strengthening (alpha only),
            ~35%: strong-side disadvantage (alpha + distance bias),
            remainder: early weaker-side boost.
        """
        s = self.adjustment_strength
        mode = self.current_mode

        if mode == BalanceMode.NORMAL:
            return ERScenario()

        u = float(self.rng.random())

        # 1) Pure weaker-side strengthening via alpha
        if u < 0.45:
            if mode == BalanceMode.HELP_HUMAN:
                # Make map more human-favoured: shift alpha negative.
                return ERScenario(
                    alpha_shift=-(0.25 + 0.55 * s),
                )
            else:  # HELP_ALIEN
                return ERScenario(
                    alpha_shift=(0.25 + 0.55 * s),
                )

        # 2) Strong-side disadvantage: alpha + distance bias
        if u < 0.80:
            if mode == BalanceMode.HELP_HUMAN:
                # Slight negative alpha shift, shorter human→exit, longer alien→human.
                return ERScenario(
                    alpha_shift=-(0.15 + 0.35 * s),
                    target_player_exit_scale=0.8 - 0.2 * s,
                    target_alien_player_scale=1.15 + 0.25 * s,
                    strong_side_disadvantage=True,
                )
            else:
                # Slight positive alpha shift, longer human→exit, shorter alien→human.
                return ERScenario(
                    alpha_shift=(0.15 + 0.35 * s),
                    target_player_exit_scale=1.15 + 0.25 * s,
                    target_alien_player_scale=0.8 - 0.2 * s,
                    strong_side_disadvantage=True,
                )

        # 3) Early-game weaker-side boost
        if mode == BalanceMode.HELP_HUMAN:
            return ERScenario(
                alpha_shift=-(0.10 + 0.25 * s),
                early_weaker_side_boost=True,
                duration_steps=int(20 + 40 * s),
            )
        else:
            return ERScenario(
                alpha_shift=(0.10 + 0.25 * s),
                early_weaker_side_boost=True,
                duration_steps=int(20 + 40 * s),
            )

    # ── Map randomization via MapGenerator ─────────────────────────────────────
    def randomize_map(
        self,
        base_map: np.ndarray,
        base_metadata: Dict,
        scenario: ERScenario,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Regenerate a biased map using MapGenerator.

        Args:
            base_map:      original fixed map (ignored except for width/height)
            base_metadata: original metadata dict (must contain width, height, alpha, dists)
            scenario:      ERScenario sampled for this episode
            seed:          optional base seed for regeneration

        Returns:
            (new_map, new_metadata)
        """
        alpha0 = float(base_metadata.get("alpha", 0.0))
        alpha = float(np.clip(alpha0 + scenario.alpha_shift, -1.0, 1.0))

        width = int(base_metadata["width"])
        height = int(base_metadata["height"])

        pe = base_metadata.get("dist_player_exit")
        ap = base_metadata.get("dist_alien_player")

        target_pe = (
            None
            if pe is None
            else max(1, int(round(pe * scenario.target_player_exit_scale)))
        )
        target_ap = (
            None
            if ap is None
            else max(1, int(round(ap * scenario.target_alien_player_scale)))
        )

        gen = MapGenerator(
            width=width,
            height=height,
            alpha=alpha,
            seed=seed,
            target_player_exit_dist=target_pe,
            target_alien_player_dist=target_ap,
        )

        new_map, new_meta = gen.regenerate_with_bias(
            alpha=alpha,
            seed=seed,
            target_player_exit_dist=target_pe,
            target_alien_player_dist=target_ap,
            tries=24,
        )

        # Attach ER info for logging/analysis
        new_meta["er_scenario"] = {
            "mode": self.current_mode.value,
            "strength": self.adjustment_strength,
            "alpha_shift": scenario.alpha_shift,
            "target_player_exit_scale": scenario.target_player_exit_scale,
            "target_alien_player_scale": scenario.target_alien_player_scale,
            "early_weaker_side_boost": scenario.early_weaker_side_boost,
            "strong_side_disadvantage": scenario.strong_side_disadvantage,
            "duration_steps": scenario.duration_steps,
        }

        return new_map, new_meta

    # ── Optional transient modifiers (vision/speed) ────────────────────────────
    def get_episode_modifiers(self, scenario: ERScenario) -> Dict[str, float | bool | int]:
        """
        Small, optional transient tweaks that can be applied inside Game/agents.

        They are intentionally mild; the main balancing mechanism is the map.
        """
        if self.current_mode == BalanceMode.NORMAL:
            return {
                "alien_vision": 1.0,
                "human_vision": 1.0,
                "alien_speed": 1.0,
                "human_speed": 1.0,
                "early_weaker_side_boost": False,
                "strong_side_disadvantage": False,
                "duration_steps": 0,
            }

        s = self.adjustment_strength

        if self.current_mode == BalanceMode.HELP_HUMAN:
            return {
                "alien_vision": 0.9 - 0.15 * s if scenario.strong_side_disadvantage else 1.0,
                "human_vision": 1.0,
                "alien_speed": 0.95 - 0.10 * s if scenario.strong_side_disadvantage else 1.0,
                "human_speed": 1.03 + 0.05 * s if scenario.early_weaker_side_boost else 1.0,
                "early_weaker_side_boost": scenario.early_weaker_side_boost,
                "strong_side_disadvantage": scenario.strong_side_disadvantage,
                "duration_steps": scenario.duration_steps,
            }

        # HELP_ALIEN
        return {
            "alien_vision": 1.0 + 0.10 * s if scenario.early_weaker_side_boost else 1.0,
            "human_vision": 1.0,
            "alien_speed": 1.03 + 0.07 * s if scenario.early_weaker_side_boost else 1.0,
            "human_speed": 0.97 - 0.05 * s if scenario.strong_side_disadvantage else 1.0,
            "early_weaker_side_boost": scenario.early_weaker_side_boost,
            "strong_side_disadvantage": scenario.strong_side_disadvantage,
            "duration_steps": scenario.duration_steps,
        }


def create_balanced_er_config(
    er: EnvironmentRandomizerAdvanced,
    alien_wr: float,
    human_wr: float,
    apply_now: bool = True,
) -> Dict:
    """
    Helper for logging/inspection.

    NOTE: er must be a persistent instance across rounds; do not recreate it
    each call or you will lose imbalance history.
    """
    mode = er.update_imbalance(alien_wr, human_wr)
    enabled = er.should_apply_er() if apply_now else (mode != BalanceMode.NORMAL)
    scenario = er.sample_scenario() if enabled else ERScenario()
    return {
        "er_enabled": enabled,
        "mode": mode.value,
        "strength": er.adjustment_strength,
        "scenario": scenario,
        "episode_modifiers": er.get_episode_modifiers(scenario),
    }