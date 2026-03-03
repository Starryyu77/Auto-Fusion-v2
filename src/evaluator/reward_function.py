"""
Reward Function

Multi-objective reward calculation balancing accuracy, efficiency, and constraints.
"""

from typing import Dict, Any
import numpy as np


class RewardFunction:
    """
    Reward function for architecture evaluation.

    Combines:
    - Accuracy reward
    - Efficiency reward (FLOPs and Params)
    - Constraint penalty (for exceeding limits)
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        penalty_type: str = "exponential"  # "linear" or "exponential"
    ):
        self.weights = weights or {
            "accuracy": 1.0,
            "efficiency": 0.5,
            "constraint": 2.0
        }
        self.penalty_type = penalty_type

    def calculate(
        self,
        accuracy: float,
        flops: int,
        params: int,
        constraints: Dict[str, Any]
    ) -> float:
        """
        Calculate the multi-objective reward.

        Args:
            accuracy: Task accuracy (0-1)
            flops: Number of FLOPs
            params: Number of parameters
            constraints: Dictionary with max_flops, max_params, etc.

        Returns:
            Scalar reward value
        """
        # Accuracy reward (primary objective)
        acc_reward = accuracy

        # Efficiency reward (encourage efficiency)
        target_flops = constraints.get("max_flops", 10e6)
        target_params = constraints.get("max_params", 50e6)

        flops_ratio = flops / target_flops if target_flops > 0 else 0
        params_ratio = params / target_params if target_params > 0 else 0

        # Reward for being under budget (1 at 0%, 0 at 100%)
        flops_efficiency = max(0, 1 - flops_ratio)
        params_efficiency = max(0, 1 - params_ratio)
        efficiency_reward = (flops_efficiency + params_efficiency) / 2

        # Constraint penalty (punish violations)
        violation_penalty = 0.0

        if flops > target_flops:
            violation = (flops - target_flops) / target_flops
            if self.penalty_type == "exponential":
                violation_penalty += self._exponential_penalty(violation)
            else:
                violation_penalty += violation

        if params > target_params:
            violation = (params - target_params) / target_params
            if self.penalty_type == "exponential":
                violation_penalty += self._exponential_penalty(violation)
            else:
                violation_penalty += violation

        # Combined reward
        reward = (
            self.weights["accuracy"] * acc_reward +
            self.weights["efficiency"] * efficiency_reward -
            self.weights["constraint"] * violation_penalty
        )

        return float(reward)

    def _exponential_penalty(self, violation: float) -> float:
        """Calculate exponential penalty for constraint violation."""
        # Exponential growth: penalty increases rapidly with violation
        return np.exp(violation) - 1

    def get_reward_breakdown(
        self,
        accuracy: float,
        flops: int,
        params: int,
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get detailed breakdown of reward components."""
        target_flops = constraints.get("max_flops", 10e6)
        target_params = constraints.get("max_params", 50e6)

        return {
            "accuracy_component": self.weights["accuracy"] * accuracy,
            "efficiency_component": self.weights["efficiency"] * max(0, 1 - flops / target_flops),
            "flops_violation": max(0, flops - target_flops) / target_flops if target_flops > 0 else 0,
            "params_violation": max(0, params - target_params) / target_params if target_params > 0 else 0,
            "final_reward": self.calculate(accuracy, flops, params, constraints)
        }
