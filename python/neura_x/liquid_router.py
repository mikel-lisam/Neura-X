# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Liquid Router — Dynamic Sparsity Engine
# ==========================================================

"""
Liquid Router for Neura-X.
Dynamically determines which neurons are awake and which are asleep.
"""

import numpy as np
from typing import Optional
from neura_x.module import Module, Parameter


class RoutingStats:
    """Statistics for routing operations."""
    def __init__(self):
        self.total_routed = 0
        self.total_awake = 0
        self.total_asleep = 0
        self.sparsity_ratio = 0.0
        self.avg_awake_fraction = 0.0


class LiquidRouter(Module):
    """Dynamic sparsity router for Neura-X."""

    def __init__(
        self,
        num_neurons: int,
        awake_threshold: float = 0.05,
        temperature: float = 1.0,
        temperature_decay: float = 0.999,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.awake_threshold = awake_threshold
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self._step = 0

        self.register_parameter(
            "gate_logits",
            Parameter(np.zeros(num_neurons, dtype=np.float32))
        )

        self._total_routed = 0
        self._total_awake = 0

    def _gumbel_softmax(self, logits: np.ndarray, hard: bool = False) -> np.ndarray:
        if hard or self.temperature < 0.01:
            return (logits > 0).astype(np.float32)

        gumbel_noise = -np.log(-np.log(np.random.rand(*logits.shape) + 1e-10) + 1e-10)
        noisy_logits = (logits + gumbel_noise) / self.temperature

        exp_logits = np.exp(noisy_logits - np.max(noisy_logits))
        soft_gates = exp_logits / exp_logits.sum()
        return soft_gates

    def route(self, x: np.ndarray, hard: bool = False) -> np.ndarray:
        logits = self._parameters["gate_logits"].data
        gates = self._gumbel_softmax(logits, hard=hard or not self.training)

        num_awake = max(1, int(self.num_neurons * self.awake_threshold))
        top_indices = np.argsort(gates)[-num_awake:]

        mask = np.zeros(self.num_neurons, dtype=np.float32)
        mask[top_indices] = 1.0

        self._total_routed += self.num_neurons
        self._total_awake += int(mask.sum())

        return mask

    def forward(self, x: np.ndarray) -> np.ndarray:
        mask = self.route(x, hard=not self.training)
        return x * mask

    def step(self):
        self._step += 1
        self.temperature *= self.temperature_decay
        self.temperature = max(self.temperature, 0.01)

    @property
    def sparsity(self) -> float:
        if self._total_routed == 0:
            return 0.0
        return 1.0 - (self._total_awake / self._total_routed)

    @property
    def effective_awake(self) -> int:
        return max(1, int(self.num_neurons * self.awake_threshold))

    def get_stats(self) -> RoutingStats:
        """Get routing statistics."""
        stats = RoutingStats()
        stats.total_routed = self._total_routed
        stats.total_awake = self._total_awake
        stats.total_asleep = self._total_routed - self._total_awake
        if self._total_routed > 0:
            stats.sparsity_ratio = 1.0 - (self._total_awake / self._total_routed)
            stats.avg_awake_fraction = self._total_awake / self._total_routed
        return stats

    def memory_savings(self) -> dict:
        total_params = self.num_neurons
        awake_params = self.effective_awake
        return {
            "total_neurons": total_params,
            "awake_neurons": awake_params,
            "asleep_neurons": total_params - awake_params,
            "sparsity_percent": (1 - awake_params / total_params) * 100,
            "memory_reduction_factor": total_params / max(awake_params, 1),
        }

    def reset_stats(self):
        self._total_routed = 0
        self._total_awake = 0

    def __repr__(self) -> str:
        savings = self.memory_savings()
        return (
            f"LiquidRouter("
            f"neurons={self.num_neurons}, "
            f"awake={self.effective_awake}, "
            f"sparsity={savings['sparsity_percent']:.1f}%, "
            f"temp={self.temperature:.3f})"
        )