# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Mixture of Experts
# ==========================================================

"""Mixture of Experts via Liquid Router."""

import numpy as np
from typing import Optional
from neura_x.module import Module, FractalLayer
from neura_x.liquid_router import LiquidRouter


class MoE(Module):
    """
    Mixture of Experts model.

    The Liquid Router serves as the native MoE routing mechanism.
    Each expert is a FractalLayer that can be paged in/out of RAM.
    """

    def __init__(
        self,
        num_experts: int = 1000,
        experts_per_token: int = 3,
        conceptual_params_per_expert: int = 500_000_000,
        expert_dim: int = 4096,
        fractal_compression: bool = True,
        dynamic_expert_growth: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.conceptual_params_per_expert = conceptual_params_per_expert
        self.dynamic_expert_growth = dynamic_expert_growth
        self.expert_dim = expert_dim  # recorded so experts match the input.

        # Router
        self.register_module("router", LiquidRouter(
            num_neurons=num_experts,
            awake_threshold=experts_per_token / num_experts,
        ))

        # Experts (lazy-loaded Fractal Layers)
        self.experts = {}
        self._fractal_compression = fractal_compression

    def _get_expert(self, expert_id: int, in_dim: Optional[int] = None) -> FractalLayer:
        """Lazy-load an expert.

        The first expert requested dictates the input/output feature
        dimensions; subsequent experts match so callers can mix
        activations across them without shape mismatches.
        """
        if in_dim is None:
            in_dim = self.expert_dim
        key = (expert_id, in_dim)
        if key not in self.experts:
            self.experts[key] = FractalLayer(
                in_features=in_dim,
                out_features=in_dim,
                conceptual_params=self.conceptual_params_per_expert,
            )
        return self.experts[key]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Route input to active experts."""
        # Flatten all but the last axis so experts see a 2-D input.
        if x.ndim > 2:
            batch = x.reshape(x.shape[0], -1)
        elif x.ndim == 1:
            batch = x.reshape(1, -1)
        else:
            batch = x

        in_dim = batch.shape[-1]
        mask = self.router.route(batch[0]) if batch.shape[0] > 0 else np.zeros(self.num_experts)
        active_expert_ids = np.where(mask > 0)[0][:self.experts_per_token]

        output = np.zeros_like(batch)
        for expert_id in active_expert_ids:
            expert = self._get_expert(int(expert_id), in_dim)
            output += expert(batch)

        denom = max(len(active_expert_ids), 1)
        out = output / denom
        return out.reshape(x.shape)

    @property
    def total_conceptual_params(self) -> int:
        return self.num_experts * self.conceptual_params_per_expert

    @property
    def active_params_per_token(self) -> int:
        return self.experts_per_token * self.conceptual_params_per_expert