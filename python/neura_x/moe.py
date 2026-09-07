# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Mixture of Experts
# ==========================================================

"""Mixture of Experts via Liquid Router."""

import numpy as np
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

        # Router
        self.register_module("router", LiquidRouter(
            num_neurons=num_experts,
            awake_threshold=experts_per_token / num_experts,
        ))

        # Experts (lazy-loaded Fractal Layers)
        self.experts = {}
        self._fractal_compression = fractal_compression

    def _get_expert(self, expert_id: int) -> FractalLayer:
        """Lazy-load an expert."""
        if expert_id not in self.experts:
            self.experts[expert_id] = FractalLayer(
                in_features=4096,
                out_features=4096,
                conceptual_params=self.conceptual_params_per_expert,
            )
        return self.experts[expert_id]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Route input to active experts."""
        mask = self.router.route(x)
        active_expert_ids = np.where(mask > 0)[0][:self.experts_per_token]

        output = np.zeros_like(x)
        for expert_id in active_expert_ids:
            expert = self._get_expert(int(expert_id))
            output += expert(x)

        return output / max(len(active_expert_ids), 1)

    @property
    def total_conceptual_params(self) -> int:
        return self.num_experts * self.conceptual_params_per_expert

    @property
    def active_params_per_token(self) -> int:
        return self.experts_per_token * self.conceptual_params_per_expert