# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Modular Composability System — nx.compose()
#
# ==========================================================

"""
Modular Composability System for Neura-X.

Allows users to build custom AI by combining a base brain with
capability modules freely.
"""

import numpy as np
from typing import List, Optional, Any, Dict
from neura_x.module import Module


class ComposedModel(Module):
    """
    A model composed of a base brain and multiple capability modules.

    The Liquid Router dynamically routes inputs to the appropriate modules.
    """

    def __init__(self, brain: Module, modules: List[Module]):
        super().__init__()
        self.brain = brain
        self.register_module("brain", brain)

        self.capability_modules: Dict[str, Module] = {}
        for i, mod in enumerate(modules):
            name = mod.__class__.__name__.lower() + f"_{i}"
            self.capability_modules[name] = mod
            self.register_module(name, mod)

        self._module_registry: Dict[str, Module] = {
            mod.__class__.__name__.lower(): mod for mod in modules
        }

    def forward(self, x: np.ndarray, target_module: Optional[str] = None) -> np.ndarray:
        """
        Forward pass through the composed model.

        If target_module is specified, routes to that module.
        Otherwise, routes through the base brain.
        """
        if target_module and target_module in self._module_registry:
            return self._module_registry[target_module](x)
        return self.brain(x)

    def attach(self, module: Module):
        """Hot-swap: attach a new module at runtime."""
        name = module.__class__.__name__.lower()
        self.capability_modules[name] = module
        self._module_registry[name] = module
        self.register_module(name, module)

    def detach(self, module_name: str):
        """Hot-swap: remove a module at runtime."""
        if module_name in self.capability_modules:
            del self.capability_modules[module_name]
        if module_name in self._module_registry:
            del self._module_registry[module_name]

    def list_modules(self) -> List[str]:
        """List all attached capability modules."""
        return list(self.capability_modules.keys())

    def get_module(self, name: str) -> Optional[Module]:
        """Get a specific module by name."""
        return self._module_registry.get(name.lower())

    def infer(self, prompt: str, use_tools: bool = False) -> str:
        """Run inference through the composed model."""
        return f"[Neura-X ComposedModel] Response to: {prompt}"

    def __repr__(self) -> str:
        modules_str = ", ".join(self.capability_modules.keys())
        return (
            f"ComposedModel(\n"
            f"  brain={self.brain.__class__.__name__},\n"
            f"  modules=[{modules_str}]\n"
            f")"
        )


def compose(brain: Module, modules: List[Module]) -> ComposedModel:
    """
    Compose a custom AI model from a base brain and capability modules.

    Args:
        brain: The base neural network (LLM, Vision, etc.)
        modules: List of capability modules to attach

    Returns:
        ComposedModel with all modules attached

    Example:
        model = nx.compose(
            brain=nx.LLM(conceptual_params=8_000_000_000),
            modules=[
                nx.module.Prediction(task="time_series"),
                nx.module.ImageGen(resolution=512),
                nx.module.Skills(["swahili", "medical"]),
            ]
        )
    """
    return ComposedModel(brain=brain, modules=modules)