# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Skill Support System
# ==========================================================

"""Skill system for Neura-X. Skills are modular capability packs."""

import numpy as np
from typing import Optional, Dict, Any
from neura_x.module import Module


class Skill(Module):
    """
    A Skill is a small, self-contained capability module.

    Skills are stored as .nexs files and can be dynamically
    loaded/unloaded from any model.
    """

    def __init__(self, name: str, description: str = "", **kwargs):
        super().__init__()
        self.skill_name = name
        self.skill_description = description
        self._knowledge: Dict[str, Any] = {}

    def ingest(self, data: Any):
        """Ingest knowledge into the skill."""
        self._knowledge["data"] = data

    def compress_to_fractal(self):
        """Compress skill knowledge into a Fractal Seed."""
        pass  # Implemented in compiled backend

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def __repr__(self) -> str:
        return f"Skill(name='{self.skill_name}')"


def skill(name: str = None, description: str = None):
    """Decorator to create a Neura-X skill."""
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            Skill.__init__(self, name=name or cls.__name__, description=description or "")
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        cls._is_skill = True
        cls._skill_name = name or cls.__name__
        return cls

    return decorator


# Built-in skill registry
_BUILT_IN_SKILLS = {
    "swahili": "Swahili language understanding and generation",
    "medical": "Medical knowledge and terminology",
    "coding_python": "Python code generation and understanding",
    "legal_ke": "Kenyan legal knowledge",
    "music_theory": "Music composition and theory",
    "finance": "Financial analysis and economics",
    "creative_writing": "Creative writing and storytelling",
    "research": "Research methodology and summarization",
    "quality_check": "Quality assurance and review",
}


def get_skill(name: str) -> Optional[Skill]:
    """Get a built-in skill by name."""
    if name in _BUILT_IN_SKILLS:
        return Skill(name=name, description=_BUILT_IN_SKILLS[name])
    return None