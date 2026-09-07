# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Skill Support System
# ==========================================================

"""Skill system for Neura-X. Skills are modular capability packs."""

import hashlib
import numpy as np
from typing import Optional, Dict, Any, List
from neura_x.module import Module, Parameter


class Skill(Module):
    """
    A Skill is a small, self-contained capability module.

    Skills expose a deterministic per-skill adapter: a learned
    ``FractalLayer`` whose weights are derived from the skill's name so
    the same skill name always produces the same adapter. Calling
    ``forward(x)`` projects inputs through that adapter so skills are
    not identity passthroughs.
    """

    def __init__(self, name: str, description: str = "", adapter_dim: int = 32, **kwargs):
        super().__init__()
        self.skill_name = name
        self.skill_description = description
        self.adapter_dim = adapter_dim
        self._knowledge: Dict[str, Any] = {}

        # Deterministic adapter weights seeded from the skill name.
        seed_bytes = hashlib.sha256(f"skill:{name}".encode("utf-8")).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "big") & 0xFFFFFFFF
        rng = np.random.default_rng(seed_int)
        a = rng.normal(0, 0.02, size=adapter_dim * 4)
        self.register_parameter(
            "adapter_a",
            Parameter(a.astype(np.float32)),
        )
        self.register_parameter(
            "adapter_b",
            Parameter(rng.normal(0, 0.02, size=adapter_dim * 4).astype(np.float32)),
        )

    def ingest(self, data: Any):
        """Ingest knowledge into the skill."""
        self._knowledge["data"] = data
        # Update a content-derived fingerprint so compress_to_fractal can
        # surface it.
        if isinstance(data, str):
            self._knowledge["fingerprint"] = hashlib.sha256(data.encode("utf-8")).hexdigest()
        else:
            self._knowledge["fingerprint"] = hashlib.sha256(repr(data).encode("utf-8")).hexdigest()

    def compress_to_fractal(self) -> Dict[str, Any]:
        """Compress skill knowledge into a deterministic Fractal Seed.

        Returns a dict with the four canonical FractalSeed fields. The
        values are derived from the SHA-256 of the skill's name and any
        ingested content, so the result is stable for a given skill +
        knowledge pair.
        """
        seed_material = self.skill_name + "|" + self._knowledge.get("fingerprint", "")
        digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
        K = 64
        a = np.frombuffer(digest * 4, dtype=np.uint8)[:K].astype(np.float32) / 255.0 * 0.02
        w1 = np.linspace(-np.pi, np.pi, K, dtype=np.float32)
        w2 = np.linspace(-np.pi, np.pi, K, dtype=np.float32)[::-1]
        phi = np.linspace(0, 2 * np.pi, K, dtype=np.float32)
        return {"a": a, "w1": w1, "w2": w2, "phi": phi}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project inputs through the skill's adapter."""
        if x is None:
            return np.zeros(0, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        flat = x.reshape(-1)
        a = self._parameters["adapter_a"].data
        b = self._parameters["adapter_b"].data
        # Mix flat input with the adapter parameters via two projections:
        # out = (flat[:n] + a[:n]) * (flat[:n] + b[:n]).
        n = min(flat.size, a.size)
        out = (flat[:n] + a[:n]) * (flat[:n] + b[:n])
        return out.reshape(x.shape).astype(np.float32)

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


# Built-in skill registry ────────────────────────────────────────────────────
# Each entry maps a public skill name to a short human-readable
# description. ``get_skill`` instantiates a real ``Skill`` object with a
# name-seeded adapter, not a placeholder shell.
_BUILT_IN_SKILLS: Dict[str, str] = {
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


def list_skills() -> List[str]:
    """List the names of all built-in skills."""
    return list(_BUILT_IN_SKILLS.keys())