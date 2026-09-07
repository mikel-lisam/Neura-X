# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Safety & Content Filtering
# ==========================================================

"""Safety filtering for Neura-X.

Provides a dependency-free, deterministic content filter based on a curated
blocklist of high-risk categories (CSAM references, sexual content involving
minors, graphic violence, instructions for serious wrongdoing, and personal
data). The filter is intentionally conservative: it errs on the side of
flagging ambiguous content so downstream callers can decide.

This is a CPU-only, deterministic filter — not a learned classifier. It is
intended to be a reliable, reproducible safety layer that does not require
external models or network access.
"""

import re
from typing import List, Tuple


class SafetyFilter:
    """Content safety filter for training data and model outputs.

    Args:
        strictness: float in [0, 1]. 0 = permissive, 1 = maximum strictness.
            Higher values flag ambiguous patterns more readily.
    """

    # High-risk categories with regex patterns. All patterns are case-
    # insensitive and designed to match natural-language phrasings.
    _BLOCKLIST: List[Tuple[str, re.Pattern]] = [
        (
            "csam",
            re.compile(
                r"\b(child|children|kid|minor|underage|juvenile|infant|baby|toddler)"
                r"\b.{0,80}\b(sex|sexual|rape|abuse|exploit|porn|nude|naked)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "graphic_violence",
            re.compile(
                r"\b(how to|instructions? for|guide to|tutorial)\b.{0,80}\b"
                r"(kill|murder|assassinate|torture|massacre|bomb)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "weapons",
            re.compile(
                r"\b(synthesis|synthesise|synthesize|build|make|construct|manufacture)\b"
                r".{0,40}\b(sarin|vx|anthrax|ricin|bioweapon|nerve agent|dirty bomb)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "pii",
            re.compile(
                r"\b(ssn|social security|credit card|passport number|"
                r"drivers license|api[_\s-]?key|password|secret[_\s-]?key)\b"
                r".{0,30}[:=].{0,40}\b[A-Za-z0-9]{6,}\b",
                re.IGNORECASE,
            ),
        ),
    ]

    def __init__(self, strictness: float = 0.5):
        self.strictness = float(strictness)

    def _score(self, text: str) -> Tuple[float, List[str]]:
        """Return (risk_score, triggered_categories)."""
        if not text:
            return 0.0, []
        triggered: List[str] = []
        score = 0.0
        # Normalise whitespace for matching.
        normalised = re.sub(r"\s+", " ", text).strip()
        for name, pattern in self._BLOCKLIST:
            if pattern.search(normalised):
                triggered.append(name)
                # CSAM is non-negotiable; everything else scales with strictness.
                if name == "csam":
                    score = max(score, 1.0)
                else:
                    score = max(score, 0.5 + 0.5 * self.strictness)
        return score, triggered

    def filter(self, text: str) -> str:
        """Return the input text, or a safe placeholder if blocked.

        Always returns a string; callers can check ``is_safe`` for a
        boolean verdict and ``categories`` for the triggered categories.
        """
        score, triggered = self._score(text)
        if score >= 0.99:
            return "[REDACTED BY NEURA-X SAFETY]"
        return text

    def is_safe(self, text: str) -> bool:
        """Return True if the text passes the filter at the current strictness."""
        score, _ = self._score(text)
        # Block anything with score above the strictness threshold.
        return score <= self.strictness

    @property
    def categories(self) -> List[str]:
        """Categories checked by this filter."""
        return [name for name, _ in self._BLOCKLIST]