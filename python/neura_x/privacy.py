# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Differential Privacy Engine
# ==========================================================

"""Differential privacy for Neura-X training."""

import numpy as np


class PrivacyEngine:
    """
    Differential Privacy engine.

    Adds calibrated noise to gradients to prevent memorization
    of individual training samples.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, max_grad_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

    def add_noise(self, gradient: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to gradient for differential privacy."""
        # Clip gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.max_grad_norm:
            gradient = gradient * (self.max_grad_norm / grad_norm)

        # Add noise
        noise_scale = self.max_grad_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, noise_scale, size=gradient.shape)

        return gradient + noise

    def __repr__(self) -> str:
        return f"PrivacyEngine(ε={self.epsilon}, δ={self.delta})"