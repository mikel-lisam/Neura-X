# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Standard Optimizers & Learning Rate Schedulers
#
# ==========================================================

"""
Standard optimizers and learning rate schedulers for Neura-X.
These use the same mathematics as PyTorch/TensorFlow.
"""

import numpy as np
from typing import List, Optional
from neura_x.module import Parameter


class AdamW:
    """
    AdamW optimizer (standard implementation).

    Uses the same mathematics as PyTorch's AdamW.
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # Optimizer states
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.astype(np.float32)

            # Decoupled weight decay
            if self.weight_decay > 0:
                param.data -= self.lr * self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)

            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                param.data -= self.lr * self.velocities[i]
            else:
                param.data -= self.lr * grad


class Adam:
    """Adam optimizer (without decoupled weight decay)."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        self.step_count += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop:
    """RMSprop optimizer."""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            grad = param.grad
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
            param.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)


class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler."""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        progress = self.step_count / self.T_max
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * cosine_decay


class LinearLR:
    """Linear learning rate scheduler."""

    def __init__(self, optimizer, start_factor: float = 1.0, end_factor: float = 0.0, total_steps: int = 1000):
        self.optimizer = optimizer
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_steps = total_steps
        self.base_lr = optimizer.lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        progress = min(self.step_count / self.total_steps, 1.0)
        factor = self.start_factor + (self.end_factor - self.start_factor) * progress
        self.optimizer.lr = self.base_lr * factor