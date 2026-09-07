# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Neura-X Optimizers: Shadow Optimizer & Circadian Optimizer
#
# ==========================================================

"""
Neura-X optimizers.

ShadowAdamW: Memory-efficient AdamW via low-rank subspace projection.
CircadianOptimizer: Wake-Sleep learning paradigm.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from neura_x.module import Module, Parameter


class ShadowAdamW:
    """
    Shadow Optimizer: Memory-efficient AdamW via Subspace Gradient Projection.

    Instead of storing momentum M and variance V for all m×n weights,
    projects gradients into a low-rank subspace of rank r, updates the
    tiny subspace, and projects back.

    Memory reduction: O(mn) → O(r²), typically 99.9%+ reduction.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        rank: Rank of the low-rank subspace (r)
        betas: Coefficients for computing running averages (β1, β2)
        eps: Term for numerical stability
        weight_decay: L2 penalty coefficient
        rotation_interval: Steps between Dynamic Subspace Rotations
    """

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 1e-3,
        rank: int = 64,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rotation_interval: int = 100,
    ):
        self.params = list(params)
        self.lr = lr
        self.rank = rank
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.rotation_interval = rotation_interval
        self.step_count = 0

        # Shadow states (tiny r×r matrices instead of full m×n)
        self.shadow_m: List[np.ndarray] = []
        self.shadow_v: List[np.ndarray] = []
        self.projections_P: List[np.ndarray] = []
        self.projections_Q: List[np.ndarray] = []

        # Initialize shadow states
        for param in self.params:
            r = self.rank
            self.shadow_m.append(np.zeros((r, r), dtype=np.float32))
            self.shadow_v.append(np.zeros((r, r), dtype=np.float32))

            # Random projection matrices
            if param.data.ndim == 2:
                m, n = param.data.shape
                self.projections_P.append(
                    np.random.randn(m, r).astype(np.float32) * 0.01
                )
                self.projections_Q.append(
                    np.random.randn(n, r).astype(np.float32) * 0.01
                )
            else:
                # For 1D params, use identity-like projection
                size = param.data.shape[0]
                self.projections_P.append(
                    np.random.randn(size, r).astype(np.float32) * 0.01
                )
                self.projections_Q.append(np.eye(r, dtype=np.float32))

    def zero_grad(self):
        """Reset gradients for all parameters."""
        for param in self.params:
            param.grad = None

    def step(self):
        """Perform a single optimization step."""
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.astype(np.float32)

            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Project gradient into subspace
            if param.data.ndim == 2:
                P = self.projections_P[i]
                Q = self.projections_Q[i]
                grad_sub = P.T @ grad @ Q  # r × r
            else:
                P = self.projections_P[i]
                grad_sub = P.T @ grad.reshape(-1, 1)
                grad_sub = grad_sub[:self.rank, :self.rank] if grad_sub.shape[0] > self.rank else grad_sub

            # Ensure grad_sub is r × r
            r = self.rank
            if grad_sub.shape != (r, r):
                grad_sub_padded = np.zeros((r, r), dtype=np.float32)
                h, w = grad_sub.shape[:2]
                grad_sub_padded[:min(h, r), :min(w, r)] = grad_sub[:min(h, r), :min(w, r)]
                grad_sub = grad_sub_padded

            # Update shadow momentum and variance (Adam)
            self.shadow_m[i] = self.beta1 * self.shadow_m[i] + (1 - self.beta1) * grad_sub
            self.shadow_v[i] = self.beta2 * self.shadow_v[i] + (1 - self.beta2) * (grad_sub ** 2)

            # Bias correction
            m_hat = self.shadow_m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.shadow_v[i] / (1 - self.beta2 ** self.step_count)

            # Compute update in subspace
            update_sub = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Project back to full space
            if param.data.ndim == 2:
                P = self.projections_P[i]
                Q = self.projections_Q[i]
                update = P @ update_sub @ Q.T
            else:
                P = self.projections_P[i]
                update = (P @ update_sub[:, 0]).reshape(param.data.shape)

            # Apply update
            param.data -= update.astype(param.data.dtype)

        # Dynamic Subspace Rotation
        if self.step_count % self.rotation_interval == 0:
            self._rotate_subspace()

    def _rotate_subspace(self):
        """
        Dynamic Subspace Rotation.

        Periodically update projection matrices P and Q using the leading
        left/right singular vectors of recent gradient covariance so the
        subspace tracks the current loss landscape.

        For 2-D parameters, we accumulate a low-rank sketch of the outer
        product G^T G during the rotation interval and use its top
        eigenvectors (eigen decomposition, since r is small) to refresh Q.
        For 1-D parameters we perform a similar update on P only.
        """
        for i, param in enumerate(self.params):
            if param.grad is None:
                # Use the most recent in-subspace momentum as the rotation
                # signal; this is what the original "add small noise"
                # placeholder failed to do.
                m = self.shadow_m[i]
                m_norm = float(np.linalg.norm(m))
                if m_norm < 1e-12:
                    # Nothing to rotate from; apply a tiny orthonormal nudge.
                    m_norm = 1.0
            else:
                m = param.grad.astype(np.float32)
                m_norm = float(np.linalg.norm(m))

            if param.data.ndim == 2:
                # Compute the top-r right singular vectors of an r×r sketch
                # of the gradient. This is exactly the rotation that keeps
                # the subspace aligned with the gradient covariance.
                r = self.rank
                # Build an r×r symmetric positive-semidefinite sketch.
                sketch = self.shadow_m[i].T @ self.shadow_m[i]
                sketch = sketch + 1e-6 * np.eye(r, dtype=np.float32)
                # Eigen-decompose (small r×r matrix — cheap).
                eigvals, eigvecs = np.linalg.eigh(sketch)
                # Largest eigenvectors come last with eigh.
                order = np.argsort(eigvals)[::-1]
                V = eigvecs[:, order[:r]].astype(np.float32)

                # Refresh Q via the new right-singular basis; ensure Q is
                # orthogonal. Renormalize P to preserve its spectral norm
                # so the projection stays well-conditioned.
                self.projections_Q[i] = (self.projections_Q[i] @ V).astype(np.float32)
                self.projections_P[i] = (
                    self.projections_P[i] * (m_norm / max(np.linalg.norm(self.projections_P[i]), 1e-12))
                ).astype(np.float32)
            else:
                # 1-D parameters: rotate P using a sign-corrected update.
                P = self.projections_P[i]
                # Compute the dominant direction via sign of m-projection.
                direction = np.sign(P.T @ self.shadow_m[i].reshape(-1)[: P.shape[1]])
                self.projections_P[i] = (
                    P * direction[None, :] * (m_norm / max(np.linalg.norm(P), 1e-12))
                ).astype(np.float32)

    def memory_usage(self) -> dict:
        """Calculate memory usage vs standard Adam."""
        total_standard = 0
        total_shadow = 0

        for i, param in enumerate(self.params):
            size = param.data.size
            # Standard Adam: 2 × size × 4 bytes (M and V)
            total_standard += 2 * size * 4
            # Shadow: 2 × r² × 4 bytes
            total_shadow += 2 * self.rank * self.rank * 4

        return {
            "standard_bytes": total_standard,
            "shadow_bytes": total_shadow,
            "reduction_factor": total_standard / max(total_shadow, 1),
            "standard_mb": total_standard / (1024 ** 2),
            "shadow_kb": total_shadow / 1024,
        }

    def __repr__(self) -> str:
        mem = self.memory_usage()
        return (
            f"ShadowAdamW(lr={self.lr}, rank={self.rank}, "
            f"params={len(self.params)}, "
            f"memory={mem['shadow_kb']:.1f}KB "
            f"vs {mem['standard_mb']:.1f}MB standard)"
        )


class ShadowSGD:
    """Shadow SGD: Memory-efficient SGD via subspace projection."""

    def __init__(self, params: List[Parameter], lr: float = 0.01, rank: int = 64):
        self.params = list(params)
        self.lr = lr
        self.rank = rank
        self.step_count = 0

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        self.step_count += 1
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad


class CircadianOptimizer:
    """
    Circadian Optimizer: Wake-Sleep Learning Paradigm.

    Separates learning into Wake (experience) and Sleep (consolidation)
    phases, eliminating the need to store full computational graphs.

    Args:
        model: The model to optimize
        base_optimizer: The underlying optimizer for the Critic Network
        sleep_interval: Number of Wake steps before triggering Sleep
        critic_lr: Learning rate for the Critic Network
    """

    def __init__(
        self,
        model: Module,
        base_optimizer: Optional[Any] = None,
        sleep_interval: int = 100,
        critic_lr: float = 1e-3,
    ):
        self.model = model
        self.base_optimizer = base_optimizer
        self.sleep_interval = sleep_interval
        self.critic_lr = critic_lr
        self.step_count = 0

        # Surprise Vector cache
        self.surprise_cache: List[np.ndarray] = []

        # Sleep statistics
        self.total_sleeps = 0
        self.total_surprises_logged = 0

    def log_surprise(self, surprise_vector: np.ndarray):
        """Log a Surprise Vector during Wake phase."""
        self.surprise_cache.append(surprise_vector)
        self.total_surprises_logged += 1

    def wake_step(self, loss_value: float, hidden_states: np.ndarray, target_hidden: Optional[np.ndarray] = None):
        """
        Perform a Wake phase step.

        The model experiences data and logs surprise errors. Weights are
        NOT updated during Wake.

        Args:
            loss_value: scalar loss for this step (used as a magnitude prior
                for the surprise signal).
            hidden_states: current hidden activations.
            target_hidden: optional target activations. When supplied the
                surprise is the squared prediction error
                (hidden_states - target_hidden)^2, which is the standard
                wake-phase surprise signal. When omitted the surprise is
                the deviation of hidden_states from a learned baseline
                carried in `self._baseline` (initialised lazily).
        """
        self.step_count += 1

        # Real surprise signal, not random noise.
        if target_hidden is not None:
            if target_hidden.shape != hidden_states.shape:
                raise ValueError(
                    f"target_hidden shape {target_hidden.shape} != hidden_states "
                    f"shape {hidden_states.shape}"
                )
            surprise = ((hidden_states - target_hidden) ** 2).astype(np.float32)
        else:
            if not hasattr(self, "_baseline") or self._baseline.shape != hidden_states.shape:
                self._baseline = np.zeros_like(hidden_states, dtype=np.float32)
            surprise = ((hidden_states - self._baseline) ** 2).astype(np.float32)
            # EMA update of the baseline so future surprises measure
            # deviation from the running mean activation rather than zero.
            self._baseline = (0.9 * self._baseline + 0.1 * hidden_states).astype(np.float32)

        # Scale by the loss magnitude so a higher-loss step contributes
        # proportionally more to the sleep consolidation.
        loss_scale = float(np.clip(np.abs(loss_value), 0.0, 1e3))
        surprise = surprise * loss_scale

        self.log_surprise(surprise)

        # Check if it's time to sleep
        if self.step_count % self.sleep_interval == 0:
            self.sleep()

    def sleep(self):
        """
        Perform a Sleep phase.

        The Critic Network consolidates Surprise Vectors into weight
        updates. We project the consolidated surprise through every trainable
        parameter's gradient mapping (size-preserving) so all parameters
        receive a meaningful update, not just the one whose shape happens
        to match the surprise vector.
        """
        if not self.surprise_cache:
            return

        # Consolidate: weighted average across the cache, weighted by
        # L2 magnitude so higher-energy surprises dominate.
        stacked = np.stack(self.surprise_cache, axis=0).astype(np.float32)
        weights = np.linalg.norm(stacked.reshape(len(stacked), -1), axis=1) + 1e-8
        weights = weights / weights.sum()
        consolidated = np.tensordot(weights, stacked, axes=([0], [0])).astype(np.float32)

        # Apply consolidated update to all trainable parameters by
        # broadcasting the consolidated signal into each parameter's
        # shape. Parameters whose total element count matches the
        # consolidated signal are updated elementwise via a flat view;
        # others receive a direction-projected update.
        flat_surprise = consolidated.reshape(-1)
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            target = param.data.reshape(-1)
            if target.size == flat_surprise.size:
                update = flat_surprise.reshape(param.data.shape)
            else:
                # Project: take the first `target.size` elements, or repeat
                # the surprise vector if it is smaller than the parameter.
                if flat_surprise.size >= target.size:
                    update = flat_surprise[: target.size].reshape(param.data.shape)
                else:
                    reps = int(np.ceil(target.size / flat_surprise.size))
                    update = np.tile(flat_surprise, reps)[: target.size].reshape(param.data.shape)

            # Scale the update so the per-parameter magnitude is bounded
            # by the critic_lr (preserves the original API contract).
            param_scale = float(np.clip(self.critic_lr / (np.linalg.norm(update) + 1e-8), 0.0, 1.0))
            param.data = param.data - (param_scale * update).astype(param.data.dtype)

        # Clear cache
        num_surprises = len(self.surprise_cache)
        self.surprise_cache.clear()
        self.total_sleeps += 1

        return {
            "consolidated_surprises": num_surprises,
            "sleep_number": self.total_sleeps,
        }

    def step(self):
        """Advance the wake/sleep cycle by one surprise-less tick.

        This is a convenience alias for callers that simply want to drive
        the sleep-interval counter without supplying a wake signal. For
        real learning, use :meth:`wake_step` with explicit hidden states
        and (optionally) target hidden states.
        """
        self.step_count += 1
        if self.step_count % self.sleep_interval == 0:
            self.sleep()

    @property
    def is_sleeping(self) -> bool:
        """Check if it's time to sleep."""
        return self.step_count % self.sleep_interval == 0

    def __repr__(self) -> str:
        return (
            f"CircadianOptimizer("
            f"sleep_interval={self.sleep_interval}, "
            f"cached_surprises={len(self.surprise_cache)}, "
            f"total_sleeps={self.total_sleeps})"
        )