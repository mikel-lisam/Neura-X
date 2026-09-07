# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Video Generation Models
# ==========================================================

"""Video generation models: Temporal Diffusion, Autoregressive.

Modular architecture:

* :class:`_TextEncoder` — deterministic text→vector; replace with CLIP
  when available without touching the rest of the pipeline.
* :class:`VideoGen` — per-frame denoise + temporal mixing.
  ``generate()`` runs a DDIM-style loop with optional classifier-free
  guidance.
* The temporal mixing matrix is resized lazily to whatever frame count
  the caller asks for, so :py:meth:`VideoGen.generate` works for any T.
"""

import numpy as np
from typing import Optional
from neura_x.module import Module, FractalLayer, Parameter


class _TextEncoder:
    """Deterministic 128-D text encoder used for prompt conditioning."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        chars = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float32)
        idx = np.arange(self.dim, dtype=np.float32)
        phases = chars[:, None] * (idx[None, :] / self.dim)
        out = np.sin(phases).sum(axis=0).astype(np.float32)
        norm = np.linalg.norm(out)
        if norm > 0:
            out /= norm
        return out


class VideoGen(Module):
    """Video generation model with per-frame denoising + temporal mixing.

    Args:
        architecture: backbone architecture identifier.
        conceptual_params: nominal parameter budget (informational).
        resolution: per-frame spatial resolution (square).
        fps: frames per second.
        duration_seconds: default video length when ``frames`` is omitted
            in :py:meth:`generate`.
        time_dim: dimensionality of the time embedding.
        cond_dim: dimensionality of the text conditioning vector.
        noise_schedule: "linear", "cosine", or "sigmoid".
    """

    def __init__(
        self,
        architecture: str = "temporal_diffusion",
        conceptual_params: int = 5_000_000_000,
        resolution: int = 16,
        fps: int = 8,
        duration_seconds: int = 2,
        time_dim: int = 64,
        cond_dim: int = 128,
        noise_schedule: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.conceptual_params = conceptual_params
        self.resolution = resolution
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.total_frames = fps * duration_seconds
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.noise_schedule = noise_schedule

        self._text_encoder = _TextEncoder(dim=cond_dim)

        feat = resolution * resolution * 3
        self.register_module("denoise", FractalLayer(in_features=feat, out_features=feat, K=128))

        # Temporal mixing matrix — initialised lazily to the actual T.
        self.register_parameter(
            "temporal_w",
            Parameter(np.eye(1, dtype=np.float32)),
        )
        self._temporal_w_T: int = 0
        self.register_parameter(
            "time_w",
            Parameter(np.random.randn(time_dim, feat).astype(np.float32) * 0.02),
        )
        self.register_parameter(
            "cond_w",
            Parameter(np.random.randn(cond_dim, feat).astype(np.float32) * 0.02),
        )

    # ─────────────────────────────────────────────────────────────────
    # Extension hooks
    # ─────────────────────────────────────────────────────────────────

    def encode_text(self, prompt: str) -> np.ndarray:
        """Public hook so callers can pre-compute the conditioning vector."""
        return self._text_encoder(prompt)

    def set_temporal_init(self, init: np.ndarray) -> None:
        """Replace the temporal mixing matrix with a custom initialisation.

        ``init`` must be a 2-D square ``np.ndarray``.  The model will
        reinitialise the temporal mixing matrix on the next call to
        :py:meth:`generate` with ``init.shape[0]`` frames.
        """
        if init.ndim != 2 or init.shape[0] != init.shape[1]:
            raise ValueError("temporal init must be a square 2-D matrix")
        self._parameters["temporal_w"].data = init.astype(np.float32)
        self._temporal_w_T = init.shape[0]

    # ─────────────────────────────────────────────────────────────────
    # Forward / sampling
    # ─────────────────────────────────────────────────────────────────

    def _time_embedding(self, t_values: np.ndarray) -> np.ndarray:
        """Vectorised sinusoidal time embedding for a sequence of timesteps."""
        half = self.time_dim // 2
        freqs = np.exp(
            -np.log(10000.0) * np.arange(half, dtype=np.float32) / max(half - 1, 1)
        )
        args = t_values.astype(np.float32)[:, None] * freqs[None, :]
        return np.concatenate([np.sin(args), np.cos(args)], axis=-1).astype(np.float32)

    def _denoise_step(self, frames: np.ndarray, t: np.ndarray, cond: np.ndarray) -> np.ndarray:
        T = frames.shape[0]
        feat = self.resolution * self.resolution * 3
        flat = frames.reshape(T, feat).astype(np.float32)

        # Resize temporal mixing matrix to match runtime T.
        tw = self._parameters["temporal_w"].data
        if tw.shape != (T, T):
            new = np.eye(T, dtype=np.float32) * (tw.diagonal().mean() if tw.size > 0 else 1.0)
            n = min(tw.shape[0], T)
            new[:n, :n] = tw[:n, :n]
            self._parameters["temporal_w"].data = new.astype(np.float32)
            tw = new
            self._temporal_w_T = T

        # Sinusoidal time embedding per frame.
        t_emb = self._time_embedding(t)
        if t_emb.shape[-1] < self.time_dim:
            pad = np.zeros((t_emb.shape[0], self.time_dim - t_emb.shape[-1]), dtype=np.float32)
            t_emb = np.concatenate([t_emb, pad], axis=-1)
        t_proj = t_emb @ self._parameters["time_w"].data
        c_proj = np.broadcast_to(cond[None, :], (T, self.cond_dim)) @ self._parameters["cond_w"].data

        # Pad/trim projections to feature width.
        if t_proj.shape[-1] != feat:
            new = np.zeros((T, feat), dtype=np.float32)
            n = min(t_proj.shape[-1], feat)
            new[:, :n] = t_proj[:, :n]
            t_proj = new
        if c_proj.shape[-1] != feat:
            new = np.zeros((T, feat), dtype=np.float32)
            n = min(c_proj.shape[-1], feat)
            new[:, :n] = c_proj[:, :n]
            c_proj = new

        h = flat + t_proj + c_proj
        h = self.denoise(h) + h
        h = tw @ h
        return h.reshape(frames.shape).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[None, ...]
        cond = self._text_encoder("")
        return self._denoise_step(x, np.zeros(x.shape[0], dtype=np.float32), cond)

    def generate(
        self,
        prompt: str,
        frames: Optional[int] = None,
        steps: int = 8,
        seed: int = 0,
        guidance_scale: float = 1.0,
    ) -> np.ndarray:
        """Generate a video from a text prompt."""
        T = int(frames or self.total_frames)
        rng = np.random.default_rng(seed)
        R = self.resolution
        x = rng.standard_normal((T, R, R, 3)).astype(np.float32) * 0.5

        cond = self._text_encoder(prompt)
        uncond = self._text_encoder("")
        timesteps = self._ddim_timesteps(steps)

        for i, t in enumerate(timesteps):
            t_arr = np.full(T, float(t), dtype=np.float32)
            noise_pred = self._denoise_step(x, t_arr, cond)
            if guidance_scale > 1.0:
                uncond_pred = self._denoise_step(x, t_arr, uncond)
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self._ddim_step(x, noise_pred, t, t_prev)

        x = np.clip(x, -3.0, 3.0)
        x_min, x_max = float(x.min()), float(x.max())
        if x_max - x_min > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x)
        return x.astype(np.float32)

    def _ddim_timesteps(self, num_steps: int) -> list:
        if num_steps <= 0:
            return [0]
        return list(reversed([
            int(round(i * 1000.0 / num_steps))
            for i in range(num_steps)
        ]))

    def _ddim_step(
        self,
        x_t: np.ndarray,
        noise_pred: np.ndarray,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> np.ndarray:
        """DDIM update rule with linear alpha schedule (β=0.0001→0.02)."""
        beta_start, beta_end = 0.0001, 0.02
        betas = np.linspace(beta_start, beta_end, 1001, dtype=np.float32)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        a_t = float(alphas_cumprod[max(0, min(t, 1000))])
        a_prev = float(alphas_cumprod[max(0, min(t_prev, 1000))]) if t_prev >= 0 else 1.0

        x0_pred = (x_t - np.sqrt(max(1.0 - a_t, 1e-8)) * noise_pred) / np.sqrt(max(a_t, 1e-8))
        dir_xt = np.sqrt(max(1.0 - a_prev, 1e-8)) * noise_pred
        x_prev = np.sqrt(a_prev) * x0_pred + dir_xt
        return x_prev.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"VideoGen(arch={self.architecture}, "
            f"params={self.conceptual_params:,}, "
            f"res={self.resolution}, fps={self.fps})"
        )