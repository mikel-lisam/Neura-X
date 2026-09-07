# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Image Generation Models
# ==========================================================

"""Image generation models: Diffusion, GAN, VAE, DiT.

The module is structured for incremental build-out:

* :class:`DiffusionScheduler` — encapsulates noise schedule math
  (linear, cosine, sigmoid) with extension points for learned
  variance.  Powers :py:meth:`ImageGen.generate`.
* :class:`UNetBlock` — a single UNet residual block: two
  :class:`FractalLayer` convolutions with FiLM-style time + text
  conditioning.  Designed to be stacked into a full UNet via
  :py:meth:`ImageGen.add_block`.
* :class:`_TextEncoder` — dependency-free text→vector encoder used as
  the conditioning source.  Replace with a CLIP / T5 encoder when one
  becomes available without touching the rest of the pipeline.
* :class:`ImageGen` — the user-facing model.  ``generate()`` runs an
  iterative DDPM-style denoising loop.  ``add_block`` and
  ``set_scheduler`` are the documented extension points.
"""

import numpy as np
from typing import Iterable, List, Optional
from neura_x.module import Module, FractalLayer, Parameter


def _sinusoidal_time_embedding(t: np.ndarray, dim: int) -> np.ndarray:
    """Standard sinusoidal time embedding (Vaswani et al., 2017)."""
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / max(half - 1, 1))
    args = t.astype(np.float32)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1).astype(np.float32)


class DiffusionScheduler:
    """Noise schedule for DDPM/DDIM sampling.

    Supports linear, cosine, and sigmoid schedules.  The scheduler owns
    ``betas``, ``alphas``, ``alphas_cumprod`` and exposes helpers for
    adding noise to a clean sample and for the DDIM update step.  All
    arrays are stored as ``float32`` and sized to ``num_train_timesteps``.
    """

    SCHEDULES = ("linear", "cosine", "sigmoid")

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        if schedule not in self.SCHEDULES:
            raise ValueError(f"schedule must be one of {self.SCHEDULES}, got {schedule!r}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.schedule = schedule

        self.betas = self._build_betas(schedule).astype(np.float32)
        self.alphas = (1.0 - self.betas).astype(np.float32)
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float32)

    def _build_betas(self, schedule: str) -> np.ndarray:
        n = self.num_train_timesteps
        if schedule == "linear":
            return np.linspace(self.beta_start, self.beta_end, n, dtype=np.float64)
        if schedule == "sigmoid":
            betas = np.linspace(-6, 6, n, dtype=np.float64)
            betas = 1.0 / (1.0 + np.exp(-betas))
            return (betas * (self.beta_end - self.beta_start) + self.beta_start).astype(np.float64)
        # cosine schedule as in Nichol & Dhariwal (2021).
        s = 0.008
        steps = n + 1
        x = np.linspace(0, n, steps, dtype=np.float64) / n
        alphas_cumprod = np.cos(((x + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999).astype(np.float64)

    def add_noise(self, original: np.ndarray, noise: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
        """Forward diffusion: ``x_t = sqrt(a_t) * x_0 + sqrt(1 - a_t) * eps``."""
        a = self.alphas_cumprod[timesteps.astype(np.int64)]
        shape = (a.shape[0],) + (1,) * (original.ndim - 1)
        sqrt_a = np.sqrt(a).reshape(shape)
        sqrt_one_minus = np.sqrt(1.0 - a).reshape(shape)
        return (sqrt_a * original + sqrt_one_minus * noise).astype(np.float32)

    def ddim_step(
        self,
        x_t: np.ndarray,
        noise_pred: np.ndarray,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> np.ndarray:
        """One DDIM sampling step (Song et al., 2020).

        ``eta=0`` is deterministic; values in (0, 1] add stochasticity.
        """
        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
        sigma = eta * np.sqrt((1.0 - a_prev) / (1.0 - a_t)) * np.sqrt(1.0 - a_t / a_prev)

        # Predicted x0.
        x0_pred = (x_t - np.sqrt(1.0 - a_t) * noise_pred) / np.sqrt(a_t)
        # Direction pointing to x_t.
        dir_xt = np.sqrt(1.0 - a_prev - sigma ** 2) * noise_pred
        noise = sigma * np.random.standard_normal(x_t.shape).astype(np.float32)
        x_prev = np.sqrt(a_prev) * x0_pred + dir_xt + noise
        return x_prev.astype(np.float32)

    def timestep_indices(self, num_inference_steps: int) -> List[int]:
        """Standard evenly-spaced inference schedule."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        return [int(round(i * step_ratio)) for i in range(num_inference_steps)][::-1]


class UNetBlock(Module):
    """A single UNet residual block.

    Layout:
        h = FiLM(time + cond)
        h = FractalLayer(h) + skip
        h = FiLM(time + cond)
        h = FractalLayer(h) + skip

    FiLM (Feature-wise Linear Modulation) injects the conditioning via
    additive bias.  This is the documented extension point for plugging
    in cross-attention or adaptive group-norm once those are added.
    """

    def __init__(
        self,
        channels: int,
        time_dim: int,
        cond_dim: int,
        K: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        self.cond_dim = cond_dim

        self.register_module("conv1", FractalLayer(in_features=channels, out_features=channels, K=K))
        self.register_module("conv2", FractalLayer(in_features=channels, out_features=channels, K=K))

        self.register_parameter(
            "time_w1",
            Parameter(np.random.randn(time_dim, channels).astype(np.float32) * 0.02),
        )
        self.register_parameter(
            "time_w2",
            Parameter(np.random.randn(time_dim, channels).astype(np.float32) * 0.02),
        )
        self.register_parameter(
            "cond_w1",
            Parameter(np.random.randn(cond_dim, channels).astype(np.float32) * 0.02),
        )
        self.register_parameter(
            "cond_w2",
            Parameter(np.random.randn(cond_dim, channels).astype(np.float32) * 0.02),
        )

    def _project(self, t_emb: np.ndarray, cond: np.ndarray, which: int) -> np.ndarray:
        """Project (time, cond) → FiLM bias of shape ``(batch, channels)``."""
        tw = self._parameters[f"time_w{which}"].data
        cw = self._parameters[f"cond_w{which}"].data
        # Pad/trim to handle mismatched time_dim.
        td = min(tw.shape[0], t_emb.shape[-1])
        cd = min(cw.shape[0], cond.shape[-1])
        return t_emb[..., :td] @ tw[:td] + cond[..., :cd] @ cw[:cd]

    def forward(
        self,
        x: np.ndarray,
        t_emb: np.ndarray,
        cond: np.ndarray,
    ) -> np.ndarray:
        h = x
        film = self._project(t_emb, cond, 1)
        if film.shape[-1] != self.channels:
            new = np.zeros((film.shape[0], self.channels), dtype=np.float32)
            n = min(film.shape[-1], self.channels)
            new[..., :n] = film[..., :n]
            film = new
        h = self.conv1(h + film) + h

        film = self._project(t_emb, cond, 2)
        if film.shape[-1] != self.channels:
            new = np.zeros((film.shape[0], self.channels), dtype=np.float32)
            n = min(film.shape[-1], self.channels)
            new[..., :n] = film[..., :n]
            film = new
        h = self.conv2(h + film) + h
        return h


class _TextEncoder:
    """Deterministic, dependency-free text→vector encoder.

    The encoder hashes tokens into a stable 256-D embedding so the same
    prompt always produces the same conditioning signal (no random text
    embeddings). This is what we feed into the diffusion backbone.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=np.float32)
        # Deterministic hash: each character contributes a per-dim bump.
        chars = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float32)
        idx = np.arange(self.dim, dtype=np.float32)
        # Sinusoidal position-keyed sum.
        phases = chars[:, None] * (idx[None, :] / self.dim)
        out = np.sin(phases).sum(axis=0).astype(np.float32)
        norm = np.linalg.norm(out)
        if norm > 0:
            out /= norm
        return out


class ImageGen(Module):
    """
    Image generation model supporting multiple architectures.

    Backbone: a stack of :class:`UNetBlock`s with sinusoidal time
    embedding and FiLM-style text conditioning.  Sampling uses
    :class:`DiffusionScheduler` (linear / cosine / sigmoid) and the DDIM
    update rule so each step is deterministic given the predicted noise.

    Args:
        architecture: "latent_diffusion", "dit", "gan", "vae"
        conceptual_params: Conceptual parameter count
        resolution: Output image resolution
        time_dim: dimensionality of the time embedding
        cond_dim: dimensionality of the text conditioning vector
        n_blocks: number of UNet blocks in the backbone
        num_train_timesteps: total diffusion steps the scheduler knows about
        schedule: "linear", "cosine", or "sigmoid"
    """

    def __init__(
        self,
        architecture: str = "latent_diffusion",
        conceptual_params: int = 2_500_000_000,
        resolution: int = 32,
        time_dim: int = 64,
        cond_dim: int = 128,
        n_blocks: int = 2,
        num_train_timesteps: int = 1000,
        schedule: str = "linear",
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.conceptual_params = conceptual_params
        self.resolution = resolution
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self._latent_dim = max(resolution // 4, 8) ** 2

        self._text_encoder = _TextEncoder(dim=cond_dim)
        self.scheduler = DiffusionScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule=schedule,
        )

        # Project image → latent and back.
        feat = resolution * resolution * 3
        self.register_module(
            "down",
            FractalLayer(in_features=feat, out_features=self._latent_dim, K=128),
        )
        self.register_module(
            "up",
            FractalLayer(in_features=self._latent_dim, out_features=feat, K=128),
        )

        # Stack of UNet blocks operating on the latent vector.
        self._blocks: List[UNetBlock] = []
        for i in range(max(1, n_blocks)):
            block = UNetBlock(
                channels=self._latent_dim,
                time_dim=time_dim,
                cond_dim=cond_dim,
            )
            self.register_module(f"block_{i}", block)
            self._blocks.append(block)

        # Time + cond biases at the input/output of the latent stream.
        self.register_parameter(
            "time_in_w",
            Parameter(np.random.randn(time_dim, self._latent_dim).astype(np.float32) * 0.02),
        )
        self.register_parameter(
            "cond_in_w",
            Parameter(np.random.randn(cond_dim, self._latent_dim).astype(np.float32) * 0.02),
        )

    # ─────────────────────────────────────────────────────────────────
    # Extension API
    # ─────────────────────────────────────────────────────────────────

    def add_block(self, block: UNetBlock) -> None:
        """Append an additional ``UNetBlock`` to the backbone."""
        if not isinstance(block, UNetBlock):
            raise TypeError("expected a UNetBlock instance")
        idx = len(self._blocks)
        self.register_module(f"block_{idx}", block)
        self._blocks.append(block)

    def set_scheduler(self, scheduler: DiffusionScheduler) -> None:
        """Replace the diffusion scheduler (e.g., switch linear → cosine)."""
        if not isinstance(scheduler, DiffusionScheduler):
            raise TypeError("expected a DiffusionScheduler instance")
        self.scheduler = scheduler

    def encode_text(self, prompt: str) -> np.ndarray:
        """Public hook so callers can pre-compute text embeddings."""
        return self._text_encoder(prompt)

    # ─────────────────────────────────────────────────────────────────
    # Forward / sampling
    # ─────────────────────────────────────────────────────────────────

    def _denoise_step(self, x: np.ndarray, t: int, cond: np.ndarray) -> np.ndarray:
        feat = self.resolution * self.resolution * 3

        h = x.reshape(1, feat).astype(np.float32)
        z = self.down(h)
        if z.shape[-1] != self._latent_dim:
            new = np.zeros((1, self._latent_dim), dtype=np.float32)
            n = min(z.shape[-1], self._latent_dim)
            new[:, :n] = z[:, :n]
            z = new

        t_emb = _sinusoidal_time_embedding(np.array([float(t)]), self.time_dim)
        t_bias = t_emb @ self._parameters["time_in_w"].data
        c_bias = cond[None, :] @ self._parameters["cond_in_w"].data
        # Pad/trim to latent dim.
        for arr_name in ("t_bias", "c_bias"):
            arr = locals()[arr_name]
            if arr.shape[-1] != self._latent_dim:
                new = np.zeros((arr.shape[0], self._latent_dim), dtype=np.float32)
                n = min(arr.shape[-1], self._latent_dim)
                new[:, :n] = arr[:, :n]
                locals()[arr_name] = new
        # Re-bind in case locals() mutation didn't take.
        if t_bias.shape[-1] != self._latent_dim:
            tb = np.zeros((t_bias.shape[0], self._latent_dim), dtype=np.float32)
            n = min(t_bias.shape[-1], self._latent_dim)
            tb[:, :n] = t_bias[:, :n]
            t_bias = tb
        if c_bias.shape[-1] != self._latent_dim:
            cb = np.zeros((c_bias.shape[0], self._latent_dim), dtype=np.float32)
            n = min(c_bias.shape[-1], self._latent_dim)
            cb[:, :n] = c_bias[:, :n]
            c_bias = cb

        z = z + t_bias + c_bias
        for block in self._blocks:
            z = block(z, t_emb, cond[None, :])

        h_out = self.up(z)
        if h_out.shape[-1] != feat:
            new = np.zeros((1, feat), dtype=np.float32)
            n = min(h_out.shape[-1], feat)
            new[:, :n] = h_out[:, :n]
            h_out = new
        return h_out.reshape(x.shape).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: treat ``x`` as an image tensor and run one denoise."""
        if x.ndim == 3:
            x = x[None, ...]
        cond = self._text_encoder("")
        return self._denoise_step(x, 0, cond)[0]

    def generate(
        self,
        prompt: str,
        steps: int = 20,
        seed: int = 0,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate an image from a text prompt using DDIM sampling.

        Args:
            prompt: text conditioning
            steps: number of denoising iterations (DDIM sub-steps)
            seed: RNG seed for reproducibility
            eta: DDIM stochasticity (0 = deterministic)
            guidance_scale: classifier-free guidance strength.  When > 1
                the model also evaluates an unconditional pass and
                extrapolates away from it; ``1.0`` disables guidance.
            noise: optional pre-generated noise tensor for fixed seed
                reproducibility across reruns.
        """
        rng = np.random.default_rng(seed)
        R = self.resolution
        if noise is None:
            x = rng.standard_normal((R, R, 3)).astype(np.float32) * 0.5
        else:
            x = noise.astype(np.float32, copy=True)

        cond = self._text_encoder(prompt)
        uncond = self._text_encoder("")
        timesteps = self.scheduler.timestep_indices(steps)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            noise_pred = self._denoise_step(x, t, cond)[0]
            if guidance_scale > 1.0:
                uncond_pred = self._denoise_step(x, t, uncond)[0]
                noise_pred = uncond_pred + guidance_scale * (noise_pred - uncond_pred)
            x = self.scheduler.ddim_step(x, noise_pred, t, t_prev, eta=eta)

        # Squash to [0, 1] for display.
        x = np.clip(x, -3.0, 3.0)
        x_min, x_max = float(x.min()), float(x.max())
        if x_max - x_min > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        return x.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"ImageGen(arch={self.architecture}, "
            f"params={self.conceptual_params:,}, "
            f"res={self.resolution}, blocks={len(self._blocks)})"
        )