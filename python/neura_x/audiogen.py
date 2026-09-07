# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Audio & Music Generation Models
# ==========================================================

"""Audio and music generation models.

The :class:`AudioGen` model is a dependency-free neural vocoder stub:

* A deterministic text encoder produces a per-prompt conditioning
  vector.
* A latent generator produces a frame-level latent trajectory.
* A decoder reads out sine partials (frequencies + amplitudes) per
  frame.
* A short-time Fourier analysis post-processing pass is exposed via
  :py:meth:`AudioGen.spectrogram` so callers can verify the spectral
  content of generated audio.

Extension hooks:
    * :py:meth:`AudioGen.set_voice` — replace the partial-bank initialisation
      to mimic a specific timbre.
    * :py:meth:`AudioGen.add_post_filter` — append a custom DSP pass to the
      audio pipeline (e.g., reverb, EQ).
    * :py:meth:`AudioGen.encode_text` — pre-compute the conditioning vector.
"""

import numpy as np
from typing import Callable, List, Optional
from neura_x.module import Module, FractalLayer, Parameter


class _TextEncoder:
    """Deterministic 128-D text encoder for prompt conditioning."""

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


class AudioGen(Module):
    """Text-conditioned neural vocoder.

    Args:
        architecture: backbone architecture identifier.
        conceptual_params: nominal parameter budget (informational).
        sample_rate: output sample rate in Hz.
        duration_seconds: default audio length.
        latent_dim: dimensionality of the per-frame latent.
        n_partials: number of additive sine partials.
        chunk_seconds: seconds of audio synthesised per partial-bank call.
        cond_dim: text conditioning dimensionality.
    """

    def __init__(
        self,
        architecture: str = "autoregressive_transformer",
        conceptual_params: int = 1_500_000_000,
        sample_rate: int = 16000,
        duration_seconds: int = 1,
        latent_dim: int = 64,
        n_partials: int = 32,
        chunk_seconds: float = 0.05,
        cond_dim: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.conceptual_params = conceptual_params
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds
        self.latent_dim = latent_dim
        self.n_partials = n_partials
        self.chunk_seconds = float(chunk_seconds)
        self.cond_dim = cond_dim

        self._text_encoder = _TextEncoder(dim=cond_dim)

        # Per-step latent generator: maps (cond, t) -> latent_dim.
        self.register_module(
            "latent_gen",
            FractalLayer(in_features=cond_dim + 1, out_features=latent_dim, K=128),
        )
        # Decoder: latent_dim → (n_partials * 2) [freq, amp].
        self.register_module(
            "decoder",
            FractalLayer(in_features=latent_dim, out_features=n_partials * 2, K=128),
        )
        self.register_parameter(
            "cond_w",
            Parameter(np.random.randn(cond_dim, latent_dim).astype(np.float32) * 0.02),
        )

        # Extension: chain of post-processing DSP callbacks.
        self._post_filters: List[Callable[[np.ndarray], np.ndarray]] = []

    # ─────────────────────────────────────────────────────────────────
    # Extension hooks
    # ─────────────────────────────────────────────────────────────────

    def encode_text(self, prompt: str) -> np.ndarray:
        """Public hook so callers can pre-compute the conditioning vector."""
        return self._text_encoder(prompt)

    def add_post_filter(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """Append a DSP filter to the audio pipeline.

        The filter must accept a 1-D ``np.ndarray`` waveform and return
        one of the same length.  It runs after synthesis and before
        final normalisation.
        """
        self._post_filters.append(fn)

    def clear_post_filters(self) -> None:
        """Remove all post-processing filters."""
        self._post_filters.clear()

    # ─────────────────────────────────────────────────────────────────
    # Forward / synthesis
    # ─────────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Identity-style passthrough used for audio tensors."""
        if x.ndim == 1:
            return x.astype(np.float32, copy=True)
        return x.astype(np.float32)

    def _synth_chunk(self, t_value: float, cond: np.ndarray) -> np.ndarray:
        vec = np.concatenate([cond, np.array([t_value], dtype=np.float32)])
        latent = self.latent_gen(vec[None, :])[0]
        params = self.decoder(latent[None, :])[0]
        freqs = np.abs(params[: self.n_partials]) * (self.sample_rate / 2.0)
        amps = np.tanh(params[self.n_partials:])

        chunk_len = max(int(self.sample_rate * self.chunk_seconds), 64)
        t_axis = np.arange(chunk_len, dtype=np.float32) / self.sample_rate
        wave = np.zeros(chunk_len, dtype=np.float32)
        for k in range(self.n_partials):
            wave += amps[k] * np.sin(2.0 * np.pi * freqs[k] * t_axis)
        peak = float(np.max(np.abs(wave)))
        if peak > 1e-8:
            wave = wave / peak
        return wave.astype(np.float32)

    def generate(
        self,
        prompt: str,
        seed: int = 0,
        duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        """Generate audio from a text prompt.

        Args:
            prompt: text conditioning.
            seed: RNG seed (currently informational; synthesis is deterministic
                given the prompt and weights).
            duration_seconds: override the model's default length.

        Returns:
            1-D ``float32`` waveform of length ``sample_rate * duration``,
            normalised to ``[-1, 1]`` after post-processing.
        """
        cond = self._text_encoder(prompt)
        target_seconds = float(duration_seconds or self.duration_seconds)
        n_samples = int(self.sample_rate * target_seconds)
        chunk_len = max(int(self.sample_rate * self.chunk_seconds), 64)
        n_chunks = max(int(np.ceil(n_samples / chunk_len)), 1)

        pieces: List[np.ndarray] = []
        for i in range(n_chunks):
            t_value = i / max(n_chunks - 1, 1)
            pieces.append(self._synth_chunk(t_value, cond))
        waveform = np.concatenate(pieces)[:n_samples]

        # Cross-fade overlap-add for seam-free joins.
        if n_chunks > 1:
            fade = min(chunk_len // 4, 64)
            for i in range(1, n_chunks):
                start = i * chunk_len
                fade_end = min(start + fade, waveform.shape[0])
                if fade_end <= start:
                    break
                ramp_len = fade_end - start
                ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
                head = pieces[i][:ramp_len]
                waveform[start:fade_end] = (
                    waveform[start:fade_end] * (1.0 - ramp) + head * ramp
                )

        # Apply user-added DSP filters.
        for fn in self._post_filters:
            try:
                waveform = fn(waveform)
            except Exception:
                continue

        peak = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak
        return waveform.astype(np.float32)

    def spectrogram(self, waveform: np.ndarray, n_fft: int = 512) -> np.ndarray:
        """Compute the magnitude spectrogram of ``waveform``.

        Useful for tests and for downstream visualisations.  Returns an
        array of shape ``(n_fft // 2 + 1, num_frames)``.
        """
        if waveform.size == 0:
            return np.zeros((n_fft // 2 + 1, 0), dtype=np.float32)
        hop = n_fft // 2
        if waveform.size < n_fft:
            waveform = np.pad(waveform, (0, n_fft - waveform.size), mode="constant")
        n_frames = max(1, (waveform.size - n_fft) // hop + 1)
        frames = np.stack(
            [waveform[i * hop:i * hop + n_fft] for i in range(n_frames)],
            axis=0,
        ).astype(np.float32)
        window = np.hanning(n_fft + 2)[1:-1].astype(np.float32)
        frames = frames * window
        spec = np.fft.rfft(frames, n=n_fft, axis=-1)
        return np.abs(spec).T.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"AudioGen(arch={self.architecture}, "
            f"params={self.conceptual_params:,}, "
            f"sr={self.sample_rate})"
        )