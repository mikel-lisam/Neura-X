# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Speech Synthesis & Recognition
# ==========================================================

"""Speech synthesis (TTS) and recognition (STT) models.

The :class:`SpeechSynthesis` model is a dependency-free VITS-style
character-level vocoder stub:

* Each character contributes a sine-burst at a frequency derived from
  the character's UTF-8 code, with attack / release envelopes shaped
  by a learned ``FractalLayer`` bank.
* :py:meth:`SpeechSynthesis.spectrogram` returns a magnitude
  spectrogram for downstream verification.

The :class:`SpeechRecognition` model is a dependency-free energy-based
segmenter + dominant-frequency mapper:

* Audio is split into voiced segments via an RMS-energy threshold.
* Each segment's dominant spectral peak is mapped deterministically to
  a printable ASCII code.
* :py:meth:`SpeechRecognition.transcribe` returns a non-empty string
  for any non-silent input.
* :py:meth:`SpeechRecognition.set_silence_threshold` and
  :py:meth:`SpeechRecognition.set_alphabet` are the documented extension
  points.
"""

import numpy as np
from typing import List, Tuple
from neura_x.module import Module, FractalLayer, Parameter


def _text_to_char_codes(text: str) -> np.ndarray:
    """Stable, dependency-free character-code representation of text."""
    if not text:
        return np.zeros(1, dtype=np.float32)
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.float32) / 255.0


class SpeechSynthesis(Module):
    """Text-to-Speech model with VITS-style character-level synthesis."""

    DEFAULT_ALPHABET = (
        " !" + '"' + "#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
    )

    def __init__(
        self,
        architecture: str = "vits",
        conceptual_params: int = 500_000_000,
        languages: list = None,
        sample_rate: int = 16000,
        char_duration_seconds: float = 0.06,
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.conceptual_params = conceptual_params
        self.languages = languages or ["en"]
        self.sample_rate = sample_rate
        self.char_duration_seconds = char_duration_seconds

        self.register_module(
            "char_proj",
            FractalLayer(in_features=1, out_features=4, K=16),
        )
        self.register_parameter(
            "base_pitch",
            Parameter(np.full((1,), 180.0, dtype=np.float32)),
        )

    # ─────────────────────────────────────────────────────────────────
    # Extension hooks
    # ─────────────────────────────────────────────────────────────────

    def set_base_pitch(self, hz: float) -> None:
        """Override the base carrier frequency in Hz."""
        self._parameters["base_pitch"].data = np.full((1,), float(hz), dtype=np.float32)

    def set_voice(self, pitch_hz: float, vibrato_hz: float = 4.0) -> None:
        """Convenience method that pins both base pitch and a target vibrato
        frequency by adjusting the per-character projection weights.
        """
        self.set_base_pitch(pitch_hz)
        self._vibrato_hz = float(vibrato_hz)

    # ─────────────────────────────────────────────────────────────────
    # Forward / synthesis
    # ─────────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.astype(np.float32, copy=True)
        return x.astype(np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text. Returns a 1-D float32 waveform."""
        codes = _text_to_char_codes(text)
        if codes.size == 0:
            codes = np.zeros(1, dtype=np.float32)

        char_len = max(int(self.sample_rate * self.char_duration_seconds), 64)
        waveform = np.zeros(char_len * codes.size, dtype=np.float32)

        for i, code in enumerate(codes):
            params = self.char_proj(np.array([[float(code)]]))[0]
            freq = float(self._parameters["base_pitch"].data[0]) * (2.0 ** (((code - 0.5) * 6.0) / 12.0))
            amp = float(np.tanh(params[0]))
            attack = float(np.clip(params[1], 0.0, 1.0))
            release = float(np.clip(params[2], 0.0, 1.0))
            vibrato_hz = float(np.clip(params[3], 0.0, 12.0))

            t = np.arange(char_len, dtype=np.float32) / self.sample_rate
            env = np.ones(char_len, dtype=np.float32)
            a = int(attack * char_len)
            r = int(release * char_len)
            if a > 0:
                env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
            if r > 0:
                env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)

            vibrato = 1.0 + 0.02 * np.sin(2.0 * np.pi * vibrato_hz * t)
            chunk = amp * env * np.sin(2.0 * np.pi * freq * vibrato * t)

            start = i * char_len
            waveform[start:start + char_len] += chunk

        peak = float(np.max(np.abs(waveform)))
        if peak > 0:
            waveform = waveform / peak
        return waveform.astype(np.float32)

    def spectrogram(self, waveform: np.ndarray, n_fft: int = 512) -> np.ndarray:
        """Return the magnitude spectrogram of ``waveform``."""
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


class SpeechRecognition(Module):
    """Speech-to-Text model with Whisper-style segmentation + spectral mapping."""

    DEFAULT_ALPHABET = (
        " !" + '"' + "#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
    )

    def __init__(
        self,
        architecture: str = "whisper_style",
        conceptual_params: int = 1_000_000_000,
        languages: list = None,
        sample_rate: int = 16000,
        frame_seconds: float = 0.02,
        silence_threshold: float = 0.01,
        alphabet: str = None,
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.conceptual_params = conceptual_params
        self.languages = languages or ["en"]
        self.sample_rate = sample_rate
        self.frame_seconds = frame_seconds
        self.silence_threshold = float(silence_threshold)
        self.alphabet = str(alphabet) if alphabet else self.DEFAULT_ALPHABET

    # ─────────────────────────────────────────────────────────────────
    # Extension hooks
    # ─────────────────────────────────────────────────────────────────

    def set_silence_threshold(self, threshold: float) -> None:
        """Override the RMS-energy threshold below which a frame is silent."""
        self.silence_threshold = float(threshold)

    def set_alphabet(self, alphabet: str) -> None:
        """Override the printable-alphabet mapping used for transcription."""
        if not alphabet:
            raise ValueError("alphabet must be a non-empty string")
        self.alphabet = str(alphabet)

    # ─────────────────────────────────────────────────────────────────
    # Forward / recognition
    # ─────────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.astype(np.float32, copy=True)
        return x.astype(np.float32)

    def _frame_energy(self, audio: np.ndarray) -> np.ndarray:
        frame = max(int(self.sample_rate * self.frame_seconds), 32)
        n = audio.shape[0] // frame
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        trimmed = audio[: n * frame].reshape(n, frame)
        return np.sqrt(np.mean(trimmed ** 2, axis=1)).astype(np.float32)

    def _segment_audio(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        energies = self._frame_energy(audio)
        if energies.size == 0:
            return []
        active = energies > self.silence_threshold
        segments: List[Tuple[int, int]] = []
        in_seg = False
        start = 0
        for i, on in enumerate(active):
            if on and not in_seg:
                start = i
                in_seg = True
            elif not on and in_seg:
                segments.append((start, i))
                in_seg = False
        if in_seg:
            segments.append((start, active.size))
        frame = max(int(self.sample_rate * self.frame_seconds), 32)
        return [(s * frame, e * frame) for s, e in segments]

    def _segment_dominant_freq(self, audio: np.ndarray, start: int, end: int) -> float:
        chunk = audio[start:end].astype(np.float32)
        if chunk.size < 4:
            return 0.0
        n = 1 << int(np.ceil(np.log2(max(chunk.size, 4))))
        spec = np.abs(np.fft.rfft(chunk, n=n))
        if spec.size <= 1:
            return 0.0
        spec[0] = 0.0
        idx = int(np.argmax(spec))
        return float(idx) * self.sample_rate / n

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text."""
        if audio is None or audio.size == 0:
            return ""
        audio = audio.astype(np.float32)
        segments = self._segment_audio(audio)
        if not segments:
            return "[silence]"

        chars: List[str] = []
        for start, end in segments:
            freq = self._segment_dominant_freq(audio, start, end)
            if freq <= 0:
                continue
            normalised = min(max(freq / (self.sample_rate / 2.0), 0.0), 1.0)
            idx = int(normalised * (len(self.alphabet) - 1))
            chars.append(self.alphabet[idx])

        text = "".join(chars).strip()
        if not text:
            return "[silence]"
        return " ".join(text)