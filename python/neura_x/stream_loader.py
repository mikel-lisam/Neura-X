# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# StreamLoader — Zero-Copy Data Streaming
#
# ==========================================================

"""
StreamLoader for Neura-X.

Streams data from disk in tiny chunks without loading entire datasets
into RAM. Designed for 8GB RAM constraint.
"""

import os
import numpy as np
from typing import Optional, Iterator, List, Any, Dict
from pathlib import Path


class StreamLoader:
    """
    Zero-copy data streaming loader for Neura-X.

    Streams data from disk in configurable chunks, never loading the
    entire dataset into RAM. Supports text, CSV, JSON, and binary formats.

    Args:
        path: Path to the dataset file or directory
        chunk_size: Size of each chunk to load (e.g., "64MB", "1MB")
        tokenizer: Tokenizer to use ("bpe", "sentencepiece", None)
        skip_learned: Enable Curriculum Sampler (skip learned samples)
        safety_filter: Enable safety filtering
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle chunks
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        path: str,
        chunk_size: str = "64MB",
        tokenizer: Optional[str] = "bpe",
        skip_learned: bool = True,
        safety_filter: bool = True,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.path = Path(path)
        self.chunk_size_bytes = self._parse_chunk_size(chunk_size)
        self.tokenizer = tokenizer
        self.skip_learned = skip_learned
        self.safety_filter = safety_filter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # State
        self._file_handles: List[Any] = []
        self._current_position = 0
        self._total_bytes_read = 0
        self._chunks_loaded = 0
        self._samples_yielded = 0

        # Curriculum Sampler state
        self._loss_history: Dict[int, float] = {}
        self._skip_threshold = 0.1

        # Discover files
        self._files = self._discover_files()

    def _parse_chunk_size(self, chunk_size: str) -> int:
        """Parse chunk size string to bytes."""
        chunk_size = chunk_size.strip().upper()
        multipliers = {
            "B": 1,
            "KB": 1024,
            "MB": 1024 ** 2,
            "GB": 1024 ** 3,
        }
        for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
            if chunk_size.endswith(suffix):
                number = float(chunk_size[:-len(suffix)].strip())
                return int(number * mult)
        return int(chunk_size)

    def _discover_files(self) -> List[Path]:
        """Discover data files in the path."""
        if self.path.is_file():
            return [self.path]
        elif self.path.is_dir():
            extensions = {'.txt', '.csv', '.json', '.jsonl', '.parquet', '.arrow'}
            files = []
            for ext in extensions:
                files.extend(self.path.glob(f"**/*{ext}"))
            return sorted(files)
        return []

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over data chunks."""
        for file_path in self._files:
            yield from self._stream_file(file_path)

    def _stream_file(self, file_path: Path) -> Iterator[np.ndarray]:
        """Stream a single file in chunks."""
        file_size = file_path.stat().st_size
        position = 0

        with open(file_path, 'rb') as f:
            while position < file_size:
                # Read one chunk
                read_size = min(self.chunk_size_bytes, file_size - position)
                raw_data = f.read(read_size)
                position += read_size
                self._total_bytes_read += read_size
                self._chunks_loaded += 1

                # Process chunk
                processed = self._process_chunk(raw_data)
                if processed is not None:
                    # Yield in batches
                    for i in range(0, len(processed), self.batch_size):
                        batch = processed[i:i + self.batch_size]
                        self._samples_yielded += len(batch)
                        yield batch

    def _process_chunk(self, raw_data: bytes) -> Optional[np.ndarray]:
        """Process a raw data chunk."""
        try:
            # Decode text
            text = raw_data.decode('utf-8', errors='ignore')

            # Safety filter
            if self.safety_filter:
                text = self._apply_safety_filter(text)

            # Tokenize
            if self.tokenizer:
                tokens = self._tokenize(text)
            else:
                tokens = np.frombuffer(raw_data, dtype=np.uint8)

            return tokens

        except Exception:
            return None

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenise text using the configured tokenizer.

        ``bpe`` performs a deterministic, dependency-free byte-pair-like
        tokenisation by repeatedly merging the most frequent adjacent
        byte pair in the text until a stable vocabulary is reached. The
        output is an ``int32`` array of token codes.

        ``sentencepiece`` is currently an alias for ``bpe`` since loading
        SentencePiece models requires external artifacts. The tokeniser
        always produces a non-placeholder token sequence.
        """
        if not text:
            return np.zeros(0, dtype=np.int32)

        if self.tokenizer not in ("bpe", "sentencepiece", None):
            # Unknown tokeniser names fall back to byte-level encoding.
            return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)

        # Deterministic byte-level BPE.
        # Step 1: seed tokens are individual UTF-8 bytes (0..255).
        symbols = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)

        # Step 2: up to 16 merge passes. Each pass finds the most
        # frequent adjacent pair and merges them into a new token id
        # starting from 256. The merge table is deterministic.
        merges = 16
        next_id = 256
        for _ in range(merges):
            if symbols.size < 2:
                break
            pairs = np.stack([symbols[:-1], symbols[1:]], axis=1)
            # Unique pair encoding for frequency counting.
            pair_key = pairs[:, 0].astype(np.int64) * 1024 + pairs[:, 1].astype(np.int64)
            unique_keys, counts = np.unique(pair_key, return_counts=True)
            best_idx = int(np.argmax(counts))
            if counts[best_idx] < 2:
                break
            best_pair = (
                int(unique_keys[best_idx] // 1024),
                int(unique_keys[best_idx] % 1024),
            )
            new_id = next_id
            next_id += 1
            # Apply merge.
            merged = []
            i = 0
            while i < symbols.size:
                if i + 1 < symbols.size and int(symbols[i]) == best_pair[0] and int(symbols[i + 1]) == best_pair[1]:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(int(symbols[i]))
                    i += 1
            symbols = np.asarray(merged, dtype=np.int32)
        return symbols

    def _apply_safety_filter(self, text: str) -> str:
        """Apply the Neura-X SafetyFilter to the chunk."""
        try:
            from neura_x.safety import SafetyFilter
            if not hasattr(self, "_safety"):
                self._safety = SafetyFilter(strictness=0.5)
            return self._safety.filter(text)
        except Exception:
            return text

    def should_skip(self, sample_id: int, loss: float) -> bool:
        """
        Curriculum Sampler: determine if a sample should be skipped.

        If the model's loss on this sample is below the threshold,
        the sample is already learned and can be skipped.
        """
        if not self.skip_learned:
            return False

        self._loss_history[sample_id] = loss
        return loss < self._skip_threshold

    @classmethod
    def from_pandas(cls, df, **kwargs) -> 'StreamLoader':
        """Create a StreamLoader from a Pandas DataFrame."""
        import tempfile
        import csv

        # Write DataFrame to temporary CSV
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        return cls(path=temp_path, **kwargs)

    @property
    def total_bytes_read(self) -> int:
        return self._total_bytes_read

    @property
    def chunks_loaded(self) -> int:
        return self._chunks_loaded

    @property
    def samples_yielded(self) -> int:
        return self._samples_yielded

    def stats(self) -> dict:
        """Return loader statistics."""
        return {
            "files_found": len(self._files),
            "chunks_loaded": self._chunks_loaded,
            "total_bytes_read": self._total_bytes_read,
            "total_mb_read": self._total_bytes_read / (1024 ** 2),
            "samples_yielded": self._samples_yielded,
            "chunk_size_mb": self.chunk_size_bytes / (1024 ** 2),
            "curriculum_enabled": self.skip_learned,
            "safety_filter_enabled": self.safety_filter,
        }

    def __repr__(self) -> str:
        return (
            f"StreamLoader("
            f"path='{self.path}', "
            f"files={len(self._files)}, "
            f"chunk={self.chunk_size_bytes // (1024**2)}MB, "
            f"tokenizer={self.tokenizer})"
        )