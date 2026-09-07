# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Fractal Tensor Engine
#
# ==========================================================

"""
Fractal Tensor implementation for Neura-X.

Fractal Tensors store weight matrices as tiny mathematical formulas
(Fractal Seeds) instead of dense numerical arrays, reducing memory
by up to 4,000x.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class FractalSeed:
    """
    The tiny parameter set that generates a Fractal Tensor.

    Instead of storing m × n weights, we store 4K numbers:
    - a: amplitudes (K numbers)
    - w1: row frequencies (K numbers)
    - w2: column frequencies (K numbers)
    - phi: phase shifts (K numbers)

    Total: 4K numbers instead of m × n numbers.
    """
    a: np.ndarray      # Amplitudes
    w1: np.ndarray     # Row frequencies
    w2: np.ndarray     # Column frequencies
    phi: np.ndarray    # Phase shifts

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=np.float32)
        self.w1 = np.asarray(self.w1, dtype=np.float32)
        self.w2 = np.asarray(self.w2, dtype=np.float32)
        self.phi = np.asarray(self.phi, dtype=np.float32)

    @property
    def K(self) -> int:
        """Number of harmonic components."""
        return len(self.a)

    @property
    def num_parameters(self) -> int:
        """Total number of seed parameters."""
        return 4 * self.K

    @property
    def memory_bytes(self) -> int:
        """Memory usage in bytes."""
        return self.num_parameters * 4  # float32 = 4 bytes

    @classmethod
    def random(cls, K: int = 1024, seed: Optional[int] = None) -> 'FractalSeed':
        """Create a random Fractal Seed (Genesis Initialization)."""
        rng = np.random.default_rng(seed)
        return cls(
            a=rng.normal(0, 0.02, size=K).astype(np.float32),
            w1=rng.uniform(-np.pi, np.pi, size=K).astype(np.float32),
            w2=rng.uniform(-np.pi, np.pi, size=K).astype(np.float32),
            phi=rng.uniform(0, 2 * np.pi, size=K).astype(np.float32),
        )

    @classmethod
    def from_hash(cls, hash_string: str, K: int = 1024) -> 'FractalSeed':
        """
        Create a Fractal Seed from a cryptographic hash.
        Used for the Founder's Watermark.
        """
        import hashlib
        hash_bytes = hashlib.sha256(hash_string.encode()).digest()
        seed_int = int.from_bytes(hash_bytes[:4], 'big')
        return cls.random(K=K, seed=seed_int)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "a": self.a.tolist(),
            "w1": self.w1.tolist(),
            "w2": self.w2.tolist(),
            "phi": self.phi.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FractalSeed':
        """Deserialize from dictionary."""
        return cls(
            a=np.array(d["a"], dtype=np.float32),
            w1=np.array(d["w1"], dtype=np.float32),
            w2=np.array(d["w2"], dtype=np.float32),
            phi=np.array(d["phi"], dtype=np.float32),
        )


class FractalTensor:
    """
    A tensor represented as a Fractal Seed rather than dense numbers.

    The full weight matrix is never stored. Instead, it is generated
    (bloomed) on demand from the Fractal Seed, used for computation,
    and then discarded.

    Args:
        rows: Number of rows in the conceptual matrix
        cols: Number of columns in the conceptual matrix
        seed: The Fractal Seed
        K: Number of harmonic components (used if seed is None)
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        seed: Optional[FractalSeed] = None,
        K: int = 1024,
    ):
        self.rows = rows
        self.cols = cols
        self.K = K

        if seed is not None:
            self.seed = seed
        else:
            self.seed = FractalSeed.random(K=K)

        # Cache for recently bloomed rows (small, bounded)
        self._bloom_cache: dict = {}
        self._cache_max_size = 16  # Cache at most 16 rows

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def conceptual_size(self) -> int:
        """Total number of conceptual elements."""
        return self.rows * self.cols

    @property
    def actual_memory_bytes(self) -> int:
        """Actual memory used by the Fractal Seed."""
        return self.seed.memory_bytes

    @property
    def conceptual_memory_bytes(self) -> int:
        """Memory that would be needed for a dense matrix."""
        return self.rows * self.cols * 4  # float32

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs dense storage."""
        if self.actual_memory_bytes == 0:
            return float('inf')
        return self.conceptual_memory_bytes / self.actual_memory_bytes

    def bloom_element(self, i: int, j: int) -> float:
        """
        Generate a single element W(i,j) from the Fractal Seed.

        W(i,j) = Σ(k=1 to K) a_k × sin(w1_k × i + w2_k × j + φ_k)
        """
        result = 0.0
        for k in range(self.K):
            result += float(
                self.seed.a[k] * np.sin(
                    self.seed.w1[k] * i + self.seed.w2[k] * j + self.seed.phi[k]
                )
            )
        return result

    def bloom_row(self, i: int) -> np.ndarray:
        """Generate a full row of the weight matrix."""
        if i in self._bloom_cache:
            return self._bloom_cache[i]

        j_indices = np.arange(self.cols, dtype=np.float32)
        row = np.zeros(self.cols, dtype=np.float32)

        for k in range(self.K):
            row += self.seed.a[k] * np.sin(
                self.seed.w1[k] * i + self.seed.w2[k] * j_indices + self.seed.phi[k]
            )

        # Cache management
        if len(self._bloom_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._bloom_cache))
            del self._bloom_cache[oldest_key]
        self._bloom_cache[i] = row

        return row

    def bloom_block(self, row_start: int, row_end: int) -> np.ndarray:
        """Generate a block of rows."""
        rows = []
        for i in range(row_start, min(row_end, self.rows)):
            rows.append(self.bloom_row(i))
        return np.stack(rows, axis=0)

    def bloom_all(self) -> np.ndarray:
        """
        Generate the entire weight matrix.
        WARNING: This defeats the purpose of Fractal Tensors for large matrices.
        Use bloom_row() or bloom_block() for memory-efficient access.
        """
        return self.bloom_block(0, self.rows)

    def matmul(self, x: np.ndarray, block_size: int = 64) -> np.ndarray:
        """
        Memory-efficient matrix multiplication: W × x

        Instead of blooming the entire W matrix, we bloom one block at a
        time, multiply, and accumulate. This keeps memory usage bounded.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        output = np.zeros((x.shape[0], self.rows), dtype=np.float32)

        for start in range(0, self.rows, block_size):
            end = min(start + block_size, self.rows)
            W_block = self.bloom_block(start, end)
            output[:, start:end] = x @ W_block.T

        return output

    def __repr__(self) -> str:
        return (
            f"FractalTensor("
            f"shape={self.shape}, "
            f"K={self.K}, "
            f"seed_params={self.seed.num_parameters}, "
            f"memory={self.actual_memory_bytes / 1024:.1f}KB, "
            f"conceptual={self.conceptual_memory_bytes / (1024**2):.1f}MB, "
            f"compression={self.compression_ratio:.0f}x)"
        )