# ==========================================================
# Neura-X Tests: Fractal Tensor
# ==========================================================

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.fractal_layer import FractalSeed, FractalTensor


def test_fractal_seed_creation():
    """Test that a Fractal Seed can be created."""
    seed = FractalSeed.random(K=1024, seed=42)
    assert seed.K == 1024
    assert seed.num_parameters == 4096
    assert seed.memory_bytes == 4096 * 4
    print("✅ test_fractal_seed_creation passed")


def test_fractal_tensor_bloom():
    """Test that a Fractal Tensor can bloom elements."""
    ft = FractalTensor(rows=100, cols=100, K=64)
    element = ft.bloom_element(0, 0)
    assert isinstance(element, float)
    print("✅ test_fractal_tensor_bloom passed")


def test_fractal_tensor_bloom_row():
    """Test blooming a full row."""
    ft = FractalTensor(rows=10, cols=50, K=32)
    row = ft.bloom_row(0)
    assert row.shape == (50,)
    print("✅ test_fractal_tensor_bloom_row passed")


def test_fractal_compression():
    """Test that compression ratio is significant."""
    ft = FractalTensor(rows=4096, cols=4096, K=1024)
    assert ft.compression_ratio > 100
    print(f"✅ test_fractal_compression passed (ratio: {ft.compression_ratio:.0f}x)")


def test_fractal_matmul():
    """Test memory-efficient matrix multiplication."""
    ft = FractalTensor(rows=64, cols=32, K=16)
    x = np.random.randn(1, 32).astype(np.float32)
    result = ft.matmul(x)
    assert result.shape == (1, 64)
    print("✅ test_fractal_matmul passed")


def test_founder_seed():
    """Test Founder's Watermark seed generation."""
    seed = FractalSeed.from_hash("Edusei Mikel Lisamba", K=1024)
    assert seed.K == 1024
    print("✅ test_founder_seed passed")


if __name__ == "__main__":
    test_fractal_seed_creation()
    test_fractal_tensor_bloom()
    test_fractal_tensor_bloom_row()
    test_fractal_compression()
    test_fractal_matmul()
    test_founder_seed()
    print("\n🎉 All Fractal Tensor tests passed!")