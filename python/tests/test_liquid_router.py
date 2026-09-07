# ==========================================================
# Neura-X Tests: Liquid Router
# ==========================================================

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.liquid_router import LiquidRouter


def test_router_creation():
    """Test LiquidRouter initialization."""
    router = LiquidRouter(num_neurons=1000, awake_threshold=0.05)
    assert router.num_neurons == 1000
    assert router.awake_threshold == 0.05
    print("✅ test_router_creation passed")


def test_router_routing():
    """Test that routing produces correct sparsity."""
    router = LiquidRouter(num_neurons=1000, awake_threshold=0.05)
    x = np.random.randn(1000).astype(np.float32)
    mask = router.route(x, hard=True)

    assert mask.shape == (1000,)
    assert mask.sum() == 50  # 5% of 1000
    print("✅ test_router_routing passed")


def test_router_sparsity():
    """Test sparsity tracking."""
    router = LiquidRouter(num_neurons=100, awake_threshold=0.1)
    x = np.random.randn(100).astype(np.float32)

    for _ in range(10):
        router.route(x, hard=True)

    savings = router.memory_savings()
    assert savings["sparsity_percent"] > 80
    print(f"✅ test_router_sparsity passed (sparsity: {savings['sparsity_percent']:.1f}%)")


if __name__ == "__main__":
    test_router_creation()
    test_router_routing()
    test_router_sparsity()
    print("\n🎉 All Liquid Router tests passed!")