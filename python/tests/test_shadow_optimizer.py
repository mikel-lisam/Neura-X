# ==========================================================
# Neura-X Tests: Shadow Optimizer
# ==========================================================

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.module import Parameter
from neura_x.optimizer import ShadowAdamW


def test_shadow_adamw_creation():
    """Test ShadowAdamW initialization."""
    params = [Parameter(np.random.randn(100, 100).astype(np.float32))]
    opt = ShadowAdamW(params, lr=1e-3, rank=16)
    assert len(opt.params) == 1
    assert opt.rank == 16
    print("✅ test_shadow_adamw_creation passed")


def test_shadow_adamw_step():
    """Test that ShadowAdamW performs an optimization step."""
    params = [Parameter(np.random.randn(50, 50).astype(np.float32))]
    opt = ShadowAdamW(params, lr=0.01, rank=8)

    # Simulate gradient
    params[0].grad = np.random.randn(50, 50).astype(np.float32)

    old_data = params[0].data.copy()
    opt.step()

    # Parameters should have changed
    assert not np.allclose(params[0].data, old_data)
    print("✅ test_shadow_adamw_step passed")


def test_shadow_memory_savings():
    """Test that Shadow Optimizer uses less memory."""
    params = [Parameter(np.random.randn(1000, 1000).astype(np.float32))]
    opt = ShadowAdamW(params, lr=1e-3, rank=64)

    mem = opt.memory_usage()
    assert mem["shadow_bytes"] < mem["standard_bytes"]
    assert mem["reduction_factor"] > 10
    print(f"✅ test_shadow_memory_savings passed (reduction: {mem['reduction_factor']:.0f}x)")


if __name__ == "__main__":
    test_shadow_adamw_creation()
    test_shadow_adamw_step()
    test_shadow_memory_savings()
    print("\n🎉 All Shadow Optimizer tests passed!")