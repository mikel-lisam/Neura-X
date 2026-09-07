# ==========================================================
# Neura-X Tests: Circadian Optimizer
# ==========================================================

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.module import Module, Parameter
from neura_x.optimizer import CircadianOptimizer


def test_circadian_creation():
    """Test CircadianOptimizer initialization."""
    model = Module()
    opt = CircadianOptimizer(model, sleep_interval=10)
    assert opt.sleep_interval == 10
    print("✅ test_circadian_creation passed")


def test_circadian_wake():
    """Test Wake phase logs surprise vectors."""
    model = Module()
    opt = CircadianOptimizer(model, sleep_interval=5)

    hidden = np.random.randn(10).astype(np.float32)
    opt.wake_step(loss_value=0.5, hidden_states=hidden)

    assert len(opt.surprise_cache) == 1
    print("✅ test_circadian_wake passed")


def test_circadian_sleep():
    """Test Sleep phase consolidates surprises."""
    model = Module()
    opt = CircadianOptimizer(model, sleep_interval=5)

    # Log some surprises
    for i in range(5):
        hidden = np.random.randn(10).astype(np.float32)
        opt.wake_step(loss_value=0.5, hidden_states=hidden)

    # Sleep should have triggered
    assert opt.total_sleeps >= 1
    assert len(opt.surprise_cache) == 0  # Cache cleared after sleep
    print("✅ test_circadian_sleep passed")


if __name__ == "__main__":
    test_circadian_creation()
    test_circadian_wake()
    test_circadian_sleep()
    print("\n🎉 All Circadian tests passed!")