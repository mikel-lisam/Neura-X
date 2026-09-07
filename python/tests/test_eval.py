# ==========================================================
# Neura-X Tests: Evaluation Suite
# ==========================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.eval import EvalSuite


def test_eval_suite_creation():
    """Test EvalSuite initialization."""
    suite = EvalSuite()
    assert len(suite.SUPPORTED_BENCHMARKS) == 7
    print("✅ test_eval_suite_creation passed")


def test_eval_run():
    """Test running evaluation."""
    suite = EvalSuite()
    results = suite.run(model=None, benchmarks=["mmlu", "gsm8k"])
    assert "mmlu" in results
    assert "gsm8k" in results
    print("✅ test_eval_run passed")


if __name__ == "__main__":
    test_eval_suite_creation()
    test_eval_run()
    print("\n🎉 All Eval tests passed!")