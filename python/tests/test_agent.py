# ==========================================================
# Neura-X Tests: Agent System
# ==========================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.agent import Agent, AgentTeam, HolographicMemory
from neura_x.module import FractalLayer


def test_holographic_memory():
    """Test Holographic Memory store and recall."""
    mem = HolographicMemory(capacity=1000, D=1000)
    mem.store(concept="test", data="test data", importance=0.8)

    result = mem.recall("test")
    assert result is not None
    assert mem.size == 1
    print("✅ test_holographic_memory passed")


def test_agent_creation():
    """Test Agent creation."""
    brain = FractalLayer(in_features=10, out_features=10)
    agent = Agent(brain=brain, skills=["swahili"])
    assert agent.brain is not None
    assert "swahili" in agent.skills
    print("✅ test_agent_creation passed")


def test_agent_run():
    """Test Agent task execution."""
    brain = FractalLayer(in_features=10, out_features=10)
    agent = Agent(brain=brain)
    result = agent.run("Test task")
    assert "Completed" in result
    print("✅ test_agent_run passed")


if __name__ == "__main__":
    test_holographic_memory()
    test_agent_creation()
    test_agent_run()
    print("\n🎉 All Agent tests passed!")