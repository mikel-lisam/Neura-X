# ==========================================================
# Neura-X Tests: Tool System
# ==========================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neura_x.tools import tool, ToolRegistry, web_search, run_code


def test_tool_decorator():
    """Test the @tool decorator."""
    @tool(description="Test tool")
    def my_tool(x: str) -> str:
        return f"Result: {x}"

    assert hasattr(my_tool, '_is_tool')
    assert my_tool._is_tool is True
    print("✅ test_tool_decorator passed")


def test_tool_execution():
    """Test executing a tool."""
    result = web_search("test query")
    assert "test query" in result
    print("✅ test_tool_execution passed")


def test_tool_registry():
    """Test the tool registry."""
    registry = ToolRegistry()

    @tool(description="Registry test")
    def registry_tool():
        return "ok"

    registry.register(registry_tool, name="test_tool")
    assert "test_tool" in registry.list_tools()
    print("✅ test_tool_registry passed")


if __name__ == "__main__":
    test_tool_decorator()
    test_tool_execution()
    test_tool_registry()
    print("\n🎉 All Tool tests passed!")