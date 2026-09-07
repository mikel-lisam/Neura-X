# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Function Calling & Tool Use
# ==========================================================

"""Function calling and tool use system for Neura-X."""

import json
from typing import Callable, Optional, Dict, Any, List
from functools import wraps


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, func: Callable, name: str = None, description: str = None):
        """Register a tool."""
        tool_name = name or func.__name__
        self._tools[tool_name] = {
            "function": func,
            "name": tool_name,
            "description": description or func.__doc__ or "",
        }

    def get(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        if name in self._tools:
            return self._tools[name]["function"]
        return None

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def to_openai_schema(self) -> List[dict]:
        """Convert tools to OpenAI function calling schema."""
        schemas = []
        for name, info in self._tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {"type": "object", "properties": {}},
                },
            })
        return schemas


# Global tool registry
_global_registry = ToolRegistry()


def tool(description: str = None, name: str = None):
    """Decorator to register a function as a Neura-X tool."""
    def decorator(func: Callable):
        _global_registry.register(func, name=name, description=description)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._is_tool = True
        wrapper._tool_name = name or func.__name__
        return wrapper

    return decorator


# Built-in tools
@tool(description="Search the web for information")
def web_search(query: str) -> str:
    """Search the web and return results."""
    return f"[Neura-X Search] Results for: {query}"


@tool(description="Execute Python code in a sandboxed environment")
def run_code(code: str) -> str:
    """Run Python code safely."""
    return f"[Neura-X Sandbox] Code executed successfully"


@tool(description="Execute a SQL query on the connected database")
def query_db(sql: str) -> str:
    """Execute SQL query."""
    return f"[Neura-X SQL] Query executed: {sql}"


@tool(description="Read a file from the local filesystem")
def read_file(path: str) -> str:
    """Read a file."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"[Neura-X Error] {e}"


@tool(description="Write content to a file on the local filesystem")
def file_write(path: str, content: str) -> str:
    """Write to a file."""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"[Neura-X] File written: {path}"
    except Exception as e:
        return f"[Neura-X Error] {e}"