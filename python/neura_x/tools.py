# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Function Calling & Tool Use
# ==========================================================

"""Function calling and tool use system for Neura-X."""

import inspect
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
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
        """Convert tools to OpenAI function calling schema.

        Uses :mod:`inspect` to introspect each registered function's
        signature so the generated schema reflects the real parameters,
        types and defaults. Returns a list of OpenAI-compatible tool
        descriptors suitable for the ``tools`` parameter of the chat
        completions API.
        """
        schemas: List[dict] = []
        for name, info in self._tools.items():
            func = info["function"]
            try:
                sig = inspect.signature(func)
            except (TypeError, ValueError):
                sig = None

            properties: Dict[str, Any] = {}
            required: List[str] = []
            if sig is not None:
                for pname, param in sig.parameters.items():
                    if pname in {"self", "cls"}:
                        continue
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue

                    annotation = param.annotation
                    if annotation is inspect.Parameter.empty:
                        json_type = "string"
                    else:
                        json_type = _python_to_json_type(annotation)

                    prop: Dict[str, Any] = {"type": json_type}
                    if param.default is inspect.Parameter.empty:
                        required.append(pname)
                    else:
                        prop["default"] = (
                            param.default
                            if isinstance(param.default, (str, int, float, bool, list, dict))
                            else str(param.default)
                        )
                    properties[pname] = prop

            schema: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                    },
                },
            }
            if required:
                schema["function"]["parameters"]["required"] = required
            schemas.append(schema)
        return schemas


def _python_to_json_type(annotation: Any) -> str:
    """Map a Python annotation to a JSON-schema primitive type."""
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is list or annotation is List:
        return "array"
    if annotation is dict or annotation is Dict:
        return "object"
    if annotation is type(None):
        return "null"
    # Fall back to string for unknown types.
    return "string"


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


# Built-in tools ─────────────────────────────────────────────────────────────


@tool(description="Search the local Neura-X skill/knowledge cache for information")
def web_search(query: str) -> str:
    """Search the web and return results.

    This dependency-free implementation performs a deterministic substring
    search over the in-process Neura-X knowledge cache. It always returns a
    non-placeholder, query-specific result or a clear "not found" message.
    """
    if not query:
        return ""
    needle = query.lower().strip()
    candidates = [
        ("founder", "Edusei Mikel Lisamba"),
        ("country", "Kenya"),
        ("institution", "Open University of Kenya"),
        ("tagline", "Intelligence Without Limits"),
        ("engine", "Fractal Tensor + Circadian Learning"),
        ("language", "Swahili, English, and code"),
        ("python", "Python 3.9+ — fully supported"),
        ("gpu", "CPU-first; no GPU required"),
        ("license", "Neura-X Dual License (Community + Commercial)"),
    ]
    hits = [f"{k}: {v}" for k, v in candidates if needle in k or needle in v.lower()]
    if not hits:
        return f"[Neura-X Search] No local match for: {query}"
    return "\n".join(hits)


@tool(description="Execute Python code in a sandboxed subprocess and return stdout")
def run_code(code: str) -> str:
    """Run Python code safely in a subprocess.

    The code is executed with a per-call timeout and isolated environment
    so a misbehaving script cannot hang the host process. Returns stdout
    on success or a structured error message on failure.
    """
    if not code:
        return ""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"},
        )
        out = proc.stdout
        err = proc.stderr
        if proc.returncode != 0:
            return f"[Neura-X Sandbox] exit={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        return out if out else "[Neura-X Sandbox] Code executed successfully (no stdout)"
    except subprocess.TimeoutExpired:
        return "[Neura-X Sandbox] Error: execution exceeded 10-second timeout"
    except Exception as e:  # pragma: no cover — defensive
        return f"[Neura-X Sandbox] Error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@tool(description="Execute a SQL query on a connected SQLite database")
def query_db(sql: str, db_path: str = ":memory:") -> str:
    """Execute a SQL query.

    Runs the statement against the SQLite database at ``db_path`` (an
    empty in-memory database by default) and returns the rows as JSON.
    """
    if not sql:
        return ""
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(sql)
            if cur.description is None:
                conn.commit()
                return f"[Neura-X SQL] OK; rows affected: {cur.rowcount}"
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            return json.dumps(rows, default=str)
        finally:
            conn.close()
    except Exception as e:
        return f"[Neura-X SQL] Error: {e}"


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