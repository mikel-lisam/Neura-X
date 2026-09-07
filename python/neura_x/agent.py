# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Agentic Systems
# ==========================================================

"""Autonomous AI agents for Neura-X."""

import hashlib
import re
from collections import Counter
from typing import List, Optional, Dict, Any
import numpy as np
from neura_x.module import Module


def _split_words(text: str) -> List[str]:
    return [w for w in re.split(r"\W+", text.lower()) if w]


def _hash_to_hypervector(seed_bytes: bytes, D: int) -> "np.ndarray":
    """Map a deterministic byte stream to a bipolar hypervector of length D."""
    bits: List[int] = []
    counter = 0
    while len(bits) < D:
        h = hashlib.sha512(seed_bytes + counter.to_bytes(8, "big")).digest()
        for byte in h:
            for shift in range(8):
                if len(bits) >= D:
                    break
                bits.append(1 if (byte >> shift) & 1 else -1)
        counter += 1
    return np.asarray(bits[:D], dtype=np.int8)


def _bind(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    return (a * b).astype(np.int8)


class HolographicMemory:
    """Holographic Memory using content-addressed bipolar hypervectors."""

    def __init__(self, capacity: int = 10_000, D: int = 10_000):
        self.capacity = capacity
        self.D = D
        self._memories: Dict[str, np.ndarray] = {}
        self._importance: Dict[str, float] = {}
        self._data: Dict[str, str] = {}

    def _item_memory(self, role: str) -> "np.ndarray":
        return _hash_to_hypervector(f"role:{role}".encode("utf-8"), self.D)

    def store(self, concept: str, data: str, importance: float = 0.5):
        """Store a memory with content-derived hypervector."""
        concept_hv = _hash_to_hypervector(f"concept:{concept}".encode("utf-8"), self.D)
        role_hv = self._item_memory("memory")
        hv = _bind(concept_hv, role_hv)
        if len(self._memories) >= self.capacity:
            evict_key = min(self._importance, key=self._importance.get)
            self._memories.pop(evict_key, None)
            self._importance.pop(evict_key, None)
            self._data.pop(evict_key, None)
        self._memories[concept] = hv
        self._importance[concept] = float(importance)
        self._data[concept] = str(data)

    def recall(self, query: str) -> Optional[str]:
        """Recall a memory by exact key or by Hamming-similarity search."""
        if query in self._memories:
            imp = self._importance.get(query, 0.0)
            return f"[Memory: {query}] importance={imp:.2f} data={self._data.get(query, '')[:120]}"

        if not self._memories:
            return None
        query_hv = _hash_to_hypervector(f"concept:{query}".encode("utf-8"), self.D)
        best_key = None
        best_score = -1.0
        for key, hv in self._memories.items():
            score = float(np.mean(query_hv * hv))
            if score > best_score:
                best_score = score
                best_key = key
        if best_key is None or best_score <= 0.0:
            return None
        imp = self._importance.get(best_key, 0.0)
        return (
            f"[Memory: {best_key}] similarity={best_score:.3f} "
            f"importance={imp:.2f} data={self._data.get(best_key, '')[:120]}"
        )

    @property
    def size(self) -> int:
        return len(self._memories)


# Mapping of action verbs to their natural-language tool equivalents.
_ACTION_VERBS = {
    "search": "web_search",
    "find": "web_search",
    "look up": "web_search",
    "look": "web_search",
    "compute": "calculator",
    "calculate": "calculator",
    "calc": "calculator",
    "read": "read_file",
    "open": "read_file",
    "write": "file_write",
    "save": "file_write",
    "store": "file_write",
    "execute": "run_code",
    "run": "run_code",
    "code": "run_code",
    "query": "query_db",
    "sql": "query_db",
}


def _extract_plan(task: str, max_steps: int = 6) -> List[str]:
    """Decompose ``task`` into ordered executable steps."""
    words = _split_words(task)
    if not words:
        return ["verify"]

    matched_verb = None
    for verb, tool_name in _ACTION_VERBS.items():
        for w in words:
            if w == verb or verb.startswith(w + " ") or w in verb.split():
                matched_verb = tool_name
                break
        if matched_verb:
            break

    plan: List[str] = [matched_verb or "analyze"]
    if any(re.search(r"\d", w) for w in words):
        if "calculator" not in plan:
            plan.append("calculator")
    plan.append("verify")
    return plan[:max_steps]


class Agent:
    """Autonomous AI agent with perception, planning, action, and memory."""

    def __init__(
        self,
        brain: Any,
        skills: List[str] = None,
        tools: List[Any] = None,
        memory: HolographicMemory = None,
        circadian: bool = True,
    ):
        self.brain = brain
        self.skills = skills or []
        self.tools = tools or []
        self.memory = memory or HolographicMemory()
        self.circadian = circadian
        self._action_log: List[Dict[str, Any]] = []

    def run(self, task: str) -> str:
        """Execute a task autonomously."""
        self._action_log.append({"action": "plan", "task": task})

        plan = self._plan(task)
        self._action_log.append({"action": "planned", "steps": plan})

        results: List[str] = []
        for step in plan:
            result = self._execute_step(step, task)
            results.append(result)

        if self.circadian:
            self._reflect(task, results)

        summary = "; ".join(results)
        return f"[Agent] Completed task: {task} (steps={len(plan)}): {summary}"

    def _plan(self, task: str) -> List[str]:
        return _extract_plan(task)

    def _execute_step(self, step: str, task: str) -> str:
        try:
            import neura_x.tools as _tools
            tool_fn = _tools._global_registry.get(step)
            if tool_fn is not None:
                return str(tool_fn(task))
        except Exception:
            pass
        return f"step:{step}"

    def _reflect(self, task: str, results: List[str]):
        if not results:
            importance = 0.1
        else:
            importance = sum(1 for r in results if r and r.strip()) / len(results)
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
        self.memory.store(
            concept=f"task:{digest}",
            data=str(results),
            importance=float(importance),
        )


class AgentTeam:
    """A team of coordinated agents."""

    def __init__(self, agents: List[Agent], coordinator: Any = None):
        self.agents = agents
        self.coordinator = coordinator

    def run(self, task: str) -> str:
        """Distribute the task across agents and merge results."""
        results: List[str] = []
        for agent in self.agents:
            results.append(agent.run(task))

        if self.coordinator is not None and callable(self.coordinator):
            try:
                merged = self.coordinator(results)
                if isinstance(merged, str) and merged.strip():
                    return merged
            except Exception:
                pass

        return "\n".join(results)