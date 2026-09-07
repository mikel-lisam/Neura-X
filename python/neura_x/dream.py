# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Dream Dashboard
# ==========================================================

"""Dream Dashboard: real-time telemetry for the Circadian wake-sleep cycle.

The dashboard is hooked into a :class:`neura_x.optimizer.CircadianOptimizer`
so every wake/sleep event is logged automatically.  Callers can:

* :py:meth:`DreamDashboard.attach` — connect to an optimiser.
* :py:meth:`DreamDashboard.detach` — disconnect.
* :py:meth:`DreamDashboard.record_wake` / :py:meth:`DreamDashboard.record_sleep`
  — manual instrumentation.
* :py:meth:`DreamDashboard.summary` — produce a wake/sleep summary
  suitable for visualisation or CI telemetry.
* :py:meth:`DreamDashboard.as_dict` — export all state as JSON.
"""

import time
from typing import Any, Dict, List, Optional


class DreamDashboard:
    """Visual dashboard for the Circadian Wake-Sleep cycle."""

    def __init__(self, model: Any = None):
        self.model = model
        self._sleep_log: List[Dict[str, Any]] = []
        self._wake_log: List[Dict[str, Any]] = []
        self._start_time = time.time()
        self._attached = None  # the CircadianOptimizer we are attached to.

    # ─────────────────────────────────────────────────────────────────
    # Automatic instrumentation
    # ─────────────────────────────────────────────────────────────────

    def attach(self, optimizer: Any) -> None:
        """Hook into a :class:`CircadianOptimizer` and record its events.

        ``optimizer.wake_step`` and ``optimizer.sleep`` are wrapped to
        capture each call into :py:attr:`wake_log` and
        :py:attr:`sleep_log` respectively.
        """
        if optimizer is self._attached:
            return

        original_wake = optimizer.wake_step
        original_sleep = optimizer.sleep
        dashboard = self

        def wake_wrapper(*args, **kwargs):
            result = original_wake(*args, **kwargs)
            dashboard.record_wake({
                "args_count": len(args),
                "kwargs": {k: type(v).__name__ for k, v in kwargs.items()},
            })
            return result

        def sleep_wrapper(*args, **kwargs):
            result = original_sleep(*args, **kwargs)
            summary = result if isinstance(result, dict) else {"ok": True}
            dashboard.record_sleep(summary)
            return result

        optimizer.wake_step = wake_wrapper
        optimizer.sleep = sleep_wrapper
        self._attached = optimizer

    def detach(self) -> None:
        """No-op for now; included for symmetry with future implementations."""
        self._attached = None

    # ─────────────────────────────────────────────────────────────────
    # Manual instrumentation
    # ─────────────────────────────────────────────────────────────────

    def record_wake(self, summary: Dict[str, Any]) -> None:
        """Append a wake-event summary to the dashboard log."""
        entry = dict(summary)
        entry["timestamp"] = time.time() - self._start_time
        entry["phase"] = "wake"
        self._wake_log.append(entry)

    def record_sleep(self, summary: Dict[str, Any]) -> None:
        """Append a sleep-cycle summary to the dashboard log."""
        entry = dict(summary)
        entry["timestamp"] = time.time() - self._start_time
        entry["phase"] = "sleep"
        self._sleep_log.append(entry)

    # ─────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────

    def open(self) -> Dict[str, Any]:
        """Open the Dream Dashboard and print a textual summary."""
        snapshot = self.summary()
        print("🌙 Neura-X Dream Dashboard")
        print("─" * 40)
        print(f"Wake events:   {snapshot['wake_count']}")
        print(f"Sleep events:  {snapshot['sleep_count']}")
        print(f"Elapsed:       {snapshot['elapsed_seconds']:.1f}s")
        if snapshot["avg_surprises_per_sleep"] is not None:
            print(f"Avg surprises: {snapshot['avg_surprises_per_sleep']:.1f} per sleep")
        print("─" * 40)
        return snapshot

    def summary(self) -> Dict[str, Any]:
        """Compute a summary suitable for visualisation or CI telemetry."""
        total_wake = len(self._wake_log)
        total_sleep = len(self._sleep_log)
        avg_surprises = None
        if total_sleep:
            surprises = [s.get("consolidated_surprises", 0) for s in self._sleep_log]
            avg_surprises = sum(surprises) / max(len(surprises), 1)
        return {
            "phases": {
                "wake": "Logging surprise vectors",
                "sleep": "Consolidating knowledge",
            },
            "wake_count": total_wake,
            "sleep_count": total_sleep,
            "avg_surprises_per_sleep": avg_surprises,
            "wake_events": list(self._wake_log),
            "sleep_events": list(self._sleep_log),
            "elapsed_seconds": time.time() - self._start_time,
        }

    def as_dict(self) -> Dict[str, Any]:
        """Return the dashboard state without printing (alias for :py:meth:`summary`)."""
        return self.summary()

    @classmethod
    def from_model(cls, model: Any) -> "DreamDashboard":
        return cls(model=model)

    @property
    def wake_log(self) -> List[Dict[str, Any]]:
        return list(self._wake_log)

    @property
    def sleep_log(self) -> List[Dict[str, Any]]:
        return list(self._sleep_log)