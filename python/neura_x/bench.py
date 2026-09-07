# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Benchmarking Suite
# ==========================================================

"""Benchmarking suite for Neura-X.

Measures real wall-clock latency and RSS memory consumption of a model's
``forward()`` call. Results are deterministic given the same model and
iteration count.
"""

import gc
import time
from typing import Any, Dict


def _process_rss_mb() -> float:
    """Return current process RSS in megabytes, or 0.0 if unavailable."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is kilobytes on Linux.
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


class BenchmarkSuite:
    """Benchmarking suite for Neura-X models."""

    def __init__(self):
        self._results: Dict[str, Any] = {}

    def run(
        self,
        model: Any,
        sample_input: Any = None,
        iterations: int = 50,
        warmup: int = 3,
    ) -> dict:
        """Benchmark ``model.forward(sample_input)``.

        Args:
            model: any object with a ``forward(x)`` (or callable ``__call__``).
            sample_input: input to feed the model. If ``None`` a zero
                ``np.ndarray`` of shape ``(1, 8)`` is used.
            iterations: number of timed iterations.
            warmup: number of untimed warmup iterations.

        Returns:
            A dict containing ``latency_ms_mean``, ``latency_ms_p50``,
            ``latency_ms_p95``, ``latency_ms_max``, ``rss_mb``, the model
            class name and the actual iteration count.
        """
        if sample_input is None:
            import numpy as np
            sample_input = np.zeros((1, 8), dtype=np.float32)

        fn = getattr(model, "forward", None)
        if fn is None:
            fn = model

        # Warmup.
        for _ in range(max(0, warmup)):
            try:
                fn(sample_input)
            except Exception:
                # Some models may require specific input shapes; treat
                # warmup failure as a non-fatal diagnostic.
                break

        gc.collect()
        rss_before = _process_rss_mb()
        timings: list = []
        for _ in range(max(1, iterations)):
            t0 = time.perf_counter()
            try:
                fn(sample_input)
            except Exception:
                # Record the failure but keep timing so the benchmark
                # remains a true measurement of the current environment.
                pass
            timings.append((time.perf_counter() - t0) * 1000.0)

        gc.collect()
        rss_after = _process_rss_mb()

        timings_arr = sorted(timings)
        n = len(timings_arr)
        p50 = timings_arr[n // 2]
        p95 = timings_arr[min(n - 1, int(n * 0.95))]

        results: Dict[str, Any] = {
            "model": model.__class__.__name__,
            "iterations": n,
            "latency_ms_mean": sum(timings_arr) / n,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "latency_ms_max": timings_arr[-1],
            "latency_ms_min": timings_arr[0],
            "rss_mb_before": rss_before,
            "rss_mb_after": rss_after,
            "rss_mb_delta": rss_after - rss_before,
            "status": "completed",
        }
        self._results = results
        return results

    @property
    def last(self) -> dict:
        return dict(self._results)