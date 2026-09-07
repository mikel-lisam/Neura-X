# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# X-Ray Debugging & Telemetry
# ==========================================================

"""X-Ray: debugging and telemetry for Neura-X models.

Reports on a model's structure, parameter inventory, and forward-pass
profile. All measurements are computed against the live model object.
"""

import gc
import time
from typing import Any, Dict


def _forward_profile(model: Any, sample: Any) -> Dict[str, float]:
    """Run one forward pass and measure wall-clock + RSS impact."""
    fn = getattr(model, "forward", None) or model
    try:
        import resource
        rss_before = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        rss_before = 0.0
    t0 = time.perf_counter()
    try:
        fn(sample)
    except Exception as e:
        return {"error": f"forward failed: {e}", "latency_ms": (time.perf_counter() - t0) * 1000.0}
    latency_ms = (time.perf_counter() - t0) * 1000.0
    try:
        import resource
        rss_after = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        rss_after = 0.0
    gc.collect()
    return {
        "latency_ms": latency_ms,
        "rss_mb": max(rss_after, rss_before),
        "rss_delta_mb": rss_after - rss_before,
    }


def xray(model: Any, sample_input: Any = None) -> Dict[str, Any]:
    """
    Inspect a Neura-X model and return a telemetry report.

    Args:
        model: any object with a ``forward(x)`` method.
        sample_input: optional sample input. Defaults to a zero
            ``np.ndarray`` of shape ``(1, 8)``.

    Returns:
        Dict containing model class, parameter count, forward-pass
        latency, RSS memory, and module inventory.
    """
    if sample_input is None:
        try:
            import numpy as np
            sample_input = np.zeros((1, 8), dtype=np.float32)
        except Exception:
            sample_input = None

    report: Dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "training_mode": getattr(model, "training", None),
    }

    if hasattr(model, "parameter_count"):
        try:
            report["parameter_count"] = int(model.parameter_count())
        except Exception:
            pass

    if hasattr(model, "_modules"):
        modules = [m.__class__.__name__ for m in model._modules.values()]
        report["submodules"] = modules
        report["submodule_count"] = len(modules)

    if hasattr(model, "_parameters"):
        params = list(model._parameters.keys())
        report["top_level_params"] = params

    report["forward_profile"] = _forward_profile(model, sample_input) if sample_input is not None else {"skipped": True}

    print("🔍 Neura-X X-Ray")
    print("─" * 40)
    print(f"Model: {report['model_class']}")
    if "parameter_count" in report:
        print(f"Parameters: {report['parameter_count']:,}")
    if "submodule_count" in report:
        print(f"Submodules: {report['submodule_count']} ({', '.join(report.get('submodules', []))})")
    if "top_level_params" in report:
        print(f"Top-level parameters: {', '.join(report['top_level_params'])}")
    fp = report["forward_profile"]
    if "latency_ms" in fp:
        print(f"Forward latency: {fp['latency_ms']:.3f} ms")
    if "rss_mb" in fp:
        print(f"Process RSS: {fp['rss_mb']:.1f} MB")
    print("─" * 40)
    print("Status: Telemetry collected")

    return report