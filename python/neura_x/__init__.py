# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Neura-X Python Package — Main Entry Point
#
# Usage:
#   import neura_x as nx
#
# ==========================================================

"""
Neura-X: Intelligence Without Limits.

A CPU-first, universal AI framework that can train, fine-tune, and deploy
any type of AI model on hardware as modest as a Core i5 5th Gen laptop
with 4GB of RAM.

Founded by Edusei Mikel Lisamba.
Open University of Kenya.
"""

__version__ = "1.0.0"
__author__ = "Edusei Mikel Lisamba"
__tagline__ = "Intelligence Without Limits."
__institution__ = "Open University of Kenya"
__country__ = "Kenya"
__license__ = "Neura-X Dual License"

# ──────────────────────────────────────────────────────────
# RUST CORE AUTO-DISCOVERY
# ──────────────────────────────────────────────────────────

_core = None
_core_available = False


def _load_rust_core():
    """
    Attempt to load the compiled Rust core from multiple locations.
    Tries 4 strategies in order until one succeeds.
    """
    global _core, _core_available

    # Strategy 1: Direct import (works if installed via pip/maturin develop)
    try:
        from neura_x import _core as _rust_core
        _core = _rust_core
        _core_available = True
        return
    except ImportError:
        pass

    # Strategy 2: Look for .so/.pyd/.dylib in the package directory
    try:
        import importlib.util
        import glob
        import os

        package_dir = os.path.dirname(os.path.abspath(__file__))
        patterns = [
            os.path.join(package_dir, "_core*.so"),
            os.path.join(package_dir, "_core*.pyd"),
            os.path.join(package_dir, "_core*.dylib"),
        ]

        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                spec = importlib.util.spec_from_file_location(
                    "neura_x._core", matches[0]
                )
                if spec and spec.loader:
                    _core = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(_core)
                    _core_available = True
                    return
    except Exception:
        pass

    # Strategy 3: Look in the Rust build directory
    try:
        import importlib.util
        import os

        package_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(package_dir))

        build_paths = [
            os.path.join(project_root, "core", "rust", "python_bridge",
                         "target", "release", "lib_core.so"),
            os.path.join(project_root, "core", "rust", "python_bridge",
                         "target", "debug", "lib_core.so"),
            os.path.join(project_root, "build", "lib_core.so"),
        ]

        for path in build_paths:
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location(
                    "neura_x._core", path
                )
                if spec and spec.loader:
                    _core = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(_core)
                    _core_available = True
                    return
    except Exception:
        pass

    # Strategy 4: Search site-packages
    try:
        import importlib.util
        import site
        import glob
        import os

        search_dirs = []
        try:
            search_dirs.extend(site.getsitepackages())
        except AttributeError:
            pass
        try:
            search_dirs.append(site.getusersitepackages())
        except AttributeError:
            pass

        for site_dir in search_dirs:
            pattern = os.path.join(site_dir, "neura_x", "_core*.so")
            matches = glob.glob(pattern)
            if matches:
                spec = importlib.util.spec_from_file_location(
                    "neura_x._core", matches[0]
                )
                if spec and spec.loader:
                    _core = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(_core)
                    _core_available = True
                    return
    except Exception:
        pass

    # Core not found — set flag and continue with Python fallbacks
    _core_available = False


# Load the Rust core immediately on import
_load_rust_core()

# ──────────────────────────────────────────────────────────
# CORE IMPORTS
# ──────────────────────────────────────────────────────────

from neura_x.module import Module, FractalLayer, Parameter
from neura_x.fractal_layer import FractalTensor, FractalSeed
from neura_x.liquid_router import LiquidRouter
from neura_x.optimizer import ShadowAdamW, CircadianOptimizer, ShadowSGD
from neura_x.stream_loader import StreamLoader
from neura_x.compose import compose
from neura_x.loss import CrossEntropyLoss, MSELoss, MAELoss, L1Loss
from neura_x.optim import AdamW, SGD, Adam, RMSprop, CosineAnnealingLR, LinearLR
from neura_x.imagegen import ImageGen
from neura_x.videogen import VideoGen
from neura_x.audiogen import AudioGen
from neura_x.speech import SpeechSynthesis, SpeechRecognition
from neura_x.predictive import Predictive
from neura_x.moe import MoE
from neura_x.tools import tool, ToolRegistry
from neura_x.skills import skill, Skill
from neura_x.agent import Agent, AgentTeam
from neura_x.eval import EvalSuite
from neura_x.serve import serve
from neura_x.convert import convert
from neura_x.hub import hub
from neura_x.dream import DreamDashboard
from neura_x.xray import xray
from neura_x.safety import SafetyFilter
from neura_x.privacy import PrivacyEngine
from neura_x.bench import BenchmarkSuite

# ──────────────────────────────────────────────────────────
# MODEL TYPE ALIASES
# ──────────────────────────────────────────────────────────

LLM = None
TrainingEngine = None
PreTrainingEngine = None
FineTuneEngine = None

# ──────────────────────────────────────────────────────────
# MODULE NAMESPACE (nx.module)
# ──────────────────────────────────────────────────────────

from neura_x import compose as _compose_module


class _ModuleNamespace:
    """Namespace for nx.module.* access."""
    Prediction = None
    ImageGen = None
    ImageUnderstand = None
    VideoGen = None
    AudioGen = None
    SpeechSynthesis = None
    SpeechRecognition = None
    Tools = None
    Skills = None
    Memory = None
    Reasoning = None
    CodeExecution = None
    MultiModal = None
    AgentCore = None
    Custom = None


module = _ModuleNamespace()

# ──────────────────────────────────────────────────────────
# TOOLS NAMESPACE (nx.tools)
# ──────────────────────────────────────────────────────────

from neura_x import tools as _tools_module
tools = _tools_module

# ──────────────────────────────────────────────────────────
# LOSS NAMESPACE (nx.loss)
# ──────────────────────────────────────────────────────────

from neura_x import loss as loss

# ──────────────────────────────────────────────────────────
# OPTIM NAMESPACE (nx.optim)
# ──────────────────────────────────────────────────────────

from neura_x import optim as optim

# ──────────────────────────────────────────────────────────
# MATH NAMESPACE (nx.math)
# ──────────────────────────────────────────────────────────


class _MathNamespace:
    """Namespace for nx.math.* access."""

    @staticmethod
    def genesis_gaussian(K=1024, seed=None):
        """Generate a Fractal Seed with Gaussian-distributed initialization."""
        import numpy as np
        rng = np.random.default_rng(seed)
        a = rng.normal(0, 0.02, size=K)
        w1 = rng.uniform(-np.pi, np.pi, size=K)
        w2 = rng.uniform(-np.pi, np.pi, size=K)
        phi = rng.uniform(0, 2 * np.pi, size=K)
        return FractalSeed(a=a, w1=w1, w2=w2, phi=phi)

    @staticmethod
    def fibonacci_spiral(K=1024):
        """Generate a Fractal Seed based on Fibonacci spiral."""
        import numpy as np
        a = np.ones(K) * 0.01
        golden = (1 + np.sqrt(5)) / 2
        w1 = np.array([golden ** (i % 20) for i in range(K)])
        w2 = np.array([golden ** ((i + 1) % 20) for i in range(K)])
        phi = np.linspace(0, 2 * np.pi, K)
        return FractalSeed(a=a, w1=w1, w2=w2, phi=phi)


math = _MathNamespace()

# ──────────────────────────────────────────────────────────
# UTILITY NAMESPACE (nx.utils)
# ──────────────────────────────────────────────────────────


class _UtilsNamespace:
    """Namespace for nx.utils.* access."""

    @staticmethod
    def search(query):
        """Placeholder for web search tool."""
        return f"[Neura-X Search] Results for: {query}"

    @staticmethod
    def sandbox(code):
        """Placeholder for sandboxed code execution."""
        return f"[Neura-X Sandbox] Executed code safely."

    @staticmethod
    def sql(query):
        """Placeholder for SQL query execution."""
        return f"[Neura-X SQL] Query executed: {query}"


utils = _UtilsNamespace()

# ──────────────────────────────────────────────────────────
# SYSTEM DETECTION
# ──────────────────────────────────────────────────────────


def _detect_cpu() -> str:
    """Detect CPU model name using Rust core if available, else fallback."""
    # Try Rust core first
    if _core_available and _core is not None:
        try:
            info = _core.get_cpu_info()
            for line in info.split('\n'):
                if 'Intel' in line or 'AMD' in line or 'ARM' in line:
                    return line.strip()
        except Exception:
            pass

    # Fallback: read /proc/cpuinfo (Linux)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.strip().startswith('model name'):
                    return line.split(':')[1].strip()
    except (FileNotFoundError, PermissionError, IndexError):
        pass

    # Fallback: sysctl (macOS)
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fallback: platform module
    import platform
    processor = platform.processor()
    if processor:
        return processor

    machine = platform.machine()
    if machine:
        return f"{machine} Processor"

    return "Unknown CPU"


def _detect_ram() -> str:
    """Detect total system RAM."""
    # Try psutil first
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        return f"{ram_gb:.1f} GB"
    except ImportError:
        pass

    # Fallback: /proc/meminfo (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split(':')[1].strip().split()[0])
                    gb = kb / (1024 ** 2)
                    return f"{gb:.1f} GB"
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass

    # Fallback: sysctl (macOS)
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            bytes_total = int(result.stdout.strip())
            gb = bytes_total / (1024 ** 3)
            return f"{gb:.1f} GB"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        pass

    return "Unknown"


# ──────────────────────────────────────────────────────────
# BOOT BANNER
# ──────────────────────────────────────────────────────────


def _print_banner():
    """Print the Neura-X boot banner."""
    import platform

    cpu_info = _detect_cpu()
    ram_str = _detect_ram()

    print("⚡ Neura-X v" + __version__)
    print(f"   {__tagline__}")
    print("─" * 48)
    print(f"🧠 Founded by {__author__}")
    print(f"🌍 Built in {__country__} | {__institution__}")
    print("─" * 48)
    print(f"CPU:     {cpu_info}")
    print(f"RAM:     {ram_str}")
    print(f"Python:  {platform.python_version()}")
    print(f"Engine:  Fractal Tensor + Circadian Learning")
    if _core_available:
        print(f"Core:    Rust/C/C++ Native ✅")
    else:
        print(f"Core:    Python Fallback ⚠️")
    print("─" * 48)
    print("Status:  All systems operational. Ready to train.")
    print()


# Print banner on import
_print_banner()

# ──────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────

__all__ = [
    # Core
    "Module",
    "FractalLayer",
    "FractalTensor",
    "FractalSeed",
    "Parameter",
    "LiquidRouter",
    "ShadowAdamW",
    "ShadowSGD",
    "CircadianOptimizer",
    "StreamLoader",
    "compose",
    # Loss
    "CrossEntropyLoss",
    "MSELoss",
    "MAELoss",
    "L1Loss",
    # Optimizers
    "AdamW",
    "SGD",
    "Adam",
    "RMSprop",
    "CosineAnnealingLR",
    "LinearLR",
    # Model Types
    "ImageGen",
    "VideoGen",
    "AudioGen",
    "SpeechSynthesis",
    "SpeechRecognition",
    "Predictive",
    "MoE",
    # Tools & Skills
    "tool",
    "ToolRegistry",
    "skill",
    "Skill",
    # Agents
    "Agent",
    "AgentTeam",
    # Evaluation
    "EvalSuite",
    # Serving
    "serve",
    # Conversion
    "convert",
    # Hub
    "hub",
    # Dashboards
    "DreamDashboard",
    "xray",
    # Safety & Privacy
    "SafetyFilter",
    "PrivacyEngine",
    # Benchmarking
    "BenchmarkSuite",
    # Namespaces
    "module",
    "tools",
    "loss",
    "optim",
    "math",
    "utils",
    # Rust Core
    "_core",
    "_core_available",
    # Metadata
    "__version__",
    "__author__",
    "__tagline__",
]