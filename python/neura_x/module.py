# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Base Module System
# ==========================================================

"""
Base module system for Neura-X. Provides the foundational Module class
(analogous to torch.nn.Module) and the Parameter class.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Iterator, Tuple
from collections import OrderedDict


class Parameter:
    """A trainable parameter in Neura-X."""

    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data.astype(np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def zero_grad(self):
        self.grad = None

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"


class Module:
    """Base class for all Neura-X modules."""

    def __init__(self):
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._modules: OrderedDict[str, 'Module'] = OrderedDict()
        self._buffers: OrderedDict[str, np.ndarray] = OrderedDict()
        self.training: bool = True
        self._name: str = self.__class__.__name__

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, name: str):
        """Allow accessing parameters and modules as attributes."""
        _parameters = self.__dict__.get('_parameters')
        if _parameters is not None and name in _parameters:
            return _parameters[name]
        _modules = self.__dict__.get('_modules')
        if _modules is not None and name in _modules:
            return _modules[name]
        _buffers = self.__dict__.get('_buffers')
        if _buffers is not None and name in _buffers:
            return _buffers[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            params = self.__dict__.get('_parameters')
            if params is None:
                params = OrderedDict()
                self.__dict__['_parameters'] = params
            params[name] = value
        elif isinstance(value, Module):
            modules = self.__dict__.get('_modules')
            if modules is None:
                modules = OrderedDict()
                self.__dict__['_modules'] = modules
            modules[name] = value
        else:
            super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Parameter):
        self._parameters[name] = param

    def register_module(self, name: str, module: 'Module'):
        self._modules[name] = module

    def register_buffer(self, name: str, tensor: np.ndarray):
        self._buffers[name] = tensor

    def parameters(self) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        for mod_name, module in self._modules.items():
            mod_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(mod_prefix)

    def children(self) -> Iterator['Module']:
        yield from self._modules.values()

    def modules(self) -> Iterator['Module']:
        yield self
        for module in self._modules.values():
            yield from module.modules()

    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        return self.train(False)

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def parameter_count(self) -> int:
        return sum(p.size for p in self.parameters() if p.requires_grad)

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data
        for name, buffer in self._buffers.items():
            state[name] = buffer
        for name, module in self._modules.items():
            state[name] = module.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        for name, param in self._parameters.items():
            if name in state:
                param.data = state[name]
        for name, buffer in self._buffers.items():
            if name in state:
                self._buffers[name] = state[name]
        for name, module in self._modules.items():
            if name in state:
                module.load_state_dict(state[name])

    def __repr__(self) -> str:
        lines = [f"{self._name}("]
        for name, module in self._modules.items():
            mod_str = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        param_count = self.parameter_count()
        lines.append(f"  [parameters: {param_count:,}]")
        lines.append(")")
        return "\n".join(lines)


class FractalLayer(Module):
    """A neural network layer using Fractal Tensors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        conceptual_params: Optional[int] = None,
        K: int = 1024,
        bias: bool = True,
        seed=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conceptual_params = conceptual_params or (in_features * out_features)
        self.K = K
        self._has_bias = bias

        from neura_x.fractal_layer import FractalSeed
        if seed is not None:
            self.fractal_seed = seed
        else:
            rng = np.random.default_rng()
            self.fractal_seed = FractalSeed(
                a=rng.normal(0, 0.02, size=K),
                w1=rng.uniform(-np.pi, np.pi, size=K),
                w2=rng.uniform(-np.pi, np.pi, size=K),
                phi=rng.uniform(0, 2 * np.pi, size=K),
            )

        self.register_parameter(
            "fractal_a", Parameter(self.fractal_seed.a, requires_grad=True)
        )
        self.register_parameter(
            "fractal_w1", Parameter(self.fractal_seed.w1, requires_grad=True)
        )
        self.register_parameter(
            "fractal_w2", Parameter(self.fractal_seed.w2, requires_grad=True)
        )
        self.register_parameter(
            "fractal_phi", Parameter(self.fractal_seed.phi, requires_grad=True)
        )

        if bias:
            self.register_parameter(
                "bias", Parameter(np.zeros(out_features, dtype=np.float32))
            )

    @property
    def bias(self):
        """Access bias parameter, returns None if no bias."""
        if self._has_bias:
            return self._parameters.get("bias")
        return None

    def bloom(self, rows: int, cols: int) -> np.ndarray:
        a = self._parameters["fractal_a"].data
        w1 = self._parameters["fractal_w1"].data
        w2 = self._parameters["fractal_w2"].data
        phi = self._parameters["fractal_phi"].data

        i_indices = np.arange(rows, dtype=np.float32)[:, None]
        j_indices = np.arange(cols, dtype=np.float32)[None, :]

        W = np.zeros((rows, cols), dtype=np.float32)
        for k in range(self.K):
            W += a[k] * np.sin(w1[k] * i_indices + w2[k] * j_indices + phi[k])
        return W

    def forward(self, x: np.ndarray) -> np.ndarray:
        W = self.bloom(self.out_features, self.in_features)
        output = x @ W.T
        bias_param = self.bias
        if bias_param is not None:
            output = output + bias_param.data
        return output

    def memory_usage(self) -> dict:
        standard_bytes = self.in_features * self.out_features * 4
        fractal_bytes = 4 * self.K * 4
        return {
            "standard_bytes": standard_bytes,
            "fractal_bytes": fractal_bytes,
            "reduction_factor": standard_bytes / max(fractal_bytes, 1),
            "standard_mb": standard_bytes / (1024 ** 2),
            "fractal_kb": fractal_bytes / 1024,
        }

    def __repr__(self) -> str:
        mem = self.memory_usage()
        return (
            f"FractalLayer("
            f"in={self.in_features}, out={self.out_features}, "
            f"K={self.K}, "
            f"conceptual_params={self.conceptual_params:,}, "
            f"memory={mem['fractal_kb']:.1f}KB "
            f"vs standard {mem['standard_mb']:.1f}MB)"
        )