# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Predictive Models
# ==========================================================

"""Predictive models: time series, regression, classification."""

import numpy as np
from neura_x.module import Module, FractalLayer, Parameter


class Predictive(Module):
    """Predictive model for time series, regression, classification.

    Uses a small recurrent architecture: a `FractalLayer` block maps each
    time step's features into a hidden state, a learnable temporal
    recurrence mixes the state with the previous hidden state, and a
    final `FractalLayer` projects the hidden state to the prediction
    horizon. ``forward(x)`` accepts inputs of shape ``(batch, time,
    features)`` or ``(time, features)`` and returns ``(..., horizon)``.
    """

    def __init__(
        self,
        architecture: str = "transformer",
        task: str = "time_series",
        input_features: int = 12,
        horizon: int = 30,
        hidden_dim: int = 64,
        conceptual_params: int = 50_000_000,
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.task = task
        self.input_features = input_features
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.conceptual_params = conceptual_params

        self.register_module(
            "encoder",
            FractalLayer(in_features=input_features, out_features=hidden_dim, K=128),
        )
        self.register_parameter(
            "recurrent_w",
            Parameter(np.eye(hidden_dim, dtype=np.float32) * 0.9),
        )
        self.register_parameter(
            "recurrent_u",
            Parameter(np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02),
        )
        self.register_module(
            "decoder",
            FractalLayer(in_features=hidden_dim, out_features=horizon, K=128),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1, self.input_features)
        elif x.ndim == 2:
            x = x.reshape(1, *x.shape)
        # x is now (batch, time, features).
        batch, time_steps, _ = x.shape
        h = np.zeros((batch, self.hidden_dim), dtype=np.float32)
        for t in range(time_steps):
            step = x[:, t, :]
            h_new = self.encoder(step)  # (batch, hidden_dim)
            # Recurrence: h_t = tanh(W h_{t-1} + U x_t)
            h = np.tanh(h_new @ self._parameters["recurrent_u"].data.T + h @ self._parameters["recurrent_w"].data)
        return self.decoder(h)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        return self.forward(data)