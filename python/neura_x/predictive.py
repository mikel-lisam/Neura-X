# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Predictive Models
# ==========================================================

"""Predictive models: time series, regression, classification."""

import numpy as np
from neura_x.module import Module, FractalLayer


class Predictive(Module):
    """Predictive model for time series, regression, classification."""

    def __init__(
        self,
        architecture: str = "transformer",
        task: str = "time_series",
        input_features: int = 12,
        horizon: int = 30,
        conceptual_params: int = 50_000_000,
        **kwargs,
    ):
        super().__init__()
        self.architecture = architecture
        self.task = task
        self.input_features = input_features
        self.horizon = horizon
        self.conceptual_params = conceptual_params

        self.register_module("backbone", FractalLayer(
            in_features=input_features,
            out_features=horizon,
            conceptual_params=conceptual_params,
        ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.backbone(x)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        return self.forward(data)