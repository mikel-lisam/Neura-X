# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
#
# Loss Functions
#
# ==========================================================

"""
Standard loss functions for Neura-X.
All loss functions use standard, proven mathematics.
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for classification and language modeling.

    L = -Σ y_i × log(ŷ_i)
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        # Softmax
        exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
        probs = exp_preds / exp_preds.sum(axis=-1, keepdims=True)

        # Clip for numerical stability
        probs = np.clip(probs, 1e-10, 1.0)

        # Cross-entropy
        if targets.ndim == 1:
            # Integer targets
            batch_size = predictions.shape[0]
            log_probs = -np.log(probs[np.arange(batch_size), targets.astype(int)])
        else:
            # One-hot targets
            log_probs = -np.sum(targets * np.log(probs), axis=-1)

        if self.reduction == "mean":
            return float(np.mean(log_probs))
        elif self.reduction == "sum":
            return float(np.sum(log_probs))
        return log_probs


class MSELoss:
    """
    Mean Squared Error Loss for regression.

    L = (1/n) × Σ (y - ŷ)²
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        diff = predictions - targets
        squared = diff ** 2

        if self.reduction == "mean":
            return float(np.mean(squared))
        elif self.reduction == "sum":
            return float(np.sum(squared))
        return squared


class MAELoss:
    """
    Mean Absolute Error Loss.

    L = (1/n) × Σ |y - ŷ|
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        abs_diff = np.abs(predictions - targets)

        if self.reduction == "mean":
            return float(np.mean(abs_diff))
        elif self.reduction == "sum":
            return float(np.sum(abs_diff))
        return abs_diff


class L1Loss:
    """Alias for MAELoss."""

    def __init__(self, reduction: str = "mean"):
        self._mae = MAELoss(reduction=reduction)

    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return self._mae(predictions, targets)