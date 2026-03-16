from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .metrics import evaluate_regression
from .types import ArrayLike1D, DataFrameLike, MetricsDict


class BaseModel(ABC):
    """Abstract base class for all Aoratos models."""

    def __init__(self, name: str | None = None) -> None:
        self.name: str = name or self.__class__.__name__
        self.fitted: bool = False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    @abstractmethod
    def fit(
        self, X: DataFrameLike, y: ArrayLike1D | None = None, **kwargs
    ) -> "BaseModel":
        """Train model parameters from input data."""

    @abstractmethod
    def predict(self, X: DataFrameLike, **kwargs) -> np.ndarray:
        """Predict targets for rows in X."""

    def score(
        self,
        y_true: ArrayLike1D,
        y_pred: ArrayLike1D,
        *,
        include_n_samples: bool = True,
        **kwargs,
    ) -> MetricsDict:
        """Evaluate predictions using universal regression metrics."""

        y_true = self._coerce_1d_array(y_true, name="y_true")
        y_pred = self._coerce_1d_array(y_pred, name="y_pred")
        return evaluate_regression(
            y_true,
            y_pred,
            include_n_samples=include_n_samples,
        )

    @staticmethod
    def _coerce_1d_array(values: Any, *, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if np.isnan(arr).any():
            raise ValueError(f"{name} must not contain NaN values")
        return arr
