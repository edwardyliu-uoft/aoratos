"""Aoratos modeling subsystem public API."""

from .base import BaseModel
from .baseline import BaselineCFModel
from .errors import (
    BaselineModelError,
    ModelError,
    ModelNotFittedError,
    SchemaValidationError,
)
from .metrics import accuracy, evaluate_regression, mae, precision, recall, rmse
from .preprocessing import cosine_similarity_matrix, scale_features
from .types import ArrayLike1D, DataFrameLike, MetricsDict

__all__ = [
    "ArrayLike1D",
    "BaseModel",
    "BaselineCFModel",
    "BaselineModelError",
    "DataFrameLike",
    "MetricsDict",
    "ModelError",
    "ModelNotFittedError",
    "SchemaValidationError",
    "rmse",
    "mae",
    "precision",
    "recall",
    "accuracy",
    "evaluate_regression",
    "scale_features",
    "cosine_similarity_matrix",
]
