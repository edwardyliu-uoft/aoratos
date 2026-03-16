from __future__ import annotations


class ModelError(RuntimeError):
    """Base class for model package errors."""


class SchemaValidationError(ModelError):
    """Raised when model input schema is invalid."""


class ModelNotFittedError(ModelError):
    """Raised when inference/evaluation is called before fit."""


class BaselineModelError(ModelError):
    """Raised for baseline-model-specific failures."""
