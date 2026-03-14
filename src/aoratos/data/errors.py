from __future__ import annotations


class DataError(RuntimeError):
    """Base class for data module errors."""


class DataNotFoundError(DataError):
    """Raised when expected dataset input cannot be found."""


class DataAmbiguityError(DataError):
    """Raised when multiple dataset matches are found for one name."""


class UnsafeNameError(DataError):
    """Raised when a provided savepoint name is unsafe."""
