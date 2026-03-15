from __future__ import annotations


class DataError(RuntimeError):
    """Base class for data module errors."""


class DataNotFoundError(DataError):
    """Raised when expected dataset input cannot be found."""


class DataAmbiguityError(DataError):
    """Raised when multiple dataset matches are found for one name."""


class UnsafeNameError(DataError):
    """Raised when a provided savepoint name is unsafe."""


class SupplementConfigurationError(DataError):
    """Raised when supplement runtime configuration is invalid."""


class SupplementInternalError(DataError):
    """Raised for unexpected internal supplement control-flow failures."""


class SupplementSchemaError(DataError):
    """Raised when supplement source data does not match required schema."""


class TMDBAuthenticationError(SupplementConfigurationError):
    """Raised when TMDB_API_KEY is missing for supplement execution."""


class TMDBRequestError(DataError):
    """Raised when TMDB request handling fails terminally."""
