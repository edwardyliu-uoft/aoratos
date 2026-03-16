from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import BaseModel
from .errors import ModelNotFittedError, SchemaValidationError
from .types import ArrayLike1D, DataFrameLike


@dataclass(slots=True)
class BaselineCFModel(BaseModel):
    """Regularized collaborative-filtering baseline with user/movie biases.

    The baseline prediction rule is:

        r_hat(u, m) = mu + b_u + b_m

    where:

    - ``mu`` is the global mean rating,
    - ``b_u`` is a per-user bias,
    - ``b_m`` is a per-movie bias.

    Bias terms are learned with an alternating least-squares style procedure and
    L2-style shrinkage through ``reg_user`` and ``reg_movie``.

    This class keeps the baseline intentionally simple and deterministic while still
    being strong enough to serve as a practical starting point for more advanced models.
    """

    name: str = "CollaborativeFiltering"
    user_column: str = "customer_id"
    movie_column: str = "movie_id"

    rating_min: float = 1.0
    rating_max: float = 5.0
    reg_user: float = 10.0
    reg_movie: float = 10.0
    n_iters: int = 15
    clip_predictions: bool = True

    _mu: float = field(init=False, default=0.0)
    _user_biases: dict[int | str, float] = field(init=False, default_factory=dict)
    _movie_biases: dict[int | str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hyperparameters and initialize internal state."""

        BaseModel.__init__(self, name=self.name)

        if self.rating_min >= self.rating_max:
            raise ValueError("rating_min must be smaller than rating_max")
        if self.reg_user < 0 or self.reg_movie < 0:
            raise ValueError("regularization terms must be non-negative")
        if self.n_iters < 1:
            raise ValueError("n_iters must be >= 1")

        self._mu = 0.0
        self._user_biases = {}
        self._movie_biases = {}

    def _validate_X(self, X: DataFrameLike) -> None:
        """Validate required input dataframe X; shape and schema."""

        if not isinstance(X, pd.DataFrame):
            raise SchemaValidationError("X must be a pandas DataFrame")

        missing = [
            column
            for column in (self.user_column, self.movie_column)
            if column not in X.columns
        ]
        if missing:
            raise SchemaValidationError(
                f"X is missing required columns: {missing}. Required: {self.user_column}, {self.movie_column}"
            )

        if X.empty:
            raise SchemaValidationError("X must contain at least one row")

    def _validate_resolve_y(
        self,
        X: DataFrameLike,
        y: ArrayLike1D | None,
    ) -> np.ndarray:
        """Resolve target vector from explicit `y` or `X['rating']` and validate."""

        if y is None:
            if "rating" not in X.columns:
                raise SchemaValidationError(
                    "y is required unless X includes a 'rating' column"
                )
            values = X["rating"].to_numpy(dtype=float)
        else:
            values = np.asarray(y, dtype=float)

        if values.ndim != 1:
            raise SchemaValidationError("y must be one-dimensional")
        if values.shape[0] != len(X):
            raise SchemaValidationError("X and y must have the same number of rows")
        if np.isnan(values).any():
            raise SchemaValidationError("y must not contain NaN values")

        return values

    def fit(
        self,
        X: DataFrameLike,
        y: ArrayLike1D | None = None,
        **kwargs,
    ) -> "BaselineCFModel":
        """Fit user/movie bias terms from training interactions.

        `X` must include user and movie id columns.
        If `y` is omitted, `X` must include a `rating` column.

        Training pipeline:

        1. Validate schema and resolve the rating target vector.
        2. Encode user/movie ids into dense integer codes.
        3. Iteratively update user and movie biases with vectorized aggregated residuals.
        4. Materialize learned biases into dictionaries for fast prediction lookups.

        The update equations are:

        - ``b_u = sum_{m in M(u)} (r_um - mu - b_m) / (reg_user + |M(u)|)``
        - ``b_m = sum_{u in U(m)} (r_um - mu - b_u) / (reg_movie + |U(m)|)``
        """

        self._validate_X(X)
        ratings = self._validate_resolve_y(X, y)

        # Step 1: pull raw id arrays and compute global mean.
        user_ids = X[self.user_column].to_numpy()
        movies_ids = X[self.movie_column].to_numpy()
        self._mu = float(ratings.mean())

        # Step 2: encode arbitrary ids to dense [0..n) codes for vectorized math.
        user_codes, unique_users = pd.factorize(user_ids, sort=False)
        movie_codes, unique_movies = pd.factorize(movies_ids, sort=False)

        n_users = unique_users.shape[0]
        n_movies = unique_movies.shape[0]

        user_counts = np.bincount(user_codes, minlength=n_users).astype(float)
        movie_counts = np.bincount(movie_codes, minlength=n_movies).astype(float)

        user_biases = np.zeros(n_users, dtype=float)
        movie_biases = np.zeros(n_movies, dtype=float)

        # Step 3: alternating updates (users then movies) using residual aggregation.
        for _ in range(self.n_iters):
            user_residual = ratings - self._mu - movie_biases[movie_codes]
            user_sums = np.bincount(
                user_codes,
                weights=user_residual,
                minlength=n_users,
            )
            user_biases = user_sums / (self.reg_user + user_counts)

            movie_residual = ratings - self._mu - user_biases[user_codes]
            movie_sums = np.bincount(
                movie_codes,
                weights=movie_residual,
                minlength=n_movies,
            )
            movie_biases = movie_sums / (self.reg_movie + movie_counts)

        # Step 4: store learned parameters as id->bias lookups for predict().
        self._user_biases = {
            user_id: float(user_biases[index])
            for index, user_id in enumerate(unique_users)
        }
        self._movie_biases = {
            movie_id: float(movie_biases[index])
            for index, movie_id in enumerate(unique_movies)
        }
        self.fitted = True
        return self

    def predict(self, X: DataFrameLike, **kwargs) -> np.ndarray:
        """Predict ratings for rows in `X` containing user/movie identifiers.

        Unknown users/movies default to zero bias, so prediction gracefully falls back
        toward the global mean when encountering cold-start entities.
        """

        if not self.fitted:
            raise ModelNotFittedError("fit must be called before predict/score")
        self._validate_X(X)

        user_component = (
            X[self.user_column].map(self._user_biases).fillna(0.0).to_numpy(dtype=float)
        )
        movie_component = (
            X[self.movie_column]
            .map(self._movie_biases)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        predictions = self._mu + user_component + movie_component

        if self.clip_predictions:
            predictions = np.clip(predictions, self.rating_min, self.rating_max)

        return predictions
