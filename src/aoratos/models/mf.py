from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseModel
from .constants import DEFAULT_MODEL_DIR
from .errors import ModelNotFittedError, SchemaValidationError
from .types import ArrayLike1D, DataFrameLike


@dataclass(slots=True)
class MatrixFactorizationModel(BaseModel):
    """Regularized biased matrix factorization (Funk-SVD style).

    Prediction rule:

        r_hat(u, m, t) = mu + b_u + b_m + <p_u, q_m> + b_{u,t} + b_{m,t}

    where time-dependent terms are optional and controlled by ``use_time_bias``.
    """

    user_column: str = "customer_id"
    movie_column: str = "movie_id"
    date_column: str = "date"

    n_factors: int = 64
    lr: float = 0.01
    reg: float = 0.02
    reg_bias: float = 0.01
    n_epochs: int = 10
    max_samples_per_epoch: int | None = 2_000_000
    track_train_rmse: bool = False
    use_time_bias: bool = True
    time_bin_freq: str = "M"
    clip_predictions: bool = True
    rating_min: float = 1.0
    rating_max: float = 5.0
    random_state: int | None = 42

    _mu: float = field(init=False, default=0.0)
    _user_id_to_index: dict[int | str, int] = field(init=False, default_factory=dict)
    _movie_id_to_index: dict[int | str, int] = field(init=False, default_factory=dict)
    _user_biases: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    _movie_biases: np.ndarray = field(init=False, default_factory=lambda: np.array([]))
    _user_factors: np.ndarray = field(
        init=False, default_factory=lambda: np.empty((0, 0))
    )
    _movie_factors: np.ndarray = field(
        init=False, default_factory=lambda: np.empty((0, 0))
    )
    _time_key_to_index: dict[str, int] = field(init=False, default_factory=dict)
    _user_time_biases: dict[tuple[int, int], float] = field(
        init=False, default_factory=dict
    )
    _movie_time_biases: dict[tuple[int, int], float] = field(
        init=False, default_factory=dict
    )

    train_rmse_history: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        """Validate hyperparameters and initialize BaseModel state."""

        BaseModel.__init__(self, name="MatrixFactorization")

        if self.n_factors < 1:
            raise ValueError("n_factors must be >= 1")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.max_samples_per_epoch is not None and self.max_samples_per_epoch < 1:
            raise ValueError("max_samples_per_epoch must be >= 1 when provided")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.reg < 0 or self.reg_bias < 0:
            raise ValueError("regularization terms must be non-negative")
        if self.rating_min >= self.rating_max:
            raise ValueError("rating_min must be smaller than rating_max")

    def _validate_X(self, X: DataFrameLike, *, require_date: bool) -> None:
        """Validate required schema for train/predict inputs.

        Parameters
        ----------
        X:
            Input interactions dataframe.
        require_date:
            Whether the date column is required (enabled when using time biases).
        """

        if not isinstance(X, pd.DataFrame):
            raise SchemaValidationError("X must be a pandas DataFrame")

        required = [self.user_column, self.movie_column]
        if require_date:
            required.append(self.date_column)

        missing = [column for column in required if column not in X.columns]
        if missing:
            raise SchemaValidationError(
                f"X is missing required columns: {missing}. Required: {required}"
            )

        if X.empty:
            raise SchemaValidationError("X must contain at least one row")

    def _validate_resolve_y(
        self,
        X: DataFrameLike,
        y: ArrayLike1D | None,
    ) -> np.ndarray:
        """Resolve target values from explicit `y` or fallback to `X['rating']`."""

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

    def _encode_time_bins_fit(self, date_series: pd.Series) -> np.ndarray:
        """Encode date values into integer time-bin codes and fit the lookup map."""

        timestamps = pd.to_datetime(date_series, errors="coerce")
        if timestamps.isna().any():
            raise SchemaValidationError("date column contains non-parsable values")

        period_values = timestamps.dt.to_period(self.time_bin_freq).astype(str)
        codes, uniques = pd.factorize(period_values, sort=True)
        self._time_key_to_index = {
            time_key: int(index) for index, time_key in enumerate(uniques.tolist())
        }
        return codes.astype(np.int64)

    def _encode_time_bins_predict(self, date_series: pd.Series) -> np.ndarray:
        """Encode date values into known time-bin codes for inference.

        Unknown bins map to -1 and receive zero time-bias contribution.
        """

        timestamps = pd.to_datetime(date_series, errors="coerce")
        if timestamps.isna().any():
            raise SchemaValidationError("date column contains non-parsable values")

        period_values = timestamps.dt.to_period(self.time_bin_freq).astype(str)
        return np.array(
            [self._time_key_to_index.get(time_key, -1) for time_key in period_values],
            dtype=np.int64,
        )

    @staticmethod
    def _safe_time_bias(
        biases: dict[tuple[int, int], float],
        key: tuple[int, int],
    ) -> float:
        """Return sparse time bias value, defaulting to zero."""

        return float(biases.get(key, 0.0))

    def _get_time_bias_terms(
        self,
        *,
        user_index: int,
        movie_index: int,
        time_code: int,
    ) -> tuple[float, float]:
        """Fetch user/movie time-bias terms for one interaction."""

        if (not self.use_time_bias) or time_code < 0:
            return 0.0, 0.0

        user_term = self._safe_time_bias(
            self._user_time_biases,
            (user_index, time_code),
        )
        movie_term = self._safe_time_bias(
            self._movie_time_biases,
            (movie_index, time_code),
        )
        return user_term, movie_term

    def _update_time_bias_terms(
        self,
        *,
        user_index: int,
        movie_index: int,
        time_code: int,
        error: float,
    ) -> None:
        """Apply SGD updates to sparse user/movie time-bias terms."""

        if (not self.use_time_bias) or time_code < 0:
            return

        user_key = (user_index, time_code)
        movie_key = (movie_index, time_code)

        user_bias = self._safe_time_bias(self._user_time_biases, user_key)
        movie_bias = self._safe_time_bias(self._movie_time_biases, movie_key)

        self._user_time_biases[user_key] = user_bias + self.lr * (
            error - self.reg_bias * user_bias
        )
        self._movie_time_biases[movie_key] = movie_bias + self.lr * (
            error - self.reg_bias * movie_bias
        )

    def fit(
        self,
        X: DataFrameLike,
        y: ArrayLike1D | None = None,
        **kwargs: Any,
    ) -> "MatrixFactorizationModel":
        """Fit MF parameters with SGD over observed explicit-rating interactions."""

        # Validate schema early to fail fast on missing columns/empty data.
        self._validate_X(X, require_date=self.use_time_bias)
        # Resolve supervised target from explicit y or X["rating"].
        ratings = self._validate_resolve_y(X, y).astype(np.float32, copy=False)

        # Extract raw ids and compute global mean baseline term.
        user_ids = X[self.user_column].to_numpy()
        movie_ids = X[self.movie_column].to_numpy()
        self._mu = float(ratings.mean())

        # Convert sparse/raw ids into dense contiguous integer codes for fast array math.
        user_codes, unique_users = pd.factorize(user_ids, sort=False)
        movie_codes, unique_movies = pd.factorize(movie_ids, sort=False)

        # Store reverse lookups so predict() can map incoming ids to trained indices.
        self._user_id_to_index = {
            user_id: int(index) for index, user_id in enumerate(unique_users.tolist())
        }
        self._movie_id_to_index = {
            movie_id: int(index)
            for index, movie_id in enumerate(unique_movies.tolist())
        }

        n_users = len(unique_users)
        n_movies = len(unique_movies)
        n_samples = ratings.shape[0]

        # Use one RNG for both parameter initialization and epoch shuffling for determinism.
        rng = np.random.default_rng(self.random_state)
        # Initialize additive bias terms at zero.
        self._user_biases = np.zeros(n_users, dtype=np.float32)
        self._movie_biases = np.zeros(n_movies, dtype=np.float32)
        # Initialize latent factors with small random values to break symmetry.
        self._user_factors = rng.normal(
            0.0, 0.1, size=(n_users, self.n_factors)
        ).astype(np.float32)
        self._movie_factors = rng.normal(
            0.0, 0.1, size=(n_movies, self.n_factors)
        ).astype(np.float32)

        # Keep sparse time-bias state in dicts because only observed (entity, time) pairs matter.
        self._user_time_biases = {}
        self._movie_time_biases = {}
        # Encode dates to compact bins once so SGD loop uses integer codes only.
        if self.use_time_bias:
            time_codes = self._encode_time_bins_fit(X[self.date_column])
        else:
            time_codes = np.full(n_samples, -1, dtype=np.int64)

        # Track epoch-level train RMSE to monitor convergence behavior.
        self.train_rmse_history = []
        indices = np.arange(n_samples, dtype=np.int64)

        for _ in range(self.n_epochs):
            # Shuffle rows each epoch to reduce cyclic update bias in SGD.
            shuffled = rng.permutation(indices)
            if (
                self.max_samples_per_epoch is not None
                and self.max_samples_per_epoch < n_samples
            ):
                shuffled = shuffled[: self.max_samples_per_epoch]

            for idx in shuffled:
                # Read one observed interaction.
                u = int(user_codes[idx])
                m = int(movie_codes[idx])
                r = float(ratings[idx])

                # Fetch current parameters for this user/item.
                bu = float(self._user_biases[u])
                bm = float(self._movie_biases[m])
                pu = self._user_factors[u]
                qm = self._movie_factors[m]

                # Add optional time-bin adjustments when configured and known.
                t = int(time_codes[idx])
                but, bit = self._get_time_bias_terms(
                    user_index=u,
                    movie_index=m,
                    time_code=t,
                )
                time_term = but + bit

                # Compute prediction and residual for this single SGD step.
                pred = self._mu + bu + bm + float(np.dot(pu, qm)) + time_term
                err = r - pred

                # Update scalar biases with L2-style shrinkage.
                self._user_biases[u] = bu + self.lr * (err - self.reg_bias * bu)
                self._movie_biases[m] = bm + self.lr * (err - self.reg_bias * bm)

                # Use previous user vector so coupled updates remain mathematically consistent.
                pu_prev = pu.copy()
                # Update latent vectors with gradient step and factor regularization.
                self._user_factors[u] = pu + self.lr * (err * qm - self.reg * pu)
                self._movie_factors[m] = qm + self.lr * (err * pu_prev - self.reg * qm)

                # Update sparse time-bias terms if active.
                self._update_time_bias_terms(
                    user_index=u,
                    movie_index=m,
                    time_code=t,
                    error=err,
                )

            # RMSE tracking can be disabled to reduce training runtime on large datasets.
            if self.track_train_rmse:
                epoch_rmse = self._compute_rmse_chunked(
                    user_codes=user_codes[shuffled],
                    movie_codes=movie_codes[shuffled],
                    time_codes=time_codes[shuffled],
                    ratings=ratings[shuffled],
                )
                self.train_rmse_history.append(epoch_rmse)

        self.fitted = True
        return self

    def _predict(
        self,
        *,
        user_codes: np.ndarray,
        movie_codes: np.ndarray,
        time_codes: np.ndarray,
    ) -> np.ndarray:
        """Predict from pre-encoded user/movie/time arrays.

        This helper is shared by both training-time RMSE tracking and public inference.
        """

        # Start from global mean so every row has a valid fallback prediction.
        predictions = np.full(user_codes.shape[0], self._mu, dtype=np.float32)

        # Mark which entities are known from training; unknowns skip their specific terms.
        known_users = user_codes >= 0
        known_movies = movie_codes >= 0

        # Add available user and movie biases independently.
        if known_users.any():
            predictions[known_users] += self._user_biases[user_codes[known_users]]
        if known_movies.any():
            predictions[known_movies] += self._movie_biases[movie_codes[known_movies]]

        # Add latent interaction only when both user and movie were seen in training.
        known_both = known_users & known_movies
        if known_both.any():
            both_rows = np.flatnonzero(known_both)
            interaction_batch_size = 50_000
            for start in range(0, both_rows.shape[0], interaction_batch_size):
                rows = both_rows[start : start + interaction_batch_size]
                u_idx = user_codes[rows]
                m_idx = movie_codes[rows]
                interaction = np.einsum(
                    "ij,ij->i",
                    self._user_factors[u_idx],
                    self._movie_factors[m_idx],
                )
                predictions[rows] += interaction

        # Time-bias terms are additive and only applied to known user/movie + known time bin.
        if self.use_time_bias:
            for row in np.where(known_users)[0]:
                t = int(time_codes[row])
                if t >= 0:
                    predictions[row] += self._safe_time_bias(
                        self._user_time_biases,
                        (int(user_codes[row]), t),
                    )

            for row in np.where(known_movies)[0]:
                t = int(time_codes[row])
                if t >= 0:
                    predictions[row] += self._safe_time_bias(
                        self._movie_time_biases,
                        (int(movie_codes[row]), t),
                    )

        # Keep outputs on the explicit rating scale when clipping is enabled.
        if self.clip_predictions:
            predictions = np.clip(predictions, self.rating_min, self.rating_max)

        return predictions

    def _compute_rmse_chunked(
        self,
        *,
        user_codes: np.ndarray,
        movie_codes: np.ndarray,
        time_codes: np.ndarray,
        ratings: np.ndarray,
    ) -> float:
        """Compute RMSE in chunks to avoid allocating full-size prediction arrays."""

        if ratings.shape[0] == 0:
            return 0.0

        sse = 0.0
        rmse_batch_size = 200_000

        for start in range(0, ratings.shape[0], rmse_batch_size):
            end = min(start + rmse_batch_size, ratings.shape[0])
            preds = self._predict(
                user_codes=user_codes[start:end],
                movie_codes=movie_codes[start:end],
                time_codes=time_codes[start:end],
            )
            residual = ratings[start:end] - preds
            sse += float(np.dot(residual, residual))

        return float(np.sqrt(sse / ratings.shape[0]))

    def predict(self, X: DataFrameLike, **kwargs: Any) -> np.ndarray:
        """Predict ratings for rows in `X` using learned factors and biases."""

        if not self.fitted:
            raise ModelNotFittedError("fit must be called before predict/score")

        # Reuse training-time schema guardrails for consistent inference behavior.
        self._validate_X(X, require_date=self.use_time_bias)

        # Map external ids to training indices; unseen ids become -1 for cold-start fallback.
        user_codes = np.array(
            [
                self._user_id_to_index.get(user_id, -1)
                for user_id in X[self.user_column]
            ],
            dtype=np.int64,
        )
        movie_codes = np.array(
            [
                self._movie_id_to_index.get(movie_id, -1)
                for movie_id in X[self.movie_column]
            ],
            dtype=np.int64,
        )

        # Encode inference timestamps; unknown bins map to -1 and skip time bias.
        if self.use_time_bias:
            time_codes = self._encode_time_bins_predict(X[self.date_column])
        else:
            time_codes = np.full(len(X), -1, dtype=np.int64)

        return self._predict(
            user_codes=user_codes,
            movie_codes=movie_codes,
            time_codes=time_codes,
        )

    @classmethod
    def _resolve_pkl(
        cls,
        *,
        name: str | None = None,
        path: Path | str | None = None,
    ) -> Path:
        candidate = Path(path) if path is not None else Path(name or cls().name)
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".pkl")

        if not candidate.is_absolute():
            candidate = DEFAULT_MODEL_DIR / candidate

        return candidate

    def save(self, path: Path | str | None = None) -> Path:
        """Serialize fitted model state to disk.

        Default path: ``data/models/<name>.pkl``.
        """

        if not self.fitted:
            raise ModelNotFittedError("fit must be called before save")

        target_path = self._resolve_pkl(name=self.name, path=path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with target_path.open("wb") as file:
            pickle.dump(self, file)

        return target_path

    @classmethod
    def load(
        cls,
        *,
        name: str | None = None,
        path: Path | str | None = None,
    ) -> "MatrixFactorizationModel":
        """Load model state from disk.

        Default path: ``data/models/<name>.pkl``.
        """

        source_path = cls._resolve_pkl(name=name, path=path)
        if not source_path.exists():
            raise FileNotFoundError(f"Saved model not found at {source_path}")

        with source_path.open("rb") as file_obj:
            loaded = pickle.load(file_obj)

        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded artifact type {type(loaded).__name__} does not match {cls.__name__}"
            )

        return loaded
