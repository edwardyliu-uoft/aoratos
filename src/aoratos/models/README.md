# Models Package (`aoratos.models`)

This package contains reusable model abstractions, baseline recommender logic, shared metrics, and preprocessing helpers for rating prediction tasks.

## Architecture

The package is organized around a small, composable core:

- `base.py`
  - Defines `BaseModel`, the abstract interface for all models.
  - Standard lifecycle: `fit(...)`, `predict(...)`, `score(...)`.
  - Owns common runtime state:
    - `name`: human-readable model identifier
    - `fitted`: boolean flag set by concrete models

- `baseline.py`
  - Implements `BaselineCFModel`, a deterministic collaborative-filtering baseline.
  - Uses global mean + user bias + movie bias:
    - $\hat r(u,m)=\mu+b_u+b_m$
  - Biases are learned with alternating residual updates and regularization.

- `metrics.py`
  - Universal metrics for all models:
    - `rmse`, `mae`, `precision`, `recall`, `accuracy`
  - `evaluate_regression(...)` returns a fixed metric suite for consistency.

- `preprocessing.py`
  - Shared utility wrappers over scikit-learn:
    - `scale_features(...)` via `StandardScaler`
    - `cosine_similarity_matrix(...)` via `cosine_similarity`

- `errors.py`
  - Shared exception hierarchy for model-layer failures.

- `types.py`
  - Shared type aliases (`ArrayLike1D`, `DataFrameLike`, `MetricsDict`).

- `__init__.py`
  - Public exports for package-level imports.

## Design Principles

- Minimal, explicit API surface.
- Deterministic baseline behavior (no random state required).
- Shared evaluation and preprocessing utilities to avoid duplication.
- Typed interfaces with clear validation errors.
- Favor mature libraries (`numpy`, `pandas`, `scikit-learn`) for core operations.

## `BaseModel` Contract

Every model should implement:

- `fit(X, y=None, **kwargs) -> BaseModel`
- `predict(X, **kwargs) -> np.ndarray`

Inherited helper:

- `score(y_true, y_pred, include_n_samples=True, **kwargs) -> MetricsDict`
  - Evaluates an explicit `(y_true, y_pred)` pair with `evaluate_regression(...)`.
  - Useful both inside models and from external evaluation pipelines.

## Baseline Model

`BaselineCFModel` supports user-tunable configuration at initialization (with defaults):

- `name="CollaborativeFiltering"`
- `user_column="customer_id"`
- `movie_column="movie_id"`
- `rating_min=1.0`, `rating_max=5.0`
- `reg_user=10.0`, `reg_movie=10.0`
- `n_iters=15`
- `clip_predictions=True`

### Input Contract

- `X`: `pandas.DataFrame`
  - Must include `user_column` and `movie_column`
  - For training without explicit `y`, must include `rating`
- `y` (optional in `fit`)
  - 1D numeric array-like
  - Same row count as `X`
  - No NaN values

### Output Contract

- `predict(...)` returns `np.ndarray` of predicted ratings.
- `score(...)` returns a metric dictionary including:
  - `rmse`, `mae`, `precision`, `recall`, `accuracy`
  - optional `n_samples`

## Usage

### 1) Load train/test from data package

```python
import aoratos.data as data
from aoratos.models import BaselineCFModel

train_df = data.read("train", source="train")
test_df = data.read("test", source="test")
```

### 2) Fit baseline model

```python
model = BaselineCFModel(
    name="baseline_cf_v1",
    reg_user=8.0,
    reg_movie=12.0,
    n_iters=20,
)

model.fit(train_df)  # uses train_df["rating"] when y is omitted
```

### 3) Predict

```python
X_test = test_df[["customer_id", "movie_id"]]
y_pred = model.predict(X_test)
```

### 4) Evaluate

```python
y_test = test_df["rating"]
y_pred = model.predict(X_test)
metrics = model.score(y_true=y_test, y_pred=y_pred)
print(metrics)
```

## Extending with New Models

When adding a new model:

1. Subclass `BaseModel`.
2. Implement `fit(...)` and `predict(...)`.
3. Set `self.fitted = True` at the end of successful training.
4. Reuse `evaluate_regression(...)` for scoring consistency.
5. Raise package errors from `errors.py` for validation/runtime issues.
6. Add tests in `tests/models/test_<module_name>.py`.

## Notes

- `precision`, `recall`, and `accuracy` are computed on rounded/clipped label space inside `metrics.py`.
- The baseline model intentionally avoids side information and latent factors; use it as a robust reference point before more complex architectures.
