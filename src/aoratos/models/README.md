# Models Package (`aoratos.models`)

This package contains reusable model abstractions, baseline recommender logic, shared metrics, and preprocessing helpers for rating prediction tasks.

## Architecture

The package is organized around a small, composable core:

- `base.py`
  - Defines `BaseModel`, the abstract interface for all models.
  - Standard lifecycle: `fit(...)`, `predict(...)`, `save(...)`, `load(...)`, `score(...)`.
  - Owns common runtime state:
    - `name`: human-readable model identifier
    - `fitted`: boolean flag set by concrete models

- `constants.py`
  - Defines shared model persistence constants.
  - Includes `DEFAULT_MODEL_DIR = Path("data/models")` used by model save/load logic.

- `baseline.py`
  - Implements `BaselineCFModel`, a deterministic collaborative-filtering baseline.
  - Uses global mean + user bias + movie bias:
    - $\hat r(u,m)=\mu+b_u+b_m$
  - Biases are learned with alternating residual updates and regularization.

- `mf.py`
  - Implements `MatrixFactorizationModel`, a regularized biased MF (Funk-SVD style) model.
  - Uses global mean + user bias + movie bias + latent interaction:
    - $\hat r(u,m)=\mu+b_u+b_m+\langle p_u,q_m\rangle$
  - Optionally adds time-bin user/movie drift terms from `date`:
    - $\hat r(u,m,t)=\mu+b_u+b_m+\langle p_u,q_m\rangle+b_{u,t}+b_{m,t}$
  - Trained with deterministic SGD using `random_state`, with cold-start fallback to available terms.

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
- `save(path=None) -> Path`
- `load(path=None, **kwargs) -> BaseModel` (classmethod)

Inherited helper:

- `score(y_true, y_pred, include_n_samples=True, **kwargs) -> MetricsDict`
  - Evaluates an explicit `(y_true, y_pred)` pair with `evaluate_regression(...)`.
  - Useful both inside models and from external evaluation pipelines.

## Baseline Model

`BaselineCFModel` supports user-tunable configuration at initialization (with defaults):

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
- `save(...)` writes to `data/models/<name>.pkl` by default.
- `load(...)` restores model parameters from disk.

## Matrix Factorization Model

`MatrixFactorizationModel` supports user-tunable configuration at initialization (with defaults):

- `user_column="customer_id"`
- `movie_column="movie_id"`
- `date_column="date"`
- `n_factors=64`
- `lr=0.01`
- `reg=0.02`
- `reg_bias=0.01`
- `n_epochs=10`
- `use_time_bias=True`
- `time_bin_freq="M"`
- `rating_min=1.0`, `rating_max=5.0`
- `clip_predictions=True`
- `random_state=42`

### Input Contract

- `X`: `pandas.DataFrame`
  - Must include `user_column` and `movie_column`
  - Must include `date_column` when `use_time_bias=True`
  - For training without explicit `y`, must include `rating`
- `y` (optional in `fit`)
  - 1D numeric array-like
  - Same row count as `X`
  - No NaN values

### Output Contract

- `predict(...)` returns `np.ndarray` of predicted ratings.
- Unknown users/movies fall back toward global/user/item terms when available.
- Unknown time bins during inference default to zero time-bias contribution.
- `save(...)` writes to `data/models/<name>.pkl` by default.
- `load(...)` restores model parameters from disk.

## Usage

Use this section as a pattern for all current and future model classes.

### Shared setup (all models)

```python
import aoratos.data as data

train_df = data.read("train", source="train")
test_df = data.read("test", source="test")
```

### Shared persistence behavior (all models)

- Relative names/paths resolve under `data/models`.
- If no extension is provided, `.pkl` is appended.
- Absolute paths are used as provided.

### Baseline model usage

#### 1) Fit

```python
from aoratos.models import BaselineCFModel

model = BaselineCFModel(
    reg_user=8.0,
    reg_movie=12.0,
    n_iters=20,
)

model.fit(train_df)  # uses train_df["rating"] when y is omitted
```

#### 2) Predict

```python
X_test = test_df[["customer_id", "movie_id"]]
y_pred = model.predict(X_test)
```

#### 3) Evaluate

```python
y_test = test_df["rating"]
y_pred = model.predict(X_test)
metrics = model.score(y_true=y_test, y_pred=y_pred)
print(metrics)
```

#### 4) Save and load

```python
save_path = model.save()  # data/models/CollaborativeFiltering.pkl
loaded_model = BaselineCFModel.load()

# optional custom relative name/path (still rooted in data/models)
model.save("baseline_v2")
loaded_v2 = BaselineCFModel.load(name="baseline_v2")
```

### Matrix factorization model usage

#### 1) Fit

```python
from aoratos.models import MatrixFactorizationModel

mf = MatrixFactorizationModel(
    n_factors=64,
    lr=0.01,
    reg=0.02,
    reg_bias=0.01,
    n_epochs=10,
    use_time_bias=True,
    time_bin_freq="M",
    random_state=42,
)

mf.fit(train_df)
```

#### 2) Predict and evaluate

```python
X_test_mf = test_df[["customer_id", "movie_id", "date"]]
y_pred_mf = mf.predict(X_test_mf)
metrics_mf = mf.score(y_true=test_df["rating"], y_pred=y_pred_mf)
print(metrics_mf)
```

#### 3) Save and load

```python
save_path = mf.save()  # data/models/MatrixFactorization.pkl
loaded_mf = MatrixFactorizationModel.load()

# optional custom relative name/path (still rooted in data/models)
mf.save("mf_experiment_a")
loaded_exp = MatrixFactorizationModel.load(name="mf_experiment_a")
```

### Adding more model examples

As additional models are introduced, add one subsection per model under `Usage` using this structure:

1. `Fit`
2. `Predict`
3. `Evaluate`

This keeps examples consistent and easy to scan across the package.

## Extending with New Models

When adding a new model:

1. Subclass `BaseModel`.
2. Implement `fit(...)`, `predict(...)`, `save(...)`, and `load(...)`.
3. Set `self.fitted = True` at the end of successful training.
4. Reuse `evaluate_regression(...)` for scoring consistency.
5. Raise package errors from `errors.py` for validation/runtime issues.
6. Add tests in `tests/models/test_<module_name>.py`.

## Notes

- `precision`, `recall`, and `accuracy` are computed on rounded/clipped label space inside `metrics.py`.
- The baseline model intentionally avoids side information and latent factors; use it as a robust reference point before more complex architectures.
- The matrix factorization model tracks per-epoch training RMSE in `train_rmse_history` for sanity checks and debugging.
