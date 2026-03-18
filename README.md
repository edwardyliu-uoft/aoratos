# Aoratos

Aoratos is a Python recommender-system project built around the Netflix Prize dataset. It provides:

- A complete data pipeline (download, compression, train/test build, read/save helpers)
- Metadata enrichment for movies via TMDB
- Baseline and matrix-factorization recommendation models
- Shared evaluation metrics and preprocessing helpers

The package is organized under `src/aoratos` with two public subpackages:

- `aoratos.data`
- `aoratos.models`

---

## Requirements

- Python 3.12+
- Main dependencies (from `pyproject.toml`):
	- `kaggle`
	- `pandas`
	- `pyarrow`
	- `scikit-learn`
	- `tqdm`

---

## Installation

### Option 1: `uv` (recommended)

```bash
uv sync
```

### Option 2: `pip`

```bash
python -m pip install -e .
python -m pip install pytest
```

---

## Environment Variables

Two environment variables are used by the data pipeline:

1. `KAGGLE_API_TOKEN` (required for `download()` and `build()`)
2. `TMDB_API_KEY` (required for `supplement()`)

PowerShell:

```powershell
$env:KAGGLE_API_TOKEN = "<your-kaggle-token>"
$env:TMDB_API_KEY = "<your-tmdb-api-key>"
```

Bash:

```bash
export KAGGLE_API_TOKEN="<your-kaggle-token>"
export TMDB_API_KEY="<your-tmdb-api-key>"
```

---

## Project Data Layout

By default, Aoratos reads/writes under `data/`:

- `data/raw` — downloaded Netflix text files
- `data/compressed` — parquet outputs (`ratings/`, `movies.parquet`, `probe.parquet`, `qualifying.parquet`)
- `data/train` — `train.parquet`
- `data/test` — `test.parquet`
- `data/savepoints` — custom savepoints and `movies_metadata.parquet`
- `data/models` — serialized model artifacts (`.pkl`)

---

## Quickstart: End-to-End Pipeline

```python
from aoratos import data

# Requires KAGGLE_API_TOKEN
summary = data.build(force=False)
print(summary)
# {
#   "raw_dir": "data/raw",
#   "compressed_dir": "data/compressed",
#   "train_dir": "data/train",
#   "test_dir": "data/test",
# }
```

`build()` orchestrates:

1. `download()`
2. `compress()`
3. `build_train()`
4. `build_test()`

---

## Data API

Public exports in `aoratos.data`:

- `download`
- `compress`
- `read`
- `build_train`
- `build_test`
- `build`
- `save`
- `supplement`

### Typical step-by-step usage

```python
from aoratos import data

data.download()
data.compress()
train_df = data.build_train()
test_df = data.build_test()

ratings_df = data.read("ratings", source="compressed")
subset_path = data.save(train_df.head(10_000), "train_head")
```

`read(name, source=...)` supports `source` values:

- `compressed`
- `savepoints`
- `train`
- `test`

### Metadata enrichment with TMDB

```python
from aoratos import data

# Requires TMDB_API_KEY
movies_metadata = data.supplement(
	source_file="data/compressed/movies.parquet",
	target_file="data/savepoints/movies_metadata.parquet",
	force=False,
	resume=True,
)
print(movies_metadata.columns.tolist())
```

Output columns:

- `movie_id`
- `tmdb_id`
- `title`
- `year`
- `genre`
- `description`
- `director`
- `actor`

Notes:

- `supplement()` supports checkpoint/resume with a progress parquet file.
- No-match titles are preserved with null TMDB-derived fields.
- Request throttling and retries are built in.

---

## Modeling API

Public exports in `aoratos.models` include:

- `BaseModel`
- `BaselineCFModel`
- `MatrixFactorizationModel`
- `rmse`, `mae`, `precision`, `recall`, `accuracy`, `evaluate_regression`
- `scale_features`, `cosine_similarity_matrix`

### Baseline collaborative filtering model

`BaselineCFModel` predicts:

$$
\hat r(u,m)=\mu+b_u+b_m
$$

Example:

```python
from aoratos import data
from aoratos.models import BaselineCFModel

train_df = data.read("train", source="train")
test_df = data.read("test", source="test")

model = BaselineCFModel(reg_user=8.0, reg_movie=12.0, n_iters=20)
model.fit(train_df)

X_test = test_df[["customer_id", "movie_id"]]
y_test = test_df["rating"]
y_pred = model.predict(X_test)
metrics = model.score(y_true=y_test, y_pred=y_pred)
print(metrics)

model_path = model.save()  # data/models/CollaborativeFiltering.pkl
loaded = BaselineCFModel.load()
```

### Matrix factorization model

`MatrixFactorizationModel` predicts:

$$
\hat r(u,m)=\mu+b_u+b_m+\langle p_u,q_m\rangle
$$

With `use_time_bias=True`, it additionally learns user/movie time-bin drift terms using `date`.

```python
from aoratos import data
from aoratos.models import MatrixFactorizationModel

train_df = data.read("train", source="train")
test_df = data.read("test", source="test")

mf = MatrixFactorizationModel(
	n_factors=64,
	lr=0.01,
	reg=0.02,
	reg_bias=0.01,
	n_epochs=10,
	use_time_bias=True,
	random_state=42,
)
mf.fit(train_df)

X_test = test_df[["customer_id", "movie_id", "date"]]
y_pred = mf.predict(X_test)
metrics = mf.score(y_true=test_df["rating"], y_pred=y_pred)
print(metrics)

mf_path = mf.save()  # data/models/MatrixFactorization.pkl
loaded_mf = MatrixFactorizationModel.load()
```

---

## Development

Run tests:

```bash
pytest
```

Run focused suites:

```bash
pytest tests/data
pytest tests/models
```

---

## Additional Docs

- `docs/strategy.md` — recommender-model strategy and implementation roadmap
- `docs/supplement.md` — TMDB metadata enrichment plan
- `src/aoratos/data/README.md` — detailed data module usage
- `src/aoratos/models/README.md` — detailed models module usage
