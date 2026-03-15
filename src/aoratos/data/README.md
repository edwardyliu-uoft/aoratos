# `aoratos.data` usage

This module provides a full data pipeline for the Netflix Prize dataset:

1. `download()` raw files from Kaggle
2. `compress()` raw text files into parquet tables
3. `build_train()` and `build_test()` datasets
4. `read()` datasets back by logical name
5. `save()` custom dataframe savepoints

Public API exports:

- `download`
- `compress`
- `read`
- `build_train`
- `build_test`
- `build`
- `save`
- `supplement`

## Requirements

- Python 3.12+
- Packages: `kaggle`, `pandas`, `pyarrow`

## Important: Kaggle token required for downloads

`download()` (and the orchestration `build()` function, which calls `download()`) requires `KAGGLE_API_TOKEN` to be set.

If `KAGGLE_API_TOKEN` is not present, `download()` raises `DataError` with:

`Kaggle credentials are missing. Set KAGGLE_API_TOKEN before running download().`

Example (PowerShell):

```powershell
$env:KAGGLE_API_TOKEN = "<your-kaggle-api-token>"
```

Example (bash):

```bash
export KAGGLE_API_TOKEN="<your-kaggle-api-token>"
```

## Important: TMDB key required for metadata enrichment

`supplement()` requires `TMDB_API_KEY` to be set.

If `TMDB_API_KEY` is missing, `supplement()` raises `DataError` with:

`TMDB API key is missing. Set TMDB_API_KEY before running supplement().`

Example (PowerShell):

```powershell
$env:TMDB_API_KEY = "<your-tmdb-api-key>"
```

Example (bash):

```bash
export TMDB_API_KEY="<your-tmdb-api-key>"
```

## Default directories

By default, this module reads/writes under `data/`:

- Raw: `data/raw`
- Compressed parquet: `data/compressed`
- Train: `data/train`
- Test: `data/test`
- Savepoints: `data/savepoints`

## Quickstart (end-to-end)

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

`build()` runs the full flow in order:

1. `download()`
2. `compress()`
3. `build_train()`
4. `build_test()`

## Step-by-step usage

### 1) Download raw data

```python
from aoratos.data import download

raw_dir = download(
	target_dir="data/raw",
	force=False,
	dataset_slug="netflix-inc/netflix-prize-data",
)
print(raw_dir)
```

Notes:

- If required raw files already exist and `force=False`, download is skipped.
- `force=True` re-runs download/extraction.

### 2) Compress raw files to parquet

```python
from aoratos.data import compress

compressed = compress(
	source_dir="data/raw",
	target_dir="data/compressed",
	force=False,
	target_part_size_mb=90,
)
print(compressed["ratings_dir"])
print(len(compressed["ratings_parts"]))
```

Notes:

- Requires raw files to exist; otherwise raises `DataNotFoundError`.
- Ratings are split into `ratings_part_*.parquet` under `data/compressed/ratings`.

### 3) Build train/test datasets

```python
from aoratos.data import build_train, build_test

train_df = build_train(
	source_dir="data/compressed",
	target_dir="data/train",
	force=False,
)

test_df = build_test(
	source_dir="data/compressed",
	target_dir="data/test",
	force=False,
)

print(train_df.shape, test_df.shape)
```

### 4) Read datasets by logical name

```python
from aoratos.data import read

ratings_df = read("ratings", source="compressed")
movies_df = read("movies", source="compressed")
train_df = read("train", source="train")
test_df = read("test", source="test")
```

Supported `source` values:

- `compressed`
- `savepoints`
- `train`
- `test`

### 5) Save custom savepoints

```python
from aoratos.data import read, save

train_df = read("train", source="train")
subset = train_df[["movie_id", "customer_id", "rating"]].head(1000)

path = save(subset, "train_sample")
print(path)  # data/savepoints/train_sample.parquet
```

Savepoint names must be safe (no path separators or `..`).

## Typical workflow

```python
from aoratos import data

data.download()
data.compress()
data.build_train()
data.build_test()

train = data.read("train", "train")
data.save(train.head(10_000), "train_head")
```

## Supplement: movie metadata enrichment

```python
from aoratos import data

metadata = data.supplement(
	source_dir="data/compressed",
	target_dir="data/savepoints",
	force=False,
	resume=True,
)

print(metadata.shape)
```

Output file:

- `data/savepoints/movies_metadata.parquet`

This output is the enriched replacement metadata table for the original `movies.parquet` table, preserving `movie_id`, `title`, and `year` while adding TMDB fields.

No-match behavior:

- If TMDB returns no search results, the row still keeps source `title` and `year`; only TMDB-derived fields (`tmdb_id`, `genre`, `description`, `director`, `actor`) remain null.
- If TMDB returns any candidates, `supplement()` selects the highest-likelihood candidate deterministically.

Progress behavior:

- `supplement()` emits periodic progress and checkpoint logs during long runs so users can confirm the pipeline is still advancing.

Output columns:

- `movie_id`
- `tmdb_id`
- `title`
- `year`
- `genre`
- `description`
- `director`
- `actor`

Implementation notes:

- `supplement()` uses shared defaults from `aoratos.data.constants` (TMDB URL, throttle/retry defaults, matching thresholds, source/target defaults).
- Supplement-specific exception classes are centralized in `aoratos.data.errors` and used by `supplement()`.

## Error behavior at a glance

- `DataError`: generic data pipeline errors (including missing `KAGGLE_API_TOKEN` for download)
- `DataNotFoundError`: expected files/datasets were not found
- `DataAmbiguityError`: `read()` matched more than one dataset for a name
- `UnsafeNameError`: unsafe savepoint name passed to `save()`
- `SupplementConfigurationError`: invalid supplement runtime configuration
- `TMDBAuthenticationError`: `TMDB_API_KEY` not set for supplement
- `DataNotFoundError`: supplement source parquet missing
- `SupplementSchemaError`: source schema missing required supplement columns
- `TMDBRequestError`: terminal TMDB API/network failure after bounded retries
