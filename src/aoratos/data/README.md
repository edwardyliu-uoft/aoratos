# Aoratos Data Module

This module manages end-to-end preparation and access for the Netflix Prize dataset.

## Directory Layout

Expected repository data directories:

- `data/raw/`: Extracted source files (`combined_data_*.txt`, `probe.txt`, `qualifying.txt`, `movie_titles.csv`)
- `data/compressed/`: Parquet outputs
- `data/compressed/ratings/`: Chunked ratings files named `ratings_part_000.parquet`, `ratings_part_001.parquet`, ...
- `data/train/`: Final train output (`train.parquet`)
- `data/test/`: Final test output (`test.parquet`)
- `data/savepoints/`: Named savepoints (`<name>.parquet`)

## Public API

The following functions are exported as `aoratos.data.<fn>`:

- `download(...)`
- `compress(...)`
- `read(name, source="compressed", ...)`
- `build_train(...)`
- `build_test(...)`
- `save(df, name, ...)`
- `build(...)`

## Usage

```python
import aoratos.data as data

# 1) Download and extract raw files from Kaggle
data.download()

# 2) Convert raw files to parquet
data.compress()

# 3) Read full ratings dataset (all parquet parts in ratings folder)
ratings = data.read("ratings", source="compressed")

# 4) Build train and test outputs enriched with movie metadata
train_df = data.build_train()
test_df = data.build_test()

# 5) Save a checkpoint
data.save(train_df.head(1000), "checkpoint")

# 6) Run one-shot build flow
summary = data.build(save_intermediate=True)
print(summary["train_path"], summary["test_path"])
```

## Read Resolution Rules

`read(name, source="compressed")` resolves names in this order:

1. Folder-first lookup: if `<root>/<name>` exists and is a directory, read all `*.parquet` files in lexicographic order and concatenate.
2. Otherwise, recursively search for parquet files where either:
   - filename equals `<name>.parquet`, or
   - path stem equals `<name>`.
3. Exactly one match: read and return it.
4. Multiple matches: raise an ambiguity error listing all candidates.
5. No matches: raise a not-found error with searched root and top-level entries.

`source` options:

- `compressed`: maps to `data/compressed`
- `savepoints`: maps to `data/savepoints`
- `train`: maps to `data/train`
- `test`: maps to `data/test`

Examples:

- `read("ratings", source="compressed")` reads all ratings parts from `data/compressed/ratings/`
- `read("ratings_part_000", source="compressed")` reads one matching file
- `read("checkpoint", source="savepoints")` reads `data/savepoints/checkpoint.parquet`
- `read("train", source="train")` reads `data/train/train.parquet`
- `read("test", source="test")` reads `data/test/test.parquet`

## Build Behavior

`build(...)` performs:

1. `download()`
2. `compress()`
3. `build_train()`
4. `build_test()`

It is idempotent by default and supports `force=True` to regenerate outputs.

`build()` returns a summary containing at least `train_path` and `test_path`.

With `save_intermediate=True`, build also stores stable savepoints:

- `train_intermediate.parquet`
- `test_intermediate.parquet`

## Troubleshooting

### Kaggle credentials

- `download()` requires `KAGGLE_API_TOKEN` in the environment.
- If missing, `download()` raises a clear error.

### Missing raw files

- If `compress()` cannot find required raw files under `data/raw/`, run `download()` first.

### Schema mismatches

- Ratings rows must match `CustomerID,Rating,Date`
- Qualifying rows must match `CustomerID,Date`
- Probe rows must match `CustomerID`
- Movie titles are parsed via CSV reader to safely handle commas in quoted titles.

## Operational Runbook: Remove Existing Large Blobs From Git History

Ignoring directories prevents future commits only. If large files are already in history, clean history and force-push:

```bash
# Install once (macOS)
brew install git-filter-repo

# Run from repository root
git filter-repo \
  --path data/raw --invert-paths \
  --path data/compressed --invert-paths \
  --path data/train --invert-paths \
  --path data/test --invert-paths \
  --path data/savepoints --invert-paths

# Reconnect remote if needed (filter-repo may remove it)
git remote add origin <your-remote-url>

# Force push rewritten history
git push --force --set-upstream origin main
```

Coordinate with collaborators before rewriting shared branch history.
