from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DEFAULT_COMPRESSED_ROOT, DEFAULT_SAVEPOINT_ROOT
from .constants import DEFAULT_TEST_ROOT, DEFAULT_TRAIN_ROOT
from .errors import DataAmbiguityError, DataNotFoundError
from .paths import resolve_path


def read_parquet_dir(path: Path) -> pd.DataFrame:
    parquet_files = sorted(path.glob("*.parquet"))
    if not parquet_files:
        raise DataNotFoundError(f"Directory {path} contains no parquet files")
    if len(parquet_files) == 1:
        return pd.read_parquet(parquet_files[0])
    frames = [pd.read_parquet(p) for p in parquet_files]
    return pd.concat(frames, ignore_index=True)


def find_parquet_matches(root: Path, name: str) -> list[Path]:
    matches: list[Path] = []
    for p in root.rglob("*.parquet"):
        if p.name == f"{name}.parquet" or p.stem == name:
            matches.append(p)
    return sorted(matches)


def read(
    name: str,
    source: str = "compressed",
    *,
    compressed_root: Path | str | None = None,
    savepoints_root: Path | str | None = None,
    train_root: Path | str | None = None,
    test_root: Path | str | None = None,
) -> pd.DataFrame:
    """Read parquet datasets by logical name from supported source roots."""

    if source not in {"compressed", "savepoints", "train", "test"}:
        raise ValueError("source must be one of: compressed, savepoints, train, test")

    compressed_root = resolve_path(compressed_root, DEFAULT_COMPRESSED_ROOT)
    savepoints_root = resolve_path(savepoints_root, DEFAULT_SAVEPOINT_ROOT)
    train_root = resolve_path(train_root, DEFAULT_TRAIN_ROOT)
    test_root = resolve_path(test_root, DEFAULT_TEST_ROOT)

    source_roots = {
        "compressed": compressed_root,
        "savepoints": savepoints_root,
        "train": train_root,
        "test": test_root,
    }
    root = source_roots[source]

    direct = root / name
    if direct.exists() and direct.is_dir():
        return read_parquet_dir(direct)

    matches = find_parquet_matches(root, name)
    if len(matches) == 1:
        return pd.read_parquet(matches[0])
    if len(matches) > 1:
        candidates = "\n".join(str(p) for p in matches)
        raise DataAmbiguityError(
            f"Ambiguous parquet dataset name '{name}'. Candidates:\n{candidates}"
        )

    top_entries = sorted(p.name for p in root.glob("*")) if root.exists() else []
    raise DataNotFoundError(
        f"No dataset named '{name}' found under {root}. Top-level entries: {top_entries}"
    )
