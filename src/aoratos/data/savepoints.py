from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DEFAULT_SAVEPOINTS_DIR
from .errors import UnsafeNameError
from .paths import ensure_dir, resolve_path


def safe_save_name(name: str) -> str:
    if not name or name.strip() != name:
        raise UnsafeNameError("Savepoint name cannot be empty or padded with spaces")
    if "/" in name or "\\" in name or ".." in name:
        raise UnsafeNameError("Savepoint name must not contain path separators or '..'")
    return name


def save(
    dataframe: pd.DataFrame,
    name: str,
    target_dir: Path | str | None = None,
) -> Path:
    """Save a dataframe into data/savepoints/<name>.parquet (overwrite)."""

    savepoints_dir = resolve_path(target_dir, DEFAULT_SAVEPOINTS_DIR)
    ensure_dir(savepoints_dir)
    target_file = savepoints_dir / f"{safe_save_name(name)}.parquet"
    dataframe.to_parquet(
        target_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    return target_file
