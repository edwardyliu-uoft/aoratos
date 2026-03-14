from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import DEFAULT_SAVEPOINT_ROOT
from .errors import UnsafeNameError
from .paths import ensure_dir, resolve_path


def safe_save_name(name: str) -> str:
    if not name or name.strip() != name:
        raise UnsafeNameError("Savepoint name cannot be empty or padded with spaces")
    if "/" in name or "\\" in name or ".." in name:
        raise UnsafeNameError("Savepoint name must not contain path separators or '..'")
    return name


def save(
    df: pd.DataFrame,
    name: str,
    *,
    savepoints_root: Path | str | None = None,
) -> Path:
    """Save a dataframe into data/savepoints/<name>.parquet (overwrite)."""

    savepoints_root = resolve_path(savepoints_root, DEFAULT_SAVEPOINT_ROOT)
    ensure_dir(savepoints_root)
    out = savepoints_root / f"{safe_save_name(name)}.parquet"
    df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    return out
