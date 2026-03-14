from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from .compress import compress
from .constants import (
    DEFAULT_COMPRESSED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_SAVEPOINTS_DIR,
    DEFAULT_TEST_DIR,
    DEFAULT_TRAIN_DIR,
)
from .download import download
from .paths import ensure_dir, resolve_path
from .reader import read


def build_train(
    source_dir: Path | str | None = None,
    target_dir: Path | str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build train dataset from qualifying + movie metadata."""

    source_dir = resolve_path(source_dir, DEFAULT_COMPRESSED_DIR)
    target_dir = resolve_path(target_dir, DEFAULT_TRAIN_DIR)
    ensure_dir(target_dir)
    target_file = target_dir / "train.parquet"

    if target_file.exists() and not force:
        return read("train", "train")

    ratings_df = read("ratings", "compressed")
    probe_df = read("probe", "compressed")
    movies_df = read("movies", "compressed")

    train_df = ratings_df.merge(
        probe_df,
        on=["movie_id", "customer_id"],
        how="left_anti",
        sort=False,
    )
    train_df = train_df.merge(movies_df, on="movie_id", how="left", sort=False)
    train_df.to_parquet(
        target_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    return train_df


def build_test(
    source_dir: Path | str | None = None,
    target_dir: Path | str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build test dataset from probe + movie metadata."""

    source_dir = resolve_path(source_dir, DEFAULT_COMPRESSED_DIR)
    target_dir = resolve_path(target_dir, DEFAULT_TEST_DIR)
    ensure_dir(target_dir)
    target_file = target_dir / "test.parquet"

    if target_file.exists() and not force:
        return read("test", "test")

    probe_df = read("probe", "compressed")
    ratings_df = read("ratings", "compressed")
    movies_df = read("movies", "compressed")

    test_df = probe_df.merge(
        ratings_df,
        on=["movie_id", "customer_id"],
        how="left",
        sort=False,
    )
    test_df = test_df.merge(movies_df, on="movie_id", how="left", sort=False)
    test_df.to_parquet(target_file, engine="pyarrow", compression="snappy", index=False)
    return test_df


def build(
    raw_dir: Path | str | None = None,
    compressed_dir: Path | str | None = None,
    train_dir: Path | str | None = None,
    test_dir: Path | str | None = None,
    savepoints_dir: Path | str | None = None,
    *,
    force: bool = False,
    dataset_slug: str = "netflix-inc/netflix-prize-data",
    download_fn: Callable[[Path, str], Path] | None = None,
    extract_fn: Callable[[Path, Path], None] | None = None,
    rows_per_part: int | None = None,
    target_part_size_mb: int = 90,
) -> dict[str, str]:
    """Run complete data build flow (download, compress, build train/test)."""

    raw_dir = resolve_path(raw_dir, DEFAULT_RAW_DIR)
    compressed_dir = resolve_path(compressed_dir, DEFAULT_COMPRESSED_DIR)
    train_dir = resolve_path(train_dir, DEFAULT_TRAIN_DIR)
    test_dir = resolve_path(test_dir, DEFAULT_TEST_DIR)
    savepoints_dir = resolve_path(savepoints_dir, DEFAULT_SAVEPOINTS_DIR)

    download(
        target_dir=raw_dir,
        force=force,
        dataset_slug=dataset_slug,
        download_fn=download_fn,
        extract_fn=extract_fn,
    )
    compress(
        source_dir=raw_dir,
        target_dir=compressed_dir,
        force=force,
        target_part_size_mb=target_part_size_mb,
        rows_per_part=rows_per_part,
    )
    build_train(
        source_dir=compressed_dir,
        target_dir=train_dir,
        force=force,
    )
    build_test(
        source_dir=compressed_dir,
        target_dir=test_dir,
        force=force,
    )

    return {
        "raw_dir": str(raw_dir),
        "compressed_dir": str(compressed_dir),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
    }
