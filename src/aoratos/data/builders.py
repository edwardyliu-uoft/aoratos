from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from .compress import compress
from .constants import (
    DEFAULT_COMPRESSED_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_SAVEPOINT_ROOT,
    DEFAULT_TEST_ROOT,
    DEFAULT_TRAIN_ROOT,
)
from .download import download
from .parsers import iter_movie_blocks, read_movies_csv
from .paths import default_raw_root_from_compressed, ensure_dir, resolve_path
from .reader import read as read_dataset
from .savepoints import save


def read_with_fallback(
    parquet_path: Path,
    fallback_loader: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return fallback_loader()


def _normalize_join_columns(df: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in join_cols:
        if col in {"movie_id", "customer_id"} and col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype("Int64")
        elif col == "date" and col in normalized.columns:
            parsed = pd.to_datetime(normalized[col], errors="coerce")
            formatted = parsed.dt.strftime("%Y-%m-%d")
            normalized[col] = formatted.fillna(normalized[col].astype("string")).astype(
                "string"
            )
    return normalized


def _read_ratings_from_raw(
    raw_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for combined_file in sorted(raw_root.glob("combined_data_*.txt")):
        for rec in iter_movie_blocks(combined_file, has_rating=True, has_date=True):
            rows.append(
                {
                    "movie_id": rec.movie_id,
                    "customer_id": rec.customer_id,
                    "date": rec.date,
                    "rating": rec.rating,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["movie_id", "customer_id", "date", "rating"])

    return pd.DataFrame(rows)


def _read_full_ratings(
    compressed_root: Path,
    fallback_raw_root: Path,
) -> pd.DataFrame:
    ratings_dir = compressed_root / "ratings"
    if ratings_dir.exists():
        try:
            return read_dataset(
                "ratings",
                source="compressed",
                compressed_root=compressed_root,
            )
        except Exception:
            # Fall through to raw parsing if compressed ratings are unavailable/corrupt.
            pass
    return _read_ratings_from_raw(fallback_raw_root)


def build_train(
    compressed_root: Path | str | None = None,
    out_dir: Path | str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build train dataset from qualifying + movie metadata."""

    compressed_root = resolve_path(compressed_root, DEFAULT_COMPRESSED_ROOT)
    out_dir = resolve_path(out_dir, DEFAULT_TRAIN_ROOT)
    ensure_dir(out_dir)
    out_file = out_dir / "train.parquet"

    if out_file.exists() and not force:
        return pd.read_parquet(out_file)

    fallback_raw_root = default_raw_root_from_compressed(compressed_root)

    qualifying_df = read_with_fallback(
        compressed_root / "qualifying.parquet",
        lambda: pd.DataFrame(
            [
                {
                    "movie_id": rec.movie_id,
                    "customer_id": rec.customer_id,
                    "date": rec.date,
                }
                for rec in iter_movie_blocks(
                    fallback_raw_root / "qualifying.txt",
                    has_rating=False,
                    has_date=True,
                )
            ]
        ),
    )
    qualifying_df = _normalize_join_columns(
        qualifying_df,
        ["movie_id", "customer_id", "date"],
    )
    ratings_df = _normalize_join_columns(
        _read_full_ratings(compressed_root, fallback_raw_root),
        ["movie_id", "customer_id", "date"],
    )
    movies_df = read_with_fallback(
        compressed_root / "movies.parquet",
        lambda: read_movies_csv(fallback_raw_root / "movie_titles.csv"),
    )

    train_df = qualifying_df.merge(
        ratings_df[["movie_id", "customer_id", "date", "rating"]],
        on=["movie_id", "customer_id", "date"],
        how="left",
        sort=False,
    )
    train_df = train_df.merge(movies_df, on="movie_id", how="left", sort=False)
    train_df.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)
    return train_df


def build_test(
    compressed_root: Path | str | None = None,
    out_dir: Path | str | None = None,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Build test dataset from probe + movie metadata."""

    compressed_root = resolve_path(compressed_root, DEFAULT_COMPRESSED_ROOT)
    out_dir = resolve_path(out_dir, DEFAULT_TEST_ROOT)
    ensure_dir(out_dir)
    out_file = out_dir / "test.parquet"

    if out_file.exists() and not force:
        return pd.read_parquet(out_file)

    fallback_raw_root = default_raw_root_from_compressed(compressed_root)

    probe_df = read_with_fallback(
        compressed_root / "probe.parquet",
        lambda: pd.DataFrame(
            [
                {
                    "movie_id": rec.movie_id,
                    "customer_id": rec.customer_id,
                }
                for rec in iter_movie_blocks(
                    fallback_raw_root / "probe.txt",
                    has_rating=False,
                    has_date=False,
                )
            ]
        ),
    )
    probe_df = _normalize_join_columns(
        probe_df,
        ["movie_id", "customer_id"],
    )
    ratings_df = _normalize_join_columns(
        _read_full_ratings(compressed_root, fallback_raw_root),
        ["movie_id", "customer_id", "date"],
    )
    movies_df = read_with_fallback(
        compressed_root / "movies.parquet",
        lambda: read_movies_csv(fallback_raw_root / "movie_titles.csv"),
    )

    test_df = probe_df.merge(
        ratings_df[["movie_id", "customer_id", "date", "rating"]],
        on=["movie_id", "customer_id"],
        how="left",
        sort=False,
    )
    test_df = test_df.merge(movies_df, on="movie_id", how="left", sort=False)
    test_df.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)
    return test_df


def build(
    raw_root: Path | str | None = None,
    compressed_root: Path | str | None = None,
    train_root: Path | str | None = None,
    test_root: Path | str | None = None,
    savepoints_root: Path | str | None = None,
    *,
    force: bool = False,
    save_intermediate: bool = False,
    dataset_slug: str = "netflix-inc/netflix-prize-data",
    download_archive_fn: Callable[[Path, str], Path] | None = None,
    extract_archive_fn: Callable[[Path, Path], None] | None = None,
    rows_per_chunk: int | None = None,
    target_part_size_mb: int = 90,
) -> dict[str, str]:
    """Run complete data build flow (download, compress, build train/test)."""

    raw_root = resolve_path(raw_root, DEFAULT_RAW_ROOT)
    compressed_root = resolve_path(compressed_root, DEFAULT_COMPRESSED_ROOT)
    train_root = resolve_path(train_root, DEFAULT_TRAIN_ROOT)
    test_root = resolve_path(test_root, DEFAULT_TEST_ROOT)
    savepoints_root = resolve_path(savepoints_root, DEFAULT_SAVEPOINT_ROOT)

    download(
        raw_root=raw_root,
        force=force,
        dataset_slug=dataset_slug,
        download_archive_fn=download_archive_fn,
        extract_archive_fn=extract_archive_fn,
    )
    compress(
        raw_root=raw_root,
        compressed_root=compressed_root,
        force=force,
        rows_per_chunk=rows_per_chunk,
        target_part_size_mb=target_part_size_mb,
    )

    train_df = build_train(
        compressed_root=compressed_root, out_dir=train_root, force=force
    )
    test_df = build_test(
        compressed_root=compressed_root, out_dir=test_root, force=force
    )

    if save_intermediate:
        save(train_df, "train_intermediate", savepoints_root=savepoints_root)
        save(test_df, "test_intermediate", savepoints_root=savepoints_root)

    return {
        "raw_root": str(raw_root),
        "compressed_root": str(compressed_root),
        "train_path": str(train_root / "train.parquet"),
        "test_path": str(test_root / "test.parquet"),
    }
