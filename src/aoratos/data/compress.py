from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

from .constants import (
    DEFAULT_COMPRESSED_DIR,
    DEFAULT_RAW_DIR,
    MB,
    REQUIRED_RAW_FILENAMES,
)
from .errors import DataNotFoundError
from .parsers import (
    ParsedKVRecord,
    ParsedRatingRecord,
    iterate_kv,
    iterate_ratings,
    kv_to_dataframe,
    ratings_to_dataframe,
    read_movies_csv,
)
from .paths import (
    compressed_tables_exist,
    ensure_dir,
    files_exist,
    resolve_path,
)


def _get_compression_summary(
    compressed_dir: Path,
    ratings_dir: Path,
) -> dict[str, Path | list[Path]]:
    return {
        "ratings_dir": ratings_dir,
        "ratings_parts": sorted(ratings_dir.glob("ratings_part_*.parquet")),
        "movies": compressed_dir / "movies.parquet",
        "probe": compressed_dir / "probe.parquet",
        "qualifying": compressed_dir / "qualifying.parquet",
    }


def _estimate_rows_per_part(
    source_files: list[Path],
    *,
    target_part_size_mb: int,
    sample_rows: int = 250_000,
) -> int:
    sample: list[ParsedRatingRecord] = []
    for source_file in source_files:
        for record in iterate_ratings(source_file):
            sample.append(record)
            if len(sample) >= sample_rows:
                break
        if len(sample) >= sample_rows:
            break

    if not sample:
        return 50_000

    sample_df = ratings_to_dataframe(sample)
    buffer = io.BytesIO()
    sample_df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    bytes_per_row = max(1.0, buffer.tell() / len(sample))
    target_bytes = target_part_size_mb * MB
    estimated_rows = int(target_bytes / bytes_per_row)
    return max(50_000, min(12_000_000, estimated_rows))


def _max_part_size_bytes(parts_dir: Path) -> int:
    sizes = [p.stat().st_size for p in parts_dir.glob("*.parquet")]
    return max(sizes) if sizes else 0


def write_parts(
    source_files: Iterable[Path],
    parts_dir: Path,
    *,
    rows_per_part: int,
) -> int:
    ensure_dir(parts_dir)
    for old_part in parts_dir.glob("*.parquet"):
        old_part.unlink()

    part_idx = 0
    buffer: list[ParsedRatingRecord] = []

    for source_file in source_files:
        for record in iterate_ratings(source_file):
            buffer.append(record)
            if len(buffer) >= rows_per_part:
                dataframe = ratings_to_dataframe(buffer)
                target_file = (
                    parts_dir / f"{parts_dir.name}_part_{part_idx:03d}.parquet"
                )
                dataframe.to_parquet(
                    target_file,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                )
                part_idx += 1
                buffer = []

    if buffer:
        dataframe = ratings_to_dataframe(buffer)
        target_file = parts_dir / f"{parts_dir.name}_part_{part_idx:03d}.parquet"
        dataframe.to_parquet(
            target_file,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        part_idx += 1

    return part_idx


def write_block(
    source_file: Path,
    target_file: Path,
) -> None:
    records = list(iterate_kv(source_file))
    df = kv_to_dataframe(records)
    df.to_parquet(target_file, engine="pyarrow", compression="snappy", index=False)


def compress(
    source_dir: Path | str | None = None,
    target_dir: Path | str | None = None,
    *,
    force: bool = False,
    target_part_size_mb: int = 90,
    rows_per_part: int | None = None,
) -> dict[str, Path | list[Path]]:
    """Convert raw Netflix text files to parquet datasets."""

    source_dir = resolve_path(source_dir, DEFAULT_RAW_DIR)
    target_dir = resolve_path(target_dir, DEFAULT_COMPRESSED_DIR)
    ratings_dir = target_dir / "ratings"
    ensure_dir(target_dir)
    ensure_dir(ratings_dir)

    if compressed_tables_exist(target_dir) and not force:
        return _get_compression_summary(target_dir, ratings_dir)

    if not files_exist(source_dir, REQUIRED_RAW_FILENAMES):
        raise DataNotFoundError(
            f"Raw files are missing in {source_dir}. Run download() first."
        )

    source_files = sorted(source_dir.glob("combined_data_*.txt"))
    if not source_files:
        raise DataNotFoundError(f"No combined_data_*.txt files found in {source_dir}")

    effective_rows_per_part = rows_per_part
    if effective_rows_per_part is None:
        # Estimate chunk rows from a sampled parquet footprint for better target sizing.
        effective_rows_per_part = _estimate_rows_per_part(
            source_files,
            target_part_size_mb=target_part_size_mb,
        )

    write_parts(
        source_files,
        ratings_dir,
        rows_per_part=effective_rows_per_part,
    )

    if rows_per_part is None:
        target_bytes = target_part_size_mb * MB
        max_allowed = 100 * MB
        largest_part = _max_part_size_bytes(ratings_dir)
        part_count = len(list(ratings_dir.glob("ratings_part_*.parquet")))

        needs_upsize = part_count > 1 and largest_part < int(target_bytes * 0.7)
        needs_downsize = largest_part > max_allowed

        if needs_upsize or needs_downsize:
            if needs_upsize:
                scale = target_bytes / max(1, largest_part)
                tuned_rows = int(effective_rows_per_part * min(10.0, scale))
            else:
                scale = (95 * MB) / max(1, largest_part)
                tuned_rows = int(effective_rows_per_part * scale)

            write_parts(
                source_files,
                ratings_dir,
                rows_per_part=max(10_000, tuned_rows),
            )

    movies_df = read_movies_csv(source_dir / "movie_titles.csv")
    movies_df.to_parquet(
        target_dir / "movies.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    write_block(
        source_dir / "qualifying.txt",
        target_dir / "qualifying.parquet",
    )
    write_block(
        source_dir / "probe.txt",
        target_dir / "probe.parquet",
    )

    return _get_compression_summary(target_dir, ratings_dir)
