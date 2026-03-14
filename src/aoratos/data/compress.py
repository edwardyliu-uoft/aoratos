from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

from .constants import DEFAULT_COMPRESSED_ROOT, DEFAULT_RAW_ROOT
from .errors import DataNotFoundError
from .parsers import (
    ParsedRecord,
    iter_movie_blocks,
    read_movies_csv,
    records_to_dataframe,
)
from .paths import (
    compressed_outputs_present,
    ensure_dir,
    raw_files_present,
    resolve_path,
)

MB = 1024 * 1024


def _compression_summary(
    compressed_root: Path, ratings_dir: Path
) -> dict[str, Path | list[Path]]:
    return {
        "ratings_dir": ratings_dir,
        "ratings_parts": sorted(ratings_dir.glob("ratings_part_*.parquet")),
        "movies": compressed_root / "movies.parquet",
        "probe": compressed_root / "probe.parquet",
        "qualifying": compressed_root / "qualifying.parquet",
    }


def write_ratings_parts(
    combined_files: Iterable[Path],
    ratings_dir: Path,
    *,
    rows_per_chunk: int,
) -> int:
    ensure_dir(ratings_dir)
    for old_part in ratings_dir.glob("ratings_part_*.parquet"):
        old_part.unlink()

    part_idx = 0
    buffer: list[ParsedRecord] = []

    for combined_file in combined_files:
        for rec in iter_movie_blocks(combined_file, has_rating=True, has_date=True):
            buffer.append(rec)
            if len(buffer) >= rows_per_chunk:
                df = records_to_dataframe(
                    buffer, include_rating=True, include_date=True
                )
                out = ratings_dir / f"ratings_part_{part_idx:03d}.parquet"
                df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
                part_idx += 1
                buffer = []

    if buffer:
        df = records_to_dataframe(buffer, include_rating=True, include_date=True)
        out = ratings_dir / f"ratings_part_{part_idx:03d}.parquet"
        df.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
        part_idx += 1

    return part_idx


def _estimate_rows_per_chunk(
    combined_files: list[Path],
    *,
    target_part_size_mb: int,
    sample_rows: int = 250_000,
) -> int:
    sample: list[ParsedRecord] = []
    for combined_file in combined_files:
        for rec in iter_movie_blocks(combined_file, has_rating=True, has_date=True):
            sample.append(rec)
            if len(sample) >= sample_rows:
                break
        if len(sample) >= sample_rows:
            break

    if not sample:
        return 50_000

    sample_df = records_to_dataframe(sample, include_rating=True, include_date=True)
    buffer = io.BytesIO()
    sample_df.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    bytes_per_row = max(1.0, buffer.tell() / len(sample))
    target_bytes = target_part_size_mb * MB
    estimated_rows = int(target_bytes / bytes_per_row)
    return max(50_000, min(12_000_000, estimated_rows))


def _max_part_size_bytes(ratings_dir: Path) -> int:
    sizes = [p.stat().st_size for p in ratings_dir.glob("ratings_part_*.parquet")]
    return max(sizes) if sizes else 0


def write_table_from_block_file(
    source_file: Path,
    output_file: Path,
    *,
    has_rating: bool,
    has_date: bool,
) -> None:
    records = list(
        iter_movie_blocks(source_file, has_rating=has_rating, has_date=has_date)
    )
    df = records_to_dataframe(
        records,
        include_rating=has_rating,
        include_date=has_date,
    )
    df.to_parquet(output_file, engine="pyarrow", compression="snappy", index=False)


def compress(
    raw_root: Path | str | None = None,
    compressed_root: Path | str | None = None,
    *,
    force: bool = False,
    target_part_size_mb: int = 90,
    rows_per_chunk: int | None = None,
) -> dict[str, Path | list[Path]]:
    """Convert raw Netflix text files to parquet datasets."""

    raw_root = resolve_path(raw_root, DEFAULT_RAW_ROOT)
    compressed_root = resolve_path(compressed_root, DEFAULT_COMPRESSED_ROOT)
    ratings_dir = compressed_root / "ratings"
    ensure_dir(compressed_root)
    ensure_dir(ratings_dir)

    if compressed_outputs_present(compressed_root) and not force:
        return _compression_summary(compressed_root, ratings_dir)

    if not raw_files_present(raw_root):
        raise DataNotFoundError(
            f"Raw files are missing in {raw_root}. Run download() first."
        )

    combined_files = sorted(raw_root.glob("combined_data_*.txt"))
    if not combined_files:
        raise DataNotFoundError(f"No combined_data_*.txt files found in {raw_root}")

    effective_rows_per_chunk = rows_per_chunk
    if effective_rows_per_chunk is None:
        # Estimate chunk rows from a sampled parquet footprint for better target sizing.
        effective_rows_per_chunk = _estimate_rows_per_chunk(
            combined_files,
            target_part_size_mb=target_part_size_mb,
        )

    write_ratings_parts(
        combined_files,
        ratings_dir,
        rows_per_chunk=effective_rows_per_chunk,
    )

    if rows_per_chunk is None:
        target_bytes = target_part_size_mb * MB
        max_allowed = 100 * MB
        largest_part = _max_part_size_bytes(ratings_dir)
        part_count = len(list(ratings_dir.glob("ratings_part_*.parquet")))

        needs_upsize = part_count > 1 and largest_part < int(target_bytes * 0.7)
        needs_downsize = largest_part > max_allowed

        if needs_upsize or needs_downsize:
            if needs_upsize:
                scale = target_bytes / max(1, largest_part)
                tuned_rows = int(effective_rows_per_chunk * min(10.0, scale))
            else:
                scale = (95 * MB) / max(1, largest_part)
                tuned_rows = int(effective_rows_per_chunk * scale)

            write_ratings_parts(
                combined_files,
                ratings_dir,
                rows_per_chunk=max(10_000, tuned_rows),
            )

    movies_df = read_movies_csv(raw_root / "movie_titles.csv")
    movies_df.to_parquet(
        compressed_root / "movies.parquet",
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    write_table_from_block_file(
        raw_root / "qualifying.txt",
        compressed_root / "qualifying.parquet",
        has_rating=False,
        has_date=True,
    )
    write_table_from_block_file(
        raw_root / "probe.txt",
        compressed_root / "probe.parquet",
        has_rating=False,
        has_date=False,
    )

    return _compression_summary(compressed_root, ratings_dir)
