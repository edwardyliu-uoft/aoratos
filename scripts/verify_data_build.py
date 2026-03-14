from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

MB = 1024 * 1024
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parquet_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def parquet_dir_row_count(path: Path) -> int:
    return sum(parquet_row_count(f) for f in sorted(path.glob("*.parquet")))


def parquet_stats_from_metadata(files: list[Path], col: str):
    gmin = None
    gmax = None
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema.names
        if col not in names:
            return None, None
        idx = names.index(col)
        for rg_i in range(pf.metadata.num_row_groups):
            c = pf.metadata.row_group(rg_i).column(idx)
            s = c.statistics
            if s is None or not s.has_min_max:
                continue
            cmin, cmax = s.min, s.max
            if isinstance(cmin, bytes):
                cmin = cmin.decode("utf-8", errors="ignore")
            if isinstance(cmax, bytes):
                cmax = cmax.decode("utf-8", errors="ignore")
            gmin = cmin if gmin is None or cmin < gmin else gmin
            gmax = cmax if gmax is None or cmax > gmax else gmax
    return gmin, gmax


def sample_dates_ok(ratings_dir: Path, sample_limit: int = 10000) -> bool:
    scanner = ds.dataset(ratings_dir, format="parquet").scanner(
        columns=["date"], batch_size=2048
    )
    seen = 0
    for batch in scanner.to_batches():
        for value in batch.column(0).to_pylist():
            if value is None or not DATE_RE.match(value):
                return False
            seen += 1
            if seen >= sample_limit:
                return True
    return True


def null_count_streaming(parquet_file: Path, column: str) -> int:
    scanner = ds.dataset(parquet_file, format="parquet").scanner(
        columns=[column], batch_size=65536
    )
    total = 0
    for batch in scanner.to_batches():
        total += int(pc.sum(pc.is_null(batch.column(0))).as_py())
    return total


def main() -> None:
    root = Path("data")
    compressed = root / "compressed"
    ratings_dir = compressed / "ratings"
    movies_file = compressed / "movies.parquet"
    qualifying_file = compressed / "qualifying.parquet"
    probe_file = compressed / "probe.parquet"
    train_file = root / "train" / "train.parquet"
    test_file = root / "test" / "test.parquet"

    ratings_files = sorted(ratings_dir.glob("ratings_part_*.parquet"))
    ratings_sizes = [
        round(file_path.stat().st_size / MB, 2) for file_path in ratings_files
    ]

    ratings_rows = parquet_dir_row_count(ratings_dir)
    ratings_movie_min, ratings_movie_max = parquet_stats_from_metadata(
        ratings_files, "movie_id"
    )
    ratings_customer_min, ratings_customer_max = parquet_stats_from_metadata(
        ratings_files, "customer_id"
    )
    ratings_rating_min, ratings_rating_max = parquet_stats_from_metadata(
        ratings_files, "rating"
    )
    ratings_date_min, ratings_date_max = parquet_stats_from_metadata(
        ratings_files, "date"
    )

    movies_df = pd.read_parquet(movies_file)
    year_non_null = movies_df["year"].dropna()
    year_min = int(year_non_null.min()) if len(year_non_null) else None
    year_max = int(year_non_null.max()) if len(year_non_null) else None

    qualifying_rows = parquet_row_count(qualifying_file)
    probe_rows = parquet_row_count(probe_file)
    train_rows = parquet_row_count(train_file)
    test_rows = parquet_row_count(test_file)

    train_schema = pq.read_schema(train_file).names
    test_schema = pq.read_schema(test_file).names

    report = {
        "ratings": {
            "parts_count": len(ratings_files),
            "part_size_mb": {
                "min": min(ratings_sizes) if ratings_sizes else None,
                "max": max(ratings_sizes) if ratings_sizes else None,
                "avg": round(sum(ratings_sizes) / len(ratings_sizes), 2)
                if ratings_sizes
                else None,
            },
            "rows": ratings_rows,
            "movie_id_range": [ratings_movie_min, ratings_movie_max],
            "customer_id_range": [ratings_customer_min, ratings_customer_max],
            "rating_range": [ratings_rating_min, ratings_rating_max],
            "date_range": [ratings_date_min, ratings_date_max],
            "date_format_sample_valid": sample_dates_ok(ratings_dir),
            "checks": {
                "all_parts_lt_100mb": all(size < 100 for size in ratings_sizes),
                "largest_part_in_target_band_70_to_100": (len(ratings_sizes) == 1)
                or (max(ratings_sizes) >= 70 and max(ratings_sizes) < 100),
                "movie_ids_within_1_17770": ratings_movie_min >= 1
                and ratings_movie_max <= 17770,
                "customer_ids_within_1_2649429": ratings_customer_min >= 1
                and ratings_customer_max <= 2649429,
                "ratings_within_1_5": ratings_rating_min >= 1
                and ratings_rating_max <= 5,
            },
        },
        "movies": {
            "rows": int(len(movies_df)),
            "movie_id_min": int(movies_df["movie_id"].min()),
            "movie_id_max": int(movies_df["movie_id"].max()),
            "movie_id_nunique": int(movies_df["movie_id"].nunique()),
            "movie_ids_sequential_1_17770": int(movies_df["movie_id"].nunique())
            == 17770
            and int(movies_df["movie_id"].min()) == 1
            and int(movies_df["movie_id"].max()) == 17770,
            "year_range_non_null": [year_min, year_max],
            "year_within_1890_2005": (year_min is None or year_min >= 1890)
            and (year_max is None or year_max <= 2005),
        },
        "qualifying": {"rows": qualifying_rows},
        "probe": {"rows": probe_rows},
        "train": {
            "rows": train_rows,
            "has_required_columns": {
                "movie_id",
                "customer_id",
                "date",
                "year",
                "title",
            }.issubset(set(train_schema)),
            "row_count_matches_qualifying": train_rows == qualifying_rows,
            "null_title_count": null_count_streaming(train_file, "title"),
        },
        "test": {
            "rows": test_rows,
            "has_required_columns": {
                "movie_id",
                "customer_id",
                "year",
                "title",
            }.issubset(set(test_schema)),
            "row_count_matches_probe": test_rows == probe_rows,
            "null_title_count": null_count_streaming(test_file, "title"),
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
