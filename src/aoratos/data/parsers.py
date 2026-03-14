from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from .errors import DataError


@dataclass(frozen=True)
class ParsedRecord:
    movie_id: int
    customer_id: int
    rating: int | None = None
    date: str | None = None


def iter_movie_blocks(
    file_path: Path,
    *,
    has_rating: bool,
    has_date: bool,
) -> Iterator[ParsedRecord]:
    movie_id: int | None = None
    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(":"):
                movie_id = int(line[:-1])
                continue
            if movie_id is None:
                raise DataError(f"Found data row before movie header in {file_path}")

            parts = line.split(",")
            if has_rating and has_date and len(parts) != 3:
                raise DataError(f"Unexpected ratings row format in {file_path}: {line}")
            if (not has_rating) and has_date and len(parts) != 2:
                raise DataError(
                    f"Unexpected qualifying row format in {file_path}: {line}"
                )
            if (not has_rating) and (not has_date) and len(parts) != 1:
                raise DataError(f"Unexpected probe row format in {file_path}: {line}")

            customer_id = int(parts[0])
            rating = int(parts[1]) if has_rating else None
            date = (
                parts[2]
                if (has_rating and has_date)
                else (parts[1] if has_date else None)
            )

            yield ParsedRecord(
                movie_id=movie_id,
                customer_id=customer_id,
                rating=rating,
                date=date,
            )


def records_to_dataframe(
    records: list[ParsedRecord], include_rating: bool, include_date: bool
) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "movie_id": [r.movie_id for r in records],
        "customer_id": [r.customer_id for r in records],
    }
    if include_rating:
        data["rating"] = [r.rating for r in records]
    if include_date:
        data["date"] = [r.date for r in records]
    return pd.DataFrame(data)


def read_movies_csv(movie_file: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with movie_file.open("r", encoding="latin-1", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            movie_id = int(row[0])
            year = pd.NA
            if row[1].strip():
                try:
                    year = int(row[1])
                except ValueError:
                    year = pd.NA
            title = ",".join(row[2:])
            rows.append({"movie_id": movie_id, "year": year, "title": title})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["year"] = df["year"].astype("Int64")
    return df
