from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aoratos.data.errors import DataError
from aoratos.data.parsers import (
    ParsedRecord,
    iter_movie_blocks,
    read_movies_csv,
    records_to_dataframe,
)


def test_iter_movie_blocks_ratings_and_probe(tmp_path: Path) -> None:
    ratings_file = tmp_path / "ratings.txt"
    ratings_file.write_text("1:\n10,5,2005-01-01\n", encoding="utf-8")
    ratings = list(iter_movie_blocks(ratings_file, has_rating=True, has_date=True))
    assert ratings == [
        ParsedRecord(movie_id=1, customer_id=10, rating=5, date="2005-01-01")
    ]

    probe_file = tmp_path / "probe.txt"
    probe_file.write_text("2:\n11\n", encoding="utf-8")
    probe = list(iter_movie_blocks(probe_file, has_rating=False, has_date=False))
    assert probe == [ParsedRecord(movie_id=2, customer_id=11, rating=None, date=None)]


def test_iter_movie_blocks_invalid_row_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("1:\n10,2005-01-01\n", encoding="utf-8")
    with pytest.raises(DataError):
        list(iter_movie_blocks(bad_file, has_rating=True, has_date=True))


def test_records_to_dataframe_and_read_movies_csv(tmp_path: Path) -> None:
    df = records_to_dataframe(
        [ParsedRecord(movie_id=1, customer_id=2, rating=3, date="2005-01-01")],
        include_rating=True,
        include_date=True,
    )
    assert df.to_dict(orient="records") == [
        {"movie_id": 1, "customer_id": 2, "rating": 3, "date": "2005-01-01"}
    ]

    movie_file = tmp_path / "movie_titles.csv"
    movie_file.write_text(
        '1,2001,"Movie, One"\n2,,Movie Two\n',
        encoding="utf-8",
    )
    movies = read_movies_csv(movie_file)
    assert movies["title"].tolist() == ["Movie, One", "Movie Two"]
    assert str(movies["year"].dtype) == "Int64"
    assert pd.isna(movies.loc[1, "year"])
