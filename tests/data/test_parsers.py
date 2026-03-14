from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aoratos.data.errors import DataError
from aoratos.data.parsers import (
    ParsedKVRecord,
    ParsedRatingRecord,
    iterate_kv,
    iterate_ratings,
    kv_to_dataframe,
    read_movies_csv,
    ratings_to_dataframe,
)


def test_iterators_parse_ratings_and_kv(tmp_path: Path) -> None:
    ratings_file = tmp_path / "ratings.txt"
    ratings_file.write_text("1:\n10,5,2005-01-01\n", encoding="utf-8")
    ratings = list(iterate_ratings(ratings_file))
    assert ratings == [
        ParsedRatingRecord(movie_id=1, customer_id=10, rating=5, date="2005-01-01")
    ]

    probe_file = tmp_path / "probe.txt"
    probe_file.write_text(
        "2:\n11\n3:\n27,2025-02-12\n81,2011-10-12\n",
        encoding="utf-8",
    )
    probe = list(iterate_kv(probe_file))
    assert probe == [
        ParsedKVRecord(movie_id=2, customer_id=11),
        ParsedKVRecord(movie_id=3, customer_id=27),
        ParsedKVRecord(movie_id=3, customer_id=81),
    ]


def test_iterators_invalid_row_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("1:\n10,2005-01-01\n", encoding="utf-8")
    with pytest.raises(DataError):
        list(iterate_ratings(bad_file))

    bad_probe = tmp_path / "bad_probe.txt"
    bad_probe.write_text("1:\n11,2005-01-01,extra\n", encoding="utf-8")
    with pytest.raises(DataError):
        list(iterate_kv(bad_probe))


def test_dataframe_builders_and_read_movies_csv(tmp_path: Path) -> None:
    ratings_df = ratings_to_dataframe(
        [ParsedRatingRecord(movie_id=1, customer_id=2, rating=3, date="2005-01-01")]
    )
    assert ratings_df.to_dict(orient="records") == [
        {"movie_id": 1, "customer_id": 2, "rating": 3, "date": "2005-01-01"}
    ]

    kv_df = kv_to_dataframe([ParsedKVRecord(movie_id=1, customer_id=2)])
    assert kv_df.to_dict(orient="records") == [{"movie_id": 1, "customer_id": 2}]

    movie_file = tmp_path / "movie_titles.csv"
    movie_file.write_text(
        '1,2001,"Movie, One"\n2,,Movie Two\n',
        encoding="utf-8",
    )
    movies = read_movies_csv(movie_file)
    assert movies["title"].tolist() == ["Movie, One", "Movie Two"]
    assert str(movies["year"].dtype) == "Int64"
    assert pd.isna(movies.loc[1, "year"])


def test_read_movies_csv_cp1252_fallback(tmp_path: Path) -> None:
    movie_file = tmp_path / "movie_titles.csv"
    movie_file.write_bytes("1,2001,Café\n".encode("cp1252"))

    movies = read_movies_csv(movie_file)

    assert movies["title"].tolist() == ["Café"]
    assert movies["movie_id"].tolist() == [1]
    assert movies["year"].tolist() == [2001]
