from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aoratos.data.errors import DataAmbiguityError, DataNotFoundError
from aoratos.data.reader import find_parquet_matches, read, read_parquet_dir


def test_read_parquet_dir_and_find_matches(tmp_path: Path) -> None:
    root = tmp_path / "compressed"
    ratings = root / "ratings"
    ratings.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(
        ratings / "ratings_part_000.parquet", index=False
    )
    pd.DataFrame({"x": [2]}).to_parquet(
        ratings / "ratings_part_001.parquet", index=False
    )

    full = read_parquet_dir(ratings)
    assert full["x"].tolist() == [1, 2]

    matches = find_parquet_matches(root, "ratings_part_000")
    assert len(matches) == 1


def test_read_folder_precedence_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", tmp_path)

    root = tmp_path / "compressed"
    ratings = root / "ratings"
    ratings.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(ratings / "a.parquet", index=False)
    pd.DataFrame({"x": [9]}).to_parquet(root / "ratings.parquet", index=False)

    folder_df = read("ratings", source="compressed")
    assert folder_df["x"].tolist() == [1]

    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "b").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(root / "a" / "dup.parquet", index=False)
    pd.DataFrame({"x": [2]}).to_parquet(root / "b" / "dup.parquet", index=False)
    with pytest.raises(DataAmbiguityError):
        read("dup", source="compressed")

    with pytest.raises(DataNotFoundError):
        read("missing", source="compressed")

    with pytest.raises(ValueError):
        read("ratings", source="invalid")


def test_read_train_and_test_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", tmp_path)

    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    train_root.mkdir(parents=True)
    test_root.mkdir(parents=True)

    pd.DataFrame({"movie_id": [1], "customer_id": [10]}).to_parquet(
        train_root / "train.parquet", index=False
    )
    pd.DataFrame({"movie_id": [2], "customer_id": [20]}).to_parquet(
        test_root / "test.parquet", index=False
    )

    train_df = read("train", source="train")
    test_df = read("test", source="test")

    assert train_df["movie_id"].tolist() == [1]
    assert test_df["movie_id"].tolist() == [2]
