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


def test_read_folder_precedence_and_errors(tmp_path: Path) -> None:
    root = tmp_path / "compressed"
    ratings = root / "ratings"
    ratings.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(ratings / "a.parquet", index=False)
    pd.DataFrame({"x": [9]}).to_parquet(root / "ratings.parquet", index=False)

    folder_df = read("ratings", source="compressed", compressed_root=root)
    assert folder_df["x"].tolist() == [1]

    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "b").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(root / "a" / "dup.parquet", index=False)
    pd.DataFrame({"x": [2]}).to_parquet(root / "b" / "dup.parquet", index=False)
    with pytest.raises(DataAmbiguityError):
        read("dup", source="compressed", compressed_root=root)

    with pytest.raises(DataNotFoundError):
        read("missing", source="compressed", compressed_root=root)

    with pytest.raises(ValueError):
        read("ratings", source="invalid", compressed_root=root)
