from __future__ import annotations

from pathlib import Path

import pandas as pd

from aoratos.data.builders import build_test, build_train


def test_build_train_and_test_from_parquet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", tmp_path)

    compressed = tmp_path / "compressed"
    (compressed / "ratings").mkdir(parents=True)
    pd.DataFrame([{"movie_id": 1, "customer_id": 10}]).to_parquet(
        compressed / "qualifying.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "customer_id": 11}]).to_parquet(
        compressed / "probe.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "year": 2001, "title": "Movie One"}]).to_parquet(
        compressed / "movies.parquet", index=False
    )
    pd.DataFrame(
        [
            {"movie_id": 1, "customer_id": 10, "rating": 5, "date": "2005-01-01"},
            {"movie_id": 1, "customer_id": 11, "rating": 4, "date": "2005-01-02"},
        ]
    ).to_parquet(compressed / "ratings" / "ratings_part_000.parquet", index=False)

    train_out = tmp_path / "train"
    test_out = tmp_path / "test"
    train_df = build_train(source_dir=compressed, target_dir=train_out, force=True)
    test_df = build_test(source_dir=compressed, target_dir=test_out, force=True)

    assert (train_out / "train.parquet").exists()
    assert (test_out / "test.parquet").exists()
    assert {"movie_id", "customer_id", "date", "rating", "year", "title"}.issubset(
        train_df.columns
    )
    assert {"movie_id", "customer_id", "date", "rating", "year", "title"}.issubset(
        test_df.columns
    )


def test_build_train_idempotent_when_output_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", tmp_path)

    compressed = tmp_path / "compressed"
    (compressed / "ratings").mkdir(parents=True)
    pd.DataFrame([{"movie_id": 1, "customer_id": 10}]).to_parquet(
        compressed / "qualifying.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "customer_id": 11}]).to_parquet(
        compressed / "probe.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "year": 2001, "title": "Movie One"}]).to_parquet(
        compressed / "movies.parquet", index=False
    )
    pd.DataFrame(
        [{"movie_id": 1, "customer_id": 10, "rating": 5, "date": "2005-01-01"}]
    ).to_parquet(compressed / "ratings" / "ratings_part_000.parquet", index=False)

    train_out = tmp_path / "train"
    first = build_train(source_dir=compressed, target_dir=train_out, force=True)
    second = build_train(source_dir=compressed, target_dir=train_out, force=False)
    assert len(first) == len(second)


def test_build_train_matches_ratings_when_join_key_dtype_differs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", tmp_path)

    compressed = tmp_path / "compressed"
    (compressed / "ratings").mkdir(parents=True)

    pd.DataFrame([{"movie_id": "1", "customer_id": "10"}]).to_parquet(
        compressed / "qualifying.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "customer_id": 11}]).to_parquet(
        compressed / "probe.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "year": 2001, "title": "Movie One"}]).to_parquet(
        compressed / "movies.parquet", index=False
    )
    pd.DataFrame(
        [{"movie_id": 1, "customer_id": 10, "rating": 5, "date": "2005-01-01"}]
    ).to_parquet(compressed / "ratings" / "ratings_part_000.parquet", index=False)

    out_dir = tmp_path / "train"
    train_df = build_train(source_dir=compressed, target_dir=out_dir, force=True)

    assert train_df.shape[0] == 1
    assert train_df["rating"].isnull().sum() == 0
    assert int(train_df.loc[0, "rating"]) == 5
