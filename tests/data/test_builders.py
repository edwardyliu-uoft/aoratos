from __future__ import annotations

from pathlib import Path

import pandas as pd

from aoratos.data.builders import build, build_test, build_train


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


def test_build_orchestrates_dependencies(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def _download(**kwargs):
        calls.append("download")
        target_dir = kwargs["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def _compress(**kwargs):
        calls.append("compress")
        target_dir = kwargs["target_dir"]
        (target_dir / "ratings").mkdir(parents=True, exist_ok=True)
        return {}

    def _build_train(**kwargs):
        calls.append("build_train")
        target_dir = kwargs["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "movie_id": 1,
                    "customer_id": 10,
                    "date": "2005-01-01",
                    "year": 2001,
                    "title": "M",
                }
            ]
        )
        df.to_parquet(target_dir / "train.parquet", index=False)
        return df

    def _build_test(**kwargs):
        calls.append("build_test")
        target_dir = kwargs["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [{"movie_id": 1, "customer_id": 11, "year": 2001, "title": "M"}]
        )
        df.to_parquet(target_dir / "test.parquet", index=False)
        return df

    monkeypatch.setattr("aoratos.data.builders.download", _download)
    monkeypatch.setattr("aoratos.data.builders.compress", _compress)
    monkeypatch.setattr("aoratos.data.builders.build_train", _build_train)
    monkeypatch.setattr("aoratos.data.builders.build_test", _build_test)

    summary = build(
        raw_dir=tmp_path / "raw",
        compressed_dir=tmp_path / "compressed",
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        savepoints_dir=tmp_path / "savepoints",
        force=True,
    )

    assert calls == ["download", "compress", "build_train", "build_test"]
    assert Path(summary["raw_dir"]).exists()
    assert Path(summary["compressed_dir"]).exists()
    assert Path(summary["train_dir"]).exists()
    assert Path(summary["test_dir"]).exists()
