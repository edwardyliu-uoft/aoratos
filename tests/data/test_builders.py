from __future__ import annotations

from pathlib import Path

import pandas as pd

from aoratos.data.builders import build_test, build_train


def test_build_train_and_test_from_parquet(tmp_path: Path) -> None:
    compressed = tmp_path / "compressed"
    compressed.mkdir()
    pd.DataFrame([{"movie_id": 1, "customer_id": 10, "date": "2005-01-01"}]).to_parquet(
        compressed / "qualifying.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "customer_id": 11}]).to_parquet(
        compressed / "probe.parquet", index=False
    )
    pd.DataFrame([{"movie_id": 1, "year": 2001, "title": "Movie One"}]).to_parquet(
        compressed / "movies.parquet", index=False
    )

    train_out = tmp_path / "train"
    test_out = tmp_path / "test"
    train_df = build_train(compressed_root=compressed, out_dir=train_out, force=True)
    test_df = build_test(compressed_root=compressed, out_dir=test_out, force=True)

    assert (train_out / "train.parquet").exists()
    assert (test_out / "test.parquet").exists()
    assert {"movie_id", "customer_id", "date", "year", "title"}.issubset(
        train_df.columns
    )
    assert {"movie_id", "customer_id", "year", "title"}.issubset(test_df.columns)


def test_builders_fallback_to_raw_and_idempotent(
    fixture_raw_dir: Path, tmp_path: Path
) -> None:
    base = tmp_path / "data"
    raw = base / "raw"
    raw.mkdir(parents=True)
    for p in fixture_raw_dir.iterdir():
        (raw / p.name).write_bytes(p.read_bytes())

    compressed = base / "compressed"
    train_out = base / "train"

    first = build_train(compressed_root=compressed, out_dir=train_out, force=True)
    second = build_train(compressed_root=compressed, out_dir=train_out, force=False)
    assert len(first) == len(second)
