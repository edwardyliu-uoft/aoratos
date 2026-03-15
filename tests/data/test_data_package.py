from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import aoratos.data as data
from aoratos.data.errors import DataAmbiguityError, DataNotFoundError


def _download_from_fixture(archive: Path):
    def _fn(_slug: str, _target_dir: Path) -> Path:
        return archive

    return _fn


def test_full_flow_download_compress_build_read_build(
    tmp_path: Path,
    fixture_archive: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-token")
    data_root = tmp_path / "data"
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", data_root)

    raw_root = data_root / "raw"
    compressed_root = data_root / "compressed"
    train_root = data_root / "train"
    test_root = data_root / "test"
    savepoints_root = data_root / "savepoints"

    data.download(
        target_dir=raw_root,
        force=True,
        download_fn=_download_from_fixture(fixture_archive),
    )
    compression_summary = data.compress(
        source_dir=raw_root,
        target_dir=compressed_root,
        force=True,
        rows_per_part=2,
    )

    ratings_parts = compression_summary["ratings_parts"]
    assert len(ratings_parts) >= 2
    assert ratings_parts[0].name == "ratings_part_000.parquet"
    assert ratings_parts[1].name == "ratings_part_001.parquet"

    ratings_df = data.read("ratings", source="compressed")
    assert len(ratings_df) == 4

    part_df = data.read("ratings_part_000", source="compressed")
    assert len(part_df) == 2

    train_df = data.build_train(
        source_dir=compressed_root, target_dir=train_root, force=True
    )
    assert (train_root / "train.parquet").exists()
    assert {"movie_id", "customer_id", "date", "rating", "year", "title"}.issubset(
        train_df.columns
    )

    test_df = data.build_test(
        source_dir=compressed_root, target_dir=test_root, force=True
    )
    assert (test_root / "test.parquet").exists()
    assert {"movie_id", "customer_id", "date", "rating", "year", "title"}.issubset(
        test_df.columns
    )

    ckpt_1 = pd.DataFrame({"value": [1]})
    ckpt_2 = pd.DataFrame({"value": [2]})
    data.save(ckpt_1, "checkpoint", target_dir=savepoints_root)
    data.save(ckpt_2, "checkpoint", target_dir=savepoints_root)
    loaded_ckpt = data.read("checkpoint", source="savepoints")
    assert loaded_ckpt["value"].tolist() == [2]

    build_raw = tmp_path / "build" / "raw"
    build_compressed = tmp_path / "build" / "compressed"
    build_train = tmp_path / "build" / "train"
    build_test = tmp_path / "build" / "test"
    build_savepoints = tmp_path / "build" / "savepoints"

    build_summary = data.build(
        raw_dir=build_raw,
        compressed_dir=build_compressed,
        train_dir=build_train,
        test_dir=build_test,
        savepoints_dir=build_savepoints,
        force=True,
        rows_per_part=2,
        download_fn=_download_from_fixture(fixture_archive),
    )

    assert Path(build_summary["raw_dir"]).exists()
    assert Path(build_summary["compressed_dir"]).exists()
    assert Path(build_summary["train_dir"]).exists()
    assert Path(build_summary["test_dir"]).exists()
    build_train_df = pd.read_parquet(Path(build_summary["train_dir"]) / "train.parquet")
    build_test_df = pd.read_parquet(Path(build_summary["test_dir"]) / "test.parquet")
    assert not build_train_df.empty
    assert not build_test_df.empty


def test_read_ambiguity_and_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", tmp_path)

    compressed_root = tmp_path / "compressed"
    (compressed_root / "a").mkdir(parents=True, exist_ok=True)
    (compressed_root / "b").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(
        compressed_root / "a" / "dup.parquet", index=False
    )
    pd.DataFrame({"x": [2]}).to_parquet(
        compressed_root / "b" / "dup.parquet", index=False
    )

    with pytest.raises(DataAmbiguityError) as ambiguity:
        data.read("dup", source="compressed")
    assert "Candidates" in str(ambiguity.value)

    with pytest.raises(DataNotFoundError) as not_found:
        data.read("missing", source="compressed")
    assert "Top-level entries" in str(not_found.value)
