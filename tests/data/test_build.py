from __future__ import annotations

from pathlib import Path

import pandas as pd

from aoratos.data.builders import build


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
