from __future__ import annotations

from pathlib import Path

import pandas as pd

from aoratos.data.builders import build


def test_build_orchestrates_dependencies(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def _download(**kwargs):
        calls.append("download")
        raw_root = kwargs["raw_root"]
        raw_root.mkdir(parents=True, exist_ok=True)
        return raw_root

    def _compress(**kwargs):
        calls.append("compress")
        compressed_root = kwargs["compressed_root"]
        (compressed_root / "ratings").mkdir(parents=True, exist_ok=True)
        return {}

    def _build_train(**kwargs):
        calls.append("build_train")
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
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
        df.to_parquet(out_dir / "train.parquet", index=False)
        return df

    def _build_test(**kwargs):
        calls.append("build_test")
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [{"movie_id": 1, "customer_id": 11, "year": 2001, "title": "M"}]
        )
        df.to_parquet(out_dir / "test.parquet", index=False)
        return df

    monkeypatch.setattr("aoratos.data.builders.download", _download)
    monkeypatch.setattr("aoratos.data.builders.compress", _compress)
    monkeypatch.setattr("aoratos.data.builders.build_train", _build_train)
    monkeypatch.setattr("aoratos.data.builders.build_test", _build_test)

    summary = build(
        raw_root=tmp_path / "raw",
        compressed_root=tmp_path / "compressed",
        train_root=tmp_path / "train",
        test_root=tmp_path / "test",
        savepoints_root=tmp_path / "savepoints",
        save_intermediate=True,
        force=True,
    )

    assert calls == ["download", "compress", "build_train", "build_test"]
    assert Path(summary["train_path"]).exists()
    assert Path(summary["test_path"]).exists()
    assert (tmp_path / "savepoints" / "train_intermediate.parquet").exists()
    assert (tmp_path / "savepoints" / "test_intermediate.parquet").exists()
