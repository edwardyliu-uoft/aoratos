from __future__ import annotations

from pathlib import Path

from aoratos.data.paths import (
    compressed_tables_exist,
    ensure_dir,
    files_exist,
    resolve_path,
)
from aoratos.data.constants import REQUIRED_RAW_FILENAMES


def test_resolve_path_and_ensure_dir(tmp_path: Path) -> None:
    default = tmp_path / "default"
    assert resolve_path(None, default) == default

    p = resolve_path(str(tmp_path / "x"), default)
    assert isinstance(p, Path)
    assert p.name == "x"

    out = ensure_dir(tmp_path / "a" / "b")
    assert out.exists() and out.is_dir()


def test_files_exist(fixture_raw_dir: Path, tmp_path: Path) -> None:
    assert files_exist(fixture_raw_dir, REQUIRED_RAW_FILENAMES)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "combined_data_1.txt").write_text("", encoding="utf-8")
    assert not files_exist(incomplete, REQUIRED_RAW_FILENAMES)


def test_compressed_tables_exist(tmp_path: Path) -> None:
    root = tmp_path / "compressed"
    ratings = root / "ratings"
    ratings.mkdir(parents=True)
    (ratings / "ratings_part_000.parquet").write_text("x", encoding="utf-8")
    (root / "movies.parquet").write_text("x", encoding="utf-8")
    (root / "probe.parquet").write_text("x", encoding="utf-8")
    (root / "qualifying.parquet").write_text("x", encoding="utf-8")
    assert compressed_tables_exist(root)
