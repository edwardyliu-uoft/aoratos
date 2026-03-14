from __future__ import annotations

from pathlib import Path

from aoratos.data.paths import (
    compressed_outputs_present,
    default_raw_root_from_compressed,
    ensure_dir,
    raw_files_present,
    resolve_path,
)


def test_resolve_path_and_ensure_dir(tmp_path: Path) -> None:
    default = tmp_path / "default"
    assert resolve_path(None, default) == default

    p = resolve_path(str(tmp_path / "x"), default)
    assert isinstance(p, Path)
    assert p.name == "x"

    out = ensure_dir(tmp_path / "a" / "b")
    assert out.exists() and out.is_dir()


def test_raw_files_present(fixture_raw_dir: Path, tmp_path: Path) -> None:
    assert raw_files_present(fixture_raw_dir)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "combined_data_1.txt").write_text("", encoding="utf-8")
    assert not raw_files_present(incomplete)


def test_compressed_outputs_present(tmp_path: Path) -> None:
    root = tmp_path / "compressed"
    ratings = root / "ratings"
    ratings.mkdir(parents=True)
    (ratings / "ratings_part_000.parquet").write_text("x", encoding="utf-8")
    (root / "movies.parquet").write_text("x", encoding="utf-8")
    (root / "probe.parquet").write_text("x", encoding="utf-8")
    (root / "qualifying.parquet").write_text("x", encoding="utf-8")
    assert compressed_outputs_present(root)

    assert default_raw_root_from_compressed(root) == root.parent / "raw"
