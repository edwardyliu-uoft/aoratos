from __future__ import annotations

from pathlib import Path


def resolve_path(path: Path | str | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def files_exist(target_dir: Path, required_files: set[str]) -> bool:
    existing = {p.name for p in target_dir.glob("*") if p.is_file()}
    return required_files.issubset(existing)


def compressed_tables_exist(compressed_dir: Path) -> bool:
    ratings_dir = compressed_dir / "ratings"
    return (
        ratings_dir.exists()
        and any(ratings_dir.glob("ratings_part_*.parquet"))
        and (compressed_dir / "movies.parquet").exists()
        and (compressed_dir / "probe.parquet").exists()
        and (compressed_dir / "qualifying.parquet").exists()
    )
