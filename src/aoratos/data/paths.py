from __future__ import annotations

from pathlib import Path

from .constants import REQUIRED_RAW_FILENAMES


def resolve_path(path: Path | str | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_files_present(raw_root: Path) -> bool:
    existing = {p.name for p in raw_root.glob("*") if p.is_file()}
    return REQUIRED_RAW_FILENAMES.issubset(existing)


def compressed_outputs_present(compressed_root: Path) -> bool:
    ratings_dir = compressed_root / "ratings"
    return (
        ratings_dir.exists()
        and any(ratings_dir.glob("ratings_part_*.parquet"))
        and (compressed_root / "movies.parquet").exists()
        and (compressed_root / "probe.parquet").exists()
        and (compressed_root / "qualifying.parquet").exists()
    )


def default_raw_root_from_compressed(compressed_root: Path) -> Path:
    return compressed_root.parent / "raw"
