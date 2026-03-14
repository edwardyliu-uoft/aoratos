from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Callable

from .constants import DEFAULT_RAW_ROOT
from .errors import DataError, DataNotFoundError
from .paths import ensure_dir, raw_files_present, resolve_path


def import_kaggle_api() -> object:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download_archive_from_kaggle(raw_root: Path, slug: str) -> Path:
    api = import_kaggle_api()
    ensure_dir(raw_root)
    api.dataset_download_files(slug, path=str(raw_root), unzip=False)
    archive_name = slug.split("/")[-1] + ".zip"
    archive_path = raw_root / archive_name
    if not archive_path.exists():
        raise DataNotFoundError(
            f"Expected downloaded archive at {archive_path}, but it was not found"
        )
    return archive_path


def extract_zip_archive(archive_path: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)


def download(
    raw_root: Path | str | None = None,
    *,
    force: bool = False,
    dataset_slug: str = "netflix-inc/netflix-prize-data",
    download_archive_fn: Callable[[Path, str], Path] | None = None,
    extract_archive_fn: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Download and extract Netflix Prize raw data into data/raw."""

    raw_root = resolve_path(raw_root, DEFAULT_RAW_ROOT)
    ensure_dir(raw_root)

    if raw_files_present(raw_root) and not force:
        return raw_root

    if not os.getenv("KAGGLE_API_TOKEN"):
        raise DataError(
            "Kaggle credentials are missing. Set KAGGLE_API_TOKEN before running download()."
        )

    downloader = download_archive_fn or download_archive_from_kaggle
    extractor = extract_archive_fn or extract_zip_archive

    archive_path = downloader(raw_root, dataset_slug)
    extractor(archive_path, raw_root)
    return raw_root
