from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Callable

from .constants import DEFAULT_RAW_DIR, REQUIRED_RAW_FILENAMES
from .errors import DataError, DataNotFoundError
from .paths import ensure_dir, files_exist, resolve_path


def import_kaggle_api() -> object:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download_zip_from_kaggle(slug: str, target_dir: Path) -> Path:
    api = import_kaggle_api()
    ensure_dir(target_dir)
    api.dataset_download_files(slug, path=str(target_dir), unzip=False)
    zip_name = slug.split("/")[-1] + ".zip"
    zip_path = target_dir / zip_name
    if not zip_path.exists():
        raise DataNotFoundError(
            f"Expected downloaded zip file at '{zip_path}', but it was not found"
        )
    return zip_path


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def download(
    target_dir: Path | str | None = None,
    *,
    force: bool = False,
    dataset_slug: str = "netflix-inc/netflix-prize-data",
    download_fn: Callable[[str, Path], Path] | None = None,
    extract_fn: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Download and extract Netflix Prize raw data into data/raw."""

    target_dir = resolve_path(target_dir, DEFAULT_RAW_DIR)
    ensure_dir(target_dir)

    if files_exist(target_dir, REQUIRED_RAW_FILENAMES) and not force:
        return target_dir

    if not os.getenv("KAGGLE_API_TOKEN"):
        raise DataError(
            "Kaggle credentials are missing. Set KAGGLE_API_TOKEN before running download()."
        )

    downloader = download_fn or download_zip_from_kaggle
    extractor = extract_fn or extract_zip

    zip_path = downloader(dataset_slug, target_dir)
    extractor(zip_path, target_dir)
    return target_dir
