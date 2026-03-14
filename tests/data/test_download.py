from __future__ import annotations

from pathlib import Path

import pytest

from aoratos.data.download import download
from aoratos.data.errors import DataError


def test_download_requires_kaggle_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    with pytest.raises(DataError):
        download(raw_root=tmp_path / "raw", force=True)


def test_download_skips_when_raw_already_present(
    fixture_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    def _fail_download(_raw: Path, _slug: str) -> Path:
        raise AssertionError("downloader should not be called")

    out = download(raw_root=fixture_raw_dir, download_archive_fn=_fail_download)
    assert out == fixture_raw_dir


def test_download_uses_injected_downloader_and_extractor(
    tmp_path: Path,
    fixture_archive: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "token")
    raw = tmp_path / "raw"

    called = {"download": False, "extract": False}

    def _download(_raw: Path, _slug: str) -> Path:
        called["download"] = True
        return fixture_archive

    def _extract(archive: Path, target: Path) -> None:
        called["extract"] = True
        import zipfile

        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)

    download(
        raw_root=raw,
        force=True,
        download_archive_fn=_download,
        extract_archive_fn=_extract,
    )
    assert called == {"download": True, "extract": True}
    assert (raw / "movie_titles.csv").exists()
