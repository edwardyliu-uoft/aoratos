from __future__ import annotations

import zipfile
from pathlib import Path

import pytest


def write_fixture_raw_files(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "combined_data_1.txt").write_text(
        "1:\n10,5,2005-01-01\n11,3,2005-01-02\n2:\n12,4,2005-01-03\n",
        encoding="utf-8",
    )
    (target / "combined_data_2.txt").write_text(
        "3:\n13,2,2005-01-04\n",
        encoding="utf-8",
    )
    (target / "combined_data_3.txt").write_text("", encoding="utf-8")
    (target / "combined_data_4.txt").write_text("", encoding="utf-8")
    (target / "qualifying.txt").write_text(
        "1:\n10\n2:\n12\n",
        encoding="utf-8",
    )
    (target / "probe.txt").write_text(
        "1:\n11\n3:\n13,2005-01-04\n",
        encoding="utf-8",
    )
    (target / "movie_titles.csv").write_text(
        '1,2001,"Movie, One"\n2,2002,Movie Two\n3,,Movie Three\n',
        encoding="utf-8",
    )


@pytest.fixture
def fixture_raw_dir(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "source_raw"
    write_fixture_raw_files(raw_dir)
    return raw_dir


@pytest.fixture
def fixture_archive(tmp_path: Path, fixture_raw_dir: Path) -> Path:
    archive = tmp_path / "netflix-prize-data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in fixture_raw_dir.iterdir():
            zf.write(p, arcname=p.name)
    return archive
