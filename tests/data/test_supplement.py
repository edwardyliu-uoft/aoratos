from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from aoratos.data.constants import (
    DEFAULT_SUPPLEMENT_PROGRESS_SUFFIX,
    DEFAULT_SUPPLEMENT_TABLE,
)
from aoratos.data.errors import DataError
from aoratos.data.supplement import supplement


def test_supplement_requires_tmdb_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "compressed" / "movies.parquet"
    target_file = tmp_path / "savepoints" / f"{DEFAULT_SUPPLEMENT_TABLE}.parquet"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"movie_id": 1, "year": 2001, "title": "Movie One"}]).to_parquet(
        source_file, index=False
    )

    monkeypatch.delenv("TMDB_API_KEY", raising=False)

    with pytest.raises(DataError) as error:
        supplement(
            source_file=source_file,
            target_file=target_file,
        )
    assert "TMDB_API_KEY" in str(error.value)


def test_supplement_enriches_and_saves_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "compressed" / "movies.parquet"
    target_file = tmp_path / "savepoints" / f"{DEFAULT_SUPPLEMENT_TABLE}.parquet"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"movie_id": 1, "year": 2001, "title": "Movie One"},
            {"movie_id": 2, "year": 2002, "title": "Unknown Title"},
        ]
    ).to_parquet(source_file, index=False)

    monkeypatch.setenv("TMDB_API_KEY", "fake-key")
    supplement_module = importlib.import_module("aoratos.data.supplement")

    def _search_movie(self, title: str, year: int | None):
        if title == "Movie One":
            return [{"id": 100, "title": "Movie One", "release_date": "2001-01-01"}]
        return []

    def _movie_details(self, tmdb_id: int):
        assert tmdb_id == 100
        return {
            "overview": "A movie overview",
            "genres": [{"id": 1, "name": "Drama"}, {"id": 2, "name": "Comedy"}],
        }

    def _movie_credits(self, tmdb_id: int):
        assert tmdb_id == 100
        return {
            "crew": [{"job": "Director", "name": "Director Name", "id": 200}],
            "cast": [{"order": 0, "name": "Actor Name", "id": 300}],
        }

    monkeypatch.setattr(supplement_module.TMDBClient, "search_movie", _search_movie)
    monkeypatch.setattr(supplement_module.TMDBClient, "movie_details", _movie_details)
    monkeypatch.setattr(supplement_module.TMDBClient, "movie_credits", _movie_credits)

    output = supplement(
        source_file=source_file,
        target_file=target_file,
        force=True,
        batch_size=1,
        requests_per_second=40.0,
    )

    assert target_file.exists()
    assert set(output.columns) == {
        "movie_id",
        "title",
        "year",
        "tmdb_id",
        "genre",
        "description",
        "director",
        "actor",
    }

    matched = output[output["movie_id"] == 1].iloc[0]
    assert int(matched["tmdb_id"]) == 100
    assert matched["genre"] == "Drama"
    assert matched["description"] == "A movie overview"
    assert matched["director"] == "Director Name"
    assert matched["actor"] == "Actor Name"

    unmatched = output[output["movie_id"] == 2].iloc[0]
    assert unmatched["title"] == "Unknown Title"
    assert int(unmatched["year"]) == 2002
    assert pd.isna(unmatched["tmdb_id"])
    assert pd.isna(unmatched["genre"])
    assert pd.isna(unmatched["director"])
    assert pd.isna(unmatched["actor"])


def test_supplement_resumes_from_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_file = tmp_path / "compressed" / "movies.parquet"
    target_file = tmp_path / "savepoints" / f"{DEFAULT_SUPPLEMENT_TABLE}.parquet"
    progress_file = (
        tmp_path
        / "savepoints"
        / f"{DEFAULT_SUPPLEMENT_TABLE}{DEFAULT_SUPPLEMENT_PROGRESS_SUFFIX}.parquet"
    )
    source_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"movie_id": 1, "year": 2001, "title": "Movie One"},
            {"movie_id": 2, "year": 2002, "title": "Movie Two"},
        ]
    ).to_parquet(source_file, index=False)
    pd.DataFrame(
        [
            {
                "movie_id": 1,
                "tmdb_id": 111,
                "genre": "Drama",
                "description": "Already done",
                "director": "Existing Director",
                "actor": "Existing Actor",
            }
        ]
    ).to_parquet(progress_file, index=False)

    monkeypatch.setenv("TMDB_API_KEY", "fake-key")
    supplement_module = importlib.import_module("aoratos.data.supplement")

    calls = {"search": 0}

    def _search_movie(self, title: str, year: int | None):
        calls["search"] += 1
        assert title == "Movie Two"
        return [{"id": 222, "title": "Movie Two", "release_date": "2002-01-01"}]

    def _movie_details(self, tmdb_id: int):
        assert tmdb_id == 222
        return {"overview": "Second movie", "genres": [{"id": 2, "name": "Action"}]}

    def _movie_credits(self, tmdb_id: int):
        assert tmdb_id == 222
        return {
            "crew": [{"job": "Director", "name": "Second Director", "id": 201}],
            "cast": [{"order": 0, "name": "Second Actor", "id": 301}],
        }

    monkeypatch.setattr(supplement_module.TMDBClient, "search_movie", _search_movie)
    monkeypatch.setattr(supplement_module.TMDBClient, "movie_details", _movie_details)
    monkeypatch.setattr(supplement_module.TMDBClient, "movie_credits", _movie_credits)

    output = supplement(
        source_file=source_file,
        target_file=target_file,
        resume=True,
        requests_per_second=40.0,
    )

    assert calls["search"] == 4
    first = output[output["movie_id"] == 1].iloc[0]
    second = output[output["movie_id"] == 2].iloc[0]
    assert int(first["tmdb_id"]) == 111
    assert int(second["tmdb_id"]) == 222
    assert not progress_file.exists()


def test_supplement_emits_periodic_progress_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_file = tmp_path / "compressed" / "movies.parquet"
    target_file = tmp_path / "savepoints" / f"{DEFAULT_SUPPLEMENT_TABLE}.parquet"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"movie_id": 1, "year": 2001, "title": "Movie One"},
            {"movie_id": 2, "year": 2002, "title": "Movie Two"},
            {"movie_id": 3, "year": 2003, "title": "Movie Three"},
        ]
    ).to_parquet(source_file, index=False)

    monkeypatch.setenv("TMDB_API_KEY", "fake-key")
    supplement_module = importlib.import_module("aoratos.data.supplement")

    def _search_movie(self, title: str, year: int | None):
        return [{"id": int(year or 0), "title": title, "release_date": f"{year}-01-01"}]

    def _movie_details(self, tmdb_id: int):
        return {
            "overview": f"Overview {tmdb_id}",
            "genres": [{"id": 1, "name": "Drama"}],
        }

    def _movie_credits(self, tmdb_id: int):
        return {
            "crew": [{"job": "Director", "name": "Director", "id": 1}],
            "cast": [{"order": 0, "name": "Actor", "id": 1}],
        }

    monkeypatch.setattr(supplement_module.TMDBClient, "search_movie", _search_movie)
    monkeypatch.setattr(supplement_module.TMDBClient, "movie_details", _movie_details)
    monkeypatch.setattr(supplement_module.TMDBClient, "movie_credits", _movie_credits)

    caplog.set_level("INFO")
    supplement(
        source_file=source_file,
        target_file=target_file,
        force=True,
        batch_size=1,
        requests_per_second=40.0,
    )

    progress_logs = [
        record.message
        for record in caplog.records
        if "TMDB supplement progress:" in record.message
    ]
    checkpoint_logs = [
        record.message
        for record in caplog.records
        if "TMDB supplement checkpoint saved:" in record.message
    ]

    assert progress_logs
    assert checkpoint_logs
