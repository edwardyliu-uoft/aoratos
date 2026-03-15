from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .constants import (
    DEFAULT_SUPPLEMENT_PROGRESS_SUFFIX,
    DEFAULT_SUPPLEMENT_SOURCE_FILE,
    DEFAULT_SUPPLEMENT_TARGET_FILE,
    DEFAULT_TMDB_BACKOFF_BASE_SECONDS,
    DEFAULT_TMDB_BACKOFF_JITTER_SECONDS,
    DEFAULT_TMDB_BACKOFF_MAX_SECONDS,
    DEFAULT_TMDB_BATCH_SIZE,
    DEFAULT_TMDB_MAX_RETRIES,
    DEFAULT_TMDB_REQUESTS_PER_SECOND,
    DEFAULT_TMDB_TIMEOUT_SECONDS,
    REQUIRED_SUPPLEMENT_SOURCE_COLUMNS,
    REQUIRED_SUPPLEMENT_TARGET_COLUMNS,
    TMDB_API_URL,
    TMDB_MAX_REQUESTS_PER_SECOND,
)
from .errors import (
    DataNotFoundError,
    SupplementConfigurationError,
    SupplementInternalError,
    SupplementSchemaError,
    TMDBAuthenticationError,
    TMDBRequestError,
)
from .paths import ensure_dir, resolve_path

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise SupplementConfigurationError("requests_per_second must be > 0")
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()


@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = DEFAULT_TMDB_MAX_RETRIES
    base_backoff_seconds: float = DEFAULT_TMDB_BACKOFF_BASE_SECONDS
    max_backoff_seconds: float = DEFAULT_TMDB_BACKOFF_MAX_SECONDS
    jitter_seconds: float = DEFAULT_TMDB_BACKOFF_JITTER_SECONDS


def _emit_status(message: str) -> None:
    print(message, flush=True)
    logger.info(message)


def _normalize_title(value: str | None) -> str:
    if not value:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _extract_year(release_date: str | None) -> int | None:
    if not release_date:
        return None
    try:
        return datetime.strptime(release_date, "%Y-%m-%d").year
    except ValueError:
        return None


def _title_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _build_confidence_score(
    query_title: str,
    query_year: int | None,
    candidate: dict[str, Any],
) -> tuple[float, float, int, float]:
    normalized_query = _normalize_title(query_title)
    candidate_title = _normalize_title(
        candidate.get("title") or candidate.get("original_title") or ""
    )
    title_score = _title_similarity(normalized_query, candidate_title)

    release_year = _extract_year(candidate.get("release_date"))
    year_difference = 99
    if query_year is not None and release_year is not None:
        year_difference = abs(query_year - release_year)
        if year_difference == 0:
            year_score = 1.0
        elif year_difference == 1:
            year_score = 0.8
        elif year_difference == 2:
            year_score = 0.5
        else:
            year_score = 0.0
    else:
        year_score = 0.4

    popularity = float(candidate.get("popularity") or 0.0)
    popularity_score = min(popularity / 100.0, 1.0)
    confidence = (title_score * 0.75) + (year_score * 0.2) + (popularity_score * 0.05)
    return confidence, title_score, year_difference, popularity


def _get_primary_director(crew: list[dict[str, Any]]) -> str | None:
    directors = [item for item in crew if item.get("job") == "Director"]
    if not directors:
        return None
    ranked = sorted(
        directors,
        key=lambda item: (
            -float(item.get("popularity") or 0.0),
            str(item.get("name") or "").lower(),
            int(item.get("id") or 0),
        ),
    )
    return ranked[0].get("name")


def _get_primary_actor(cast: list[dict[str, Any]]) -> str | None:
    if not cast:
        return None
    ranked = sorted(
        cast,
        key=lambda item: (
            int(item.get("order") if item.get("order") is not None else 1_000_000),
            -float(item.get("popularity") or 0.0),
            int(item.get("id") or 0),
        ),
    )
    return ranked[0].get("name")


def _get_primary_genre(details: dict[str, Any]) -> str | None:
    genres = details.get("genres")
    if not genres or not isinstance(genres, list):
        return None
    ranked = sorted(
        [item for item in genres if item.get("name")],
        key=lambda item: (int(item.get("id") or 0), str(item.get("name")).lower()),
    )
    if not ranked:
        return None
    return ranked[0].get("name")


def _backoff_duration(attempt: int, policy: RetryPolicy) -> float:
    exponential = policy.base_backoff_seconds * (2**attempt)
    bounded = min(exponential, policy.max_backoff_seconds)
    jitter = random.uniform(0.0, policy.jitter_seconds)
    return bounded + jitter


def _http_get_json(
    endpoint: str,
    params: dict[str, Any],
    *,
    timeout_seconds: float,
    api_key: str,
    limiter: RateLimiter,
    retry_policy: RetryPolicy,
) -> dict[str, Any]:
    url_params = {**params, "api_key": api_key}
    url = f"{TMDB_API_URL}{endpoint}?{urlencode(url_params)}"

    for attempt in range(retry_policy.max_retries + 1):
        limiter.wait()
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except HTTPError as exc:
            should_retry = exc.code == 429 or 500 <= exc.code < 600
            if not should_retry or attempt >= retry_policy.max_retries:
                logger.error(
                    "TMDB request terminal failure: endpoint=%s, status=%s, attempt=%s, params=%s",
                    endpoint,
                    exc.code,
                    attempt,
                    params,
                )
                raise TMDBRequestError(
                    f"TMDB request terminal failure: endpoint={endpoint}, status={exc.code}, attempt={attempt}, params={params}"
                ) from exc

            retry_after_header = exc.headers.get("Retry-After") if exc.headers else None
            if retry_after_header and retry_after_header.isdigit():
                sleep_seconds = float(retry_after_header)
            else:
                sleep_seconds = _backoff_duration(attempt, retry_policy)
            time.sleep(sleep_seconds)
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt >= retry_policy.max_retries:
                logger.error(
                    "TMDB request terminal failure: endpoint=%s, status=%s, attempt=%s, params=%s",
                    endpoint,
                    type(exc).__name__,
                    attempt,
                    params,
                )
                raise TMDBRequestError(
                    f"TMDB request terminal failure: endpoint={endpoint}, error={type(exc).__name__}, attempt={attempt}, params={params}"
                ) from exc
            time.sleep(_backoff_duration(attempt, retry_policy))

    raise SupplementInternalError("Unreachable retry loop in _http_get_json")


class TMDBClient:
    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        limiter: RateLimiter,
        retry_policy: RetryPolicy,
    ) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._limiter = limiter
        self._retry_policy = retry_policy

    def search_movie(self, title: str, year: int | None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query": title,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        }
        if year is not None:
            params["primary_release_year"] = year
        payload = _http_get_json(
            "/search/movie",
            params,
            timeout_seconds=self._timeout_seconds,
            api_key=self._api_key,
            limiter=self._limiter,
            retry_policy=self._retry_policy,
        )
        return payload.get("results", [])

    def movie_details(self, tmdb_id: int) -> dict[str, Any]:
        return _http_get_json(
            f"/movie/{tmdb_id}",
            {"language": "en-US"},
            timeout_seconds=self._timeout_seconds,
            api_key=self._api_key,
            limiter=self._limiter,
            retry_policy=self._retry_policy,
        )

    def movie_credits(self, tmdb_id: int) -> dict[str, Any]:
        return _http_get_json(
            f"/movie/{tmdb_id}/credits",
            {"language": "en-US"},
            timeout_seconds=self._timeout_seconds,
            api_key=self._api_key,
            limiter=self._limiter,
            retry_policy=self._retry_policy,
        )


def _resolve_best_match(
    tmdb: TMDBClient,
    title: str,
    year: int | None,
) -> tuple[int | None, float]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    year_candidates: list[int | None]
    if year is None:
        year_candidates = [None]
    else:
        year_candidates = [year, year - 1, year + 1, None]

    for candidate_year in year_candidates:
        results = tmdb.search_movie(title=title, year=candidate_year)
        for item in results:
            item_id = int(item.get("id") or 0)
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                candidates.append(item)
        if len(candidates) >= 20:
            break

    if not candidates:
        return None, 0.0

    scored: list[tuple[float, float, int, float, int]] = []
    for item in candidates:
        confidence, title_score, year_diff, popularity = _build_confidence_score(
            query_title=title,
            query_year=year,
            candidate=item,
        )
        scored.append((confidence, title_score, year_diff, popularity, int(item["id"])))

    scored.sort(key=lambda value: (-value[0], -value[1], value[2], -value[3], value[4]))
    best = scored[0]
    confidence, title_score, year_diff, _, tmdb_id = best
    return tmdb_id, confidence


def _create_empty_record(movie_id: Any, title: str, year: int | None) -> dict[str, Any]:
    return {
        "movie_id": movie_id,
        "tmdb_id": pd.NA,
        "title": title,
        "year": year,
        "genre": pd.NA,
        "description": pd.NA,
        "director": pd.NA,
        "actor": pd.NA,
    }


def _supplement_one_record(
    row: pd.Series,
    tmdb: TMDBClient,
) -> dict[str, Any]:
    movie_id = row["movie_id"]
    title = str(row["title"])
    year_raw = row["year"]
    year = None if pd.isna(year_raw) else int(year_raw)

    tmdb_id, confidence = _resolve_best_match(tmdb, title=title, year=year)
    if tmdb_id is None:
        logger.debug(
            "TMDB no-matches for: "
            f"movie_id={movie_id} title={title} year={year} confidence={round(confidence, 4)}"
        )
        return _create_empty_record(movie_id, title, year)

    details = tmdb.movie_details(tmdb_id)
    credits = tmdb.movie_credits(tmdb_id)

    return {
        "movie_id": movie_id,
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "genre": _get_primary_genre(details),
        "description": details.get("overview") or pd.NA,
        "director": _get_primary_director(credits.get("crew", [])) or pd.NA,
        "actor": _get_primary_actor(credits.get("cast", [])) or pd.NA,
    }


def _apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["tmdb_id"] = output["tmdb_id"].astype("Int64")
    output["genre"] = output["genre"].astype("string")
    output["description"] = output["description"].astype("string")
    output["director"] = output["director"].astype("string")
    output["actor"] = output["actor"].astype("string")
    return output


def supplement(
    source_file: Path | str | None = None,
    target_file: Path | str | None = None,
    *,
    force: bool = False,
    resume: bool = True,
    batch_size: int = DEFAULT_TMDB_BATCH_SIZE,
    requests_per_second: float = DEFAULT_TMDB_REQUESTS_PER_SECOND,
    timeout_seconds: float = DEFAULT_TMDB_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_TMDB_MAX_RETRIES,
) -> pd.DataFrame:
    """Enrich movies metadata using TMDB and save to savepoints/movies_metadata.parquet."""

    if batch_size <= 0:
        raise SupplementConfigurationError("batch_size must be > 0")
    if requests_per_second > TMDB_MAX_REQUESTS_PER_SECOND:
        raise SupplementConfigurationError(
            f"requests_per_second must be <= {TMDB_MAX_REQUESTS_PER_SECOND}"
        )

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise TMDBAuthenticationError(
            "TMDB API key is missing. Set TMDB_API_KEY before running supplement()."
        )

    source_file = resolve_path(source_file, DEFAULT_SUPPLEMENT_SOURCE_FILE)
    target_file = resolve_path(target_file, DEFAULT_SUPPLEMENT_TARGET_FILE)
    ensure_dir(target_file.parent)

    if not source_file.exists():
        raise DataNotFoundError(f"Source file not found: {source_file}")

    progress_file = target_file.parent / (
        f"{target_file.stem}{DEFAULT_SUPPLEMENT_PROGRESS_SUFFIX}{target_file.suffix}"
    )
    if target_file.exists() and not force:
        return pd.read_parquet(target_file)

    source_df = pd.read_parquet(source_file)
    required_columns = set(REQUIRED_SUPPLEMENT_SOURCE_COLUMNS)
    missing_columns = required_columns.difference(source_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SupplementSchemaError(
            f"movies.parquet is missing required columns: {missing}"
        )

    base_df = source_df[list(REQUIRED_SUPPLEMENT_SOURCE_COLUMNS)].drop_duplicates(
        "movie_id"
    )
    supplement_df = pd.DataFrame(columns=list(REQUIRED_SUPPLEMENT_TARGET_COLUMNS))

    if progress_file.exists() and resume and not force:
        supplement_df = pd.read_parquet(progress_file)
        supplement_df = supplement_df.drop_duplicates("movie_id", keep="last")

    processed_ids = (
        set(supplement_df["movie_id"].tolist()) if not supplement_df.empty else set()
    )
    pending_df = base_df[~base_df["movie_id"].isin(processed_ids)].reset_index(
        drop=True
    )

    limiter = RateLimiter(requests_per_second=requests_per_second)
    retry_policy = RetryPolicy(max_retries=max_retries)
    tmdb = TMDBClient(
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        limiter=limiter,
        retry_policy=retry_policy,
    )

    _emit_status(
        "TMDB supplement start: "
        f"total={len(base_df)}, pending={len(pending_df)}, resumed={len(processed_ids)}"
    )

    if not pending_df.empty:
        total_pending = len(pending_df)
        completed = 0
        progress_interval = max(1, batch_size)
        batch_records: list[dict[str, Any]] = []
        for _, row in pending_df.iterrows():
            try:
                batch_records.append(_supplement_one_record(row, tmdb))
            except Exception:
                logger.exception(
                    "TMDB supplement terminal failure: movie_id=%s, title=%s, year=%s",
                    row.get("movie_id"),
                    row.get("title"),
                    row.get("year"),
                )
                row_title = str(row.get("title") or "")
                row_year_raw = row.get("year")
                row_year = None if pd.isna(row_year_raw) else int(row_year_raw)
                batch_records.append(
                    _create_empty_record(row["movie_id"], row_title, row_year)
                )

            completed += 1
            if completed % progress_interval == 0 or completed == total_pending:
                progress_pct = (completed / total_pending) * 100.0
                _emit_status(
                    "TMDB supplement progress: "
                    f"completed={completed}/{total_pending} ({progress_pct:.1f}%)"
                )

            if len(batch_records) >= batch_size:
                batch_df = pd.DataFrame(batch_records)
                supplement_df = pd.concat([supplement_df, batch_df], ignore_index=True)
                supplement_df = supplement_df.drop_duplicates("movie_id", keep="last")
                _apply_schema(supplement_df).to_parquet(
                    progress_file,
                    engine="pyarrow",
                    compression="snappy",
                    index=False,
                )
                _emit_status(
                    "TMDB supplement checkpoint saved: "
                    f"processed={completed}/{total_pending} path={progress_file}"
                )
                batch_records = []

        if batch_records:
            batch_df = pd.DataFrame(batch_records)
            supplement_df = pd.concat([supplement_df, batch_df], ignore_index=True)
            supplement_df = supplement_df.drop_duplicates("movie_id", keep="last")
            _apply_schema(supplement_df).to_parquet(
                progress_file,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )
            _emit_status(
                "TMDB supplement checkpoint saved: "
                f"processed={completed}/{total_pending} path={progress_file}"
            )

    enrichment_columns = [
        column
        for column in REQUIRED_SUPPLEMENT_TARGET_COLUMNS
        if column not in {"movie_id", "title", "year"}
    ]
    target_df = base_df.merge(
        supplement_df[["movie_id", *enrichment_columns]],
        on="movie_id",
        how="left",
    )
    target_df = target_df[list(REQUIRED_SUPPLEMENT_TARGET_COLUMNS)]
    target_df = _apply_schema(target_df)
    target_df.to_parquet(
        target_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    if progress_file.exists():
        progress_file.unlink()

    _emit_status(f"TMDB supplement end: count={len(target_df)}, path={target_file}")
    return target_df
