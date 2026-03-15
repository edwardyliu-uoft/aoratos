# TMDB Metadata Enrichment Plan

## 1) Phased implementation plan

1. **Discovery and contract lock**
   - Confirm source contract from `movies.parquet`: `movie_id`, `title`, `year`.
   - Confirm output contract: `savepoints/movies_metadata.parquet`.
   - Validate presence of `TMDB_API_KEY` and fail fast when missing.

2. **TMDB retrieval layer**
   - Implement TMDB GET client using bearer-style API key query auth (`api_key`).
   - Implement proactive request pacing below hard cap (`<= 40 req/s`) with deterministic inter-request spacing.
   - Implement retries for transient network failures, HTTP `429`, and HTTP `5xx`.

3. **Deterministic title/year matching**
   - Normalize titles (lowercase, punctuation stripped, whitespace collapsed).
   - Search with year-first strategy (`year`, `year-1`, `year+1`, then no-year fallback).
   - Score and rank candidates with weighted confidence (`title similarity`, `year distance`, `popularity`).
   - Reject low-confidence or ambiguous matches using fixed thresholds.

4. **Enrichment and shaping**
   - For accepted `tmdb_id`, retrieve movie details and credits.
  - Extract target fields: one primary genre, overview, one primary director, one primary actor.
  - Build nullable metadata rows for no-match/failed lookup cases while retaining source `title` and `year`.

5. **Shared module reuse**
  - Define supplement defaults/thresholds/path constants in `aoratos.data.constants`.
  - Define supplement-specific exception classes in `aoratos.data.errors`.
  - Keep `supplement.py` focused on orchestration and retrieval logic only.

6. **Checkpoint/resume and output**
   - Persist progress batches to savepoints progress parquet during runtime.
   - Resume from progress if available.
   - Build final merged dataset and save atomically to `savepoints/movies_metadata.parquet`.
   - Remove progress file after successful completion.

7. **Quality and observability**
   - Log terminal request failures and unmatched titles with structured fields.
   - Emit periodic progress and checkpoint logs during enrichment.
   - Report match-rate diagnostics and null-rate diagnostics.
   - Validate duplicate `movie_id` handling and deterministic tie-breaking.

## 2) Endpoint-to-field mapping

| Endpoint                          | Required params                                                                           | Auth                                      | Pagination                                                                     | Fields used                                                   | Output columns                           |
| --------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------- | ---------------------------------------- |
| `GET /3/search/movie`             | `query`, `page`, optional `primary_release_year`, `include_adult=false`, `language=en-US` | `api_key` query param from `TMDB_API_KEY` | Uses `page`; default flow consumes page 1 only for conservative request budget | `id`, `title`, `original_title`, `release_date`, `popularity` | candidate scoring only; yields `tmdb_id` |
| `GET /3/movie/{movie_id}`         | `movie_id`, `language=en-US`                                                              | same                                      | n/a                                                                            | `genres[].name`, `overview`                                   | `genre`, `description`                   |
| `GET /3/movie/{movie_id}/credits` | `movie_id`, `language=en-US`                                                              | same                                      | n/a                                                                            | `crew` (job=`Director`), `cast` (`order`, `popularity`)       | `director`, `actor`                      |

## 3) Matching strategy (`title/year -> tmdb_id`)

- **Normalization:** lowercase, remove non-alphanumeric chars, collapse spaces.
- **Year tolerance:** search in order `year`, `year-1`, `year+1`, fallback without year.
- **Confidence score:**
  - $score = 0.75 \cdot title\_similarity + 0.20 \cdot year\_score + 0.05 \cdot popularity\_score$
  - `year_score`: exact=1.0, ±1=0.8, ±2=0.5, else 0.0.
- **Accept thresholds:** confidence >= 0.65, title similarity >= 0.75, year delta <= 2.
- **Deterministic tie-break:** confidence desc, title similarity desc, year delta asc, popularity desc, tmdb id asc.
- **Fallback:** if no accepted candidate, emit null metadata with `tmdb_id = null`.

## 4) Target output schema (`movies_metadata.parquet`)

| Column        | Type                   | Null policy                      |
| ------------- | ---------------------- | -------------------------------- |
| `movie_id`    | inherited (int-like)   | non-null (source key)            |
| `title`       | string                 | non-null unless source is null   |
| `year`        | nullable int (`Int64`) | nullable if source missing       |
| `tmdb_id`     | nullable int (`Int64`) | null when no accepted match      |
| `genre`       | nullable string        | null when details unavailable    |
| `description` | nullable string        | null when overview unavailable   |
| `director`    | nullable string        | null when no director in credits |
| `actor`       | nullable string        | null when no cast in credits     |

Shared configuration location:

- `aoratos.data.constants`: TMDB URL, request/backoff defaults, matching thresholds, supplement source/target defaults.
- `aoratos.data.errors`: supplement-specific error classes used by `supplement()`.

## 5) Explicit rate-limit and retry policy

- **Hard cap compliance:** enforce `requests_per_second <= 40` (default set to `20`).
- **Pacing:** deterministic minimum inter-request interval of `1 / requests_per_second`.
- **Retry scope:** network errors, HTTP `429`, HTTP `5xx`.
- **Retry schedule:** exponential backoff with jitter.
  - `max_retries = 6`
  - `base_backoff = 0.5s`
  - `max_backoff = 20s`
  - `jitter = [0, 0.25]s`
- **429 handling:** if `Retry-After` is present and numeric, sleep exactly that duration before retry.
- **Terminal failures:** structured error log entry with endpoint, params, status/error type, attempts.

## 6) Risk table

| Risk                             | Impact                    | Mitigation                                                                         |
| -------------------------------- | ------------------------- | ---------------------------------------------------------------------------------- |
| Ambiguous title matches          | Wrong metadata attached   | Weighted confidence score + deterministic thresholds + year tolerance bounds       |
| TMDB throttling/429 bursts       | Pipeline stalls/failures  | Conservative default RPS, request spacing, `Retry-After` honoring, bounded retries |
| Transient API/network failures   | Partial enrichment        | Retry with backoff + jitter; checkpoint batches for resume                         |
| Resume inconsistency after crash | Duplicate or missing rows | Progress parquet keyed by `movie_id`, dedupe keep-last on resume                   |
| Missing credits/details fields   | Null-heavy output         | Nullable schema with explicit null policy and quality metrics                      |
| API/key misconfiguration         | Immediate runtime failure | Fail-fast validation of `TMDB_API_KEY` at entrypoint                               |

## 7) `data.supplement()` interface contract

- **Inputs**
  - `source_dir` (optional): location of `movies.parquet` (default `data/compressed`)
  - `target_dir` (optional): output/savepoint folder (default `data/savepoints`)
  - controls: `force`, `resume`, `batch_size`, `requests_per_second`, `timeout_seconds`, `max_retries`

- **Behavior**
  - Reads `movies.parquet` and validates required columns.
  - Uses `TMDB_API_KEY` only (environment variable).
  - Enriches each movie with deterministic TMDB match + details + credits.
  - Writes progress periodically and resumes when requested.

- **Outputs**
  - Returns enriched DataFrame.
  - Persists final dataset to `savepoints/movies_metadata.parquet` with schema: `movie_id`, `tmdb_id`, `title`, `year`, `genre`, `description`, `director`, `actor`.
  - Serves as the enriched replacement metadata table for the original `movies.parquet` contract.

- **Failure modes**
  - Missing `TMDB_API_KEY` -> `TMDBAuthenticationError`.
  - Missing source file -> `DataNotFoundError`.
  - Missing required columns -> `SupplementSchemaError`.
  - Invalid runtime config -> `SupplementConfigurationError`.
  - Terminal TMDB request failures -> `TMDBRequestError` (logged and downgraded to null metadata row at record-level processing).

## Ready-to-implement checklist

- [ ] `data.supplement()` entrypoint exported in package API
- [ ] TMDB request client with auth + conservative throttling
- [ ] Retry/backoff + explicit 429 `Retry-After` support
- [ ] Deterministic matching and tie-break rules
- [ ] Single director/single actor extraction rules
- [ ] Batch checkpoint + resume support
- [ ] Final parquet output to `savepoints/movies_metadata.parquet`
- [ ] Targeted tests for key behaviors
