from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_COMPRESSED_DIR = Path("data/compressed")
DEFAULT_TRAIN_DIR = Path("data/train")
DEFAULT_TEST_DIR = Path("data/test")
DEFAULT_SAVEPOINTS_DIR = Path("data/savepoints")

REQUIRED_RAW_FILENAMES = {
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
    "probe.txt",
    "qualifying.txt",
    "movie_titles.csv",
}

MB = 1024 * 1024

DEFAULT_SUPPLEMENT_TABLE = "movies_metadata"
DEFAULT_SUPPLEMENT_PROGRESS_SUFFIX = "_wip"

DEFAULT_SUPPLEMENT_SOURCE_FILE = DEFAULT_COMPRESSED_DIR / "movies.parquet"
DEFAULT_SUPPLEMENT_TARGET_FILE = (
    DEFAULT_SAVEPOINTS_DIR / f"{DEFAULT_SUPPLEMENT_TABLE}.parquet"
)

REQUIRED_SUPPLEMENT_SOURCE_COLUMNS = ("movie_id", "title", "year")
REQUIRED_SUPPLEMENT_TARGET_COLUMNS = (
    "movie_id",
    "tmdb_id",
    "title",
    "year",
    "genre",
    "description",
    "director",
    "actor",
)

TMDB_API_URL = "https://api.themoviedb.org/3"
TMDB_MAX_REQUESTS_PER_SECOND = 40.0
DEFAULT_TMDB_REQUESTS_PER_SECOND = 20.0
DEFAULT_TMDB_TIMEOUT_SECONDS = 20.0
DEFAULT_TMDB_BATCH_SIZE = 100
DEFAULT_TMDB_MAX_RETRIES = 6

DEFAULT_TMDB_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_TMDB_BACKOFF_MAX_SECONDS = 20.0
DEFAULT_TMDB_BACKOFF_JITTER_SECONDS = 0.25
