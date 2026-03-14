from __future__ import annotations

from pathlib import Path

DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_COMPRESSED_ROOT = Path("data/compressed")
DEFAULT_TRAIN_ROOT = Path("data/train")
DEFAULT_TEST_ROOT = Path("data/test")
DEFAULT_SAVEPOINT_ROOT = Path("data/savepoints")

REQUIRED_RAW_FILENAMES = {
    "combined_data_1.txt",
    "combined_data_2.txt",
    "combined_data_3.txt",
    "combined_data_4.txt",
    "probe.txt",
    "qualifying.txt",
    "movie_titles.csv",
}
