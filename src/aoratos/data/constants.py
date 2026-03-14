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
