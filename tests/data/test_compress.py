from __future__ import annotations

from pathlib import Path

import pandas as pd

from aoratos.data.compress import compress, write_parts


def test_write_parts_creates_expected_names(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    in_file = raw / "combined_data_1.txt"
    in_file.write_text("1:\n10,5,2005-01-01\n11,4,2005-01-02\n", encoding="utf-8")
    out_dir = tmp_path / "compressed" / "ratings"

    parts = write_parts([in_file], out_dir, rows_per_part=1)
    assert parts == 2
    assert (out_dir / "ratings_part_000.parquet").exists()
    assert (out_dir / "ratings_part_001.parquet").exists()


def test_compress_end_to_end_with_fixture_raw(
    fixture_raw_dir: Path, tmp_path: Path
) -> None:
    compressed = tmp_path / "compressed"
    summary = compress(
        source_dir=fixture_raw_dir,
        target_dir=compressed,
        force=True,
        rows_per_part=2,
    )
    parts = summary["ratings_parts"]
    assert len(parts) >= 2
    assert parts[0].name == "ratings_part_000.parquet"
    assert (compressed / "movies.parquet").exists()
    assert (compressed / "probe.parquet").exists()
    assert (compressed / "qualifying.parquet").exists()

    ratings_df = pd.read_parquet(parts[0])
    assert {"movie_id", "customer_id", "rating", "date"}.issubset(ratings_df.columns)
