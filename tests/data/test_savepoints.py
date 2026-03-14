from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aoratos.data.errors import UnsafeNameError
from aoratos.data.savepoints import safe_save_name, save


def test_safe_save_name_rejects_unsafe_values() -> None:
    for bad in ["", " bad", "a/b", "a\\b", "..", "x..y"]:
        with pytest.raises(UnsafeNameError):
            safe_save_name(bad)

    assert safe_save_name("checkpoint") == "checkpoint"


def test_save_overwrites_file(tmp_path: Path) -> None:
    root = tmp_path / "savepoints"
    save(pd.DataFrame({"v": [1]}), "checkpoint", target_dir=root)
    save(pd.DataFrame({"v": [2]}), "checkpoint", target_dir=root)

    loaded = pd.read_parquet(root / "checkpoint.parquet")
    assert loaded["v"].tolist() == [2]
