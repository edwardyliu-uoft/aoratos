from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aoratos.data as data
from aoratos.models import BaseModel
from aoratos.models.baseline import BaselineCFModel
from aoratos.models.constants import DEFAULT_MODEL_DIR
from aoratos.models.errors import ModelNotFittedError, SchemaValidationError


def _tiny_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10, "rating": 5.0},
            {"customer_id": 1, "movie_id": 20, "rating": 4.0},
            {"customer_id": 2, "movie_id": 10, "rating": 3.0},
            {"customer_id": 2, "movie_id": 20, "rating": 2.0},
            {"customer_id": 3, "movie_id": 10, "rating": 4.0},
            {"customer_id": 3, "movie_id": 20, "rating": 3.0},
        ]
    )


def test_baseline_inherits_base_model() -> None:
    model = BaselineCFModel()
    assert isinstance(model, BaseModel)


def test_fit_predict_score_is_deterministic_and_clipped() -> None:
    train_df = _tiny_train_df()

    model_a = BaselineCFModel(reg_user=0.1, reg_movie=0.1, n_iters=20)
    model_b = BaselineCFModel(reg_user=0.1, reg_movie=0.1, n_iters=20)

    model_a.fit(train_df.drop(columns=["rating"]), train_df["rating"])
    model_b.fit(train_df.drop(columns=["rating"]), train_df["rating"])

    predict_X = pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10},
            {"customer_id": 2, "movie_id": 20},
            {"customer_id": 999, "movie_id": 10},
            {"customer_id": 1, "movie_id": 999},
        ]
    )

    pred_a = model_a.predict(predict_X)
    pred_b = model_b.predict(predict_X)

    assert pred_a.shape == (4,)
    assert np.allclose(pred_a, pred_b)
    assert np.all(pred_a >= model_a.rating_min)
    assert np.all(pred_a <= model_a.rating_max)

    train_X = train_df.drop(columns=["rating"])
    train_y = train_df["rating"]
    train_pred = model_a.predict(train_X)
    metrics = model_a.score(y_true=train_y, y_pred=train_pred)
    assert set(metrics) == {
        "rmse",
        "mae",
        "precision",
        "recall",
        "accuracy",
        "n_samples",
    }
    assert metrics["n_samples"] == 6.0
    assert metrics["rmse"] < 0.35
    assert metrics["mae"] < 0.35


def test_baseline_score_uses_universal_metrics() -> None:
    frame = pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10, "rating": 5.0},
            {"customer_id": 1, "movie_id": 11, "rating": 4.0},
            {"customer_id": 2, "movie_id": 10, "rating": 3.0},
            {"customer_id": 2, "movie_id": 11, "rating": 2.0},
        ]
    )

    model = BaselineCFModel(n_iters=10)
    model.fit(frame)

    frame_X = frame.drop(columns=["rating"])
    frame_y = frame["rating"]
    frame_pred = model.predict(frame_X)
    metrics = model.score(y_true=frame_y, y_pred=frame_pred)
    assert set(metrics) == {
        "rmse",
        "mae",
        "precision",
        "recall",
        "accuracy",
        "n_samples",
    }
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_score_accepts_explicit_y_true_and_y_pred() -> None:
    train_df = _tiny_train_df()
    model = BaselineCFModel()
    model.fit(train_df)

    train_X = train_df.drop(columns=["rating"])
    train_y = train_df["rating"]
    train_pred = model.predict(train_X)
    metrics = model.score(y_true=train_y, y_pred=train_pred)
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_invalid_input_contracts_raise_schema_errors() -> None:
    model = BaselineCFModel()
    good_X = pd.DataFrame({"customer_id": [1, 2], "movie_id": [10, 20]})

    with pytest.raises(SchemaValidationError):
        model.fit(pd.DataFrame({"customer_id": [1]}), pd.Series([4.0]))

    with pytest.raises(SchemaValidationError):
        model.fit(good_X, pd.Series([4.0]))

    with pytest.raises(SchemaValidationError):
        model.fit(good_X, pd.Series([4.0, np.nan]))


def test_predict_before_fit_raises() -> None:
    model = BaselineCFModel()
    with pytest.raises(ModelNotFittedError):
        model.predict(pd.DataFrame({"customer_id": [1], "movie_id": [1]}))


def test_save_before_fit_raises() -> None:
    model = BaselineCFModel()
    with pytest.raises(ModelNotFittedError):
        model.save()


def test_baseline_save_load_roundtrip(tmp_path: Path) -> None:
    train_df = _tiny_train_df()
    model = BaselineCFModel(reg_user=0.1, reg_movie=0.1, n_iters=20)
    model.fit(train_df)

    predict_X = pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10},
            {"customer_id": 999, "movie_id": 10},
            {"customer_id": 1, "movie_id": 999},
        ]
    )
    pred_before = model.predict(predict_X)

    save_path = tmp_path / "models" / "baseline_roundtrip.pkl"
    written_path = model.save(save_path)
    loaded = BaselineCFModel.load(path=written_path)
    pred_after = loaded.predict(predict_X)

    assert written_path == save_path
    assert written_path.exists()
    assert np.allclose(pred_before, pred_after)


def test_baseline_default_save_path_uses_data_models() -> None:
    model = BaselineCFModel()
    resolved = model._resolve_pkl(name="my_baseline")
    assert resolved == DEFAULT_MODEL_DIR / "my_baseline.pkl"


def test_baseline_relative_path_resolves_under_default_model_dir() -> None:
    model = BaselineCFModel()
    resolved = model._resolve_pkl(path="custom_name")
    assert resolved == DEFAULT_MODEL_DIR / "custom_name.pkl"


def test_works_with_existing_data_read_interface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aoratos.data.reader.DEFAULT_DATA_DIR", tmp_path)

    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    train_df = pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10, "rating": 5.0},
            {"customer_id": 1, "movie_id": 20, "rating": 3.0},
            {"customer_id": 2, "movie_id": 10, "rating": 4.0},
            {"customer_id": 2, "movie_id": 20, "rating": 2.0},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10, "rating": 5.0},
            {"customer_id": 2, "movie_id": 20, "rating": 2.0},
        ]
    )

    train_df.to_parquet(train_dir / "train.parquet", index=False)
    test_df.to_parquet(test_dir / "test.parquet", index=False)

    loaded_train = data.read("train", source="train")
    loaded_test = data.read("test", source="test")

    model = BaselineCFModel()
    model.fit(loaded_train)
    test_X = loaded_test.drop(columns=["rating"])
    test_y = loaded_test["rating"]
    test_pred = model.predict(test_X)
    metrics = model.score(y_true=test_y, y_pred=test_pred)

    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
