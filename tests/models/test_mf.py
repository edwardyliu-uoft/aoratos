from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aoratos.models import BaseModel
from aoratos.models.constants import DEFAULT_MODEL_DIR
from aoratos.models.errors import ModelNotFittedError, SchemaValidationError
from aoratos.models.mf import MatrixFactorizationModel


def _tiny_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10, "rating": 5.0, "date": "2005-01-01"},
            {"customer_id": 1, "movie_id": 20, "rating": 4.0, "date": "2005-01-05"},
            {"customer_id": 2, "movie_id": 10, "rating": 2.0, "date": "2005-02-01"},
            {"customer_id": 2, "movie_id": 20, "rating": 1.0, "date": "2005-02-03"},
            {"customer_id": 3, "movie_id": 10, "rating": 4.0, "date": "2005-03-01"},
            {"customer_id": 3, "movie_id": 30, "rating": 5.0, "date": "2005-03-10"},
            {"customer_id": 4, "movie_id": 20, "rating": 2.0, "date": "2005-04-01"},
            {"customer_id": 4, "movie_id": 30, "rating": 3.0, "date": "2005-04-05"},
        ]
    )


def test_mf_inherits_base_model() -> None:
    model = MatrixFactorizationModel()
    assert isinstance(model, BaseModel)


def test_mf_fit_predict_deterministic_and_clipped() -> None:
    train_df = _tiny_train_df()

    model_a = MatrixFactorizationModel(
        n_factors=8,
        n_epochs=12,
        lr=0.01,
        reg=0.02,
        reg_bias=0.01,
        use_time_bias=True,
        random_state=7,
    )
    model_b = MatrixFactorizationModel(
        n_factors=8,
        n_epochs=12,
        lr=0.01,
        reg=0.02,
        reg_bias=0.01,
        use_time_bias=True,
        random_state=7,
    )

    model_a.fit(train_df.drop(columns=["rating"]), train_df["rating"])
    model_b.fit(train_df.drop(columns=["rating"]), train_df["rating"])

    predict_X = pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10, "date": "2005-01-20"},
            {"customer_id": 2, "movie_id": 20, "date": "2005-02-10"},
            {"customer_id": 999, "movie_id": 10, "date": "2005-01-20"},
            {"customer_id": 1, "movie_id": 999, "date": "2005-01-20"},
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


def test_mf_cold_start_falls_back_to_global_mean() -> None:
    train_df = _tiny_train_df()
    model = MatrixFactorizationModel(
        n_factors=6,
        n_epochs=10,
        use_time_bias=False,
        random_state=11,
    )
    model.fit(train_df)

    cold_start_X = pd.DataFrame([{"customer_id": 99999, "movie_id": 88888}])
    pred = model.predict(cold_start_X)
    assert pred.shape == (1,)
    assert pred[0] == pytest.approx(model._mu)


def test_mf_train_rmse_trend_improves() -> None:
    train_df = _tiny_train_df()
    model = MatrixFactorizationModel(
        n_factors=10,
        n_epochs=14,
        track_train_rmse=True,
        lr=0.012,
        reg=0.02,
        reg_bias=0.01,
        use_time_bias=True,
        random_state=99,
    )

    model.fit(train_df)

    assert len(model.train_rmse_history) == model.n_epochs
    assert model.train_rmse_history[-1] <= model.train_rmse_history[0]


def test_mf_requires_date_when_time_bias_enabled() -> None:
    train_df = _tiny_train_df().drop(columns=["date"])
    model = MatrixFactorizationModel(use_time_bias=True)

    with pytest.raises(SchemaValidationError):
        model.fit(train_df)


def test_mf_predict_before_fit_raises() -> None:
    model = MatrixFactorizationModel(use_time_bias=False)
    with pytest.raises(ModelNotFittedError):
        model.predict(pd.DataFrame({"customer_id": [1], "movie_id": [1]}))


def test_mf_save_before_fit_raises() -> None:
    model = MatrixFactorizationModel(use_time_bias=False)
    with pytest.raises(ModelNotFittedError):
        model.save()


def test_mf_save_load_roundtrip(tmp_path: Path) -> None:
    train_df = _tiny_train_df()
    model = MatrixFactorizationModel(
        n_factors=8,
        n_epochs=12,
        lr=0.01,
        reg=0.02,
        reg_bias=0.01,
        use_time_bias=True,
        random_state=7,
    )
    model.fit(train_df)

    predict_X = pd.DataFrame(
        [
            {"customer_id": 1, "movie_id": 10, "date": "2005-01-20"},
            {"customer_id": 999, "movie_id": 10, "date": "2005-01-20"},
            {"customer_id": 1, "movie_id": 999, "date": "2005-01-20"},
        ]
    )
    pred_before = model.predict(predict_X)

    save_path = tmp_path / "models" / "mf_roundtrip.pkl"
    written_path = model.save(save_path)
    loaded = MatrixFactorizationModel.load(path=written_path)
    pred_after = loaded.predict(predict_X)

    assert written_path == save_path
    assert written_path.exists()
    assert np.allclose(pred_before, pred_after)


def test_mf_default_save_path_uses_data_models() -> None:
    model = MatrixFactorizationModel()
    resolved = model._resolve_pkl(name="my_mf")
    assert resolved == DEFAULT_MODEL_DIR / "my_mf.pkl"


def test_mf_relative_path_resolves_under_default_model_dir() -> None:
    model = MatrixFactorizationModel()
    resolved = model._resolve_pkl(path="custom_name")
    assert resolved == DEFAULT_MODEL_DIR / "custom_name.pkl"
