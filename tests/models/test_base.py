from __future__ import annotations

import numpy as np
import pytest

from aoratos.models import BaseModel


class _TestModel(BaseModel):
    def fit(self, X, y=None, **kwargs) -> "_TestModel":
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return np.asarray(X, dtype=float)


def test_base_model_has_default_name() -> None:
    model = _TestModel()

    assert model.name == "_TestModel"
    assert "_TestModel" in str(model)


def test_base_model_accepts_custom_name() -> None:
    model = _TestModel(name="_TestCostomModel")

    assert model.name == "_TestCostomModel"
    assert "_TestCostomModel" in str(model)


def test_base_score_uses_evaluate_regression() -> None:
    model = _TestModel().fit(X=[1.0, 2.0, 3.0])
    metrics = model.score(X=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0])

    assert set(metrics) == {
        "rmse",
        "mae",
        "precision",
        "recall",
        "accuracy",
        "n_samples",
    }
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["n_samples"] == 3.0
