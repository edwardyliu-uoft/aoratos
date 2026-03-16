from __future__ import annotations

import numpy as np
import pytest

from aoratos.models import accuracy, evaluate_regression, mae, precision, recall, rmse


def test_metrics_functions() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.0])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((0.0 + 0.25 + 1.0) / 3.0))
    assert mae(y_true, y_pred) == pytest.approx((0.0 + 0.5 + 1.0) / 3.0)
    assert 0.0 <= precision(y_true, y_pred) <= 1.0
    assert 0.0 <= recall(y_true, y_pred) <= 1.0
    assert 0.0 <= accuracy(y_true, y_pred) <= 1.0


def test_evaluate_regression_with_n_samples() -> None:
    y_true = np.array([3.0, 4.0])
    y_pred = np.array([2.0, 5.0])

    metrics = evaluate_regression(y_true, y_pred, include_n_samples=True)
    assert set(metrics) == {
        "rmse",
        "mae",
        "precision",
        "recall",
        "accuracy",
        "n_samples",
    }
    assert metrics["n_samples"] == 2.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_regression_without_n_samples() -> None:
    metrics = evaluate_regression(np.array([1.0]), np.array([1.0]))
    assert "n_samples" not in metrics
