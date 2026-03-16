from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def _validate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return y_true, y_pred


def _round(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = _validate(y_true, y_pred)

    y_true_rint = np.rint(y_true).astype(int)
    y_pred_rint = np.rint(y_pred).astype(int)

    label_min = int(y_true_rint.min())
    label_max = int(y_true_rint.max())
    y_pred_rint = np.clip(y_pred_rint, label_min, label_max)

    return y_true_rint, y_pred_rint


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""

    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean absolute error."""

    y_true, y_pred = _validate(y_true, y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return macro precision on rounded labels."""

    y_true_rint, y_pred_rint = _round(y_true, y_pred)
    return float(
        precision_score(
            y_true_rint,
            y_pred_rint,
            average="macro",
            zero_division=0,
        )
    )


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return macro recall on rounded labels."""

    y_true_rint, y_pred_rint = _round(y_true, y_pred)
    return float(
        recall_score(
            y_true_rint,
            y_pred_rint,
            average="macro",
            zero_division=0,
        )
    )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return exact-match accuracy on rounded labels."""

    y_true_rint, y_pred_rint = _round(y_true, y_pred)
    return float(accuracy_score(y_true_rint, y_pred_rint))


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    include_n_samples: bool = False,
) -> dict[str, float]:
    """Evaluate predictions with a fixed universal metric suite.

    Returned metrics:

    - `rmse`
    - `mae`
    - `precision` (macro-averaged on rounded labels)
    - `recall` (macro-averaged on rounded labels)
    - `accuracy` (rounded-label exact match)
    - `n_samples` (optional)
    """

    y_true, y_pred = _validate(y_true, y_pred)

    results: dict[str, float] = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }

    if include_n_samples:
        results["n_samples"] = float(y_true.shape[0])

    return results
