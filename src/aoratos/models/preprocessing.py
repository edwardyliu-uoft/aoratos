from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def scale_features(
    X: np.ndarray,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    return_scaler: bool = False,
) -> np.ndarray | tuple[np.ndarray, StandardScaler]:
    """Scale feature matrix using scikit-learn StandardScaler."""

    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    X = np.asarray(X, dtype=float)
    transformed = scaler.fit_transform(X)

    if return_scaler:
        return transformed, scaler

    return transformed


def cosine_similarity_matrix(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Compute pairwise cosine similarity using scikit-learn."""

    X = np.asarray(X, dtype=float)
    y = None if y is None else np.asarray(y, dtype=float)
    return cosine_similarity(X, y)
