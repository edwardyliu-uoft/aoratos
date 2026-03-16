from __future__ import annotations

import numpy as np
import pytest

from aoratos.models import cosine_similarity_matrix, scale_features


def test_scale_features_standardizes_columns() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    X_scaled = scale_features(X)

    assert X_scaled.shape == X.shape
    assert X_scaled[:, 0].mean() == pytest.approx(0.0, abs=1e-12)
    assert X_scaled[:, 1].mean() == pytest.approx(0.0, abs=1e-12)
    assert X_scaled[:, 0].std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert X_scaled[:, 1].std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_scale_features_can_return_fitted_scaler() -> None:
    X = np.array([[1.0], [2.0], [3.0]])

    X_scaled, scaler = scale_features(X, return_scaler=True)

    assert X_scaled.shape == X.shape
    transformed = scaler.transform(np.array([[2.0]]))
    assert transformed.shape == (1, 1)


def test_cosine_similarity_matrix_pairwise_scores() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    similarity = cosine_similarity_matrix(X)

    assert similarity.shape == (3, 3)
    assert np.diag(similarity) == pytest.approx(np.ones(3))
    assert similarity[0, 1] == pytest.approx(0.0)
    assert similarity[0, 2] == pytest.approx(1.0 / np.sqrt(2.0))
