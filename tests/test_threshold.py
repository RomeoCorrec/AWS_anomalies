from pathlib import Path

import numpy as np

from src.eval.threshold import roc_points, threshold_candidates


def test_threshold_candidates_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])

    rows = threshold_candidates(scores, labels, thresholds=np.array([0.5]))

    assert rows == [{"threshold": 0.5, "precision": 1.0, "recall": 1.0, "f1": 1.0}]


def test_threshold_candidates_uses_quantile_grid_by_default() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])

    rows = threshold_candidates(scores, labels)

    assert len(rows) == 21
    assert all({"threshold", "precision", "recall", "f1"} == set(row.keys()) for row in rows)


def test_roc_points_returns_fpr_tpr_thresholds() -> None:
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])

    fpr, tpr, thresholds = roc_points(scores, labels)

    assert fpr[0] == 0.0
    assert tpr[-1] == 1.0
    assert len(fpr) == len(tpr) == len(thresholds)
