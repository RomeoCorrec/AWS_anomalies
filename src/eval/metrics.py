"""Extraction des métriques utiles depuis les résultats de Engine.test()."""
from __future__ import annotations

METRIC_KEYS = ("image_AUROC", "image_F1Score", "pixel_AUROC")


def extract_metrics(test_results: list[dict[str, float]]) -> dict[str, float]:
    """Aplatit les résultats de Engine.test() en un dict des métriques suivies."""
    if not test_results:
        raise ValueError("test_results est vide, aucune métrique à extraire.")
    raw = test_results[0]
    return {key: float(raw[key]) for key in METRIC_KEYS if key in raw}
