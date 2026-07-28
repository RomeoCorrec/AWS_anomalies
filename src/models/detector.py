"""Charge un modèle PatchCore entraîné et prédit un score d'anomalie par image."""
from __future__ import annotations

from pathlib import Path

from anomalib.engine import Engine

from src.config import load_experiment_config
from src.models.train import build_model


class AnomalyDetector:
    """Prédit un score d'anomalie et une décision binaire pour une image, à partir d'un checkpoint déployé."""

    def __init__(self, experiment_path: Path, checkpoint_path: Path, threshold: float) -> None:
        cfg = load_experiment_config(experiment_path)
        self.model = build_model(cfg)
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        self.engine = Engine()

    def predict(self, image_path: Path) -> dict[str, float | bool]:
        """Retourne {"score": float, "is_anomaly": bool} pour une image donnée."""
        predictions = self.engine.predict(
            model=self.model,
            ckpt_path=self.checkpoint_path,
            data_path=image_path,
        )
        score = float(predictions[0].pred_score.item())
        return {"score": score, "is_anomaly": score >= self.threshold}
