"""Charge un modèle PatchCore entraîné et prédit un score d'anomalie par image."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from anomalib.engine import Engine

from src.config import load_experiment_config
from src.models.train import build_model


class AnomalyDetector:
    """Prédit un score d'anomalie et une décision binaire pour une image, à partir d'un checkpoint déployé."""

    def __init__(self, experiment_path: Path, checkpoint_path: Path, threshold: float) -> None:
        cfg = load_experiment_config(experiment_path)
        self.model = build_model(cfg, visualizer=False)
        self.checkpoint_path = checkpoint_path
        self.threshold = threshold
        # anomalib's visualization callback writes result images under default_root_dir;
        # the SageMaker Serverless Inference container's filesystem is read-only outside
        # /tmp (and /opt/ml), so the default "results" relative path fails with a
        # PermissionError at inference time. tempfile.gettempdir() is writable everywhere
        # this runs: locally, in the packaging Docker image, and in SageMaker containers.
        self.engine = Engine(default_root_dir=tempfile.gettempdir())

    def predict(self, image_path: Path) -> dict[str, float | bool]:
        """Retourne {"score": float, "is_anomaly": bool} pour une image donnée."""
        predictions = self.engine.predict(
            model=self.model,
            ckpt_path=self.checkpoint_path,
            data_path=image_path,
        )
        score = float(predictions[0].pred_score.item())
        return {"score": score, "is_anomaly": score >= self.threshold}


def main() -> None:
    """CLI : prédit le score d'anomalie d'une image via un checkpoint déployé."""
    parser = argparse.ArgumentParser(description="Prédit le score d'anomalie d'une image avec PatchCore.")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()

    detector = AnomalyDetector(
        experiment_path=args.experiment,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    result = detector.predict(args.image)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
