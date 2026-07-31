"""Serveur HTTP minimal exposant /ping et /invocations pour un endpoint SageMaker Serverless Inference."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError

from src.config import load_experiment_config
from src.models.detector import AnomalyDetector

DEFAULT_EXPERIMENT_PATH = Path("config/experiment/bottle_wideresnet50.yaml")
DEFAULT_CHECKPOINT_PATH = Path("/opt/ml/model/model.ckpt")
DEFAULT_THRESHOLD_PATH = Path("config/threshold.yaml")


def load_threshold(experiment_path: Path, threshold_path: Path = DEFAULT_THRESHOLD_PATH) -> float:
    """Lit le seuil de décision de la catégorie de l'expérience depuis threshold.yaml."""
    cfg = load_experiment_config(experiment_path)
    thresholds = OmegaConf.load(threshold_path)
    return float(thresholds[cfg.category])


def build_detector(
    experiment_path: Path = DEFAULT_EXPERIMENT_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
) -> AnomalyDetector:
    """Construit l'AnomalyDetector utilisé par le serveur, seuil inclus."""
    threshold = load_threshold(experiment_path, threshold_path)
    return AnomalyDetector(experiment_path, checkpoint_path, threshold)


def create_app(detector: AnomalyDetector) -> Flask:
    """Construit l'app Flask exposant /ping et /invocations pour un AnomalyDetector donné."""
    app = Flask(__name__)

    @app.get("/ping")
    def ping():
        return "", 200

    @app.post("/invocations")
    def invocations():
        try:
            image = Image.open(io.BytesIO(request.get_data())).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "invalid image"}), 400

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            result = detector.predict(Path(tmp.name))

        return jsonify(result), 200

    return app


def main() -> None:
    """Point d'entrée exécuté par le container au lancement du serveur d'inférence."""
    detector = build_detector()
    app = create_app(detector)
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
