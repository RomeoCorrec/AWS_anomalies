"""Point d'entrée du container de training SageMaker : entraîne PatchCore avec les
données montées par SageMaker et écrit le checkpoint dans SM_MODEL_DIR."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from anomalib.engine import Engine

from src.config import load_experiment_config
from src.models.train import build_datamodule, build_model

HYPERPARAMETERS_PATH = Path("/opt/ml/input/config/hyperparameters.json")
DEFAULT_MODEL_DIR = Path("/opt/ml/model")


def load_experiment_path(hyperparameters_path: Path = HYPERPARAMETERS_PATH) -> Path:
    """Lit le chemin de la config d'expérience depuis les hyperparamètres SageMaker."""
    hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    return Path(hyperparameters["experiment"])


def run_training(experiment_path: Path, data_root: Path, model_dir: Path) -> Path:
    """Entraîne PatchCore avec root surchargé par les données SageMaker, copie le checkpoint vers model_dir."""
    cfg = load_experiment_config(experiment_path)
    cfg.root = str(data_root)

    datamodule = build_datamodule(cfg)
    model = build_model(cfg)

    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)

    checkpoint_path = Path(engine.trainer.checkpoint_callback.best_model_path)
    destination = model_dir / "model.ckpt"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination)
    return destination


def main() -> None:
    """Point d'entrée exécuté par le container au lancement du Training Job."""
    experiment_path = load_experiment_path()
    data_root = Path(os.environ["SM_CHANNEL_TRAINING"])
    model_dir = Path(os.environ.get("SM_MODEL_DIR", str(DEFAULT_MODEL_DIR)))

    destination = run_training(experiment_path, data_root, model_dir)
    print(f"Checkpoint écrit dans {destination}")


if __name__ == "__main__":
    main()
