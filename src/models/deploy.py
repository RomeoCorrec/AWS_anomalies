"""Entraîne PatchCore et copie le checkpoint résultant vers un chemin de déploiement stable."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from anomalib.engine import Engine

from src.config import load_experiment_config
from src.models.train import build_datamodule, build_model

DEFAULT_DEPLOYED_ROOT = Path("results/deployed")


def deploy_checkpoint(experiment_path: Path, deployed_root: Path = DEFAULT_DEPLOYED_ROOT) -> Path:
    """Entraîne PatchCore et copie le meilleur checkpoint vers deployed_root/<catégorie>/model.ckpt."""
    cfg = load_experiment_config(experiment_path)
    datamodule = build_datamodule(cfg)
    model = build_model(cfg)

    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)

    checkpoint_path = Path(engine.trainer.checkpoint_callback.best_model_path)
    destination = deployed_root / cfg.category / "model.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination)
    return destination


def main() -> None:
    """CLI : entraîne et déploie le checkpoint pour une expérience donnée."""
    parser = argparse.ArgumentParser(description="Entraîne PatchCore et copie le checkpoint vers un chemin stable.")
    parser.add_argument("--experiment", required=True, type=Path)
    args = parser.parse_args()

    destination = deploy_checkpoint(args.experiment)
    print(f"Checkpoint déployé : {destination}")


if __name__ == "__main__":
    main()
