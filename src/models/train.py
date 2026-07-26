"""Entraîne PatchCore sur une catégorie MVTec AD à partir d'une config d'expérience."""
from __future__ import annotations

import argparse
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from omegaconf import DictConfig

from src.config import load_experiment_config
from src.eval.log_results import log_experiment
from src.eval.metrics import extract_metrics

DEFAULT_RESULTS_CSV = Path("results/experiments.csv")


def build_datamodule(cfg: DictConfig) -> MVTecAD:
    """Construit le datamodule MVTec AD à partir de la config fusionnée."""
    return MVTecAD(
        root=cfg.root,
        category=cfg.category,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        num_workers=cfg.num_workers,
    )


def build_model(cfg: DictConfig) -> Patchcore:
    """Construit le modèle PatchCore à partir de la config fusionnée."""
    return Patchcore(
        backbone=cfg.backbone,
        layers=list(cfg.layers),
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
    )


def run_experiment(experiment_path: Path, results_csv: Path = DEFAULT_RESULTS_CSV) -> dict[str, float]:
    """Charge la config, entraîne PatchCore, teste, et log les résultats en CSV."""
    cfg = load_experiment_config(experiment_path)
    datamodule = build_datamodule(cfg)
    model = build_model(cfg)

    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)

    metrics = extract_metrics(test_results)
    log_experiment(cfg, metrics, results_csv)
    return metrics


def main() -> None:
    """CLI : entraîne et évalue PatchCore pour une config d'expérience donnée."""
    parser = argparse.ArgumentParser(description="Entraîne PatchCore sur MVTec AD.")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--results-csv", default=DEFAULT_RESULTS_CSV, type=Path)
    args = parser.parse_args()

    metrics = run_experiment(args.experiment, args.results_csv)
    print(f"Résultats : {metrics}")


if __name__ == "__main__":
    main()
