"""Calcule des candidats de seuil de décision à partir des scores d'anomalie du set de test.

Ne choisit pas de seuil : fournit les scores/étiquettes et les métriques par seuil candidat
pour que la décision et sa justification métier restent manuelles.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from anomalib.engine import Engine
from sklearn.metrics import precision_recall_fscore_support, roc_curve

from src.config import load_experiment_config
from src.models.train import build_datamodule, build_model


def collect_test_scores(experiment_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Entraîne PatchCore puis retourne (scores, labels binaires) sur le set de test."""
    cfg = load_experiment_config(experiment_path)
    datamodule = build_datamodule(cfg)
    model = build_model(cfg)

    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)
    predictions = engine.predict(datamodule=datamodule, model=model)

    scores = torch.cat([batch.pred_score for batch in predictions]).numpy()
    labels = torch.cat([batch.gt_label for batch in predictions]).numpy().astype(int)
    return scores, labels


def threshold_candidates(
    scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Calcule precision/recall/F1 pour chaque seuil candidat (score >= seuil => anomalie)."""
    if thresholds is None:
        thresholds = np.quantile(scores, np.linspace(0.0, 1.0, 21))

    rows = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return rows


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (fpr, tpr, thresholds) de la courbe ROC."""
    return roc_curve(labels, scores)


def _write_candidates_csv(rows: list[dict[str, float]], csv_path: Path) -> None:
    """Écrit les candidats de seuil dans un CSV (une ligne par seuil)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI : entraîne PatchCore et écrit les candidats de seuil pour une expérience donnée."""
    parser = argparse.ArgumentParser(
        description="Calcule des candidats de seuil de décision à partir des scores de test."
    )
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument(
        "--output-csv",
        default=None,
        type=Path,
        help="Défaut : results/threshold_candidates_<catégorie>.csv",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.experiment)
    scores, labels = collect_test_scores(args.experiment)
    rows = threshold_candidates(scores, labels)

    output_csv = args.output_csv or Path(f"results/threshold_candidates_{cfg.category}.csv")
    _write_candidates_csv(rows, output_csv)
    print(f"Candidats de seuil écrits dans {output_csv}")


if __name__ == "__main__":
    main()
