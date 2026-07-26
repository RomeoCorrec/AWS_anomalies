"""Écrit les résultats d'une run (config + métriques) dans un CSV d'expériences."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def log_experiment(cfg: DictConfig, metrics: dict[str, float], csv_path: Path) -> None:
    """Append une ligne (config aplatie + métriques) dans csv_path, en créant l'en-tête si besoin."""
    flat_cfg: dict[str, Any] = _flatten(OmegaConf.to_container(cfg, resolve=True))
    row = {**flat_cfg, **metrics}

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplatit un dict imbriqué en clés séparées par des points ; les listes sont jointes par virgule."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{full_key}."))
        elif isinstance(value, list):
            flat[full_key] = ",".join(str(v) for v in value)
        else:
            flat[full_key] = value
    return flat
