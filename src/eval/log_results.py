"""Écrit les résultats d'une run (config + métriques) dans un CSV d'expériences."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def log_experiment(cfg: DictConfig, metrics: dict[str, float], csv_path: Path) -> None:
    """Append une ligne (config aplatie + métriques), en réconciliant le header si les colonnes diffèrent entre runs."""
    flat_cfg: dict[str, Any] = _flatten(OmegaConf.to_container(cfg, resolve=True))
    row = {**flat_cfg, **metrics}

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        existing_fieldnames = list(existing_rows[0].keys()) if existing_rows else []
    else:
        existing_rows = []
        existing_fieldnames = []

    fieldnames = existing_fieldnames + [key for key in row if key not in existing_fieldnames]

    if fieldnames != existing_fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow({key: existing_row.get(key, "") for key in fieldnames})
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    else:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
