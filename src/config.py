"""Chargement et fusion des configs YAML (data + model + overrides d'expérience)."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_experiment_config(experiment_path: Path) -> DictConfig:
    """Fusionne data.yaml + model.yaml + les overrides du fichier d'expérience."""
    experiment_cfg = OmegaConf.load(experiment_path)

    data_cfg_path = Path(experiment_cfg.data_config)
    model_cfg_path = Path(experiment_cfg.model_config)

    base_cfg = OmegaConf.merge(
        OmegaConf.load(data_cfg_path),
        OmegaConf.load(model_cfg_path),
    )
    overrides = experiment_cfg.get("overrides", OmegaConf.create({}))
    return OmegaConf.merge(base_cfg, overrides)
