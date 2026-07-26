import csv
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.eval.log_results import log_experiment
from src.eval.metrics import extract_metrics


def test_extract_metrics_picks_known_keys() -> None:
    test_results = [{"image_AUROC": 0.987, "pixel_AUROC": 0.912, "unused_key": 1}]

    metrics = extract_metrics(test_results)

    assert metrics == {"image_AUROC": 0.987, "pixel_AUROC": 0.912}


def test_extract_metrics_raises_on_empty_results() -> None:
    with pytest.raises(ValueError):
        extract_metrics([])


def test_log_experiment_appends_row_with_header(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {"category": "bottle", "coreset_sampling_ratio": 0.1, "layers": ["layer2", "layer3"]}
    )
    metrics = {"image_AUROC": 0.98}
    csv_path = tmp_path / "experiments.csv"

    log_experiment(cfg, metrics, csv_path)
    log_experiment(cfg, metrics, csv_path)

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert "category" in lines[0]
    assert "image_AUROC" in lines[0]
    assert "layer2,layer3" in lines[1]


def test_log_experiment_reconciles_differing_schemas(tmp_path: Path) -> None:
    csv_path = tmp_path / "experiments.csv"

    cfg_a = OmegaConf.create({"category": "bottle", "backbone": "wide_resnet50_2"})
    log_experiment(cfg_a, {"image_AUROC": 0.9}, csv_path)

    cfg_b = OmegaConf.create({"category": "screw", "backbone": "dinov2_vits14", "extra_param": 1})
    log_experiment(cfg_b, {"image_AUROC": 0.95}, csv_path)

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    assert len(rows) == 2
    assert "extra_param" in rows[0]
    assert rows[0]["extra_param"] == ""
    assert rows[1]["extra_param"] == "1"
    assert rows[0]["category"] == "bottle"
    assert rows[1]["category"] == "screw"
