import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models import detector


class _FakeBatch:
    def __init__(self, score: float) -> None:
        self.pred_score = torch.tensor([score])


class _FakeEngine:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.predict_calls: list[tuple] = []

    def predict(self, model, ckpt_path, data_path):
        self.predict_calls.append((model, ckpt_path, data_path))
        return [_FakeBatch(0.8)]


def test_predict_returns_score_and_is_anomaly_above_threshold(monkeypatch) -> None:
    monkeypatch.setattr(detector, "load_experiment_config", lambda path: OmegaConf.create({}))
    monkeypatch.setattr(detector, "build_model", lambda cfg: object())
    monkeypatch.setattr(detector, "Engine", _FakeEngine)

    det = detector.AnomalyDetector(
        experiment_path=Path("config/experiment/bottle_wideresnet50.yaml"),
        checkpoint_path=Path("results/deployed/bottle/model.ckpt"),
        threshold=0.523,
    )
    result = det.predict(Path("some/image.png"))

    assert result["score"] == pytest.approx(0.8)
    assert result["is_anomaly"] is True


def test_predict_below_threshold_is_not_anomaly(monkeypatch) -> None:
    class _LowScoreEngine(_FakeEngine):
        def predict(self, model, ckpt_path, data_path):
            return [_FakeBatch(0.1)]

    monkeypatch.setattr(detector, "load_experiment_config", lambda path: OmegaConf.create({}))
    monkeypatch.setattr(detector, "build_model", lambda cfg: object())
    monkeypatch.setattr(detector, "Engine", _LowScoreEngine)

    det = detector.AnomalyDetector(
        experiment_path=Path("config/experiment/bottle_wideresnet50.yaml"),
        checkpoint_path=Path("results/deployed/bottle/model.ckpt"),
        threshold=0.523,
    )
    result = det.predict(Path("some/image.png"))

    assert result["score"] == pytest.approx(0.1)
    assert result["is_anomaly"] is False


def test_engine_uses_writable_default_root_dir(monkeypatch) -> None:
    # Regression test: anomalib's visualization callback writes under default_root_dir.
    # A relative "results" path fails with PermissionError in read-only-filesystem
    # containers (e.g. SageMaker Serverless Inference) — it must be a writable tempdir.
    monkeypatch.setattr(detector, "load_experiment_config", lambda path: OmegaConf.create({}))
    monkeypatch.setattr(detector, "build_model", lambda cfg: object())
    monkeypatch.setattr(detector, "Engine", _FakeEngine)

    det = detector.AnomalyDetector(
        experiment_path=Path("config/experiment/bottle_wideresnet50.yaml"),
        checkpoint_path=Path("results/deployed/bottle/model.ckpt"),
        threshold=0.523,
    )

    assert det.engine.kwargs["default_root_dir"] == tempfile.gettempdir()
