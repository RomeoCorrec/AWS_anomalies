from pathlib import Path

from omegaconf import OmegaConf

from src.models import deploy


class _FakeCheckpointCallback:
    best_model_path = ""


class _FakeTrainer:
    checkpoint_callback = _FakeCheckpointCallback()


class _FakeEngine:
    def __init__(self) -> None:
        self.trainer = _FakeTrainer()

    def fit(self, datamodule, model) -> None:
        pass


def test_deploy_checkpoint_copies_best_checkpoint_to_stable_path(tmp_path, monkeypatch) -> None:
    fake_checkpoint = tmp_path / "source" / "model.ckpt"
    fake_checkpoint.parent.mkdir(parents=True)
    fake_checkpoint.write_bytes(b"fake-weights")
    _FakeCheckpointCallback.best_model_path = str(fake_checkpoint)

    monkeypatch.setattr(deploy, "load_experiment_config", lambda path: OmegaConf.create({"category": "bottle"}))
    monkeypatch.setattr(deploy, "build_datamodule", lambda cfg: object())
    monkeypatch.setattr(deploy, "build_model", lambda cfg: object())
    monkeypatch.setattr(deploy, "Engine", _FakeEngine)

    destination = deploy.deploy_checkpoint(
        Path("config/experiment/bottle_wideresnet50.yaml"), deployed_root=tmp_path / "deployed"
    )

    assert destination == tmp_path / "deployed" / "bottle" / "model.ckpt"
    assert destination.read_bytes() == b"fake-weights"
