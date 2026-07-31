import json
from pathlib import Path

from omegaconf import OmegaConf

from src.aws import train_entrypoint


class _FakeCheckpointCallback:
    best_model_path = ""


class _FakeTrainer:
    checkpoint_callback = _FakeCheckpointCallback()


class _FakeEngine:
    def __init__(self) -> None:
        self.trainer = _FakeTrainer()

    def fit(self, datamodule, model) -> None:
        pass


def test_load_experiment_path_reads_hyperparameters_json(tmp_path: Path) -> None:
    hp_path = tmp_path / "hyperparameters.json"
    hp_path.write_text(json.dumps({"experiment": "config/experiment/bottle_wideresnet50.yaml"}))

    result = train_entrypoint.load_experiment_path(hp_path)

    assert result == Path("config/experiment/bottle_wideresnet50.yaml")


def test_run_training_overrides_root_and_copies_checkpoint(tmp_path, monkeypatch) -> None:
    fake_checkpoint = tmp_path / "source" / "model.ckpt"
    fake_checkpoint.parent.mkdir(parents=True)
    fake_checkpoint.write_bytes(b"fake-weights")
    _FakeCheckpointCallback.best_model_path = str(fake_checkpoint)

    captured_cfg = {}

    def _fake_build_datamodule(cfg):
        captured_cfg["root"] = cfg.root
        return object()

    monkeypatch.setattr(
        train_entrypoint, "load_experiment_config", lambda path: OmegaConf.create({"root": "data/mvtec"})
    )
    monkeypatch.setattr(train_entrypoint, "build_datamodule", _fake_build_datamodule)
    monkeypatch.setattr(train_entrypoint, "build_model", lambda cfg: object())
    monkeypatch.setattr(train_entrypoint, "Engine", _FakeEngine)

    model_dir = tmp_path / "model_dir"
    destination = train_entrypoint.run_training(
        experiment_path=Path("config/experiment/bottle_wideresnet50.yaml"),
        data_root=tmp_path / "sagemaker_data",
        model_dir=model_dir,
    )

    assert captured_cfg["root"] == str(tmp_path / "sagemaker_data")
    assert destination == model_dir / "model.ckpt"
    assert destination.read_bytes() == b"fake-weights"


def test_main_defaults_to_byoc_paths_when_env_vars_absent(monkeypatch) -> None:
    monkeypatch.delenv("SM_CHANNEL_TRAINING", raising=False)
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    monkeypatch.setattr(
        train_entrypoint, "load_experiment_path", lambda: Path("config/experiment/bottle_wideresnet50.yaml")
    )

    captured_args = {}

    def _fake_run_training(experiment_path, data_root, model_dir):
        captured_args["experiment_path"] = experiment_path
        captured_args["data_root"] = data_root
        captured_args["model_dir"] = model_dir
        return model_dir / "model.ckpt"

    monkeypatch.setattr(train_entrypoint, "run_training", _fake_run_training)

    train_entrypoint.main()

    assert captured_args["data_root"] == Path("/opt/ml/input/data/training")
    assert captured_args["model_dir"] == Path("/opt/ml/model")
