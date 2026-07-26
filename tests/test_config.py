from pathlib import Path

from omegaconf import OmegaConf

from src.config import load_experiment_config


def test_load_experiment_config_merges_and_overrides(tmp_path: Path) -> None:
    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text("root: data/mvtec\ncategory: bottle\nimage_size: [256, 256]\n")

    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text("backbone: wide_resnet50_2\ncoreset_sampling_ratio: 0.1\n")

    experiment_cfg = tmp_path / "experiment.yaml"
    experiment_cfg.write_text(
        f"data_config: {data_cfg}\n"
        f"model_config: {model_cfg}\n"
        "overrides:\n"
        "  category: screw\n"
    )

    cfg = load_experiment_config(experiment_cfg)

    assert cfg.category == "screw"
    assert cfg.backbone == "wide_resnet50_2"
    assert cfg.coreset_sampling_ratio == 0.1
    assert OmegaConf.to_container(cfg.image_size) == [256, 256]


def test_repo_bottle_experiment_config_loads() -> None:
    cfg = load_experiment_config(Path("config/experiment/bottle_wideresnet50.yaml"))

    assert cfg.category == "bottle"
    assert cfg.backbone == "wide_resnet50_2"
