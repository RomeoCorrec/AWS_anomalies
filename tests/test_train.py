from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from omegaconf import OmegaConf

from src.models.train import build_datamodule, build_model


def test_build_datamodule_constructs_without_error() -> None:
    cfg = OmegaConf.create(
        {
            "root": "data/mvtec",
            "category": "bottle",
            "train_batch_size": 16,
            "eval_batch_size": 8,
            "num_workers": 2,
        }
    )

    datamodule = build_datamodule(cfg)

    assert isinstance(datamodule, MVTecAD)


def test_build_model_constructs_without_error() -> None:
    cfg = OmegaConf.create(
        {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
        }
    )

    model = build_model(cfg)

    assert isinstance(model, Patchcore)
