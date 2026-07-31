from pathlib import Path

from src.aws import launch_training, train_entrypoint


class _FakeEstimator:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.hyperparameters: dict = {}
        self.fit_inputs = None
        self.model_data = "s3://aws-anomalies-mvtec-romeo/output/job/model.tar.gz"

    def set_hyperparameters(self, **kwargs) -> None:
        self.hyperparameters.update(kwargs)

    def fit(self, inputs) -> None:
        self.fit_inputs = inputs


def test_launch_training_wires_estimator_and_returns_model_data(monkeypatch) -> None:
    monkeypatch.setattr(launch_training, "Estimator", _FakeEstimator)

    model_data = launch_training.launch_training(
        image_uri="155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-train:latest",
        role_arn="arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution",
        training_data_uri="s3://aws-anomalies-mvtec-romeo/mvtec/",
        output_path="s3://aws-anomalies-mvtec-romeo/output/",
        experiment_path="config/experiment/bottle_wideresnet50.yaml",
    )

    assert model_data == "s3://aws-anomalies-mvtec-romeo/output/job/model.tar.gz"


def test_channel_name_matches_train_entrypoint_default_data_root(monkeypatch) -> None:
    captured_estimator = {}

    class _CapturingFakeEstimator(_FakeEstimator):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            captured_estimator["instance"] = self

    monkeypatch.setattr(launch_training, "Estimator", _CapturingFakeEstimator)

    launch_training.launch_training(
        image_uri="155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-train:latest",
        role_arn="arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution",
        training_data_uri="s3://aws-anomalies-mvtec-romeo/mvtec/",
        output_path="s3://aws-anomalies-mvtec-romeo/output/",
        experiment_path="config/experiment/bottle_wideresnet50.yaml",
    )

    # the channel name launch_training.py uses in estimator.fit({...}) must match the
    # fixed path train_entrypoint.py falls back to, or a real job silently finds no data
    # at the expected subdirectory and re-downloads the full dataset instead of failing fast.
    used_channel_names = set(captured_estimator["instance"].fit_inputs)
    assert used_channel_names == {"training"}
    assert train_entrypoint.DEFAULT_DATA_ROOT == Path("/opt/ml/input/data") / next(iter(used_channel_names))
