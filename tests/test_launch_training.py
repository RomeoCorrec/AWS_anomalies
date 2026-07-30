from src.aws import launch_training


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
