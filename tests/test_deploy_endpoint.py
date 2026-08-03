from src.aws import deploy_endpoint


class _FakeModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.deploy_kwargs: dict = {}

    def deploy(self, **kwargs) -> None:
        self.deploy_kwargs.update(kwargs)


def test_deploy_endpoint_wires_model_and_serverless_config(monkeypatch) -> None:
    captured_model = {}

    class _CapturingFakeModel(_FakeModel):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            captured_model["instance"] = self

    monkeypatch.setattr(deploy_endpoint, "Model", _CapturingFakeModel)
    monkeypatch.setattr(deploy_endpoint, "ServerlessInferenceConfig", lambda **kwargs: kwargs)

    endpoint_name = deploy_endpoint.deploy_endpoint(
        image_uri="155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest",
        role_arn="arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution",
        model_data_url="s3://aws-anomalies-mvtec-romeo/output/job/model.tar.gz",
        endpoint_name="aws-anomalies-bottle",
        memory_size_in_mb=2048,
        max_concurrency=1,
    )

    assert endpoint_name == "aws-anomalies-bottle"
    model = captured_model["instance"]
    assert model.kwargs["image_uri"] == "155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest"
    assert model.kwargs["model_data"] == "s3://aws-anomalies-mvtec-romeo/output/job/model.tar.gz"
    assert model.kwargs["role"] == "arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution"
    assert model.deploy_kwargs["endpoint_name"] == "aws-anomalies-bottle"
    assert model.deploy_kwargs["serverless_inference_config"] == {
        "memory_size_in_mb": 2048,
        "max_concurrency": 1,
    }
