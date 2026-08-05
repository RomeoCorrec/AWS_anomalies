from pathlib import Path

from botocore.exceptions import ClientError

from src.aws import deploy_api


def test_zip_source_produces_a_valid_zip_containing_the_file(tmp_path) -> None:
    import zipfile

    source = tmp_path / "handler.py"
    source.write_text("def handler(event, context):\n    return {}\n")

    zip_bytes = deploy_api.zip_source(source)

    with zipfile.ZipFile(__import__("io").BytesIO(zip_bytes)) as zf:
        assert zf.namelist() == ["handler.py"]
        assert b"def handler" in zf.read("handler.py")


class _FakeLambdaClient:
    class exceptions:
        class ResourceNotFoundException(Exception):
            pass

    def __init__(self, existing: bool) -> None:
        self._existing = existing
        self.create_calls: list[dict] = []
        self.update_calls: list[dict] = []

    def get_function(self, FunctionName: str) -> dict:
        if not self._existing:
            raise self.exceptions.ResourceNotFoundException()
        return {"Configuration": {"FunctionArn": f"arn:aws:lambda:eu-west-1:155466261331:function:{FunctionName}"}}

    def create_function(self, **kwargs) -> dict:
        self.create_calls.append(kwargs)
        return {"FunctionArn": f"arn:aws:lambda:eu-west-1:155466261331:function:{kwargs['FunctionName']}"}

    def update_function_code(self, **kwargs) -> dict:
        self.update_calls.append(kwargs)
        return {}


def test_deploy_lambda_function_creates_when_absent() -> None:
    client = _FakeLambdaClient(existing=False)

    arn = deploy_api.deploy_lambda_function(
        client,
        function_name="aws-anomalies-predict",
        handler="src.aws.lambda_predict.handler",
        role_arn="arn:aws:iam::155466261331:role/aws-anomalies-lambda-predict-execution",
        zip_bytes=b"fake-zip-bytes",
        environment={"SAGEMAKER_ENDPOINT_NAME": "aws-anomalies-bottle"},
        timeout=30,
    )

    assert arn == "arn:aws:lambda:eu-west-1:155466261331:function:aws-anomalies-predict"
    assert len(client.create_calls) == 1
    call = client.create_calls[0]
    assert call["FunctionName"] == "aws-anomalies-predict"
    assert call["Handler"] == "src.aws.lambda_predict.handler"
    assert call["Runtime"] == deploy_api.LAMBDA_RUNTIME
    assert call["Role"] == "arn:aws:iam::155466261331:role/aws-anomalies-lambda-predict-execution"
    assert call["Code"] == {"ZipFile": b"fake-zip-bytes"}
    assert call["Timeout"] == 30
    assert call["Environment"] == {"Variables": {"SAGEMAKER_ENDPOINT_NAME": "aws-anomalies-bottle"}}
    assert not client.update_calls


def test_deploy_lambda_function_updates_when_present() -> None:
    client = _FakeLambdaClient(existing=True)

    arn = deploy_api.deploy_lambda_function(
        client,
        function_name="aws-anomalies-predict",
        handler="src.aws.lambda_predict.handler",
        role_arn="arn:aws:iam::155466261331:role/aws-anomalies-lambda-predict-execution",
        zip_bytes=b"fake-zip-bytes",
        environment=None,
        timeout=30,
    )

    assert arn == "arn:aws:lambda:eu-west-1:155466261331:function:aws-anomalies-predict"
    assert not client.create_calls
    assert len(client.update_calls) == 1
    assert client.update_calls[0]["FunctionName"] == "aws-anomalies-predict"
    assert client.update_calls[0]["ZipFile"] == b"fake-zip-bytes"
