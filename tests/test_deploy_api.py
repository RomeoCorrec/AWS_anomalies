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


class _FakeApiGatewayClient:
    def __init__(self) -> None:
        self.create_api_calls: list[dict] = []
        self.create_integration_calls: list[dict] = []
        self.create_authorizer_calls: list[dict] = []
        self.create_route_calls: list[dict] = []
        self.create_stage_calls: list[dict] = []

    def create_api(self, **kwargs) -> dict:
        self.create_api_calls.append(kwargs)
        return {"ApiId": "abc123", "ApiEndpoint": "https://abc123.execute-api.eu-west-1.amazonaws.com"}

    def create_integration(self, **kwargs) -> dict:
        self.create_integration_calls.append(kwargs)
        return {"IntegrationId": "integ1"}

    def create_authorizer(self, **kwargs) -> dict:
        self.create_authorizer_calls.append(kwargs)
        return {"AuthorizerId": "auth1"}

    def create_route(self, **kwargs) -> dict:
        self.create_route_calls.append(kwargs)
        return {"RouteId": "route1"}

    def create_stage(self, **kwargs) -> dict:
        self.create_stage_calls.append(kwargs)
        return {}


class _FakeLambdaClientForApi(_FakeLambdaClient):
    def __init__(self) -> None:
        super().__init__(existing=False)
        self.permission_calls: list[dict] = []

    def add_permission(self, **kwargs) -> dict:
        self.permission_calls.append(kwargs)
        return {}


def test_deploy_api_wires_lambdas_authorizer_route_and_permissions(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        deploy_api, "PREDICT_SOURCE_PATH", tmp_path / "lambda_predict.py"
    )
    monkeypatch.setattr(
        deploy_api, "AUTHORIZER_SOURCE_PATH", tmp_path / "lambda_authorizer.py"
    )
    (tmp_path / "lambda_predict.py").write_text("def handler(event, context):\n    return {}\n")
    (tmp_path / "lambda_authorizer.py").write_text("def handler(event, context):\n    return {}\n")

    lambda_client = _FakeLambdaClientForApi()
    apigw_client = _FakeApiGatewayClient()

    invoke_url = deploy_api.deploy_api(
        lambda_client,
        apigw_client,
        predict_role_arn="arn:aws:iam::155466261331:role/aws-anomalies-lambda-predict-execution",
        authorizer_role_arn="arn:aws:iam::155466261331:role/aws-anomalies-lambda-authorizer-execution",
        api_key="secret-value",
        region="eu-west-1",
        account_id="155466261331",
    )

    assert invoke_url == "https://abc123.execute-api.eu-west-1.amazonaws.com/predict"

    # Two functions deployed
    assert {c["FunctionName"] for c in lambda_client.create_calls} == {
        deploy_api.PREDICT_FUNCTION_NAME,
        deploy_api.AUTHORIZER_FUNCTION_NAME,
    }
    authorizer_call = next(
        c for c in lambda_client.create_calls if c["FunctionName"] == deploy_api.AUTHORIZER_FUNCTION_NAME
    )
    assert authorizer_call["Environment"] == {"Variables": {"API_KEY": "secret-value"}}

    # HTTP API created
    assert apigw_client.create_api_calls[0]["ProtocolType"] == "HTTP"

    # Integration targets the predict Lambda
    integration_kwargs = apigw_client.create_integration_calls[0]
    assert integration_kwargs["ApiId"] == "abc123"
    assert integration_kwargs["IntegrationType"] == "AWS_PROXY"
    assert "aws-anomalies-predict" in integration_kwargs["IntegrationUri"]

    # Authorizer targets the authorizer Lambda
    authorizer_kwargs = apigw_client.create_authorizer_calls[0]
    assert authorizer_kwargs["ApiId"] == "abc123"
    assert authorizer_kwargs["AuthorizerType"] == "REQUEST"
    assert authorizer_kwargs["IdentitySource"] == ["$request.header.x-api-key"]
    assert authorizer_kwargs["EnableSimpleResponses"] is True

    # Route wires integration + authorizer together
    route_kwargs = apigw_client.create_route_calls[0]
    assert route_kwargs["ApiId"] == "abc123"
    assert route_kwargs["RouteKey"] == "POST /predict"
    assert route_kwargs["Target"] == "integrations/integ1"
    assert route_kwargs["AuthorizerId"] == "auth1"

    # Stage deployed
    assert apigw_client.create_stage_calls[0]["ApiId"] == "abc123"
    assert apigw_client.create_stage_calls[0]["StageName"] == "$default"
    assert apigw_client.create_stage_calls[0]["AutoDeploy"] is True

    # Both functions grant API Gateway invoke permission
    assert len(lambda_client.permission_calls) == 2
    predict_permission = next(
        c for c in lambda_client.permission_calls if c["FunctionName"] == deploy_api.PREDICT_FUNCTION_NAME
    )
    assert predict_permission["SourceArn"] == "arn:aws:execute-api:eu-west-1:155466261331:abc123/*/*/predict"
    authorizer_permission = next(
        c for c in lambda_client.permission_calls if c["FunctionName"] == deploy_api.AUTHORIZER_FUNCTION_NAME
    )
    assert authorizer_permission["SourceArn"] == "arn:aws:execute-api:eu-west-1:155466261331:abc123/authorizers/auth1"
