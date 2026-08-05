import json

from botocore.exceptions import ClientError

from src.aws import lambda_predict


class _FakeBody:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeRuntimeClient:
    def __init__(self, response=None, error=None) -> None:
        self._response = response
        self._error = error
        self.invoke_calls: list[dict] = []

    def invoke_endpoint(self, **kwargs):
        self.invoke_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _event(body: bytes, is_base64: bool = True, content_type: str = "image/png") -> dict:
    import base64

    return {
        "body": base64.b64encode(body).decode("ascii") if is_base64 else body.decode("utf-8"),
        "isBase64Encoded": is_base64,
        "headers": {"content-type": content_type},
    }


def test_handler_relays_prediction_for_valid_image(monkeypatch) -> None:
    fake_client = _FakeRuntimeClient(
        response={"Body": _FakeBody(json.dumps({"score": 0.91, "is_anomaly": True}).encode())}
    )
    monkeypatch.setattr(lambda_predict, "_get_runtime_client", lambda: fake_client)

    result = lambda_predict.handler(_event(b"fake-png-bytes"), context=None)

    assert result["statusCode"] == 200
    assert json.loads(result["body"]) == {"score": 0.91, "is_anomaly": True}
    assert fake_client.invoke_calls[0]["EndpointName"] == "aws-anomalies-bottle"
    assert fake_client.invoke_calls[0]["Body"] == b"fake-png-bytes"
    assert fake_client.invoke_calls[0]["ContentType"] == "image/png"


def test_handler_relays_model_error_status_and_body(monkeypatch) -> None:
    error = ClientError(
        error_response={
            "Error": {
                "Code": "ModelError",
                "Message": "boom",
                "OriginalStatusCode": 400,
                "OriginalMessage": json.dumps({"error": "invalid image"}),
            }
        },
        operation_name="InvokeEndpoint",
    )
    fake_client = _FakeRuntimeClient(error=error)
    monkeypatch.setattr(lambda_predict, "_get_runtime_client", lambda: fake_client)

    result = lambda_predict.handler(_event(b"not-an-image"), context=None)

    assert result["statusCode"] == 400
    assert json.loads(result["body"]) == {"error": "invalid image"}


def test_handler_returns_503_when_endpoint_missing(monkeypatch) -> None:
    error = ClientError(
        error_response={"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}},
        operation_name="InvokeEndpoint",
    )
    fake_client = _FakeRuntimeClient(error=error)
    monkeypatch.setattr(lambda_predict, "_get_runtime_client", lambda: fake_client)

    result = lambda_predict.handler(_event(b"irrelevant"), context=None)

    assert result["statusCode"] == 503
    assert json.loads(result["body"]) == {"error": "endpoint unavailable"}
