from src.aws import lambda_authorizer


def test_handler_authorizes_matching_key(monkeypatch) -> None:
    monkeypatch.setattr(lambda_authorizer, "API_KEY", "secret-value")

    result = lambda_authorizer.handler({"headers": {"x-api-key": "secret-value"}}, context=None)

    assert result == {"isAuthorized": True}


def test_handler_rejects_wrong_key(monkeypatch) -> None:
    monkeypatch.setattr(lambda_authorizer, "API_KEY", "secret-value")

    result = lambda_authorizer.handler({"headers": {"x-api-key": "wrong"}}, context=None)

    assert result == {"isAuthorized": False}


def test_handler_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.setattr(lambda_authorizer, "API_KEY", "secret-value")

    result = lambda_authorizer.handler({"headers": {}}, context=None)

    assert result == {"isAuthorized": False}


def test_handler_rejects_when_no_key_configured(monkeypatch) -> None:
    # An empty configured API_KEY must never match an empty provided key.
    monkeypatch.setattr(lambda_authorizer, "API_KEY", "")

    result = lambda_authorizer.handler({"headers": {"x-api-key": ""}}, context=None)

    assert result == {"isAuthorized": False}
