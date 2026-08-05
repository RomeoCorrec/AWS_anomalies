# Lambda + API Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the SageMaker Serverless Inference endpoint (`aws-anomalies-bottle`) behind a public HTTP API — API Gateway (HTTP API) → Lambda authorizer (API key) → Lambda `predict` → `sagemaker-runtime.invoke_endpoint` — so a client can classify an image without holding AWS credentials.

**Architecture:** Two Lambda functions (`predict`, `authorizer`), both plain Python 3.12 zip packages with no external dependencies (boto3 is preinstalled in the Lambda runtime). A new `src/aws/deploy_api.py` script (raw `boto3` calls — there is no high-level SDK for Lambda/API Gateway the way `sagemaker` provides one for training/serving) provisions both functions, an API Gateway HTTP API, a `POST /predict` route, the Lambda authorizer, and the resource-based permissions letting API Gateway invoke both functions. Data flow: client → API Gateway → authorizer Lambda (validates `x-api-key`) → `predict` Lambda → SageMaker endpoint → JSON response relayed back unchanged.

**Tech Stack:** Python 3.10+ (Lambda runtime itself is 3.12), `boto3` (new direct dependency — previously only pulled in transitively via `sagemaker`), AWS Lambda, API Gateway HTTP API, IAM.

**Prior state:** Sub-project 3 (`docs/superpowers/specs/2026-07-31-aws-serverless-endpoint-design.md`, `docs/superpowers/plans/2026-07-31-aws-serverless-endpoint.md`) is complete — `src/aws/deploy_endpoint.py` deploys `aws-anomalies-bottle` on demand (destroyed between test sessions; see `docs/aws-architecture.md`). AWS account ID `155466261331`, region `eu-west-1`. IAM role `aws-anomalies-sagemaker-execution` and IAM user `aws-anomalies-local` already exist.

## Global Constraints

- Region `eu-west-1` only.
- Least privilege: every IAM change is additive and scoped to specific ARNs (no `*` on resources except where AWS mandates it).
- Lambda timeout ≥30s on the `predict` function (Serverless endpoint cold start — project guardrail).
- No hardcoded secrets in source — the authorizer's expected API key is read from an environment variable, never a literal in code.
- Binary passthrough only — the request body reaching `predict` is the raw image bytes, never base64/JSON-wrapped in application code (API Gateway's own base64 encoding of binary bodies over HTTP is transport-level and handled by decoding `event["isBase64Encoded"]`, not a design choice).
- No test hits the network or AWS — all `boto3` clients are passed as parameters and replaced with fakes in tests, same dependency-injection style already used for `Model`/`ServerlessInferenceConfig` monkeypatching in `tests/test_deploy_endpoint.py`.
- Category `bottle` only — the endpoint name `aws-anomalies-bottle` is the only one wired through.
- **Destroy the endpoint after verification** (same guardrail as the prior sub-project). Lambda and API Gateway do not bill at rest and may be left deployed or torn down at the user's discretion after Task 6.

---

### Task 1: Lambda `predict` handler

**Files:**
- Create: `src/aws/lambda_predict.py`
- Test: `tests/test_lambda_predict.py`

**Interfaces:**
- Consumes: nothing from other tasks at import time; reads `SAGEMAKER_ENDPOINT_NAME` and `AWS_REGION` environment variables at call time (both defaulted).
- Produces: `handler(event: dict, context) -> dict` — the Lambda entry point, referenced by name (`lambda_predict.handler`) as the `Handler` value when Task 4/5's `deploy_lambda_function` creates the function. Response shape `{"statusCode": int, "headers": dict, "body": str}` (API Gateway HTTP API Lambda proxy integration v2.0 format).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lambda_predict.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lambda_predict.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aws.lambda_predict'`.

- [ ] **Step 3: Write `src/aws/lambda_predict.py`**

```python
"""Lambda "predict" : relaie une image binaire vers l'endpoint SageMaker Serverless."""
from __future__ import annotations

import base64
import json
import os

import boto3
from botocore.exceptions import ClientError

ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "aws-anomalies-bottle")
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")

_runtime_client = None


def _get_runtime_client():
    """Retourne un client sagemaker-runtime, créé une seule fois par exécution froide."""
    global _runtime_client
    if _runtime_client is None:
        _runtime_client = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
    return _runtime_client


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event: dict, context) -> dict:
    """Point d'entrée Lambda : relaie le corps de la requête API Gateway vers l'endpoint SageMaker."""
    raw_body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(raw_body)
    else:
        body = raw_body.encode("utf-8")

    content_type = event.get("headers", {}).get("content-type", "application/x-image")

    try:
        response = _get_runtime_client().invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType=content_type, Body=body
        )
    except ClientError as exc:
        error = exc.response.get("Error", {})
        if error.get("Code") == "ModelError":
            status_code = error.get("OriginalStatusCode", 500)
            try:
                payload = json.loads(error.get("OriginalMessage", "{}"))
            except (TypeError, ValueError):
                payload = {"error": error.get("OriginalMessage", "model error")}
            return _response(status_code, payload)
        return _response(503, {"error": "endpoint unavailable"})

    payload = json.loads(response["Body"].read())
    return _response(200, payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lambda_predict.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/aws/lambda_predict.py tests/test_lambda_predict.py
git commit -m "feat: add predict Lambda handler relaying to SageMaker endpoint"
```

---

### Task 2: Lambda `authorizer` handler

**Files:**
- Create: `src/aws/lambda_authorizer.py`
- Test: `tests/test_lambda_authorizer.py`

**Interfaces:**
- Consumes: nothing from other tasks; reads the `API_KEY` environment variable at call time.
- Produces: `handler(event: dict, context) -> dict` — HTTP API Lambda authorizer, simple response format (`{"isAuthorized": bool}`), referenced as `lambda_authorizer.handler` by Task 5's `deploy_api`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lambda_authorizer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lambda_authorizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aws.lambda_authorizer'`.

- [ ] **Step 3: Write `src/aws/lambda_authorizer.py`**

```python
"""Lambda authorizer HTTP API : valide le header x-api-key contre le secret configuré."""
from __future__ import annotations

import os

API_KEY = os.environ.get("API_KEY", "")


def handler(event: dict, context) -> dict:
    """Point d'entrée Lambda authorizer (format simple response, payload v2.0)."""
    provided_key = event.get("headers", {}).get("x-api-key", "")
    is_authorized = bool(API_KEY) and provided_key == API_KEY
    return {"isAuthorized": is_authorized}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lambda_authorizer.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/aws/lambda_authorizer.py tests/test_lambda_authorizer.py
git commit -m "feat: add API-key Lambda authorizer for the HTTP API"
```

---

### Task 3: IAM — Lambda execution roles and local-user policy extension

**This task has no repo files — it is AWS console/CLI provisioning, done by the user with the assistant providing exact JSON and verification commands, same pattern as the prior sub-project's Task 3.**

**Files:** none.

**Interfaces:**
- Consumes: nothing from other tasks at the code level.
- Produces: two role ARNs consumed as CLI parameters by Task 6 (`--predict-role-arn`, `--authorizer-role-arn`); permissions consumed by `aws-anomalies-local` running Task 4/5's `deploy_api.py` and Task 6's manual deployment.

- [ ] **Step 1: Create the `aws-anomalies-lambda-predict-execution` role**

Console: IAM → Roles → Create role → Trusted entity: AWS service → Use case:
Lambda. Name: `aws-anomalies-lambda-predict-execution`. Attach this inline
policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeSageMakerEndpoint",
      "Effect": "Allow",
      "Action": "sagemaker:InvokeEndpoint",
      "Resource": "arn:aws:sagemaker:eu-west-1:155466261331:endpoint/aws-anomalies-bottle"
    },
    {
      "Sid": "WriteLambdaLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:eu-west-1:155466261331:log-group:/aws/lambda/aws-anomalies-predict:*"
    }
  ]
}
```

- [ ] **Step 2: Create the `aws-anomalies-lambda-authorizer-execution` role**

Same trust policy (Lambda service). Name:
`aws-anomalies-lambda-authorizer-execution`. Attach this inline policy — no
AWS service access needed beyond its own logs, since the authorizer only
compares strings in memory:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "WriteLambdaLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:eu-west-1:155466261331:log-group:/aws/lambda/aws-anomalies-authorizer:*"
    }
  ]
}
```

- [ ] **Step 3: Extend the `aws-anomalies-local` policy**

Add the following statements to the existing customer-managed policy
attached to `aws-anomalies-local`:

```json
{
  "Sid": "ManageLambdaFunctions",
  "Effect": "Allow",
  "Action": [
    "lambda:CreateFunction",
    "lambda:UpdateFunctionCode",
    "lambda:GetFunction",
    "lambda:AddPermission",
    "lambda:DeleteFunction"
  ],
  "Resource": [
    "arn:aws:lambda:eu-west-1:155466261331:function:aws-anomalies-predict",
    "arn:aws:lambda:eu-west-1:155466261331:function:aws-anomalies-authorizer"
  ]
},
{
  "Sid": "PassLambdaExecutionRoles",
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": [
    "arn:aws:iam::155466261331:role/aws-anomalies-lambda-predict-execution",
    "arn:aws:iam::155466261331:role/aws-anomalies-lambda-authorizer-execution"
  ]
},
{
  "Sid": "ManageHttpApi",
  "Effect": "Allow",
  "Action": [
    "apigateway:GET",
    "apigateway:POST",
    "apigateway:DELETE",
    "apigateway:PATCH"
  ],
  "Resource": "arn:aws:apigateway:eu-west-1::/apis/*"
}
```

**Note:** API Gateway's resource-level ARNs cannot be scoped to a specific,
not-yet-created API ID up front — `apis/*` is the narrowest practical scope
for creating a new HTTP API via this IAM user, consistent with the
least-privilege intent (no `*` service-wide, but the API Gateway resource
hierarchy itself doesn't support per-name scoping before creation, unlike
the `aws-anomalies-*`-prefixed ARNs used elsewhere in this project).

- [ ] **Step 4: Verify**

Run: `aws iam get-role --role-name aws-anomalies-lambda-predict-execution`
and `aws iam get-role --role-name aws-anomalies-lambda-authorizer-execution`
Expected: both roles exist, trust policy is `lambda.amazonaws.com`.

- [ ] **Step 5: User confirms completion**

The user says when all three policy edits are done before Task 6 proceeds
(Tasks 4/5 are pure code and do not require these permissions to write or
unit-test).

---

### Task 4: Lambda packaging and deployment helpers

**Files:**
- Create: `src/aws/deploy_api.py`
- Test: `tests/test_deploy_api.py`

**Interfaces:**
- Consumes: `src/aws/lambda_predict.py` (Task 1), `src/aws/lambda_authorizer.py` (Task 2) as the source files zipped for deployment.
- Produces: `zip_source(source_path: Path) -> bytes`, `deploy_lambda_function(lambda_client, function_name: str, handler: str, role_arn: str, zip_bytes: bytes, environment: dict[str, str] | None, timeout: int) -> str` (returns the function ARN) — both consumed by Task 5's `deploy_api` orchestrator in this same file.

- [ ] **Step 1: Add the `boto3` dependency**

Run: `uv add boto3`
Expected: `pyproject.toml` and `uv.lock` updated (`boto3` was previously only
a transitive dependency of `sagemaker`; `deploy_api.py` imports it
directly, so it becomes an explicit one, same reasoning as `flask` in the
prior sub-project).

- [ ] **Step 2: Write the failing tests**

Create `tests/test_deploy_api.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_deploy_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aws.deploy_api'`.

- [ ] **Step 4: Write `src/aws/deploy_api.py` (packaging/deployment helpers only)**

```python
"""Déploie les Lambdas predict/authorizer et l'API Gateway HTTP API devant l'endpoint SageMaker."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

LAMBDA_RUNTIME = "python3.12"


def zip_source(source_path: Path) -> bytes:
    """Zippe un unique fichier Python source pour le déploiement Lambda."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_path, arcname=source_path.name)
    return buffer.getvalue()


def deploy_lambda_function(
    lambda_client,
    function_name: str,
    handler: str,
    role_arn: str,
    zip_bytes: bytes,
    environment: dict[str, str] | None = None,
    timeout: int = 30,
) -> str:
    """Crée la fonction Lambda si absente, sinon met à jour son code. Retourne l'ARN de la fonction."""
    try:
        existing = lambda_client.get_function(FunctionName=function_name)
        lambda_client.update_function_code(FunctionName=function_name, ZipFile=zip_bytes)
        return existing["Configuration"]["FunctionArn"]
    except lambda_client.exceptions.ResourceNotFoundException:
        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime=LAMBDA_RUNTIME,
            Role=role_arn,
            Handler=handler,
            Code={"ZipFile": zip_bytes},
            Timeout=timeout,
            Environment={"Variables": environment or {}},
        )
        return response["FunctionArn"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_deploy_api.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/aws/deploy_api.py tests/test_deploy_api.py
git commit -m "feat: add Lambda packaging and deployment helpers"
```

---

### Task 5: API Gateway wiring and CLI

**Files:**
- Modify: `src/aws/deploy_api.py` (append to the file created in Task 4)
- Modify: `tests/test_deploy_api.py` (append to the file created in Task 4)

**Interfaces:**
- Consumes: `deploy_lambda_function` and `zip_source` (Task 4, same file); the ClientError import already used across `src/aws`.
- Produces: `deploy_api(lambda_client, apigw_client, predict_role_arn: str, authorizer_role_arn: str, api_key: str, region: str = "eu-west-1", account_id: str = "155466261331") -> str` (returns the API's invoke URL) and a `main()` CLI entry point — both run manually in Task 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_deploy_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_api.py -v`
Expected: FAIL — `AttributeError: module 'src.aws.deploy_api' has no attribute 'deploy_api'`.

- [ ] **Step 3: Append to `src/aws/deploy_api.py`**

```python
import argparse

from botocore.exceptions import ClientError

PREDICT_FUNCTION_NAME = "aws-anomalies-predict"
AUTHORIZER_FUNCTION_NAME = "aws-anomalies-authorizer"
API_NAME = "aws-anomalies-api"
PREDICT_SOURCE_PATH = Path(__file__).parent / "lambda_predict.py"
AUTHORIZER_SOURCE_PATH = Path(__file__).parent / "lambda_authorizer.py"


def deploy_api(
    lambda_client,
    apigw_client,
    predict_role_arn: str,
    authorizer_role_arn: str,
    api_key: str,
    region: str = "eu-west-1",
    account_id: str = "155466261331",
) -> str:
    """Déploie les 2 Lambdas et l'API Gateway HTTP API, retourne l'URL d'invocation de POST /predict."""
    predict_arn = deploy_lambda_function(
        lambda_client,
        function_name=PREDICT_FUNCTION_NAME,
        handler="lambda_predict.handler",
        role_arn=predict_role_arn,
        zip_bytes=zip_source(PREDICT_SOURCE_PATH),
        environment={"SAGEMAKER_ENDPOINT_NAME": "aws-anomalies-bottle", "AWS_REGION": region},
        timeout=30,
    )
    authorizer_arn = deploy_lambda_function(
        lambda_client,
        function_name=AUTHORIZER_FUNCTION_NAME,
        handler="lambda_authorizer.handler",
        role_arn=authorizer_role_arn,
        zip_bytes=zip_source(AUTHORIZER_SOURCE_PATH),
        environment={"API_KEY": api_key},
        timeout=10,
    )

    api = apigw_client.create_api(Name=API_NAME, ProtocolType="HTTP")
    api_id = api["ApiId"]

    integration = apigw_client.create_integration(
        ApiId=api_id,
        IntegrationType="AWS_PROXY",
        IntegrationUri=predict_arn,
        PayloadFormatVersion="2.0",
        IntegrationMethod="POST",
    )

    authorizer = apigw_client.create_authorizer(
        ApiId=api_id,
        AuthorizerType="REQUEST",
        Name="api-key-authorizer",
        AuthorizerUri=(
            f"arn:aws:apigateway:{region}:lambda:path/2015-03-31/functions/{authorizer_arn}/invocations"
        ),
        AuthorizerPayloadFormatVersion="2.0",
        EnableSimpleResponses=True,
        IdentitySource=["$request.header.x-api-key"],
    )

    apigw_client.create_route(
        ApiId=api_id,
        RouteKey="POST /predict",
        Target=f"integrations/{integration['IntegrationId']}",
        AuthorizationType="CUSTOM",
        AuthorizerId=authorizer["AuthorizerId"],
    )

    apigw_client.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)

    lambda_client.add_permission(
        FunctionName=PREDICT_FUNCTION_NAME,
        StatementId="apigateway-invoke-predict",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*/predict",
    )
    lambda_client.add_permission(
        FunctionName=AUTHORIZER_FUNCTION_NAME,
        StatementId="apigateway-invoke-authorizer",
        Action="lambda:InvokeFunction",
        Principal="apigateway.amazonaws.com",
        SourceArn=f"arn:aws:execute-api:{region}:{account_id}:{api_id}/authorizers/{authorizer['AuthorizerId']}",
    )

    return f"{api['ApiEndpoint']}/predict"


def main() -> None:
    """CLI : déploie les Lambdas et l'API Gateway HTTP API devant l'endpoint SageMaker."""
    import boto3

    parser = argparse.ArgumentParser(description="Déploie Lambda + API Gateway devant l'endpoint SageMaker.")
    parser.add_argument("--predict-role-arn", required=True)
    parser.add_argument("--authorizer-role-arn", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--account-id", default="155466261331")
    args = parser.parse_args()

    lambda_client = boto3.client("lambda", region_name=args.region)
    apigw_client = boto3.client("apigatewayv2", region_name=args.region)

    invoke_url = deploy_api(
        lambda_client,
        apigw_client,
        predict_role_arn=args.predict_role_arn,
        authorizer_role_arn=args.authorizer_role_arn,
        api_key=args.api_key,
        region=args.region,
        account_id=args.account_id,
    )
    print(f"API déployée : {invoke_url}")


if __name__ == "__main__":
    main()
```

Move the `if __name__ == "__main__": main()` block from wherever Task 4
left the bottom of the file (Task 4 did not add one — `main()` did not
exist yet) so it appears exactly once, at the end of the file, after this
step.

Note: `Handler` for each function is the bare module-level name
(`lambda_predict.handler`, not `src.aws.lambda_predict.handler`) because
`zip_source` zips each file at the root of the archive (`arcname=source_path.name`)
— the deployed package has no `src/aws/` directory structure, only the
single file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_deploy_api.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/aws/deploy_api.py tests/test_deploy_api.py
git commit -m "feat: wire API Gateway HTTP API, authorizer, and route to the predict Lambda"
```

---

### Task 6: Manual end-to-end deployment, verification, and teardown

**No repo files — real AWS execution, cross-checking against the already-validated endpoint responses, and cleanup.**

- [ ] **Step 1: Redeploy the SageMaker endpoint**

Run (same command as the prior sub-project's Task 6 / already used
manually since):
```bash
uv run python -m src.aws.deploy_endpoint \
  --image-uri 155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest \
  --role-arn arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution \
  --model-data-url s3://aws-anomalies-mvtec-romeo/output/aws-anomalies-train-2026-07-31-09-04-02-108/output/model.tar.gz \
  --endpoint-name aws-anomalies-bottle
```
Expected: endpoint reaches `InService` (verify with
`aws sagemaker describe-endpoint --endpoint-name aws-anomalies-bottle --region eu-west-1 --query EndpointStatus`).

- [ ] **Step 2: Choose an API key and deploy Lambda + API Gateway**

Pick any random string as the API key (e.g.
`uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`).

Run:
```bash
uv run python -m src.aws.deploy_api \
  --predict-role-arn arn:aws:iam::155466261331:role/aws-anomalies-lambda-predict-execution \
  --authorizer-role-arn arn:aws:iam::155466261331:role/aws-anomalies-lambda-authorizer-execution \
  --api-key "<the chosen key>"
```
Expected: prints `API déployée : https://<api-id>.execute-api.eu-west-1.amazonaws.com/predict`.

- [ ] **Step 3: Invoke the API on a defective and a good test image, with the key**

Run:
```bash
curl -s -o defect_result.json -w "%{http_code}\n" \
  -X POST "<invoke-url>" \
  -H "x-api-key: <the chosen key>" \
  -H "Content-Type: image/png" \
  --data-binary @data/mvtec/bottle/test/broken_large/000.png
cat defect_result.json

curl -s -o good_result.json -w "%{http_code}\n" \
  -X POST "<invoke-url>" \
  -H "x-api-key: <the chosen key>" \
  -H "Content-Type: image/png" \
  --data-binary @data/mvtec/bottle/test/good/000.png
cat good_result.json
```
Expected: both return HTTP 200; scores close to the reference values
`0.9098` (defective, `is_anomaly: true`) and `0.2531` (good,
`is_anomaly: false`) already measured in the prior sub-project and
reconfirmed via direct `invoke-endpoint` calls.

- [ ] **Step 4: Verify the authorizer actually rejects unauthorized calls**

Run:
```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST "<invoke-url>" \
  -H "Content-Type: image/png" --data-binary @data/mvtec/bottle/test/good/000.png

curl -s -o /dev/null -w "%{http_code}\n" -X POST "<invoke-url>" \
  -H "x-api-key: wrong-key" -H "Content-Type: image/png" \
  --data-binary @data/mvtec/bottle/test/good/000.png
```
Expected: both return `403` (no key, wrong key) — confirms the authorizer
is actually enforced, not bypassed.

- [ ] **Step 5: Verify the invalid-image path**

Run:
```bash
curl -s -X POST "<invoke-url>" -H "x-api-key: <the chosen key>" \
  -H "Content-Type: image/png" --data-binary "not an image"
```
Expected: `400` with body `{"error": "invalid image"}`, confirming
`lambda_predict.handler`'s `ModelError` branch relays the endpoint's own
error faithfully end-to-end.

- [ ] **Step 6: Tear down the SageMaker endpoint — mandatory**

Run:
```bash
aws sagemaker delete-endpoint --endpoint-name aws-anomalies-bottle --region eu-west-1
aws sagemaker delete-endpoint-config --endpoint-config-name aws-anomalies-bottle --region eu-west-1
```
Expected: `describe-endpoint` afterward returns `ValidationException: Could
not find endpoint` — same verification pattern as the prior sub-project.
**Not optional — the Serverless endpoint bills per request/idle-memory
until deleted, unlike Lambda/API Gateway below.**

- [ ] **Step 7: Decide on Lambda/API Gateway teardown**

Unlike the SageMaker endpoint, the 2 Lambdas and the HTTP API do not bill
at rest (Lambda: per invocation + duration; API Gateway HTTP API: per
request). Ask the user whether to leave them deployed (so the API can be
re-tested later without redeploying — only the SageMaker endpoint needs
recreating each time) or delete them now:

```bash
aws lambda delete-function --function-name aws-anomalies-predict --region eu-west-1
aws lambda delete-function --function-name aws-anomalies-authorizer --region eu-west-1
aws apigatewayv2 delete-api --api-id <api-id> --region eu-west-1
```

- [ ] **Step 8: Report to the user**

Summarize: deployment results for both endpoint and API, the 4 invocation
outcomes (valid defective, valid good, unauthorized, invalid image) with
their exact status codes and bodies, and explicit confirmation of what was
torn down vs. left running.

---

## Self-Review Notes

- **Spec coverage:** `predict` Lambda relaying to `invoke_endpoint` (Task 1), API-key `authorizer` Lambda (Task 2), IAM execution roles + local-user extension (Task 3), Lambda packaging/deployment helpers (Task 4), API Gateway HTTP API + route + authorizer + permissions wiring and CLI (Task 5), real end-to-end deployment + verification (valid images, unauthorized rejection, invalid-image passthrough) + endpoint teardown (Task 6) — matches every section of `2026-08-05-aws-lambda-api-gateway-design.md`.
- **Placeholder scan:** no TBD/TODO. Account ID (`155466261331`), region (`eu-west-1`), reference scores (`0.9098`/`0.2531`), and the exact `model.tar.gz` S3 path are filled in literally, reused from `docs/aws-architecture.md`. `<the chosen key>` and `<invoke-url>` in Task 6 are real runtime values only known after Step 2 runs — not unresolved design placeholders.
- **Type/interface consistency:** `handler(event: dict, context) -> dict` has the identical signature in both `lambda_predict.py` (Task 1) and `lambda_authorizer.py` (Task 2), matching what AWS Lambda actually calls. `deploy_lambda_function` (Task 4) is called twice in `deploy_api` (Task 5) with consistent parameter names. `zip_source` (Task 4) is consumed directly by `deploy_api` (Task 5) via `PREDICT_SOURCE_PATH`/`AUTHORIZER_SOURCE_PATH`, both defined in Task 5's addition to the same file. The `Handler` string format (`lambda_predict.handler`, no `src.aws.` prefix) is explicitly reconciled with how `zip_source` flattens the archive — called out as a note in Task 5 to prevent a real deployment-time mismatch.
- **Ambiguity check:** clarified that HTTP API (not REST API) is used despite wanting an API key, resolved via a Lambda authorizer instead of native API Gateway API keys (which HTTP API does not support); clarified that `ModelError` (image validation failures from the endpoint) and `ValidationException`/other `ClientError`s (endpoint missing) are handled as two distinct branches in `lambda_predict.handler`, each tested separately; clarified that Lambda/API Gateway teardown in Task 6 is optional (unlike the mandatory SageMaker endpoint teardown), since neither bills at rest.
