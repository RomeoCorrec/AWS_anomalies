# AWS Phase — SageMaker Serverless Inference Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the already-trained `bottle` PatchCore checkpoint via a SageMaker Serverless Inference endpoint, using a BYOC HTTP server (`/ping`, `/invocations`) that reuses the existing `AnomalyDetector`, and verify the deployed endpoint returns the same predictions as the local checkpoint.

**Architecture:** A new `src/aws/serve.py` (Flask HTTP server), a new `Dockerfile.sagemaker-serve`, a new ECR repo `aws-anomalies-serve`, extended permissions on the existing `aws-anomalies-sagemaker-execution` role and `aws-anomalies-local` user, and a new `src/aws/deploy_endpoint.py` launcher using the `sagemaker` SDK's `Model.deploy()`. Data flow: S3 `model.tar.gz` (already produced by the training sub-project) → SageMaker Model → Serverless EndpointConfig → Endpoint → real HTTP invocation → cross-checked against the local `AnomalyDetector`.

**Tech Stack:** Same as the local project (Python 3.10+, `uv`, `anomalib`, `torch` CPU, `sagemaker<3` SDK already pinned) plus `flask` (new dependency) for the HTTP server, Docker, ECR, IAM.

**Prior state:** Sub-project 2 (`docs/superpowers/specs/2026-07-29-aws-sagemaker-training-design.md`, `docs/superpowers/plans/2026-07-29-aws-sagemaker-training.md`) is complete — a real SageMaker Training Job produced a `model.tar.gz` on `s3://aws-anomalies-mvtec-romeo/output/.../model.tar.gz` for `bottle`. IAM role `aws-anomalies-sagemaker-execution` and IAM user `aws-anomalies-local` already exist with training-scoped permissions. ECR repo `aws-anomalies-train` already exists (training image only). AWS account ID `155466261331`, region `eu-west-1`.

## Global Constraints

- Region `eu-west-1` only.
- Least privilege: every IAM change is additive and scoped to specific ARNs (no `*` on resources except where AWS mandates it, e.g. `ecr:GetAuthorizationToken`).
- Serverless config: `memory_size_in_mb=2048`, `max_concurrency=1` (confirmed with the user).
- No hardcoded values that should be config — the threshold comes from `config/threshold.yaml`, never inlined as a literal in Python source.
- Reuse `AnomalyDetector` (`src/models/detector.py`) as-is — no duplicated prediction logic.
- Reuse `load_experiment_config` (`src/config.py`) — no duplicated config-merging logic.
- No test hits the network, AWS, or loads a real model — `AnomalyDetector` is monkeypatched in all automated tests, same style as `tests/test_launch_training.py` and `tests/test_train_entrypoint.py`.
- Category `bottle` only for this plan.
- **Destroy the endpoint after verification** — a Serverless Inference endpoint bills continuously once created, unlike the one-off Training Job from the prior sub-project. This is the single most important reminder for Task 6.

---

### Task 1: Inference HTTP server

**Files:**
- Create: `src/aws/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `AnomalyDetector` (`src/models/detector.py`), `load_experiment_config` (`src/config.py`).
- Produces: `load_threshold(experiment_path: Path, threshold_path: Path) -> float`, `build_detector(experiment_path: Path, checkpoint_path: Path, threshold_path: Path) -> AnomalyDetector`, `create_app(detector: AnomalyDetector) -> Flask` — `create_app` and `build_detector` are consumed by `main()` in this same task, which `Dockerfile.sagemaker-serve` (Task 2) invokes as its entrypoint. `DEFAULT_CHECKPOINT_PATH` (constant, `Path("/opt/ml/model/model.ckpt")`) must match the filename `train_entrypoint.py`'s `run_training` (prior sub-project) copies the checkpoint to — both hardcode the literal string `"model.ckpt"`.

- [ ] **Step 1: Add the `flask` dependency**

Run: `uv add flask`
Expected: `pyproject.toml` and `uv.lock` updated, `uv sync` succeeds.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_serve.py`:

```python
import io
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from src.aws import serve


class _FakeDetector:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.predict_calls: list[Path] = []

    def predict(self, image_path: Path) -> dict:
        self.predict_calls.append(image_path)
        return self.result


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_threshold_reads_category_from_experiment_config(tmp_path, monkeypatch) -> None:
    threshold_path = tmp_path / "threshold.yaml"
    threshold_path.write_text("bottle: 0.523\ncarpet: 0.510\n")
    monkeypatch.setattr(
        serve, "load_experiment_config", lambda path: OmegaConf.create({"category": "bottle"})
    )

    result = serve.load_threshold(Path("config/experiment/bottle_wideresnet50.yaml"), threshold_path)

    assert result == 0.523


def test_ping_returns_200() -> None:
    app = serve.create_app(_FakeDetector({"score": 0.1, "is_anomaly": False}))
    client = app.test_client()

    response = client.get("/ping")

    assert response.status_code == 200


def test_invocations_returns_prediction_for_valid_image() -> None:
    detector = _FakeDetector({"score": 0.91, "is_anomaly": True})
    app = serve.create_app(detector)
    client = app.test_client()

    response = client.post("/invocations", data=_png_bytes(), content_type="image/png")

    assert response.status_code == 200
    assert response.get_json() == {"score": 0.91, "is_anomaly": True}
    assert len(detector.predict_calls) == 1


def test_invocations_rejects_invalid_image_body() -> None:
    app = serve.create_app(_FakeDetector({"score": 0.0, "is_anomaly": False}))
    client = app.test_client()

    response = client.post("/invocations", data=b"not an image", content_type="image/png")

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_default_checkpoint_path_matches_train_entrypoint_artifact_filename() -> None:
    # train_entrypoint.py (prior sub-project) copies the checkpoint to model_dir / "model.ckpt";
    # SageMaker untars model.tar.gz into /opt/ml/model/ on the serving side, so this constant
    # must reference the same filename or the server fails to find the checkpoint at startup.
    assert serve.DEFAULT_CHECKPOINT_PATH.name == "model.ckpt"
    assert serve.DEFAULT_CHECKPOINT_PATH.parent == Path("/opt/ml/model")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aws.serve'`.

- [ ] **Step 4: Write `src/aws/serve.py`**

```python
"""Serveur HTTP minimal exposant /ping et /invocations pour un endpoint SageMaker Serverless Inference."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from omegaconf import OmegaConf
from PIL import Image, UnidentifiedImageError

from src.config import load_experiment_config
from src.models.detector import AnomalyDetector

DEFAULT_EXPERIMENT_PATH = Path("config/experiment/bottle_wideresnet50.yaml")
DEFAULT_CHECKPOINT_PATH = Path("/opt/ml/model/model.ckpt")
DEFAULT_THRESHOLD_PATH = Path("config/threshold.yaml")


def load_threshold(experiment_path: Path, threshold_path: Path = DEFAULT_THRESHOLD_PATH) -> float:
    """Lit le seuil de décision de la catégorie de l'expérience depuis threshold.yaml."""
    cfg = load_experiment_config(experiment_path)
    thresholds = OmegaConf.load(threshold_path)
    return float(thresholds[cfg.category])


def build_detector(
    experiment_path: Path = DEFAULT_EXPERIMENT_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    threshold_path: Path = DEFAULT_THRESHOLD_PATH,
) -> AnomalyDetector:
    """Construit l'AnomalyDetector utilisé par le serveur, seuil inclus."""
    threshold = load_threshold(experiment_path, threshold_path)
    return AnomalyDetector(experiment_path, checkpoint_path, threshold)


def create_app(detector: AnomalyDetector) -> Flask:
    """Construit l'app Flask exposant /ping et /invocations pour un AnomalyDetector donné."""
    app = Flask(__name__)

    @app.get("/ping")
    def ping():
        return "", 200

    @app.post("/invocations")
    def invocations():
        try:
            image = Image.open(io.BytesIO(request.get_data())).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "invalid image"}), 400

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            result = detector.predict(Path(tmp.name))

        return jsonify(result), 200

    return app


def main() -> None:
    """Point d'entrée exécuté par le container au lancement du serveur d'inférence."""
    detector = build_detector()
    app = create_app(detector)
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_serve.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/aws/serve.py tests/test_serve.py
git commit -m "feat: add SageMaker BYOC inference HTTP server"
```

---

### Task 2: Serving container image

**Files:**
- Create: `Dockerfile.sagemaker-serve`

**Interfaces:**
- Consumes: `src/aws/serve.py` (Task 1), `pyproject.toml`/`uv.lock`.
- Produces: a Docker image buildable and runnable locally; pushed to ECR in Task 4.

- [ ] **Step 1: Write `Dockerfile.sagemaker-serve`**

```dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY config/ config/

EXPOSE 8080
ENTRYPOINT ["uv", "run", "--no-sync", "python", "-m", "src.aws.serve"]
```

**Note:** `--no-sync` avoids re-resolving the `uv` environment (a network
call) at container start — `uv sync --frozen` already ran at build time.
Same reasoning as the existing `Dockerfile` and `Dockerfile.sagemaker-train`.

- [ ] **Step 2: Manual verification — build locally**

Run: `docker build --platform linux/amd64 -f Dockerfile.sagemaker-serve -t aws-anomalies-serve:local .`
Expected: build succeeds (same dependency layer as the existing images, cached if built recently).

- [ ] **Step 3: Manual verification — smoke test the server locally**

First, produce a local checkpoint to serve (reuse the one already deployed
locally in the packaging sub-project, or the one downloaded from the real
training job if `/tmp/sagemaker-model/model.ckpt` from the prior
sub-project's Task 6 is still present):

```bash
ls results/deployed/bottle/model.ckpt || ls /tmp/sagemaker-model/model.ckpt
```

**Important (Git Bash / Windows only):** as with the training container,
Git Bash's MSYS layer rewrites leading-`/`-style `docker run` arguments —
including `-v` mount destinations like `/opt/ml/model` — into bogus Windows
paths before Docker sees them. Prefix the `docker run` command with
`MSYS_NO_PATHCONV=1`, and convert host-side paths through `cygpath -w`, as
done for the training smoke test.

Run (mounts a local checkpoint at the exact path SageMaker would produce
after untarring `model.tar.gz`, maps the container's port 8080 to the host):

```bash
MSYS_NO_PATHCONV=1 docker run --rm -d --name aws-anomalies-serve-smoke \
  -p 8080:8080 \
  -v "$(cygpath -w "$(pwd)/results/deployed/bottle/model.ckpt"):/opt/ml/model/model.ckpt:ro" \
  aws-anomalies-serve:local

sleep 5
curl -f http://localhost:8080/ping
curl -f -X POST --data-binary @data/mvtec/bottle/test/broken_large/000.png \
  -H "Content-Type: image/png" http://localhost:8080/invocations

docker stop aws-anomalies-serve-smoke
```

Expected: `/ping` returns an empty `200` body; `/invocations` returns
`{"score": ..., "is_anomaly": true}` for the defective test image (the
score should be close to the `0.910` observed for this same image and
checkpoint in the prior sub-project's Task 6 verification, modulo the
known non-determinism already documented there).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.sagemaker-serve
git commit -m "feat: add SageMaker BYOC serving container"
```

---

### Task 3: IAM — execution role and local-user policy extensions

**This task has no repo files — it is AWS console/CLI provisioning, done by the user with the assistant providing exact JSON and verification commands.**

**Files:** none.

**Interfaces:**
- Consumes: existing role `aws-anomalies-sagemaker-execution`, existing user `aws-anomalies-local` (both from the prior sub-project).
- Produces: permissions consumed by Task 5 (`deploy_endpoint.py`, run as `aws-anomalies-local`) and by the real endpoint (assuming `aws-anomalies-sagemaker-execution` at runtime) in Task 6.

- [ ] **Step 1: Extend the `aws-anomalies-sagemaker-execution` role's inline policy**

Console: IAM → Roles → `aws-anomalies-sagemaker-execution` → the existing
inline policy (`aws-anomalies-sagemaker-execution-policy`) → Edit. Add the
following two statements (ECR pull on the new serving repo, and CloudWatch
Logs write for the endpoint's own log group — separate from the training
job's log group already granted) to the existing statement array, alongside
the S3/ECR/Logs statements already there from the prior sub-project:

```json
{
  "Sid": "PullServingImage",
  "Effect": "Allow",
  "Action": [
    "ecr:GetDownloadUrlForLayer",
    "ecr:BatchGetImage",
    "ecr:BatchCheckLayerAvailability"
  ],
  "Resource": "arn:aws:ecr:eu-west-1:155466261331:repository/aws-anomalies-serve"
},
{
  "Sid": "WriteEndpointLogs",
  "Effect": "Allow",
  "Action": [
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": "arn:aws:logs:eu-west-1:155466261331:log-group:/aws/sagemaker/Endpoints/*"
}
```

The existing `S3ReadTrainingData`/`S3WriteOutput`/`ecr:GetAuthorizationToken`
statements from the prior sub-project are unaffected — the role also needs
`s3:GetObject` on `arn:aws:s3:::aws-anomalies-mvtec-romeo/output/*` to read
`model.tar.gz`, which the existing `S3WriteOutput` statement's resource
already covers if it was scoped to that prefix; if it was scoped to a
narrower path, add `s3:GetObject` there too.

- [ ] **Step 2: User extends the `aws-anomalies-local` policy**

Add the following statements to the existing customer-managed policy
attached to `aws-anomalies-local`:

```json
{
  "Sid": "PushServingImageTemporary",
  "Effect": "Allow",
  "Action": [
    "ecr:CreateRepository",
    "ecr:PutImage",
    "ecr:InitiateLayerUpload",
    "ecr:UploadLayerPart",
    "ecr:CompleteLayerUpload",
    "ecr:BatchCheckLayerAvailability"
  ],
  "Resource": "arn:aws:ecr:eu-west-1:155466261331:repository/aws-anomalies-serve"
},
{
  "Sid": "ManageServerlessEndpoint",
  "Effect": "Allow",
  "Action": [
    "sagemaker:CreateModel",
    "sagemaker:CreateEndpointConfig",
    "sagemaker:CreateEndpoint",
    "sagemaker:DescribeEndpoint",
    "sagemaker:DescribeEndpointConfig",
    "sagemaker:DescribeModel",
    "sagemaker:DeleteEndpoint",
    "sagemaker:DeleteEndpointConfig",
    "sagemaker:DeleteModel"
  ],
  "Resource": [
    "arn:aws:sagemaker:eu-west-1:155466261331:model/aws-anomalies-*",
    "arn:aws:sagemaker:eu-west-1:155466261331:endpoint-config/aws-anomalies-*",
    "arn:aws:sagemaker:eu-west-1:155466261331:endpoint/aws-anomalies-*"
  ]
},
{
  "Sid": "InvokeEndpoint",
  "Effect": "Allow",
  "Action": "sagemaker:InvokeEndpoint",
  "Resource": "arn:aws:sagemaker:eu-west-1:155466261331:endpoint/aws-anomalies-*"
}
```

Note: `ecr:CreateRepository` here follows the same temporary-grant pattern
used for the S3 bucket creation in sub-project 1 — remove it once the
`aws-anomalies-serve` repo exists (Task 4), same as before.

- [ ] **Step 3: Verify**

Run: `aws iam get-role --role-name aws-anomalies-sagemaker-execution`
Expected: role exists, trust policy unchanged (`sagemaker.amazonaws.com`),
inline policy now includes the two new statements from Step 1.

- [ ] **Step 4: User confirms completion**

The user says when both policy edits are done before Task 4 proceeds.

---

### Task 4: ECR repository and image push

**This task has no repo files — it is AWS console/CLI provisioning plus a Docker push, done by the assistant once Task 3's temporary `ecr:CreateRepository` grant is in place.**

**Files:** none.

**Interfaces:**
- Consumes: `aws-anomalies-serve:local` image built in Task 2.
- Produces: `155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest` — consumed by Task 6 as `--image-uri`.

- [ ] **Step 1: Create the ECR repository**

Run: `aws ecr create-repository --repository-name aws-anomalies-serve --region eu-west-1`
Expected: repository created, ARN
`arn:aws:ecr:eu-west-1:155466261331:repository/aws-anomalies-serve` returned
— matches the ARN already granted in Task 3.

- [ ] **Step 2: Authenticate Docker to ECR**

Run: `aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 155466261331.dkr.ecr.eu-west-1.amazonaws.com`
Expected: `Login Succeeded`.

- [ ] **Step 3: Tag and push the image**

Run:
```bash
docker tag aws-anomalies-serve:local 155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest
docker push 155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest
```
Expected: push succeeds, digest printed.

- [ ] **Step 4: User removes the temporary `ecr:CreateRepository` grant**

Once the repo exists, the user removes the `ecr:CreateRepository` action
from the `PushServingImageTemporary` statement added in Task 3 (keeping the
push/pull actions, which remain needed for future image updates) — same
least-privilege pattern as the training sub-project's ECR setup.

---

### Task 5: Endpoint deployment launcher

**Files:**
- Create: `src/aws/deploy_endpoint.py`
- Test: `tests/test_deploy_endpoint.py`

**Interfaces:**
- Consumes: nothing from prior tasks at the code level (talks to AWS via the `sagemaker` SDK); consumes the image pushed in Task 4 and the role extended in Task 3 as runtime parameters.
- Produces: `deploy_endpoint(image_uri, role_arn, model_data_url, endpoint_name, memory_size_in_mb, max_concurrency) -> str` (deployed endpoint name) — run manually in Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_deploy_endpoint.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deploy_endpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aws.deploy_endpoint'`.

- [ ] **Step 3: Write `src/aws/deploy_endpoint.py`**

```python
"""Déploie le checkpoint entraîné derrière un endpoint SageMaker Serverless Inference (BYOC)."""
from __future__ import annotations

import argparse

from sagemaker.model import Model
from sagemaker.serverless import ServerlessInferenceConfig

DEFAULT_MEMORY_SIZE_MB = 2048
DEFAULT_MAX_CONCURRENCY = 1


def deploy_endpoint(
    image_uri: str,
    role_arn: str,
    model_data_url: str,
    endpoint_name: str,
    memory_size_in_mb: int = DEFAULT_MEMORY_SIZE_MB,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> str:
    """Déploie le modèle derrière un endpoint Serverless, retourne le nom de l'endpoint."""
    model = Model(image_uri=image_uri, model_data=model_data_url, role=role_arn)
    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=memory_size_in_mb, max_concurrency=max_concurrency
    )
    model.deploy(endpoint_name=endpoint_name, serverless_inference_config=serverless_config)
    return endpoint_name


def main() -> None:
    """CLI : déploie l'endpoint Serverless Inference pour un checkpoint PatchCore donné."""
    parser = argparse.ArgumentParser(description="Déploie un endpoint SageMaker Serverless Inference (BYOC).")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--model-data-url", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--memory-size-mb", type=int, default=DEFAULT_MEMORY_SIZE_MB)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    args = parser.parse_args()

    endpoint_name = deploy_endpoint(
        image_uri=args.image_uri,
        role_arn=args.role_arn,
        model_data_url=args.model_data_url,
        endpoint_name=args.endpoint_name,
        memory_size_in_mb=args.memory_size_mb,
        max_concurrency=args.max_concurrency,
    )
    print(f"Endpoint déployé : {endpoint_name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_deploy_endpoint.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/aws/deploy_endpoint.py tests/test_deploy_endpoint.py
git commit -m "feat: add SageMaker Serverless Inference endpoint launcher"
```

---

### Task 6: Manual end-to-end deployment, verification, and teardown

**No repo files — real AWS execution, cross-checking against the existing local inference code, and mandatory cleanup.**

- [ ] **Step 1: Deploy the real endpoint**

Run:
```bash
uv run python -m src.aws.deploy_endpoint \
  --image-uri 155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-serve:latest \
  --role-arn arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution \
  --model-data-url s3://aws-anomalies-mvtec-romeo/output/.../model.tar.gz \
  --endpoint-name aws-anomalies-bottle
```
Expected: the SDK polls until the endpoint reaches `InService` (a Serverless
cold start typically takes a few minutes), then prints `Endpoint déployé :
aws-anomalies-bottle`.

**Note on the Unicode console crash:** the same `sagemaker` SDK log-tip
emoji crash documented in the prior sub-project's Task 6 can occur here
too, on Windows terminals using the `cp1252` codec. If it happens, the
deploy call itself has usually already succeeded server-side — confirm with
`aws sagemaker describe-endpoint --endpoint-name aws-anomalies-bottle` via
direct `boto3` (or the AWS CLI, mindful of Git Bash path-mangling on any
argument that looks like a path) rather than relying on the crashed
process's exit code.

- [ ] **Step 2: Invoke the endpoint on a defective and a good test image**

Run:
```bash
uv run python -c "
import boto3

client = boto3.client('sagemaker-runtime', region_name='eu-west-1')

with open('data/mvtec/bottle/test/broken_large/000.png', 'rb') as f:
    anomaly_response = client.invoke_endpoint(
        EndpointName='aws-anomalies-bottle', ContentType='image/png', Body=f.read()
    )
print('anomaly image:', anomaly_response['Body'].read())

with open('data/mvtec/bottle/test/good/000.png', 'rb') as f:
    good_response = client.invoke_endpoint(
        EndpointName='aws-anomalies-bottle', ContentType='image/png', Body=f.read()
    )
print('good image:', good_response['Body'].read())
"
```
Expected: `is_anomaly: true` for the defective image, `is_anomaly: false` for
the good image.

- [ ] **Step 3: Cross-check against the local `AnomalyDetector`**

Compare the two scores from Step 2 against the scores obtained locally with
the same checkpoint via `AnomalyDetector` (already measured in the prior
sub-project's Task 6: `0.910` for the defective image, `0.253` for the good
image). Scores should match closely — same checkpoint, same code path,
only the HTTP transport differs. Exact bit-for-bit equality is not expected
(no fixed seed anywhere in the pipeline, same documented non-determinism as
the prior sub-project), but both classification outcomes must agree.

- [ ] **Step 4: Tear down the endpoint — mandatory**

Run:
```bash
aws sagemaker delete-endpoint --endpoint-name aws-anomalies-bottle --region eu-west-1
aws sagemaker delete-endpoint-config --endpoint-config-name aws-anomalies-bottle --region eu-west-1
aws sagemaker delete-model --model-name <model-name-printed-by-deploy> --region eu-west-1
```
Expected: all three resources deleted. Verify with
`aws sagemaker list-endpoints --region eu-west-1` — `aws-anomalies-bottle`
must not appear. **Unlike the Training Job in the prior sub-project (which
stops billing on its own once `Completed`), a Serverless Inference endpoint
keeps billing per-request-served and per-idle-memory-reserved until
explicitly deleted — this step is not optional.**

- [ ] **Step 5: Report to the user**

Summarize: deployment time, invocation results (scores and classification
outcomes for both test images), the cross-check against the local
`AnomalyDetector`, and explicit confirmation that the endpoint, its config,
and the model were all deleted.

---

## Self-Review Notes

- **Spec coverage:** HTTP server with `/ping`/`/invocations` (Task 1), BYOC serving container (Task 2), IAM execution-role and local-user extensions (Task 3), ECR repo + push (Task 4), endpoint deployment launcher using the `sagemaker` SDK (Task 5), real end-to-end deployment + verification + mandatory teardown (Task 6) — matches every section of `2026-07-31-aws-serverless-endpoint-design.md`.
- **Placeholder scan:** no TBD/TODO; account ID (`155466261331`), bucket name (`aws-anomalies-mvtec-romeo`), region (`eu-west-1`), and the measured local scores (`0.910`/`0.253`) from the prior sub-project are filled in literally rather than left as placeholders. The `model.tar.gz` S3 path in Tasks 5/6 uses `.../` because the exact job-specific path is only known once Task 6 re-reads it from the prior sub-project's actual output location — this is a real runtime lookup, not an unresolved placeholder.
- **Type/interface consistency:** `create_app(detector: AnomalyDetector) -> Flask` (Task 1) is called by `main()`, which `Dockerfile.sagemaker-serve`'s `ENTRYPOINT` invokes (Task 2). `DEFAULT_CHECKPOINT_PATH` (Task 1) is asserted against the exact filename (`model.ckpt`) that `train_entrypoint.py`'s `run_training` (prior sub-project) produces — a coupling test, mirroring the channel-name coupling test from the training sub-project. `deploy_endpoint(...) -> str` (Task 5) returns the endpoint name consumed directly by Task 6's invocation and teardown commands. No signature mismatches across tasks.
- **Ambiguity check:** clarified that the request body for `/invocations` is the raw image bytes (not JSON/base64), that the threshold is read from the checked-in `config/threshold.yaml` inside the image (not an environment variable), and that endpoint teardown in Task 6 is mandatory rather than optional, given the billing model difference from the one-off Training Job.
