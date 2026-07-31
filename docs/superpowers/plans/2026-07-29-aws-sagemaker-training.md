# AWS Phase — SageMaker Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train PatchCore on `bottle` via a SageMaker Training Job using a custom (BYOC) container, reusing the existing `build_datamodule`/`build_model` code, and verify the resulting checkpoint loads through the existing `AnomalyDetector`.

**Architecture:** A new `src/aws/` package (`train_entrypoint.py`, `launch_training.py`), a new `Dockerfile.sagemaker-train`, a new ECR repo `aws-anomalies-train`, a new IAM execution role `aws-anomalies-sagemaker-execution`, and an incremental extension of the existing `aws-anomalies-local` IAM user's policy. Data flows S3 (`bottle`, already uploaded) → SageMaker Training Job (BYOC) → S3 (`model.tar.gz`) → downloaded locally → verified via `AnomalyDetector`.

**Tech Stack:** Same as the local project (Python 3.10+, `uv`, `anomalib`, `torch` CPU) plus `sagemaker` (Python SDK, new dependency), AWS CLI, Docker, ECR, IAM.

**Prior state:** Sub-project 1 (`docs/superpowers/specs/2026-07-29-aws-s3-account-design.md`) is complete — bucket `aws-anomalies-mvtec-romeo` (`eu-west-1`), IAM user `aws-anomalies-local`, `bottle` dataset uploaded to `s3://aws-anomalies-mvtec-romeo/mvtec/bottle/`. AWS account ID: `155466261331`.

## Global Constraints

- Region `eu-west-1` only.
- Least privilege: every IAM change is additive and scoped to specific ARNs (no `*` on resources except where AWS mandates it, e.g. `ecr:GetAuthorizationToken`).
- Instance `ml.m5.xlarge` (billed per usage hour) — confirmed with the user before any job launch.
- No hardcoded values in Python source — experiment config path, image URI, role ARN, S3 URIs are all parameters (CLI args or config), never inlined.
- Reuse `build_datamodule`/`build_model` from `src/models/train.py` — no duplicated model-construction logic.
- No test hits the network, AWS, or trains a real model — `Engine`, `Estimator`, `build_datamodule`, `build_model` are monkeypatched in all automated tests, same style as `tests/test_deploy.py`.
- Category `bottle` only for this plan.

---

### Task 1: SageMaker training entrypoint

**Files:**
- Create: `src/aws/__init__.py`
- Create: `src/aws/train_entrypoint.py`
- Test: `tests/test_train_entrypoint.py`

**Interfaces:**
- Consumes: `load_experiment_config` (`src/config.py`), `build_datamodule`/`build_model` (`src/models/train.py`).
- Produces: `load_experiment_path(hyperparameters_path: Path) -> Path`, `run_training(experiment_path: Path, data_root: Path, model_dir: Path) -> Path` — consumed by Task 2 (baked into the training image's entrypoint) and exercised for real in Task 6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train_entrypoint.py`:

```python
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.aws import train_entrypoint


class _FakeCheckpointCallback:
    best_model_path = ""


class _FakeTrainer:
    checkpoint_callback = _FakeCheckpointCallback()


class _FakeEngine:
    def __init__(self) -> None:
        self.trainer = _FakeTrainer()

    def fit(self, datamodule, model) -> None:
        pass


def test_load_experiment_path_reads_hyperparameters_json(tmp_path: Path) -> None:
    hp_path = tmp_path / "hyperparameters.json"
    hp_path.write_text(json.dumps({"experiment": "config/experiment/bottle_wideresnet50.yaml"}))

    result = train_entrypoint.load_experiment_path(hp_path)

    assert result == Path("config/experiment/bottle_wideresnet50.yaml")


def test_run_training_overrides_root_and_copies_checkpoint(tmp_path, monkeypatch) -> None:
    fake_checkpoint = tmp_path / "source" / "model.ckpt"
    fake_checkpoint.parent.mkdir(parents=True)
    fake_checkpoint.write_bytes(b"fake-weights")
    _FakeCheckpointCallback.best_model_path = str(fake_checkpoint)

    captured_cfg = {}

    def _fake_build_datamodule(cfg):
        captured_cfg["root"] = cfg.root
        return object()

    monkeypatch.setattr(
        train_entrypoint, "load_experiment_config", lambda path: OmegaConf.create({"root": "data/mvtec"})
    )
    monkeypatch.setattr(train_entrypoint, "build_datamodule", _fake_build_datamodule)
    monkeypatch.setattr(train_entrypoint, "build_model", lambda cfg: object())
    monkeypatch.setattr(train_entrypoint, "Engine", _FakeEngine)

    model_dir = tmp_path / "model_dir"
    destination = train_entrypoint.run_training(
        experiment_path=Path("config/experiment/bottle_wideresnet50.yaml"),
        data_root=tmp_path / "sagemaker_data",
        model_dir=model_dir,
    )

    assert captured_cfg["root"] == str(tmp_path / "sagemaker_data")
    assert destination == model_dir / "model.ckpt"
    assert destination.read_bytes() == b"fake-weights"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_entrypoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aws'`.

- [ ] **Step 3: Write `src/aws/__init__.py`** (empty file)

- [ ] **Step 4: Write `src/aws/train_entrypoint.py`**

```python
"""Point d'entrée du container de training SageMaker : entraîne PatchCore avec les
données montées par SageMaker et écrit le checkpoint dans SM_MODEL_DIR."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from anomalib.engine import Engine

from src.config import load_experiment_config
from src.models.train import build_datamodule, build_model

HYPERPARAMETERS_PATH = Path("/opt/ml/input/config/hyperparameters.json")
DEFAULT_MODEL_DIR = Path("/opt/ml/model")
DEFAULT_DATA_ROOT = Path("/opt/ml/input/data/training")


def load_experiment_path(hyperparameters_path: Path = HYPERPARAMETERS_PATH) -> Path:
    """Lit le chemin de la config d'expérience depuis les hyperparamètres SageMaker."""
    hyperparameters = json.loads(hyperparameters_path.read_text(encoding="utf-8"))
    return Path(hyperparameters["experiment"])


def run_training(experiment_path: Path, data_root: Path, model_dir: Path) -> Path:
    """Entraîne PatchCore avec root surchargé par les données SageMaker, copie le checkpoint vers model_dir."""
    cfg = load_experiment_config(experiment_path)
    cfg.root = str(data_root)

    datamodule = build_datamodule(cfg)
    model = build_model(cfg)

    engine = Engine()
    engine.fit(datamodule=datamodule, model=model)

    checkpoint_path = Path(engine.trainer.checkpoint_callback.best_model_path)
    destination = model_dir / "model.ckpt"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination)
    return destination


def main() -> None:
    """Point d'entrée exécuté par le container au lancement du Training Job."""
    experiment_path = load_experiment_path()
    # SM_CHANNEL_TRAINING is only injected by the SageMaker Training Toolkit, which this
    # minimal BYOC image doesn't include; a real job mounts the channel at this fixed path.
    data_root = Path(os.environ.get("SM_CHANNEL_TRAINING", str(DEFAULT_DATA_ROOT)))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", str(DEFAULT_MODEL_DIR)))

    destination = run_training(experiment_path, data_root, model_dir)
    print(f"Checkpoint écrit dans {destination}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_train_entrypoint.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (21 passed — 19 prior + 2 new).

- [ ] **Step 7: Commit**

```bash
git add src/aws/__init__.py src/aws/train_entrypoint.py tests/test_train_entrypoint.py
git commit -m "feat: add SageMaker training entrypoint reusing build_datamodule/build_model"
```

---

### Task 2: Training container image

**Files:**
- Create: `Dockerfile.sagemaker-train`

**Interfaces:**
- Consumes: `src/aws/train_entrypoint.py` (Task 1), `pyproject.toml`/`uv.lock`.
- Produces: a Docker image buildable and runnable locally; pushed to ECR in Task 4.

- [ ] **Step 1: Write `Dockerfile.sagemaker-train`**

```dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY config/ config/

ENTRYPOINT ["uv", "run", "python", "-m", "src.aws.train_entrypoint"]
```

- [ ] **Step 2: Manual verification — build locally**

Run: `docker build --platform linux/amd64 -f Dockerfile.sagemaker-train -t aws-anomalies-train:local .`
Expected: build succeeds (same dependency layer as the existing inference `Dockerfile`, cached if built recently).

- [ ] **Step 3: Manual verification — smoke test the entrypoint locally**

**Important #1:** `anomalib`'s `MVTecAD` datamodule checks `(root / category).is_dir()` and
auto-downloads the *full* MVTec AD archive (all 15 categories, ~5GB) if that path is
missing — triggered automatically by `Engine.fit()` (Lightning calls `prepare_data()`
even though our own code never does). The mount below must therefore expose a `bottle/`
subdirectory under the mounted root — mount the *parent* of the local `bottle` folder
(`data/mvtec`), not `data/mvtec/bottle` directly, so the container sees
`/data/training/bottle/...` and never re-downloads. This mirrors how the real SageMaker
channel will be structured in Task 6 (S3 URI one level above `bottle/`).

**Important #2 (Git Bash / Windows only):** Git Bash's MSYS layer silently rewrites any
argument that looks like a leading `/path` — including `-v` mount destinations like
`/opt/ml/input/data/training` — into a Windows path (e.g. `C:/Program
Files/Git/opt/ml/input/data/training`) before Docker ever sees it. Inside the Linux
container this would make the mount destination wrong, so `(root / category).is_dir()`
would be `False` and the full-archive download would trigger even with the correct host
path from Important #1. Prefix the `docker run` invocation with `MSYS_NO_PATHCONV=1` to
stop this rewrite. This is purely a local-verification artifact of running Docker from
Git Bash on Windows, not something the real SageMaker platform is exposed to — the
production container mounts the `training` channel directly at
`/opt/ml/input/data/training` (no env var involved; see `DEFAULT_DATA_ROOT` in Task 1),
so the smoke test below mounts at that same fixed path rather than relying on an
`SM_CHANNEL_TRAINING` override, to exercise the exact fallback branch production uses.

Run (mounts local MVTec AD data at the real SageMaker BYOC training-channel path, and a
hyperparameters.json as SageMaker would write it):

```bash
mkdir -p /tmp/sm-smoke/config /tmp/sm-smoke/model
echo '{"experiment": "config/experiment/bottle_wideresnet50.yaml"}' > /tmp/sm-smoke/config/hyperparameters.json

MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$(cygpath -w "$(pwd)/data/mvtec"):/opt/ml/input/data/training:ro" \
  -v "$(cygpath -w "/tmp/sm-smoke/config"):/opt/ml/input/config:ro" \
  -v "$(cygpath -w "/tmp/sm-smoke/model"):/opt/ml/model" \
  aws-anomalies-train:local
```

Expected: `Checkpoint écrit dans /opt/ml/model/model.ckpt`; `/tmp/sm-smoke/model/model.ckpt`
exists locally afterward; container logs show **no** MVTec AD download step (training
starts directly with feature extraction — a run of a couple minutes, not ~20+ minutes).
This proves the container works, and that it does not silently redownload the dataset,
before spending SageMaker instance-hours on it.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.sagemaker-train
git commit -m "feat: add SageMaker BYOC training container"
```

---

### Task 3: IAM — execution role and extended local-user policy

**This task has no repo files — it is AWS console/CLI provisioning, done by the user with the assistant providing exact JSON and verification commands.**

- [ ] **Step 1: User creates the SageMaker execution role**

Console: IAM → Roles → Create role → Trusted entity type "AWS service" → Use case "SageMaker". Name: `aws-anomalies-sagemaker-execution`.

Attach this inline policy (`aws-anomalies-sagemaker-execution-policy`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DataAccess",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::aws-anomalies-mvtec-romeo",
        "arn:aws:s3:::aws-anomalies-mvtec-romeo/mvtec/*"
      ]
    },
    {
      "Sid": "OutputAccess",
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::aws-anomalies-mvtec-romeo/output/*"
    },
    {
      "Sid": "EcrAuth",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "EcrPullRepo",
      "Effect": "Allow",
      "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"],
      "Resource": "arn:aws:ecr:eu-west-1:155466261331:repository/aws-anomalies-train"
    },
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:eu-west-1:155466261331:log-group:/aws/sagemaker/TrainingJobs:*"
    }
  ]
}
```

- [ ] **Step 2: User extends the `aws-anomalies-local` policy**

Add these statements to the existing `aws-anomalies-s3-access` policy:

```json
{
  "Sid": "AwsAnomaliesEcrAuth",
  "Effect": "Allow",
  "Action": "ecr:GetAuthorizationToken",
  "Resource": "*"
},
{
  "Sid": "AwsAnomaliesEcrRepo",
  "Effect": "Allow",
  "Action": [
    "ecr:CreateRepository",
    "ecr:DescribeRepositories",
    "ecr:BatchCheckLayerAvailability",
    "ecr:PutImage",
    "ecr:InitiateLayerUpload",
    "ecr:UploadLayerPart",
    "ecr:CompleteLayerUpload"
  ],
  "Resource": "arn:aws:ecr:eu-west-1:155466261331:repository/aws-anomalies-train"
},
{
  "Sid": "AwsAnomaliesSageMakerTraining",
  "Effect": "Allow",
  "Action": ["sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob"],
  "Resource": "arn:aws:sagemaker:eu-west-1:155466261331:training-job/aws-anomalies-*"
},
{
  "Sid": "AwsAnomaliesPassExecutionRole",
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": "arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution"
},
{
  "Sid": "AwsAnomaliesLogsRead",
  "Effect": "Allow",
  "Action": ["logs:GetLogEvents", "logs:DescribeLogStreams"],
  "Resource": "arn:aws:logs:eu-west-1:155466261331:log-group:/aws/sagemaker/TrainingJobs:*"
}
```

(The `sagemaker` Python SDK streams training logs to the local terminal via `logs:GetLogEvents`/`DescribeLogStreams` — needed for `.fit()` to show progress in Task 6.)

- [ ] **Step 3: Assistant verifies**

Run: `aws iam get-role --role-name aws-anomalies-sagemaker-execution`
Expected: role exists, `AssumeRolePolicyDocument` shows `sagemaker.amazonaws.com` as principal.

---

### Task 4: ECR repository and image push

**No new repo files — AWS provisioning + a `docker push`, run by the assistant now that Task 3's permissions are in place.**

- [ ] **Step 1: Assistant creates the ECR repo**

Run: `aws ecr create-repository --repository-name aws-anomalies-train --region eu-west-1`
Expected: JSON output with `repositoryUri` (e.g. `155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-train`).

- [ ] **Step 2: Assistant authenticates Docker to ECR**

Run: `aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 155466261331.dkr.ecr.eu-west-1.amazonaws.com`
Expected: `Login Succeeded`.

- [ ] **Step 3: Assistant tags and pushes the image**

Run:
```bash
docker tag aws-anomalies-train:local 155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-train:latest
docker push 155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-train:latest
```
Expected: push completes, digest printed.

- [ ] **Step 4: Assistant verifies**

Run: `aws ecr describe-images --repository-name aws-anomalies-train --region eu-west-1`
Expected: one image listed, tag `latest`.

---

### Task 5: Training job launcher

**Files:**
- Modify: `pyproject.toml` (add `sagemaker` dependency)
- Create: `src/aws/launch_training.py`
- Test: `tests/test_launch_training.py`

**Interfaces:**
- Consumes: nothing from prior tasks at the code level (talks to AWS directly via the `sagemaker` SDK); consumes the image pushed in Task 4 and the role created in Task 3 as runtime parameters.
- Produces: `launch_training(image_uri, role_arn, training_data_uri, output_path, experiment_path, instance_type) -> str` (S3 URI of `model.tar.gz`) — run manually in Task 6.

- [ ] **Step 1: Add the dependency**

Run: `uv add sagemaker`
Expected: `pyproject.toml` and `uv.lock` updated, `uv sync` succeeds.

- [ ] **Step 2: Write the failing test**

Create `tests/test_launch_training.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_launch_training.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.aws.launch_training'`.

- [ ] **Step 4: Write `src/aws/launch_training.py`**

```python
"""Lance un SageMaker Training Job PatchCore (BYOC) sur les données déjà uploadées sur S3."""
from __future__ import annotations

import argparse

from sagemaker.estimator import Estimator

DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"


def build_estimator(image_uri: str, role_arn: str, output_path: str, instance_type: str = DEFAULT_INSTANCE_TYPE) -> Estimator:
    """Construit l'Estimator SageMaker pour l'image de training BYOC."""
    return Estimator(
        image_uri=image_uri,
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
    )


def launch_training(
    image_uri: str,
    role_arn: str,
    training_data_uri: str,
    output_path: str,
    experiment_path: str,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
) -> str:
    """Lance le Training Job, attend sa fin, retourne l'URI S3 du model.tar.gz produit."""
    estimator = build_estimator(image_uri, role_arn, output_path, instance_type)
    estimator.set_hyperparameters(experiment=experiment_path)
    estimator.fit({"training": training_data_uri})
    return estimator.model_data


def main() -> None:
    """CLI : lance un Training Job SageMaker pour une config d'expérience donnée."""
    parser = argparse.ArgumentParser(description="Lance un SageMaker Training Job PatchCore (BYOC).")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--training-data-uri", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    args = parser.parse_args()

    model_data = launch_training(
        image_uri=args.image_uri,
        role_arn=args.role_arn,
        training_data_uri=args.training_data_uri,
        output_path=args.output_path,
        experiment_path=args.experiment,
        instance_type=args.instance_type,
    )
    print(f"Modèle entraîné disponible sur S3 : {model_data}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_launch_training.py -v`
Expected: PASS (1 passed).

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (22 passed).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/aws/launch_training.py tests/test_launch_training.py
git commit -m "feat: add SageMaker training job launcher"
```

---

### Task 6: Manual end-to-end run and verification

**No repo files — real AWS execution and cross-checking against the existing local inference code.**

- [ ] **Step 1: Launch the real Training Job**

**Forward-looking cost note:** `--training-data-uri` points at the shared `mvtec/`
prefix (parent of `bottle/`), not a `bottle`-specific prefix, because the datamodule
needs `bottle/` to be a subdirectory of the mounted root. A SageMaker channel downloads
every key under the prefix it's given — today only `bottle` lives under `mvtec/`, so this
is free, but once `screw` or `carpet` is uploaded there too, every `bottle` training job
will silently download all categories present, tripling transfer time and billed
instance-time for no benefit. When a second category lands, give each category its own
channel prefix (e.g. `s3://…/channels/bottle/bottle/`) rather than pointing at the shared
parent.

Run:
```bash
uv run python -m src.aws.launch_training \
  --image-uri 155466261331.dkr.ecr.eu-west-1.amazonaws.com/aws-anomalies-train:latest \
  --role-arn arn:aws:iam::155466261331:role/aws-anomalies-sagemaker-execution \
  --training-data-uri s3://aws-anomalies-mvtec-romeo/mvtec/ \
  --output-path s3://aws-anomalies-mvtec-romeo/output/ \
  --experiment config/experiment/bottle_wideresnet50.yaml
```
Expected: SageMaker logs streamed to the terminal, job reaches `Completed`, prints `Modèle entraîné disponible sur S3 : s3://aws-anomalies-mvtec-romeo/output/.../model.tar.gz`.

- [ ] **Step 2: Download and extract the artifact**

Run:
```bash
aws s3 cp s3://aws-anomalies-mvtec-romeo/output/.../model.tar.gz /tmp/sagemaker-model.tar.gz
mkdir -p /tmp/sagemaker-model
tar -xzf /tmp/sagemaker-model.tar.gz -C /tmp/sagemaker-model
```
Expected: `/tmp/sagemaker-model/model.ckpt` exists.

- [ ] **Step 3: Verify the checkpoint through the existing `AnomalyDetector`**

Run:
```bash
uv run python -c "
from pathlib import Path
from src.models.detector import AnomalyDetector

det = AnomalyDetector(
    experiment_path=Path('config/experiment/bottle_wideresnet50.yaml'),
    checkpoint_path=Path('/tmp/sagemaker-model/model.ckpt'),
    threshold=0.523,
)
print('anomaly image:', det.predict(Path('data/mvtec/bottle/test/broken_large/000.png')))
print('good image:', det.predict(Path('data/mvtec/bottle/test/good/000.png')))
"
```
Expected: `is_anomaly: True` for the defective image, `is_anomaly: False` for the good image — same qualitative behavior as the local and Docker-based checkpoints verified in the prior sub-projects. Exact score need not match bit-for-bit (no fixed seed anywhere in the pipeline, documented as a known non-determinism in `docs/superpowers/specs/2026-07-29-aws-sagemaker-training-design.md`), but the classification outcome must be correct on both.

- [ ] **Step 4: Report to the user**

Summarize: job status, training duration, instance-hours consumed (cost estimate), and the verification outcome above. Remind the user this SageMaker Training Job is a one-off run — no persistent billed resource remains after it completes (unlike an endpoint), so no cleanup action is required at this stage.

---

## Self-Review Notes

- **Spec coverage:** training entrypoint (Task 1), BYOC container (Task 2), IAM execution role + extended local-user policy (Task 3), ECR repo + push (Task 4), job launcher (Task 5), real end-to-end run verified through `AnomalyDetector` (Task 6) — matches every section of `2026-07-29-aws-sagemaker-training-design.md`.
- **Placeholder scan:** no TBD/TODO; account ID (`155466261331`), bucket name (`aws-anomalies-mvtec-romeo`), and region (`eu-west-1`) are filled in literally since they're already known from sub-project 1, rather than left as placeholders.
- **Type/interface consistency:** `run_training(experiment_path, data_root, model_dir) -> Path` (Task 1) is called by `train_entrypoint.main()`, which is what `Dockerfile.sagemaker-train`'s `ENTRYPOINT` invokes (Task 2). `launch_training(...) -> str` (Task 5) returns the S3 URI consumed directly by Task 6's `aws s3 cp`. No signature mismatches across tasks.
- **Ambiguity check:** clarified that hyperparameters are read directly from `/opt/ml/input/config/hyperparameters.json` (always written by the SageMaker platform for every training job, BYOC or not) rather than assuming SageMaker Training Toolkit env-var injection, which this minimal container does not include.
