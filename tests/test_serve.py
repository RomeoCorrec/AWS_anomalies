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
