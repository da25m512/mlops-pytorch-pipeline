"""API contract tests for the serving app."""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import get_model  # noqa: E402


def _write_checkpoint(path: Path) -> None:
    model = get_model("simple_cnn", 10)
    torch.save(
        {
            "epoch": 1,
            "architecture": "simple_cnn",
            "num_classes": 10,
            "dataset": "cifar10",
            "class_names": [f"class_{i}" for i in range(10)],
            "model_state_dict": model.state_dict(),
            "val_loss": 1.23,
            "val_accuracy": 0.42,
        },
        path,
    )


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 90, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "classifier_v1.pt"
    _write_checkpoint(checkpoint)
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))

    import serve

    importlib.reload(serve)
    with TestClient(serve.app) as test_client:
        yield test_client


def test_health_reports_ok_when_model_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_ready_is_200_when_model_loaded(client) -> None:
    assert client.get("/ready").status_code == 200


def test_metadata_exposes_provenance(client) -> None:
    body = client.get("/metadata").json()
    assert body["architecture"] == "simple_cnn"
    assert body["val_accuracy"] == pytest.approx(0.42)


def test_predict_returns_normalised_probabilities(client) -> None:
    response = client.post(
        "/predict", files={"image": ("test_image.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200

    body = response.json()
    assert len(body["probabilities"]) == 10
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["predicted_class"] in body["probabilities"]
    assert len(body["top_k"]) == 5


def test_predict_rejects_non_image_payload(client) -> None:
    response = client.post(
        "/predict", files={"image": ("notes.txt", b"this is not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_liveness_stays_200_without_a_checkpoint(tmp_path, monkeypatch) -> None:
    """A pod waiting for the training Job is not ready, but it is not broken:
    /health must stay 200 so the liveness probe does not restart it."""
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "absent.pt"))

    import serve

    importlib.reload(serve)
    with TestClient(serve.app, raise_server_exceptions=False) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        assert test_client.get("/ready").status_code == 503



def test_ready_picks_up_a_checkpoint_that_appears_after_startup(tmp_path, monkeypatch) -> None:
    """The serving pod starts before the training Job finishes. Once the
    checkpoint lands, the next readiness probe must load it and return 200."""
    checkpoint = tmp_path / "classifier_v1.pt"
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))

    import serve

    importlib.reload(serve)
    with TestClient(serve.app, raise_server_exceptions=False) as test_client:
        assert test_client.get("/ready").status_code == 503

        _write_checkpoint(checkpoint)

        assert test_client.get("/ready").status_code == 200
        assert test_client.get("/health").json()["model_loaded"] is True
