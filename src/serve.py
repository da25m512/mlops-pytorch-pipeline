"""FastAPI inference service.

Endpoints:
  GET  /health   liveness - 200 whenever the process is serving.
  GET  /ready    readiness - 200 once a checkpoint is loaded, 503 while not.
  POST /predict  multipart image upload, returns class probabilities.
  GET  /metadata checkpoint provenance (architecture, dataset, val accuracy).

The checkpoint path is configurable so the same image works when the
checkpoint arrives from a bind mount (Docker) or a PersistentVolumeClaim
(Kubernetes).
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import CLASS_NAMES, STATS  # noqa: E402
from model import get_model  # noqa: E402

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
MODEL_NAME = os.environ.get("MODEL_NAME", "classifier_v1.pt")
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", CHECKPOINT_DIR / MODEL_NAME))
TOP_K = int(os.environ.get("TOP_K", "5"))

app = FastAPI(title="mlops-pytorch-pipeline serving", version="1.0.0")

STATE: dict = {"model": None, "meta": {}, "transform": None, "error": None}


def _build_transform(dataset: str) -> transforms.Compose:
    mean, std = STATS[dataset]
    size = 32 if dataset == "cifar10" else 28
    mode = "RGB" if dataset == "cifar10" else "L"
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert(mode)),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_checkpoint() -> None:
    """Load the checkpoint into module state. Never raises - failures are
    recorded so /health can report 503 rather than crash-looping the pod."""
    try:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"checkpoint not found at {CHECKPOINT_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        architecture = checkpoint.get("architecture", "resnet18")
        num_classes = checkpoint.get("num_classes", 10)
        dataset = checkpoint.get("dataset", "cifar10")

        model = get_model(architecture=architecture, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        STATE["model"] = model
        STATE["transform"] = _build_transform(dataset)
        STATE["meta"] = {
            "architecture": architecture,
            "dataset": dataset,
            "num_classes": num_classes,
            "class_names": checkpoint.get("class_names", CLASS_NAMES[dataset]),
            "trained_epochs": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "checkpoint_path": str(CHECKPOINT_PATH),
        }
        STATE["error"] = None
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        STATE["model"] = None
        STATE["error"] = f"{type(exc).__name__}: {exc}"


@app.on_event("startup")
def on_startup() -> None:
    load_checkpoint()


@app.get("/health")
def health() -> JSONResponse:
    """Liveness: the process is up and serving HTTP.

    This deliberately does NOT depend on the checkpoint. The liveness probe
    restarts the container when it answers non-200, and a pod that is merely
    waiting for the training Job to write its checkpoint is not broken - it is
    not ready yet. Conflating the two produces a restart loop the pod can never
    escape, because each restart resets the clock before the checkpoint can
    appear.
    """
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "model_loaded": STATE["model"] is not None},
    )


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness: a checkpoint is loaded and the pod can answer /predict.

    Answering 503 here keeps the pod out of the Service's endpoints without
    killing it, so it joins automatically once the checkpoint lands.
    """
    if STATE["model"] is None:
        # Retry the load on every probe. The pod may have started before the
        # training Job wrote the checkpoint; loading only at startup would
        # leave it permanently unready even after the file appears.
        load_checkpoint()

    if STATE["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "reason": STATE["error"]},
        )
    return JSONResponse(status_code=200, content={"status": "ready", "model_loaded": True})


@app.get("/metadata")
def metadata() -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "model not loaded")
    return STATE["meta"]


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "model not loaded")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

    tensor = STATE["transform"](pil_image).unsqueeze(0)
    with torch.no_grad():
        probabilities = F.softmax(STATE["model"](tensor), dim=1)[0]

    class_names = STATE["meta"]["class_names"]
    k = min(TOP_K, len(class_names))
    top_scores, top_indices = torch.topk(probabilities, k)

    return {
        "filename": image.filename,
        "predicted_class": class_names[int(probabilities.argmax())],
        "probabilities": {
            class_names[i]: round(float(p), 6)
            for i, p in enumerate(probabilities.tolist())
        },
        "top_k": [
            {"class": class_names[int(i)], "probability": round(float(s), 6)}
            for s, i in zip(top_scores, top_indices, strict=True)
        ],
    }
