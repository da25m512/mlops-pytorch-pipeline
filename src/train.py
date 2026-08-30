"""Training entrypoint.

Reads hyperparameters from a YAML config, trains an image classifier, emits
one JSON object per line to stdout so the logs are machine-parsable when the
job runs under Docker or Kubernetes, and writes checkpoints to a configurable
directory. Early stopping halts training once validation loss has failed to
improve for ``early_stopping_patience`` consecutive epochs.

Config resolution order:
  1. ``TRAINING_CONFIG`` environment variable
  2. ``/app/configs/training_config.yaml``   (ConfigMap / volume mount)
  3. ``configs/training_config.yaml``        (repository checkout)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import CLASS_NAMES, get_dataloaders  # noqa: E402
from model import get_model  # noqa: E402

CONFIG_CANDIDATES = (
    Path("/app/configs/training_config.yaml"),
    Path("configs/training_config.yaml"),
)


def log(**fields) -> None:
    """Emit a single structured JSON line and flush immediately."""
    print(json.dumps(fields), flush=True)


def resolve_config_path() -> Path:
    override = os.environ.get("TRAINING_CONFIG")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"TRAINING_CONFIG points at a missing file: {path}")
        return path
    for candidate in CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "no training config found; set TRAINING_CONFIG or mount one at "
        + " or ".join(str(c) for c in CONFIG_CANDIDATES)
    )


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as handle:
        return yaml.safe_load(handle)


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def main() -> int:
    config_path = resolve_config_path()
    config = load_config(config_path)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    out_cfg = config["output"]

    dataset_name = data_cfg.get("dataset", "cifar10")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(
        event="run_start",
        config_path=str(config_path),
        device=str(device),
        architecture=model_cfg["architecture"],
        dataset=dataset_name,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
    )

    model = get_model(
        architecture=model_cfg["architecture"],
        num_classes=model_cfg["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 2),
        dataset=dataset_name,
        subset_size=data_cfg.get("subset_size"),
    )
    log(
        event="data_ready",
        train_batches=len(train_loader),
        val_batches=len(val_loader),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(out_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / out_cfg["model_name"]

    patience = train_cfg["early_stopping_patience"]
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(train_cfg["epochs"]):
        started = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log(
            epoch=epoch + 1,
            train_loss=round(train_loss, 4),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(val_acc, 4),
            seconds=round(time.time() - started, 1),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "architecture": model_cfg["architecture"],
                    "num_classes": model_cfg["num_classes"],
                    "dataset": dataset_name,
                    "class_names": CLASS_NAMES[dataset_name],
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            log(event="checkpoint_saved", path=str(save_path), val_loss=round(val_loss, 4))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(event="early_stopping", epoch=epoch + 1, patience=patience)
                break

    log(
        event="training_complete",
        best_val_loss=round(best_val_loss, 4),
        checkpoint=str(save_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
