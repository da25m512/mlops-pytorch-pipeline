"""Tests for config resolution and the train/evaluate loops."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train as train_module  # noqa: E402
from model import get_model  # noqa: E402


def _fake_loader(n: int = 16) -> DataLoader:
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text("model:\n  architecture: simple_cnn\n  num_classes: 10\n")
    config = train_module.load_config(config_file)
    assert config["model"]["architecture"] == "simple_cnn"


def test_config_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("model: {}\n")
    monkeypatch.setenv("TRAINING_CONFIG", str(config_file))
    assert train_module.resolve_config_path() == config_file


def test_missing_config_override_raises(monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_CONFIG", "/nope/missing.yaml")
    with pytest.raises(FileNotFoundError):
        train_module.resolve_config_path()


def test_train_one_epoch_returns_finite_metrics() -> None:
    model = get_model("simple_cnn", 10)
    loader = _fake_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, accuracy = train_module.train_one_epoch(
        model, loader, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert loss == pytest.approx(loss)  # not NaN
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_does_not_update_weights() -> None:
    model = get_model("simple_cnn", 10)
    before = [p.detach().clone() for p in model.parameters()]
    train_module.evaluate(
        model, _fake_loader(), torch.nn.CrossEntropyLoss(), torch.device("cpu")
    )
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_log_emits_one_json_object_per_line(capsys) -> None:
    train_module.log(epoch=1, train_loss=0.5)
    captured = capsys.readouterr().out.strip()
    assert json.loads(captured) == {"epoch": 1, "train_loss": 0.5}
