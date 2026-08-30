"""Unit tests for the model factory and the training helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import ARCHITECTURES, get_model  # noqa: E402


@pytest.mark.parametrize("architecture", sorted(ARCHITECTURES))
def test_forward_pass_shape(architecture: str) -> None:
    """Every architecture maps a batch of 32x32 RGB images to class logits."""
    model = get_model(architecture=architecture, num_classes=10)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 32, 32))
    assert logits.shape == (4, 10)


def test_num_classes_is_respected() -> None:
    model = get_model(architecture="simple_cnn", num_classes=7)
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 7)


def test_unknown_architecture_raises() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        get_model(architecture="not-a-model")


def test_resnet_stem_is_cifar_sized() -> None:
    """The ImageNet stem would downsample 32x32 inputs too aggressively."""
    model = get_model(architecture="resnet18", num_classes=10)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)


def test_model_is_trainable() -> None:
    """One optimiser step must actually move the parameters."""
    model = get_model(architecture="simple_cnn", num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    before = model.classifier[-1].weight.detach().clone()
    loss = criterion(model(torch.randn(8, 3, 32, 32)), torch.randint(0, 10, (8,)))
    loss.backward()
    optimizer.step()

    assert not torch.allclose(before, model.classifier[-1].weight)
