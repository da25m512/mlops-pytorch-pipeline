"""Model definitions for the CIFAR-10 image classifier.

Two architectures are supported:

* ``resnet18`` - torchvision ResNet-18 adapted for 32x32 inputs. The stock
  ImageNet stem (7x7 stride-2 conv + maxpool) throws away far too much spatial
  information on CIFAR-sized images, so the stem is replaced with a 3x3
  stride-1 conv and the maxpool is dropped.
* ``simple_cnn`` - a small three-block CNN used for fast smoke tests and for
  CPU-only environments where ResNet-18 is unnecessarily slow.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """A compact three-block convolutional classifier for 32x32 RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _resnet18_for_cifar(num_classes: int) -> nn.Module:
    """ResNet-18 with a CIFAR-friendly stem and a resized classification head."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


ARCHITECTURES = {
    "resnet18": _resnet18_for_cifar,
    "simple_cnn": SimpleCNN,
}


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Build a model by name.

    Raises:
        ValueError: if ``architecture`` is not one of the supported names.
    """
    key = architecture.lower()
    if key not in ARCHITECTURES:
        supported = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unknown architecture {architecture!r}; supported: {supported}")
    return ARCHITECTURES[key](num_classes=num_classes)
