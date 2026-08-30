"""Write a single CIFAR-10 test image to disk for use with POST /predict.

Falls back to a synthetic image when the dataset is not available locally, so
the script never blocks a verification run on a dataset download.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_image.png")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision import datasets

        dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=False)
        image, label = dataset[args.index]
        image.save(out)
        print(f"wrote {out} from CIFAR-10 test split (label index {label})")
    except Exception as exc:  # noqa: BLE001
        from PIL import Image

        Image.new("RGB", (32, 32), (70, 130, 180)).save(out)
        print(f"wrote synthetic {out} (dataset unavailable: {type(exc).__name__})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
