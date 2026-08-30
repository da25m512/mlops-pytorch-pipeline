"""Dataset and DataLoader construction for CIFAR-10 / Fashion-MNIST."""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Per-channel statistics for the two supported datasets.
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
}

CLASS_NAMES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
    "fashion_mnist": [
        "t_shirt_top", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle_boot",
    ],
}


def get_transforms(train: bool = True, dataset: str = "cifar10") -> transforms.Compose:
    """Build the transform pipeline.

    Augmentation (random flip + random crop) is applied to the training split
    only; the validation split gets the deterministic tensor/normalise pair so
    that metrics are comparable across epochs.
    """
    mean, std = STATS[dataset]
    normalise = transforms.Normalize(mean=mean, std=std)

    if not train:
        return transforms.Compose([transforms.ToTensor(), normalise])

    augment = [transforms.RandomHorizontalFlip()]
    if dataset == "cifar10":
        augment.append(transforms.RandomCrop(32, padding=4))
    return transforms.Compose(augment + [transforms.ToTensor(), normalise])


def _dataset_factory(dataset: str):
    if dataset == "cifar10":
        return datasets.CIFAR10
    if dataset == "fashion_mnist":
        return datasets.FashionMNIST
    raise ValueError(f"unsupported dataset {dataset!r}; expected cifar10 or fashion_mnist")


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset: str = "cifar10",
    download: bool = True,
    subset_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)``.

    Args:
        data_dir: Root directory the dataset is read from / downloaded into.
            In containers this is a mounted volume so the download survives
            container restarts.
        subset_size: When set, truncate both splits to this many samples. Used
            by smoke tests and by the Kubernetes Job's fast validation run so a
            full end-to-end pass does not take an hour on CPU.
    """
    factory = _dataset_factory(dataset)

    train_dataset = factory(
        root=data_dir,
        train=True,
        download=download,
        transform=get_transforms(train=True, dataset=dataset),
    )
    val_dataset = factory(
        root=data_dir,
        train=False,
        download=download,
        transform=get_transforms(train=False, dataset=dataset),
    )

    if subset_size:
        from torch.utils.data import Subset

        train_dataset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(subset_size, len(val_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
