"""CIFAR-10 dataladdning för exp_010."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10_loaders(batch_size: int = 128,
                        eval_n: int = 2000,
                        seed: int = 0,
                        data_dir: str = "/tmp/cifar10",
                        num_workers: int | None = None,
                        pin_memory: bool | None = None):
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=tf_train)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=tf_test)

    rng = torch.Generator()
    rng.manual_seed(seed)
    idx = torch.randperm(len(test_ds), generator=rng)[:eval_n].tolist()
    eval_ds = Subset(test_ds, idx)

    has_cuda = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 2 if has_cuda else 0
    if pin_memory is None:
        pin_memory = has_cuda

    train_rng = torch.Generator()
    train_rng.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=train_rng,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, eval_loader
