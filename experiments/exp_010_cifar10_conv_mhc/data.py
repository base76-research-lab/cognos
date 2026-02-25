"""CIFAR-10 dataladdning för exp_010."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10_loaders(batch_size: int = 128,
                        eval_n: int = 2000,
                        seed: int = 0,
                        data_dir: str = "/tmp/cifar10"):
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, eval_loader
