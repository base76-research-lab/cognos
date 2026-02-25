"""MNIST-dataladdning för exp_009."""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_mnist_loaders(batch_size: int = 256,
                      eval_n: int = 2000,
                      seed: int = 0,
                      data_dir: str = "/tmp/mnist"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)

    rng = torch.Generator()
    rng.manual_seed(seed)
    idx = torch.randperm(len(test_ds), generator=rng)[:eval_n].tolist()
    eval_ds = Subset(test_ds, idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader
