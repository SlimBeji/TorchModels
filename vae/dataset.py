import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    datapath: str, device: str, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), lambda x: x.to(device)]
    )

    train_dataset = datasets.MNIST(
        datapath, train=True, transform=mnist_transforms, download=False
    )
    test_dataset = datasets.MNIST(
        datapath, train=False, transform=mnist_transforms, download=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
