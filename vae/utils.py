from random import randint

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchinfo import summary


def show_tensor(x: torch.Tensor):
    plt.imshow(x.permute(1, 2, 0).cpu())


def display_random_image(dataset: Dataset, random_index: int = None) -> int:
    if not random_index:
        random_index = randint(0, len(dataset))
    img, label = dataset[random_index]
    img = img.permute(1, 2, 0).cpu()
    class_ = dataset.classes[label]
    print(f"Picked index is {random_index}")
    print(f"Image class is {class_}")
    plt.imshow(img)
    plt.show()
    return random_index


def print_model(model: torch.nn.Module, shape: tuple = (128, 1, 28, 28)):
    print(
        summary(
            model,
            shape,
            col_names=["input_size", "output_size", "num_params", "trainable"],
        )
    )
