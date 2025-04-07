import torch


class Reshape(torch.nn.Module):
    def __init__(self, start_dim: int, new_shape: tuple):
        super().__init__()
        self.start_dim = start_dim
        self.new_shape = new_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[: self.start_dim], *self.new_shape)
