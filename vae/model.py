import torch

from vae.layers.decoder import VAEDecoder
from vae.layers.encoder import VAEEncoder


class VAE(torch.nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],  # CWH
        latent_space_dim: int,
        conv_layers: list[tuple[int, int]],
        perceptron_layers: list[tuple[int, int]],
    ):
        super().__init__()
        self.latent_space_dim = latent_space_dim
        self.encoder = VAEEncoder(
            image_shape, latent_space_dim, conv_layers, perceptron_layers
        )
        self.deocder = VAEDecoder(
            image_shape,
            latent_space_dim,
            self.reverse_layers(conv_layers),
            self.reverse_layers(perceptron_layers),
        )

    @classmethod
    def reverse_layers(self, layers: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [tuple(reversed(l)) for l in list(reversed(layers))]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.encoder(x)
        mean = y[:, : self.latent_space_dim]
        logvar = y[:, self.latent_space_dim :]
        return mean, logvar

    def decode(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return self.deocder.decode(mean, logvar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deocder(self.encoder(x))
