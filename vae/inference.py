import torch

from vae.model import VAE


def encode_decode(model: VAE, image: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        output = model(image.unsqueeze(dim=0))
        return output.squeeze(dim=0)


def encode(model: VAE, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        mean, logvar = model.encode(image.unsqueeze(dim=0))
        return mean, logvar


def decode(model: VAE, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        example = model.decode(mean, logvar)
        return example.squeeze(dim=0)
