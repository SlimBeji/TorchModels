import torch


class Sample(torch.nn.Module):
    def __init__(self, latent_space_dim: int):
        super().__init__()
        self.dim = latent_space_dim

    def extract_parameters(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2
        assert x.shape[-1] == 2 * self.dim
        mean = x[:, : self.dim]
        logvar = x[:, self.dim :]
        return mean, logvar

    def generate(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if len(args) == 1:
            mean, logvar = self.extract_parameters(args[0])
        elif len(args) == 2:
            mean, logvar = args
        else:
            raise TypeError("Sample layer accepts only one or two tensors")

        return self.generate(mean, logvar)
