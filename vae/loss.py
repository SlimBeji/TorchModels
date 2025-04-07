import torch


class VAELoss(torch.nn.Module):
    def __init__(self, beta: float = 1):
        super().__init__()
        self.beta = beta
        self.mse = torch.nn.MSELoss(reduction="sum")

    def forward(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = self.mse(output, input_)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss
