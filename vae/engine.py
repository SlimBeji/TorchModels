import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vae.loss import VAELoss
from vae.model import VAE


def train(
    model: VAE,
    dataloader: DataLoader,
    loss_fn: VAELoss,
    optimizer: torch.optim.Optimizer,
) -> float:
    train_loss = 0
    model.train()
    for x_batch, _ in tqdm(dataloader):
        # Forward pass
        mean_batch, logvar_batch = model.encode(x_batch)
        output_batch = model.decode(mean_batch, logvar_batch)

        # Loss computations
        loss = loss_fn(output_batch, x_batch, mean_batch, logvar_batch)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader.dataset)
    return round(train_loss, 8)


def test(model: VAE, dataloader: DataLoader, loss_fn: VAELoss) -> float:
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for x_batch, _ in tqdm(dataloader):
            # Forward pass
            mean_batch, logvar_batch = model.encode(x_batch)
            output_batch = model.decode(mean_batch, logvar_batch)

            # Loss computations
            loss = loss_fn(output_batch, x_batch, mean_batch, logvar_batch)
            test_loss += loss.item()

    test_loss /= len(dataloader.dataset)
    return round(test_loss, 8)


def train_model(
    model: VAE,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    loss_fn: VAELoss,
    optimizer: torch.optim.Optimizer,
    epochs: int,
):
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch} Training -----")
        train_loss = train(model, train_dataloader, loss_fn, optimizer)
        print(f"Epoch {epoch} Testing -----")
        test_loss = test(model, test_dataloader, loss_fn)
        print(f"Epoch {epoch} | Train Loss: {train_loss} | Test Loss {test_loss}")
        print("=" * 50)
        print("=" * 50)
        print("=" * 50)
        print("=" * 50)
