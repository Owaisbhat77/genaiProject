import argparse
import os
import time
import yaml
import torch
from torch.utils.data import DataLoader, Dataset

from .data import read_data, build_vocab, encode
from .model import MiniGPT


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def estimate_loss(model, data_loader, device, max_batches=2):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(data_loader):
            if i >= max_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["train"]["seed"])
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    text = read_data(cfg["train"]["data_path"])
    chars, stoi, itos = build_vocab(text)

    data = encode(text, stoi)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    vocab_size = len(chars)
    cfg["model"]["vocab_size"] = vocab_size

    train_ds = TextDataset(train_data, cfg["model"]["block_size"])
    val_ds = TextDataset(val_data, cfg["model"]["block_size"])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    model = MiniGPT(
        vocab_size=cfg["model"]["vocab_size"],
        block_size=cfg["model"]["block_size"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_embd=cfg["model"]["n_embd"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    os.makedirs("checkpoints", exist_ok=True)
    start = time.time()

    for it in range(cfg["train"]["max_iters"]):
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % cfg["train"]["eval_interval"] == 0:
            train_loss = estimate_loss(model, train_loader, device, max_batches=2)
            val_loss = estimate_loss(model, val_loader, device, max_batches=2)
            elapsed = time.time() - start
            print(
                f"step {it} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {elapsed:.1f}s"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "itos": itos,
            "config": cfg,
        },
        cfg["train"]["checkpoint_path"],
    )
    print(f"Saved checkpoint to {cfg['train']['checkpoint_path']}")


if __name__ == "__main__":
    main()
