import argparse
import torch

from .model import MiniGPT
from .data import encode, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    model = MiniGPT(
        vocab_size=cfg["model"]["vocab_size"],
        block_size=cfg["model"]["block_size"],
        n_layers=cfg["model"]["n_layers"],
        n_heads=cfg["model"]["n_heads"],
        n_embd=cfg["model"]["n_embd"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([encode(args.prompt, stoi)], dtype=torch.long)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(decode(out[0].tolist(), itos))


if __name__ == "__main__":
    main()
