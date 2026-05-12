import argparse
import torch
from langchain.llms.base import LLM
from typing import List, Optional

from .model import MiniGPT
from .data import encode, decode


class MiniGPTLLM(LLM):
    checkpoint_path: str = "checkpoints/model.pt"

    @property
    def _llm_type(self) -> str:
        return "mini-gpt"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
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

        idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long)
        out = model.generate(idx, max_new_tokens=200, temperature=1.0, top_k=50)
        text = decode(out[0].tolist(), itos)

        if stop:
            for s in stop:
                text = text.split(s)[0]
        return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Write a short poem about the ocean")
    args = parser.parse_args()

    llm = MiniGPTLLM()
    print(llm(args.prompt))


if __name__ == "__main__":
    main()
