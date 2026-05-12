# GenAI Project — Mini LLM From Scratch (Python + PyTorch + LangChain)

A compact, **from‑scratch** Transformer language model you can train on a small dataset (e.g., Tiny Shakespeare). Includes:
- training a GPT‑style model from scratch
- text generation
- a LangChain demo wrapper to use your model in chains

> Designed to be a **decent, impressive project** that runs locally while keeping the code readable.

## 1) Setup

```bash
# clone

git clone https://github.com/Owaisbhat77/genaiProject.git
cd genaiProject

# create venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install deps
pip install -r requirements.txt
```

## 2) Download dataset

```bash
python -m src.data --download
```

This fetches Tiny Shakespeare and creates `data/tiny_shakespeare.txt`.

## 3) Train the model

```bash
python -m src.train --config configs/base.yaml
```

The trained checkpoint is saved to `checkpoints/model.pt`.

## 4) Generate text

```bash
python -m src.generate --prompt "To be, or not to be" --checkpoint checkpoints/model.pt
```

## 5) LangChain demo

```bash
python -m src.langchain_demo --prompt "Write a short poem about the moon"
```

## Project Layout

```
.
├── configs/
│   └── base.yaml
├── data/
│   └── tiny_shakespeare.txt
├── checkpoints/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── langchain_demo.py
├── requirements.txt
└── README.md
```

## Notes
- This is a **from‑scratch Transformer** (token embedding + positional embedding + multi‑head self‑attention + MLP + layer norm).
- Keep batch size and model size small for CPU.
- Adjust `configs/base.yaml` to scale up if you have a GPU.

---

If you want extras (tokenizers, more datasets, Weights & Biases logging, evaluation, etc.), ask and I can add them.
