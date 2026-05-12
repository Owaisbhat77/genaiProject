# GenAI Project — Mini LLM From Scratch (Python + PyTorch + LangChain)

A compact, **from‑scratch** Transformer language model that runs **fast on CPU**. Includes:
- training a GPT‑style model from scratch
- text generation
- a LangChain demo wrapper to use your model in chains
- a simple Markov Chain poetry generator

> Optimized for **quick runs** on CPU so you can demo it immediately.

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

## 3) Train the model (FAST CPU)

```bash
python -u -m src.train --config configs/fast.yaml
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

## 6) Markov Chain poetry (from scratch)

```bash
python -m src.markov_poetry --input data/tiny_shakespeare.txt --order 2 --words 120
```

Try different settings:
- `--order 1` (more random)
- `--order 3` (more coherent)
- `--words 200` (longer poem)

## Project Layout

```
.
├── configs/
│   ├── base.yaml
│   └── fast.yaml
├── data/
│   └── tiny_shakespeare.txt
├── checkpoints/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── langchain_demo.py
│   └── markov_poetry.py
├── requirements.txt
└── README.md
```

## Notes
- This is a **from‑scratch Transformer** (token embedding + positional embedding + multi‑head self‑attention + MLP + layer norm).
- Use `configs/fast.yaml` for quick CPU demos.
- If you have a GPU, increase block size, layers, and `max_iters`.

---

If you want a higher‑quality config or extra features (tokenizers, more datasets, eval, UI), ask and I’ll add them.
