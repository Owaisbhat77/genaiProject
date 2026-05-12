"""Simple Markov Chain poetry generator (from scratch).

Builds a word-level Markov chain from a text file and generates new text.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict, deque
from pathlib import Path


def tokenize(text: str) -> list[str]:
    # Keep punctuation as separate tokens for more natural line endings.
    tokens = re.findall(r"[A-Za-z']+|[.,!?;:]|\n", text)
    return [t for t in tokens if t.strip() != ""]


def build_chain(tokens: list[str], order: int) -> dict[tuple[str, ...], list[str]]:
    chain: dict[tuple[str, ...], list[str]] = defaultdict(list)
    window: deque[str] = deque(maxlen=order)

    for token in tokens:
        if len(window) == order:
            chain[tuple(window)].append(token)
        window.append(token)

    return chain


def generate(chain: dict[tuple[str, ...], list[str]], order: int, words: int, seed: str | None) -> str:
    if not chain:
        return ""

    rng = random.Random(seed)
    start_state = rng.choice(list(chain.keys()))
    state = deque(start_state, maxlen=order)
    output: list[str] = list(state)

    while len([w for w in output if w != "\n"]) < words:
        choices = chain.get(tuple(state))
        if not choices:
            state = deque(rng.choice(list(chain.keys())), maxlen=order)
            output.extend(list(state))
            continue
        next_token = rng.choice(choices)
        output.append(next_token)
        state.append(next_token)

    # Post-process spacing: no space before punctuation, preserve newlines.
    text_parts: list[str] = []
    for token in output:
        if token == "\n":
            text_parts.append("\n")
            continue
        if token in {".", ",", "!", "?", ";", ":"}:
            if text_parts:
                text_parts[-1] = text_parts[-1].rstrip() + token
            else:
                text_parts.append(token)
        else:
            text_parts.append(token + " ")
    return "".join(text_parts).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov Chain poetry generator")
    parser.add_argument("--input", default="data/tiny_shakespeare.txt", help="Path to training text")
    parser.add_argument("--order", type=int, default=2, help="Markov order (n-gram size)")
    parser.add_argument("--words", type=int, default=120, help="Number of words to generate")
    parser.add_argument("--seed", default=None, help="Random seed for reproducible output")

    args = parser.parse_args()
    text_path = Path(args.input)

    if not text_path.exists():
        raise SystemExit(f"Input file not found: {text_path}")

    text = text_path.read_text(encoding="utf-8")
    tokens = tokenize(text)
    chain = build_chain(tokens, args.order)
    generated = generate(chain, args.order, args.words, args.seed)
    print(generated)


if __name__ == "__main__":
    main()
