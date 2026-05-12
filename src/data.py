import argparse
import os
import urllib.request

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading dataset to {path}...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    print("Download complete.")


def read_data(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi):
    return [stoi[c] for c in text]


def decode(tokens, itos):
    return "".join([itos[i] for i in tokens])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download Tiny Shakespeare dataset")
    parser.add_argument("--path", default="data/tiny_shakespeare.txt")
    args = parser.parse_args()

    if args.download:
        download_data(args.path)
    else:
        print(read_data(args.path)[:1000])


if __name__ == "__main__":
    main()
