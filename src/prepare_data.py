"""
prepare_data.py - Download CIFAR-10 and cache it as compressed NumPy arrays.

Primary source : Hugging Face mirror (uoft-cs/cifar10, parquet) - fast.
Fallback       : torchvision's official download (cs.toronto.edu) - was very slow
                 on our network (~3 MB/min), see DEMO.md "Issues".

Output: data/cifar10_train.npz, data/cifar10_test.npz   (x: uint8 N,32,32,3 ; y: int64 N)

Usage:
    python src/prepare_data.py
"""
import io
import os
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
EXPECTED = {"train": 50000, "test": 10000}


def from_parquet(split):
    import pandas as pd
    from PIL import Image
    path = os.path.join(DATA_DIR, f"{split}.parquet")
    if not os.path.exists(path):
        print(f"[data] downloading {split} split from Hugging Face ...")
        urllib.request.urlretrieve(HF_URL.format(split=split), path)
    df = pd.read_parquet(path)
    img_col = "img" if "img" in df.columns else df.columns[0]
    x = np.stack([np.array(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in df[img_col]])
    y = df["label"].to_numpy().astype(np.int64)
    return x, y


def from_torchvision(split):
    import torchvision
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=(split == "train"), download=True)
    return ds.data, np.array(ds.targets, dtype=np.int64)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ("train", "test"):
        out = os.path.join(DATA_DIR, f"cifar10_{split}.npz")
        if os.path.exists(out):
            print(f"[data] {out} already exists")
            continue
        try:
            x, y = from_parquet(split)
        except Exception as e:                       # network / format problem -> official source
            print(f"[data] HF mirror failed ({e}); falling back to torchvision download")
            x, y = from_torchvision(split)
        assert x.shape == (EXPECTED[split], 32, 32, 3) and x.dtype == np.uint8, x.shape
        assert set(np.unique(y)) == set(range(10))
        np.savez_compressed(out, x=x, y=y)
        print(f"[data] saved {out}: x{x.shape} y{y.shape} class counts {np.bincount(y).tolist()}")


if __name__ == "__main__":
    main()
