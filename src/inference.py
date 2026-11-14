"""
inference.py - Unified deployment / inference pipeline for ALL FIVE models (Q1.4, Q2.5).

    fp32          FP32 ResNet-18 (PyTorch state_dict)
    ptq           PTQ INT8 (TorchScript, onednn quantized kernels)
    qat           QAT INT8 (TorchScript, onednn quantized kernels)
    unstructured  80%-sparse unstructured-pruned ResNet-18
    structured    channel-pruned, physically smaller ResNet-18

Everything runs on CPU (edge-deployment scenario).

Examples
    # classify your own image(s) with every model
    python src/inference.py --model all --image samples/cat_0.png
    # classify a few CIFAR-10 test images with one model
    python src/inference.py --model qat --testset 8
    # dump some test images to samples/ for the demo
    python src/inference.py --save-samples 10
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image

from common import (CIFAR10_CLASSES, MODEL_NAMES, ROOT, _load_split, load_model, normalize)

SAMPLE_DIR = os.path.join(ROOT, "samples")


def preprocess_image(path):
    """Any RGB image -> normalised 1x3x32x32 tensor (resized to CIFAR resolution)."""
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).unsqueeze(0)
    return normalize(x)


class Predictor:
    """Load once, predict many times."""

    def __init__(self, name):
        self.name = name
        self.model = load_model(name)
        with torch.no_grad():                 # warm-up: TorchScript profiles/optimises on first runs
            for _ in range(3):
                self.model(torch.zeros(1, 3, 32, 32))

    @torch.no_grad()
    def __call__(self, x, topk=3):
        t = time.perf_counter()
        probs = self.model(x).float().softmax(1)
        ms = (time.perf_counter() - t) * 1000
        p, k = probs.topk(topk, 1)
        return [[(CIFAR10_CLASSES[int(c)], float(v)) for v, c in zip(pr, kr)] for pr, kr in zip(p, k)], ms


def save_samples(n):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    x, y = _load_split(False)
    for i in range(n):
        name = f"{CIFAR10_CLASSES[int(y[i])]}_{i}.png"
        Image.fromarray(x[i].permute(1, 2, 0).numpy()).save(os.path.join(SAMPLE_DIR, name))
    print(f"saved {n} test images to {SAMPLE_DIR}")


def main():
    ap = argparse.ArgumentParser(description="Inference with FP32 / PTQ / QAT / pruned models")
    ap.add_argument("--model", default="all", choices=MODEL_NAMES + ["all"])
    ap.add_argument("--image", nargs="*", help="image file(s) or glob pattern(s)")
    ap.add_argument("--testset", type=int, default=0, help="classify the first N CIFAR-10 test images")
    ap.add_argument("--save-samples", type=int, default=0, help="write N test images to samples/")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    if args.save_samples:
        save_samples(args.save_samples)
        if not (args.image or args.testset):
            return

    names = MODEL_NAMES if args.model == "all" else [args.model]
    predictors = [Predictor(n) for n in names]

    if args.image:
        files = sorted({f for pat in args.image for f in glob.glob(pat)})
        for f in files:
            x = preprocess_image(f)
            print(f"\nImage: {os.path.basename(f)}")
            print(f"  {'model':<13} {'prediction':<11} {'conf':>6}  {'top-3':<45} {'latency':>8}")
            for p in predictors:
                (top,), ms = p(x)
                t3 = ", ".join(f"{c}:{v:.2f}" for c, v in top)
                print(f"  {p.name:<13} {top[0][0]:<11} {top[0][1]:6.3f}  {t3:<45} {ms:6.2f}ms")

    if args.testset:
        xs, ys = _load_split(False)
        x = normalize(xs[:args.testset])
        truth = [CIFAR10_CLASSES[int(c)] for c in ys[:args.testset]]
        header = f"{'#':>3} {'ground truth':<12}" + "".join(f" {n:<13}" for n in names)
        print("\n" + header + "\n" + "-" * len(header))
        results = [p(x, topk=1)[0] for p in predictors]
        for i in range(args.testset):
            row = f"{i:>3} {truth[i]:<12}"
            for r in results:
                c = r[i][0][0]
                row += f" {c + ('' if c == truth[i] else ' X'):<13}"
            print(row)
        print("(X = misclassified)")


if __name__ == "__main__":
    main()
