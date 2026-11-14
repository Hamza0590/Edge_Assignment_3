"""
train_baseline.py - Q1.1 / Q2.1: train the FP32 ResNet-18 baseline on CIFAR-10.

Output: models/resnet18_fp32.pth (state_dict) + results/baseline_training_log.csv

Usage:
    python src/train_baseline.py --epochs 30 --lr 0.1 --batch-size 256
"""
import argparse
import os

import pandas as pd
import torch

from common import CKPT, RESULT_DIR, ResNet18, get_device, get_loaders, set_seed, train_epochs, evaluate


def main():
    ap = argparse.ArgumentParser(description="Train FP32 ResNet-18 on CIFAR-10")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[baseline] device = {device}")
    train_loader, test_loader = get_loaders(args.batch_size, device=device)

    model = ResNet18().to(device).to(memory_format=torch.channels_last)
    # per-epoch training state -> the run can be paused (create a STOP file) and resumed
    state_path = os.path.join(os.path.dirname(CKPT["fp32"]), "baseline_train_state.pth")
    hist = train_epochs(model, train_loader, test_loader, args.epochs, args.lr, device,
                        log_prefix="[baseline] ", resume_path=state_path)
    if os.path.exists(state_path):
        os.remove(state_path)              # training finished -> resume state no longer needed

    model = model.to(memory_format=torch.contiguous_format).cpu()
    torch.save(model.state_dict(), CKPT["fp32"])
    pd.DataFrame(hist).to_csv(os.path.join(RESULT_DIR, "baseline_training_log.csv"), index=False)

    acc = evaluate(model.to(device), test_loader, device)["accuracy"]
    print(f"[baseline] final FP32 test accuracy = {acc:.2f}%  -> saved {CKPT['fp32']}")


if __name__ == "__main__":
    main()
