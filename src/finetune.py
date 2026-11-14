"""
finetune.py - Shared fine-tuning pipeline used by BOTH pruning approaches (Q2.4).

* For unstructured pruning the torch.nn.utils.prune masks stay attached
  while fine-tuning, so pruned weights remain exactly zero (weight = weight_orig * mask
  is recomputed in a forward-pre-hook every step).
* For structured pruning the network is already physically smaller, so a
  normal training loop is enough.

Can also be run stand-alone to fine-tune any saved pruned model further:
    python src/finetune.py --model structured --epochs 5 --lr 0.01
"""
import argparse

import torch

from common import CKPT, get_device, get_loaders, load_model, train_epochs, evaluate


def finetune(model, epochs, lr, batch_size=256, log_prefix="[finetune] ", loaders=None):
    """Fine-tune `model` on CIFAR-10 to recover accuracy. Returns (model, history)."""
    device = get_device()
    train_loader, test_loader = loaders or get_loaders(batch_size, device=device)
    hist = train_epochs(model, train_loader, test_loader, epochs, lr, device,
                        schedule="cosine", log_prefix=log_prefix)
    return model, hist


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a pruned model")
    ap.add_argument("--model", choices=["unstructured", "structured"], required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    model = load_model(args.model)
    if args.model == "unstructured":
        # re-attach masks from the zero pattern so zeros stay zero during fine-tuning
        import torch.nn.utils.prune as prune
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                prune.custom_from_mask(m, "weight", (m.weight != 0).float())
    model, _ = finetune(model, args.epochs, args.lr)
    model.cpu()
    if args.model == "unstructured":
        import torch.nn.utils.prune as prune
        for m in model.modules():
            if prune.is_pruned(m) and hasattr(m, "weight_mask"):
                prune.remove(m, "weight")
        torch.save(model.state_dict(), CKPT["unstructured"])
    else:
        torch.save({"mid_channels": model.mid_channels, "state_dict": model.state_dict()}, CKPT["structured"])
    _, test_loader = get_loaders(500, device="cpu", train_aug=False)
    print(f"[finetune] accuracy after fine-tuning = {evaluate(model, test_loader)['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
