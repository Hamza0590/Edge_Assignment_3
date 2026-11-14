"""
qat.py - Q1.3: Quantization-Aware Training (QAT) + fine-tuning script.

Pipeline
  1. Initialise from the pretrained FP32 checkpoint
  2. Fuse modules with the QAT-aware fuser (Conv+BN kept as ConvBn2d so BN
     can still be updated / folded during training)
  3. prepare_qat() -> inserts FakeQuantize modules (weights + activations)
  4. Fine-tune with a small learning rate (GPU).  After `--freeze-observer-epoch`
     the observers are frozen; after `--freeze-bn-epoch` BN statistics are frozen
     (standard QAT recipe).
  5. Move to CPU and convert() -> real INT8 model
  6. Export: TorchScript INT8 model + state_dict (+ fake-quant model for ONNX QDQ export)

Usage:
    python src/qat.py --epochs 5 --lr 0.001
"""
import argparse
import copy
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.ao.quantization as tq

from common import (CKPT, CKPT_EXTRA, QUANT_ENGINE, RESULT_DIR, evaluate, file_size_mb, get_device,
                    get_loaders, load_fp32, set_seed)


def prepare_qat_model(fp32_model):
    torch.backends.quantized.engine = QUANT_ENGINE
    model = copy.deepcopy(fp32_model).cpu()
    model.eval()
    model.fuse_model(is_qat=True)                  # ConvBnReLU2d / ConvBn2d (QAT-friendly)
    model.qconfig = tq.get_default_qat_qconfig(QUANT_ENGINE)
    model.train()
    tq.prepare_qat(model, inplace=True)            # inserts FakeQuantize modules
    return model


def qat_finetune(model, train_loader, test_loader, device, epochs, lr, freeze_obs_ep, freeze_bn_ep):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    hist = []
    # Observers must see at least one epoch of data before being frozen, otherwise their
    # min/max stay at +inf/-inf and convert() fails ("min should be less than or equal to max").
    freeze_obs_ep, freeze_bn_ep = max(1, freeze_obs_ep), max(1, freeze_bn_ep)
    for ep in range(1, epochs + 1):
        if ep > freeze_obs_ep:
            model.apply(tq.disable_observer)                       # freeze scale / zero-point
        if ep > freeze_bn_ep:
            model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)  # freeze BN running stats
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(xb)                       # FP32 (no AMP: fake-quant needs fp32)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            seen += xb.size(0)
        acc = evaluate(model, test_loader, device)["accuracy"]
        hist.append(dict(epoch=ep, loss=loss_sum / seen, train_acc=100 * correct / seen,
                         fakequant_test_acc=acc, time_s=time.time() - t0))
        print(f"[QAT] epoch {ep}/{epochs} | loss {loss_sum / seen:.4f} | train {100 * correct / seen:.2f}% "
              f"| fake-quant test {acc:.2f}% | {time.time() - t0:.1f}s", flush=True)
    return hist


def main():
    ap = argparse.ArgumentParser(description="Quantization-Aware Training")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--freeze-observer-epoch", type=int, default=3)
    ap.add_argument("--freeze-bn-epoch", type=int, default=2)
    args = ap.parse_args()
    set_seed(1)
    device = get_device()

    train_loader, test_loader = get_loaders(args.batch_size, device=device)
    qat_model = prepare_qat_model(load_fp32())
    print("[QAT] fake-quant inserted, example block:\n", qat_model.layer1[0])

    hist = qat_finetune(qat_model, train_loader, test_loader, device, args.epochs, args.lr,
                        args.freeze_observer_epoch, args.freeze_bn_epoch)
    pd.DataFrame(hist).to_csv(os.path.join(RESULT_DIR, "qat_training_log.csv"), index=False)

    # Keep the fake-quant model (used for ONNX QDQ export)
    qat_model.cpu().eval()
    torch.save(qat_model.state_dict(), CKPT_EXTRA["qat_fakequant"])

    # Convert to real INT8
    int8 = tq.convert(copy.deepcopy(qat_model).eval(), inplace=False)
    torch.jit.save(torch.jit.script(int8), CKPT["qat"])
    torch.save(int8.state_dict(), CKPT_EXTRA["qat_state"])
    print(f"[QAT] saved {CKPT['qat']} ({file_size_mb(CKPT['qat']):.2f} MB)")

    _, cpu_test = get_loaders(500, device="cpu", train_aug=False)
    acc = evaluate(torch.jit.load(CKPT["qat"]), cpu_test)["accuracy"]
    print(f"[QAT] INT8 test accuracy (reloaded TorchScript) = {acc:.2f}%")


if __name__ == "__main__":
    main()
