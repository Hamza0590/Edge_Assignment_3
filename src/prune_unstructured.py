"""
prune_unstructured.py - Q2.2: iterative global magnitude (L1) unstructured pruning.

Algorithm
  * Parameters pruned: weights of every Conv2d and the final Linear layer.
  * Global magnitude pruning: in each iteration the smallest-|w| weights across
    ALL these layers are masked (torch.nn.utils.prune.global_unstructured +
    L1Unstructured).  Global ranking lets sensitive layers keep more weights.
  * Iterative schedule: N iterations reach the target sparsity S with the
    per-iteration amount  a = 1 - (1 - S)^(1/N)  (applied to *remaining* weights,
    PyTorch's PruningContainer composes the masks).
  * After each iteration: fine-tune (masks attached -> zeros stay zero).
  * At the end: prune.remove() makes the pruning permanent (weight_orig*mask
    is baked into .weight, hooks / masks removed).
  * Exports the dense checkpoint (drop-in for ResNet18) and a sparse (COO)
    checkpoint that actually stores fewer numbers.

Usage:
    python src/prune_unstructured.py --target-sparsity 0.8 --iterations 4 --ft-epochs 2 --final-ft-epochs 3
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import (CKPT, CKPT_EXTRA, RESULT_DIR, count_params, evaluate, file_size_mb, get_device,
                    get_loaders, load_fp32, set_seed)
from finetune import finetune


def prunable_params(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def global_sparsity(model):
    tot = zeros = 0
    for m, n in prunable_params(model):
        w = getattr(m, n)
        tot += w.numel()
        zeros += int((w == 0).sum())
    return 100.0 * zeros / tot


def layer_sparsity_table(model):
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight
            rows.append(dict(layer=name, params=w.numel(), sparsity_pct=100.0 * float((w == 0).sum()) / w.numel()))
    return pd.DataFrame(rows)


def to_sparse_state_dict(state_dict):
    """Compressed storage for pruned weights: 1-bit packed mask + non-zero values.

    (torch COO format was tried first but it stores an int64 index per dimension
    for every non-zero -> 4x8 = 32 extra bytes per weight for conv kernels, which made
    the file BIGGER than the dense one.  A bitmask costs only 1 bit per weight.)
    """
    out = {}
    for k, v in state_dict.items():
        if k.endswith("weight") and v.dim() > 1 and (v == 0).float().mean() > 0.3:
            mask = (v != 0).flatten().numpy()
            out[k] = {"__packed__": True, "shape": tuple(v.shape),
                      "mask": torch.from_numpy(np.packbits(mask)),
                      "values": v.flatten()[torch.from_numpy(mask)].clone()}
        else:
            out[k] = v
    return out


def from_sparse_state_dict(packed):
    """Inverse of to_sparse_state_dict -> normal dense state_dict."""
    out = {}
    for k, v in packed.items():
        if isinstance(v, dict) and v.get("__packed__"):
            n = int(np.prod(v["shape"]))
            mask = torch.from_numpy(np.unpackbits(v["mask"].numpy())[:n].astype(bool))
            dense = torch.zeros(n, dtype=v["values"].dtype)
            dense[mask] = v["values"]
            out[k] = dense.view(v["shape"])
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="Iterative magnitude-based unstructured pruning")
    ap.add_argument("--target-sparsity", type=float, default=0.8)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--ft-epochs", type=int, default=2, help="fine-tune epochs after each iteration")
    ap.add_argument("--final-ft-epochs", type=int, default=3, help="extra fine-tune epochs at the end")
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()
    set_seed(7)
    device = get_device()
    loaders = get_loaders(256, device=device)

    model = load_fp32().to(device)
    base_acc = evaluate(model, loaders[1], device)["accuracy"]
    print(f"[unstructured] baseline accuracy {base_acc:.2f}%")
    log = [dict(stage="baseline", iteration=0, sparsity_pct=0.0, acc_before_ft=base_acc, acc_after_ft=base_acc)]

    per_iter = 1 - (1 - args.target_sparsity) ** (1 / args.iterations)
    print(f"[unstructured] per-iteration amount = {per_iter:.4f} of remaining weights")
    for it in range(1, args.iterations + 1):
        prune.global_unstructured(prunable_params(model), pruning_method=prune.L1Unstructured, amount=per_iter)
        sp = global_sparsity(model)
        acc_b = evaluate(model, loaders[1], device)["accuracy"]
        print(f"[unstructured] iter {it}: sparsity {sp:.2f}% | acc before FT {acc_b:.2f}%")
        epochs = args.ft_epochs + (args.final_ft_epochs if it == args.iterations else 0)
        finetune(model, epochs, args.lr, log_prefix=f"[unstructured it{it}] ", loaders=loaders)
        acc_a = evaluate(model, loaders[1], device)["accuracy"]
        log.append(dict(stage="prune+finetune", iteration=it, sparsity_pct=sp, acc_before_ft=acc_b, acc_after_ft=acc_a))

    # ---- make pruning permanent -----------------------------------------
    for m, n in prunable_params(model):
        prune.remove(m, n)
    model.cpu().eval()
    assert not any(k.endswith("_mask") or k.endswith("_orig") for k in model.state_dict())

    torch.save(model.state_dict(), CKPT["unstructured"])
    torch.save(to_sparse_state_dict(model.state_dict()), CKPT_EXTRA["unstructured_sparse"])
    # round-trip check of the compressed checkpoint
    restored = from_sparse_state_dict(torch.load(CKPT_EXTRA["unstructured_sparse"], weights_only=False))
    assert all(torch.equal(restored[k], v) for k, v in model.state_dict().items())
    tot, nz = count_params(model)
    print(f"[unstructured] final global sparsity (conv+fc) {global_sparsity(model):.2f}% | params {tot:,} "
          f"non-zero {nz:,}")
    print(f"[unstructured] dense ckpt {file_size_mb(CKPT['unstructured']):.2f} MB | sparse ckpt "
          f"{file_size_mb(CKPT_EXTRA['unstructured_sparse']):.2f} MB")

    pd.DataFrame(log).to_csv(os.path.join(RESULT_DIR, "unstructured_pruning_log.csv"), index=False)
    layer_sparsity_table(model).to_csv(os.path.join(RESULT_DIR, "unstructured_layer_sparsity.csv"), index=False)


if __name__ == "__main__":
    main()
