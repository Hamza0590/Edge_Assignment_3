"""
prune_structured.py - Q2.3: structured (filter / channel) pruning with architecture reconstruction.

Why only the *internal* channels of each BasicBlock?
  In ResNet the output channels of every block in a stage are tied together by
  the identity shortcut (out = F(x) + x requires identical channel counts).
  Removing a block-output channel would force removing the same channel from
  every block + shortcut of that stage.  The channel between conv1 and conv2
  inside a BasicBlock, however, is "private": it is produced by conv1/bn1 and
  consumed only by conv2.  Pruning it keeps every tensor dimension valid.

Steps
  1. Importance score of each internal channel c of block b:
        s_c = ||W_conv1[c]||_1 * |gamma_bn1[c]|
     (L1-norm of the filter, scaled by the BN gain that multiplies its output.)
  2. Keep the top (1 - ratio) channels in each block (at least `min_channels`).
  3. Reconstruct a NEW, physically smaller ResNet18(mid_channels=[...]):
        conv1.weight[keep, :, :, :]   bn1.*[keep]   conv2.weight[:, keep, :, :]
     -> no masks and no zero channels; FLOPs and parameters really drop.
  4. Verify dimensions and output equivalence of the copy, fine-tune, export.

Usage:
    python src/prune_structured.py --ratio 0.5 --ft-epochs 10
"""
import argparse
import os

import pandas as pd
import torch

from common import (CKPT, RESULT_DIR, ResNet18, count_params, evaluate, file_size_mb, get_device,
                    get_loaders, load_fp32, set_seed)
from finetune import finetune


def channel_importance(block):
    w = block.conv1.weight.detach()                         # [mid, in, 3, 3]
    l1 = w.abs().sum(dim=(1, 2, 3))
    gamma = block.bn1.weight.detach().abs()
    return l1 * gamma


def select_channels(model, ratio, min_channels=8):
    keep_idx = []
    for blk in model.blocks():
        s = channel_importance(blk)
        n_keep = max(min_channels, int(round(s.numel() * (1 - ratio))))
        keep = torch.argsort(s, descending=True)[:n_keep].sort().values
        keep_idx.append(keep)
    return keep_idx


@torch.no_grad()
def reconstruct(model, keep_idx):
    """Build the smaller architecture and copy the surviving weights into it."""
    new = ResNet18(mid_channels=[len(k) for k in keep_idx])
    old_sd, new_sd = model.state_dict(), new.state_dict()
    # copy everything that is shape-compatible (stem, conv2 outputs, shortcuts, fc ...)
    for k in new_sd:
        if old_sd[k].shape == new_sd[k].shape:
            new_sd[k].copy_(old_sd[k])
    # now the pruned tensors
    for (bi, (ob, nb)), keep in zip(enumerate(zip(model.blocks(), new.blocks())), keep_idx):
        nb.conv1.weight.copy_(ob.conv1.weight[keep])
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(nb.bn1, attr).copy_(getattr(ob.bn1, attr)[keep])
        nb.conv2.weight.copy_(ob.conv2.weight[:, keep])
        nb.bn2.load_state_dict(ob.bn2.state_dict())
    return new.eval()


def check_dimensions(model):
    """Assert every conv's in_channels matches its producer's out_channels."""
    prev = model.conv1.out_channels
    for blk in model.blocks():
        assert blk.conv1.in_channels == prev
        assert blk.conv1.out_channels == blk.bn1.num_features == blk.conv2.in_channels
        assert blk.conv2.out_channels == blk.bn2.num_features
        if blk.downsample is not None:
            assert blk.downsample[0].in_channels == prev and blk.downsample[0].out_channels == blk.conv2.out_channels
        else:
            assert prev == blk.conv2.out_channels
        prev = blk.conv2.out_channels
    assert model.fc.in_features == prev
    model(torch.randn(2, 3, 32, 32))      # real forward pass
    return True


def main():
    ap = argparse.ArgumentParser(description="Structured channel pruning + reconstruction")
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of internal channels removed per block")
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()
    set_seed(11)
    device = get_device()
    loaders = get_loaders(256, device=device)

    base = load_fp32()
    base_acc = evaluate(base.to(device), loaders[1], device)["accuracy"]
    base.cpu()

    keep_idx = select_channels(base, args.ratio)
    pruned = reconstruct(base, keep_idx)
    check_dimensions(pruned)

    # sanity: reconstruction == masking channels to zero in the original model (proves correct copy)
    masked = load_fp32()
    with torch.no_grad():
        for blk, keep in zip(masked.blocks(), keep_idx):
            drop = torch.ones(blk.bn1.num_features, dtype=torch.bool)
            drop[keep] = False
            blk.bn1.weight[drop] = 0
            blk.bn1.bias[drop] = 0
        x = torch.randn(8, 3, 32, 32)
        diff = (masked(x) - pruned(x)).abs().max().item()
    print(f"[structured] reconstruction check: max |masked - reconstructed| = {diff:.2e}")

    rows = []
    for i, (blk_o, blk_n) in enumerate(zip(base.blocks(), pruned.blocks())):
        rows.append(dict(block=i, original_mid_channels=blk_o.conv1.out_channels,
                         pruned_mid_channels=blk_n.conv1.out_channels))
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "structured_channels.csv"), index=False)

    acc_before = evaluate(pruned.to(device), loaders[1], device)["accuracy"]
    tb, _ = count_params(base)
    tp, _ = count_params(pruned)
    print(f"[structured] mid channels: {pruned.mid_channels}")
    print(f"[structured] params {tb:,} -> {tp:,} ({100 * (1 - tp / tb):.1f}% fewer) | acc {base_acc:.2f}% -> "
          f"{acc_before:.2f}% before fine-tune")

    finetune(pruned, args.ft_epochs, args.lr, log_prefix="[structured FT] ", loaders=loaders)
    pruned.cpu().eval()
    acc_after = evaluate(pruned.to(device), loaders[1], device)["accuracy"]
    pruned.cpu()
    torch.save({"mid_channels": pruned.mid_channels, "state_dict": pruned.state_dict()}, CKPT["structured"])
    print(f"[structured] after fine-tune {acc_after:.2f}% -> saved {CKPT['structured']} "
          f"({file_size_mb(CKPT['structured']):.2f} MB vs {file_size_mb(CKPT['fp32']):.2f} MB)")
    pd.DataFrame([dict(baseline_acc=base_acc, ratio=args.ratio, params_before=tb, params_after=tp,
                       acc_after_prune=acc_before, acc_after_finetune=acc_after)]).to_csv(
        os.path.join(RESULT_DIR, "structured_pruning_log.csv"), index=False)


if __name__ == "__main__":
    main()
