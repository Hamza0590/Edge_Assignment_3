"""
benchmark.py - Automated benchmarking of all models on the full CIFAR-10 test set (CPU).

Q1 suite (quantization): fp32, ptq, qat
Q2 suite (pruning)     : fp32 (original), unstructured, structured

Generated files (results/)
  q1_quantization_benchmark.csv  summary: accuracy, size, latency, throughput, speed-up
  q1_test_predictions.csv        per-image predictions + confidence of fp32/ptq/qat
  q2_pruning_benchmark.csv       summary: accuracy, params, sparsity, MACs, size, latency
  q2_test_predictions.csv        per-image predictions + confidence of fp32/unstructured/structured
  all_models_benchmark.csv       the five models side-by-side
  benchmark_plots.png            accuracy / size / latency bar charts

Usage:
    python src/benchmark.py --suite all
"""
import argparse
import gzip
import io
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (CKPT, CKPT_EXTRA, CIFAR10_CLASSES, QUANT_ENGINE, RESULT_DIR, count_params, evaluate,
                    file_size_mb, get_loaders, load_model, measure_latency)

LABELS = {"fp32": "FP32 baseline", "ptq": "PTQ INT8", "qat": "QAT INT8",
          "unstructured": "Unstructured-pruned", "structured": "Structured-pruned"}


def gzip_size_mb(path):
    with open(path, "rb") as f:
        return len(gzip.compress(f.read(), 6)) / 1024 ** 2


def count_macs(model):
    """Dense and effective (non-zero weight) multiply-accumulates for one 32x32 image."""
    dense, eff = [0], [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            per_out = out.shape[2] * out.shape[3]
            dense[0] += m.weight.numel() * per_out
            eff[0] += int((m.weight != 0).sum()) * per_out
        else:
            dense[0] += m.weight.numel()
            eff[0] += int((m.weight != 0).sum())

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32))
    for h in hs:
        h.remove()
    return dense[0], eff[0]


def bench_one(name, test_loader, lat_iters):
    model = load_model(name)
    t0 = time.perf_counter()
    res = evaluate(model, test_loader, return_preds=True)
    total_s = time.perf_counter() - t0
    lat_mean, lat_p95 = measure_latency(model, batch_size=1, n_iters=lat_iters)
    path = CKPT[name]
    row = dict(model=LABELS[name], key=name, accuracy_pct=round(res["accuracy"], 2),
               checkpoint=os.path.basename(path), size_mb=round(file_size_mb(path), 3),
               gzip_size_mb=round(gzip_size_mb(path), 3),
               latency_bs1_ms=round(lat_mean, 3), latency_bs1_p95_ms=round(lat_p95, 3),
               throughput_img_s=round(len(res["preds"]) / total_s, 1),
               testset_time_s=round(total_s, 2))
    if isinstance(model, nn.Module) and name not in ("ptq", "qat"):
        tot, nz = count_params(model)
        dm, em = count_macs(model)
        row.update(params=tot, nonzero_params=nz, sparsity_pct=round(100 * (1 - nz / tot), 2),
                   macs_m=round(dm / 1e6, 1), effective_macs_m=round(em / 1e6, 1))
    else:   # INT8 TorchScript: parameters live in packed params, count from FP32 reference
        row.update(params=count_params(load_model("fp32"))[0], weight_dtype="int8")
    if name == "unstructured" and os.path.exists(CKPT_EXTRA["unstructured_sparse"]):
        row["sparse_storage_mb"] = round(file_size_mb(CKPT_EXTRA["unstructured_sparse"]), 3)
    print(f"  {LABELS[name]:<20} acc {row['accuracy_pct']:6.2f}% | size {row['size_mb']:6.2f} MB | "
          f"bs1 {row['latency_bs1_ms']:6.2f} ms | {row['throughput_img_s']:8.1f} img/s", flush=True)
    return row, res


def predictions_frame(results, labels):
    df = pd.DataFrame({"image_idx": np.arange(len(labels)), "true_label": labels.numpy(),
                       "true_class": [CIFAR10_CLASSES[i] for i in labels]})
    for name, r in results.items():
        df[f"pred_{name}"] = r["preds"].numpy()
        df[f"pred_class_{name}"] = [CIFAR10_CLASSES[i] for i in r["preds"]]
        df[f"conf_{name}"] = r["confs"].numpy().round(4)
        df[f"correct_{name}"] = (r["preds"] == labels).numpy().astype(int)
    return df


def run_suite(names, tag, test_loader, lat_iters, cache):
    print(f"\n=== {tag} ===")
    rows, results = [], {}
    for n in names:
        if n not in cache:
            cache[n] = bench_one(n, test_loader, lat_iters)
        row, res = cache[n]
        rows.append(dict(row))
        results[n] = res
    df = pd.DataFrame(rows)
    base = df.loc[df.key == "fp32"].iloc[0]
    df["accuracy_drop_pct"] = (base.accuracy_pct - df.accuracy_pct).round(2)
    df["compression_x"] = (base.size_mb / df.size_mb).round(2)
    df["speedup_bs1_x"] = (base.latency_bs1_ms / df.latency_bs1_ms).round(2)
    df["speedup_throughput_x"] = (df.throughput_img_s / base.throughput_img_s).round(2)
    labels = results["fp32"]["labels"]
    for n in names:
        df.loc[df.key == n, "agreement_with_fp32_pct"] = round(
            100 * (results[n]["preds"] == results["fp32"]["preds"]).float().mean().item(), 2)
    return df, predictions_frame(results, labels)


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(df)]
    for a, col, title in zip(ax, ["accuracy_pct", "size_mb", "latency_bs1_ms"],
                             ["Top-1 accuracy (%)", "Checkpoint size (MB)", "CPU latency, batch=1 (ms)"]):
        bars = a.bar(df.model, df[col], color=colors)
        a.set_title(title)
        a.tick_params(axis="x", rotation=25)
        for b, v in zip(bars, df[col]):
            a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        if col == "accuracy_pct":
            a.set_ylim(max(0, df[col].min() - 10), 100)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"plot -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark FP32 / INT8 / pruned models on CIFAR-10 test set")
    ap.add_argument("--suite", choices=["quant", "pruning", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--latency-iters", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = QUANT_ENGINE
    torch.set_grad_enabled(False)
    print(f"CPU: {platform.processor()} | threads: {torch.get_num_threads()} | torch {torch.__version__} "
          f"| quant engine {QUANT_ENGINE}")

    _, test_loader = get_loaders(args.batch_size, device="cpu", train_aug=False)
    cache, frames = {}, []
    if args.suite in ("quant", "all"):
        s, p = run_suite(["fp32", "ptq", "qat"], "Q1: quantization", test_loader, args.latency_iters, cache)
        s.to_csv(os.path.join(RESULT_DIR, "q1_quantization_benchmark.csv"), index=False)
        p.to_csv(os.path.join(RESULT_DIR, "q1_test_predictions.csv"), index=False)
        frames.append(s)
    if args.suite in ("pruning", "all"):
        s, p = run_suite(["fp32", "unstructured", "structured"], "Q2: pruning", test_loader, args.latency_iters, cache)
        s.to_csv(os.path.join(RESULT_DIR, "q2_pruning_benchmark.csv"), index=False)
        p.to_csv(os.path.join(RESULT_DIR, "q2_test_predictions.csv"), index=False)
        frames.append(s)
    if args.suite == "all":
        allm, _ = run_suite(["fp32", "ptq", "qat", "unstructured", "structured"], "All five models",
                            test_loader, args.latency_iters, cache)
        allm.to_csv(os.path.join(RESULT_DIR, "all_models_benchmark.csv"), index=False)
        plot(allm, os.path.join(RESULT_DIR, "benchmark_plots.png"))
        cols = ["model", "accuracy_pct", "size_mb", "compression_x", "latency_bs1_ms", "speedup_bs1_x",
                "throughput_img_s", "agreement_with_fp32_pct"]
        print("\n" + allm[cols].to_string(index=False))
    print(f"\nCSV files written to {RESULT_DIR}")


if __name__ == "__main__":
    main()
