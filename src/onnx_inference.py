"""
onnx_inference.py - CPU inference + benchmark with ONNX Runtime (CPUExecutionProvider).

No PyTorch model code is needed at deployment time: only numpy, PIL and onnxruntime.

Examples
    python src/onnx_inference.py --benchmark                 # all 5 ONNX models on full test set
    python src/onnx_inference.py --image samples/*.png       # classify image files
"""
import argparse
import glob
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(ROOT, "models", "onnx")
RESULT_DIR = os.path.join(ROOT, "results")
NAMES = ["fp32", "ptq", "qat", "unstructured", "structured"]
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(uint8_nhwc):
    x = uint8_nhwc.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    return ((x - MEAN) / STD).astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def session(name, threads=0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(os.path.join(ONNX_DIR, f"{name}.onnx"), so, providers=["CPUExecutionProvider"])


def benchmark(threads, lat_iters):
    d = np.load(os.path.join(ROOT, "data", "cifar10_test.npz"))
    x_all, y_all = preprocess(d["x"]), d["y"]
    rows, preds = [], {}
    for n in NAMES:
        s = session(n, threads)
        t0 = time.perf_counter()
        out = np.concatenate([s.run(None, {"input": x_all[i:i + 256]})[0] for i in range(0, len(x_all), 256)])
        tot = time.perf_counter() - t0
        preds[n] = out.argmax(1)
        one = x_all[:1]
        for _ in range(10):
            s.run(None, {"input": one})
        lat = []
        for _ in range(lat_iters):
            t = time.perf_counter()
            s.run(None, {"input": one})
            lat.append((time.perf_counter() - t) * 1000)
        rows.append(dict(model=n, onnx_file=f"{n}.onnx",
                         size_mb=round(os.path.getsize(os.path.join(ONNX_DIR, f"{n}.onnx")) / 1024 ** 2, 3),
                         accuracy_pct=round(100 * float((preds[n] == y_all).mean()), 2),
                         latency_bs1_ms=round(float(np.mean(lat)), 3),
                         latency_bs1_p95_ms=round(float(np.percentile(lat, 95)), 3),
                         throughput_img_s=round(len(x_all) / tot, 1)))
        print(f"  {n:<13} acc {rows[-1]['accuracy_pct']:6.2f}% | {rows[-1]['size_mb']:6.2f} MB | "
              f"bs1 {rows[-1]['latency_bs1_ms']:6.3f} ms | {rows[-1]['throughput_img_s']:8.1f} img/s", flush=True)
    df = pd.DataFrame(rows)
    df["agreement_with_fp32_pct"] = [round(100 * float((preds[n] == preds["fp32"]).mean()), 2) for n in NAMES]
    df.to_csv(os.path.join(RESULT_DIR, "onnx_cpu_benchmark.csv"), index=False)
    print(f"-> {os.path.join(RESULT_DIR, 'onnx_cpu_benchmark.csv')}")


def classify(files, threads):
    sessions = {n: session(n, threads) for n in NAMES}
    for f in files:
        img = np.asarray(Image.open(f).convert("RGB").resize((32, 32)))[None]
        x = preprocess(img)
        print(f"\nImage: {os.path.basename(f)}")
        for n, s in sessions.items():
            p = softmax(s.run(None, {"input": x})[0])[0]
            k = int(p.argmax())
            print(f"  {n:<13} {CLASSES[k]:<11} {p[k]:.3f}")


def main():
    ap = argparse.ArgumentParser(description="ONNX Runtime CPU inference")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--image", nargs="*")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--latency-iters", type=int, default=200)
    args = ap.parse_args()
    print(f"onnxruntime {ort.__version__} | providers {ort.get_available_providers()}")
    if args.image:
        classify(sorted({f for p in args.image for f in glob.glob(p)}), args.threads)
    if args.benchmark or not args.image:
        benchmark(args.threads, args.latency_iters)


if __name__ == "__main__":
    main()
