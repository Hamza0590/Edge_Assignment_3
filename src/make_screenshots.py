"""
make_screenshots.py - Render the console logs of every pipeline step as terminal-style
PNG "screenshots" for the technical report (report/screenshots/).

Usage:
    python src/make_screenshots.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
OUT_DIR = os.path.join(ROOT, "report", "screenshots")

# noise produced by PyTorch deprecation warnings - not part of our program output
NOISE = re.compile(r"(Warning|deprecat|For migrations of users|^\s*\d\. (Eager|FX|pt2e)|see https://github)", re.I)


def read_text(path):
    """Logs written by Windows PowerShell 5.1 (Tee-Object / *>) are UTF-16LE with BOM."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def clean(lines, max_lines=60):
    keep = [l.rstrip("\n") for l in lines if l.strip() and not NOISE.search(l)]
    if len(keep) > max_lines:
        keep = keep[:max_lines // 2] + ["   ...  (truncated, see full log in logs/)  ..."] + keep[-max_lines // 2:]
    return keep


def render(title, lines, path):
    width = min(170, max(60, max((len(l) for l in lines), default=60)))
    h = 0.6 + 0.19 * (len(lines) + 2)
    fig = plt.figure(figsize=(width * 0.08 + 0.4, h))
    fig.patch.set_facecolor("#1e1e1e")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#1e1e1e")
    ax.text(0.01, 1 - 0.25 / h, f"PS D:\\...\\Assignment\\3> {title}", color="#4ec9b0", family="monospace",
            fontsize=9, va="top", transform=ax.transAxes)
    ax.text(0.01, 1 - 0.55 / h, "\n".join(l[:width] for l in lines), color="#d4d4d4", family="monospace",
            fontsize=9, va="top", transform=ax.transAxes, linespacing=1.45)
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


CMDS = {
    "00": "python src/prepare_data.py", "01": "python src/train_baseline.py --epochs 30",
    "02": "python src/ptq.py --calib-images 2000", "03": "python src/qat.py --epochs 5",
    "04": "python src/prune_unstructured.py --target-sparsity 0.8 --iterations 4",
    "05": "python src/prune_structured.py --ratio 0.5 --ft-epochs 10", "06": "python src/export_onnx.py",
    "07": "python src/benchmark.py --suite all", "08": "python src/onnx_inference.py --benchmark",
    "09": "python src/inference.py --model all --testset 10 --image samples/cat_0.png",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        base = os.path.splitext(os.path.basename(log))[0]
        if not base[:2].isdigit():          # skip pipeline_master.log (duplicate of the step logs)
            continue
        lines = clean(read_text(log).splitlines())
        out = os.path.join(OUT_DIR, base + ".png")
        render(CMDS.get(base[:2], base), lines, out)
        print("screenshot ->", out)


if __name__ == "__main__":
    main()
