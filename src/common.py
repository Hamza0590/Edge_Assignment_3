"""
common.py - Shared utilities for Assignment 3 (Edge Computing & Intelligent Systems).

Contents
--------
* Paths / reproducibility helpers
* CIFAR-10 data loading (whole dataset kept as tensors -> very fast, no
  Windows multiprocessing DataLoader problems)
* A quantization-ready ResNet-18 for CIFAR-10 (QuantStub / DeQuantStub,
  FloatFunctional for the residual add, separate ReLU modules so they can be
  fused).  The same class is used for pruning; `mid_channels` lets us build
  the *physically smaller* structured-pruned architecture.
* Training / evaluation / latency-measurement helpers
* Model-loading helpers used by every inference / benchmark script
"""

import os
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.ao.quantization import QuantStub, DeQuantStub

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
ONNX_DIR = os.path.join(MODEL_DIR, "onnx")
RESULT_DIR = os.path.join(ROOT, "results")
for _d in (DATA_DIR, MODEL_DIR, ONNX_DIR, RESULT_DIR):
    os.makedirs(_d, exist_ok=True)

# Checkpoint file names (single source of truth for every script)
CKPT = {
    "fp32": os.path.join(MODEL_DIR, "resnet18_fp32.pth"),
    "ptq": os.path.join(MODEL_DIR, "resnet18_ptq_int8.pt"),            # TorchScript
    "qat": os.path.join(MODEL_DIR, "resnet18_qat_int8.pt"),            # TorchScript
    "unstructured": os.path.join(MODEL_DIR, "resnet18_unstructured_pruned.pth"),
    "structured": os.path.join(MODEL_DIR, "resnet18_structured_pruned.pth"),
}
# Extra artefacts
CKPT_EXTRA = {
    "ptq_state": os.path.join(MODEL_DIR, "resnet18_ptq_int8_state_dict.pth"),
    "qat_state": os.path.join(MODEL_DIR, "resnet18_qat_int8_state_dict.pth"),
    "qat_fakequant": os.path.join(MODEL_DIR, "resnet18_qat_fakequant.pth"),
    "unstructured_sparse": os.path.join(MODEL_DIR, "resnet18_unstructured_pruned_sparse.pth"),
}

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

QUANT_ENGINE = "onednn"   # only INT8 engine available in this PyTorch build (see DEMO.md)


def set_seed(seed: int = 42):
    """Make runs reproducible (as far as cuDNN allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True     # autotune conv kernels (fixed 32x32 input)


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_SPLIT_CACHE = {}


def _load_split(train: bool):
    """Load a CIFAR-10 split from the .npz cache created by prepare_data.py."""
    split = "train" if train else "test"
    if split not in _SPLIT_CACHE:
        path = os.path.join(DATA_DIR, f"cifar10_{split}.npz")
        if not os.path.exists(path):
            import prepare_data
            prepare_data.main()
        d = np.load(path)
        x = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).contiguous()   # uint8 N,3,32,32
        y = torch.from_numpy(d["y"]).long()
        _SPLIT_CACHE[split] = (x, y)
    return _SPLIT_CACHE[split]


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [N,3,32,32] -> normalised float tensor."""
    mean = torch.tensor(CIFAR10_MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def augment(x_uint8: torch.Tensor) -> torch.Tensor:
    """Random-crop (pad 4) + horizontal flip, done on whole batch on-device."""
    n = x_uint8.size(0)
    padded = F.pad(x_uint8, (4, 4, 4, 4), mode="constant", value=0)
    ix = torch.randint(0, 9, (n,), device=x_uint8.device)
    iy = torch.randint(0, 9, (n,), device=x_uint8.device)
    # gather crops with advanced indexing
    ar = torch.arange(32, device=x_uint8.device)
    rows = (iy.view(n, 1) + ar.view(1, 32))            # n,32
    cols = (ix.view(n, 1) + ar.view(1, 32))            # n,32
    out = padded[torch.arange(n, device=x_uint8.device).view(n, 1, 1, 1),
                 torch.arange(3, device=x_uint8.device).view(1, 3, 1, 1),
                 rows.view(n, 1, 32, 1), cols.view(n, 1, 1, 32)]
    flip = torch.rand(n, device=x_uint8.device) < 0.5
    out[flip] = out[flip].flip(3)
    return out


class TensorLoader:
    """Minimal, fast replacement for DataLoader over in-memory tensors."""

    def __init__(self, x, y, batch_size=256, shuffle=False, train_aug=False, device="cpu"):
        self.x, self.y = x.to(device), y.to(device)
        self.bs, self.shuffle, self.aug, self.device = batch_size, shuffle, train_aug, device

    def __len__(self):
        return (len(self.x) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.x)
        idx = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        for i in range(0, n, self.bs):
            b = idx[i:i + self.bs]
            xb = self.x[b]
            if self.aug:
                xb = augment(xb)
            yield normalize(xb), self.y[b]


def get_loaders(batch_size=256, device="cpu", train_aug=True):
    xtr, ytr = _load_split(True)
    xte, yte = _load_split(False)
    train = TensorLoader(xtr, ytr, batch_size, shuffle=True, train_aug=train_aug, device=device)
    test = TensorLoader(xte, yte, batch_size, shuffle=False, train_aug=False, device=device)
    return train, test


def get_calibration_loader(num_images=2000, batch_size=100, seed=0):
    """Random subset of the TRAINING set (no augmentation) used for PTQ calibration."""
    xtr, ytr = _load_split(True)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(xtr), generator=g)[:num_images]
    return TensorLoader(xtr[idx], ytr[idx], batch_size, shuffle=False, train_aug=False, device="cpu")


# ---------------------------------------------------------------------------
# Model: quantization-ready, prunable ResNet-18 for 32x32 inputs
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mid_planes=None):
        super().__init__()
        mid = mid_planes or planes          # mid_planes < planes after structured pruning
        self.conv1 = nn.Conv2d(in_planes, mid, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))
        # FloatFunctional makes the residual addition quantizable
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, identity)
        return self.relu2(out)


class ResNet18(nn.Module):
    """ResNet-18 (CIFAR variant: 3x3 stem, no max-pool).

    mid_channels: optional list of 8 ints - number of internal channels of
    each BasicBlock (used to rebuild the structured-pruned network).
    """
    STAGES = [(64, 1), (128, 2), (256, 2), (512, 2)]

    def __init__(self, num_classes=10, mid_channels=None):
        super().__init__()
        self.mid_channels = list(mid_channels) if mid_channels else [p for p, _ in self.STAGES for _ in range(2)]
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        in_planes, k = 64, 0
        layers = []
        for planes, stride in self.STAGES:
            blocks = []
            for s in (stride, 1):
                blocks.append(BasicBlock(in_planes, planes, s, self.mid_channels[k]))
                in_planes, k = planes, k + 1
            layers.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def blocks(self):
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            yield from layer

    # ---- fusion (Conv+BN(+ReLU)) for quantization -------------------------
    def fuse_model(self, is_qat=False):
        fuse = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for blk in self.blocks():
            fuse(blk, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
            if blk.downsample is not None:
                fuse(blk.downsample, [["0", "1"]], inplace=True)


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return total, nonzero


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
STOP_FILE = os.path.join(ROOT, "STOP")


def train_epochs(model, train_loader, test_loader, epochs, lr, device, weight_decay=5e-4,
                 schedule="onecycle", log_prefix="", after_step=None, use_amp=True,
                 resume_path=None):
    """Generic SGD training loop (used by baseline training and every fine-tuning stage).

    after_step : optional callback executed after every optimizer step
                 (not needed for torch.nn.utils.prune - masks are re-applied automatically).
    resume_path: if given, a full training state (model/optimizer/scheduler/scaler/epoch)
                 is saved there after every epoch and training resumes from it if it exists.
    Pause      : create a file named STOP in the project root -> training stops cleanly after
                 the current epoch (state saved); rerun the same command to resume.
    """
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    steps = epochs * len(train_loader)
    if schedule == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.25)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    history, start_ep = [], 1
    if resume_path and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        history, start_ep = st["history"], st["epoch"] + 1
        torch.set_rng_state(st["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(st["rng_cuda"].cpu())
        print(f"{log_prefix}resumed from {resume_path} at epoch {start_ep}/{epochs}", flush=True)
    for ep in range(start_ep, epochs + 1):
        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if after_step:
                after_step()
            run_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            seen += xb.size(0)
        test_acc = evaluate(model, test_loader, device)["accuracy"]
        rec = dict(epoch=ep, loss=run_loss / seen, train_acc=100 * correct / seen, test_acc=test_acc,
                   time_s=time.time() - t0)
        history.append(rec)
        print(f"{log_prefix}epoch {ep:3d}/{epochs} | loss {rec['loss']:.4f} | train {rec['train_acc']:.2f}% "
              f"| test {test_acc:.2f}% | {rec['time_s']:.1f}s", flush=True)
        if resume_path:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), epoch=ep, history=history,
                            rng_cpu=torch.get_rng_state(),
                            rng_cuda=torch.cuda.get_rng_state() if device.type == "cuda" else None),
                       resume_path)
        if os.path.exists(STOP_FILE) and ep < epochs:
            os.remove(STOP_FILE)
            print(f"{log_prefix}STOP file found -> paused after epoch {ep}. "
                  f"Run the same command again to resume.", flush=True)
            raise SystemExit(3)
    return history


@torch.no_grad()
def evaluate(model, loader, device=torch.device("cpu"), return_preds=False):
    """Top-1 accuracy (+ optional per-sample predictions/confidences)."""
    if isinstance(model, nn.Module):
        model.eval()
    preds, confs, labels = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).float()
        p = out.softmax(1)
        c, k = p.max(1)
        preds.append(k.cpu()); confs.append(c.cpu()); labels.append(yb.cpu())
    preds, confs, labels = torch.cat(preds), torch.cat(confs), torch.cat(labels)
    res = {"accuracy": 100.0 * (preds == labels).float().mean().item()}
    if return_preds:
        res.update(preds=preds, confs=confs, labels=labels)
    return res


@torch.no_grad()
def measure_latency(model, batch_size=1, n_warmup=10, n_iters=100, device=torch.device("cpu")):
    """Mean / p95 latency (ms per batch) on random input of the CIFAR shape."""
    if isinstance(model, nn.Module):
        model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    for _ in range(n_warmup):
        model(x)
    times = []
    for _ in range(n_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    return float(times.mean()), float(np.percentile(times, 95))


# ---------------------------------------------------------------------------
# Loading every deployed model (shared by inference.py / benchmark scripts)
# ---------------------------------------------------------------------------
MODEL_NAMES = ["fp32", "ptq", "qat", "unstructured", "structured"]


def load_fp32(path=None):
    m = ResNet18()
    m.load_state_dict(torch.load(path or CKPT["fp32"], map_location="cpu"))
    return m.eval()


def load_structured(path=None):
    ck = torch.load(path or CKPT["structured"], map_location="cpu")
    m = ResNet18(mid_channels=ck["mid_channels"])          # physically smaller architecture
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def load_model(name: str):
    """Return an eval-mode, CPU, ready-to-run model for one of MODEL_NAMES."""
    if name == "fp32":
        return load_fp32()
    if name in ("ptq", "qat"):
        torch.backends.quantized.engine = QUANT_ENGINE
        return torch.jit.load(CKPT[name], map_location="cpu").eval()   # self-contained INT8 TorchScript
    if name == "unstructured":
        return load_fp32(CKPT["unstructured"])
    if name == "structured":
        return load_structured()
    raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")
