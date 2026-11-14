"""
ptq.py - Q1.2: Post-Training Static Quantization (FP32 -> INT8).

Pipeline
  1. Load trained FP32 checkpoint
  2. Fuse Conv+BN+ReLU / Conv+BN / downsample Conv+BN
  3. Configure the quantization backend + qconfig (onednn)
  4. prepare()  -> insert observers
  5. calibrate() on a training-data subset   (calibration.py)
  6. convert()  -> real INT8 kernels (quantized::conv2d, quantized::add ...)
  7. Save: TorchScript (self-contained, executable) + state_dict
  8. Quick deployment check: evaluate on the CIFAR-10 test set (CPU)

Usage:
    python src/ptq.py --calib-images 2000
"""
import argparse
import copy

import torch
import torch.ao.quantization as tq

from calibration import calibrate
from common import (CKPT, CKPT_EXTRA, QUANT_ENGINE, evaluate, file_size_mb, get_loaders,
                    load_fp32, set_seed)


def build_ptq_model(fp32_model, calib_images=2000, calib_batch=100):
    model = copy.deepcopy(fp32_model).cpu().eval()

    # 1) fusion
    model.fuse_model(is_qat=False)

    # 2) backend + qconfig
    torch.backends.quantized.engine = QUANT_ENGINE
    model.qconfig = tq.get_default_qconfig(QUANT_ENGINE)
    print(f"[PTQ] backend = {QUANT_ENGINE}")
    print(f"[PTQ] qconfig = {model.qconfig}")

    # 3) observers
    tq.prepare(model, inplace=True)

    # 4) calibration on training subset
    calibrate(model, num_images=calib_images, batch_size=calib_batch)

    # 5) convert to INT8
    tq.convert(model, inplace=True)
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description="Post-Training Static Quantization")
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--calib-batch", type=int, default=100)
    args = ap.parse_args()
    set_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    fp32 = load_fp32()
    q = build_ptq_model(fp32, args.calib_images, args.calib_batch)
    print(q.layer1[0])   # show INT8 modules

    # Save: TorchScript (executable without the Python class) + plain state_dict
    scripted = torch.jit.script(q)
    torch.jit.save(scripted, CKPT["ptq"])
    torch.save(q.state_dict(), CKPT_EXTRA["ptq_state"])
    print(f"[PTQ] saved {CKPT['ptq']} ({file_size_mb(CKPT['ptq']):.2f} MB)  "
          f"vs FP32 {file_size_mb(CKPT['fp32']):.2f} MB")

    # Deploy check: reload the saved file and run the test set
    _, test_loader = get_loaders(500, device="cpu", train_aug=False)
    deployed = torch.jit.load(CKPT["ptq"])
    acc = evaluate(deployed, test_loader)["accuracy"]
    print(f"[PTQ] INT8 test accuracy (reloaded TorchScript) = {acc:.2f}%")


if __name__ == "__main__":
    main()
