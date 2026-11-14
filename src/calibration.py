"""
calibration.py - Calibration pipeline for Post-Training Static Quantization.

After `torch.ao.quantization.prepare()` the model contains *observers*
(HistogramObserver for activations, PerChannelMinMaxObserver for weights).
Feeding representative data through the model lets the observers record the
activation ranges, from which scale / zero-point are computed at convert time.

We use a fixed random subset of the TRAINING set (default 2,000 images,
no augmentation) - the test set is never touched during calibration.
"""
import time

import torch

from common import get_calibration_loader


@torch.no_grad()
def calibrate(prepared_model, num_images=2000, batch_size=100, seed=0, verbose=True):
    """Run the calibration subset through a prepared (observed) model on CPU."""
    loader = get_calibration_loader(num_images, batch_size, seed)
    prepared_model.eval()
    t0 = time.time()
    for i, (xb, _) in enumerate(loader, 1):
        prepared_model(xb)
        if verbose and (i % 5 == 0 or i == len(loader)):
            print(f"[calibration] batch {i}/{len(loader)} ({i * batch_size} images)", flush=True)
    if verbose:
        print(f"[calibration] done: {num_images} images in {time.time() - t0:.1f}s")
    return prepared_model


if __name__ == "__main__":
    # Stand-alone demo: show the observer ranges collected for the first layer
    import copy
    import torch.ao.quantization as tq
    from common import QUANT_ENGINE, load_fp32

    torch.backends.quantized.engine = QUANT_ENGINE
    m = copy.deepcopy(load_fp32())
    m.fuse_model()
    m.qconfig = tq.get_default_qconfig(QUANT_ENGINE)
    tq.prepare(m, inplace=True)
    calibrate(m, num_images=500)
    obs = m.quant.activation_post_process
    print("Input observer range:", float(obs.min_val), float(obs.max_val))
    print("Input qparams (scale, zero_point):", obs.calculate_qparams())
