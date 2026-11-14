"""
export_onnx.py - Deliverable 8: export all five models to ONNX for CPU deployment.

  fp32, unstructured, structured : direct torch.onnx export of the float model
  qat  : the fine-tuned fake-quant QAT model is exported -> FakeQuantize nodes become
         QuantizeLinear/DequantizeLinear (QDQ) pairs that ONNX Runtime fuses into
         INT8 kernels.  (PyTorch's *converted* eager INT8 modules (quantized::conv2d
         with packed params) cannot be exported to ONNX - see DEMO.md "Issues".)
  ptq  : ONNX Runtime static quantization of fp32.onnx with the SAME calibration
         subset (2,000 training images), QDQ format, per-channel symmetric INT8 weights.

Output: models/onnx/{fp32,ptq,qat,unstructured,structured}.onnx

Usage:
    python src/export_onnx.py
"""
import argparse
import os

import numpy as np
import onnx
import torch
import torch.ao.quantization as tq

from common import (CKPT_EXTRA, ONNX_DIR, QUANT_ENGINE, ResNet18, file_size_mb, get_calibration_loader,
                    load_model)

OPSET = 17


def onnx_path(name):
    return os.path.join(ONNX_DIR, f"{name}.onnx")


def export_float(model, path, opset=OPSET):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    # Legacy TorchScript-based exporter: stable for eager models and fake-quant ops.
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    print(f"[onnx] {os.path.basename(path):<18} {file_size_mb(path):6.2f} MB")


class ExportableFakeQuant(torch.nn.Module):
    """Replacement for FusedMovingAvgObsFakeQuantize, whose op
    (aten::fused_moving_avg_obs_fake_quant) has no ONNX symbolic.  Uses the frozen,
    learned scale / zero-point with the standard fake-quant ops -> ONNX Q/DQ nodes."""

    def __init__(self, fq):
        super().__init__()
        scale, zp = fq.scale.detach(), fq.zero_point.detach()     # the frozen qparams actually used
        self.per_channel = fq.is_per_channel
        self.ch_axis = fq.ch_axis
        self.qmin, self.qmax = fq.quant_min, fq.quant_max
        self.register_buffer("scale", scale.float().reshape(-1))
        self.register_buffer("zero_point", zp.to(torch.int32).reshape(-1))

    def forward(self, x):
        if self.per_channel:
            return torch.fake_quantize_per_channel_affine(x, self.scale, self.zero_point, self.ch_axis,
                                                          self.qmin, self.qmax)
        return torch.fake_quantize_per_tensor_affine(x, float(self.scale[0]), int(self.zero_point[0]),
                                                     self.qmin, self.qmax)


class FoldedQConv(torch.nn.Module):
    """QAT Conv+BN(+ReLU) with BN folded ONCE at export time.

    A QAT ConvBn2d re-computes w * gamma / sqrt(var + eps) on every forward pass;
    exported as-is, ONNX Runtime sees float weights flowing through Mul/Div before the
    Q/DQ pair and cannot use INT8 conv kernels (our first qat.onnx was 42.8 MB and 3x
    slower than FP32).  Folding first gives a constant weight -> QDQ -> INT8 Conv."""

    def __init__(self, qat_mod):
        super().__init__()
        with torch.no_grad():
            std = torch.sqrt(qat_mod.bn.running_var + qat_mod.bn.eps)
            factor = qat_mod.bn.weight / std
            w = qat_mod.weight * factor.reshape(-1, 1, 1, 1)
            b0 = qat_mod.bias if qat_mod.bias is not None else torch.zeros_like(std)
            b = (b0 - qat_mod.bn.running_mean) * factor + qat_mod.bn.bias
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.bias = torch.nn.Parameter(b, requires_grad=False)
        self.stride, self.padding = qat_mod.stride, qat_mod.padding
        self.relu = isinstance(qat_mod, torch.ao.nn.intrinsic.qat.ConvBnReLU2d)
        self.wfq = ExportableFakeQuant(qat_mod.weight_fake_quant)
        self.afq = ExportableFakeQuant(qat_mod.activation_post_process)

    def forward(self, x):
        y = torch.nn.functional.conv2d(x, self.wfq(self.weight), self.bias, self.stride, self.padding)
        if self.relu:
            y = torch.relu(y)
        return self.afq(y)


def make_fakequant_exportable(model):
    nniqat = torch.ao.nn.intrinsic.qat
    # In eager-mode INT8, relu2 (after the residual add) runs directly on the quantized
    # tensor and inherits the add's scale/zero-point, so it has no observer of its own.
    # Exported naively, the Relu output is float and the NEXT conv cannot be fused into
    # QLinearConv.  Re-quantizing with the add's qparams is exactly what INT8 PyTorch does.
    for blk in model.blocks():
        blk.relu2 = torch.nn.Sequential(torch.nn.ReLU(),
                                        ExportableFakeQuant(blk.skip_add.activation_post_process))
    for mod in list(model.modules()):
        for attr, child in list(mod.named_children()):
            if isinstance(child, (nniqat.ConvBn2d, nniqat.ConvBnReLU2d)):
                setattr(mod, attr, FoldedQConv(child))
    for mod in list(model.modules()):
        for attr, child in list(mod.named_children()):
            if isinstance(child, tq.FakeQuantizeBase):
                setattr(mod, attr, ExportableFakeQuant(child))
    return model


def export_qat():
    from qat import prepare_qat_model
    torch.backends.quantized.engine = QUANT_ENGINE
    m = prepare_qat_model(ResNet18())
    m.load_state_dict(torch.load(CKPT_EXTRA["qat_fakequant"], map_location="cpu"))
    m.apply(tq.disable_observer)
    m.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
    m.eval()
    with torch.no_grad():
        x = torch.randn(4, 3, 32, 32)
        ref = m(x)
        make_fakequant_exportable(m)
        print(f"[onnx] QAT fake-quant swap check: max diff {(m(x) - ref).abs().max():.2e}")
    export_float(m, onnx_path("qat"))
    fold_weight_quantize(onnx_path("qat"))


def fold_weight_quantize(path):
    """Replace QuantizeLinear(constant FP32 weight) by a pre-computed INT8 initializer.

    torch.onnx exports weight fake-quant as  W_fp32 -> QuantizeLinear -> DequantizeLinear.
    ONNX Runtime does not constant-fold that QuantizeLinear, so the file keeps FP32
    weights (no size saving) and quantizes them on every run.  We do it once offline,
    giving the same  W_int8 -> DequantizeLinear  layout ORT's own quantizer produces."""
    from onnx import numpy_helper
    model = onnx.load(path)
    g = model.graph
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    for n in g.node:                      # exporter de-duplicates shared constants via Identity
        if n.op_type == "Identity" and n.input[0] in consts:
            consts[n.output[0]] = consts[n.input[0]]
    new_inits, remove, folded = [], [], 0
    for n in g.node:
        if n.op_type == "QuantizeLinear" and n.input[0] in consts and consts[n.input[0]].ndim > 1:
            w, s, zp = consts[n.input[0]], consts[n.input[1]], consts[n.input[2]]
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            shape = [1] * w.ndim
            if s.ndim == 1 and s.size > 1:
                shape[axis] = -1
            info = np.iinfo(zp.dtype)
            q = np.clip(np.rint(w / s.reshape(shape)) + zp.reshape(shape).astype(np.int32), info.min, info.max)
            new_inits.append(numpy_helper.from_array(q.astype(zp.dtype), n.output[0]))
            remove.append(n)
            folded += 1
    for n in remove:
        g.node.remove(n)
    g.initializer.extend(new_inits)

    # Quantize Conv biases to INT32 (scale = s_input * s_weight, zp = 0).  ORT only fuses
    # DQ -> Conv -> Q into an INT8 QLinearConv kernel when the bias is also a DQ'd int32 tensor;
    # with a float bias it silently keeps the FP32 Conv (that is why it ran at FP32 speed).
    producer = {o: n for n in g.node for o in n.output}
    init_by_name = {i.name: i for i in g.initializer}
    bias_nodes, bias_inits, nb = [], [], 0
    for n in g.node:
        if n.op_type != "Conv" or len(n.input) < 3:
            continue
        dq_x, dq_w = producer.get(n.input[0]), producer.get(n.input[1])
        if not (dq_x and dq_w and dq_x.op_type == dq_w.op_type == "DequantizeLinear"):
            continue
        b = consts.get(n.input[2])
        if b is None and n.input[2] in init_by_name:
            b = numpy_helper.to_array(init_by_name[n.input[2]])
        if b is None:
            continue
        sx = float(consts[dq_x.input[1]].reshape(-1)[0])
        sw = consts[dq_w.input[1]].reshape(-1).astype(np.float64)
        bscale = (sx * sw).astype(np.float32)
        qb = np.rint(b.astype(np.float64) / bscale).astype(np.int32)
        base = n.input[2] + "_q"
        bias_inits += [numpy_helper.from_array(qb, base), numpy_helper.from_array(bscale, base + "_scale"),
                       numpy_helper.from_array(np.zeros_like(qb), base + "_zp")]
        bias_nodes.append(onnx.helper.make_node("DequantizeLinear", [base, base + "_scale", base + "_zp"],
                                                [base + "_dq"], axis=0))
        n.input[2] = base + "_dq"
        nb += 1
    g.initializer.extend(bias_inits)
    for bn in bias_nodes:
        g.node.insert(0, bn)
    print(f"[onnx] quantized {nb} conv biases to INT32")
    # drop Constant nodes / initializers that are no longer used
    used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
    for n in [n for n in g.node if n.op_type == "Constant" and n.output[0] not in used]:
        g.node.remove(n)
    for init in [i for i in g.initializer if i.name not in used]:
        g.initializer.remove(init)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"[onnx] folded {folded} weight QuantizeLinear nodes into INT8 initializers -> "
          f"qat.onnx {file_size_mb(path):.2f} MB")


class CalibReader:
    """onnxruntime CalibrationDataReader over the PTQ calibration subset."""

    def __init__(self, n=2000):
        self.batches = iter([{"input": xb.numpy()} for xb, _ in get_calibration_loader(n, 100)])

    def get_next(self):
        return next(self.batches, None)


def export_ptq(calib_images):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    pre = onnx_path("fp32_preprocessed")
    quant_pre_process(onnx_path("fp32"), pre)
    quantize_static(pre, onnx_path("ptq"), CalibReader(calib_images), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    os.remove(pre)
    print(f"[onnx] {'ptq.onnx':<18} {file_size_mb(onnx_path('ptq')):6.2f} MB (ORT static INT8)")


def main():
    ap = argparse.ArgumentParser(description="Export models to ONNX")
    ap.add_argument("--calib-images", type=int, default=2000)
    args = ap.parse_args()
    for name in ("fp32", "unstructured", "structured"):
        export_float(load_model(name), onnx_path(name))
    export_qat()
    export_ptq(args.calib_images)
    print(f"[onnx] all models exported to {ONNX_DIR}")


if __name__ == "__main__":
    main()
