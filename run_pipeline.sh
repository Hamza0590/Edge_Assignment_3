#!/usr/bin/env bash
# run_pipeline.sh - reproducible end-to-end pipeline (Linux / macOS / Git-Bash)
# Usage: bash run_pipeline.sh            (full run)
#        SKIP_TRAIN=1 bash run_pipeline.sh  (reuse models/resnet18_fp32.pth)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONWARNINGS=ignore
EPOCHS=${EPOCHS:-30}

step () {  # step "<title>" <logfile> <python args...>
  local title=$1 log=$2; shift 2
  echo -e "\n==== $title ===="
  local t0=$SECONDS
  python -u "$@" 2>&1 | tee "logs/$log"
  echo "---- $title finished in $((SECONDS - t0))s"
}

step "0. Prepare CIFAR-10"             00_prepare_data.log       src/prepare_data.py
[ "${SKIP_TRAIN:-0}" = "1" ] || \
step "1. Train FP32 baseline"          01_train_baseline.log     src/train_baseline.py --epochs "$EPOCHS"
step "2. Post-Training Quantization"   02_ptq.log                src/ptq.py --calib-images 2000
step "3. Quantization-Aware Training"  03_qat.log                src/qat.py --epochs 5
step "4. Unstructured pruning"         04_prune_unstructured.log src/prune_unstructured.py --target-sparsity 0.8 --iterations 4
step "5. Structured channel pruning"   05_prune_structured.log   src/prune_structured.py --ratio 0.5 --ft-epochs 10
step "6. ONNX export"                  06_export_onnx.log        src/export_onnx.py
step "7. PyTorch benchmark (5 models)" 07_benchmark.log          src/benchmark.py --suite all
step "8. ONNX Runtime CPU benchmark"   08_onnx_benchmark.log     src/onnx_inference.py --benchmark
step "9. Inference demo"               09_inference_demo.log     src/inference.py --save-samples 10 --model all --testset 10
echo "All done. Results in ./results, models in ./models"
