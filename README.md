# Assignment 3: Model Optimization for Edge Deployment
**Edge Computing & Intelligent Systems**: quantization (PTQ and QAT) and pruning (unstructured and structured) of ResNet-18 on CIFAR-10, with ONNX CPU deployment.

| Question | What is implemented |
|---|---|
| Q1 (50) | FP32 baseline → **Post-Training Static Quantization** (fusion, onednn backend, calibration, INT8 convert) and **Quantization-Aware Training** (fake-quant, fine-tune, INT8 convert), plus an inference pipeline and an automated benchmark |
| Q2 (50) | **Iterative global magnitude unstructured pruning** (masks, then made permanent) and **structured channel pruning** with architecture reconstruction, a fine-tuning pipeline, an inference pipeline and an automated benchmark |
| Extra | ONNX export of all 5 models, ONNX Runtime CPU inference/benchmark, technical report, demo guide |

---

## 1. Folder structure
```
.
├── src/
│   ├── common.py              # data, ResNet-18 (quant-ready, prunable), train/eval/latency helpers, model loaders
│   ├── prepare_data.py        # downloads CIFAR-10 (HF mirror, torchvision fallback) -> data/*.npz
│   ├── train_baseline.py      # Q1.1 / Q2.1  FP32 training script
│   ├── calibration.py         # Q1.2  PTQ calibration pipeline (training subset)
│   ├── ptq.py                 # Q1.2  Post-Training Static Quantization
│   ├── qat.py                 # Q1.3  Quantization-Aware Training + fine-tuning
│   ├── finetune.py            # Q2.4  shared fine-tuning pipeline for both pruning methods
│   ├── prune_unstructured.py  # Q2.2  iterative magnitude pruning
│   ├── prune_structured.py    # Q2.3  channel pruning + architecture reconstruction
│   ├── inference.py           # Q1.4 / Q2.5  inference for all 5 models (images or test set)
│   ├── benchmark.py           # Q1.5 / Q2.6  automated benchmark -> CSV + plot
│   ├── export_onnx.py         # ONNX export (FP32, PTQ, QAT, both pruned)
│   ├── onnx_inference.py      # ONNX Runtime CPU inference + benchmark
│   └── make_screenshots.py    # renders logs as PNG screenshots for the report
├── models/                    # 5 executable checkpoints (+ extras), models/onnx/*.onnx
├── results/                   # all CSV results + benchmark_plots.png
├── logs/                      # console output of every pipeline step
├── samples/                   # 10 CIFAR-10 test images for the inference demo
├── report/REPORT.md           # technical report (+ report/screenshots/)
├── decisions/DECISIONS.md     # design decisions and their rationale
├── DEMO.md                    # demo script, issues faced and their solutions
├── run_pipeline.ps1 / .sh     # one-command reproducible pipeline
└── requirements.txt
```

## 2. Installation
Python 3.10+ is required (tested on Python 3.13.5, Windows 11, RTX 5060 Ti, CUDA 12.8).
```powershell
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
A GPU is optional: training and fine-tuning use CUDA when available. All quantized inference and benchmarking runs on the **CPU**.

## 3. Run everything (reproducible pipeline)
```powershell
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1              # full run (~28 min on RTX 5060 Ti + Ryzen 7 7700)
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipTrain   # reuse the trained FP32 model
```
```bash
bash run_pipeline.sh                    # Linux / macOS / Git-Bash
SKIP_TRAIN=1 bash run_pipeline.sh
```
**Pause / resume training:** create an empty file named `STOP` in the project root (`New-Item STOP`). Training stops cleanly after the current epoch and saves its full state to `models/baseline_train_state.pth`. Run the same command again to resume from that epoch.

Seeds are fixed (`set_seed`) and every hyper-parameter is a command-line argument with the default used for the reported results. Each step's console output goes to `logs/NN_step.log`.

## 4. Step-by-step execution
| # | Command | Produces |
|---|---|---|
| 0 | `python src/prepare_data.py` | `data/cifar10_{train,test}.npz` |
| 1 | `python src/train_baseline.py --epochs 30` | `models/resnet18_fp32.pth`, `results/baseline_training_log.csv` |
| 2 | `python src/ptq.py --calib-images 2000` | `models/resnet18_ptq_int8.pt` (TorchScript INT8) |
| 3 | `python src/qat.py --epochs 5` | `models/resnet18_qat_int8.pt`, `results/qat_training_log.csv` |
| 4 | `python src/prune_unstructured.py --target-sparsity 0.8 --iterations 4` | `models/resnet18_unstructured_pruned.pth` (+ `_sparse.pth`) |
| 5 | `python src/prune_structured.py --ratio 0.5 --ft-epochs 10` | `models/resnet18_structured_pruned.pth` |
| 6 | `python src/export_onnx.py` | `models/onnx/{fp32,ptq,qat,unstructured,structured}.onnx` |
| 7 | `python src/benchmark.py --suite all` | `results/q1_*.csv`, `results/q2_*.csv`, `results/all_models_benchmark.csv`, plot |
| 8 | `python src/onnx_inference.py --benchmark` | `results/onnx_cpu_benchmark.csv` |
| 9 | `python src/inference.py --model all --testset 10` | console predictions |

Run step 2 on its own as `python src/calibration.py`; it prints the observer ranges and the resulting qparams.

### Inference examples
```powershell
python src/inference.py --model all --image samples/cat_0.png      # all 5 models on one image
python src/inference.py --model qat --image "samples/*.png"        # one model, many images
python src/inference.py --model structured --testset 20            # first 20 test images
python src/onnx_inference.py --image samples/ship_1.png            # ONNX Runtime (no PyTorch model code)
python src/benchmark.py --suite quant                              # only Q1 models
python src/benchmark.py --suite pruning --threads 1                # only Q2 models, single core
```

## 5. Checkpoints
| Model | File | Format | Load with |
|---|---|---|---|
| FP32 baseline | `models/resnet18_fp32.pth` | state_dict | `common.load_model("fp32")` |
| PTQ INT8 | `models/resnet18_ptq_int8.pt` | TorchScript (self-contained) | `torch.jit.load(...)` |
| QAT INT8 | `models/resnet18_qat_int8.pt` | TorchScript (self-contained) | `torch.jit.load(...)` |
| Unstructured-pruned | `models/resnet18_unstructured_pruned.pth` | state_dict, pruning permanent (80% zeros) | `common.load_model("unstructured")` |
| Structured-pruned | `models/resnet18_structured_pruned.pth` | `{mid_channels, state_dict}`, smaller architecture | `common.load_model("structured")` |

Extras: `*_int8_state_dict.pth` (INT8 state dicts), `resnet18_qat_fakequant.pth` (fake-quant QAT model, used for the ONNX QDQ export), `resnet18_unstructured_pruned_sparse.pth` (bitmask + values compressed storage).

## 6. Results (CIFAR-10 test set, 10,000 images, CPU: Ryzen 7 7700)
| Model | Top-1 acc. | Size | PyTorch latency bs=1 | ONNX Runtime latency bs=1 |
|---|---|---|---|---|
| FP32 baseline | 93.73% | 42.70 MB | 5.92 ms | 2.12 ms |
| PTQ INT8 | 93.75% | 10.79 MB | 1.59 ms | 0.60 ms |
| QAT INT8 | 93.83% | 10.79 MB | 1.55 ms | 0.67 ms |
| Unstructured-pruned (80%) | 93.85% | 42.70 MB (10.19 MB sparse) | 5.94 ms | 2.13 ms |
| Structured-pruned (50% mid-channels) | 93.19% | 21.73 MB | 3.61 ms | 0.88 ms |

The full discussion is in `report/REPORT.md`, the demo script and issues are in `DEMO.md`, and the raw numbers are in `results/*.csv`.

## 7. Submission
Zip the folder **without** `data/` (dataset), `.venv/` and `__pycache__/`, and name it `ROLL-NUM_SECTION.zip` (e.g. `22i-0001_A.zip`):
```powershell
Compress-Archive -Path src,models,results,logs,samples,report,decisions,README.md,DEMO.md,requirements.txt,run_pipeline.ps1,run_pipeline.sh -DestinationPath 22i-0001_A.zip
```
