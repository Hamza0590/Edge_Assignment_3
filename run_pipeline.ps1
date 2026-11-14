# run_pipeline.ps1 - reproducible end-to-end pipeline (Windows PowerShell)
# Usage:  powershell -ExecutionPolicy Bypass -File run_pipeline.ps1            (full run)
#         powershell -ExecutionPolicy Bypass -File run_pipeline.ps1 -SkipTrain (reuse models/resnet18_fp32.pth)
param([switch]$SkipTrain, [int]$Epochs = 30)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$env:PYTHONWARNINGS = "ignore"

function Step($name, $log, $cmd) {
    Write-Host "`n==== $name ====" -ForegroundColor Cyan
    $start = Get-Date
    # PowerShell 5.1 turns every stderr line of a native exe into an error record; with
    # ErrorActionPreference=Stop a harmless library warning would abort the pipeline.
    # We rely on python's exit code instead.
    $ErrorActionPreference = "Continue"
    # (Tee-Object in PS 5.1 writes UTF-16 files -> write UTF-8 ourselves)
    $logPath = Join-Path $PSScriptRoot "logs/$log"
    Set-Content -Path $logPath -Value $null -Encoding UTF8
    & python -u @cmd 2>&1 | ForEach-Object { $line = "$_"; Add-Content -Path $logPath -Value $line -Encoding UTF8; $line }
    $ErrorActionPreference = "Stop"
    if ($LASTEXITCODE -eq 3) { Write-Host "Paused (STOP file). Re-run the same command to resume." -ForegroundColor Yellow; exit 3 }
    if ($LASTEXITCODE -ne 0) { throw "$name failed (see logs/$log)" }
    Write-Host ("---- {0} finished in {1:N0}s" -f $name, ((Get-Date) - $start).TotalSeconds) -ForegroundColor Green
}

Step "0. Prepare CIFAR-10"             "00_prepare_data.log"       @("src/prepare_data.py")
if (-not $SkipTrain) {
Step "1. Train FP32 baseline"          "01_train_baseline.log"     @("src/train_baseline.py", "--epochs", "$Epochs")
}
Step "2. Post-Training Quantization"   "02_ptq.log"                @("src/ptq.py", "--calib-images", "2000")
Step "3. Quantization-Aware Training"  "03_qat.log"                @("src/qat.py", "--epochs", "5")
Step "4. Unstructured pruning"         "04_prune_unstructured.log" @("src/prune_unstructured.py", "--target-sparsity", "0.8", "--iterations", "4")
Step "5. Structured channel pruning"   "05_prune_structured.log"   @("src/prune_structured.py", "--ratio", "0.5", "--ft-epochs", "10")
Step "6. ONNX export"                  "06_export_onnx.log"        @("src/export_onnx.py")
Step "7. PyTorch benchmark (5 models)" "07_benchmark.log"          @("src/benchmark.py", "--suite", "all")
Step "8. ONNX Runtime CPU benchmark"   "08_onnx_benchmark.log"     @("src/onnx_inference.py", "--benchmark")
Step "9. Inference demo"               "09_inference_demo.log"     @("src/inference.py", "--save-samples", "10", "--model", "all", "--testset", "10")
Write-Host "`nAll done. Results in ./results, models in ./models" -ForegroundColor Green
