#!/usr/bin/env bash
set -euo pipefail
cd /workspace/dita-structure-translation-poc
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libxml2-utils git
bash scripts/setup_dtd.sh
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python3 -m pip uninstall -y torchaudio || true
python3 -m pip install -q -r requirements.txt
# Fast kernels for Qwen3.5 linear attention. Failure is recorded, not fatal.
if python3 -m pip install -q causal-conv1d flash-linear-attention; then
  echo KERNELS_OK
else
  echo KERNELS_FAIL
fi
python3 - << 'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
for name in ("causal_conv1d", "fla"):
    try:
        __import__(name)
        print(name, "import_ok")
    except Exception as e:
        print(name, "MISSING", type(e).__name__, e)
PY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export SCAN_WALL_S="${SCAN_WALL_S:-10800}"
export TRAIN_WALL_S="${TRAIN_WALL_S:-3000}"
python3 -u scripts/run_fr_poc.py
