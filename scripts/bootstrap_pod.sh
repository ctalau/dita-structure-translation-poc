#!/usr/bin/env bash
set -euo pipefail
cd /workspace/dita-structure-translation-poc
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libxml2-utils git
bash scripts/setup_dtd.sh
if [[ ! -d data/userguide/DITA ]]; then
  rm -rf data/userguide
  git clone --depth 1 --filter=blob:none --sparse https://github.com/oxygenxml/userguide.git data/userguide
  git -C data/userguide sparse-checkout set DITA
fi
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements.txt
python3 - << 'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers", transformers.__version__)
PY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRAIN_WALL_S="${TRAIN_WALL_S:-2400}"
python3 -u scripts/run_gpu_poc.py
