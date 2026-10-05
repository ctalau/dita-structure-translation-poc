#!/usr/bin/env bash
# Runs on the pod. Unpacks the code, installs vLLM + PEFT, pre-downloads the model.
set -euxo pipefail
mkdir -p /workspace/repo && tar xzf /workspace/code.tgz -C /workspace/repo
python3 -m pip install -q --upgrade pip
# vllm 0.11.0 ships torch 2.8 + cu128, which runs on CUDA 12.8 host drivers.
python3 -m pip install -q vllm==0.11.0 peft sacrebleu lxml
python3 - <<'PY'
import torch, vllm, transformers, peft
print("VERSIONS", torch.__version__, vllm.__version__, transformers.__version__, peft.__version__, torch.cuda.get_device_name(0))
PY
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507')"
nvidia-smi
echo SETUP_DONE
