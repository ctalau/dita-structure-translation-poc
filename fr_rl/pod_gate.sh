#!/usr/bin/env bash
# Pod health gate: prints min(HF, PyPI) download speed in bytes/s, or 0 if CUDA
# cannot initialise or the driver is older than 570 (torch cu128 needs >= 570).
a=$(curl -s -o /dev/null -m 10 -r 0-104857599 -w '%{speed_download}' -L \
  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model-00001-of-00003.safetensors)
b=$(curl -s -o /dev/null -m 10 -w '%{speed_download}' \
  https://files.pythonhosted.org/packages/47/33/d19e0763c34392ec956534536fa837c060495bfff31ed83452135ea7608d/vllm-0.11.0-cp38-abi3-manylinux1_x86_64.whl)
c=$(python3 -c 'import ctypes; print(ctypes.CDLL("libcuda.so.1").cuInit(0))' 2>/dev/null || echo nolib)
d=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
[ "${d:-0}" -ge 570 ] || c="old_driver_$d"
python3 -c "print(min($a, $b) if '$c' == '0' else 0)"
echo "HF=$a PYPI=$b cuInit=$c driver=$d"
