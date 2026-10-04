#!/usr/bin/env bash
# Fast-kernel install, one-topic tok/s check, then the violation-count run.
set -uo pipefail
cd /workspace/dita-structure-translation-poc
mkdir -p results/fr_punct
exec > >(tee -a results/fr_punct/bootstrap.log) 2>&1
echo "BOOT $(date -Is)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libxml2-utils git rsync
bash scripts/setup_dtd.sh
if [[ ! -d data/userguide/DITA ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/oxygenxml/userguide.git data/userguide
  git -C data/userguide fetch --depth 1 origin db722d7af0c705cc3d6969f52bd1299a20ba1770
  git -C data/userguide checkout db722d7af0c705cc3d6969f52bd1299a20ba1770
fi
git -C data/userguide rev-parse HEAD
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python3 -m pip uninstall -y torchaudio || true
python3 -m pip install -q -r requirements.txt ninja packaging wheel
echo "---- causal-conv1d (no build isolation) ----"
set +e
python3 -m pip install --no-build-isolation "causal-conv1d>=1.5.0" > results/fr_punct/causal_conv1d_build.log 2>&1
echo CAUSAL_EXIT:$? | tee -a results/fr_punct/causal_conv1d_build.log
python3 -m pip install --no-build-isolation "flash-linear-attention>=0.3.0" > results/fr_punct/fla_build.log 2>&1
echo FLA_EXIT:$? | tee -a results/fr_punct/fla_build.log
set -e
python3 - << 'PY'
mods = {}
for name in ("causal_conv1d", "fla", "torch"):
    try:
        m = __import__(name)
        mods[name] = getattr(m, "__version__", "import_ok") + " @ " + getattr(m, "__file__", "")
    except Exception as e:
        mods[name] = f"MISSING {type(e).__name__}: {e}"
print(mods)
open("results/fr_punct/kernel_import.json","w").write(__import__("json").dumps(mods, indent=2))
PY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export POD_START_EPOCH="${POD_START_EPOCH:-1791092757}"
export BUDGET_USD="${BUDGET_USD:-3.8}"
export GPU_USD_PER_HOUR="${GPU_USD_PER_HOUR:-0.22}"
export SCAN_WALL_S="${SCAN_WALL_S:-54000}"
export TRAIN_WALL_S="${TRAIN_WALL_S:-20000}"
export TARGET_BROKEN=100
python3 -u scripts/run_fr_punct.py --bench-only
set +e
python3 - << 'PY'
import json, sys
info=json.load(open("results/fr_punct/bench.json"))
print("BENCH", info)
open("results/fr_punct/bench_gate.txt","w").write(str(info.get("tok_s")))
sys.exit(0 if float(info.get("tok_s") or 0) >= 35 else 2)
PY
GATE=$?
set -e
echo BENCH_GATE:$GATE
if [[ "$GATE" != "0" ]]; then
  echo "---- transformers bench under 35 tok/s; try vLLM without touching the training env ----"
  python3 -m venv /opt/vllm-venv || true
  set +e
  /opt/vllm-venv/bin/pip install -U pip
  /opt/vllm-venv/bin/pip install vllm > results/fr_punct/vllm_install.log 2>&1
  echo VLLM_EXIT:$? | tee -a results/fr_punct/vllm_install.log
  set -e
  if /opt/vllm-venv/bin/python -c "import vllm; print(vllm.__version__)"; then
    /opt/vllm-venv/bin/python scripts/vllm_bench.py || echo VLLM_BENCH_FAIL
  else
    echo "vLLM did not import; staying on transformers. See vllm_install.log" | tee results/fr_punct/vllm_skip.txt
  fi
fi
echo "---- full run ----"
python3 -u scripts/run_fr_punct.py
echo "DONE $(date -Is)"
