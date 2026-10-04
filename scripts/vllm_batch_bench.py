"""Batch 1/4/8 tok/s and GPU util for Qwen3.5-4B on vLLM. Same short topic, 128 new tokens."""
from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path("/workspace/dita-structure-translation-poc")
OUT = ROOT / "results" / "fr_punct"
SYSTEM = (
    "You translate DITA XML from English to French. "
    "Output one XML document and nothing else. "
    "Keep every element, attribute, attribute value, and child order identical. "
    "Translate only human-readable text. "
    "Do not translate tag names, attribute values, ids, or hrefs. "
    "Use French guillemets (« ») for quotations, not English quotation marks. "
    "Insert a non-breaking space (U+00A0 or the entity &nbsp;) before ? ! : and ;. "
    "Keep the DOCTYPE. "
    "Do not add markdown fences or explanations."
)


def shortest():
    cands = json.loads((ROOT / "split" / "fr_candidates.json").read_text())["candidates"]
    rel = min(cands, key=lambda r: (ROOT / "data" / "userguide" / "DITA" / r).stat().st_size)
    return rel


def smi_during(fn):
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            line = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0]
            util_s, mem_s = [x.strip() for x in line.split(",")]
            samples.append({"util": int(util_s), "mem_mib": int(mem_s)})
            stop.wait(0.3)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    t0 = time.perf_counter()
    try:
        out = fn()
    finally:
        dt = time.perf_counter() - t0
        stop.set()
        th.join(timeout=1.5)
    utils = [s["util"] for s in samples]
    mems = [s["mem_mib"] for s in samples]
    summary = {
        "n": len(utils),
        "util_median": None if not utils else statistics.median(utils),
        "util_mean": None if not utils else round(sum(utils) / len(utils), 1),
        "util_max": None if not utils else max(utils),
        "mem_mib_max": None if not mems else max(mems),
        "util_samples": utils,
    }
    return out, dt, summary


def main():
    rel = shortest()
    src = (ROOT / "data" / "userguide" / "DITA" / rel).read_text(encoding="utf-8")
    print("topic", rel, "bytes", len(src.encode()), flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
    )
    messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": src}]]
    params_warm = SamplingParams(temperature=0, max_tokens=16)
    llm.chat(messages, params_warm, chat_template_kwargs={"enable_thinking": False})
    rows = {}
    params = SamplingParams(temperature=0, max_tokens=128)
    for b in (1, 4, 8):
        batch = messages * b

        def run(batch=batch):
            return llm.chat(batch, params, chat_template_kwargs={"enable_thinking": False})

        outs, dt, smi = smi_during(run)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        rec = {
            "batch": b,
            "seconds": round(dt, 3),
            "new_tokens": ntok,
            "tok_s": round(ntok / dt, 2),
            "per_seq_tok_s": round((ntok / b) / dt, 2),
            "smi": {k: smi[k] for k in smi if k != "util_samples"},
            "util_samples": smi["util_samples"],
        }
        rows[str(b)] = rec
        print("VLLM", json.dumps(rec), flush=True)
    dest = OUT / "vllm_batch_bench.json"
    dest.write_text(json.dumps({"topic": rel, "max_new": 128, "rows": rows}, indent=2))
    print("WROTE", dest, flush=True)


if __name__ == "__main__":
    main()
