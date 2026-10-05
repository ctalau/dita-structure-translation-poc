"""Greedy translation of whole topics with vLLM, optionally with a LoRA.

  python3 -m fr_rl.eval_vllm --topics split/fr_rl/dev_topics.jsonl split/fr_rl/test_topics.jsonl \
      --out results/fr_rl/base_outputs.jsonl [--lora PATH]

Writes one row per topic: {"id", "split", "hyp", "n_out_tokens", "finish_reason"}.
Scoring happens separately (fr_rl.score), on CPU, so outputs can be
rescored when a reference or a checker is fixed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def load_rows(paths):
    rows = []
    for p in paths:
        split = Path(p).stem.split("_")[0]
        for line in open(p, encoding="utf-8"):
            d = json.loads(line)
            rows.append({"id": d["id"], "split": split, "src": d["src"]})
    return rows


def generate(llm, rows, lora=None, max_tokens=4096):
    from vllm import SamplingParams
    from fr_rl.prompt import messages
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.chat([messages(r["src"]) for r in rows], sp, lora_request=lora, use_tqdm=False)
    res = []
    for r, o in zip(rows, outs):
        c = o.outputs[0]
        res.append({"id": r["id"], "split": r["split"], "hyp": c.text,
                    "n_out_tokens": len(c.token_ids), "finish_reason": c.finish_reason})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora")
    args = ap.parse_args()
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
              enable_lora=bool(args.lora), max_lora_rank=32, seed=0)
    lora = LoRARequest("eval", 1, args.lora) if args.lora else None
    rows = load_rows(args.topics)
    t = time.time()
    res = generate(llm, rows, lora)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(res), "seconds": round(time.time() - t, 1),
                      "out_tokens": sum(r["n_out_tokens"] for r in res),
                      "hit_cap": sum(r["finish_reason"] == "length" for r in res)}))


if __name__ == "__main__":
    main()
