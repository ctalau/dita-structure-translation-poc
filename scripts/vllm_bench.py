"""One-topic tok/s on vLLM. Does not train. Records the error if Qwen3.5 will not load."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fr_punct"
OUT.mkdir(parents=True, exist_ok=True)
rel = json.loads((ROOT / "split" / "fr_candidates.json").read_text())["candidates"][0]
src = (ROOT / "data" / "userguide" / "DITA" / rel).read_text(encoding="utf-8")
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
try:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        language_model_only=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=256)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": src[:1500]}]
    t0 = time.perf_counter()
    outs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
    dt = time.perf_counter() - t0
    n = len(outs[0].outputs[0].token_ids)
    rec = {"stack": "vllm", "seconds": round(dt, 3), "new_tokens": n, "tok_s": round(n / dt, 2), "path": rel}
except Exception as e:
    rec = {"stack": "vllm", "error": f"{type(e).__name__}: {e}"}
(OUT / "vllm_bench.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
