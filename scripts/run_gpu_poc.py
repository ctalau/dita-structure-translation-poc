#!/usr/bin/env python3
"""Baseline eval + short GRPO QLoRA + after eval for EN->RO DITA topics.

Q4 training = bitsandbytes NF4 on Qwen/Qwen3.5-4B. Designed for one
community RTX 3090 and a hard wall clock so the pod can be deleted.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dita_reward import score_pair  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "split" / "manifest.json").read_text())
DITA = ROOT / "data" / "userguide" / "DITA"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
SURPRISES = []

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")
LR = 5e-6
TRAIN_WALL_S = int(os.environ.get("TRAIN_WALL_S", "3600"))  # 60 min of RL
DEVICE_LABEL = "RunPod Community RTX 3090, NF4 4-bit (bitsandbytes), greedy"

SYSTEM = (
    "You translate DITA XML from English to Romanian. "
    "Output one XML document and nothing else. "
    "Keep every element, attribute, and attribute value identical and in the same order. "
    "Translate only human-readable text. Do not translate tag names, attribute values, ids, or hrefs. "
    "Keep the DOCTYPE. Do not add markdown fences or explanations."
)


def note(msg: str):
    print("SURPRISE:", msg, flush=True)
    SURPRISES.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "msg": msg})


def load_src(rel: str) -> str:
    return (DITA / rel).read_text(encoding="utf-8")


def format_prompt(tok, xml: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": xml},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        note("chat template has no enable_thinking flag; patched a closed think block if needed")
    # Qwen3.5 thinks by default. A closed empty think block forces a direct answer.
    stripped = text.rstrip()
    if stripped.endswith("<think>"):
        text = stripped[:-7] + "<think>\n\n</think>\n\n"
    return text


def max_new_for(src: str) -> int:
    # tags are cheap; leave headroom so the topic is not cut mid-element
    return int(min(640, max(160, len(src) / 3.0 + 48)))


def generate_one(model, tok, prompt: str, max_new: int, sample: bool):
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = dict(
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        do_sample=sample,
    )
    if sample:
        kwargs.update(temperature=0.7, top_p=0.8, top_k=20)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    dt = time.perf_counter() - t0
    new_tokens = out.shape[1] - prompt_len
    text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
    return text, dt, int(prompt_len), int(new_tokens), out.detach().cpu()


def completion_logprob_ids(model, full_ids, prompt_len: int):
    """Sum of token logprobs of generated ids. Grad enabled. Uses the exact sampled tokens."""
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    input_ids = full_ids.to(model.device)
    if input_ids.shape[1] <= prompt_len:
        return None
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, prompt_len - 1 : -1, :]
    target = input_ids[:, prompt_len:]
    # align if the model dropped a position
    n = min(logits.shape[1], target.shape[1])
    logits = logits[:, :n, :]
    target = target[:, :n]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return tok_lp.sum()


def eval_split(model, tok, rels, tag: str):
    rows = []
    for rel in rels:
        src = load_src(rel)
        prompt = format_prompt(tok, src)
        mx = max_new_for(src)
        try:
            text, dt, plen, nnew, _ = generate_one(model, tok, prompt, mx, sample=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            note(f"OOM during {tag} generate {rel}; recorded as failure")
            rows.append({
                "path": rel, "tag": tag, "latency_s": None, "well_formed": False,
                "skeleton_match": False, "dtd_valid": False, "dtd_errors": 1,
                "reward": -1.0, "error_tail": "oom", "hyp_xml": "", "new_tokens": 0,
                "prompt_tokens": None,
            })
            continue
        scored = score_pair(src, text)
        row = {
            "path": rel,
            "tag": tag,
            "latency_s": round(dt, 3),
            "prompt_tokens": plen,
            "new_tokens": nnew,
            "max_new": mx,
            "well_formed": scored["well_formed"],
            "skeleton_match": scored["skeleton_match"],
            "dtd_valid": scored["dtd_valid"],
            "dtd_errors": scored["dtd_errors"],
            "reward": scored["reward"],
            "error_tail": scored["error_tail"],
            "hyp_xml": scored["hyp_xml"],
        }
        rows.append(row)
        print(
            f"{tag} {rel} lat={dt:.2f}s tok={nnew}/{mx} "
            f"wf={row['well_formed']} sk={row['skeleton_match']} dtd={row['dtd_valid']} err={row['dtd_errors']} r={row['reward']:.2f}",
            flush=True,
        )
    return rows


def summarize(rows):
    lats = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    p50 = statistics.median(lats) if lats else None
    return {
        "n": len(rows),
        "schema_invalid_topics": sum(1 for r in rows if not (r["well_formed"] and r["dtd_valid"])),
        "schema_error_total": int(sum((r["dtd_errors"] or 0) for r in rows)),
        "skeleton_mismatch_topics": sum(1 for r in rows if not r["skeleton_match"]),
        "valid_and_skeleton_ok": sum(1 for r in rows if r["well_formed"] and r["dtd_valid"] and r["skeleton_match"]),
        "mean_reward": round(sum(r["reward"] for r in rows) / max(1, len(rows)), 4),
        "latency_p50_s": None if p50 is None else round(p50, 3),
        "latency_device": DEVICE_LABEL,
        "decoding": "greedy",
    }


def pick_lora_targets(model):
    suffixes = set()
    for n, m in model.named_modules():
        cls = m.__class__.__name__
        if cls not in ("Linear", "Linear4bit"):
            continue
        if any(x in n for x in ("vision", "visual", "lm_head", "embed")):
            continue
        suffixes.add(n.split(".")[-1])
    preferred = [s for s in ("q_proj", "k_proj", "v_proj", "o_proj") if s in suffixes]
    if len(preferred) >= 2:
        return preferred
    # linear-attention blocks in Qwen3.5 use different projection names
    fallback = sorted(s for s in suffixes if "proj" in s or s in ("in_proj_qkv", "in_proj_z", "out_proj"))
    return fallback[:8]


def main():
    log_path = OUT / "run_log.jsonl"
    def log(event, **kw):
        rec = {"event": event, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}
        print(json.dumps(rec)[:500], flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    log("start", model=MODEL_ID, train_wall_s=TRAIN_WALL_S)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    t0 = time.perf_counter()
    load_kwargs = dict(
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    load_err = None
    model = None
    for cls in (AutoModelForImageTextToText, AutoModelForCausalLM):
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(MODEL_ID, **load_kwargs)
            log("model_class", name=cls.__name__)
            break
        except Exception as e:
            load_err = e
            note(f"{cls.__name__} failed to load: {type(e).__name__}: {e}")
    if model is None:
        raise SystemExit(f"could not load {MODEL_ID}: {load_err}")
    log("loaded", seconds=round(time.perf_counter() - t0, 1),
        vram_gb=round(torch.cuda.memory_allocated() / 1e9, 2))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = pick_lora_targets(model)
    log("lora_targets", targets=targets)
    if not targets:
        note("no Linear targets found for LoRA")
        raise SystemExit(2)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # smoke one short forward so we fail before burning the whole baseline if the arch is broken
    try:
        smoke_src = load_src(MANIFEST["test"][0])
        smoke_prompt = format_prompt(tok, smoke_src[:400])
        text, dt, plen, nnew, _ = generate_one(model, tok, smoke_prompt, 32, sample=False)
        log("smoke", seconds=round(dt, 2), new_tokens=nnew, prompt_tokens=plen, head=text[:120])
    except Exception as e:
        note(f"smoke generation failed: {type(e).__name__}: {e}")
        log("smoke_fail", error=traceback.format_exc()[-2000:])
        raise

    log("baseline_begin")
    base_rows = eval_split(model, tok, MANIFEST["test"], "baseline")
    (OUT / "baseline_test.json").write_text(json.dumps({
        "summary": summarize(base_rows),
        "rows": base_rows,
    }, indent=2))
    log("baseline_done", **summarize(base_rows))

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    train_start = time.perf_counter()
    updates = 0
    skipped_flat = 0
    train_rows = []
    for step, rel in enumerate(MANIFEST["train_update"]):
        if time.perf_counter() - train_start > TRAIN_WALL_S:
            note(f"hit train wall clock after {step} topics; stopping RL early")
            break
        src = load_src(rel)
        prompt = format_prompt(tok, src)
        mx = max_new_for(src)
        samples = []
        oom = False
        for k in range(2):
            try:
                text, dt, plen, nnew, ids = generate_one(model, tok, prompt, mx, sample=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                note(f"OOM sampling {rel} k={k}")
                oom = True
                break
            sc = score_pair(src, text)
            samples.append((text, sc, dt, nnew, plen, ids))
            print(f"sample {step} {rel} k={k} r={sc['reward']:.2f} dtd={sc['dtd_valid']} sk={sc['skeleton_match']} {dt:.1f}s", flush=True)
        if oom or len(samples) < 2:
            continue
        rewards = [s[1]["reward"] for s in samples]
        mean_r = sum(rewards) / len(rewards)
        if max(rewards) - min(rewards) < 1e-6:
            skipped_flat += 1
            train_rows.append({"path": rel, "rewards": rewards, "updated": False})
            continue
        # GRPO advantage: reward minus group mean, then divide by group std
        var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std = var ** 0.5
        advs = [(r - mean_r) / (std + 1e-6) for r in rewards]
        opt.zero_grad(set_to_none=True)
        loss_acc = None
        try:
            for (text, sc, dt, nnew, plen, ids), adv in zip(samples, advs):
                lp = completion_logprob_ids(model, ids, plen)
                if lp is None:
                    note(f"logprob empty {rel}")
                    continue
                # policy gradient. adv is a detached scalar.
                term = -(float(adv)) * lp
                (term / len(samples)).backward()
                loss_acc = float(term.detach()) if loss_acc is None else loss_acc + float(term.detach())
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
            updates += 1
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            note(f"OOM on backward {rel}; skipped update")
            continue
        train_rows.append({"path": rel, "rewards": rewards, "advs": advs, "updated": True, "loss": loss_acc})
        log("rl_step", step=step, path=rel, rewards=rewards, updates=updates, vram_gb=round(torch.cuda.memory_allocated()/1e9, 2))

    log("rl_done", updates=updates, skipped_flat=skipped_flat, train_s=round(time.perf_counter()-train_start, 1))
    adapter_dir = OUT / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    log("after_begin")
    after_rows = eval_split(model, tok, MANIFEST["test"], "after")
    (OUT / "after_test.json").write_text(json.dumps({
        "summary": summarize(after_rows),
        "rows": after_rows,
    }, indent=2))
    # val is reported but not used to pick anything
    val_rows = eval_split(model, tok, MANIFEST["val"], "val_after")
    summary = {
        "model": MODEL_ID,
        "quant": "bitsandbytes NF4 QLoRA (4-bit base)",
        "method": "GRPO-style group size 2, advantage = (r - mean) / std, one QLoRA step per topic",
        "lr": LR,
        "lora_r": 8,
        "lora_targets": targets,
        "updates": updates,
        "skipped_flat_reward": skipped_flat,
        "train_wall_s_budget": TRAIN_WALL_S,
        "baseline": summarize(base_rows),
        "after": summarize(after_rows),
        "val_after": summarize(val_rows),
        "surprises": SURPRISES,
        "language_pair": "en-ro",
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    (OUT / "train_trace.json").write_text(json.dumps(train_rows, indent=2))
    log("finished", baseline=summary["baseline"], after=summary["after"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "crash.txt").write_text(traceback.format_exc())
        raise
