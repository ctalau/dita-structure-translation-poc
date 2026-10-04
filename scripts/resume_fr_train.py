#!/usr/bin/env python3
"""Continue French RL after the scan. Does not translate the guide again.

The first training pass died when xmllint stderr was not valid UTF-8.
Those five updates were not checkpointed, so this starts LoRA from zero
on the frozen broken-topic split.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig

import run_fr_poc as R

def main():
    log = R.log_factory()
    manifest = json.loads((R.ROOT / "split" / "manifest_fr.json").read_text())
    test = manifest["test"]
    train = manifest["train"]
    base_doc = json.loads((R.OUT / "baseline_test.json").read_text())
    log("resume_train", n_train=len(train), n_test=len(test),
        note="scan not repeated; LoRA restarted from zero because the crash left no adapter")
    tok = AutoTokenizer.from_pretrained(R.MODEL_ID, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = None
    for cls in (R.AutoModelForImageTextToText, R.AutoModelForCausalLM):
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                R.MODEL_ID,
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            log("model_class", name=cls.__name__)
            break
        except Exception as e:
            R.note(f"{cls.__name__} failed: {e}")
    if model is None:
        raise SystemExit("load failed")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    targets = R.pick_lora_targets(model)
    log("lora_targets", targets=targets)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=R.LR)
    import time
    train_start = time.perf_counter()
    updates = 0
    skipped_flat = 0
    train_rows = []
    for step, rel in enumerate(train):
        if time.perf_counter() - train_start > R.TRAIN_WALL_S:
            R.note(f"train wall after {step} topics; {updates} updates")
            break
        src = R.load_src(rel)
        mx = R.max_new_for(src, 896)
        prompt = R.format_prompt(tok, src)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        samples = []
        oom = False
        for k in range(2):
            try:
                gens = R.generate_batch(model, tok, [prompt], mx, sample=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                R.note(f"OOM sampling {rel} k={k}")
                oom = True
                break
            text, dt, _plen, nnew, ids = gens[0]
            hit = nnew >= mx - 1
            try:
                sc = R.score_pair(src, text, hit_token_cap=hit)
            except Exception as e:
                R.note(f"score failed {rel} k={k}: {type(e).__name__}: {e}")
                sc = {"reward": -1.0, "reasons": ["score_error"], "primary_reason": "score_error",
                      "dtd_valid": False, "skeleton_match": False}
            samples.append((text, sc, dt, nnew, ids))
            print(f"sample {step}/{len(train)} {rel} k={k} r={sc['reward']:.3f} why={sc.get('primary_reason')} {dt:.1f}s", flush=True)
        if oom or len(samples) < 2:
            continue
        rewards = [s[1]["reward"] for s in samples]
        if max(rewards) - min(rewards) < 1e-4:
            skipped_flat += 1
            train_rows.append({"path": rel, "rewards": rewards, "reasons": [s[1].get("reasons") for s in samples], "updated": False})
            log("rl_flat", step=step, path=rel, rewards=rewards)
            continue
        mean_r = sum(rewards) / len(rewards)
        var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std = var ** 0.5
        advs = [(r - mean_r) / (std + 1e-6) for r in rewards]
        opt.zero_grad(set_to_none=True)
        loss_acc = None
        try:
            for (_text, sc, dt, nnew, ids), adv in zip(samples, advs):
                lp = R.completion_logprob_ids(model, ids, prompt_ids)
                if lp is None:
                    continue
                term = -(float(adv)) * lp
                (term / len(samples)).backward()
                loss_acc = float(term.detach()) if loss_acc is None else loss_acc + float(term.detach())
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            opt.step()
            updates += 1
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            R.note(f"OOM backward {rel}")
            continue
        train_rows.append({"path": rel, "rewards": rewards, "advs": advs, "updated": True, "loss": loss_acc,
                           "reasons": [s[1].get("reasons") for s in samples]})
        log("rl_step", step=step, path=rel, rewards=rewards, updates=updates,
            vram_gb=round(torch.cuda.memory_allocated() / 1e9, 2))
    log("rl_done", updates=updates, skipped_flat=skipped_flat, topics_attempted=len(train_rows),
        train_s=round(time.perf_counter() - train_start, 1))
    adapter_dir = R.OUT / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    model.eval()
    model.config.use_cache = True
    after_rows = []
    for rel in test:
        src = R.load_src(rel)
        mx = R.max_new_for(src, 896)
        prompt = R.format_prompt(tok, src)
        gens = R.generate_batch(model, tok, [prompt], mx, sample=False)
        text, dt, _plen, nnew, _ids = gens[0]
        row = R.row_from_gen(rel, src, text, dt, nnew, mx, "after")
        after_rows.append(row)
        print(f"after {rel} lat={dt:.1f}s broken={row['format_broken']} why={row['primary_reason']}", flush=True)
    after_summary = R.summarize(after_rows)
    (R.OUT / "after_test.json").write_text(json.dumps({"summary": after_summary, "rows": after_rows}, indent=2, ensure_ascii=False))
    (R.OUT / "train_trace.json").write_text(json.dumps(train_rows, indent=2))
    summary = {
        "model": R.MODEL_ID,
        "language_pair": "en-fr",
        "resume": True,
        "updates": updates,
        "skipped_flat_reward": skipped_flat,
        "train_topics_attempted": len(train_rows),
        "train_topics_available": len(train),
        "lora_targets": targets,
        "baseline": base_doc["summary"],
        "after": after_summary,
        "surprises": R.SURPRISES,
        "system_prompt": R.SYSTEM,
    }
    (R.OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log("finished", updates=updates, skipped_flat=skipped_flat,
        after_broken=after_summary.get("format_broken"))
    print(json.dumps(summary, indent=2)[:4000], flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (R.OUT / "crash_resume.txt").write_text(traceback.format_exc())
        raise
