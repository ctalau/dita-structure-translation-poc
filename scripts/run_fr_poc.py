#!/usr/bin/env python3
"""English to French DITA structure PoC.

1. Greedy-scan source-valid topics, in sorted path order, until 100 translations
   break formatting (or the scan wall / the fittable pool runs out).
2. Hold out 10 test and 10 val with seed 1337. Train only on the broken rest.
3. Re-decode the same 10 greedily and write before/after JSON.

The system prompt is one fixed string. Eval is greedy. Thinking is forced off.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
import traceback
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dita_reward import expand_nbsp, extract_xml, punct_stats, score_pair  # noqa: E402
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = json.loads((ROOT / "split" / "fr_candidates.json").read_text())
DITA = ROOT / "data" / "userguide" / "DITA"
OUT = ROOT / "results" / "fr"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")
LR = 5e-6
SCAN_WALL_S = int(os.environ.get("SCAN_WALL_S", "10800"))
TRAIN_WALL_S = int(os.environ.get("TRAIN_WALL_S", "3000"))
TARGET_BROKEN = int(os.environ.get("TARGET_BROKEN", "100"))
SEED = 1337
DEVICE_LABEL = "RunPod Community RTX 3090, NF4 4-bit (bitsandbytes), greedy"

# Fixed. Do not paraphrase between samples or between eval and training.
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

SURPRISES = []


def note(msg: str):
    print("SURPRISE:", msg, flush=True)
    SURPRISES.append({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "msg": msg})


def log_factory():
    path = OUT / "run_log.jsonl"
    def log(event, **kw):
        rec = {"event": event, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}
        line = json.dumps(rec, ensure_ascii=False)
        print(line[:800], flush=True)
        with path.open("a") as f:
            f.write(line + "\n")
    return log


def load_src(rel: str) -> str:
    return (DITA / rel).read_text(encoding="utf-8")


def need_tokens(src: str) -> int:
    # French expands a bit, and XML tags are token-expensive. This is only a gate.
    return int(len(src) / 2.8 + 48)


def max_new_for(src: str, cap: int) -> int:
    return int(min(cap, max(96, need_tokens(src))))


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
        note("chat template has no enable_thinking flag; patched a closed think block")
    stripped = text.rstrip()
    if stripped.endswith("<think>"):
        text = stripped[:-7] + "<think>\n\n</think>\n\n"
    return text


class XmlClosed(StoppingCriteria):
    """Stop a row once its completion parses as XML, so a finished topic does not run to the cap."""

    def __init__(self, tokenizer, prompt_len: int):
        self.tok = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        n_new = input_ids.shape[1] - self.prompt_len
        batch = input_ids.shape[0]
        if n_new < 16 or (n_new % 16 != 0):
            return torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        flags = []
        for i in range(batch):
            text = self.tok.decode(input_ids[i, self.prompt_len:], skip_special_tokens=True)
            ok = False
            try:
                ET.fromstring(expand_nbsp(extract_xml(text)))
                ok = True
            except ET.ParseError:
                ok = False
            flags.append(ok)
        return torch.tensor(flags, dtype=torch.bool, device=input_ids.device)


def _trim(ids, pad_id):
    if pad_id is None:
        return ids
    while ids and ids[-1] == pad_id:
        ids.pop()
    return ids


def generate_batch(model, tok, prompts, max_new: int, sample: bool):
    tok.padding_side = "left"
    old_pad = tok.pad_token_id
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    inputs = tok(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = dict(
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        do_sample=sample,
        stopping_criteria=[XmlClosed(tok, prompt_len)],
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
    rows = []
    pad_id = tok.pad_token_id
    for i in range(out.shape[0]):
        new_ids = _trim(out[i, prompt_len:].detach().cpu().tolist(), pad_id)
        text = tok.decode(new_ids, skip_special_tokens=True)
        rows.append((text, dt, prompt_len, len(new_ids), torch.tensor(new_ids)))
    tok.padding_side = "right"
    return rows


def completion_logprob_ids(model, gen_ids, prompt_ids):
    """Sum of token logprobs of generated ids, given the unpadded prompt ids."""
    if gen_ids is None or gen_ids.numel() == 0:
        return None
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if gen_ids.dim() == 1:
        gen_ids = gen_ids.unsqueeze(0)
    full = torch.cat([prompt_ids.to(gen_ids.device), gen_ids.to(prompt_ids.device)], dim=1).to(model.device)
    plen = prompt_ids.shape[1]
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    outputs = model(input_ids=full)
    logits = outputs.logits[:, plen - 1 : -1, :]
    target = full[:, plen:]
    n = min(logits.shape[1], target.shape[1])
    logits = logits[:, :n, :]
    target = target[:, :n]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return tok_lp.sum()


def row_from_gen(rel, src, text, dt, nnew, mx, tag):
    hit = nnew >= mx - 1
    scored = score_pair(src, text, hit_token_cap=hit)
    punct = punct_stats(scored["hyp_xml"]) if scored["well_formed"] else {"text_available": False}
    keep_xml = scored["format_broken"] or tag in ("baseline", "after", "test")
    return {
        "path": rel,
        "tag": tag,
        "bytes": len(src.encode("utf-8")),
        "latency_s": round(dt, 3),
        "new_tokens": nnew,
        "max_new": mx,
        "hit_token_cap": hit,
        "well_formed": scored["well_formed"],
        "skeleton_match": scored["skeleton_match"],
        "order_sensitive": True,
        "dtd_valid": scored["dtd_valid"],
        "dtd_errors": scored["dtd_errors"],
        "reward": scored["reward"],
        "lcs": scored["lcs"],
        "attr_recall": scored["attr_recall"],
        "reasons": scored["reasons"],
        "primary_reason": scored["primary_reason"],
        "truncated": scored["truncated"],
        "format_broken": scored["format_broken"],
        "error_tail": scored["error_tail"],
        "punct": punct,
        "hyp_xml": scored["hyp_xml"] if keep_xml else "",
    }


def summarize(rows):
    lats = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    p50 = statistics.median(lats) if lats else None
    hist = {}
    for r in rows:
        if not r.get("format_broken"):
            continue
        for reason in r.get("reasons") or ["unknown"]:
            hist[reason] = hist.get(reason, 0) + 1
    primary = {}
    for r in rows:
        if r.get("format_broken"):
            k = r.get("primary_reason") or "unknown"
            primary[k] = primary.get(k, 0) + 1
    return {
        "n": len(rows),
        "format_broken": sum(1 for r in rows if r.get("format_broken")),
        "schema_invalid_topics": sum(1 for r in rows if not (r["well_formed"] and r["dtd_valid"])),
        "schema_error_total": int(sum((r["dtd_errors"] or 0) for r in rows if r.get("dtd_errors"))),
        "skeleton_mismatch_topics": sum(1 for r in rows if not r["skeleton_match"]),
        "sibling_reorder_topics": sum(1 for r in rows if "sibling_reorder" in (r.get("reasons") or [])),
        "truncated_topics": sum(1 for r in rows if r.get("truncated")),
        "valid_and_skeleton_ok": sum(1 for r in rows if not r.get("format_broken")),
        "mean_reward": round(sum(r["reward"] for r in rows) / max(1, len(rows)), 4),
        "latency_p50_s": None if p50 is None else round(p50, 3),
        "latency_mean_s": None if not lats else round(sum(lats) / len(lats), 3),
        "latency_device": DEVICE_LABEL,
        "decoding": "greedy",
        "reason_histogram_multilabel": hist,
        "primary_reason_histogram": primary,
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
    fallback = sorted(s for s in suffixes if "proj" in s or s in ("in_proj_qkv", "in_proj_z", "out_proj"))
    return fallback[:8]


def kernel_status():
    found = {}
    for name in ("causal_conv1d", "flash_linear_attention", "fla"):
        try:
            __import__(name if name != "flash_linear_attention" else "fla")
            found[name] = True
        except Exception as e:
            found[name] = f"{type(e).__name__}: {e}"
    return found


def main():
    log = log_factory()
    log("start", model=MODEL_ID, scan_wall_s=SCAN_WALL_S, train_wall_s=TRAIN_WALL_S,
        target_broken=TARGET_BROKEN, system=SYSTEM, kernels=kernel_status(),
        source_commit=CANDIDATES.get("source_commit"), n_candidates=CANDIDATES.get("n_candidates"))
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
        vram_gb=round(torch.cuda.memory_allocated() / 1e9, 2),
        gpu=torch.cuda.get_device_name(0))

    # Zero-init LoRA before the scan so baseline and the update share one module.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    targets = pick_lora_targets(model)
    log("lora_targets", targets=targets)
    if not targets:
        note("no Linear targets found for LoRA")
        raise SystemExit(2)
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.eval()
    model.config.use_cache = True

    # smoke
    smoke_rel = CANDIDATES["candidates"][0]
    smoke_src = load_src(smoke_rel)
    smoke_prompt = format_prompt(tok, smoke_src[:500])
    try:
        rows = generate_batch(model, tok, [smoke_prompt], 48, sample=False)
        log("smoke", seconds=round(rows[0][1], 2), new_tokens=rows[0][3], head=rows[0][0][:160])
    except Exception as e:
        note(f"smoke generation failed: {type(e).__name__}: {e}")
        log("smoke_fail", error=traceback.format_exc()[-2000:])
        raise

    scan_start = time.perf_counter()
    broken = []
    passed = 0
    scanned = 0
    skipped_over_cap = 0
    phase_caps = [640, 896]
    seen = set()
    scan_rows_broken = []
    by_path = {}
    stop_reason = "pool_exhausted"

    for cap in phase_caps:
        if len(broken) >= TARGET_BROKEN:
            break
        if time.perf_counter() - scan_start > SCAN_WALL_S:
            stop_reason = "scan_wall"
            break
        log("scan_phase", cap=cap, broken=len(broken), scanned=scanned)
        for rel in CANDIDATES["candidates"]:
            if rel in seen:
                continue
            if len(broken) >= TARGET_BROKEN:
                stop_reason = "found_target"
                break
            if time.perf_counter() - scan_start > SCAN_WALL_S:
                stop_reason = "scan_wall"
                note(f"scan wall hit with {len(broken)} broken after {scanned} translations")
                break
            src = load_src(rel)
            if need_tokens(src) > cap:
                continue
            seen.add(rel)
            mx = max_new_for(src, cap)
            prompt = format_prompt(tok, src)
            # prompt ids length gate: if the prompt itself is huge, skip
            plen_est = len(tok(prompt, add_special_tokens=False)["input_ids"])
            if plen_est > 1800:
                skipped_over_cap += 1
                note(f"skip long prompt {rel} tokens={plen_est}")
                continue
            try:
                gens = generate_batch(model, tok, [prompt], mx, sample=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                note(f"OOM scan {rel}")
                continue
            text, dt, _plen, nnew, _ids = gens[0]
            # batch latency is the whole call; one row so it is this topic
            row = row_from_gen(rel, src, text, dt, nnew, mx, "scan")
            scanned += 1
            by_path[rel] = row
            if row["format_broken"]:
                broken.append(rel)
                scan_rows_broken.append({k: v for k, v in row.items()})
            else:
                passed += 1
            if scanned % 10 == 0 or row["format_broken"]:
                print(
                    f"scan {scanned} broken={len(broken)} pass={passed} {rel} "
                    f"lat={dt:.1f}s tok={nnew}/{mx} broken={row['format_broken']} "
                    f"why={row['primary_reason']}",
                    flush=True,
                )
            if scanned % 20 == 0:
                (OUT / "scan_progress.json").write_text(json.dumps({
                    "scanned": scanned, "passed": passed, "broken": len(broken),
                    "skipped_over_cap_so_far": skipped_over_cap,
                    "elapsed_s": round(time.perf_counter() - scan_start, 1),
                    "last": rel, "primary_running": summarize(list(by_path.values()))["primary_reason_histogram"],
                }, indent=2))
        else:
            continue
        break

    # files never translated because they never fit a cap we reached
    never = [rel for rel in CANDIDATES["candidates"] if rel not in seen]
    # among never, those that fit the last cap were simply not reached
    not_reached = 0
    too_big = 0
    last_cap = phase_caps[-1] if stop_reason != "found_target" else phase_caps[0]
    for rel in never:
        src_len = (DITA / rel).stat().st_size
        # approximate, bytes ~ chars for this corpus
        if int(src_len / 2.8 + 48) > phase_caps[-1]:
            too_big += 1
        else:
            not_reached += 1
    log("scan_done", stop_reason=stop_reason, scanned=scanned, passed=passed,
        broken=len(broken), skipped_over_cap=skipped_over_cap,
        not_reached=not_reached, too_big_for_896=too_big,
        elapsed_s=round(time.perf_counter() - scan_start, 1))

    if len(broken) < 20:
        note(f"only {len(broken)} broken topics; split will be smaller than 10/10/rest if needed")

    rng = random.Random(SEED)
    order = broken[:]
    rng.shuffle(order)
    n_test = 10 if len(order) >= 20 else max(1, len(order) // 5)
    n_val = 10 if len(order) >= 30 else max(0, min(10, (len(order) - n_test) // 2))
    if len(order) >= 30:
        n_test, n_val = 10, 10
    test = order[:n_test]
    val = order[n_test:n_test + n_val]
    train = order[n_test + n_val:]
    manifest = {
        "language_pair": "en-fr",
        "source_repo": "https://github.com/oxygenxml/userguide",
        "source_commit": CANDIDATES.get("source_commit"),
        "seed": SEED,
        "scan_order": "sorted relative path, files that cannot finish under the phase cap are deferred",
        "phase_caps": phase_caps,
        "order_sensitive_skeleton": True,
        "sibling_order": "structural signature (tag, attributes in document order, child signatures). A permutation is sibling_reorder. Two same-shape siblings that differ only in translated text are not a format break.",
        "target_broken": TARGET_BROKEN,
        "scanned_translations": scanned,
        "passed_kept_format": passed,
        "skipped_over_cap": skipped_over_cap,
        "not_reached": not_reached,
        "too_big_for_max_cap": too_big,
        "n_broken": len(broken),
        "stop_reason": stop_reason,
        "n_test": len(test),
        "n_val": len(val),
        "n_train": len(train),
        "test": test,
        "val": val,
        "train": train,
        "broken_in_scan_order": broken,
        "system_prompt": SYSTEM,
    }
    (ROOT / "split" / "manifest_fr.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "scan_broken.json").write_text(json.dumps(scan_rows_broken, indent=2), encoding="utf-8")

    base_rows = []
    for rel in test:
        row = dict(by_path[rel])
        row["tag"] = "baseline"
        if not row.get("hyp_xml"):
            # should already be kept because broken
            pass
        row["punct"] = row.get("punct") or {}
        base_rows.append(row)
    base_summary = summarize(base_rows)
    (OUT / "baseline_test.json").write_text(json.dumps({"summary": base_summary, "rows": base_rows}, indent=2, ensure_ascii=False))
    log("baseline_from_scan", **{k: v for k, v in base_summary.items() if "hist" not in k})

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    train_start = time.perf_counter()
    updates = 0
    skipped_flat = 0
    train_rows = []
    for step, rel in enumerate(train):
        if time.perf_counter() - train_start > TRAIN_WALL_S:
            note(f"train wall after {step} topics; {updates} updates")
            break
        src = load_src(rel)
        mx = max_new_for(src, 896)
        prompt = format_prompt(tok, src)
        prompt_ids = tok(prompt, return_tensors="pt").input_ids
        samples = []
        oom = False
        for k in range(2):
            try:
                gens = generate_batch(model, tok, [prompt], mx, sample=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                note(f"OOM sampling {rel} k={k}")
                oom = True
                break
            text, dt, _plen, nnew, ids = gens[0]
            hit = nnew >= mx - 1
            sc = score_pair(src, text, hit_token_cap=hit)
            samples.append((text, sc, dt, nnew, ids))
            print(
                f"sample {step}/{len(train)} {rel} k={k} r={sc['reward']:.3f} "
                f"why={sc['primary_reason']} dtd={sc['dtd_valid']} sk={sc['skeleton_match']} {dt:.1f}s",
                flush=True,
            )
        if oom or len(samples) < 2:
            continue
        rewards = [s[1]["reward"] for s in samples]
        if max(rewards) - min(rewards) < 1e-4:
            skipped_flat += 1
            train_rows.append({"path": rel, "rewards": rewards, "reasons": [s[1]["reasons"] for s in samples], "updated": False})
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
                lp = completion_logprob_ids(model, ids, prompt_ids)
                if lp is None:
                    note(f"logprob empty {rel}")
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
            note(f"OOM on backward {rel}; skipped update")
            continue
        train_rows.append({
            "path": rel, "rewards": rewards, "advs": advs, "updated": True, "loss": loss_acc,
            "reasons": [s[1]["reasons"] for s in samples],
        })
        log("rl_step", step=step, path=rel, rewards=rewards, updates=updates,
            vram_gb=round(torch.cuda.memory_allocated() / 1e9, 2))

    log("rl_done", updates=updates, skipped_flat=skipped_flat,
        topics_attempted=len(train_rows), train_s=round(time.perf_counter() - train_start, 1))
    adapter_dir = OUT / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    model.eval()
    model.config.use_cache = True
    after_rows = []
    for rel in test:
        src = load_src(rel)
        mx = max_new_for(src, 896)
        prompt = format_prompt(tok, src)
        try:
            gens = generate_batch(model, tok, [prompt], mx, sample=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            note(f"OOM after {rel}")
            continue
        text, dt, _plen, nnew, _ids = gens[0]
        row = row_from_gen(rel, src, text, dt, nnew, mx, "after")
        after_rows.append(row)
        print(f"after {rel} lat={dt:.1f}s broken={row['format_broken']} why={row['primary_reason']}", flush=True)
    after_summary = summarize(after_rows)
    (OUT / "after_test.json").write_text(json.dumps({"summary": after_summary, "rows": after_rows}, indent=2, ensure_ascii=False))
    summary = {
        "model": MODEL_ID,
        "quant": "bitsandbytes NF4 QLoRA (4-bit base)",
        "language_pair": "en-fr",
        "method": "Scan greedy until format breaks. GRPO-style group size 2 on broken train topics only. Advantage (r-mean)/std. Dense reward: 0.5 skeleton LCS + 0.2 attribute recall + 0.15 well-formed + 0.15 DTD, minus 0.3 if truncated. Perfect format scores 1.",
        "reward_order_sensitive": True,
        "system_prompt": SYSTEM,
        "lr": LR,
        "lora_r": 8,
        "lora_targets": targets,
        "updates": updates,
        "skipped_flat_reward": skipped_flat,
        "train_topics_attempted": len(train_rows),
        "train_topics_available": len(train),
        "scan": {
            "scanned": scanned,
            "passed": passed,
            "broken": len(broken),
            "stop_reason": stop_reason,
            "skipped_over_cap": skipped_over_cap,
            "not_reached": not_reached,
            "too_big_for_max_cap": too_big,
            "elapsed_s": round(time.perf_counter() - scan_start, 1),
        },
        "baseline": base_summary,
        "after": after_summary,
        "surprises": SURPRISES,
        "kernels": kernel_status(),
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "train_trace.json").write_text(json.dumps(train_rows, indent=2))
    log("finished", baseline_broken=base_summary.get("format_broken"), after_broken=after_summary.get("format_broken"),
        updates=updates, skipped_flat=skipped_flat)
    print(json.dumps({k: summary[k] for k in ("scan", "updates", "skipped_flat_reward", "baseline", "after")}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "crash.txt").write_text(traceback.format_exc())
        raise
