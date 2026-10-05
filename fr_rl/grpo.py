"""GRPO with LoRA for English-to-French DITA segment translation. One GPU.

Rollouts: vLLM, with the current LoRA hot-loaded after every update.
Update:   HF transformers + PEFT, one gradient step per batch, so the
          policy that sampled is the policy being updated (ratio = 1, no
          clipping needed). No KL term; the chrF-to-reference part of the
          reward is what keeps the translation anchored.
Reward:   fr_rl.checks.reward (structure gate, chrF, pattern penalties).
Advantage: group-normalised (r - mean) / std per prompt; tied groups skipped.
Loss:     token-level mean over all completion tokens in the batch.

Dev evaluation (greedy, whole dev topics) runs at step 0 and every
--eval-every steps; each evaluated adapter is kept on disk.

  python3 -m fr_rl.grpo --out /workspace/run --steps 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path

import torch

from fr_rl.checks import check, reward
from fr_rl.prompt import messages

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ROOT = Path(__file__).resolve().parents[1]


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def log(fh, **kw):
    kw["t"] = round(time.time(), 1)
    line = json.dumps(kw, ensure_ascii=False)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def dev_eval(llm, lora, dev, out_path):
    from vllm import SamplingParams
    from fr_rl.score import score_file
    sp = SamplingParams(temperature=0.0, max_tokens=4096)
    outs = llm.chat([messages(d["src"]) for d in dev], sp, lora_request=lora, use_tqdm=False)
    with open(out_path, "w") as f:
        for d, o in zip(dev, outs):
            c = o.outputs[0]
            f.write(json.dumps({"id": d["id"], "split": "dev", "hyp": c.text,
                                "n_out_tokens": len(c.token_ids), "finish_reason": c.finish_reason},
                               ensure_ascii=False) + "\n")
    return score_file(str(out_path), "dev")["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vllm-mem", type=float, default=0.40, help="vLLM gpu_memory_utilization")
    ap.add_argument("--sleep", action="store_true",
                    help="offload vLLM weights to CPU during each update (24 GB cards)")
    ap.add_argument("--wall-s", type=float, default=1e9, help="stop after this many seconds")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fh = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    train = jl(ROOT / "split/fr_rl/train_segments.jsonl")
    dev = jl(ROOT / "split/fr_rl/dev_topics.jsonl")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(MODEL, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=args.vllm_mem,
              enable_sleep_mode=args.sleep, enable_lora=True, max_lora_rank=args.rank, max_loras=1, max_cpu_loras=2,
              enable_prefix_caching=True, seed=args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    log(fh, event="setup", args=vars(args), trainable=sum(p.numel() for p in params),
        n_train=len(train), n_dev=len(dev), setup_s=round(time.time() - t0, 1),
        gpu=torch.cuda.get_device_name(0))

    lora = None
    dev_hist = []

    def do_eval(step):
        te = time.time()
        s = dev_eval(llm, lora, dev, out / f"dev_step{step}.jsonl")
        if lora is not None:
            shutil.copytree(lora.lora_path, out / f"ckpt_step{step}", dirs_exist_ok=True)
        dev_hist.append({"step": step, **s})
        log(fh, event="dev", step=step, seconds=round(time.time() - te, 1), **s)

    do_eval(0)
    order = []
    done = 0
    for step in range(1, args.steps + 1):
        if time.time() - t0 > args.wall_s:
            log(fh, event="wall_stop", step=step)
            break
        if len(order) < args.prompts:
            perm = list(range(len(train)))
            rng.shuffle(perm)
            order += perm
        batch = [train[i] for i in order[:args.prompts]]
        order = order[args.prompts:]

        tg = time.time()
        prompts = [tok.apply_chat_template(messages(b["src"]), add_generation_prompt=True, tokenize=False)
                   for b in batch]
        sp = SamplingParams(n=args.group, temperature=args.temperature, top_p=1.0,
                            max_tokens=args.max_new, seed=args.seed * 100000 + step)
        outs = llm.generate(prompts, sp, lora_request=lora, use_tqdm=False)
        tg = time.time() - tg

        ts = time.time()
        samples, stats = [], {"r": [], "chrf": [], "struct": [], "typo": [], "calque": [], "title": [], "len": [], "cap": 0}
        n_spread = 0
        for b, o in zip(batch, outs):
            rs = []
            group = []
            for c in o.outputs:
                rep = check(b["src"], c.text, b["ref"])
                r = reward(rep)
                if c.finish_reason == "length":
                    r = -1.0
                    stats["cap"] += 1
                rs.append(r)
                group.append(c)
                stats["r"].append(r)
                stats["chrf"].append(rep.chrf or 0.0)
                stats["struct"].append(rep.structure_ok)
                stats["typo"].append(rep.n_typo)
                stats["calque"].append(rep.n_calque)
                stats["title"].append(rep.titlecase)
                stats["len"].append(len(c.token_ids))
            mu, sd = statistics.mean(rs), statistics.pstdev(rs)
            if sd < 0.01:
                continue
            n_spread += 1
            pids = list(o.prompt_token_ids)
            for c, r in zip(group, rs):
                samples.append((pids, list(c.token_ids), (r - mu) / (sd + 1e-4)))
        ts = time.time() - ts

        tt = time.time()
        loss_sum, gnorm = 0.0, 0.0
        if samples:
            if args.sleep:
                llm.sleep(level=1)
                torch.cuda.empty_cache()
            model.train()
            n_tok = sum(len(c) for _, c, _ in samples)
            samples.sort(key=lambda s: len(s[0]) + len(s[1]))
            for i in range(0, len(samples), args.micro):
                mb = samples[i:i + args.micro]
                L = max(len(p) + len(c) for p, c, _ in mb)
                C = max(len(c) for _, c, _ in mb)
                ids = torch.full((len(mb), L), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(mb), L), dtype=torch.long)
                cmask = torch.zeros((len(mb), C), dtype=torch.float32)
                adv = torch.tensor([a for _, _, a in mb], dtype=torch.float32)
                for j, (p, c, _) in enumerate(mb):  # left padding: completions end-aligned
                    seq = p + c
                    ids[j, L - len(seq):] = torch.tensor(seq)
                    att[j, L - len(seq):] = 1
                    cmask[j, C - len(c):] = 1
                pos = (att.cumsum(-1) - 1).clamp(min=0)
                ids, att, pos, cmask, adv = ids.cuda(), att.cuda(), pos.cuda(), cmask.cuda(), adv.cuda()
                logits = model(input_ids=ids, attention_mask=att, position_ids=pos,
                               logits_to_keep=C + 1).logits[:, :-1].float()
                tgt = ids[:, -C:]
                logp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                loss = -(adv[:, None] * logp * cmask).sum() / n_tok
                loss.backward()
                loss_sum += loss.item()
                del logits, logp, loss
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            path = out / "lora_cur" / f"s{step}"
            model.save_pretrained(path)
            if lora is not None and "lora_cur" in lora.lora_path:
                shutil.rmtree(lora.lora_path, ignore_errors=True)
            lora = LoRARequest(f"s{step}", step, str(path))
            if args.sleep:
                torch.cuda.empty_cache()
                llm.wake_up()
        tt = time.time() - tt

        m = lambda xs: round(sum(xs) / max(1, len(xs)), 4)
        log(fh, event="step", step=step, reward=m(stats["r"]), chrf=m(stats["chrf"]),
            struct_ok=m(stats["struct"]), typo=m(stats["typo"]), calque=m(stats["calque"]),
            titlecase=m(stats["title"]), comp_len=m(stats["len"]), hit_cap=stats["cap"],
            groups_with_spread=n_spread, n_samples_trained=len(samples), loss=round(loss_sum, 5),
            grad_norm=round(gnorm, 4), gen_s=round(tg, 1), score_s=round(ts, 1), train_s=round(tt, 1),
            elapsed_s=round(time.time() - t0, 1),
            peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        done = step
        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step)

    if dev_hist[-1]["step"] != done:
        do_eval(done)
    best = max(dev_hist, key=lambda d: d["mean_reward"])
    json.dump({"dev_history": dev_hist, "best_step_by_dev_mean_reward": best["step"],
               "elapsed_s": round(time.time() - t0, 1)}, open(out / "summary.json", "w"), indent=1)
    log(fh, event="done", best_step=best["step"], elapsed_s=round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
