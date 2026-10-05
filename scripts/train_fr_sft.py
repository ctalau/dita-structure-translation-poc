#!/usr/bin/env python3
"""SFT a new Qwen/Qwen3.5-4B adapter on the French typography split.

This is not a continuation of results/fr_punct or results/fr_nbsp_sft2.
Rank 8, alpha 16, dropout 0, NF4, AdamW, cosine with 5% warmup,
effective batch 16, sequence cap 1536, at most 1 epoch, completion-token loss,
thinking off.

The 5% warmup is configs/fr_style.yaml sft.warmup_ratio. This script passes
warmup_steps, never warmup_ratio. TrainingArguments on the RunPod pytorch 2.4
image has no warmup_ratio. A 5-step pilot (--max-steps 5) uses warmup_steps 0.

Budget contract, not executed by CPU preparation: one community RTX 3090, $5
total. SFT allocation $1.80. The watchdog can terminate the pod. This file
does not call RunPod unless training is actually running and the stage cap is hit.

CPU preparation refuses to start. Pass --run on the pod.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from budget_watchdog import BudgetWatchdog, credentials_from_env  # noqa: E402
from fr_lora_targets import selection  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "fr_style.yaml"
PILOT_STEPS = 5


def positive_flag(name: str) -> int | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise SystemExit(f"{name} needs a value")
    try:
        value = int(sys.argv[index + 1])
    except ValueError as exc:
        raise SystemExit(f"{name} needs an integer") from exc
    if value < 1:
        raise SystemExit(f"{name} must be positive")
    return value


def sft_optimizer_steps(
    n_examples: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:
    """Optimizer updates for one process when max_steps is unset.

    The dataloader length is ceil(n / per_device_batch_size). Updates per epoch
    are that length floor-divided by gradient_accumulation_steps, at least 1.
    """
    if min(n_examples, per_device_batch_size, gradient_accumulation_steps, epochs) < 1:
        raise ValueError("n_examples, batch, accumulation, and epochs must be positive")
    batches = math.ceil(n_examples / per_device_batch_size)
    updates = max(batches // gradient_accumulation_steps, 1)
    return updates * epochs


def grpo_optimizer_steps(
    n_prompts: int,
    per_device_batch_size: int,
    num_generations: int,
    gradient_accumulation_steps: int,
    epochs: int,
    num_iterations: int = 1,
    num_processes: int = 1,
) -> int:
    """Optimizer updates for TRL 1.14.1 GRPOTrainer.

    Unset steps_per_generation becomes gradient_accumulation_steps. The sampler
    drops a trailing prompt chunk smaller than its batch. num_processes is 1
    for the single GPU in the budget contract.
    """
    if min(
        n_prompts,
        per_device_batch_size,
        num_generations,
        gradient_accumulation_steps,
        epochs,
        num_iterations,
        num_processes,
    ) < 1:
        raise ValueError("GRPO step inputs must be positive")
    if per_device_batch_size % num_generations != 0:
        raise ValueError("per_device_train_batch_size must be divisible by num_generations")
    steps_per_generation = gradient_accumulation_steps
    generation_batch_size = per_device_batch_size * num_processes * steps_per_generation
    sampler_batch = generation_batch_size // num_generations
    repeat_count = num_iterations * steps_per_generation
    kept_prompts = (n_prompts // sampler_batch) * sampler_batch
    if kept_prompts < 1:
        raise ValueError("not enough prompts for one GRPO generation batch")
    sampler_len = kept_prompts * num_generations * repeat_count
    dataloader_batch = per_device_batch_size * num_processes * steps_per_generation
    n_batches = sampler_len // dataloader_batch
    updates = max(n_batches // gradient_accumulation_steps, 1)
    return updates * epochs


def warmup_steps_for(num_training_steps: int, warmup_ratio: float) -> int:
    """Linear warmup covering warmup_ratio of the optimizer steps.

    Callers pass the result as warmup_steps. A 5-step pilot returns 0.
    """
    if num_training_steps < 1:
        raise ValueError("num_training_steps must be positive")
    if warmup_ratio < 0:
        raise ValueError("warmup_ratio must be non-negative")
    if num_training_steps == PILOT_STEPS:
        return 0
    return math.ceil(num_training_steps * warmup_ratio)


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_train_rows(config: dict) -> list[dict]:
    train_path = ROOT / config["data"]["train_jsonl"]
    sealed_path = ROOT / config["data"]["sealed_jsonl"]
    train = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    sealed = [json.loads(line) for line in sealed_path.read_text().splitlines() if line.strip()]
    sealed_ids = {row["id"] for row in sealed}
    overlap = sorted(sealed_ids & {row["id"] for row in train})
    if overlap:
        raise SystemExit(f"refusing to train: sealed ids in train: {overlap[:5]}")
    sealed_targets = {row["target_xml"] for row in sealed}
    leaked = [row["id"] for row in train if row["target_xml"] in sealed_targets]
    if leaked:
        raise SystemExit(f"refusing to train: sealed completions in train: {leaked[:5]}")
    return train


def build_messages(config: dict, row: dict) -> tuple[list[dict], str]:
    if row["task"] == "translate":
        user = row["source_xml"]
    elif row["task"] == "repair":
        user = "Repair French typography. Output the XML document only.\n" + row["source_xml"]
    elif row["task"] == "noop":
        user = "The French typography is already correct. Output the XML document only.\n" + row["source_xml"]
    else:
        raise ValueError(row["task"])
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": user},
    ]
    return messages, row["target_xml"]


def completion_labels(prompt_ids: list[int], completion_ids: list[int], max_length: int):
    """Loss on completion tokens only. Prompt positions are -100."""
    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    if len(prompt_ids) >= max_length:
        return None
    return input_ids, labels


def main() -> int:
    config = load_config()
    rows = load_train_rows(config)
    chosen = selection()
    sft = config["sft"]
    max_steps = positive_flag("--max-steps")
    if max_steps is None:
        num_training_steps = sft_optimizer_steps(
            len(rows),
            sft["per_device_batch_size"],
            sft["gradient_accumulation_steps"],
            sft["max_epochs"],
        )
    else:
        num_training_steps = max_steps
    warm = warmup_steps_for(num_training_steps, sft["warmup_ratio"])
    print(json.dumps({
        "started": False,
        "reason": "CPU preparation does not start training. Pass --run on the GPU pod.",
        "n_train": len(rows),
        "new_adapter_from": config["model_id"],
        "revision": config["model_revision"],
        "continue_from_old_lora": config["continue_from_old_lora"],
        "selected_targets": chosen["selected_targets"],
        "full_set_fits_24gib_static_estimate": chosen["full_set_fits_24gib_static_estimate"],
        "sft": sft,
        "num_training_steps": num_training_steps,
        "warmup_steps": warm,
        "warmup_argument": "warmup_steps",
        "budget_sft_usd": config["budget"]["allocations_usd"]["sft"],
    }, indent=2))
    if "--run" not in sys.argv:
        return 0
    # Imports stay inside the run path so CPU preparation does not need torch.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise SystemExit("refusing to start SFT: no CUDA device")
    if config["continue_from_old_lora"]:
        raise SystemExit("refusing to continue an old LoRA")

    class Rows(Dataset):
        def __init__(self, records, tokenizer, system_config):
            self.records = records
            self.tokenizer = tokenizer
            self.config = system_config

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            messages, completion = build_messages(self.config, self.records[index])
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            if prompt.rstrip().endswith("<think>"):
                prompt = prompt.rstrip()[:-7] + "<think>\n\n</think>\n\n"
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = self.tokenizer(completion + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            packed = completion_labels(prompt_ids, completion_ids, self.config["sft"]["max_seq_length"])
            if packed is None:
                input_ids = prompt_ids[: self.config["sft"]["max_seq_length"]]
                labels = [-100] * len(input_ids)
            else:
                input_ids, labels = packed
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }

    def collate(batch):
        width = max(len(row["input_ids"]) for row in batch)
        pad_id = tokenizer.pad_token_id
        input_ids, labels, mask = [], [], []
        for row in batch:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            mask.append(row["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=chosen["selected_regex"],
    )
    model = get_peft_model(model, lora)
    out_dir = ROOT / "results" / "fr_style_sft_adapter"
    argument_values = {
        "output_dir": str(out_dir),
        "num_train_epochs": sft["max_epochs"],
        "per_device_train_batch_size": sft["per_device_batch_size"],
        "gradient_accumulation_steps": sft["gradient_accumulation_steps"],
        "learning_rate": sft["learning_rate"],
        "lr_scheduler_type": sft["scheduler"],
        "warmup_steps": warm,
        "optim": "adamw_torch",
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
    }
    if max_steps is not None:
        argument_values["max_steps"] = max_steps
    args = TrainingArguments(**argument_values)
    pod_id, api_key = credentials_from_env()
    watchdog = BudgetWatchdog(stage="sft", pod_id=pod_id, api_key=api_key)

    class Cap(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            watchdog.poll()
            return super().training_step(model, inputs, num_items_in_batch)

    trainer = Cap(
        model=model,
        args=args,
        train_dataset=Rows(rows, tokenizer, config),
        data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
