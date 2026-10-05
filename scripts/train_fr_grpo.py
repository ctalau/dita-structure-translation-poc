#!/usr/bin/env python3
"""Conditional GRPO on the French typography gated reward.

Pinned trainer: trl==1.14.1 GRPOTrainer. loss_type dr_grpo, scale_rewards false,
epsilon 0.2, beta 0.02, group size 4, learning rate 5e-6. The KL reference is a
frozen copy of the SFT adapter loaded under the PEFT name "ref". TRL 1.14.1
uses that adapter when it is present. With no "ref" adapter, disabling PEFT
would score the KL against the base model, which this script refuses.

Warmup is warmup_steps from sft.warmup_ratio in configs/fr_style.yaml. This
script does not pass warmup_ratio. A 5-step pilot (--max-steps 5) uses
warmup_steps 0.

Budget: conditional GRPO $1.30 of the $5 community RTX 3090 contract.
The watchdog can terminate the pod. CPU preparation does not call RunPod
and does not start training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from budget_watchdog import BudgetWatchdog, credentials_from_env  # noqa: E402
from dita_reward import gated_scalar  # noqa: E402
from fr_lora_targets import selection  # noqa: E402
from train_fr_sft import grpo_optimizer_steps, positive_flag, warmup_steps_for  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "fr_style.yaml"
TRL_PIN = "1.14.1"
SFT_DIR = ROOT / "results" / "fr_style_sft_adapter"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def grpo_should_run(sft_dev_mean: float, budget_remaining_usd: float, config: dict) -> bool:
    return sft_dev_mean < 0.95 and budget_remaining_usd >= config["budget"]["allocations_usd"]["grpo"]


def reward_func(completions, source_xml, **kwargs):
    """TRL passes completion strings or message lists, plus the source_xml column."""
    scores = []
    for completion, source in zip(completions, source_xml):
        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = completion
        scores.append(gated_scalar(source, text))
    return scores


def assert_trl_pin():
    import trl
    from trl import GRPOConfig

    version = getattr(trl, "__version__", "")
    if version != TRL_PIN:
        raise SystemExit(f"TRL pin failed: expected {TRL_PIN}, imported {version}")
    probe = GRPOConfig(
        output_dir="/tmp/fr-style-grpo-probe",
        loss_type="dr_grpo",
        scale_rewards=False,
        epsilon=0.2,
        beta=0.02,
        num_generations=4,
        learning_rate=5e-6,
        max_completion_length=896,
    )
    if probe.loss_type != "dr_grpo":
        raise SystemExit("TRL pin failed: loss_type dr_grpo was not accepted")
    if probe.scale_rewards not in (False, "none"):
        raise SystemExit(f"TRL pin failed: scale_rewards={probe.scale_rewards}")
    return version


def _translate_count(config: dict) -> int:
    train_path = ROOT / config["data"]["train_jsonl"]
    count = 0
    for line in train_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["task"] == "translate":
            count += 1
    return count


def main() -> int:
    config = load_config()
    grpo = config["grpo"]
    max_steps = positive_flag("--max-steps")
    n_prompts = _translate_count(config)
    if max_steps is None:
        num_training_steps = grpo_optimizer_steps(
            n_prompts,
            per_device_batch_size=4,
            num_generations=grpo["num_generations"],
            gradient_accumulation_steps=1,
            epochs=1,
        )
    else:
        num_training_steps = max_steps
    warm = warmup_steps_for(num_training_steps, config["sft"]["warmup_ratio"])
    print(json.dumps({
        "started": False,
        "reason": "CPU preparation does not start training. Pass --run on the GPU pod.",
        "trl_pin": TRL_PIN,
        "trl_imported": False,
        "grpo": grpo,
        "kl_reference": grpo["kl_reference"],
        "n_prompts": n_prompts,
        "num_training_steps": num_training_steps,
        "warmup_steps": warm,
        "warmup_argument": "warmup_steps",
        "budget_grpo_usd": config["budget"]["allocations_usd"]["grpo"],
        "targets": selection()["selected_targets"],
    }, indent=2))
    if "--run" not in sys.argv:
        return 0
    try:
        version = assert_trl_pin()
    except Exception as exc:
        raise SystemExit(f"TRL pin failed: {type(exc).__name__}: {exc}") from exc
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise SystemExit("refusing to start GRPO: no CUDA device")
    if not (SFT_DIR / "adapter_config.json").is_file():
        raise SystemExit(f"refusing to start GRPO: missing new SFT adapter at {SFT_DIR}")
    dev_mean = float(sys.argv[sys.argv.index("--sft-dev-mean") + 1]) if "--sft-dev-mean" in sys.argv else 0.0
    remaining = float(sys.argv[sys.argv.index("--budget-remaining") + 1]) if "--budget-remaining" in sys.argv else 1.30
    if not grpo_should_run(dev_mean, remaining, config):
        print("GRPO condition is false. Not starting.")
        return 0

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["model_revision"])
    base = AutoModelForImageTextToText.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, str(SFT_DIR), is_trainable=True, adapter_name="default")
    model.load_adapter(str(SFT_DIR), adapter_name="ref")
    model.set_adapter("default")
    for name, param in model.named_parameters():
        if ".ref." in name:
            param.requires_grad = False
    if "ref" not in model.peft_config:
        raise SystemExit("KL reference is missing. Refusing to fall back to the base model.")

    rows = []
    train_path = ROOT / config["data"]["train_jsonl"]
    sealed_ids = {
        json.loads(line)["id"]
        for line in (ROOT / config["data"]["sealed_jsonl"]).read_text().splitlines()
        if line.strip()
    }
    for line in train_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in sealed_ids:
            raise SystemExit(f"sealed id in GRPO train: {row['id']}")
        if row["task"] != "translate":
            continue
        messages = [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": row["source_xml"]},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prompt.rstrip().endswith("<think>"):
            prompt = prompt.rstrip()[:-7] + "<think>\n\n</think>\n\n"
        rows.append({"prompt": prompt, "source_xml": row["source_xml"]})

    dataset = Dataset.from_list(rows)
    grpo_values = {
        "output_dir": str(ROOT / "results" / "fr_style_grpo"),
        "loss_type": "dr_grpo",
        "scale_rewards": False,
        "epsilon": 0.2,
        "beta": 0.02,
        "num_generations": 4,
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "warmup_steps": warm,
        "max_completion_length": grpo["max_completion_length"],
        "max_prompt_length": grpo["max_prompt_length"],
        "num_train_epochs": 1,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
    }
    if max_steps is not None:
        grpo_values["max_steps"] = max_steps
    grpo_config = GRPOConfig(**grpo_values)
    pod_id, api_key = credentials_from_env()
    watchdog = BudgetWatchdog(stage="grpo", pod_id=pod_id, api_key=api_key)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    if getattr(trainer, "ref_model", None) is not None:
        raise SystemExit("A separate base ref_model was created. Refusing to train against the base model.")
    watchdog.poll()
    trainer.train()
    model.save_pretrained(ROOT / "results" / "fr_style_grpo_adapter")
    print(json.dumps({"trl": version, "started": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
