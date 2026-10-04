"""CPU checks for the French typography plan. No GPU and no RunPod."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from budget_watchdog import (  # noqa: E402
    ALLOCATION_USD,
    HOURLY_USD,
    TOTAL_USD,
    BudgetWatchdog,
    allocation_total,
    terminate_runpod_pod,
)
from fr_lora_targets import selection  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "fr_style.yaml").read_text())


def test_budget_sums_to_five_dollars():
    assert allocation_total() == 5.00
    assert abs(sum(ALLOCATION_USD.values()) - TOTAL_USD) < 1e-9
    assert HOURLY_USD == 0.22
    allocations = CONFIG["budget"]["allocations_usd"]
    assert round(sum(allocations.values()), 2) == 5.00
    assert CONFIG["budget"]["total_usd"] == 5.00
    assert CONFIG["budget"]["do_not_call_runpod_from_cpu_prep"] is True
    assert CONFIG["budget"]["watchdog_can_terminate_pod"] is True


def test_lora_attention_matches_published_count_and_full_set_is_selected():
    chosen = selection()
    assert chosen["attention_only_trainable_params"] == 1_572_864
    assert chosen["attention_only_matches_published_1572864"] is True
    assert chosen["full_trainable_params"] == 16_232_448
    assert chosen["fallback_without_deltanet_trainable_params"] == 10_616_832
    assert chosen["full_set_fits_24gib_static_estimate"] is True
    assert chosen["selected_targets"] == (
        chosen["delta_net_projections"]
        + chosen["attention_projections"]
        + chosen["mlp_projections"]
    )
    assert "in_proj_qkv" in chosen["selected_targets"]
    assert "q_proj" in chosen["selected_targets"]
    assert "gate_proj" in chosen["selected_targets"]
    assert chosen["activation_allowance_is_not_a_measurement"] is True


def test_watchdog_delete_uses_the_injected_opener():
    calls = []

    def opener(request, timeout=None):
        calls.append((request.get_method(), request.full_url, request.get_header("Authorization")))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b""

        return Response()

    def terminate(pod_id, api_key):
        terminate_runpod_pod(pod_id, api_key, opener=opener)

    clock = {"t": 0.0}

    def now():
        return clock["t"]

    dog = BudgetWatchdog("sft", "pod123", "secret", clock=now, terminate=terminate, started_at=0.0)
    dog.poll()
    assert calls == []
    clock["t"] = (ALLOCATION_USD["sft"] / HOURLY_USD) * 3600.0
    try:
        dog.poll()
    except SystemExit as exc:
        assert "pod123" in str(exc)
    else:
        raise AssertionError("over-cap poll did not exit")
    assert calls == [("DELETE", "https://rest.runpod.io/v1/pods/pod123", "Bearer secret")]


def test_sft_and_grpo_contract():
    assert CONFIG["new_adapter"] is True
    assert CONFIG["continue_from_old_lora"] is False
    assert CONFIG["model_id"] == "Qwen/Qwen3.5-4B"
    assert CONFIG["model_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert CONFIG["lora_rank"] == 8
    assert CONFIG["lora_alpha"] == 16
    assert CONFIG["lora_dropout"] == 0.0
    sft = CONFIG["sft"]
    assert sft["learning_rate"] == 1.0e-4
    assert sft["optimizer"] == "adamw"
    assert sft["scheduler"] == "cosine"
    assert sft["warmup_ratio"] == 0.05
    assert sft["effective_batch_size"] == 16
    assert sft["per_device_batch_size"] * sft["gradient_accumulation_steps"] == 16
    assert sft["max_seq_length"] == 1536
    assert sft["max_epochs"] == 1
    assert sft["loss"] == "completion_tokens_only"
    grpo = CONFIG["grpo"]
    assert grpo["trl_pin"] == "1.14.1"
    assert grpo["loss_type"] == "dr_grpo"
    assert grpo["scale_rewards"] is False
    assert grpo["epsilon"] == 0.2
    assert grpo["beta"] == 0.02
    assert grpo["num_generations"] == 4
    assert grpo["learning_rate"] == 5.0e-6
    assert grpo["kl_reference"] == "frozen_sft_adapter"
    assert grpo["kl_reference_is_base_model"] is False


def test_frozen_split_shortfall_and_disjoint_sets():
    manifest_path = ROOT / "split" / "fr_style" / "manifest.json"
    assert manifest_path.is_file(), "run scripts/build_fr_punct_data.py first"
    manifest = json.loads(manifest_path.read_text())
    shipped = manifest["shipped"]
    assert shipped["verified_real_pairs"] == 0
    assert shipped["verified_real_pairs"] < manifest["plan_targets"]["verified_real_pairs"]
    assert shipped["deterministic_synthetic"] == 500
    assert shipped["mixed_literal_counterexamples"] == 300
    assert shipped["repair"] == 50
    assert shipped["noop"] == 50
    assert shipped["sealed_test"] == 40
    assert manifest["human_french_review_signed_off"] is False
    assert manifest["do_not_train_on_sealed_test"] is True
    train = [
        json.loads(line)
        for line in (ROOT / "split" / "fr_style" / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    sealed = [
        json.loads(line)
        for line in (ROOT / "split" / "fr_style" / "sealed_test.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train_ids = {row["id"] for row in train}
    sealed_ids = {row["id"] for row in sealed}
    assert not (train_ids & sealed_ids)
    assert {row["target_xml"] for row in train}.isdisjoint({row["target_xml"] for row in sealed})
    assert all(row["verified_real"] is False for row in train)
    assert all(row["human_french_review_signed_off"] is False for row in train)
    historical = json.loads((ROOT / "split" / "fr_style" / "historical_regression_10.json").read_text())
    assert historical["not_the_fresh_test"] is True
    assert historical["gold_translations"] is None
    assert len(historical["paths"]) == 10
    historical_names = {Path(path).name for path in historical["paths"]}
    blob = "\n".join(row["source_xml"] + row["target_xml"] for row in train)
    for name in historical_names:
        assert name not in blob
    families = manifest["families"]["partition"]
    for family in ("xml-schema-diagram", "oxy"):
        assert families[family]["partition"] == "regression_not_fresh_test"
        assert families[family]["n"] > 1
