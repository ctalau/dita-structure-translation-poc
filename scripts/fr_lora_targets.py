#!/usr/bin/env python3
"""LoRA targets for Qwen/Qwen3.5-4B, enumerated from the text-stack module names.

Names and shapes are the ones that reproduce the published trainable-parameter
count of 1,572,864 for q_proj, k_proj, v_proj, and o_proj on the 8 full-attention
layers (results/fr_nbsp_sft2/metrics.json and the pod log in docs/experiments/01-en-ro.md).

Text config Qwen/Qwen3.5-4B revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a:
hidden 2560, intermediate 9216, 32 layers, 24 Gated DeltaNet + 8 full attention,
head_dim 256, 16 query heads, 4 KV heads, linear key/value head dim 128,
16 linear key heads, 32 linear value heads.

Module names, from the transformers implementation of this checkpoint:
- Qwen3_5GatedDeltaNet: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj
- Qwen3NextAttention (full attention, attn_output_gate): q_proj, k_proj, v_proj, o_proj
- MLP: gate_proj, up_proj, down_proj

Vision and embedding modules are excluded. conv1d, norms, A_log, and dt_bias
are not projections and are not targeted.
"""
from __future__ import annotations

import json

HIDDEN = 2560
INTERMEDIATE = 9216
RANK = 8
HEAD_DIM = 256
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 4
LINEAR_KEY_DIM = 128 * 16
LINEAR_VALUE_DIM = 128 * 32
LINEAR_VALUE_HEADS = 32
N_LINEAR_LAYERS = 24
N_FULL_LAYERS = 8
N_LAYERS = 32

# Batch-1 NF4 inference peak recorded in results/fr_punct/throughput_bench.json.
# This is not a new measurement.
MEASURED_NF4_INFERENCE_MIB = 5297
ACTIVATION_ALLOWANCE_BYTES = 10 * 1024 ** 3
DEVICE_BYTES = 24 * 1024 ** 3

DELTA_NET_PROJECTIONS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
)
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Used only if a CUDA OOM is raised before the first optimizer step, or if the
# static estimate below ever exceeds 24 GiB.
FALLBACK_DROP_DELTANET = ATTENTION_PROJECTIONS + MLP_PROJECTIONS
FALLBACK_ATTENTION_ONLY = ATTENTION_PROJECTIONS


def _lora_params(in_features: int, out_features: int, rank: int = RANK) -> int:
    return rank * (in_features + out_features)


def projection_shapes() -> dict[str, tuple[int, int]]:
    q_out = NUM_ATTENTION_HEADS * HEAD_DIM * 2
    kv_out = NUM_KEY_VALUE_HEADS * HEAD_DIM
    o_in = NUM_ATTENTION_HEADS * HEAD_DIM
    qkv_out = LINEAR_KEY_DIM * 2 + LINEAR_VALUE_DIM
    return {
        "in_proj_qkv": (HIDDEN, qkv_out),
        "in_proj_z": (HIDDEN, LINEAR_VALUE_DIM),
        "in_proj_a": (HIDDEN, LINEAR_VALUE_HEADS),
        "in_proj_b": (HIDDEN, LINEAR_VALUE_HEADS),
        "out_proj": (LINEAR_VALUE_DIM, HIDDEN),
        "q_proj": (HIDDEN, q_out),
        "k_proj": (HIDDEN, kv_out),
        "v_proj": (HIDDEN, kv_out),
        "o_proj": (o_in, HIDDEN),
        "gate_proj": (HIDDEN, INTERMEDIATE),
        "up_proj": (HIDDEN, INTERMEDIATE),
        "down_proj": (INTERMEDIATE, HIDDEN),
    }


def _layer_count(name: str) -> int:
    if name in DELTA_NET_PROJECTIONS:
        return N_LINEAR_LAYERS
    if name in ATTENTION_PROJECTIONS:
        return N_FULL_LAYERS
    if name in MLP_PROJECTIONS:
        return N_LAYERS
    raise KeyError(name)


def trainable_params(names: tuple[str, ...] | list[str]) -> int:
    shapes = projection_shapes()
    total = 0
    for name in names:
        in_features, out_features = shapes[name]
        total += _lora_params(in_features, out_features) * _layer_count(name)
    return total


def estimate_bytes(trainable: int) -> int:
    """Weights from the recorded inference peak, plus AdamW state and an allowance.

    The allowance is a planning ceiling for activations at sequence length 1536
    with gradient checkpointing. It is not a measured training peak.
    """
    weight_bytes = MEASURED_NF4_INFERENCE_MIB * 1024 ** 2
    adam_and_grad = trainable * (8 + 2)
    return weight_bytes + adam_and_grad + ACTIVATION_ALLOWANCE_BYTES


def target_regex(names: tuple[str, ...] | list[str]) -> str:
    joined = "|".join(names)
    return (
        r"^(?!.*(?:visual|vision|merger|lm_head|embed)).*"
        r"\.(?:linear_attn|self_attn|mlp)\.(?:" + joined + r")$"
    )


def selection() -> dict:
    full_names = DELTA_NET_PROJECTIONS + ATTENTION_PROJECTIONS + MLP_PROJECTIONS
    full_params = trainable_params(full_names)
    attention_params = trainable_params(ATTENTION_PROJECTIONS)
    fallback_params = trainable_params(FALLBACK_DROP_DELTANET)
    full_bytes = estimate_bytes(full_params)
    fits = full_bytes < DEVICE_BYTES
    chosen = list(full_names) if fits else list(FALLBACK_DROP_DELTANET)
    return {
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "rank": RANK,
        "alpha": 16,
        "dropout": 0.0,
        "delta_net_projections": list(DELTA_NET_PROJECTIONS),
        "attention_projections": list(ATTENTION_PROJECTIONS),
        "mlp_projections": list(MLP_PROJECTIONS),
        "attention_only_trainable_params": attention_params,
        "attention_only_matches_published_1572864": attention_params == 1_572_864,
        "full_trainable_params": full_params,
        "fallback_without_deltanet_trainable_params": fallback_params,
        "estimated_bytes_full": full_bytes,
        "device_bytes": DEVICE_BYTES,
        "full_set_fits_24gib_static_estimate": fits,
        "selected_targets": chosen,
        "selected_regex": target_regex(chosen),
        "fallback_targets": list(FALLBACK_DROP_DELTANET),
        "fallback_regex": target_regex(FALLBACK_DROP_DELTANET),
        "second_fallback_targets": list(FALLBACK_ATTENTION_ONLY),
        "fallback_rule": (
            "Use the full projection set when the static estimate is under 24 GiB. "
            "If the first CUDA allocation raises out-of-memory, drop DeltaNet projections "
            "and keep attention plus MLP. If that still raises out-of-memory, keep only "
            "q_proj, k_proj, v_proj, and o_proj, which are the targets whose 1,572,864 "
            "trainable parameters were already trained on a 24 GiB 3090."
        ),
        "measured_nf4_inference_mib": MEASURED_NF4_INFERENCE_MIB,
        "measured_nf4_inference_source": "results/fr_punct/throughput_bench.json batch 1 mem_mib_max",
        "activation_allowance_bytes": ACTIVATION_ALLOWANCE_BYTES,
        "activation_allowance_is_not_a_measurement": True,
    }


if __name__ == "__main__":
    print(json.dumps(selection(), indent=2))
