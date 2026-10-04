# French DITA typography: SFT then conditional GRPO

This file is the plan. It is not an experiment report. No training run was started to produce it. A later GPU run still needs its own document under `docs/experiments/`, as `AGENTS.md` requires. This page does not weaken that rule.

Prepared against commit `a4d18f9`. The default branch contains that commit. Diagnosis of the old counter uses the result files already committed on that baseline.

## Spec

Execute the CPU preparation only. Do not rent a GPU.

Outcome: a pull request that implements the CPU half of this plan and leaves GPU training unstarted.

Must be true when you stop:

- `scripts/dita_reward.py` is a context-aware oracle. Linguistic U+00A0 before `: ; ? !` and inside guillemets. Accept `&#160;`, `&#xA0;`, and declared `&nbsp;` after safe entity decoding. Do not NFKC. Protect code, filenames, URLs, namespace-qualified names (`xs:override`, `oxy:`, `xi:`, `xsl:`), times, ratios, and literals in ordinary `p`/`title`/`i` text, not only `codeph`/`codeblock`. Score inline text across element boundaries (NBSP in a tail and at the end of an inline element). Hard gate: raw response is XML only, well-formed, DTD-valid, ordered structure and protected attributes match, protected literals preserved. Invalid output scores below every valid output. Do not reward total NBSP count. Do not penalize a namespace colon that stays glued. Adversarial fixtures must show that adding `codeph`, deleting prose, or spacing `xs:override` cannot improve the reward.
- `tests/test_fr_typography_oracle.py` plus about 250 fixtures covering positives, negatives, inline tails, entities, clusters, omitted marks, deleted prose, identifier mutation, wrapper insertion, invalid XML, and sibling swaps. Tests pass. Ambiguous spans are marked as needing a human French review; do not pretend a reviewer signed off.
- A deterministic typography postprocessor baseline, reported as code plus a small CPU score on fixtures, separate from any model.
- Frozen splits and `scripts/build_fr_punct_data.py`. Training mixture as close as the repo allows to 100 verified real pairs, 500 deterministic synthetic, 300 mixed literal counterexamples, 100 repair/no-op. If 100 real pairs cannot be verified, ship fewer and say so. Do not invent gold translations. Do not train on the sealed test. Keep the historical 10 as regressions, not the fresh test. Group near-duplicate families (`xml-schema-diagram-*`, `oxy-*`) into one partition.
- `configs/fr_style.yaml`, `scripts/train_fr_sft.py`, `scripts/train_fr_grpo.py`, `scripts/eval_fr_punct.py`. SFT starts a new adapter from pinned `Qwen/Qwen3.5-4B`, not a continuation of the old LoRA. Rank 8, alpha 16, dropout 0, NF4, lr 1e-4, AdamW, cosine 5% warmup, effective batch 16, seq cap 1536, at most 1 epoch, completion-token loss, thinking off. LoRA targets must be enumerated from the real module names and should cover DeltaNet projections, attention projections, and MLP projections, with a recorded fallback if that will not fit 24 GB. GRPO uses a pinned TRL `GRPOTrainer` (or say the pin failed), frozen SFT adapter as KL reference (not the base model), `loss_type` `dr_grpo`, `scale_rewards` false, epsilon 0.2, beta 0.02, group size 4, lr 5e-6. The reward is the gated scalar in this plan.
- Re-score historical outputs under `results/` with both the old counter and the new oracle. Keep old numbers verbatim. Publish the new counts separately with an evaluator version. Do not claim a corrected baseline that was not measured.
- Save this plan as `docs/plans/fr-punct-sft-grpo.md`, outside `docs/experiments/`. Do not write an experiment report. No training run.
- `AGENTS.md` still requires a separate honest experiment note after a real run. Do not weaken it.

Budget contract, encoded in the training scripts as comments and in `configs/fr_style.yaml`, not executed: one community RTX 3090, $5 total. Allocations: pilot $0.50, SFT $1.80, conditional GRPO $1.30, final eval $0.70, contingency $0.40, storage $0.30. The watchdog must be able to terminate the pod. Do not call RunPod from the CPU preparation.

## Gated scalar

Evaluator version: `fr-typography-oracle-1`. Function: `gated_scalar` in `scripts/dita_reward.py`. The legacy `count_violations` path is unchanged and is not this reward.

Six gates, in order: `xml_only`, `well_formed`, `dtd_valid`, `structure`, `protected_attributes`, `protected_literals`.

`xml_only` reads the raw response. Think tags, markdown fences, or prose around the document fail the gate. The XML declaration and the DOCTYPE may be peeled only so ElementTree can parse. `xmllint` validates the raw string against the DITA 1.3 catalog. No NFKC. Entities `&#160;`, `&#xA0;`, `&#x00A0;`, `&#0160;`, and `&nbsp;` expand to U+00A0 before parsing. A fullwidth colon (U+FF1A) is not an ASCII colon.

If any gate fails, and `passed` is the number of gates that passed:

```text
reward = round(-1 + 0.99 * (passed / 6), 4)
```

That range is about [-1, -0.175]. It is strictly below every valid score.

If every gate passes, the typography score is in [0, 1]:

```text
penalty = omitted_linguistic + bad_linguistic_spacing + bad_guillemet_edges
        + omitted_quotes + english_quotes_when_source_has_quotes
base = source_linguistic + source_quote_marks
reward = 1                         if penalty == 0
reward = max(0, 1 - penalty / max(base, 1))   otherwise
```

There is no term for the total number of U+00A0 characters. `reward_uses_nbsp_count` is false.

Linguistic high punctuation is `: ; ? !` whose previous character must be U+00A0. A cluster continuation (the previous character is itself high punctuation) is not its own obligation. A glued namespace, QName, time, ratio, URL, filename, or CSS pseudo (`:before`, `:after`, and the other names in the protected pattern) is not linguistic. Digit-colon-digit is protected. A spaced ratio such as `16 : 9` is not counted as a miss; it is flagged for human French review. Narrow spaces U+202F, U+2009, U+2007, and U+200A are misses and are flagged the same way. `«` must be followed by U+00A0. `»` must be preceded by U+00A0. Source quote marks outside protected spans (`"`, U+201C, U+201D, `«`, `»`) create quote obligations. English double quotes in the hypothesis are penalized only when the source has quotes. Extra correct marks do not raise a perfect score above 1 and do not improve a partial score. Omitting a source linguistic mark is a miss, so deleting the sentence does not raise the reward. Adding `codeph` changes the structure, the gate fails, and the reward is negative, below a valid score of 0. Spacing `xs:override` drops the protected literal and fails the gate.

Inline text is concatenated across inline elements. A non-breaking space at the end of an inline element, or in a tail, is visible to the next mark. Block elements are flow boundaries. Verbatim elements (`codeblock`, `codeph`, and the rest of `VERBATIM_TAGS`) are protected as wholes. Protected literals are the longest non-overlapping regex matches plus the strings inside verbatim elements. The hypothesis must contain each source literal at least as many times as the source.

Ambiguous spans set `needs_human_french_review` and `reviewer_signed_off: false`. `human_french_review_signed_off` is false. `reviewer` is null. Nothing in this preparation is a sign-off.

## Data

`scripts/build_fr_punct_data.py` writes `split/fr_style/`. Seed 1337.

| Bucket | Plan | What this repo can ship |
| --- | ---: | --- |
| Verified real pairs | 100 | 0. The Oxygen user guide at `db722d7` is English. No French gold is invented. Historical model outputs are not gold. |
| Deterministic synthetic | 500 | 500 constructed topic pairs, labeled `deterministic-synthetic`, `verified_real: false` |
| Mixed literal counterexamples | 300 | 300. Each keeps a glued literal (`xs:override`, `oxy:`, `xi:`, `xsl:`, a time, a ratio, a filename, a URL, or `:before`) next to one linguistic mark |
| Repair / no-op | 100 | 50 repair and 50 no-op, taken from the synthetic targets |
| Sealed test | a fresh held-out set, not the historical 10 | 40 synthetic topics carrying `SEALEDTOKEN`. Not 100 real pairs. Disjoint from train. |

The historical 10 paths in `split/manifest_fr_punct.json` `test` stay in `historical_regression_10.json`. They are regressions. They are not the fresh test. No French gold is attached to them.

`xml-schema-diagram-*` is one family. `oxy-*` is one family. If any member is in the old manifest train/val/test lists or in `results/fr_nbsp_sft2/metrics.json` `held_out_paths`, the whole family is `regression_not_fresh_test`. Otherwise it is `unverified_real_not_used`. Neither family enters train or the sealed test.

## Training contract, not executed

`configs/fr_style.yaml` and the two train scripts.

SFT (`scripts/train_fr_sft.py`): new LoRA from `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Not `PeftModel.from_pretrained` of `results/fr_punct` or `results/fr_nbsp_sft2`. Rank 8, alpha 16, dropout 0, NF4, AdamW (`adamw_torch`), cosine, warmup ratio 0.05, learning rate 1e-4, per-device batch 1, gradient accumulation 16, max sequence 1536, 1 epoch, completion-token labels only, thinking off. The script returns without training unless `--run` is passed on a CUDA device.

LoRA names, from the transformers classes for this checkpoint (`scripts/fr_lora_targets.py`):

- DeltaNet: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`
- Full attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` (`attn_output_gate` makes `q_proj` output `16 * 256 * 2`)
- MLP: `gate_proj`, `up_proj`, `down_proj`

Attention-only rank 8 is 1,572,864 trainable parameters, the same count as `results/fr_nbsp_sft2/metrics.json`. The full set is 16,232,448. Dropping DeltaNet leaves 10,616,832. The static 24 GiB estimate uses the recorded NF4 inference peak of 5297 MiB in `results/fr_punct/throughput_bench.json` (batch 1 `mem_mib_max`), AdamW state at 8 bytes per parameter, bf16 gradients at 2 bytes per parameter, and a 10 GiB activation allowance. That allowance is not a measurement. The full-set estimate is under 24 GiB, so the selected targets are the full set. Fallback, only if a later CUDA run raises out-of-memory: drop DeltaNet and keep attention plus MLP; if that still fails, attention only.

GRPO (`scripts/train_fr_grpo.py`): TRL pin `trl==1.14.1`. `loss_type` `dr_grpo`, `scale_rewards` false, epsilon 0.2, beta 0.02, `num_generations` 4, learning rate 5e-6, max prompt 640, max completion 896. The reward function calls `gated_scalar`. The script loads the new SFT adapter as `default` and again as `ref`, freezes parameters whose names contain `.ref.`, and does not pass a fresh `peft_config`. In TRL 1.14.1, `beta > 0` with a PEFT model and no `ref` adapter disables adapters and uses the base model as the KL reference. This script refuses that path: it refuses a missing `ref` adapter and refuses a non-null `trainer.ref_model`. The pin is the v1.14.1 source contract. Importing TRL is inside `--run` only, so a machine without TRL can still load the script and exit. Run GRPO only when the SFT dev mean gated reward is below 0.95 and at least $1.30 remains. Translate rows only.

Budget (`scripts/budget_watchdog.py`): $5.00 = 0.50 + 1.80 + 1.30 + 0.70 + 0.40 + 0.30. Hourly rate $0.22 is the community RTX 3090 rate recorded in `docs/experiments/03-en-fr-violations.md`, used only to turn a dollar cap into a wall clock. It is not a new quote. `terminate_runpod_pod` sends `DELETE https://rest.runpod.io/v1/pods/{id}` with a bearer token. Tests inject the opener. CPU preparation does not call it. A future run reads `RUNPOD_POD_ID` and `RUNPOD_API_KEY`.

## Historical numbers, copied

These are the published files. A rescore is a new measurement. It is not a replacement.

`results/fr_punct/metrics.json` and `results/fr_punct/baseline_test.json` summary, and the same figures in `after`: n 10, topics with violations 10, schema violation occurrences 5, punctuation violation occurrences 34, violation occurrences 39, mean reward -3.9.

`results/fr_nbsp_sft2/metrics.json`: punctuation occurrences 39, then 33, then 28. Schema occurrences 2, then 0, then 2. Topics still violated 10, then 9, then 9. `trainable_params` 1572864. The reward string in that file says English quotes are counted but not added to the reward, and `skip_tags` are `codeblock` and `codeph`. Recomputing `count_violations` on those outputs can disagree with 39, 33, and 28. When it does, the file wins. The disagreement is recorded in `results/fr_typography_oracle_rescore.json` and is not a corrected baseline.

## What this preparation does not do

No GPU rental. No training. No RunPod call. No human French sign-off. No experiment document for a run that did not happen.
