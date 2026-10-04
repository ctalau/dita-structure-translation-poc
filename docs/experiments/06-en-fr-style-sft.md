# English to French, new typography LoRA

This is not a before/after. `results/fr_style_sft/fr_style_eval.json` is this adapter's greedy outputs only. The file has no paired base-model column. It does not say whether the held-out metric moved.

On the historical 10, this adapter's new-oracle mean reward is **0.465**. `gate_passed` is **7**. `bad_spacing` is **1**. `penalty` is **5**. The old counter on the same outputs is punct **5**, schema **1**, violations **6**, `topics_with_any_violation` **4**. Four topics are not perfect. Six have new-oracle reward 1.0.

On the sealed 40, this adapter's new-oracle mean reward is **1.0**. `gate_passed` is **40**. `bad_spacing` is **0**. `penalty` is **0**. The old counter is punct **0**, schema **0**, violations **0**, `topics_with_any_violation` **0**. The sealed set is synthetic and built from the same templates as training. A perfect score on it is not a production result.

GRPO did not train. The probe decision is `skip`.

This run does not replace the continued NBSP run. That run is [05-en-fr-nbsp-sft2.md](05-en-fr-nbsp-sft2.md). It does not replace the violation-count run. That run is [03-en-fr-violations.md](03-en-fr-violations.md).

## Method

New LoRA from `Qwen/Qwen3.5-4B`. `results/fr_style_sft/adapter_config.json` has `base_model_name_or_path` `Qwen/Qwen3.5-4B`, `r` 8, `lora_alpha` 16, `lora_dropout` 0.0, `peft_type` `LORA`, `peft_version` `0.21.2`, `task_type` `CAUSAL_LM`, `inference_mode` true, `revision` null. The file has no `continued_from` field. The task note says this adapter is not a continuation of the old NBSP adapter.

`target_modules` in that file is:

`^(?!.*(?:visual|vision|merger|lm_head|embed)).*\.(?:linear_attn|self_attn|mlp)\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj|q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$`

`results/fr_style_sft/sft_metrics.json` records `stage` `sft`, `selected_target_count` 12, and the same twelve names. `full_trainable_params_estimate` is 16232448. `lora_A_modules` is 248. `rollout` is null.

The adapter weights exist and are not in git. The task note says they are about 65MB, and that optimizer checkpoints were not committed. This commit has `adapter_config.json` only.

Eval in `fr_style_eval.json` used `max_new_tokens` 768 and `adapter` `results/fr_style_sft_adapter`. Every historical row and every sealed row has `hit_token_cap` false. The evaluator version on the rows is `fr-typography-oracle-1`.

## Data split

Train rows are the frozen split in `split/fr_style/`: 0 verified real pairs, 500 deterministic synthetic, 300 mixed literal counterexamples, 50 repair, 50 no-op. The historical 10 are regressions, not the fresh test. The sealed 40 are synthetic topics with the token `SEALEDTOKEN`, disjoint from train. `split/fr_style/manifest.json` records `human_french_review_signed_off` false.

## Numbers

`sft_metrics.json`:

| Field | Value |
| --- | --- |
| global_step | 57 |
| max_steps | null |
| seen_tokens | 300717 |
| step_seconds | 887.955 |
| tokens_per_sec | 338.662 |
| peak_allocated_mib | 6954.1 |
| peak_reserved_mib | 7456.0 |

The task note said one epoch, 57/57 steps, about 905 seconds, about 339 tokens/s, and peak allocated about 6954 MiB. The file records `global_step` 57 and `max_steps` null. It does not record a planned step total or an epoch count. `step_seconds` is 887.955, not about 905. `tokens_per_sec` is 338.662, not about 339. `peak_allocated_mib` is 6954.1. Where those disagree, the file wins.

`configs/fr_style.yaml` sets `max_epochs` to 1. That is the training contract. It is not a field in `sft_metrics.json`.

`scripts/train_fr_sft.py` estimates optimizer steps as `ceil(n / per_device_batch_size) // gradient_accumulation_steps`. For 900 rows, batch 1, and accumulation 16, that estimate is 56. The metrics file records `global_step` 57. The file wins for the step count the trainer reached.

Historical 10, from `fr_style_eval.json`. Old-counter reward 0.0 means zero violations. New-oracle reward 1.0 means the gate passed and the typography penalty is 0.

| Topic | Old punct, schema, violations | New reward | Gate |
| --- | --- | ---: | --- |
| `topics/dcpp_how_to_change_the_page_dimension.dita` | 0, 0, 0 | 1.0 | passed |
| `topics/author-editing-tables-xhtml.dita` | 0, 0, 0 | 1.0 | passed |
| `topics/ant-preferences.dita` | 0, 0, 0 | 1.0 | passed |
| `rules/styleguide/uicontrol.dita` | 1, 0, 1 | 0.0 | passed |
| `topics/dcpp_how_to_control_titles_layout.dita` | 0, 1, 1 | -1.0 | failed, all six gates |
| `topics/ch_xml_support.dita` | 2, 0, 2 | -0.175 | failed, `protected_literals` |
| `glossary/project-options.dita` | 0, 0, 0 | 1.0 | passed |
| `topics/ch_avoiding_page_breaks.dita` | 0, 0, 0 | 1.0 | passed |
| `topics/ch_external-links.dita` | 2, 0, 2 | -0.175 | failed, `protected_literals` |
| `topics/dcpp_how_to_center_videos.dita` | 0, 0, 0 | 1.0 | passed |

`rules/styleguide/uicontrol.dita` passed the gate. Typography score is 0.0, `bad_spacing` 1, `penalty` 1, `nbsp_count` 0, `source_linguistic` 1, `hyp_linguistic` 1, `omitted_marks` 0. The bad excerpt is `Rule: uicontrol E`. Old-counter reward is -1.0. `well_formed` and `dtd_valid` are true.

`topics/dcpp_how_to_control_titles_layout.dita` failed `xml_only`, `well_formed`, `dtd_valid`, `structure`, `protected_attributes`, and `protected_literals`. New reward is -1.0. `typography` is null. Old counter: `well_formed` false, `dtd_valid` false, punct 0, schema 1, violations 1, reward -1.0.

`topics/ch_xml_support.dita` failed only `protected_literals`. New reward is -0.175. The typography object still has score 1.0, penalty 0, `nbsp_count` 1. Old counter punct 2, schema 0, violations 2, reward -2.0.

`topics/ch_external-links.dita` failed only `protected_literals`. New reward is -0.175. Typography score is 0.2, `omitted_quotes` 2, `english_quotes_penalized` 2, `source_quote_marks` 2, `hyp_guillemets` 0, `penalty` 4, `base` 5, `nbsp_count` 3, `bad_spacing` 0. Old counter punct 2, schema 0, violations 2, reward -2.0.

Sealed summary, same file: n 40, old counter all zeros, new oracle mean reward 1.0, `gate_passed` 40, `bad_spacing` 0, `penalty` 0. Every sealed row has new-oracle reward 1.0, `gate_passed` true, and old-counter violations 0. Ids run from `sealed-00000` through `sealed-00039`.

## GRPO

`results/fr_style_sft/fr_style_grpo_probe.json`: `n_prompts` 6, `group_size` 4, `n_groups_with_reward_spread` 0, `n_groups_all_tie` 6, `punctuation_errors_remain` true, `decision` `skip`, `reason` `all sampled groups tied`.

Every listed reward is 1.0. `reward_spread` is 0.0 and `tie` is true on each group.

| Id | Rewards | old_punct |
| --- | --- | --- |
| `syn00054` | 1.0, 1.0, 1.0, 1.0 | 0, 0, 0, 0 |
| `mix-00108` | 1.0, 1.0, 1.0, 1.0 | 1, 1, 1, 1 |
| `syn00205` | 1.0, 1.0, 1.0, 1.0 | 0, 0, 0, 0 |
| `syn00313` | 1.0, 1.0, 1.0, 1.0 | 0, 0, 0, 0 |
| `syn00141` | 1.0, 1.0, 1.0, 1.0 | 0, 0, 0, 0 |
| `mix-00206` | 1.0, 1.0, 1.0, 1.0 | 0, 0, 0, 0 |

`mix-00108` has old-counter punct 1 on every sample and new reward 1.0 on every sample. The probe still records `punctuation_errors_remain` true. No GRPO update is in these files.

## Cost

`sft_metrics.json` records `gpu` `NVIDIA GeForce RTX 3090 Ti` and `hourly_usd` 0.27. It has no pod id, no clock window, and no charge.

The task note says the pod API reported pod `uzhv25u68nclv4`, a community RTX 3090, API rate $0.22/hr, about 00:10 to 00:59 Europe/Bucharest, about 48 minutes, approximate charge $0.18, and that the pod was deleted. Those items are not in the attached JSON.

The two sources disagree. The metrics file says RTX 3090 Ti and $0.27. The task note says the pod API said RTX 3090 and $0.22/hr. Each number stays with its source.

## Surprises

- There is no before column. The historical mean 0.465 and the sealed mean 1.0 are this adapter's greedy scores, not a measured change.
- Four historical topics are not perfect: `rules/styleguide/uicontrol.dita` (gate passed, reward 0.0, excerpt `Rule: uicontrol E`), `topics/dcpp_how_to_control_titles_layout.dita` (reward -1.0, not well-formed), `topics/ch_xml_support.dita` (reward -0.175, `protected_literals`), `topics/ch_external-links.dita` (reward -0.175, `protected_literals`, typography score 0.2).
- The sealed 40 are all new-oracle reward 1.0. They are synthetic copies of the training templates. That is not a production result.
- GRPO skipped. All six sampled groups tied at reward 1.0. `punctuation_errors_remain` is true. `mix-00108` is old-counter punct 1 and new reward 1.0 on all four samples.
- The task note's about 905 seconds and about 339 tokens/s disagree with `step_seconds` 887.955 and `tokens_per_sec` 338.662.
- The metrics file's GPU name and hourly rate disagree with the pod API figures in the task note.

## Files and commit

Result files: `results/fr_style_sft/sft_metrics.json`, `results/fr_style_sft/fr_style_eval.json`, `results/fr_style_sft/fr_style_grpo_probe.json`, `results/fr_style_sft/adapter_config.json`.

The adapter weights and the optimizer checkpoints are not in git.

Commit that added them: pending.
