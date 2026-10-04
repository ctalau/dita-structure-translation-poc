# English to French, violation-count reward

Held-out violation counts did not move.

The test set stayed fully violated: **10/10** topics with a violation before and after. Schema occurrences stayed **5**. Punctuation occurrences stayed **34**. Total violation occurrences stayed **39**. Mean violations stayed **3.9**. Mean reward stayed **-3.9**. `zero_violation_topics` stayed 0. The part histogram is the same on both sides: `not_well_formed` 3, `bad_high_punct` 34, `attr_value` 1, `child_order` 1. English-quote occurrences on the 10 are 0 before and after.

Most train topics were skipped because the two samples tied. **63 of 80** groups did not update. **52** of those ties are both samples with no schema miss and the same positive `bad_high_punct` count (no English quotes), so the non-breaking-space count had no winner.

This document does not treat the earlier French run's single format fix as a result of this run. That run is [02-en-fr-format.md](02-en-fr-format.md).

## Method

Qwen/Qwen3.5-4B, bitsandbytes NF4 QLoRA, rank 8, alpha 16, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`. Learning rate `5e-6`. The method string in `results/fr_punct/metrics.json` is: scan greedy until a topic has at least one violation, or the pool ends, or the GPU budget cap hits; GRPO-style group size 2 on violated train topics only; advantage `(r - mean) / std`; reward is minus the violation count (schema occurrences plus French punctuation occurrences); 0 is perfect; a group updates only when the two counts differ.

`count_violations` in `scripts/dita_reward.py` counts, as schema, one not-well-formed, one truncation, each xmllint error line, each changed attribute value, and each child insertion or deletion (a reorder of unlike siblings is one deletion plus one insertion). Punctuation counts each `?` `!` `:` `;` in prose that is not immediately preceded by U+00A0, plus each English double quote (`"`, `“`, `”`) when the source text contains a quotation. `codeblock`, `codeph`, `filepath`, and the other verbatim tags are skipped when the file is well-formed. If it is not well-formed, that exclusion does not run. The system prompt still asks for guillemets and for the non-breaking space. The reward does not count a missing guillemet. It counts an English quote, and it counts a high mark that lacks U+00A0.

Eval is greedy. Training samples two completions at temperature 0.7, top-p 0.8, top-k 20, in one batched generate of the same prompt twice (`generate_batch` of `[prompt, prompt]` in `scripts/run_fr_punct.py`). The loss uses `tok_lp.mean()`. The docstring on that function still says "Sum of token logprobs". The return value is the mean. The previous French run's script returns `tok_lp.sum()`.

`results/fr_punct/metrics.json` records `"kernels": {"causal_conv1d": true, "flash_linear_attention": true, "fla": true}`.

The scan did not hit the time cap. `stop_reason` is `found_target`. `elapsed_s` is **2390.7**. The earlier page `docs/report_punct.md` (commit `f9a3524`) says the scan wall was 587 s. That number is not in the result files. `elapsed_s` in `results/fr_punct/metrics.json` is the recorded scan time.

## Data split

Same source commit `db722d7af0c705cc3d6969f52bd1299a20ba1770`. Same candidate pool as the previous French run: 1730 topics, 400–4500 bytes (`split/fr_candidates.json`).

From `split/manifest_fr_punct.json` and `results/fr_punct/metrics.json` `scan`:

| Item | Value |
| --- | --- |
| Scanned | 184 |
| Clean | 84 |
| Violations found | 100 |
| stop_reason | `found_target` |
| elapsed_s | 2390.7 |
| Not reached | 783 |
| Too big for the 896 cap | 763 |
| skipped_over_cap | 0 |
| Seed | 1337 |
| Test / val / train | 10 / 10 / 80 |

The test 10 were drawn from the 100, so 10/10 with a violation at baseline is how the split was built. The scan rate is 100/184.

On the 100, from `results/fr_punct/scan_broken.json`: punctuation on 91 topics (82 punctuation only, 9 both), schema on 18 (9 schema only). Occurrences: punctuation 190, schema 21. Of the punctuation occurrences, `bad_high_punct` is 182 and `english_quotes` is 8. Schema parts: `child_order` 8, `not_well_formed` 6, `attr_value` 5, `dtd_errors` 2, `tag_mismatch` 0, `truncation` 0. Every broken scan row has `"batch": 8`. `agg_tok_s` on those rows runs from 69.29 to 126.93.

Baseline rows are copied from that scan (`baseline_from_scan` in `scripts/run_fr_punct.py`), so the baseline latency is the wall clock of the batch-8 call, shared by the topics in the call. The after pass generates one topic at a time. The p50 drop from 26.988 s to 21.967 s is that difference in how the call was batched. It is not a change in violation count.

## Numbers

Held-out, from `results/fr_punct/metrics.json`:

| Metric | Baseline | After |
| --- | ---: | ---: |
| topics_with_violations | 10 | 10 |
| schema_topics | 5 | 5 |
| punct_topics | 9 | 9 |
| schema_only_topics | 1 | 1 |
| punct_only_topics | 5 | 5 |
| both_schema_and_punct_topics | 4 | 4 |
| schema_violation_occurrences | 5 | 5 |
| punct_violation_occurrences | 34 | 34 |
| violation_occurrences | 39 | 39 |
| mean_violations | 3.9 | 3.9 |
| mean_reward | -3.9 | -3.9 |
| zero_violation_topics | 0 | 0 |
| latency_p50_s | 26.988 | 21.967 |
| latency_mean_s | 26.692 | 18.885 |

Eight of the ten `hyp_xml` strings are byte-identical. The two that differ still have the same schema count and the same punctuation count:

| Topic | Schema | Punct (`bad_high_punct`) | Identical string |
| --- | ---: | ---: | --- |
| `dcpp_how_to_change_the_page_dimension` | 1 | 16 | yes |
| `author-editing-tables-xhtml` | 1 | 1 | yes |
| `ant-preferences` | 1 | 6 | no |
| `uicontrol` | 0 | 1 | yes |
| `dcpp_how_to_control_titles_layout` | 1 | 1 | yes |
| `ch_xml_support` | 0 | 3 | yes |
| `project-options` | 1 | 0 | no |
| `ch_avoiding_page_breaks` | 0 | 2 | yes |
| `ch_external-links` | 0 | 3 | yes |
| `dcpp_how_to_center_videos` | 0 | 1 | yes |

`project-options` is the schema-only topic (`child_order` 1). Its baseline row stores `"punct": {}` and the after row stores `"punct": 0`. The writer does `row.get("punct") or {}`, and a count of 0 is falsy, so the baseline field is an empty object. `parts.bad_high_punct` is 0 on both rows, and the summary punctuation total is 34 on both sides. The empty object is not a metric change.

`results/fr_punct/judge.json` (same agent, no API) records the two string edits. `ant-preferences`: faithfulness 4 before and 3 after; the note says the options went from `Intégré` / `Personnalisé` back to `Built-in` / `Custom`. `project-options`: French 3 before and 4 after; the note says `au niveau de projet` became `au niveau du projet`. The violation count on that topic stayed 1. `ch_external-links` is the row scored guillemets 4, with the quote `après le texte « W3C »`, and nbsp 2. Its `bad_high_punct` stayed 3.

Training, from `results/fr_punct/metrics.json` and `results/fr_punct/train_trace.json`:

| | Count |
| --- | ---: |
| Train topics attempted | 80 |
| Updated | 17 |
| Skipped, counts tied | 63 |

Tied violation pairs: `(1, 1)` 34, `(2, 2)` 17, `(3, 3)` 7, `(4, 4)` 3, `(5, 5)` 1, `(6, 6)` 1. Of the 63 ties, 62 have identical `parts` on the two samples. The exception is `topics/ch_layout.dita`, both total 1, one sample `child_order` 1 and the other `not_well_formed` 1.

Of the 63 ties, 52 are schema 0 on both samples, `english_quotes` 0, and the same `bad_high_punct` greater than 0. Six ties are schema-only (same schema count, punctuation 0). The other five ties also match on the total and include a schema miss, English quotes, or both, still with no winner. 56 of the 63 ties have equal positive `bad_high_punct` on the two samples.

Three of the 17 updates contain a 0: `topics/cf-admin-page-mail.dita` `[0, 2]`, `topics/dcpp_headers_footers_built_in_css.dita` `[1, 0]`, `glossary/canonicalize.dita` `[1, 0]`. The greedy test set still has `zero_violation_topics` 0.

`results/fr_punct/metrics.json` does not record training wall-clock seconds, and there is no `results/fr_punct/run_log.jsonl` in the repo. The script would have logged `train_s`. The writeup committed in `f9a3524` says training took 1614 s. That figure is not recomputed here.

## Speed notes that are in the result files

`results/fr_punct/throughput_bench.json` is the NF4 transformers bench, stack string `transformers NF4 + causal_conv1d + fla`, same short topic, 128 new tokens. It records a FLOP roof, not a bus-bandwidth counter. There is no bus or bandwidth field in the result files, so this document does not say whether 4-bit decode was bus-bound.

| | Batch 1 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: |
| tok/s | 21.07 | 76.62 | 148.49 |
| seconds | 6.074 | 6.682 | 6.896 |
| new tokens | 128 | 512 | 1024 |
| GPU util median | 28 | 42 | 58.0 |
| GPU memory max (MiB) | 5297 | 5497 | 6099 |

The file records `params` 4540838400, `flop_per_token` 9081676800, `fp16_tensor_roof_flop_s` 142000000000000.0, `compute_roof_tok_s` 15635.9, `batch1_fraction_of_roof` 0.001348. The roof note in the file calls 142 TFLOP/s a ballpark RTX 3090 FP16 tensor-core class figure with sparsity. Static KV cache at batch 1: 21.17 tok/s, util median 31, `mem_mib_max` 6099. `chosen_batch` is 8.

`results/fr_punct/kernel_import.json` records `causal_conv1d` missing: `undefined symbol: _ZN3c106detail14torchCheckFail...` (the C++ `torchCheckFail` / `std::__cxx11::basic_string` symbol). `fla` is `0.5.2`. `torch` is `2.6.0+cu124`. The later bench file and `metrics.json` record `causal_conv1d: true`. With those kernels flagged true, batch-1 throughput in the bench is still 21.07 tok/s. `results/fr_punct/bench.json` is one longer greedy call on the same stack: 896 new tokens, 42.051 s, 21.31 tok/s, violations 3 (schema 2, punct 1), GPU `NVIDIA GeForce RTX 3090`.

`results/fr_punct/vllm_batch_bench.json` is a side bench. The JSON does not name the stack. `scripts/vllm_batch_bench.py`, which writes that file, loads vLLM with `dtype="bfloat16"` and does not train. The training policy in `metrics.json` is the NF4 QLoRA loop.

| | Batch 1 | Batch 4 | Batch 8 |
| --- | ---: | ---: | ---: |
| tok/s | 79.92 | 285.21 | 512.01 |
| seconds | 1.602 | 1.795 | 2.0 |
| new tokens | 128 | 512 | 1024 |
| GPU util median | 100 | 100.0 | 100 |
| GPU memory max (MiB) | 19385 | 19385 | 19441 |

The earlier writeup's account of a FlashInfer JIT failure and an `nvcc` flag is not in these JSON files. It is not repeated here as a measured result.

## Cost

Not in `results/fr_punct/metrics.json`. Recorded in `docs/report_punct.md` as of commit `320714a` (the spend line; the run itself is `f9a3524`).

| Item | Recorded value |
| --- | --- |
| Pod | `54mu4jws1t3933` (dita-fr-punct) |
| GPU | Community RTX 3090, $0.22/hr, host CUDA 13.0, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Start | 2026-10-04 05:45:57 UTC |
| Train done | 2026-10-04 07:01:55 UTC |
| Process exit | 2026-10-04 07:05:05 UTC |
| Deleted | 2026-10-04 07:09 UTC. The writeup says `delete-pod` returned 204 and a follow-up `get-pod` returned 404 |
| Estimated spend | about $0.30 at $0.22/hr for 1 hour 23 minutes. Disk is extra and small. Cap for the run was $4 |

## Surprises

- Held-out schema occurrences and punctuation occurrences are the same numbers after 17 updates. Eight test strings did not change. The two that changed did not change the counts. One of those two is a regression in the judge notes (`Built-in` / `Custom`); the other is one article (`du projet`) with the violation count still 1.
- 63 of 80 groups tied. 52 of those are the same missing-non-breaking-space count on both samples and no other violation. The count reward had nothing to rank.
- Three training groups did contain a zero-violation sample. The greedy test set still has zero topics at zero violations.
- The scan stopped because it found 100 violations (`found_target`, 2390.7 s), not because a wall clock fired. The 587 s figure in the earlier writeup disagrees with `metrics.json`.
- Batch-1 NF4 throughput in the bench that records the conv and linear-attention kernels as loaded is 21.07 tok/s (util median 28, 5297 MiB). The single-topic bench at 896 new tokens is 21.31 tok/s. The files do not record a bus-bandwidth measurement.
- vLLM bf16 on the same card, in the side bench, is 79.92 tok/s at batch 1 and 512.01 tok/s at batch 8, util median 100. It is not the training policy.
- `"surprises": []` in `results/fr_punct/metrics.json`. The items above are read from the other result files.

## Files and commit

Result files: `results/fr_punct/metrics.json`, `results/fr_punct/baseline_test.json`, `results/fr_punct/after_test.json`, `results/fr_punct/scan_broken.json`, `results/fr_punct/train_trace.json`, `results/fr_punct/judge.json`, `results/fr_punct/throughput_bench.json`, `results/fr_punct/vllm_batch_bench.json`, `results/fr_punct/bench.json`, `results/fr_punct/kernel_import.json`, `results/fr_punct/adapter/`, `split/manifest_fr_punct.json`.

Commit that added them: `f9a352453d2c962585b4456bddb4c6334a35d7a2` (`Add the French violation-count run at batch 8.`). The pod-delete and $0.30 lines were edited in `320714a3dd5920c44b432d4be5859d4045a9830f`.
