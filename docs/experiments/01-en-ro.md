# English to Romanian, pass/fail reward

Held-out pass/fail did not change.

On the 10 test topics, valid-and-skeleton stayed **8/10**, schema-invalid topics stayed **1/10**, skeleton mismatches stayed **2/10**, and mean reward stayed **0.635**. Baseline latency p50 was 23.553 s. After the update it was 23.448 s. `schema_error_total` stayed 0.

Most training groups were flat because both samples tied: **20 of 24** groups were skipped (`skipped_flat_reward` in `results/metrics.json`). Four groups updated.

This run is not a step toward the later French runs. Those are separate documents.

## Method

Qwen/Qwen3.5-4B, bitsandbytes NF4 (double quant, bf16 compute), QLoRA rank 8, alpha 16, on `q_proj`, `k_proj`, `v_proj`, `o_proj`. The pod log prints `trainable params: 1,572,864 || all params: 4,540,838,400 || trainable%: 0.0346`. Device string in the metrics file: `RunPod Community RTX 3090, NF4 4-bit (bitsandbytes), greedy`. Eval is greedy. Thinking is forced off with an empty think block.

The reward in commit `1517f47` (`scripts/dita_reward.py` at that commit) is 1.0 only when the element skeleton matches and `xmllint --valid` accepts the file. Otherwise it is a penalty: `0.15` times the validator error count (at least 1) plus, when the skeleton moved, `0.5 + 0.02 *` the skeleton-length difference, floored at -1.0. Not-well-formed XML is -1.0 and returns before xmllint. The current `scripts/dita_reward.py` is a later reward. The numbers below come from the saved rows, which this reward produced.

Training is GRPO-style, group size 2, temperature 0.7, top-p 0.8, top-k 20. Advantage is `(reward - group mean) / group std`. The loss multiplies that advantage by the sum of completion-token logprobs. One AdamW step at `5e-6`, grad clip 1.0, only when the two rewards differ. The metrics method string is `GRPO-style group size 2, advantage = (r - mean) / std, one QLoRA step per topic`.

The system prompt says translate English DITA to Romanian, keep elements, attributes, and attribute values, translate only human-readable text, keep the DOCTYPE, XML only.

## Data split

Source: `https://github.com/oxygenxml/userguide` commit `db722d7af0c705cc3d6969f52bd1299a20ba1770`. Recorded in `split/manifest.json`.

| Item | Value |
| --- | --- |
| `.dita` files | 2757 |
| Eligible | 963. Standard DITA, source passes the DTD, 700–2600 bytes |
| Seed | 1337 |
| Test | 10. Not used for updates |
| Val | 10. Scored after training. Not used to pick a checkpoint |
| Train updates | 24 |
| Eligible and unused | 919 |
| Train wall budget | 2400 s (`train_wall_s_budget`). The loop finished inside it |
| Train time | 1292.0 s (`rl_done` in `results/run_log.jsonl`) |

Token cap per topic: `min(640, max(160, len(src) / 3 + 48))` in `scripts/run_gpu_poc.py`.

## Numbers

Held-out, from `results/metrics.json` (the same aggregates are in `results/baseline_test.json` and `results/after_test.json`):

| Metric | Baseline | After |
| --- | ---: | ---: |
| n | 10 | 10 |
| schema_invalid_topics | 1 | 1 |
| schema_error_total | 0 | 0 |
| skeleton_mismatch_topics | 2 | 2 |
| valid_and_skeleton_ok | 8 | 8 |
| mean_reward | 0.635 | 0.635 |
| latency_p50_s | 23.553 | 23.448 |

Six of the ten `hyp_xml` strings differ. Four are byte-identical. The pass/fail fields on every topic are unchanged, which is why the mean reward is unchanged.

The two topics that fail the pass condition, both before and after (`results/baseline_test.json`, `results/after_test.json`):

- `topics/download-database-drivers.dita`. `new_tokens` 640, `max_new` 640, `well_formed` false, `dtd_valid` false, `dtd_errors` null, reward -1.0. `error_tail`: `hypothesis not well-formed: no element found: line 28, column 140`. The saved XML stops mid-attribute.
- `topics/author-editing-tables-xhtml.dita`. Well-formed, `dtd_valid` true, `dtd_errors` 0, `skeleton_match` false, reward -0.65. The saved id is `author-editare-tabele-xhtml`. The saved `conkeyref` starts with `reusables-editare-documente`.

`schema_error_total` is 0 because the not-well-formed topic never reaches xmllint, so there is no stored xmllint error line. The earlier writeup says xmllint reported `Premature end of data in tag filepath`. That sentence is not in the result files. The `error_tail` above is the recorded failure.

Val after training, from `results/metrics.json` (`val_after`). There is no `val_after.json`. Per-topic lines are in `results/pod_run.log`:

| Metric | val_after |
| --- | ---: |
| n | 10 |
| schema_invalid_topics | 5 |
| schema_error_total | 0 |
| skeleton_mismatch_topics | 6 |
| valid_and_skeleton_ok | 4 |
| mean_reward | -0.167 |
| latency_p50_s | 28.528 |

The log's ten `val_after` lines: four topics are well-formed, skeleton-matched, DTD-valid, reward 1.00 (`api_faq_introduction`, `dcpp_controlling_the_publication_content`, `update-xml-documents-using-xquery`, `schematron-search-refactor-operations-scope`). Four are `tok=640/640`, not well-formed, reward -1.00 (`saving-documents`, `cf-organization-security`, `wa-UrlChooser-troubleshooting`, `cf-ai-positron-overview`). `cf-admin-page-license` is `tok=617/617`, not well-formed, reward -1.00. `wa-ai-positron-overview-addon` is well-formed, skeleton mismatch, DTD-valid, reward -0.67.

Training groups, from `results/train_trace.json`:

| | Count |
| --- | ---: |
| Topics | 24 |
| Updated | 4 |
| Skipped, rewards tied | 20 |

Tied pairs: `(1.0, 1.0)` on 14 topics, `(-1.0, -1.0)` on 5, `(-0.67, -0.67)` on 1. The five `-1` ties are groups where both samples failed and the update still did not run.

The four updates, rewards `[sample0, sample1]`:

| Topic | Rewards |
| --- | --- |
| `topics/dcpp_console_logging.dita` | 1.0, -0.67 |
| `topics/whr-converting-templates.dita` | 1.0, -1.0 |
| `topics/dcpp_how_to_enforce_a_number_of_lines_from_the_split_paragraphs_in_the_next_page.dita` | 1.0, -1.0 |
| `topics/dcpp_how_to_display_subtopics_in_toc.dita` | -1.0, 1.0 |

`results/run_log.jsonl`: load 12.0 s, 3.29 GB allocated; each update log line records 4.64 GB. `rl_done` at `2026-10-03T19:54:52`, `train_s` 1292.0. `finished` at `2026-10-03T20:02:47`. An earlier `start` event at `2026-10-03T19:18:06` is in the same file; the run that reached `finished` starts at `2026-10-03T19:29:16`.

Translation scores: `results/judge_fallback.json`. Judge label `fallback-lexical-not-grok`. `mean_faithfulness` is null. `mean_fluency_heuristic` is 5.0 before and after, and every topic's heuristic is 5. The file says that is a Romanian function-word count, not a faithfulness score. The saved canonicalize gloss (`results/baseline_test.json`) still has glossterm `Canonicalize` and the phrases `conversare a datelor` and `standardizare`.

## Cost

These figures are not in `results/metrics.json`. They are the estimates recorded in the writeup committed in `1517f47` (then `docs/report.md`, later `docs/report_ro.md`).

| Item | Recorded value |
| --- | --- |
| Pod | `ix70qe0fixor7g` (dita-rl-poc) |
| GPU | Community RTX 3090, list price $0.22/hr |
| Start | 2026-10-03 19:15:22 UTC |
| Deleted | about 20:04 UTC the same day |
| Wall clock | about 49 minutes |
| Estimated spend | about $0.18 of GPU time, before any small disk fee. Cap was $5 |

The result-file clock is the `19:29:16` to `20:02:47` span in `results/run_log.jsonl`, plus the unused `19:18:06` start. That is the process log, not the rental invoice.

## Surprises

- The held-out counts did not move, including the mean reward at 0.635, while 6 of 10 hypothesis strings still changed.
- 20 of 24 groups tied. Fourteen ties were both already 1.0. Five were both -1.0, so a shared failure also produced no gradient.
- The one schema failure is the 640-token cap. `schema_error_total` stays 0 because xmllint is not called. The earlier writeup's `Premature end of data in tag filepath` line is not in the saved `error_tail`.
- Val, which was not used to select anything, is 4/10 valid-and-skeleton and mean reward -0.167. Four of its failures are the same 640-token cutoff. One not-well-formed val topic ended at 617/617, not at the cap.
- No Grok scores. The fallback file records HTTP 401 `API key expired` for the OpenRouter key and no `XAI_API_KEY`. The heuristic is 5.0 on every row. The canonicalize XML in the result file is still a bad Romanian gloss.
- `results/pod_run.log` records the slow fallback: `causal_conv1d` and `flash-linear-attention` were not installed. `results/metrics.json` has `"surprises": []`. The items above are read from the other result files, not from that empty list.

## Files and commit

Result files: `results/metrics.json`, `results/baseline_test.json`, `results/after_test.json`, `results/train_trace.json`, `results/run_log.jsonl`, `results/pod_run.log`, `results/judge_fallback.json`, `results/adapter/`, `split/manifest.json`.

Commit that added them: `1517f47ab6f6d9500b9a49f98b5721af673d13be` (`Add the DITA structure translation PoC.`).
