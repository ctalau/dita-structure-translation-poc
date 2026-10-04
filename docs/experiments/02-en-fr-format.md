# English to French, train on format breaks

The scan did not reach 100 breaks. It stopped at **45** because the time cap fired.

Held-out format-broken went from **10/10** to **9/10**. Nine of the ten hypothesis strings are byte-identical to the baseline, so those nine stayed broken. Mean reward went from **0.1305** to **0.3305**. That difference is the one topic that flipped: `topics/dcpp_how_to_display_subtopics_in_toc.dita` went from reward -1.0 to 1.0, and 2.0/10 = 0.2. The other nine rewards are unchanged.

Punctuation (French guillemets and a non-breaking space before `? ! : ;`) was in the system prompt and not in the reward. It was not learned. The first training process died on a non-UTF-8 xmllint stderr. The saved adapter is a second run that started from a fresh LoRA.

## Method

Qwen/Qwen3.5-4B, bitsandbytes NF4, QLoRA rank 8, alpha 16, targets `q_proj`, `k_proj`, `v_proj`, `o_proj` (`results/fr/adapter/adapter_config.json` and `results/fr/metrics.json`). The French result files do not print a trainable-parameter count. Eval is greedy. The system prompt is the string stored on `results/fr/metrics.json`. It tells the model to keep the DITA skeleton, use French guillemets, and insert a non-breaking space (U+00A0 or `&nbsp;`) before `? ! : ;`.

What the scan calls a format break is in `split/manifest_fr.json`: not well-formed, truncated before the skeleton matches, DTD-invalid, tag or attribute change (including attribute order), or a child insertion, deletion, or reorder. Two same-shape siblings that differ only in translated text are not a break.

The training reward is not that pass/fail bit. `scripts/dita_reward.py` `score_pair` (the function this run's resume script calls) is `0.50 * skeleton LCS + 0.20 * attribute recall + 0.15` if well-formed `+ 0.15` if the DTD accepts it, minus `0.30` if truncated, clamped to [-1, 1], and forced to 1.0 when the format is not broken. `punct_stats` is a separate counter. Its docstring says it is not part of the RL reward.

GRPO-style, group size 2, temperature 0.7, top-p 0.8, top-k 20. Advantage `(reward - group mean) / group std`. This run multiplies that by the sum of completion-token logprobs (`tok_lp.sum()` in `scripts/run_fr_poc.py`). One AdamW step at `5e-6` when the two rewards differ. The saved adapter is the second pass, `scripts/resume_fr_train.py`, which imports that module. Its module docstring says the first pass's five updates were not checkpointed and this start is a zero LoRA. `SCAN_WALL_S` recorded on the run is 10800. `TRAIN_WALL_S` recorded on the run is 3000. `TARGET_BROKEN` is 100.

`results/fr/run_log.jsonl` records the kernels at start: `ModuleNotFoundError: No module named 'causal_conv1d'` and the same for `fla`.

## Data split

Source: `oxygenxml/userguide` `db722d7af0c705cc3d6969f52bd1299a20ba1770`. Pool file `split/fr_candidates.json`: 2757 `.dita` files, 1730 candidates (400–4500 bytes, standard DITA, source passes the DTD). Skipped histogram in that file: `too_big_for_scan_pool` 957, `too_small` 40, `lightweight` 29, `no_doctype` 1.

Scan and split, from `split/manifest_fr.json` and the `scan_done` event in `results/fr/run_log.jsonl`:

| Item | Value |
| --- | --- |
| Order | Sorted path. A topic that cannot finish under the phase cap is deferred, not counted as a model break |
| Phase caps | 640, then 896 |
| stop_reason | `scan_wall` |
| elapsed_s | 10825.9 |
| Scanned | 569 |
| Kept the format | 524 |
| Format breaks | 45 |
| Not reached | 398 |
| Too big for the 896 cap | 763 |
| Seed | 1337 |
| Test / val / train | 10 / 10 / 25 |

Phase 1, same log: at `2026-10-04T00:16:12`, cap 640, broken 41, scanned 536. `scan_done` at `2026-10-04T00:33:06` is the 45 and 569 above. 45/569 is the scan rate. The test 10 were drawn from the 45, so 10/10 broken at baseline is how the split was built.

Primary reason on all 45, one per topic, from `results/fr/scan_broken.json`: `child_mismatch` 17, `not_well_formed` 15, `attr_changed` 11, `truncated` 1 (`topics/ch_css_units.dita`), `sibling_reorder` 1 (`topics/xml-editor-specific-actions.dita`). Multi-label counts: `not_well_formed` 16, `dtd_invalid` 3, and the other primary labels unchanged. The three DTD labels sit on topics whose primary reason is something else.

## Numbers

Held-out, from `results/fr/metrics.json`:

| Metric | Baseline | After |
| --- | ---: | ---: |
| format_broken | 10 | 9 |
| schema_invalid_topics | 5 | 4 |
| schema_error_total | 1 | 1 |
| skeleton_mismatch_topics | 10 | 9 |
| sibling_reorder_topics | 0 | 0 |
| truncated_topics | 0 | 0 |
| valid_and_skeleton_ok | 0 | 1 |
| mean_reward | 0.1305 | 0.3305 |
| latency_p50_s | 23.932 | 23.654 |
| latency_mean_s | 20.824 | 20.521 |

Primary reasons on the 10, baseline: `not_well_formed` 4, `attr_changed` 4, `child_mismatch` 2. After: `not_well_formed` 3, `attr_changed` 4, `child_mismatch` 2.

The one string that changed is `topics/dcpp_how_to_display_subtopics_in_toc.dita`. Before: `not_well_formed`, reward -1.0, `punct.text_available` false. After: `format_broken` false, reward 1.0, `nbsp_ok_marks` 0, `bad_high_punct` 1, `guillemet_open` 0, `guillemet_close` 0, `ascii_double_quote` 8. The format fix did not produce a non-breaking space.

Punctuation on the nine unchanged strings (`punct` on each baseline row; after is the same string):

| Topic | text | guillemets « » | nbsp_ok_marks | bad_high_punct |
| --- | --- | ---: | ---: | ---: |
| `dcpp_how_to_control_titles_layout` | not well-formed |  |  |  |
| `dcpp_how_to_change_the_page_dimension` | not well-formed |  |  |  |
| `ofb-technical-support-page` | yes | 0 / 0 | 0 | 0 |
| `cf-change-admin-password` | yes | 1 / 1 | 0 | 2 |
| `canonicalize` | yes | 0 / 0 | 0 | 0 |
| `preferences-database-filters` | yes | 0 / 0 | 1 | 0 |
| `moderator` | yes | 0 / 0 | 0 | 0 |
| `author-editing-tables-xhtml` | yes | 0 / 0 | 0 | 1 |
| `verifying-signature` | not well-formed |  |  |  |

`cf-change-admin-password` has one opening and one closing guillemet and two high-punctuation marks with no non-breaking space. `preferences-database-filters` is the one row with `nbsp_ok_marks` 1. Those counts are the baseline generation. Training did not change them.

Human scores are in `results/fr/judge.json` (the agent that ran the job, no external API). The file says nine of the ten after-strings are byte-identical to the baseline. That matches the XML comparison.

### Crash, then a fresh LoRA

`results/fr/crash.txt` is a `UnicodeDecodeError`: `'utf-8' codec can't decode byte 0xa8 in position 621: invalid start byte`, raised in `dtd_validate` while reading xmllint stderr. The file does not name the topic.

`results/fr/run_log.jsonl` records the first training process through step 7, with `updates` reaching 5 (`rl_step` lines at steps 0, 3, 4, 5, 6). The last event before the resume is `rl_flat` step 7 on `topics/dcpp_cover_page.dita`, rewards `[0.9146, 0.9146]`. The next path in `split/manifest_fr.json` `train` is `topics/dcpp_how_to_remove_entries_from_the_toc.dita`. The earlier prose log names that file as the crash site. `crash.txt` does not, so this document does not treat the filename as part of the traceback.

`resume_train` at `2026-10-04T00:39:58` says `scan not repeated; LoRA restarted from zero because the crash left no adapter`. The saved trace and `results/fr/metrics.json` are that second pass: `"resume": true`, `updates` 10, `skipped_flat_reward` 15, `train_topics_attempted` 25. `rl_done` at `2026-10-04T00:58:12`, `train_s` 1089.5. `finished` at `2026-10-04T01:01:40`, `after_broken` 9.

The second-pass ties in `results/fr/train_trace.json`: `(-1.0, -1.0)` five times, `(1.0, 1.0)` four times, and one each of `(0.9643, 0.9643)`, `(0.9146, 0.9146)`, `(0.9476, 0.9476)`, `(0.9778, 0.9778)`, `(0.9552, 0.9552)`, `(0.9361, 0.9361)`. Fifteen groups, including groups that were both already 1.0 and groups that were both -1.0.

## Cost

Not in `results/fr/metrics.json`. Recorded in the writeup and run log committed in `69c81a3`.

| Item | Recorded value |
| --- | --- |
| Pod | `7qrdee52y372iy` (dita-fr-poc) |
| GPU | Community RTX 3090, list price $0.22/hr, CUDA 13.0 host, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Start | 2026-10-03 20:52:22 UTC |
| Deleted | 2026-10-04 01:02 UTC. The writeup says `delete-pod` returned 204 |
| Wall clock | about 4 hours 10 minutes |
| Estimated spend | about $0.92 of GPU time at the list price, before a small disk fee. Cap was $5 |

Process times that are in `results/fr/run_log.jsonl`: first `start` `2026-10-03T21:15:40` (load 376.5 s), second `start` `2026-10-03T21:32:34` (load 1.8 s, this is the scan), `scan_done` `2026-10-04T00:33:06`, `finished` `2026-10-04T01:01:40`. The earlier prose log says the first upload archived a symlink to `/tmp/userguide` and the topic path was missing. That sentence is not an event in the jsonl. The jsonl does show two starts, and only the second one reaches `scan_done`.

## Surprises

- The 10800 s scan cap fired at 10825.9 s with 45 breaks, not 100. 398 topics that fit the 896-token cap were not translated. 524 of 569 scanned topics already kept the format.
- The saved training run is the second one. Five updates were logged on the first process and were not in the adapter. The second process started from a zero LoRA on the same split.
- On that second pass, 15 of 25 groups tied, including both samples at 1.0, both at -1.0, and identical partial scores (`0.9643` twice, `0.9361` twice, and the other pairs listed above).
- The held-out set stayed 9/10 format-broken. The mean-reward change is one topic. Punctuation counts on the nine unchanged topics are the baseline counts.
- `causal_conv1d` and `fla` were missing for this run. The later violation-count run is a different document.
- Val was not decoded. `results/fr/metrics.json` has no `val_after`. `"surprises": []` in that file. The items above are read from the other result files.

## Files and commit

Result files: `results/fr/metrics.json`, `results/fr/baseline_test.json`, `results/fr/after_test.json`, `results/fr/scan_broken.json`, `results/fr/train_trace.json`, `results/fr/run_log.jsonl`, `results/fr/crash.txt`, `results/fr/judge.json`, `results/fr/adapter/`, `split/manifest_fr.json`, `split/fr_candidates.json`.

Commit that added them: `69c81a36bb8f426a6c267214738d5aef7f9f7e36` (`Add the English-to-French DITA structure run.`).
