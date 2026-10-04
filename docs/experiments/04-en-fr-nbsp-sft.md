# English to French, supervised non-breaking spaces

Held-out punctuation moved, and it did not reach zero. Schema occurrences stayed **5**. Punctuation occurrences went from **34** to **23**. `bad_high_punct` went from **34** to **23**. Topics with a violation went from **10** to **6**. Topics with any U+00A0 went from **0** to **9**. Four of the ten test topics reached zero violations. Six did not. The non-breaking space was partly learned. This is not a fix.

The before numbers are `before_recomputed_on_saved_baseline_strings` in `results/fr_nbsp_sft/metrics.json`, scored on the saved baseline strings from the violation-count run. `before_saved_metrics` in the same file is that run's stored baseline (schema 5, punctuation 34, topics with a violation 10, `bad_high_punct` 34). This run did not train on these 10 topics. It does not replace that run. That run is [03-en-fr-violations.md](03-en-fr-violations.md).

## Method

Not GRPO. The method string in `results/fr_nbsp_sft/metrics.json` is `supervised causal next-token loss on code-rewritten base greedy French; not GRPO`.

Qwen/Qwen3.5-4B, bitsandbytes NF4 QLoRA, rank 8, alpha 16, targets `q_proj`, `k_proj`, `v_proj`, `o_proj` (the same fields are in `results/fr_nbsp_sft/adapter/adapter_config.json`). Learning rate `2e-05`. `max_epochs` is 3. `epochs_completed` is 1. `optimizer_steps` is 71. `stop_reason` is `train_nbsp_count_dropped`. Thinking is `off`. Decoding is `greedy`. The system prompt stored on the metrics file is the same fixed French DITA prompt as the violation-count run, including guillemets and a non-breaking space before `? ! : ;`.

The training targets are the base model's saved greedy French after code inserted or replaced a space with U+00A0 before `? ! : ;` in prose. The metrics file records `train_insertions` 42 and `train_space_replacements` 83. One training target was still dirty: `train_targets_still_bad_high_punct` is 1.

The loss is ordinary causal next-token loss. There is no group, no advantage, and no reward comparison inside a step.

## Data split

Same 10 held-out topics as the violation-count run. `test_paths` in `results/fr_nbsp_sft/metrics.json` equals `test` in `split/manifest_fr_punct.json`. None of those 10 paths is in `results/fr_nbsp_sft/train_loss.json`. None of the val paths from that manifest are in the loss file either.

| Item | Value |
| --- | --- |
| Train topics in the split | 80 |
| Train topics used | 71 |
| Skipped, no prose edit | 9 |
| Skipped, too long | 0 |
| Epochs completed | 1 of max 3 |
| Optimizer steps | 71 |

The loss file has 71 paths, all inside that 80-topic train list. The metrics file records the skip reason and does not name the 9. The 9 train paths absent from the loss file are:

- `topics/dcpp_how_to_definel_hyphenation_for_a_word.dita`
- `glossary/moderator.dita`
- `glossary/anchor.dita`
- `glossary/foldable-element.dita`
- `glossary/global-options.dita`
- `topics/dcpp_cover_page.dita`
- `topics/dcpp_draft_watermark.dita`
- `glossary/canonicalize.dita`
- `topics/ch_layout.dita`

`results/fr_nbsp_sft/before_recomputed.json` has one field quirk carried from the saved baseline: `glossary/project-options.dita` stores `saved_punct` as `{}` and recomputed `punct` as 0. The recomputed totals still add to punctuation 34 and `bad_high_punct` 34. The empty object is not a second before-count.

## Numbers

Held-out aggregates from `results/fr_nbsp_sft/metrics.json`:

| Metric | Before, recomputed on the saved baseline strings | After |
| --- | ---: | ---: |
| n | 10 | 10 |
| topics_with_violations | 10 | 6 |
| schema_violation_occurrences | 5 | 5 |
| punct_violation_occurrences | 34 | 23 |
| bad_high_punct_occurrences | 34 | 23 |
| topics_with_any_nbsp | 0 | 9 |

`test_topics_gained_at_least_one_nbsp` is 9. The topic that did not is `glossary/project-options.dita` (`has_nbsp` false in `results/fr_nbsp_sft/after_test.json`). It also has no `? ! : ;` in `colon_spans`. Its violation is schema, `child_order` 1, unchanged.

Per topic, before from `results/fr_nbsp_sft/before_recomputed.json`, after from `results/fr_nbsp_sft/after_test.json`:

| Topic | Schema before | Schema after | Punct before | Punct after | After well-formed | After skeleton | After violations | After has NBSP |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| `dcpp_how_to_change_the_page_dimension` | 1 | 1 | 16 | 15 | false | false | 16 | true |
| `author-editing-tables-xhtml` | 1 | 1 | 1 | 0 | true | false | 1 | true |
| `ant-preferences` | 1 | 1 | 6 | 6 | false | false | 7 | true |
| `uicontrol` | 0 | 0 | 1 | 0 | true | true | 0 | true |
| `dcpp_how_to_control_titles_layout` | 1 | 1 | 1 | 0 | false | false | 1 | true |
| `ch_xml_support` | 0 | 0 | 3 | 2 | true | true | 2 | true |
| `project-options` | 1 | 1 | 0 | 0 | true | false | 1 | false |
| `ch_avoiding_page_breaks` | 0 | 0 | 2 | 0 | true | true | 0 | true |
| `ch_external-links` | 0 | 0 | 3 | 0 | true | true | 0 | true |
| `dcpp_how_to_center_videos` | 0 | 0 | 1 | 0 | true | true | 0 | true |

The four topics at zero violations are `uicontrol`, `ch_avoiding_page_breaks`, `ch_external-links`, and `dcpp_how_to_center_videos`. Each is well-formed, skeleton-matched, and `has_nbsp` true. Their `colon_spans` colons all have `prev_is_nbsp` true.

Schema parts on the five topics that still have a schema occurrence, after the update:

| Topic | After parts that are non-zero |
| --- | --- |
| `dcpp_how_to_change_the_page_dimension` | `not_well_formed` 1, `bad_high_punct` 15 |
| `author-editing-tables-xhtml` | `attr_value` 1 |
| `ant-preferences` | `not_well_formed` 1, `bad_high_punct` 6 |
| `dcpp_how_to_control_titles_layout` | `not_well_formed` 1 |
| `project-options` | `child_order` 1 |

`ch_xml_support` has no schema occurrence. It still has `bad_high_punct` 2.

`metrics.json` `after` does not record a latency aggregate. The after rows do. Generate time, in the order above: 25.139 s, 22.992 s, 20.211 s, 10.706 s, 22.126 s, 26.533 s, 12.35 s, 13.465 s, 23.218 s, 6.535 s. The 26.988 s p50 in `before_saved_metrics` is the violation-count run's baseline latency (`latency_device` still says `violation-count reward`). It is not a decode from this run.

### Leftovers

`topics/dcpp_how_to_change_the_page_dimension.dita` is still not well-formed. `error_tail`: `hypothesis not well-formed: mismatched tag: line 12, column 8`. The saved index terms are still `taille<indexterm>Modification` inside an unclosed nest. Punctuation went from 16 to 15. `colon_spans` lists six colons. Five still have `prev_is_nbsp` false, including the prose colon `</xref>, utilisez :` and the CSS fragments `">@page {     size:` and `ze: A4;     margin:` (each CSS fragment is listed twice, once per code block). One prose colon gained U+00A0: `e côté de la page<NBSP>:`. The file is not well-formed, so those code-block colons stay in the count. `bad_high_punct` is 15. The six `colon_spans` entries are not the whole set of 15.

`topics/ant-preferences.dita` is still not well-formed. `error_tail`: `hypothesis not well-formed: mismatched tag: line 10, column 8`. Punctuation stayed 6. The one colon in `colon_spans` did gain U+00A0 (`tions disponibles<NBSP>:`). The saved index term is escaped text, not elements: `Préférences&lt;indexterm&gt;Ant&lt;/indexterm&gt;&lt;/indexterm&gt;`. That line contains six `;` characters. `parts.bad_high_punct` is 6.

`topics/dcpp_how_to_control_titles_layout.dita` lost its punctuation hit (1 to 0) and the colon in `colon_spans` has `prev_is_nbsp` true. It is still not well-formed. `error_tail`: `hypothesis not well-formed: not well-formed (invalid token): line 10, column 36`. The saved index term is `Titres<Mise en page</Mise en page>`.

`topics/ch_xml_support.dita` is well-formed and the skeleton matches. Punctuation went from 3 to 2. The prose colon `is en charge sont<NBSP>:` has U+00A0. The two `xi:` spans do not: `                xi:` and ` les fragments  xi:`, both `prev_is_nbsp` false. The saved title of that section is `xi:include`, and the sentence has `<i>xi:inclus</i>`.

`topics/author-editing-tables-xhtml.dita` lost its punctuation hit. The colon in `colon_spans` has `prev_is_nbsp` true. Schema stayed 1, `attr_value` 1. The saved topic id is `author-edit-tables-xhtml`.

`glossary/project-options.dita` gained no U+00A0. Schema stayed `child_order` 1. The saved sentence still says `au niveau de projet`.

### Train spot check that stopped the run

`stop_reason` is `train_nbsp_count_dropped` after epoch 1, not after the held-out count hit zero. The spot check in `metrics.json` is four train topics, `base_orig_punct_sum_same4` 4, then `bad_high_punct` 0 and `topics_with_nbsp` 4:

| Path | bad_high_punct | has_nbsp | new_tokens |
| --- | ---: | --- | ---: |
| `topics/dcpp_headers_footers_built_in_css.dita` | 0 | true | 151 |
| `topics/dcpp_comments_and_change_tracking___built_in_css.dita` | 0 | true | 141 |
| `topics/ch_advanced_styling_hyphenation_how_to_hide_hyphens.dita` | 0 | true | 218 |
| `topics/configure-mathtype-editor.dita` | 0 | true | 219 |

That check is not the test set. The test set still has 23 punctuation occurrences.

`results/fr_nbsp_sft/train_loss.json` is 71 rows, every row `epoch` 1. `mean_loss_first10` in the metrics file is 0.048. `mean_loss_last10` is 0.0276. `elapsed_train_eval_s` is 315.4. `script_elapsed_s` is 507.2. `gpu` is `NVIDIA GeForce RTX 3090`.

## Cost

Not in the attached JSON. Supplied with the artifacts for this run.

| Item | Stated with the run |
| --- | --- |
| Pod | `evoql06bul63su` |
| Clock | 13:09–13:28 Europe/Bucharest |
| Estimated spend | about $0.07 |
| After the run | deleted |

The times that are in `metrics.json` are the process spans above, 315.4 s and 507.2 s, not the rental clock.

## Surprises

- Schema occurrences did not move (5 to 5). Three test topics are still not well-formed: page dimension, ant-preferences, and titles layout. Titles layout and the XHTML tables topic cleared their punctuation hits and stayed schema-invalid.
- The page-dimension prose colon `utilisez :` is still a normal space. The two `xi:` colons are still ordinary colons. Ant-preferences' listed colon did gain U+00A0, and `bad_high_punct` stayed 6 on a file that is still not well-formed.
- One test topic, `project-options`, has no U+00A0 at all. Its violation is still `child_order` 1.
- The run stopped because a 4-topic train spot check went from 4 punctuation hits to 0. The held-out punctuation count at that stop is 23, not 0.
- One training target was not clean. `surprises.json` and the metrics `surprises` entry, at `2026-10-04T10:18:45`: `target still has bad_high_punct=10 on topics/dcpp_how_to_display_subtopics_in_toc.dita`. That path is one of the 71 loss rows (loss 0.0366). The spans in the message are a code sample and escaped tags, including `map\u00a0:` written with a backslash-u in the log string, then `: &lt;map>`, and further `&lt;` spans.
- The attached result files do not record `causal_conv1d` or `flash-linear-attention`. The note supplied with the artifacts says those kernels were not installed.

## Files and commit

Result files: `results/fr_nbsp_sft/metrics.json`, `results/fr_nbsp_sft/after_test.json`, `results/fr_nbsp_sft/before_recomputed.json`, `results/fr_nbsp_sft/train_loss.json`, `results/fr_nbsp_sft/surprises.json`, `results/fr_nbsp_sft/adapter/adapter_config.json`, `results/fr_nbsp_sft/adapter/adapter_model.safetensors`.

Commit that added them: `a60788493aa8cd462ef872309549b44e5da03409` (`Add the supervised French non-breaking-space run.`).
