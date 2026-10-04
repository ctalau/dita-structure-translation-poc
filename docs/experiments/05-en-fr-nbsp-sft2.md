# English to French, continued non-breaking spaces on new topics

Held-out punctuation moved down and did not reach zero. GRPO gave back the schema improvement.

On the same 10 topics, prose punctuation occurrences were **39**, then **33** after the next-token pass, then **28** after GRPO. Missing U+00A0 hits are the same three numbers: **39**, **33**, **28**. Schema occurrences were **2**, then **0**, then **2**. Topics still violated were **10**, then **9**, then **9**. Topics with any U+00A0 were **7**, then **7**, then **8**. One topic is at zero violations. Nine are not. This is not a success.

The schema **2** at the start and the schema **2** after GRPO are not the same topic. The start **2** is `child_order` 2 on `topics/ch_advanced_styling_multiple_before_and_after_pseudo_elements.dita`. That topic is schema 0 after the next-token pass and still schema 0 after GRPO. The **2** after GRPO is `child_order` 2 on `topics/xml-schema-diagram-override-properties.dita`, which was schema 0 at the start and after the next-token pass.

This run does not replace the violation-count run. That run is [03-en-fr-violations.md](03-en-fr-violations.md). It continued from the experiment-04 French NBSP LoRA. `continued_from` in `results/fr_nbsp_sft2/metrics.json` is `/workspace/fr_nbsp_sft2/start_adapter`. That earlier run's writeup is not in this branch.

The task note for this run said the next-token pass was about 16 seconds of training. `sft_elapsed_s` in `results/fr_nbsp_sft2/metrics.json` is **1048.6**. Those disagree. The file wins.

## Method

Qwen/Qwen3.5-4B, bitsandbytes NF4 QLoRA, rank 8, alpha 16, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`. Those fields are in both `results/fr_nbsp_sft2/adapter_after_sft/adapter_config.json` and `results/fr_nbsp_sft2/adapter_after_grpo/adapter_config.json`. The two config files are byte-identical. The two `adapter_model.safetensors` files are not. `trainable_params` is 1572864. `lr_sft` is `2e-05`. The metrics file does not record a GRPO learning rate. Thinking is `off`.

The reward string in `results/fr_nbsp_sft2/metrics.json` is: minus (schema occurrences + missing U+00A0 before `?!:;` in prose). English quotes are counted but not added to the reward. `skip_tags` are `codeblock` and `codeph`. `attributes_skipped` is true. `codeblock_codeph_colons_excluded` is true.

The system prompt stored in that file says to insert a non-breaking space (U+00A0 or the entity `&nbsp;`) before `? ! :` and `;`.

New topics only. The scan was filtered to prose punctuation outside `codeblock` and `codeph`, and attributes were ignored. `results/fr_nbsp_sft2/surprises.json` says the search hit 28 min. `search_scanned` is 48, `search_clean` is 30, `search_train_queue` is 8. The train queue is the 8 topics in `results/fr_nbsp_sft2/sft_trace.json`.

The next-token pass is 8 steps. Every row in the SFT trace has `rewrite_mode` `element_scan_wellformed`. `sft_insertions` is 15. `sft_space_replacements` is 0. `sft_skipped_no_edit`, `sft_skipped_rewrite_still_bad`, `sft_skipped_long`, and `sft_oom` are 0. `sft_elapsed_s` is 1048.6. `sft_eval_missing` is false.

GRPO then ran on those same 8 topics. Each row in `results/fr_nbsp_sft2/grpo_trace.json` has two rewards, so the group size is 2. `grpo_groups` is 8. `grpo_updates` is 2. `grpo_ties` is 6. `grpo_ties_both_missing_nbsp` is 5. `grpo_ties_identical_missing_spans` is 4. `grpo_ties_both_zero` is 1. `grpo_oom` is 0. `grpo_elapsed_s` is 676.1. `grpo_eval_missing` is false.

## Data split

`n_candidates_with_prose_punct` is 1324. `n_fresh_pool` is 431. The same two counts are in `results/fr_nbsp_sft2/candidate_pool.json`, with `scan_fail` 0.

`used_old10_as_heldout` is false. `old10_all_initial_punct_errors` is false. `old10_all_source_candidates` is false. The surprise line is: `old 10 not used as held-out: all_initial_punct_errors=False all_source_prose_punct=False n_scored=10`.

The held-out 10, from `held_out_paths`, are not the previous test 10, and none of them is in the SFT trace or the GRPO trace:

- `topics/ch_advanced_styling_multiple_before_and_after_pseudo_elements.dita`
- `topics/oxy-platform.dita`
- `topics/oxy-is-editable-element.dita`
- `topics/editing-JSON.dita`
- `topics/xml-schema-diagram-override-properties.dita`
- `topics/xml-schema-diagram-include-properties.dita`
- `topics/xml-schema-diagram-redefine-properties.dita`
- `topics/dita-pdf-output.dita`
- `topics/oxy-selected-elements.dita`
- `topics/oxy-current-selected-element.dita`

Every start row has `prose_punct` greater than 0. All 30 eval rows are `well_formed` true, `dtd_valid` true, `truncated` false, `hit_token_cap` false, and `punct_best_effort` false. `english_quotes_not_in_reward` is 0 on every row. `topics_best_effort_punct` is 0 at all three stages.

## Numbers

Held-out aggregates from `results/fr_nbsp_sft2/metrics.json`:

| Metric | Start | After SFT | After GRPO |
| --- | ---: | ---: | ---: |
| n | 10 | 10 | 10 |
| schema_violation_occurrences | 2 | 0 | 2 |
| prose_punct_occurrences | 39 | 33 | 28 |
| missing_nbsp_hits | 39 | 33 | 28 |
| topics_still_violated | 10 | 9 | 9 |
| topics_with_any_nbsp | 7 | 7 | 8 |
| topics_best_effort_punct | 0 | 0 | 0 |
| english_quotes_not_in_reward | 0 | 0 | 0 |

Per topic, from `results/fr_nbsp_sft2/start_test.json`, `results/fr_nbsp_sft2/after_sft_test.json`, and `results/fr_nbsp_sft2/after_grpo_test.json`. Counts are schema, then prose punctuation.

| Topic | Start | After SFT | After GRPO | `has_nbsp` start / SFT / GRPO |
| --- | ---: | ---: | ---: | --- |
| `ch_advanced_styling_multiple_before_and_after_pseudo_elements` | 2, 6 | 0, 4 | 0, 4 | true / true / true |
| `oxy-platform` | 0, 3 | 0, 3 | 0, 3 | true / true / true |
| `oxy-is-editable-element` | 0, 4 | 0, 4 | 0, 4 | true / true / true |
| `editing-JSON` | 0, 4 | 0, 0 | 0, 0 | true / true / true |
| `xml-schema-diagram-override-properties` | 0, 5 | 0, 5 | 2, 0 | false / false / true |
| `xml-schema-diagram-include-properties` | 0, 5 | 0, 5 | 0, 5 | false / false / false |
| `xml-schema-diagram-redefine-properties` | 0, 5 | 0, 5 | 0, 5 | false / false / false |
| `dita-pdf-output` | 0, 1 | 0, 1 | 0, 1 | true / true / true |
| `oxy-selected-elements` | 0, 3 | 0, 3 | 0, 3 | true / true / true |
| `oxy-current-selected-element` | 0, 3 | 0, 3 | 0, 3 | true / true / true |

The one topic at zero violations is `editing-JSON`. It reached 0 at the next-token pass and the GRPO string is identical to the after-SFT string. U+00A0 count inside that `hyp_xml` went from 1 to 5. The four start spans were ordinary spaces: `Vidéo :`, `Vidéo :`, `Webinaire :`, `Webinaire :`. After SFT each of those four colons is preceded by U+00A0.

The 39 to 33 drop is two topics. `editing-JSON` went from 4 to 0. The styling topic went from 6 to 4, and its schema went from `child_order` 2 to 0 (`skeleton_match` false to true). The only string change on that topic is one list item: `éléments pseudo- :before ou :after normaux` became `éléments pseudo- normaux <codeph>:before</codeph> ou <codeph>:after</codeph>`. The U+00A0 count inside that `hyp_xml` stayed 2. The two colons that left the prose count moved into `codeph`, which `skip_tags` excludes. The other four colons stayed, each with `prev` `" "` and `prev_is_nbsp` false:

- `ments pseudo- CSS :`
- `CSS :before(n) et :`
- `  élément pseudo- :`
- `  élément pseudo- :`

The 33 to 28 drop is one topic. `xml-schema-diagram-override-properties` went from prose 5 to 0 and from schema 0 to 2. `has_nbsp` went from false to true. U+00A0 count inside that `hyp_xml` went from 0 to 5. `skeleton_match` went from true to false. `parts_schema` after GRPO is `child_order` 2 and zeros elsewhere. `well_formed` stayed true and `dtd_valid` stayed true. The five `xs:` spans became `xs` plus U+00A0 plus `:`. The string also dropped both `<i>` wrappers: `Le composant <i>xs:override</i>` became `Le composant xs` + U+00A0 + `:override`, and `Propriétés du <i>xs:override</i>` became `Propriétés xs` + U+00A0 + `:override`. `La construction override` became `La construction d'override`.

Seven topics did not change punctuation count at any stage: `oxy-platform` 3, `oxy-is-editable-element` 4, `xml-schema-diagram-include-properties` 5, `xml-schema-diagram-redefine-properties` 5, `dita-pdf-output` 1, `oxy-selected-elements` 3, `oxy-current-selected-element` 3.

Four `hyp_xml` strings are byte-identical at all three evals: `oxy-platform`, `xml-schema-diagram-redefine-properties`, `oxy-selected-elements`, `oxy-current-selected-element`.

`oxy-is-editable-element` changed one paragraph at the next-token pass (`Il n'a qu'un seul argument...` became `Elle ne possède qu'un argument...`) and the four colon spans stayed. GRPO left that string as it was.

`xml-schema-diagram-include-properties` changed one cell at GRPO, `Nom de propriété` to `Nom de la propriété`. Prose stayed 5. `has_nbsp` stayed false. The five spans are still `xs:` with `prev` `s`.

`dita-pdf-output` changed wording at both stages and the punctuation count stayed 1. The leftover span at every stage is `un processeur  xsl:` with `prev` `l` and `prev_is_nbsp` false. U+00A0 count inside that `hyp_xml` stayed 4.

`xml-schema-diagram-include-properties` and `xml-schema-diagram-redefine-properties` are the two topics that still have no U+00A0 after GRPO. Each still has five `xs:` spans, `prev` `s`.

The glued colons that are still missing U+00A0, `prev_is_nbsp` false:

| Topic | After GRPO spans | `prev` |
| --- | --- | --- |
| `oxy-platform` | `Fonction oxy:`, `fonction oxy:`, `Exemple<NBSP>:  oxy:` | `y`, `y`, `y` |
| `oxy-is-editable-element` | `Fonction oxy:`, `fonction oxy:`, `ns une section  xi:`, `Exemple<NBSP>:  oxy:` | `y`, `y`, `i`, `y` |
| `oxy-selected-elements` | `Fonction oxy:`, `fonction oxy:`, `Exemple<NBSP>:  oxy:` | `y`, `y`, `y` |
| `oxy-current-selected-element` | `Fonction oxy:`, `fonction oxy:`, `Exemple<NBSP>:  oxy:` | `y`, `y`, `y` |
| `xml-schema-diagram-include-properties` | `xs:`, `mode Composants xs:`, `tion Composants xs:`, `Le composant  xs:`, `Propriétés  xs:` | `s` |
| `xml-schema-diagram-redefine-properties` | `xs:`, `mode Composants xs:`, `tion Composants xs:`, `Le composant  xs:`, `Propriétés de  xs:` | `s` |
| `dita-pdf-output` | `un processeur  xsl:` | `l` |

`colon_examples_start_vs_last_eval` in the metrics file is start versus the last eval. On the styling topic it records `before_missing` 6, `after_missing` 4, `before_schema` 2, `after_schema` 0. On `editing-JSON` it records 4 then 0. On `xml-schema-diagram-override-properties` it records `before_missing` 5, `after_missing` 0, `before_schema` 0, `after_schema` 2. Those match the row files.

## Training

SFT trace, 8 steps, none of them a held-out path. `n_repl` is 0 on every row. Insertions sum to 15.

| Topic | loss | n_ins | tokens | source_prose_punct |
| --- | ---: | ---: | ---: | ---: |
| `topics/oxy-is-required-element.dita` | 0.049 | 3 | 758 | 4 |
| `topics/ch_graphics_images.dita` | 0.0438 | 1 | 1233 | 4 |
| `topics/sdk-open-custom-protocal.dita` | 0.0522 | 1 | 1181 | 4 |
| `topics/XSLT-packages.dita` | 0.0582 | 4 | 1230 | 4 |
| `topics/saxon-issues.dita` | 0.0484 | 1 | 1453 | 4 |
| `topics/dg-subject-selector.dita` | 0.0405 | 2 | 1418 | 4 |
| `topics/user-entry-sqf-operation.dita` | 0.0351 | 2 | 1457 | 4 |
| `topics/dg-base-uri-function.dita` | 0.0251 | 1 | 977 | 3 |

GRPO trace. Two rows updated.

| Topic | Updated | Rewards | missing U+00A0 | Schema | identical_spans |
| --- | --- | --- | --- | --- | --- |
| `topics/oxy-is-required-element.dita` | no | -3.0, -3.0 | 3, 3 | 0, 0 | true |
| `topics/ch_graphics_images.dita` | yes | -3.0, -1.0 | 1, 1 | 2, 0 | false |
| `topics/sdk-open-custom-protocal.dita` | no | -1.0, -1.0 | 1, 1 | 0, 0 | true |
| `topics/XSLT-packages.dita` | no | -4.0, -4.0 | 4, 4 | 0, 0 | false |
| `topics/saxon-issues.dita` | no | -1.0, -1.0 | 1, 1 | 0, 0 | true |
| `topics/dg-subject-selector.dita` | yes | -1.0, -2.0 | 1, 2 | 0, 0 | false |
| `topics/user-entry-sqf-operation.dita` | no | -2.0, -2.0 | 2, 2 | 0, 0 | true |
| `topics/dg-base-uri-function.dita` | no | 0.0, 0.0 | 0, 0 | 0, 0 | true |

The five ties that both still miss the non-breaking space are `oxy-is-required-element`, `sdk-open-custom-protocal`, `XSLT-packages`, `saxon-issues`, and `user-entry-sqf-operation`. Four of those have `identical_spans` true. `XSLT-packages` is the tie with the same missing count and `identical_spans` false. The sixth tie, `dg-base-uri-function`, is 0 and 0. The two updates are `ch_graphics_images` (rewards -3.0 and -1.0, schema 2 and 0, missing U+00A0 1 and 1; `advs` `-0.9999990000010001` and `0.9999990000010001`) and `dg-subject-selector` (rewards -1.0 and -2.0, missing U+00A0 1 and 2, schema 0 and 0; `advs` `0.999998000004` and `-0.999998000004`). Neither updated path is in the held-out 10.

## Cost

`results/fr_nbsp_sft2/metrics.json` has no cost field, no pod id, and no clock window. It records `gpu` `NVIDIA GeForce RTX 3090 Ti` and `pod_elapsed_s` **3753.2**.

The task note for this run says pod `u09noxe12eayqc`, a community RTX 3090 Ti, 14:13–15:18 Europe/Bucharest, cost $0.27, deleted. Those four items are not in the attached JSON.

`sft_elapsed_s` is 1048.6. `grpo_elapsed_s` is 676.1. The surprise file records `search hit 28 min` at `2026-10-04T11:59:00`, and the two earlier surprise lines at `2026-10-04T11:30:58`. The file does not label a timezone on those timestamps.

## Surprises

- Punctuation moved down and stopped at 28. It did not reach zero. Nine of the ten topics are still violated. The one at zero is `editing-JSON`.
- Schema went 2 to 0 to 2. The return to 2 is `child_order` on `xml-schema-diagram-override-properties`, with `skeleton_match` false, after that topic's five `xs:` colons gained U+00A0. The styling topic's `child_order` 2 stayed cleared.
- On the styling topic, the prose drop from 6 to 4 is two colons moving into `codeph`. The U+00A0 count in that string stayed 2. Four colons still have an ordinary space in front of them.
- Five of the six GRPO ties are both samples still missing the non-breaking space. Four of those five have identical spans. One tie is both zero. Two groups updated. The held-out set after those updates is still 9/10 violated.
- The search stopped at the line `search hit 28 min`, with 8 topics in the train queue. The old 10 were not the held-out set.
- `include` and `redefine` still contain no U+00A0. Their `xs:` colons, the `oxy:` and `xi:` colons, and the `xsl:` colon are still counted, with `prev` a letter.
- The task note's "about 16 seconds" disagrees with `sft_elapsed_s` 1048.6.
- `"surprises"` in the metrics file is the three lines also stored in `results/fr_nbsp_sft2/surprises.json`. The items above are read from the other result files.

## Files and commit

Result files: `results/fr_nbsp_sft2/metrics.json`, `results/fr_nbsp_sft2/start_test.json`, `results/fr_nbsp_sft2/after_sft_test.json`, `results/fr_nbsp_sft2/after_grpo_test.json`, `results/fr_nbsp_sft2/sft_trace.json`, `results/fr_nbsp_sft2/grpo_trace.json`, `results/fr_nbsp_sft2/surprises.json`, `results/fr_nbsp_sft2/candidate_pool.json`, `results/fr_nbsp_sft2/adapter_after_sft/`, `results/fr_nbsp_sft2/adapter_after_grpo/`.

Commit that added them: `0a9367e48704829056c23e7d268e77629d24c07c` (`Add the continued French NBSP run on new topics.`).
