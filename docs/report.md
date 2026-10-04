# Report: English to French, keep the DITA skeleton

The Oxygen user guide is English. This run translates topics to French and checks that the XML skeleton and the DITA DTD still accept the file. The previous Romanian run is kept in [report_ro.md](report_ro.md).

## Data

| Item | Value |
| --- | --- |
| Manual | https://github.com/oxygenxml/userguide `db722d7` |
| Files | 2757 `.dita` topics |
| Scan pool | 1730 standard topics, source file passes the DTD, 400–4500 bytes. 957 larger files were not scanned. |
| Order | Sorted path. A topic that cannot finish under the token cap is deferred, not counted as a model break. |
| Found | 45 format breaks in 569 translations. The 3 hour scan cap fired before 100. 524 kept the format. 398 fittable topics were not reached. |
| Split of the 45 | seed 1337. Test 10, val 10, train 25. Test was not used for updates. |
| Pair | English to French. |

## What counts as a broken format

The check is order-sensitive. The skeleton is the preorder of (tag, attributes in document order, child signatures). Text may change. These all count as a break:

- not well-formed
- generation hit the token cap before the skeleton matched (truncation)
- DTD invalid (`xmllint --valid`, OASIS DITA 1.3)
- tag name changed
- attribute name, value, or attribute order changed
- child elements under the same parent reordered, added, or removed

A swap of two siblings that differ only in translated text is not a format break. There is nothing structural left to tell them apart. A swap of two `<li id="a">` / `<li id="b">` is `sibling_reorder`.

Reward used for the update is not pass/fail. It is `0.50 * skeleton LCS + 0.20 * attribute recall + 0.15` if well-formed `+ 0.15` if the DTD accepts it, minus `0.30` if truncated. A perfect skeleton scores 1.

## Break reasons (all 45, not just the test 10)

Primary reason, one per topic:

| Reason | Topics |
| --- | --- |
| child_mismatch | 17 |
| not_well_formed | 15 |
| attr_changed | 11 |
| truncated | 1 |
| sibling_reorder | 1 |

A topic can carry more than one label. The multi-label counts are the same except `not_well_formed` 16 and `dtd_invalid` 3 (those three also had another structural reason, so DTD is not their primary label).

## Baseline (the 10 held-out breaks, greedy, before the update)

Stack: Qwen/Qwen3.5-4B, bitsandbytes NF4, greedy, community RTX 3090. p50 is generate time only. These 10 were chosen from the broken set, so a 10/10 break count is by construction, not a random-topic rate. The random-topic rate on this scan was 45/569 (7.9%).

| Metric | Baseline |
| --- | --- |
| Format broken | 10 / 10 |
| Not well-formed or DTD-invalid | 5 / 10 |
| xmllint errors stored | 1 (the other four never reached the DTD because the XML parser failed first) |
| Skeleton mismatches | 10 / 10 |
| Sibling reorders in this 10 | 0 |
| Mean reward | 0.131 |
| Latency p50 | 23.9 s |

Primary reasons on the 10: not well-formed 4, attribute changed 4, child mismatch 2.

## After the RL update (same 10)

| Metric | After |
| --- | --- |
| Format broken | 9 / 10 |
| Not well-formed or DTD-invalid | 4 / 10 |
| xmllint errors stored | 1 |
| Skeleton mismatches | 9 / 10 |
| Mean reward | 0.331 |
| Latency p50 | 23.7 s |

The one file that flipped is `topics/dcpp_how_to_display_subtopics_in_toc.dita`. Before, the index-term tags did not nest. After, the file is well-formed and the skeleton matches. The other nine hypothesis strings are byte-identical to the baseline.

## Translation quality

No Grok, xAI, or OpenRouter call. The same agent that ran the job read the English source and the French XML. Scores are 1–5, or n/a when the source has no quotation and no `? ! : ;`. Full notes are in `results/fr/judge.json`. Nine topics have the same French before and after.

| Topic | Faithfulness | French | Guillemets | NBSP before ?!:; |
| --- | --- | --- | --- | --- |
| dcpp_how_to_control_titles_layout | 4 | 4 | n/a | n/a |
| dcpp_how_to_change_the_page_dimension | 4 | 4 | n/a | n/a |
| ofb-technical-support-page | 3 | 4 | n/a | n/a |
| dcpp_how_to_display_subtopics_in_toc | 4 | 4 | n/a | 3 |
| cf-change-admin-password | 3 | 4 | 4 | 1 |
| canonicalize | 3 | 3 | n/a | n/a |
| preferences-database-filters | 3 | 3 | n/a | 5 |
| moderator | 3 | 4 | n/a | n/a |
| author-editing-tables-xhtml | 3 | 2 | n/a | 1 |
| verifying-signature | 4 | 4 | n/a | 2 |

Quoted misses:

- `cf-change-admin-password`: guillemets are right (`« fusion »`), but the colons are a normal space: `suivez ces étapes :` and `nouveau mot de passe :`. The command sample translated `my-new-password` to `mon-nouveau-mot-de-passe`.
- `author-editing-tables-xhtml`: `un type HTML de tableau`, `d'un XHTML tableau`, `width attribut`, and `<uicontrol>Auteur</uicontrol> mode`. Colon: `tableau XHTML incluent :`.
- `preferences-database-filters`: the id became `preferences-filtres-tables`, and SQL tables became `tableaux`. The list colon is a real NBSP: `tableaux suivants\u00a0:`.
- `moderator`: id `modérateur`, but the glossterm stayed `Moderator`. `configuration du site particulière`.
- `ofb-technical-support-page`: id `ofb-page-d'aide-technique` (apostrophe, DTD rejects it). `Technical Support` became `Aide technique`. The video sentence itself is fine: `telles qu'une démonstration vidéo sur l'installation de Oxygen Feedback`.
- `canonicalize`: the inner `<term>canonicalize</term>` was dropped. `en une standardisation qui respecte la spécification`.
- `verifying-signature`: `<uicontrol>Outils</uicontrol` is missing `>`. `Informations connexes :` has no NBSP.
- `dcpp_how_to_control_titles_layout`: `<indexterm>Titres<Mise en page</indexterm>` is missing `>`.

## Method

Load `Qwen/Qwen3.5-4B` as NF4. LoRA rank 8 on `q_proj`, `k_proj`, `v_proj`, `o_proj` (about 1.6M trainable params). One fixed system prompt for every sample and for the test. It says: translate English DITA to French, output one XML document, keep every element, attribute, attribute value, and child order, translate only human-readable text, use French guillemets, put a non-breaking space before `? ! : ;`, keep the DOCTYPE, no markdown. Thinking is forced off with an empty think block. Eval is greedy.

The scan is greedy and stops at 45 breaks because a 3 hour wall fired (`SCAN_WALL_S=10800`), not because the guide ran out. Training is GRPO-style, group size 2, temperature 0.7, advantage `(reward - group mean) / group std`, one AdamW step at 5e-6 when the two rewards differ. 25 train topics were attempted. 10 groups updated, 15 were flat (both samples scored the same, often both already 1 or both -1). The first training process died on step 8 when `xmllint` wrote a non-UTF-8 byte on stderr. Those 5 updates were not saved. Training was started again from a zero LoRA on the same split, with stderr decoded as `errors=replace`. That second pass is the one in `results/fr/metrics.json`. It took 1089 seconds.

## Money

| Item | Value |
| --- | --- |
| Pod | `7qrdee52y372iy` (dita-fr-poc) |
| GPU | Community RTX 3090, list price $0.22/hr, CUDA 13.0 host, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Start | 2026-10-03 20:52:22 UTC (23:52 Europe/Bucharest) |
| Deleted | 2026-10-04 01:02 UTC (04:02 Europe/Bucharest). `delete-pod` returned 204. |
| Wall clock | about 4 hours 10 minutes |
| Estimated spend | about $0.92 of GPU time at the list price, before a small disk fee. Cap was $5. |

## What did not get done

- 100 broken topics. The break rate was about 8%, and the in-process 3 hour scan cap fired at 45. The usable pool was not exhausted (398 topics that fit the 896-token cap were never translated). Budget was not the limit.
- `causal_conv1d` and `flash-linear-attention` were not installed. The build tried to pull a second Torch into an isolated build env, so it was killed. Generation stayed on the slow PyTorch path, about 20 tokens/s, p50 about 24 s.
- Val was not decoded. It is only a held-out list.
- Punctuation is in the prompt and in the human scores. It is not in the RL reward, so the update had no reason to learn the NBSP rule.
