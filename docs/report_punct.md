# Report: violation-count reward, English to French

This run keeps the same Oxygen user guide (`oxygenxml/userguide` `db722d7`) and the same fixed system prompt as the previous French run. The reward is no longer a blend of skeleton overlap and a DTD bit. It is minus the number of violations. Zero is perfect. The previous write-up is [report.md](report.md).

## What one violation is

Schema, each occurrence:

- the file is not well-formed (1)
- generation hit the token cap and the skeleton still does not match (1)
- each `xmllint --valid` error line
- each attribute whose value differs, or is missing or extra, on an aligned element
- each child insertion or deletion (a reorder of unlike siblings is one deletion plus one insertion)

Punctuation, each occurrence, same function for the reward, for which sample is kept, and for the test score:

- in prose text, each `?` `!` `:` `;` that is not immediately preceded by U+00A0 (a normal space counts, and so does no space)
- each English double quote (`"`, `“`, `”`) when the source text contains a quotation

`codeblock`, `codeph`, `filepath`, and the other verbatim elements are skipped, so a URL colon inside a code element is not a French-typography miss. If the file is not well-formed, that exclusion cannot run and the regex sees the raw text, including CSS.

## Speed, before the scan

Same short topic, 128 new tokens, greedy, thinking off. NVIDIA-SMI was sampled about three times a second during the call. The 3090 FP16 tensor-core class figure used here is 142 TFLOP/s with sparsity, a ballpark, not a measured spec.

Arithmetic: the checkpoint reports 4,540,838,400 parameters. At 2 FLOPs per parameter per token that is 9.08 GFLOP/token. Roof = 142e12 / 9.08e9 = **15,636 tok/s**. Batch-1 at 21 tok/s is 0.14% of that roof, so a full GPU would still not mean the tensor cores are busy. It would mean the card is stalled on something else.

| Stack | Batch | tok/s | GPU util median | GPU memory |
| --- | ---: | ---: | ---: | ---: |
| transformers NF4, causal-conv1d + fla | 1 | 21.1 | 28% | 5.3 GB |
| same | 4 | 76.6 | 42% | 5.5 GB |
| same | 8 | 148.5 | 58% | 6.1 GB |
| vLLM 0.30 bf16, Triton attention | 1 | 79.9 | 100% | 19.4 GB |
| same | 4 | 285 | 100% | 19.4 GB |
| same | 8 | 512 | 100% | 19.4 GB |

Batch 1 is not compute-saturated. Util stays near 30%, not 100%. A static KV cache in transformers did not help (21.2 tok/s). Batch 8 was the largest that still gained at least 15% and did not OOM, so the scan and the pair of RL samples used batch 8 on the NF4 model. vLLM is faster and it does sit at 100% util, but the GRPO step needs token logprobs from the NF4 LoRA policy, so the training loop stayed in transformers. The first vLLM attempt died because FlashInfer JIT called the image's CUDA 12.4 `nvcc` (`Unknown option '--compress-mode=size'`). Pointing `PATH` at the venv's CUDA 13.4 `nvcc` and setting `VLLM_USE_FLASHINFER_SAMPLER=0` is what produced the table above.

The prebuilt `causal-conv1d` wheel did not load (`torchCheckFail` was compiled with the C++11 string ABI; this torch wheel is the old ABI). Rebuilding from source with `nvcc` on `PATH` and `_GLIBCXX_USE_CXX11_ABI=0` made `causal_conv1d_fn` run. It did not move batch-1 speed off 21 tok/s. The stall is the NF4 decode, not the missing conv kernel.

## Data

| Item | Value |
| --- | --- |
| Pool | 1730 source-valid topics, 400–4500 bytes, sorted path |
| Scanned | 184 greedy translations, then 100 violations, so the scan stopped |
| Clean | 84 |
| Not reached | 783 that would have fit the 896-token cap, plus 763 over that cap |
| Split | seed 1337. Test 10, val 10, train 80. The test 10 were not updated |
| Scan wall | 587 s at batch 8. Printed aggregate throughput during the scan was about 90–125 tok/s |
| Break mix on the 100 | punctuation on 91 topics (82 of them punctuation only), schema on 18 (9 schema only), both on 9 |
| Occurrences on the 100 | punctuation 190, schema 21 |

Punctuation occurrences are almost all missing non-breaking spaces (182). English quotes are 8. Schema occurrences: child-order 8, not well-formed 6, attribute value 5, DTD error lines 2.

## The 10 held-out topics

These 10 were drawn from the 100, so 10/10 with a violation is by construction. The rate on the scan is 100/184.

Baseline latency is the wall clock of the batch-8 call, shared by every topic in that call, so the p50 of 27.0 s is not a single-topic time. The after pass generated one topic at a time.

| | Baseline | After |
| --- | ---: | ---: |
| Topics with a violation | 10 / 10 | 10 / 10 |
| Schema occurrences | 5 | 5 |
| Punctuation occurrences | 34 | 34 |
| Mean violations | 3.9 | 3.9 |
| Schema topics / punct topics | 5 / 9 | 5 / 9 |
| Latency p50 | 27.0 s (batch wall) | 22.0 s (one topic) |

Eight of the ten hypothesis strings are byte-identical. The counts did not change on the other two either.

Per topic, baseline = after for the counts:

| Topic | Schema | Punct | What |
| --- | ---: | ---: | --- |
| dcpp_how_to_change_the_page_dimension | 1 | 16 | not well-formed, CSS colons |
| author-editing-tables-xhtml | 1 | 1 | one attribute value |
| ant-preferences | 1 | 6 | not well-formed |
| uicontrol | 0 | 1 | `Règle :` |
| dcpp_how_to_control_titles_layout | 1 | 1 | not well-formed |
| ch_xml_support | 0 | 3 | prose colon plus `xi:` |
| project-options | 1 | 0 | one child-order mismatch |
| ch_avoiding_page_breaks | 0 | 2 | two prose colons |
| ch_external-links | 0 | 3 | three prose colons |
| dcpp_how_to_center_videos | 0 | 1 | `ceci :` |

## French, read by the same agent

No API. Scores are 1–5. Full notes are in `results/fr_punct/judge.json`. Where before and after are the same string, one score covers both.

| Topic | Faithfulness | French | Guillemets | NBSP |
| --- | --- | --- | --- | --- |
| page dimension | 4 | 3 | n/a | 1 |
| xhtml tables | 3 | 2 | n/a | 2 |
| ant preferences | 4, then 3 | 3 | n/a | 2 |
| uicontrol | 4 | 4 | n/a | 2 |
| titles layout | 4 | 4 | n/a | 2 |
| xml support | 3 | 3 | n/a | 2 |
| project options | 4 | 3, then 4 | n/a | n/a |
| avoiding page breaks | 4 | 4 | n/a | 2 |
| external links | 4 | 4 | 4 | 2 |
| center videos | 5 | 5 | n/a | 2 |

Quoted spans:

- external links is the one place the quotes are actually French: `après le texte « W3C »`. The colons next to it are still a normal space: `lors de leur stylisation :`.
- ant preferences got worse. Before, the options were `Intégré` and `Personnalisé`. After, they are `Built-in` and `Custom` again.
- project options got a small article right: `au niveau de projet` became `au niveau du projet`. The violation count stayed 1.
- xhtml tables still reads `un type HTML de tableau`, `width attribut`, `Auteur mode`.
- xml support says `Soutien XML` and turns `xi:included` into `xi:inclus`.
- the page-size topic's index term does not nest, and `utilisez :` has no non-breaking space.

The non-breaking space was in the reward on every training topic that had a colon. Greedy decoding on the held-out 10 still writes a normal space. Seventeen updates were not enough to move that habit, and most groups never updated because both samples had the same count.

## Method

`Qwen/Qwen3.5-4B`, bitsandbytes NF4, LoRA rank 8 on `q_proj` `k_proj` `v_proj` `o_proj` (1.6M trainable). One fixed system prompt. Thinking forced off. Eval greedy. Training draws two samples at temperature 0.7, top-p 0.8, top-k 20, in one batched generate. Advantage is `(reward - group mean) / group std` on the mean token logprob. AdamW at 5e-6, one step, only when the two violation counts differ.

80 train topics. 17 updated, 63 were flat. Training took 1614 s. The flat groups are usually both −1 or both −2: the two samples miss the same number of non-breaking spaces, so the count gives no winner.

## Money

| Item | Value |
| --- | --- |
| Pod | `54mu4jws1t3933` (dita-fr-punct) |
| GPU | Community RTX 3090, $0.22/hr, host CUDA 13.0, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Start | 2026-10-04 05:45:57 UTC (08:45 Europe/Bucharest) |
| Train done | 2026-10-04 07:01:55 UTC (10:01 Europe/Bucharest) |
| Process exit | 2026-10-04 07:05:05 UTC (10:05 Europe/Bucharest) |
| Deleted | 2026-10-04 07:09 UTC (10:09 Europe/Bucharest). `delete-pod` returned 204. A follow-up `get-pod` returned 404. |
| Estimated spend | about $0.30 at $0.22/hr for 1 hour 23 minutes. Disk is extra and small. Cap for the run was $4. |

## Surprises

- 21 tok/s is not a full 3090. Util sits at 28%. The FLOP roof is about 15,600 tok/s. Batching is what moved throughput, up to 148 tok/s at batch 8, and even then util is only 58%.
- Installing the conv kernel did not change that. The wheel was the wrong C++ ABI. The rebuild works and the speed does not.
- vLLM bf16 on the same card is 80 tok/s at batch 1 and 512 at batch 8, with util pinned at 100%. It never entered the training loop.
- The reward is no longer flat because of a pass/fail DTD bit. It is flat because two samples miss the same colon. 63 of 80 groups tied.
- The greedy test set did not gain a single violation. The only behavior change on it is a regression (`Built-in` / `Custom`) and one better article (`du projet`).
- Guillemets show up (`« W3C »`). The non-breaking space does not, even though it was the bulk of the reward.
