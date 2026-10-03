# Report: keep DITA valid while translating

Plain version. The Oxygen user guide on GitHub is English. This run translates topics to Romanian and checks that the XML skeleton and the DITA DTD still accept the file.

## Data

| Item | Value |
| --- | --- |
| Manual | https://github.com/oxygenxml/userguide `db722d7` |
| Files | 2757 `.dita` topics |
| Eligible for this budget | 963. Standard DITA (not Lightweight DITA), source file passes the DTD, size 700–2600 bytes |
| Split | seed 1337. Test 10, val 10, RL update 24, other eligible topics recorded as `train_unused` |
| Pair | English to Romanian. The manual has no Romanian source. |

Test topics are listed in `split/manifest.json` under `test`. They were not used for the reward updates.

## How a topic is scored

Same validator before and after:

1. `xmllint --valid` with the OASIS DITA 1.3 technical-content catalog from `dita-community/org.oasis-open.dita.dita13.doctypes`.
2. Element skeleton: tag name plus attributes, in order, must match the English source. Text may change. Ids and `href`s may not.

Reward used for the update: `1` only when both checks pass. Otherwise a penalty that grows with the validator error count, plus an extra penalty when the skeleton moved. Not-well-formed XML is `-1`.

## Baseline (frozen 10, before the update)

Stack: Qwen/Qwen3.5-4B in bitsandbytes NF4, greedy decoding, on a community RTX 3090. Not CPU llama.cpp. p50 is generate time only.

| Metric | Baseline |
| --- | --- |
| Topics that are not well-formed or fail the DTD | 1 / 10 |
| xmllint errors on saved XML | 1 parser error (the script's own total was 0, see below) |
| Skeleton mismatches | 2 / 10 |
| Valid DTD and same skeleton | 8 / 10 |
| Mean reward | 0.635 |
| Latency p50 | 23.6 s per topic |

The one schema failure is `topics/download-database-drivers.dita`. Generation stopped at the 640-token cap, so the last tag is unclosed. `xmllint` says `Premature end of data in tag filepath`. The training script never called xmllint for that file because the XML parser failed first, so `schema_error_total` in `results/metrics.json` is 0. Counting the parser error, the real validator total is 1.

The other skeleton miss is `topics/author-editing-tables-xhtml.dita`. The DTD still accepts it. The model translated the topic `id` (`author-editing-tables-xhtml` to `author-editare-tabele-xhtml`) and a `conkeyref` path (`reusables-editing-documents` to `reusables-editare-documente`). Those are attribute values, not visible sentences. That is exactly the kind of break a publisher cannot ship.

## After the short RL update (same 10)

| Metric | After |
| --- | --- |
| Not well-formed or DTD-invalid | 1 / 10 (same file, still truncated) |
| xmllint errors on saved XML | 1 (same premature end) |
| Skeleton mismatches | 2 / 10 (same two files) |
| Valid and same skeleton | 8 / 10 |
| Mean reward | 0.635 |
| Latency p50 | 23.4 s per topic |

Six of the ten strings changed a little. The pass/fail counts did not. The val set (never used to pick a checkpoint; there was only one) was worse: 4/10 valid-and-skeleton, mostly the same 640-token cutoff. Treat that as noise on a tiny set, not as a tuned result.

## Translation quality

Grok 4.7 was the planned judge (`x-ai/grok-4.7`, same rubric on baseline and after).

- No `XAI_API_KEY` in the box secret card or in the environment.
- The OpenRouter key on the box was tried and returned HTTP 401 `API key expired`.

So there are no Grok faithfulness or fluency scores.

Fallback, labeled as not Grok (`results/judge_fallback.json`): count Romanian function words (`și`, `în`, `pentru`, `este`, `care`, …) against English function words (`the`, `of`, `with`, …). Every test topic, before and after, landed in the top bucket (heuristic 5). That only says the model switched to Romanian. It does not say the Romanian is correct. A visible miss in the first topic: "conversare a datelor" / "standardizare" for the canonicalize glossary entry.

## Method

Load `Qwen/Qwen3.5-4B` as NF4 (double quant, bf16 compute). LoRA rank 8 on `q_proj`, `k_proj`, `v_proj`, `o_proj` (about 1.6M trainable params, 0.03%). For each of 24 training topics, sample 2 translations at temperature 0.7. If the two rewards differ, one policy-gradient step: advantage is `(reward - group mean) / group std`, loss is minus that times the sum of token logprobs of the sampled tokens, AdamW at 5e-6, grad clip 1. Group size 2 is the smallest GRPO-style update that still has a baseline inside the group. Wall clock cap on the update loop was 40 minutes. It finished in 21.5 minutes: 4 steps actually updated, 20 groups were skipped because both samples got the same reward (usually both already 1).

Prompt: system text says translate English DITA to Romanian, keep tags and attribute values, output XML only. Thinking is turned off with an empty `<think>` block so the 4B model does not spend the token budget on a scratchpad.

## Money

| Item | Value |
| --- | --- |
| Pod | `ix70qe0fixor7g` (dita-rl-poc) |
| GPU | Community RTX 3090, list price $0.22/hr |
| Start | 2026-10-03 19:15:22 UTC |
| Deleted | about 20:04 UTC the same day (confirmed: get-pod returns 404, list-pods is empty) |
| Wall clock | about 49 minutes |
| Estimated spend | about $0.18 of GPU time, before any small disk fee. Cap was $5. |

## What did not get done

- No Grok 4.7 scores. Keys were missing or expired.
- No CPU llama.cpp latency. The GPU was already paid for, and converting this architecture to GGUF was not worth more time.
- No flash-linear-attention / causal_conv1d kernels. The eager PyTorch path is why a short topic takes ~20–30 s.
- Most of the 963 eligible topics were not used for updates. 24 was the budget.
