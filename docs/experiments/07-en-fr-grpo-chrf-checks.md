# English to French, GRPO on segments with chrF + pattern-check reward

**The held-out metric moved.** On the 40 sealed test topics, mean topic reward went from **0.4135** (base) to **0.551** (GRPO step 100). Paired bootstrap Δ is +0.1375, 95% CI [0.012, 0.267], with 31 topics up, 6 down and 3 equal. chrF++ against the reference went from **80.01** to **83.04** (Δ +3.04, CI [1.73, 4.44], 32 up and 8 down). Typography findings went from **53** to **3**, and calque findings from **8** to **0**.

These things did **not** improve:
- **Structure:** 27/40 topics were structure-OK before and **26/40** after (Δ −0.025, CI [−0.125, 0.05]).
- **Title case:** findings stayed at **7**.
- **Protected text:** changes went from 12 to 13.
- **Against a regex fixer:** a plain no-break-space regex fixer on the base output scores mean reward **0.5075**. GRPO + the same fixer scores **0.557**, Δ +0.0495, CI [−0.084, 0.184], which is not significant. GRPO's clear wins over that baseline are chrF (+3.04) and calques (8 → 0).
- **Blind judgement:** Sonnet judges preferred GRPO on **22** topics, base on **17**, with **1** tie. A two-sided sign test gives p = 0.52, so this is not significant.

The calque penalty also caused a **visible side effect**. The model now avoids the whole "édit-" word family, including legitimate nouns, and garbles some index terms: "modification de conception de modification" appears 8 times on test.

Numbers come from `results/fr_rl/comparison.json` and `results/fr_rl/blind_ab/result.json`. Results commit: `df4d612` (`Add the GRPO run results: outputs, logs, scores, blind A/B, adapter.`).

This run does not reuse or continue any earlier experiment in this folder. It is a new split, new references, a new reward, a new model (`Qwen/Qwen3-4B-Instruct-2507`) and new code (`fr_rl/`).

## Method

- **Model:** `Qwen/Qwen3-4B-Instruct-2507`, bf16, with LoRA r=16, alpha=32, dropout 0, on q/k/v/o/gate/up/down. That is 33,030,144 trainable parameters (`train_log.jsonl`, `setup` event).
- **Prompt:** the same system prompt (`fr_rl/prompt.py`) for base, training and evaluation. It spells out every rule the reward checks, so the baseline is a prompted baseline, not a naive one.
- **Algorithm:** GRPO, hand-written in `fr_rl/grpo.py`.
  - vLLM 0.11.0 samples 8 completions per prompt for 16 prompts per step, at temperature 1.0.
  - HF transformers 4.57.1 + PEFT 0.17.1 takes one gradient step per batch. The sampling policy is the updated policy, so there is no ratio clipping. There is no KL term.
  - Advantages are (r − group mean) / group std. Groups whose std is below 0.01 are skipped.
  - The loss is a token-level mean over the completion tokens.
  - AdamW, lr 5e-5, betas (0.9, 0.99), grad clip 1.0.
  - After each step the LoRA is saved and hot-loaded into vLLM.
  - vLLM sleep mode offloads its weights to CPU during each update, so both engines fit on a 24 GB card.
- **Reward** (`fr_rl/checks.py:reward`), per segment:
  - −1 if the XML does not parse, if elements/attributes/order/DOCTYPE differ, or if the completion hit the 1024-token cap.
  - −0.5 if it looks untranslated.
  - Otherwise chrF++/100, minus:
    - 0.25 per changed protected element (code-like elements; at most 2 counted)
    - 0.08 per typography finding (at most 4)
    - 0.15 per calque (at most 3)
    - 0.10 per Title Case word in a title (at most 3)
    - 0.2 if the prose length ratio is below 0.75 or above 1.8
  - The result is clipped to [−1, 1].
- **Calque list:** 14 regexes (`CALQUES` in `fr_rl/checks.py`), for example *librairie*, verb *éditer*, *supporter*, *sauver*, *cliquez le*, *le dialogue*.
- **Topic score:** the same function applied to a whole topic, with chrF over the whole topic's prose.
- **Checkpoint choice:** pre-registered in `docs/plans/fr-rl-grpo.md` (commit `b821b29`, before any GPU result) as the best dev mean reward among the evaluated adapters, with step 0 eligible. That was step 100. Test was scored once, with that adapter, greedy decoding.
- **Checker fix:** one change to the checker was made after looking at outputs. It used dev base outputs only, before training started (commit `f6a01b8`). A capitalised UI name right after a generic noun ("boîte de dialogue Historique") no longer counts as a title-case error.

## Data split

From `split/fr_rl/manifest.json`:
- **Source:** oxygenxml/userguide commit `db722d7af0c705cc3d6969f52bd1299a20ba1770`, 2757 `.dita` files.
- **Eligible topics:** 376, defined as standard DITA, 700–3500 bytes, at least 250 prose characters, no conref/conkeyref, at least 2 segments.
- **Test:** 40 whole topics, seed 20261005.
- **Dev:** 16 whole topics.
- **Train:** 1000 block-level segments (`p`, `li`, `title`, `dd`, ...) from other topics, at most 3 per file.
  - 141 candidate segments were dropped because their text also appears in a dev or test topic.
  - One segment, `seg-4515c98b33`, was later excluded because its reference was ambiguous: it lists English conjunctions as examples, and the translator localised them. That leaves **999** train segments.

**References:** French references for every train segment and every dev/test topic were written by Claude Sonnet subagents, one per shard, following `fr_rl/TRANSLATION_BRIEF.md`.
- Each shard was checked by `fr_rl/verify_refs.py`, and I re-ran it myself: all 17 shards had 0 problems. I also read a random sample of 14 train pairs and found them correct.
- **The checker shaped the references.** Some translators reworded titles to clear the title-case check. Example: "Boîte de dialogue de création de patch - ajout de ressources non versionnées", with the dialog name lower-cased (before the UI-name fix above existed). One translator added a no-break space inside a CSS declaration in `<b>` (`border-collapse&#160;: separate`) because the checker flagged the colon.
- **Some reference choices are debatable.** `<keyword outputclass="label">` UI labels were kept in English because the brief lists `keyword` as protected.

Train segments are short: the mean sampled completion was 61–67 tokens per 25-step window. Test items are whole topics, with a mean of 2105.75 source bytes over the 40 (`split/fr_rl/test_topics.jsonl`).

## Numbers

**Test, 40 topics, greedy**: `results/fr_rl/*_test.scored.json`

| System | Mean reward | chrF | Structure OK | Skeleton OK | Protected changed | Typography | Calques | Title case | Clean topics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 0.4135 | 80.01 | 27 | 36 | 12 | 53 | 8 | 7 | 8 |
| Base + regex NBSP fixer | 0.5075 | 80.01 | 27 | 36 | 12 | 1 | 8 | 7 | 20 |
| GRPO step 100 | 0.551 | 83.04 | 26 | 36 | 13 | 3 | 0 | 7 | 19 |
| GRPO step 100 + regex fixer | 0.557 | 83.04 | 26 | 36 | 13 | 0 | 0 | 7 | 20 |

All four systems have 0 untranslated topics and 0 topics that hit the token cap. The fixer leaves 1 typography finding on base, but it has no effect on chrF (the score normalises no-break spaces).

**Paired bootstrap on test**, base → GRPO, 10,000 resamples (`comparison.json`):

| Metric | Δ | 95% CI | Topics up / down / equal |
| --- | ---: | --- | --- |
| reward | +0.1375 | [0.012, 0.267] | 31 / 6 / 3 |
| chrF | +3.038 | [1.73, 4.44] | 32 / 8 / 0 |
| typography count | −1.25 | [−1.725, −0.8] | 0 / 22 / 18 |
| calque count | −0.2 | [−0.35, −0.075] | 0 / 7 / 33 |
| structure OK | −0.025 | [−0.125, 0.05] | 1 / 2 / 37 |
| protected changed | +0.025 | [−0.075, 0.15] | 1 / 1 / 38 |
| title case | 0.0 | [0, 0] | 0 / 0 / 40 |

For the count rows, "up" means more errors.

**Structure changes on test.**
- Two topics broke:
  - `api_faq_highlight_content.dita`: the output is not well-formed. Two attributes lost their closing quote (`<p id="p_p2g_dgk_54b>`).
  - `whr-responsive-override-xslt-processing.dita`: two `<keyword>` labels were translated.
- One topic was fixed: `cf-task-activity.dita`.
- The remaining structure failures are the same kinds the base model makes: translated `<codeph>` literals (`by-topic` → `par-sujet` on dev), translated `<keyword>` labels, translated comments inside `<codeblock>`, and a reordered `<xref>`. Train segments rarely contain these cases, so the reward rarely separated samples on them.

**Dev (16 topics) during training**: `results/fr_rl/run1/summary.json`

| Step | Mean reward | chrF | Structure OK | Typography | Calques |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.3949 | 76.14 | 11 | 29 | 3 |
| 25 | 0.3791 | 77.5 | 10 | 5 | 3 |
| 50 | 0.5296 | 78.71 | 11 | 2 | 2 |
| 75 | 0.4961 | 80.14 | 11 | 2 | 0 |
| 100 | 0.5691 | 80.72 | 11 | 3 | 0 |
| 125 | 0.565 | 79.45 | 11 | 2 | 0 |
| 150 | 0.4965 | 79.98 | 11 | 2 | 0 |
| 175 | 0.5271 | 80.42 | 9 | 2 | 0 |
| 200 | 0.5178 | 79.86 | 12 | 2 | 0 |

Each structure failure costs a topic −1, so one topic moves the 16-topic mean by about 0.1. The dev curve rises over the first 100 steps and is flat to noisy after that.

**Run-to-run noise.** Re-scoring step 100 in the separate final evaluation gives dev 0.5659 / 80.41 (`rl_step100_dev.scored.json`), against 0.5691 / 80.72 inside training. Likewise, base dev is 0.3937 / 76.02 in the standalone run against 0.3949 / 76.14 at step 0. vLLM greedy outputs are not bit-identical across batch layouts, so expect noise of about ±0.3 chrF.

**Training reward on sampled train segments**, 25-step means (`run1/train_log.jsonl`): reward 0.719 → 0.756 → 0.78 → 0.791 → 0.82 → 0.827 → 0.849 → 0.834, and chrF 76.3 → 85.0. One epoch is about 62 steps, so the later windows revisit segments; this curve is not a held-out measure.

**Blind A/B on test**, base vs GRPO step 100, raw outputs without the fixer (`results/fr_rl/blind_ab/`):
- **Setup:** 4 Sonnet subagents with 10 pairs each, A/B order randomised per topic (seed 12345). The key was moved out of the folder while they worked. They were told to read only their pairs file and run nothing. They saw no references.
- **Result:** GRPO 22, base 17, tie 1. A two-sided sign test over 39 non-ties gives p = 0.52.
- **I checked three verdicts against the files:**
  - The GRPO malformed-XML loss on `api_faq_highlight_content.dita` is correct.
  - The base "hourly, daily" → "quotidiennement, quotidiennement" error is correct.
  - The GRPO index-term garbling on `xml-schema-diagram-selector-properties.dita` is correct.
- The judges finished 10 long pairs in about 20–28 seconds each. Their verdicts are a coarse second opinion, not a professional review.

## The calque side effect

The calque regex penalises the **verb** *éditer* (éditer, édite, édité, ...). It does not penalise the noun *édition* or *éditeur*. Even so, on test:

| Pattern | Base | GRPO | Reference |
| --- | ---: | ---: | ---: |
| `\bédit\w*` | 45 | 11 | 19 |
| `\bmodifi\w*` | 41 | 84 | 78 |
| `modification de conception` | 0 | 8 | 0 |

The model learned "avoid édit-" rather than "avoid the verb *éditer*". In the XML Schema diagram topics, the index term *Design editing mode* became "Mode de modification de conception" and "modification de conception de modification". The base had "Mode de conception d'édition". This is reward over-optimisation, and the calque count of 0 hides it. The blind judges flagged it on three topics.

## Cost

RunPod account balance went from **$4.4065** to **$3.5116** (`results/fr_rl/pod_ledger.jsonl`), so **$0.8949** was spent in total. Summing create→terminate × hourly rate per pod in the same ledger gives $0.873. The difference is billing granularity or disk.

| What | Pods | Ledger estimate |
| --- | --- | ---: |
| Training and evaluation pod `ur2t2zx6a6gpza`, community RTX 3090 at $0.22/h | 1 | 143.1 min, $0.525 |
| Rejected or failed pods: slow PyPI (0.06–1 MB/s), driver 565, `cuInit` 999, image pull that never finished, secure pods at $0.49–0.50/h | 19 | $0.348 |

About 40% of the spend went to finding a usable cheap host. Two avoidable mistakes contributed:
- A crashed speed-gate script orphaned one pod for 10.8 minutes.
- An f-string syntax error made several "no supply" retries look like supply problems.

The final gate (`fr_rl/pod_gate.sh`) checks the HF and PyPI download speed, the driver version (570 or newer) and `cuInit` within about 1 minute, and terminates the pod on any failure.

Run time on the training pod:
- Setup: pip install vLLM 0.11.0 and model download. The pod was created at 07:28:43 UTC (ledger). Base outputs were saved locally at 07:37, after one rerun caused by the transformers pin.
- Base eval: 28.9 s for 56 topics.
- Pilot: 3 steps.
- Training: 200 steps. Segment 1 took 1964 s for steps 1–50. Segment 2 took 5324 s for steps 51–200, plus dev evals of about 27 s each.
- Final eval: 33.3 s for 56 topics.
- Step time (generation + update) was 21.5–51.0 s per step. The 25-step means were 4.2–5.0 s for generation and 26.9–33.8 s for the update. vLLM logged sleep/wake at under 1 s each. Peak allocated memory was 19.78 GB.

The reference translations and the blind judges were Claude subagents in this session. They are not part of the RunPod spend.

## Surprises and failures

- **The first training segment crashed** with CUDA OOM at step 51. A micro-batch of 16 times a long completion in the fp32 log-softmax tried to allocate 4.87 GiB. The fix caps micro-batches by rows × longest completion at 3072 tokens (commit `f371ac0`). The run resumed from the step-50 adapter with the same data order. **Adam's moment estimates were not restored**, so steps 51–200 started with a fresh optimizer state. The dev table above spans both segments.
- **vLLM version conflicts on the pods.** The latest vLLM wheel needed a newer driver than the first host had, so I pinned 0.11.0. Installing `peft` then pulled transformers 5.18, which breaks vLLM 0.11, so I pinned 4.57.1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` is incompatible with vLLM sleep mode.
- **Most dev and test structure failures are unchanged by RL.** The reward did not reach them because 999 short segments rarely contain `<codeph>` literals, `<keyword>` labels or `<codeblock>` comments that the model gets wrong.
- **Title-case findings did not move** (7 → 7 on test). Train segments contained 133 titles, but the sampled title-case rate was already near 0 during training: 0.0 to 0.0781 per sample per step, mean 0.0041. The base model mostly gets short segment titles right and errs on whole-topic titles.
- **A regex alone gets most of the typography gain.** RL's typography improvement is real (53 → 3), but a 30-line regex post-processor gets 53 → 1. What RL adds over that is chrF +3.04 and the calques. On calques, a 0 count partly reflects the over-avoidance described above.

## Files

- Code: `fr_rl/`. Plan written before results: `docs/plans/fr-rl-grpo.md`.
- Split and references: `split/fr_rl/`.
- Outputs: `results/fr_rl/base_outputs.jsonl`, `rl_step100_outputs.jsonl`, and the `*_nbspfix_outputs.jsonl` files (dev and test rows).
- Scores: `results/fr_rl/{base,base_nbspfix,rl_step100,rl_step100_nbspfix}_{dev,test}.scored.json`, `comparison.json`.
- Training: `results/fr_rl/run1/train_log.jsonl` (both segments, including the `resume` event), `run1/summary.json`, `run1/dev_step*.jsonl`. Pilot: `results/fr_rl/pilot_train_log.jsonl`.
- Blind A/B: `results/fr_rl/blind_ab/`, which holds the pairs, verdicts, key and `result.json`.
- Adapter: `results/fr_rl/adapter_step100/`. This is a bf16 copy of the fp32 step-100 LoRA, 66 MB; the training run saved fp32. Evaluation used the fp32 file.
- Spend: `results/fr_rl/pod_ledger.jsonl`.
- Results commit: `df4d612`.
