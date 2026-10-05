# Plan: GRPO for English→French DITA topics (written before any GPU result)

This plan was committed before the baseline or any training run was scored.
The experiment document comes after the run, as AGENTS.md requires.

## What is measured

Held-out set: the 40 whole topics in `split/fr_rl/test_topics.jsonl`. They are
never used for training, for picking hyperparameters or for picking a checkpoint.
Dev set: the 16 whole topics in `split/fr_rl/dev_topics.jsonl`, used to monitor
training and to choose the checkpoint.

Per topic, from `fr_rl/checks.py`:
- structure: XML parses, same elements/attributes/order, same DOCTYPE, protected
  code-like text unchanged
- typography: missing no-break space before `: ; ! ?` or inside « », English quotes
- calques: a fixed list of 14 anglicism patterns
- title case: English Title Case words in French `<title>`
- untranslated: English function-word ratio
- chrF++ against a Sonnet-written, script-verified reference

Primary metric: mean topic reward (`fr_rl.checks.reward`) on test. Secondary:
each component count above and mean chrF. Paired bootstrap 95% CI over topics.

## Protocol

1. Base model `Qwen/Qwen3-4B-Instruct-2507`, same system prompt as training
   (`fr_rl/prompt.py`), greedy, on dev and test.
2. GRPO with LoRA on the 999 train segments. Dev eval every N steps.
3. Checkpoint = best dev mean reward among the evaluated ones (step 0, the base,
   is a candidate; if it wins, the result is "no improvement").
4. Test eval of that one checkpoint, greedy. Reported whatever it shows.
5. Independent quality check: blind A/B judgement of base vs trained test
   outputs by Sonnet subagents, order randomised per topic, then spot-checked.

## Known weaknesses, stated up front

- chrF is against a reference written by another model (Claude Sonnet). Moving
  toward it is partly moving toward that model's style.
- The pattern checks are heuristics I wrote. A model trained on them can learn to
  satisfy the checker rather than the rule; the blind A/B is there to catch that.
- 40 test topics is small. Differences of a few chrF points may be noise; the CI says.
- No-break spaces can also be fixed by a regex post-processor. That would not
  fix calques, title case or structure.

## Budget

Account balance at start: $4.4065 (results/fr_rl/pod_ledger.jsonl). Community
RTX 3090 at $0.22/hr. Hard stop for this experiment: $2.50 of GPU spend.
