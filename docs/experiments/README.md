# Experiments

Each run has its own document. The first lines of each one say whether the held-out metric moved. A flat or worse result stays in that document.

| Run | Held-out result | Document | Result commit |
| --- | --- | --- | --- |
| English to Romanian, pass/fail reward, GRPO | Did not move. Valid-and-skeleton stayed 8/10. Mean reward stayed 0.635. | [01-en-ro.md](01-en-ro.md) | `1517f47` |
| English to French, train on format breaks | Scan stopped at 45 breaks on a time cap. Format-broken went from 10/10 to 9/10. Nine strings are unchanged. | [02-en-fr-format.md](02-en-fr-format.md) | `69c81a3` |
| English to French, violation-count reward | Did not move. Violations stayed 10/10. Schema occurrences stayed 5. Punctuation occurrences stayed 34. | [03-en-fr-violations.md](03-en-fr-violations.md) | `f9a3524` |
| English to French, continued NBSP LoRA on new topics | Punctuation moved down and did not reach zero: 39, then 33, then 28. Schema went 2, then 0, then 2. Topics still violated went 10, then 9, then 9. | [05-en-fr-nbsp-sft2.md](05-en-fr-nbsp-sft2.md) | `0a9367e` |
| English to French, new typography LoRA | Not a before/after. This adapter's greedy historical mean reward is 0.465, gate passed 7/10. Sealed synthetic mean reward is 1.0 on 40/40. GRPO did not train. | [06-en-fr-style-sft.md](06-en-fr-style-sft.md) | `afa2726` |
| English to French, GRPO on segments, chrF + pattern-check reward (Qwen3-4B-Instruct-2507) | Moved. Test mean reward 0.4135 → 0.551 (Δ +0.1375, CI [0.012, 0.267]). chrF 80.01 → 83.04. Typography 53 → 3, calques 8 → 0. Structure 27 → 26/40 and title case 7 → 7 did not improve. Not significant over a regex NBSP fixer on reward (0.5075 vs 0.557 with the fixer). Blind A/B 22–17–1, p = 0.52. Calque penalty over-generalised to 'édit-'. | [07-en-fr-grpo-chrf-checks.md](07-en-fr-grpo-chrf-checks.md) | `df4d612` |

The continued NBSP run is document 05. It started from the experiment-04 French NBSP LoRA. That earlier writeup is not in this branch.

The older combined pages (`docs/report.md`, `docs/report_ro.md`, `docs/report_punct.md`, `docs/run_log.md`, `docs/run_log_ro.md`) now point here. Where one of those pages disagrees with a result file, the experiment document follows the file.
