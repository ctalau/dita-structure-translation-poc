# Experiments

Each run has its own document. The first lines of each one say whether the held-out metric moved. A flat or worse result stays in that document.

| Run | Held-out result | Document | Result commit |
| --- | --- | --- | --- |
| English to Romanian, pass/fail reward, GRPO | Did not move. Valid-and-skeleton stayed 8/10. Mean reward stayed 0.635. | [01-en-ro.md](01-en-ro.md) | `1517f47` |
| English to French, train on format breaks | Scan stopped at 45 breaks on a time cap. Format-broken went from 10/10 to 9/10. Nine strings are unchanged. | [02-en-fr-format.md](02-en-fr-format.md) | `69c81a3` |
| English to French, violation-count reward | Did not move. Violations stayed 10/10. Schema occurrences stayed 5. Punctuation occurrences stayed 34. | [03-en-fr-violations.md](03-en-fr-violations.md) | `f9a3524` |

The older combined pages (`docs/report.md`, `docs/report_ro.md`, `docs/report_punct.md`, `docs/run_log.md`, `docs/run_log_ro.md`) now point here. Where one of those pages disagrees with a result file, the experiment document follows the file.
