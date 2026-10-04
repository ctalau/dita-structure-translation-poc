# DITA structure translation PoC

Five finished runs. Each one has its own writeup. The first lines of each writeup say whether the held-out metric moved.

| Run | Held-out result |
| --- | --- |
| English to Romanian, pass/fail reward | Did not move. Valid-and-skeleton stayed 8/10. Mean reward stayed 0.635. |
| English to French, train on format breaks | Scan stopped at 45 breaks on a time cap. Format-broken went from 10/10 to 9/10. |
| English to French, violation counts | Did not move. The test set stayed 10/10 violated. Schema occurrences stayed 5. Punctuation occurrences stayed 34. |
| English to French, continued NBSP on new topics | Punctuation moved down and did not reach zero: 39, then 33, then 28. Schema went 2, then 0, then 2. Topics still violated went 10, then 9, then 9. |
| English to French, new typography LoRA | Not a before/after. This adapter's greedy historical mean reward is 0.465, gate passed 7/10. Sealed synthetic mean reward is 1.0 on 40/40. GRPO did not train. |

Index: [docs/experiments/README.md](docs/experiments/README.md).

The French typography CPU preparation is a plan, not a finished run: [docs/plans/fr-punct-sft-grpo.md](docs/plans/fr-punct-sft-grpo.md). It does not replace an experiment note. After a real training run, AGENTS.md still requires a separate document under `docs/experiments/`.

Source for all three: [oxygenxml/userguide](https://github.com/oxygenxml/userguide) commit `db722d7`. Base weights are not in this repo. Adapters: `results/adapter/` (Romanian), `results/fr/adapter/`, `results/fr_punct/adapter/`.

```bash
bash scripts/setup_dtd.sh
python3 scripts/prepare_split.py
python3 scripts/run_gpu_poc.py

python3 scripts/prepare_fr_candidates.py
# GPU, from the repo root, after requirements.txt on torch>=2.5:
SCAN_WALL_S=10800 TRAIN_WALL_S=3000 python3 scripts/run_fr_poc.py
# If training dies after the scan has written split/manifest_fr.json:
python3 scripts/resume_fr_train.py

SCAN_WALL_S=10800 TRAIN_WALL_S=3000 python3 scripts/run_fr_punct.py
```
