# DITA structure translation PoC

Four finished runs. Each one has its own writeup. The first lines of each writeup say whether the held-out metric moved.

| Run | Held-out result |
| --- | --- |
| English to Romanian, pass/fail reward | Did not move. Valid-and-skeleton stayed 8/10. Mean reward stayed 0.635. |
| English to French, train on format breaks | Scan stopped at 45 breaks on a time cap. Format-broken went from 10/10 to 9/10. |
| English to French, violation counts | Did not move. The test set stayed 10/10 violated. Schema occurrences stayed 5. Punctuation occurrences stayed 34. |
| English to French, supervised non-breaking spaces | Punctuation occurrences 34 to 23. Schema occurrences stayed 5. Topics with a violation 10 to 6. |

Index: [docs/experiments/README.md](docs/experiments/README.md).

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
