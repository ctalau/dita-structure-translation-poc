# DITA structure translation PoC

Two short runs. Both ask a 4-bit Qwen3.5-4B to translate Oxygen user-guide topics without breaking the DITA skeleton.

## English to French (latest)

- Source: [oxygenxml/userguide](https://github.com/oxygenxml/userguide) commit `db722d7`.
- Scan until format breaks, then train only on those. The 3 hour cap stopped at 45 breaks (target was 100). Test is 10 of those, seed 1337.
- Report: [docs/report.md](docs/report.md). Notes: [docs/run_log.md](docs/run_log.md).

```bash
bash scripts/setup_dtd.sh
python3 scripts/prepare_fr_candidates.py
# GPU, from the repo root, after requirements.txt on torch>=2.5:
SCAN_WALL_S=10800 TRAIN_WALL_S=3000 python3 scripts/run_fr_poc.py
# If training dies after the scan has written split/manifest_fr.json:
python3 scripts/resume_fr_train.py
```

## English to Romanian (previous)

See [docs/report_ro.md](docs/report_ro.md) and [docs/run_log_ro.md](docs/run_log_ro.md). `python3 scripts/run_gpu_poc.py` is that run.

Base weights are not in this repo. The French LoRA is `results/fr/adapter/` (~6 MB). The Romanian LoRA is `results/adapter/`.
