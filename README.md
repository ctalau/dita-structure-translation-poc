# DITA structure translation PoC

Short experiment: can a 4-bit Qwen3.5-4B be nudged with a DITA DTD validator so English Oxygen user-guide topics stay valid XML when translated to Romanian?

- Source: [oxygenxml/userguide](https://github.com/oxygenxml/userguide) commit `db722d7af0c705cc3d6969f52bd1299a20ba1770` (English). Direction is English to Romanian.
- Frozen test set: 10 topics. See `split/manifest.json`.
- Training: NF4 QLoRA on a community RTX 3090, GRPO-style reward from `xmllint` plus an element-skeleton check.
- Report: [docs/report.md](docs/report.md). Run notes: [docs/run_log.md](docs/run_log.md).

## Reproduce

```bash
bash scripts/setup_dtd.sh          # OASIS DITA 1.3 DTDs, SVG driver include removed
python3 scripts/prepare_split.py   # clones the userguide and rewrites the split (seed 1337)
# GPU, from the repo root, after installing requirements.txt on torch>=2.5:
TRAIN_WALL_S=2400 python3 scripts/run_gpu_poc.py
python3 scripts/score_grok.py      # needs a live OpenRouter or XAI key; see the report
```

Base weights are not in this repo. The LoRA adapter is `results/adapter/` (~6 MB). Load it on `Qwen/Qwen3.5-4B`.
