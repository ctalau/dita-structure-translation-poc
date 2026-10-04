# Run log

Times below are UTC.

## 19:15

Rented community RTX 3090 `ix70qe0fixor7g` at $0.22/hr. Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Host CUDA was 13.0. Direct SSH worked (`root@64.119.209.250 -p 13294`). The RunPod proxy shell rejects a command without a PTY, so the first uploads went through an interactive shell.

## Surprises

1. **xmllint cannot load the stock DITA 1.3 SVG domain.** `svgDomain.mod` pulls in `svg11-ditadriver.dtd` and libxml2 reports an entity-reference loop. No topic in this manual uses `svg-container`. The setup script drops that one include. After that, a sample of real topics validates. 963 of the size-filtered topics passed and became the pool.

2. **Transformers 4.57.6 does not know `qwen3_5`.** The model card architecture is `Qwen3_5ForConditionalGeneration` (text + a vision tower). Pip 5.18.0 knows it, but that release refuses to use torch 2.4.1 (`PyTorch >= 2.5 is required`). Installed torch 2.6.0+cu124.

3. **The image's torchvision and torchaudio then failed ABI checks** (`torchvision::nms` missing, `libtorchaudio.so` undefined symbol). Uninstalled torchaudio and installed torchvision 0.21.0+cu124. After that, NF4 load was 12 s and about 3.3 GB.

4. **Qwen3.5 linear-attention kernels were not installed.** Logs repeat that `causal_conv1d` and `flash-linear-attention` fell back to a slow PyTorch reference. Generation stayed correct and landed around 5–8 GB, but a topic costs tens of seconds. That is the whole reason the update set is 24 topics, not the full train split.

5. **The base model already keeps structure on most short topics.** Greedy baseline: 8/10 valid and skeleton-matched. Because of that, 20 of 24 sample pairs had identical rewards and produced no gradient. Only 4 updates ran (`dcpp_console_logging`, `whr-converting-templates`, the split-paragraphs topic, `dcpp_how_to_display_subtopics_in_toc`). Test-set pass counts did not move.

6. **The failure mode is not "random tags".** One topic is cut off at `max_new_tokens=640` (both samples then score -1, so still no gradient). The other failure is the model translating an `id` and a `conkeyref` into Romanian. The DTD allows it. The skeleton check is what catches it.

7. **Grok 4.7 was not reachable.** No xAI key. OpenRouter returned `API key expired`. Quality numbers in `results/judge_fallback.json` are a word-list check, not a model judge.

## 19:29–20:02

`results/pod_run.log` is the pod stdout. `rl_done` at 19:54:52 UTC: 4 updates, 20 flat groups, 1292 s. After-eval and val finished about 20:02. Artifacts were copied off, then the pod was deleted.
