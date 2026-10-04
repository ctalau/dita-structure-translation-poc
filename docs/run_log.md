# Run log (English to French)

Times below are UTC unless marked Bucharest. The Romanian run is in [run_log_ro.md](run_log_ro.md).

## 20:52 UTC (23:52 Bucharest)

Rented community RTX 3090 `7qrdee52y372iy` at $0.22/hr. Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`. Host CUDA 13.0. Direct SSH: `root@174.94.157.109` port `48139`.

## Surprises

1. **The user-guide checkout on the build machine was a symlink to `/tmp/userguide`.** The first upload archived the symlink, not the files. The model loaded, then the first topic path was missing. The real `.dita` files (gzipped, images left behind) were uploaded and the same process was not reused. A new process started at 21:32 UTC. That is the scan.

2. **Fast kernels were abandoned.** `pip install causal-conv1d flash-linear-attention` started an isolated build that depends on Torch. It was still installing that build env after several minutes, so it was killed (`KERNELS_FAIL` in the bootstrap log). Transformers then logged the slow fallback for `causal_conv1d` and `chunk_gated_delta_rule` on every generate. A short glossary topic was ~8 s. A 600-token topic was ~25–35 s.

3. **The base model already keeps the skeleton on most short topics.** 569 greedy translations, 524 kept the format, 45 broke it (7.9%). The 3 hour scan wall (`SCAN_WALL_S=10800`) fired at 00:33 UTC with 45 breaks, not 100. Phase 1 (cap 640) found 41 breaks in 536 files. Phase 2 (cap 896) added 4 more before the wall. 398 topics that would have fit the higher cap were not reached. 763 candidates were bigger than the 896 cap and were never sent to the model.

4. **Breaks are mostly bad tags and edited attributes, not reordered siblings.** Of 45: child mismatch 17, not well-formed 15, attribute changed 11, truncated 1, sibling reorder 1 (`topics/xml-editor-specific-actions.dita`). The order check is on. The model rarely permutes siblings. It drops a wrapper, translates an `id`, or emits `<indexterm>Titres<Mise en page</indexterm>`.

5. **`xmllint` stderr killed the first training pass.** At 00:38 UTC, scoring `topics/dcpp_how_to_remove_entries_from_the_toc.dita` raised `UnicodeDecodeError` on byte `0xa8`. Five updates had run and were not saved. `dita_reward.py` now decodes stderr with `errors=replace`. Training was run again from a zero LoRA on the same manifest (`scripts/resume_fr_train.py`), not a second scan. That pass finished at 01:01 UTC: 10 updates, 15 flat groups, 1089 s. After-eval: 9/10 still broken. Only `dcpp_how_to_display_subtopics_in_toc.dita` changed, and it became valid.

6. **Sampling often ties, so the dense reward still skips.** On the rerun, both samples scored 1.0, or both -1.0, or the same partial score (`0.9643` twice, `0.9361` twice). Greedy is the decode that fails on the test set. The two stochastic samples frequently agree, so there is no advantage.

7. **French punctuation was not learned.** The prompt asks for `« »` and a NBSP before `? ! : ;`. The reward does not. `cf-change-admin-password` does use `« fusion »`, then writes `suivez ces étapes :` with a normal space. `preferences-database-filters` is the one clear NBSP (`tableaux suivants\u00a0:`).

## 01:02 UTC (04:02 Bucharest)

Artifacts were copied off (`results/fr/`, including the 6 MB LoRA). `delete-pod` on `7qrdee52y372iy` returned 204.
