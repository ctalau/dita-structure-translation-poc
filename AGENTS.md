# Working in this repo

## Experiment documents

Every experiment gets its own document, written after the run. Do not start the document before the run has finished, and do not add the run to another experiment's document.

The first lines state whether the held-out metric moved. A flat result is a result. A worse result is a result. Write the measured before and after numbers there.

Then write the method, the data split, the numbers, the cost, and the surprises. Point at the result files and at the commit that added them.

Rules for that document:

- Do not fold a failed run into a later run's success story. A later run does not erase an earlier one.
- Do not hide a tie, a crash, a cap, or a metric that did not move.
- Do not replace a measured number with a softer summary. Write the number that is in the file.
- If a chat recollection and a result file disagree, the file wins. Say that they disagree and quote the file.
- Do not invent a number that is not in the committed results or the git history. If a figure exists only in an earlier writeup, say so.
- Bandwidth, kernel, or speed notes go in the document only when a result file records them, and only in the experiment that recorded them.

Put the document under `docs/experiments/` and add a line to `docs/experiments/README.md`. The root `README.md` should keep linking to that index.

Do not start a new training run in order to produce a document.
