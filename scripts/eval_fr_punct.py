#!/usr/bin/env python3
"""CPU evaluation for the French typography prep.

Writes two measurements:
- a postprocessor baseline on the oracle fixtures, separate from any model
- a rescore of historical outputs under results/ with the legacy counter and
  the new oracle

Published numbers are copied. They are not replaced. This script does not
train and does not call RunPod.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dita_reward import EVALUATOR_VERSION, count_violations, typography_oracle  # noqa: E402
from fr_typography_fixtures import build_cases  # noqa: E402
from fr_typography_postprocess import postprocess_french_typography  # noqa: E402

PUBLISHED_KEYS = (
    "reward", "violations", "schema", "punct", "missing_nbsp", "prose_punct",
    "format_broken", "well_formed", "dtd_valid", "skeleton_match", "path", "tag",
    "parts", "parts_schema", "english_quotes_not_in_reward", "has_nbsp",
)


def _dita_root() -> Path | None:
    candidates = []
    if os.environ.get("USERGUIDE_DITA"):
        candidates.append(Path(os.environ["USERGUIDE_DITA"]))
    candidates.append(ROOT / "data" / "userguide" / "DITA")
    candidates.append(Path("/tmp/ug/DITA"))
    for path in candidates:
        if path.is_dir():
            return path
    return None


def postprocess_baseline() -> dict:
    before = []
    after = []
    improved = 0
    regressed = 0
    unchanged = 0
    for case in build_cases():
        if case["category"] not in ("negatives", "positives", "guillemets", "protected_literals"):
            continue
        start = typography_oracle(case["source"], case["hyp"])
        if not start["gate_passed"]:
            continue
        edited = postprocess_french_typography(case["hyp"])
        end = typography_oracle(case["source"], edited)
        before.append(start["reward"])
        after.append(end["reward"])
        if end["reward"] > start["reward"]:
            improved += 1
        elif end["reward"] < start["reward"]:
            regressed += 1
        else:
            unchanged += 1
    return {
        "separate_from_any_model": True,
        "method": "deterministic typography postprocessor",
        "evaluator_version": EVALUATOR_VERSION,
        "n": len(before),
        "mean_reward_before": round(sum(before) / len(before), 4),
        "mean_reward_after": round(sum(after) / len(after), 4),
        "n_improved": improved,
        "n_regressed": regressed,
        "n_unchanged": unchanged,
    }


def _published_metrics() -> dict:
    found = {}
    for rel in (
        "results/fr_punct/metrics.json",
        "results/fr/metrics.json",
        "results/fr_nbsp_sft2/metrics.json",
        "results/metrics.json",
    ):
        path = ROOT / rel
        if path.is_file():
            found[rel] = json.loads(path.read_text())
    return found


def _copy_published(row: dict) -> dict:
    copied = {}
    for key in PUBLISHED_KEYS:
        if key in row:
            copied[key] = row[key]
    return copied


def rescore_historical(dita_root: Path | None) -> dict:
    files = sorted((ROOT / "results").rglob("*test.json"))
    reports = []
    for path in files:
        payload = json.loads(path.read_text())
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        if not isinstance(rows, list) or not rows or "hyp_xml" not in rows[0]:
            continue
        published_summary = payload.get("summary") if isinstance(payload, dict) else None
        rescored_rows = []
        old_punct = 0
        old_schema = 0
        old_violations = 0
        new_rewards = []
        new_gate = 0
        missing_source = 0
        for row in rows:
            published = _copy_published(row)
            rel = row.get("path")
            source_text = None
            if dita_root is not None and rel:
                source_path = dita_root / rel
                if source_path.is_file():
                    source_text = source_path.read_text(encoding="utf-8", errors="replace")
            item = {"published": published}
            if source_text is None:
                missing_source += 1
                item["old_counter"] = None
                item["new_oracle"] = None
                item["error"] = "source DITA not on disk"
            else:
                hyp = row.get("raw_text") or row.get("hyp_xml") or ""
                old = count_violations(source_text, hyp)
                new = typography_oracle(source_text, hyp)
                old_punct += int(old["punct"])
                old_schema += int(old["schema"])
                old_violations += int(old["violations"])
                new_rewards.append(new["reward"])
                new_gate += int(bool(new["gate_passed"]))
                item["old_counter"] = {
                    "evaluator": "count_violations",
                    "reward": old["reward"],
                    "violations": old["violations"],
                    "schema": old["schema"],
                    "punct": old["punct"],
                }
                item["new_oracle"] = {
                    "evaluator_version": EVALUATOR_VERSION,
                    "reward": new["reward"],
                    "gate_passed": new["gate_passed"],
                    "gate_failures": new["gate_failures"],
                    "nbsp_count": new["nbsp_count"],
                    "reward_uses_nbsp_count": False,
                }
            rescored_rows.append(item)
        reports.append({
            "file": str(path.relative_to(ROOT)),
            "published_summary_verbatim": published_summary,
            "n": len(rows),
            "n_missing_source": missing_source,
            "old_counter_rescore": None if missing_source == len(rows) else {
                "punct": old_punct,
                "schema": old_schema,
                "violations": old_violations,
                "mean_reward": round(sum(item["old_counter"]["reward"] for item in rescored_rows if item["old_counter"]) / max(1, len(rows) - missing_source), 4),
            },
            "new_oracle_rescore": None if not new_rewards else {
                "evaluator_version": EVALUATOR_VERSION,
                "mean_reward": round(sum(new_rewards) / len(new_rewards), 4),
                "gate_passed": new_gate,
                "n": len(new_rewards),
            },
            "rows": rescored_rows,
        })
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "note": (
            "published_summary_verbatim and published row fields are copied from the "
            "result files. old_counter_rescore and new_oracle_rescore are new "
            "measurements. A difference is not a corrected baseline."
        ),
        "language_note": (
            "results/baseline_test.json and results/after_test.json are the Romanian run. "
            "results/fr/ is the format-break run. French-oracle numbers on those files "
            "measure those XML strings. They are not Romanian metrics and they are not "
            "format-break metrics."
        ),
        "published_metrics_verbatim": _published_metrics(),
        "source_tree": None if dita_root is None else str(dita_root),
        "files": reports,
    }


def main() -> None:
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = postprocess_baseline()
    (out_dir / "fr_typography_postprocess_baseline.json").write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rescore = rescore_historical(_dita_root())
    (out_dir / "fr_typography_oracle_rescore.json").write_text(
        json.dumps(rescore, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "postprocess_mean_before": baseline["mean_reward_before"],
        "postprocess_mean_after": baseline["mean_reward_after"],
        "postprocess_n": baseline["n"],
        "rescore_files": [item["file"] for item in rescore["files"]],
    }, indent=2))


if __name__ == "__main__":
    main()
