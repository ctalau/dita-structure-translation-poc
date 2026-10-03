#!/usr/bin/env python3
"""Clone Oxygen userguide (if needed) and write a frozen split.

Eligible topics: standard DITA (not lightweight), source file DTD-valid,
size 700..2600 bytes so a 4B model can emit the whole topic under a short
token cap. Seed 1337. Test = 10, val = 10, 24 of the remaining train topics
are the RL update set. Every other topic is recorded as excluded or train-unused.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dita_reward import dtd_validate  # noqa: E402

GUIDE = ROOT / "data" / "userguide"
DITA = GUIDE / "DITA"
MIN_B, MAX_B = 700, 2600
SEED = 1337
N_TEST, N_VAL, N_UPDATE = 10, 10, 24


def ensure_guide():
    if (DITA).is_dir() and any(DITA.rglob("*.dita")):
        return
    GUIDE.parent.mkdir(parents=True, exist_ok=True)
    if not GUIDE.exists():
        subprocess.check_call(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "https://github.com/oxygenxml/userguide.git", str(GUIDE)]
        )
    subprocess.check_call(["git", "-C", str(GUIDE), "sparse-checkout", "set", "DITA"])
    # depth-1 sparse may already have files; if not, checkout
    if not any(DITA.rglob("*.dita")):
        subprocess.check_call(["git", "-C", str(GUIDE), "checkout"])


def public_id(text: str) -> str:
    # join doctype header
    head = "\n".join(text.splitlines()[:12])
    if "LIGHTWEIGHT" in head:
        return "LIGHTWEIGHT"
    if "<!DOCTYPE" not in head and "<!DOCTYPE" not in text[:1500]:
        return "NO_DOCTYPE"
    return "STANDARD"


def main():
    ensure_guide()
    rev = subprocess.check_output(["git", "-C", str(GUIDE), "rev-parse", "HEAD"], text=True).strip()
    rows = []
    eligible = []
    for p in sorted(DITA.rglob("*.dita")):
        rel = str(p.relative_to(DITA))
        raw = p.read_text(encoding="utf-8", errors="replace")
        size = p.stat().st_size
        kind = public_id(raw)
        rec = {"path": rel, "bytes": size, "public": kind, "split": "excluded", "reason": ""}
        if kind != "STANDARD":
            rec["reason"] = kind
            rows.append(rec)
            continue
        if not (MIN_B <= size <= MAX_B):
            rec["reason"] = f"size_outside_{MIN_B}_{MAX_B}"
            rows.append(rec)
            continue
        ok, nerr, err = dtd_validate(raw)
        rec["source_dtd_valid"] = ok
        rec["source_dtd_errors"] = nerr
        if not ok:
            rec["reason"] = "source_dtd_invalid"
            rec["error_tail"] = err[-300:]
            rows.append(rec)
            continue
        rec["reason"] = "eligible"
        rows.append(rec)
        eligible.append(rel)

    rng = random.Random(SEED)
    order = eligible[:]
    rng.shuffle(order)
    if len(order) < N_TEST + N_VAL + 1:
        raise SystemExit(f"only {len(order)} eligible topics")
    test = order[:N_TEST]
    val = order[N_TEST:N_TEST + N_VAL]
    update = order[N_TEST + N_VAL:N_TEST + N_VAL + N_UPDATE]
    train_rest = order[N_TEST + N_VAL + N_UPDATE:]
    label = {}
    for p in test:
        label[p] = "test"
    for p in val:
        label[p] = "val"
    for p in update:
        label[p] = "train_update"
    for p in train_rest:
        label[p] = "train_unused"

    for rec in rows:
        if rec["path"] in label:
            rec["split"] = label[rec["path"]]
            rec["reason"] = "eligible"

    out = {
        "source_repo": "https://github.com/oxygenxml/userguide",
        "source_commit": rev,
        "language_pair": "en-ro",
        "language_note": "Oxygen user guide source is English. Target is Romanian.",
        "seed": SEED,
        "size_bytes": [MIN_B, MAX_B],
        "n_dita_files": len(rows),
        "n_eligible": len(eligible),
        "n_test": len(test),
        "n_val": len(val),
        "n_train_update": len(update),
        "n_train_unused": len(train_rest),
        "test": test,
        "val": val,
        "train_update": update,
        "topics": rows,
    }
    dest = ROOT / "split" / "manifest.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["source_commit", "n_dita_files", "n_eligible", "n_test", "n_val", "n_train_update", "n_train_unused"]}, indent=2))
    print("test:")
    for p in test:
        print(" ", p)


if __name__ == "__main__":
    main()
