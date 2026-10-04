#!/usr/bin/env python3
"""List source-valid standard DITA topics the French scan is allowed to translate.

Order is sorted repo-relative path. The GPU walks this list and stops at 100
format breaks. Files that cannot finish inside the token cap are still listed;
the GPU records them as skipped_over_cap instead of pretending they broke.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DITA = ROOT / "data" / "userguide" / "DITA"
MAX_BYTES = 4500
MIN_BYTES = 400


def check(rel: str):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    path = DITA / rel
    raw = path.read_text(encoding="utf-8", errors="replace")
    size = path.stat().st_size
    head = raw[:1200]
    if "LIGHTWEIGHT" in head:
        return {"path": rel, "bytes": size, "keep": False, "reason": "lightweight"}
    if "<!DOCTYPE" not in head and "<!DOCTYPE" not in raw[:2000]:
        return {"path": rel, "bytes": size, "keep": False, "reason": "no_doctype"}
    if size < MIN_BYTES:
        return {"path": rel, "bytes": size, "keep": False, "reason": "too_small"}
    if size > MAX_BYTES:
        return {"path": rel, "bytes": size, "keep": False, "reason": "too_big_for_scan_pool"}
    from dita_reward import dtd_validate
    ok, nerr, err = dtd_validate(raw)
    if not ok:
        return {"path": rel, "bytes": size, "keep": False, "reason": "source_dtd_invalid", "dtd_errors": nerr}
    return {"path": rel, "bytes": size, "keep": True, "reason": "candidate"}


def main():
    rels = sorted(str(p.relative_to(DITA)) for p in DITA.rglob("*.dita"))
    rev = subprocess.check_output(["git", "-C", str(ROOT / "data" / "userguide"), "rev-parse", "HEAD"], text=True).strip()
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(check, rel) for rel in rels]
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"checked {done}/{len(rels)}", flush=True)
    rows.sort(key=lambda r: r["path"])
    keep = [r for r in rows if r["keep"]]
    out = {
        "source_repo": "https://github.com/oxygenxml/userguide",
        "source_commit": rev,
        "order": "sorted relative path",
        "min_bytes": MIN_BYTES,
        "max_bytes": MAX_BYTES,
        "n_files": len(rows),
        "n_candidates": len(keep),
        "candidates": [r["path"] for r in keep],
        "skipped_histogram": {},
        "topics": rows,
    }
    hist = {}
    for r in rows:
        hist[r["reason"]] = hist.get(r["reason"], 0) + 1
    out["skipped_histogram"] = hist
    dest = ROOT / "split" / "fr_candidates.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"commit": rev, "n_files": len(rows), "n_candidates": len(keep), "hist": hist}, indent=2))


if __name__ == "__main__":
    main()
