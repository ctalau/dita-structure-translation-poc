"""Blind pairwise comparison of two systems on the same topics.

  python3 -m fr_rl.blind_ab make SYS1.jsonl SYS2.jsonl --split test --out DIR --shards 4
  python3 -m fr_rl.blind_ab unblind DIR

`make` writes DIR/pairs_K.jsonl ({pair, src, A, B}) for the judges and
DIR/key.json (which system is A) that judges never see. Order is random per
topic (seeded). Judges write DIR/verdicts_K.jsonl rows {pair, winner: A|B|tie,
reason}. `unblind` maps verdicts back to systems and writes DIR/result.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from fr_rl.checks import clean_output

ROOT = Path(__file__).resolve().parents[1]


def make(a):
    topics = {d["id"]: d for d in map(json.loads, open(ROOT / f"split/fr_rl/{a.split}_topics.jsonl"))}
    s1 = {d["id"]: d["hyp"] for d in map(json.loads, open(a.sys1))}
    s2 = {d["id"]: d["hyp"] for d in map(json.loads, open(a.sys2))}
    ids = sorted(set(s1) & set(s2) & set(topics))
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key, rows = {"sys1": a.sys1, "sys2": a.sys2, "pairs": {}}, []
    for n, i in enumerate(ids):
        pid = f"p{n:03d}"
        flip = rng.random() < 0.5
        A, B = (s2[i], s1[i]) if flip else (s1[i], s2[i])
        key["pairs"][pid] = {"id": i, "A": "sys2" if flip else "sys1"}
        rows.append({"pair": pid, "src": topics[i]["src"], "A": clean_output(A), "B": clean_output(B)})
    for k in range(a.shards):
        with open(out / f"pairs_{k}.jsonl", "w") as f:
            for r in rows[k::a.shards]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(len(rows), "pairs")


def unblind(a):
    out = Path(a.dir)
    key = json.loads((out / "key.json").read_text())
    tally = {"sys1": 0, "sys2": 0, "tie": 0}
    per = []
    for f in sorted(out.glob("verdicts_*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            v = json.loads(line)
            k = key["pairs"][v["pair"]]
            w = v["winner"].strip()
            sysw = "tie" if w == "tie" else (k["A"] if w == "A" else ("sys2" if k["A"] == "sys1" else "sys1"))
            tally[sysw] += 1
            per.append({"id": k["id"], "winner": sysw, "reason": v.get("reason", "")})
    res = {"sys1": key["sys1"], "sys2": key["sys2"], "n_judged": len(per), "n_pairs": len(key["pairs"]),
           "tally": tally, "per_topic": sorted(per, key=lambda r: r["id"])}
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("sys1", "sys2", "n_judged", "n_pairs", "tally")}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("sys1")
    m.add_argument("sys2")
    m.add_argument("--split", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--shards", type=int, default=4)
    m.add_argument("--seed", type=int, default=12345)
    u = sub.add_parser("unblind")
    u.add_argument("dir")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else unblind(a)


if __name__ == "__main__":
    main()
