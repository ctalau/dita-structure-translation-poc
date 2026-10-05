"""Score translation outputs against the frozen split. CPU only.

  python3 -m fr_rl.score OUTPUTS.jsonl [OUTPUTS2.jsonl ...] --split test
  python3 -m fr_rl.score A.jsonl B.jsonl --split test --paired   # A vs B

Writes OUTPUTS.scored.json (per-topic reports + summary) next to each input.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from fr_rl.checks import check, reward, chrf, clean_output, parse, prose

ROOT = Path(__file__).resolve().parents[1]


def load_split(split: str) -> dict:
    return {d["id"]: d for d in map(json.loads, open(ROOT / f"split/fr_rl/{split}_topics.jsonl", encoding="utf-8"))}


def score_file(path: str, split: str) -> dict:
    topics = load_split(split)
    rows = [d for d in map(json.loads, open(path, encoding="utf-8")) if d["id"] in topics]
    per = []
    for d in rows:
        t = topics[d["id"]]
        rep = check(t["src"], d["hyp"], t["ref"])
        if rep.chrf is None:  # unparseable: score the tag-stripped text
            ref_root = parse(t["ref"])
            rep.chrf = round(chrf(re.sub(r"<[^>]+>", " ", clean_output(d["hyp"])),
                                  prose(ref_root, mask_ui=False)), 2)
        row = rep.as_dict()
        row.update(id=d["id"], reward=round(reward(rep), 4),
                   hit_cap=d.get("finish_reason") == "length")
        per.append(row)
    n = len(per)
    s = {
        "n": n,
        "structure_ok": sum(r["structure_ok"] for r in per),
        "xml_ok": sum(r["xml_ok"] for r in per),
        "skeleton_ok": sum(r["skeleton_ok"] for r in per),
        "header_ok": sum(r["header_ok"] for r in per),
        "protected_changed": sum(r["protected_changed"] for r in per),
        "typo_total": sum(r["n_typo"] for r in per),
        "typo_nbsp_missing": sum(r["typo"].get("nbsp_missing", 0) for r in per),
        "typo_guillemet_space": sum(r["typo"].get("guillemet_space", 0) for r in per),
        "typo_en_quotes": sum(r["typo"].get("en_quotes", 0) for r in per),
        "topics_with_typo": sum(r["n_typo"] > 0 for r in per),
        "calque_total": sum(r["n_calque"] for r in per),
        "topics_with_calque": sum(r["n_calque"] > 0 for r in per),
        "titlecase_total": sum(r["titlecase"] for r in per),
        "untranslated": sum(r["untranslated"] for r in per),
        "hit_cap": sum(r["hit_cap"] for r in per),
        "mean_chrf": round(sum(r["chrf"] for r in per) / n, 2) if n else None,
        "mean_reward": round(sum(r["reward"] for r in per) / n, 4) if n else None,
        "clean_topics": sum(r["structure_ok"] and r["n_typo"] == 0 and r["n_calque"] == 0
                            and r["titlecase"] == 0 and not r["untranslated"] for r in per),
    }
    calq = {}
    for r in per:
        for k, v in r["calques"].items():
            calq[k] = calq.get(k, 0) + v
    s["calques_by_type"] = calq
    out = {"source": path, "split": split, "summary": s, "topics": per}
    Path(path).with_suffix(".scored.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def paired(a: dict, b: dict, key: str, n_boot=10000, seed=0) -> dict:
    """Paired bootstrap of mean(b - a) over topics present in both.

    n_increased counts topics where b's value is higher than a's. For reward
    and chrF higher is better; for error counts higher is worse.
    """
    A = {r["id"]: r[key] for r in a["topics"]}
    B = {r["id"]: r[key] for r in b["topics"]}
    ids = sorted(set(A) & set(B))
    d = [float(B[i]) - float(A[i]) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"key": key, "n": len(d), "mean_delta": round(sum(d) / len(d), 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)],
            "n_increased": sum(x > 0 for x in d), "n_decreased": sum(x < 0 for x in d), "n_equal": sum(x == 0 for x in d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--split", required=True)
    ap.add_argument("--paired", action="store_true")
    args = ap.parse_args()
    res = [score_file(f, args.split) for f in args.files]
    for r in res:
        print(r["source"], json.dumps(r["summary"], ensure_ascii=False))
    if args.paired and len(res) == 2:
        for k in ("chrf", "reward", "n_typo", "n_calque", "titlecase"):
            print(json.dumps(paired(res[0], res[1], k)))


if __name__ == "__main__":
    main()
