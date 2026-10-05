"""Freeze the topic-level split and extract training segments.

Unit of evaluation: a whole DITA topic file (what we finally care about).
Unit of training: a block-level segment (<p>, <li>, <title>, <cmd>, ...)
taken from topics that are NOT in dev or test.

Leakage control: a training segment is dropped when its normalised text
also occurs anywhere in a dev or test topic (the user guide reuses a lot of
sentences across topics).

Writes split/fr_rl/{test,dev}_topics.jsonl and train_segments.jsonl.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

from lxml import etree

from fr_rl.checks import parse, prose, _norm, _tag, PROTECTED

ROOT = Path(__file__).resolve().parents[1]
DITA = ROOT / "data" / "userguide" / "DITA"
OUT = ROOT / "split" / "fr_rl"
SEED = 20261005
N_TEST, N_DEV, N_TRAIN = 40, 16, 1000
MAX_PER_FILE = 3

SEG_TAGS = {"title", "shortdesc", "p", "li", "cmd", "info", "stepresult",
            "dt", "dd", "entry", "note", "linktext", "context", "result",
            "prereq", "postreq", "glossdef", "cause", "condition", "glossterm"}
NO_INNER = {"p", "ul", "ol", "dl", "sl", "note", "fig", "codeblock", "table",
            "simpletable", "li", "section", "steps", "substeps", "lines",
            "pre", "object", "msgblock", "screen", "lq", "itemgroup"}


def has_reuse(e) -> bool:
    return any("conref" in x.attrib or "conkeyref" in x.attrib
               or (x.tag in ("ph", "keyword", "xref", "term") and "keyref" in x.attrib and not (x.text or len(x)))
               for x in e.iter() if _tag(x))


def segments(root):
    for e in root.iter():
        t = _tag(e)
        if t not in SEG_TAGS:
            continue
        if any(_tag(a) in ("prolog", "indexterm", "related-links") and t != "linktext" for a in e.iterancestors()):
            continue
        if any(_tag(d) in NO_INNER for d in e.iterdescendants()):
            continue
        if has_reuse(e):
            continue
        text = _norm(prose(e, mask_ui=False))
        if not (25 <= len(text) <= 500) or len(text.split()) < 4:
            continue
        xml = etree.tostring(e, encoding="unicode", with_tail=False)
        xml = re.sub(r'\s+xmlns:ditaarch="[^"]*"', "", xml)
        yield t, xml, text


def main():
    rev = subprocess.check_output(["git", "-C", str(DITA.parent), "rev-parse", "HEAD"], text=True).strip()
    files = sorted(DITA.rglob("*.dita"))
    topics, segs_by_file = [], {}
    for p in files:
        raw = p.read_text(encoding="utf-8")
        root = parse(raw)
        if root is None or "LIGHTWEIGHT" in raw[:600]:
            continue
        rel = str(p.relative_to(DITA))
        segs_by_file[rel] = list(segments(root))
        text = _norm(prose(root, mask_ui=False))
        eligible = (700 <= len(raw.encode()) <= 3500 and len(text) >= 250
                    and not has_reuse(root) and len(segs_by_file[rel]) >= 2
                    and "<!DOCTYPE" in raw[:400])
        if eligible:
            topics.append({"id": rel, "src": raw, "n_bytes": len(raw.encode()),
                           "root": root.tag, "prose_chars": len(text)})

    rng = random.Random(SEED)
    rng.shuffle(topics)
    test, dev = topics[:N_TEST], topics[N_TEST:N_TEST + N_DEV]
    held = {t["id"] for t in test + dev}

    held_sent = set()
    for t in test + dev:
        root = parse(t["src"])
        for e in root.iter():
            if _tag(e) in SEG_TAGS:
                held_sent.add(_norm(prose(e, mask_ui=False)).lower())

    pool, seen = [], set()
    dropped_leak = 0
    for rel, segs in segs_by_file.items():
        if rel in held:
            continue
        for tag, xml, text in segs:
            key = text.lower()
            if key in held_sent:
                dropped_leak += 1
                continue
            if key in seen:
                continue
            seen.add(key)
            pool.append({"file": rel, "tag": tag, "src": xml, "chars": len(text)})
    rng.shuffle(pool)
    per_file, train = Counter(), []
    for s in pool:
        if per_file[s["file"]] >= MAX_PER_FILE:
            continue
        per_file[s["file"]] += 1
        s["id"] = "seg-" + hashlib.sha1(s["src"].encode()).hexdigest()[:10]
        train.append(s)
        if len(train) >= N_TRAIN:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("test_topics", test), ("dev_topics", dev), ("train_segments", train)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "source_repo": "https://github.com/oxygenxml/userguide", "source_commit": rev,
        "seed": SEED, "n_dita_files": len(files), "n_eligible_topics": len(topics),
        "n_test_topics": len(test), "n_dev_topics": len(dev),
        "n_train_segments": len(train), "segment_pool": len(pool),
        "train_segments_dropped_as_heldout_duplicates": dropped_leak,
        "train_tag_counts": Counter(s["tag"] for s in train).most_common(),
        "topic_rule": "standard DITA, 700..3500 bytes, >=250 prose chars, no conref/conkeyref, >=2 segments",
        "segment_rule": "block element without nested blocks or reuse, 25..500 prose chars, >=4 words, max 3 per file",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
