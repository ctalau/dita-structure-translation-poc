"""Deterministic French typography post-processor (a baseline, not a model).

Inserts a no-break space before : ; ! ? used as punctuation and inside « »,
in text outside protected elements. Leaves everything else alone.

  python3 -m fr_rl.nbsp_fix IN.jsonl OUT.jsonl
"""
from __future__ import annotations

import json
import re
import sys

from lxml import etree

from fr_rl.checks import PROTECTED, _tag, clean_output, parse, split_prolog, _expand

_HIGH = re.compile(r"[   ]?([:;!?])(?=\s|$|[)\]»])")
_OPEN = re.compile(r"«[   ]?")
_CLOSE = re.compile(r"[   ]?»")


def fix_text(t: str | None) -> str | None:
    if not t:
        return t
    t = _HIGH.sub(lambda m: " " + m.group(1) if not (m.start() > 0 and t[m.start() - 1].isdigit() and m.group(1) == ":") else m.group(0), t)
    t = _OPEN.sub("« ", t)
    t = _CLOSE.sub(" »", t)
    return t


def fix(xml: str) -> str:
    xml = clean_output(xml)
    head, _ = split_prolog(_expand(xml))
    root = parse(xml)
    if root is None:
        return xml
    for e in root.iter():
        if not _tag(e):
            continue
        inside = _tag(e) in PROTECTED or any(_tag(a) in PROTECTED for a in e.iterancestors())
        if not inside:
            e.text = fix_text(e.text)
        parent = e.getparent()
        parent_prot = parent is not None and (_tag(parent) in PROTECTED or any(_tag(a) in PROTECTED for a in parent.iterancestors()))
        if parent is not None and not parent_prot:
            e.tail = fix_text(e.tail)
    return head + etree.tostring(root, encoding="unicode")


def main(a):
    with open(a[2], "w", encoding="utf-8") as f:
        for line in open(a[1], encoding="utf-8"):
            d = json.loads(line)
            d["hyp"] = fix(d["hyp"])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv)
