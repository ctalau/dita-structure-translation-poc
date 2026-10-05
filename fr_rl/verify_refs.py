"""Verify French reference translations against their English sources.

Usage: python3 -m fr_rl.verify_refs SOURCES.jsonl REFS.(jsonl|txt)

SOURCES rows: {"id", "src"}; REFS rows: {"id", "fr"}.
Prints one line per problem and a summary. Exit code 1 if any problem.
A reference passes when: XML parses, same skeleton/attributes/DOCTYPE,
protected text unchanged, zero typography and calque findings, no title
case findings, not untranslated, prose length ratio within 0.8..1.9.
"""
from __future__ import annotations

import json
import sys

from fr_rl.checks import check


def problems(src: str, fr: str) -> list[str]:
    r = check(src, fr)
    out = []
    if not r.xml_ok:
        return ["xml does not parse"]
    if not r.header_ok:
        out.append("XML declaration/DOCTYPE differs from source")
    if not r.skeleton_ok:
        out.append("elements/attributes differ from source")
    if r.protected_changed:
        out.append(f"{r.protected_changed} protected element(s) changed (codeph, filepath, xmlelement, keyword, ...)")
    if r.n_typo:
        out.append(f"typography {r.typo}")
    if r.n_calque:
        out.append(f"calques {r.calques}")
    if r.titlecase:
        out.append(f"{r.titlecase} Title-Case word(s) in <title>")
    if r.untranslated:
        out.append(f"looks untranslated (english_ratio={r.english_ratio})")
    if not 0.8 <= r.len_ratio <= 1.9:
        out.append(f"length ratio {r.len_ratio} outside 0.8..1.9")
    return out


def load_refs(path: str) -> dict[str, str]:
    """JSONL rows {"id","fr"}, or a .txt file of blocks headed '### <id>'."""
    refs = {}
    if path.endswith(".txt"):
        cur, buf = None, []
        for line in open(path, encoding="utf-8").read().splitlines() + ["### <end>"]:
            if line.startswith("### "):
                if cur is not None:
                    refs[cur] = "\n".join(buf).strip()
                cur, buf = line[4:].strip(), []
            else:
                buf.append(line)
        refs.pop("<end>", None)
        return refs
    for line in open(path, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            refs[d["id"]] = d["fr"]
    return refs


def main(argv):
    srcs = {}
    for line in open(argv[1], encoding="utf-8"):
        d = json.loads(line)
        srcs[d["id"]] = d["src"]
    refs = load_refs(argv[2])
    missing = [i for i in srcs if i not in refs]
    for i in missing:
        print(f"{i}: missing")
    bad = len(missing)
    for i, fr in refs.items():
        if i not in srcs:
            print(f"{i}: unknown id")
            bad += 1
            continue
        ps = problems(srcs[i], fr)
        if ps:
            bad += 1
            print(f"{i}: " + "; ".join(ps))
    print(f"checked {len(refs)} refs, {bad} with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
