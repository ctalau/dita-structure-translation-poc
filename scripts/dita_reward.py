#!/usr/bin/env python3
"""Order-sensitive DITA format reward.

A translation breaks formatting when any of these is true:
- not well-formed XML
- generation hit the token cap before the document matched the source
- DTD-invalid (xmllint --valid, OASIS DITA 1.3 technical content)
- tag name changed
- attribute name, value, or attribute order changed (attributes compared in
  document order, not sorted)
- child elements under the same parent were reordered, added, or removed

Sibling order is significant. The skeleton is the preorder of
(tag, attribute pairs in document order). Two <li> swapped under the same
<ul> is a sibling_reorder break even if the DTD still accepts the file.
Equality is positional. A sorted multiset of children is not enough.

&nbsp; / &#160; / &#xA0; are expanded to U+00A0 before the ElementTree walk.
The DITA topic module declares nbsp, so xmllint accepts the entity, but
ElementTree does not load the DTD and would otherwise call a correct file
not well-formed.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "schemas" / "dita13-doctypes" / "catalog.xml"

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:xml)?\s*|```", re.IGNORECASE)
NBSP_RE = re.compile(r"&nbsp;|&#0*160;|&#x0*a0;", re.IGNORECASE)

PRIMARY_ORDER = (
    "truncated",
    "not_well_formed",
    "tag_mismatch",
    "child_mismatch",
    "sibling_reorder",
    "attr_changed",
    "attr_order",
    "dtd_invalid",
)


def extract_xml(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    text = FENCE_RE.sub("", text)
    starts = []
    for marker in (
        "<?xml",
        "<!DOCTYPE",
        "<topic",
        "<task",
        "<concept",
        "<reference",
        "<troubleshooting",
        "<glossentry",
        "<glossgroup",
    ):
        i = text.find(marker)
        if i >= 0:
            starts.append(i)
    if not starts:
        return text.strip()
    return text[min(starts):].strip()


def expand_nbsp(xml_text: str) -> str:
    return NBSP_RE.sub("\u00A0", xml_text)


def struct_sig(el):
    """Structure only: tag, attributes in document order, children in order. No text."""
    return (
        el.tag,
        tuple(el.attrib.items()),
        tuple(struct_sig(c) for c in list(el)),
    )


def _sigs(root):
    seq = []

    def walk(el):
        seq.append((el.tag, tuple(el.attrib.items())))
        for child in list(el):
            walk(child)

    walk(root)
    return seq


def _lcs_ratio(a, b) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return prev[m] / max(n, m)


def _attr_recall(src_sigs, hyp_sigs) -> float:
    """Fraction of source (tag, key, value) triples present in the hypothesis multiset."""
    from collections import Counter

    def triples(sigs):
        c = Counter()
        for tag, attrs in sigs:
            for k, v in attrs:
                c[(tag, k, v)] += 1
        return c

    src_c = triples(src_sigs)
    hyp_c = triples(hyp_sigs)
    total = sum(src_c.values())
    if total == 0:
        return 1.0
    hit = 0
    for k, n in src_c.items():
        hit += min(n, hyp_c.get(k, 0))
    return hit / total


def _compare(src, hyp, reasons: list):
    if struct_sig(src) == struct_sig(hyp):
        return
    if src.tag != hyp.tag:
        reasons.append("tag_mismatch")
        return
    sa = list(src.attrib.items())
    ha = list(hyp.attrib.items())
    if sa != ha:
        if sorted(sa) == sorted(ha):
            reasons.append("attr_order")
        else:
            reasons.append("attr_changed")
    sc, hc = list(src), list(hyp)

    def key(el):
        return (el.tag, tuple(el.attrib.items()))

    sk = [key(c) for c in sc]
    hk = [key(c) for c in hc]
    if sk == hk:
        for c, d in zip(sc, hc):
            _compare(c, d, reasons)
        return
    if len(sk) == len(hk) and sorted(sk) == sorted(hk):
        reasons.append("sibling_reorder")
        used = set()
        for c in sc:
            k = key(c)
            for j, d in enumerate(hc):
                if j in used:
                    continue
                if key(d) == k:
                    used.add(j)
                    _compare(c, d, reasons)
                    break
        return
    st = [c.tag for c in sc]
    ht = [c.tag for c in hc]
    if st == ht:
        for c, d in zip(sc, hc):
            _compare(c, d, reasons)
        return
    if len(st) == len(ht) and sorted(st) == sorted(ht):
        reasons.append("sibling_reorder")
        used = set()
        for c in sc:
            for j, d in enumerate(hc):
                if j in used:
                    continue
                if d.tag == c.tag and key(d) == key(c):
                    used.add(j)
                    _compare(c, d, reasons)
                    break
            else:
                for j, d in enumerate(hc):
                    if j in used:
                        continue
                    if d.tag == c.tag:
                        used.add(j)
                        _compare(c, d, reasons)
                        break
        return
    reasons.append("child_mismatch")
    used = set()
    for c in sc:
        match = None
        for j, d in enumerate(hc):
            if j in used:
                continue
            if key(d) == key(c):
                match = j
                break
        if match is None:
            for j, d in enumerate(hc):
                if j in used:
                    continue
                if d.tag == c.tag:
                    match = j
                    break
        if match is None:
            continue
        used.add(match)
        _compare(c, hc[match], reasons)


def dtd_validate(xml_text: str) -> tuple[bool, int, str]:
    if not CATALOG.is_file():
        return False, 1, f"missing catalog {CATALOG}"
    env = os.environ.copy()
    env["XML_CATALOG_FILES"] = str(CATALOG)
    with tempfile.NamedTemporaryFile("w", suffix=".dita", delete=False, encoding="utf-8") as f:
        f.write(xml_text)
        path = f.name
    try:
        proc = subprocess.run(
            ["xmllint", "--noout", "--valid", path],
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    err = (proc.stderr or "").strip()
    n = 0
    for line in err.splitlines():
        low = line.lower()
        if "validity error" in low or "parser error" in low or "error :" in low or low.startswith("error"):
            n += 1
    if proc.returncode != 0 and n == 0:
        n = 1
    return proc.returncode == 0, n, err[:2000]


def _primary(reasons: list) -> str | None:
    if not reasons:
        return None
    have = set(reasons)
    for name in PRIMARY_ORDER:
        if name in have:
            return name
    return reasons[0]


def score_pair(src_xml: str, hyp_text: str, hit_token_cap: bool = False) -> dict:
    hyp_xml = extract_xml(hyp_text)
    info = {
        "well_formed": False,
        "skeleton_match": False,
        "order_sensitive": True,
        "dtd_valid": False,
        "dtd_errors": None,
        "reward": -1.0,
        "hyp_xml": hyp_xml,
        "error_tail": "",
        "reasons": [],
        "primary_reason": None,
        "truncated": False,
        "format_broken": True,
        "lcs": 0.0,
        "attr_recall": 0.0,
    }
    try:
        src_root = ET.fromstring(_for_et(src_xml))
    except ET.ParseError as e:
        info["error_tail"] = f"source not well-formed: {e}"
        info["reasons"] = ["not_well_formed"]
        info["primary_reason"] = "not_well_formed"
        return info
    src_sigs = _sigs(src_root)
    try:
        hyp_root = ET.fromstring(_for_et(hyp_xml))
        info["well_formed"] = True
    except ET.ParseError as e:
        info["error_tail"] = f"hypothesis not well-formed: {e}"
        reasons = ["not_well_formed"]
        if hit_token_cap:
            reasons.insert(0, "truncated")
            info["truncated"] = True
        info["reasons"] = reasons
        info["primary_reason"] = _primary(reasons)
        info["reward"] = -0.7 if info["truncated"] else -1.0
        return info

    reasons = []
    _compare(src_root, hyp_root, reasons)
    hyp_sigs = _sigs(hyp_root)
    info["skeleton_match"] = hyp_sigs == src_sigs and not reasons
    info["lcs"] = round(_lcs_ratio(src_sigs, hyp_sigs), 4)
    info["attr_recall"] = round(_attr_recall(src_sigs, hyp_sigs), 4)
    # A perfect preorder with no logged reasons is the definition of a match.
    # If reasons were logged, skeleton_match stays false even if a bug dropped one.
    if reasons:
        info["skeleton_match"] = False

    valid, nerr, err = dtd_validate(hyp_xml)
    info["dtd_valid"] = valid
    info["dtd_errors"] = nerr
    info["error_tail"] = err[-500:]
    if not valid:
        reasons.append("dtd_invalid")
    if hit_token_cap and not info["skeleton_match"]:
        reasons.insert(0, "truncated")
        info["truncated"] = True

    # unique, stable
    dedup = []
    for r in reasons:
        if r not in dedup:
            dedup.append(r)
    info["reasons"] = dedup
    info["primary_reason"] = _primary(dedup)
    info["format_broken"] = bool(dedup) or not info["skeleton_match"] or not valid

    lcs = info["lcs"]
    attr = info["attr_recall"]
    reward = 0.50 * lcs + 0.20 * attr
    reward += 0.15 if info["well_formed"] else 0.0
    reward += 0.15 if valid else 0.0
    if info["truncated"]:
        reward -= 0.30
    if not info["format_broken"]:
        reward = 1.0
    info["reward"] = round(max(-1.0, min(1.0, reward)), 4)
    return info



VERBATIM_TAGS = {
    "codeblock", "codeph", "filepath", "cmdname", "userinput", "systemoutput",
    "apiname", "option", "parmname", "synph", "kwd", "var",
}
HIGH_PUNCT = set("?!:;")
ENG_QUOTES = set('"\u201c\u201d')
SRC_QUOTES = set('"\u201c\u201d\u00ab\u00bb')


def _levenshtein(a, b) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (x != y)
            cur.append(ins if ins <= delete and ins <= sub else delete if delete <= sub else sub)
        prev = cur
    return prev[m]


def _lcs_index_pairs(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = dp[i]
        prow = dp[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                row[j] = prow[j - 1] + 1
            else:
                row[j] = prow[j] if prow[j] >= row[j - 1] else row[j - 1]
    pairs = []
    i, j = n, m
    while i and j:
        if a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def _text_nodes(root, skip_verbatim=True):
    chunks = []

    def walk(el, verbatim):
        here = verbatim or (skip_verbatim and el.tag in VERBATIM_TAGS)
        if el.text and not here:
            chunks.append(el.text)
        for child in list(el):
            walk(child, here)
            if child.tail and not here:
                chunks.append(child.tail)
    walk(root, False)
    return "\n".join(chunks)


def _strip_doctype(xml_text: str) -> str:
    start = xml_text.find("<!DOCTYPE")
    if start < 0:
        return xml_text
    i = start
    depth = 0
    while i < len(xml_text):
        ch = xml_text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == ">" and depth == 0:
            return xml_text[:start] + xml_text[i + 1 :]
        i += 1
    return xml_text


def _for_et(xml_text: str) -> str:
    return _strip_doctype(expand_nbsp(xml_text))


def _strip_tags(xml_text: str) -> str:
    no_decl = _strip_doctype(xml_text)
    no_decl = re.sub(r"<\?xml[^>]*\?>", " ", no_decl)
    return re.sub(r"<[^>]+>", " ", no_decl)


def _punct_on_text(src_text: str, hyp_text: str):
    bad = 0
    quotes = 0
    spans = []
    for i, ch in enumerate(hyp_text):
        if ch not in HIGH_PUNCT:
            continue
        prev = hyp_text[i - 1] if i else ""
        if prev == "\u00A0":
            continue
        bad += 1
        spans.append(hyp_text[max(0, i - 16): i + 1].replace("\u00A0", "\\u00a0"))
    source_has_quote = any(ch in SRC_QUOTES for ch in src_text)
    if source_has_quote:
        for i, ch in enumerate(hyp_text):
            if ch in ENG_QUOTES:
                quotes += 1
                spans.append(hyp_text[max(0, i - 16): i + 1])
    return {
        "bad_high_punct": bad,
        "english_quotes": quotes,
        "punct": bad + quotes,
        "spans": spans[:12],
        "source_has_quotation": source_has_quote,
        "guillemet_open": hyp_text.count("\u00ab"),
        "guillemet_close": hyp_text.count("\u00bb"),
    }


def _struct_counts(src, hyp, acc):
    if src.tag != hyp.tag:
        acc["tag_mismatch"] += 1
        return
    keys = set(src.attrib) | set(hyp.attrib)
    for k in keys:
        if src.attrib.get(k) != hyp.attrib.get(k):
            acc["attr_value"] += 1
    sc, hc = list(src), list(hyp)
    st = [c.tag for c in sc]
    ht = [c.tag for c in hc]
    pairs = _lcs_index_pairs(st, ht)
    # Insertions and deletions. A pure reorder of different tags is both.
    acc["child_order"] += (len(st) - len(pairs)) + (len(ht) - len(pairs))
    for i, j in pairs:
        _struct_counts(sc[i], hc[j], acc)


def count_violations(src_xml: str, hyp_text: str, hit_token_cap: bool = False) -> dict:
    """One shared violation total for reward, candidate selection, and eval.

    schema counts each occurrence of: not-well-formed (1), truncation (1),
    DTD error lines, attribute value changes, and child insertions/deletions
    (a reorder of unlike siblings counts as a deletion plus an insertion).
    punct counts each ? ! : ; in prose text that is not preceded by U+00A0,
    plus each English double quote when the source text contains a quotation.
    Verbatim elements (codeblock, codeph, filepath, and similar) are skipped
    so a URL colon is not a French-typography miss. Reward is minus the total.
    """
    hyp_xml = extract_xml(hyp_text)
    acc = {"tag_mismatch": 0, "attr_value": 0, "child_order": 0, "dtd_errors": 0,
           "not_well_formed": 0, "truncation": 0}
    well = False
    dtd_valid = False
    err_tail = ""
    try:
        src_root = ET.fromstring(_for_et(src_xml))
    except ET.ParseError as e:
        return {
            "violations": 1, "schema": 1, "punct": 0, "reward": -1.0,
            "well_formed": False, "dtd_valid": False, "dtd_errors": None,
            "truncated": False, "hyp_xml": hyp_xml, "error_tail": f"source not well-formed: {e}",
            "parts": {**acc, "bad_high_punct": 0, "english_quotes": 0},
            "punct_detail": {}, "skeleton_match": False,
        }
    try:
        hyp_root = ET.fromstring(_for_et(hyp_xml))
        well = True
    except ET.ParseError as e:
        acc["not_well_formed"] = 1
        if hit_token_cap:
            acc["truncation"] = 1
        err_tail = f"hypothesis not well-formed: {e}"
        src_text = _text_nodes(src_root)
        punct = _punct_on_text(src_text, _strip_tags(_for_et(hyp_xml)))
    else:
        _struct_counts(src_root, hyp_root, acc)
        structural = acc["tag_mismatch"] == 0 and acc["attr_value"] == 0 and acc["child_order"] == 0
        if hit_token_cap and not structural:
            acc["truncation"] = 1
        valid, nerr, err = dtd_validate(hyp_xml)
        dtd_valid = valid
        err_tail = err[-500:]
        if not valid:
            acc["dtd_errors"] = nerr if nerr else 1
        src_text = _text_nodes(src_root)
        punct = _punct_on_text(src_text, _text_nodes(hyp_root))
    if not well:
        structural = False
    schema = sum(acc.values())
    total = schema + punct["punct"]
    skeleton_match = bool(well and structural)
    parts = {**acc, "bad_high_punct": punct["bad_high_punct"], "english_quotes": punct["english_quotes"]}
    return {
        "violations": total,
        "schema": schema,
        "punct": punct["punct"],
        "reward": float(-total),
        "well_formed": well,
        "dtd_valid": dtd_valid,
        "dtd_errors": acc["dtd_errors"] if well else None,
        "truncated": bool(acc["truncation"]),
        "hyp_xml": hyp_xml,
        "error_tail": err_tail,
        "parts": parts,
        "punct_detail": punct,
        "skeleton_match": skeleton_match,
        "format_broken": total > 0,
    }


def punct_stats(xml_text: str) -> dict:
    """French punctuation counts over text nodes only. Not part of the RL reward."""
    raw = expand_nbsp(extract_xml(xml_text))
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {"text_available": False}
    chunks = []
    for el in root.iter():
        if el.text:
            chunks.append(el.text)
        if el.tail:
            chunks.append(el.tail)
    blob = "\n".join(chunks)
    # A double mark is bad when the previous character is not NBSP.
    bad = 0
    good = 0
    marks = []
    for i, ch in enumerate(blob):
        if ch not in "?!;:":
            continue
        prev = blob[i - 1] if i else ""
        if prev == "\u00A0":
            good += 1
        elif prev.isspace():
            bad += 1
            marks.append(blob[max(0, i - 12): i + 1])
        else:
            # no space at all, also wrong for French high punctuation
            bad += 1
            marks.append(blob[max(0, i - 12): i + 1])
    return {
        "text_available": True,
        "guillemet_open": blob.count("«"),
        "guillemet_close": blob.count("»"),
        "ascii_double_quote": blob.count('"'),
        "curly_double_quote": blob.count("“") + blob.count("”"),
        "nbsp_ok_marks": good,
        "bad_high_punct": bad,
        "bad_spans": marks[:8],
    }
