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
from collections import Counter
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


# --- Context-aware French typography oracle (evaluator fr-typography-oracle-1) ---
#
# The legacy count_violations() path above is unchanged. It still counts every
# prose colon that is not preceded by U+00A0, including namespace colons, and it
# joins text nodes with newlines, so a non-breaking space at the end of an inline
# element is invisible to the next mark.
#
# typography_oracle() is the reward used by the French SFT / GRPO scripts.
# Invalid documents score in [-1, 0). Valid documents score in [0, 1].
# The scalar does not add a term for the total number of U+00A0 characters.

EVALUATOR_VERSION = "fr-typography-oracle-1"

INLINE_TAGS = {
    "b", "i", "u", "ph", "codeph", "filepath", "xref", "term", "q",
    "uicontrol", "tm", "cite", "keyword", "apiname", "cmdname", "option",
    "parmname", "var", "varname", "userinput", "systemoutput", "synph", "kwd",
    "xmlelement", "xmlatt", "wintitle", "msgph", "msgnum", "image", "sup", "sub",
    "tt", "menucascade", "abbrev", "text",
}

_PROTECTED_RES = (
    re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE),
    re.compile(
        r"\b[\w./\\-]+\.(?:dita|xml|xsl|xslt|xsd|css|js|json|html|png|jpe?g|svg|pdf|txt|md)(?::(?!\s))?",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Za-z_][\w.-]*:[A-Za-z_][\w.-]*"),
    re.compile(r"\b(?:xs|oxy|xi|xsl|xml|xmlns):"),
    re.compile(r"\d{1,2}:\d{2}(?::\d{2})?"),
    re.compile(r"\d{1,4}:\d{1,4}"),
    re.compile(r"(?<=\d):(?=\d)"),
    re.compile(
        r":{1,2}(?:before|after|first-letter|first-line|hover|active|focus|root|empty)\b",
        re.IGNORECASE,
    ),
)

_NARROW_SPACES = set("\u202f\u2009\u2007\u200a")
_QUOTE_CHARS = set('"\u201c\u201d\u00ab\u00bb')
_EN_QUOTE_CHARS = set('"\u201c\u201d')
_GATE_NAMES = (
    "xml_only",
    "well_formed",
    "dtd_valid",
    "structure",
    "protected_attributes",
    "protected_literals",
)


def _is_block_tag(tag: str) -> bool:
    return tag not in INLINE_TAGS


def _merge_spans(spans):
    if not spans:
        return []
    ordered = sorted((a, b) for a, b in spans if b > a)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1]:
            if b > merged[-1][1]:
                merged[-1][1] = b
        else:
            merged.append([a, b])
    return [tuple(item) for item in merged]


def _covered(spans, index: int) -> bool:
    for start, end in spans:
        if start <= index < end:
            return True
        if start > index:
            return False
    return False


def protected_regex_spans(text: str):
    spans = []
    for pattern in _PROTECTED_RES:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
    return _merge_spans(spans)


def _nonoverlapping_literals(text: str):
    found = []
    for pattern in _PROTECTED_RES:
        for match in pattern.finditer(text):
            if match.end() > match.start():
                found.append((match.start(), match.end(), match.group(0)))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    taken = []
    literals = []
    for start, end, literal in found:
        if any(not (end <= left or start >= right) for left, right, _ in taken):
            continue
        taken.append((start, end, literal))
        literals.append(literal)
    return literals


class _Buf:
    def __init__(self):
        self.parts = []
        self.length = 0
        self.protected = []

    def add(self, text: str, protect: bool = False):
        if not text:
            return
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        if protect:
            self.protected.append((start, self.length))

    def text(self) -> str:
        return "".join(self.parts)


def _add_inline(buf: _Buf, el):
    protect = el.tag in VERBATIM_TAGS
    start = buf.length
    buf.add(el.text or "")
    for child in list(el):
        if _is_block_tag(child.tag):
            continue
        _add_inline(buf, child)
    end = buf.length
    if protect and end > start:
        buf.protected.append((start, end))


def inline_flows(root):
    """Block flows. Inline elements are transparent, including their tails."""
    flows = []

    def walk(el):
        buf = _Buf()
        if el.tag in VERBATIM_TAGS:
            buf.add(el.text or "", protect=False)
            # A verbatim block protects its whole flow, tails of nested inlines included.
            for child in list(el):
                if _is_block_tag(child.tag):
                    walk(child)
                else:
                    _add_inline(buf, child)
                    buf.add(child.tail or "")
            text = buf.text()
            flows.append((text, [(0, len(text))] if text else []))
            return
        buf.add(el.text or "")
        for child in list(el):
            if _is_block_tag(child.tag):
                walk(child)
            else:
                _add_inline(buf, child)
            buf.add(child.tail or "")
        text = buf.text()
        protected = _merge_spans(buf.protected + protected_regex_spans(text))
        flows.append((text, protected))

    walk(root)
    return flows


def _verbatim_strings(root):
    found = []

    def walk(el):
        if el.tag in VERBATIM_TAGS:
            buf = _Buf()
            buf.add(el.text or "")
            for child in list(el):
                _add_inline(buf, child)
                buf.add(child.tail or "")
            blob = buf.text()
            if blob.strip():
                found.append(blob)
            return
        for child in list(el):
            walk(child)

    walk(root)
    return found


def _source_literals(root):
    counts = Counter()
    for text, _spans in inline_flows(root):
        for literal in _nonoverlapping_literals(text):
            counts[literal] += 1
    verbatim_counts = Counter(_verbatim_strings(root))
    for blob, count in verbatim_counts.items():
        counts[blob] = max(counts[blob], count)
    return counts


def _literals_preserved(src_root, hyp_root):
    need = _source_literals(src_root)
    blob = "\n".join(text for text, _spans in inline_flows(hyp_root))
    missing = []
    for literal, count in sorted(need.items(), key=lambda item: (-len(item[0]), item[0])):
        if not literal.strip():
            continue
        if blob.count(literal) < count:
            missing.append({"literal": literal, "need": count, "have": blob.count(literal)})
    return not missing, missing


def _spaced_ratio(text: str, index: int) -> bool:
    left = index - 1
    while left >= 0 and text[left] == " ":
        left -= 1
    right = index + 1
    while right < len(text) and text[right] == " ":
        right += 1
    if left < 0 or right >= len(text):
        return False
    if not (text[left].isdigit() and text[right].isdigit()):
        return False
    return (index - left) > 1 or (right - index) > 1


def _excerpt(text: str, index: int) -> str:
    return text[max(0, index - 12): index + 13].replace("\u00A0", "<NBSP>").replace("\u202f", "<NNBSP>")


def analyze_high_marks(text: str, protected):
    protected = _merge_spans(protected)
    sites = []
    for index, char in enumerate(text):
        if char not in HIGH_PUNCT:
            continue
        prev = text[index - 1] if index else ""
        if _covered(protected, index):
            sites.append({
                "index": index,
                "mark": char,
                "kind": "protected",
                "ok": True,
                "ambiguous": False,
                "prev": prev,
                "excerpt": _excerpt(text, index),
            })
            continue
        if index and text[index - 1] in HIGH_PUNCT:
            sites.append({
                "index": index,
                "mark": char,
                "kind": "cluster_continuation",
                "ok": True,
                "ambiguous": False,
                "prev": prev,
                "excerpt": _excerpt(text, index),
            })
            continue
        if _spaced_ratio(text, index):
            sites.append({
                "index": index,
                "mark": char,
                "kind": "ambiguous_spaced_ratio",
                "ok": True,
                "ambiguous": True,
                "prev": prev,
                "excerpt": _excerpt(text, index),
                "reason": "spaced_digit_colon_needs_human_french_review",
            })
            continue
        ambiguous = prev in _NARROW_SPACES
        sites.append({
            "index": index,
            "mark": char,
            "kind": "linguistic",
            "ok": prev == "\u00A0",
            "ambiguous": ambiguous,
            "prev": prev,
            "excerpt": _excerpt(text, index),
            "reason": "narrow_space_not_u00a0" if ambiguous else "",
        })
    return sites


def _guillemet_sites(text: str):
    sites = []
    for index, char in enumerate(text):
        if char == "\u00ab":
            nxt = text[index + 1] if index + 1 < len(text) else ""
            sites.append({
                "index": index,
                "mark": char,
                "kind": "guillemet_open",
                "ok": nxt == "\u00A0",
                "ambiguous": nxt in _NARROW_SPACES,
                "excerpt": _excerpt(text, index),
                "reason": "narrow_space_not_u00a0" if nxt in _NARROW_SPACES else "",
            })
        elif char == "\u00bb":
            prev = text[index - 1] if index else ""
            sites.append({
                "index": index,
                "mark": char,
                "kind": "guillemet_close",
                "ok": prev == "\u00A0",
                "ambiguous": prev in _NARROW_SPACES,
                "excerpt": _excerpt(text, index),
                "reason": "narrow_space_not_u00a0" if prev in _NARROW_SPACES else "",
            })
    return sites


def _flow_bundle(root):
    bundles = []
    for text, protected in inline_flows(root):
        bundles.append({
            "text": text,
            "marks": analyze_high_marks(text, protected),
            "guillemets": _guillemet_sites(text),
            "protected": protected,
        })
    return bundles


def _count_outside_protected(text: str, protected, chars: set) -> int:
    protected = _merge_spans(protected)
    return sum(1 for i, ch in enumerate(text) if ch in chars and not _covered(protected, i))


def _typography_score(src_root, hyp_root):
    src_flows = _flow_bundle(src_root)
    hyp_flows = _flow_bundle(hyp_root)
    src_ling = []
    hyp_ling = []
    ambiguous = []
    for bundle in src_flows:
        for site in bundle["marks"]:
            if site["kind"] == "linguistic":
                src_ling.append(site)
    hyp_bad = []
    for bundle in hyp_flows:
        for site in bundle["marks"]:
            if site["ambiguous"]:
                ambiguous.append(dict(site))
            if site["kind"] == "linguistic":
                hyp_ling.append(site)
                if not site["ok"]:
                    hyp_bad.append(site)
        for site in bundle["guillemets"]:
            if site["ambiguous"]:
                ambiguous.append(dict(site))
    src_quotes = 0
    hyp_guillemets = 0
    hyp_en_quotes = 0
    guillemet_bad = []
    for bundle in src_flows:
        src_quotes += _count_outside_protected(bundle["text"], bundle["protected"], _QUOTE_CHARS)
    for bundle in hyp_flows:
        hyp_guillemets += bundle["text"].count("\u00ab") + bundle["text"].count("\u00bb")
        hyp_en_quotes += _count_outside_protected(bundle["text"], bundle["protected"], _EN_QUOTE_CHARS)
        for site in bundle["guillemets"]:
            if not site["ok"]:
                guillemet_bad.append(site)
    omitted = max(0, len(src_ling) - len(hyp_ling))
    omitted_quotes = max(0, src_quotes - hyp_guillemets)
    en_penalty = hyp_en_quotes if src_quotes else 0
    penalty = omitted + len(hyp_bad) + len(guillemet_bad) + omitted_quotes + en_penalty
    base = len(src_ling) + src_quotes
    if penalty == 0:
        score = 1.0
    else:
        score = max(0.0, 1.0 - penalty / max(base, 1))
    for site in ambiguous:
        site["needs_human_french_review"] = True
        site["reviewer_signed_off"] = False
    return {
        "score": round(score, 4),
        "source_linguistic": len(src_ling),
        "hyp_linguistic": len(hyp_ling),
        "omitted_marks": omitted,
        "bad_spacing": len(hyp_bad),
        "guillemet_bad": len(guillemet_bad),
        "omitted_quotes": omitted_quotes,
        "english_quotes_penalized": en_penalty,
        "source_quote_marks": src_quotes,
        "hyp_guillemets": hyp_guillemets,
        "penalty": penalty,
        "base": base,
        "nbsp_count": sum(bundle["text"].count("\u00A0") for bundle in hyp_flows),
        "ambiguous_spans": ambiguous,
        "bad_excerpts": [site["excerpt"] for site in hyp_bad[:12]],
    }


def _xml_only_root(text: str):
    """Return (ok, element_or_None, error). Does not search inside prose for a root."""
    if not isinstance(text, str) or not text.strip():
        return False, None, "empty"
    raw = text.strip().lstrip("\ufeff")
    if not raw.startswith("<") or not raw.endswith(">"):
        return False, None, "not_xml_framed"
    lowered = raw.lower()
    if "```" in raw or "<think>" in lowered or "</think>" in lowered:
        return False, None, "wrapper_prose"
    peeled = raw
    if peeled.startswith("<?"):
        end = peeled.find("?>")
        if end < 0:
            return False, None, "bad_declaration"
        peeled = peeled[end + 2:].lstrip()
    if peeled.startswith("<!DOCTYPE"):
        stripped = _strip_doctype(peeled).lstrip()
        if stripped == peeled:
            return False, None, "bad_doctype"
        peeled = stripped
    if not peeled.startswith("<"):
        return False, None, "missing_root"
    try:
        root = ET.fromstring(expand_nbsp(peeled))
    except ET.ParseError as exc:
        return False, None, f"not_well_formed: {exc}"
    return True, root, ""


def _preorder_attrs(root):
    seq = []

    def walk(el):
        seq.append((el.tag, tuple(el.attrib.items())))
        for child in list(el):
            walk(child)

    walk(root)
    return seq


def typography_oracle(src_xml: str, hyp_text: str) -> dict:
    """Gated French typography reward.

    Hard gate: the raw response is XML only, well-formed, DTD-valid, the ordered
    structure matches, protected attributes match, and protected literals from
    the source are preserved. A failing gate scores in [-1, 0), below every
    valid score. A passing gate scores typography in [0, 1]: linguistic
    U+00A0 before : ; ? ! , U+00A0 inside guillemets, omitted source marks, and
    unconverted English quotes. Namespace, time, ratio, URL, filename, and code
    colons are not linguistic. Total U+00A0 count is reported and is not a
    reward term. No NFKC normalization.
    """
    info = {
        "evaluator_version": EVALUATOR_VERSION,
        "reward": -1.0,
        "gate_passed": False,
        "gates": {name: False for name in _GATE_NAMES},
        "gate_failures": [],
        "typography": None,
        "nbsp_count": 0,
        "reward_uses_nbsp_count": False,
        "ambiguous_spans": [],
        "needs_human_french_review": False,
        "human_french_review_signed_off": False,
        "reviewer": None,
        "protected_literal_misses": [],
        "structure_reasons": [],
        "error": "",
    }
    try:
        src_root = ET.fromstring(_for_et(src_xml))
    except ET.ParseError as exc:
        info["error"] = f"source not well-formed: {exc}"
        info["gate_failures"] = list(_GATE_NAMES)
        return info

    xml_ok, hyp_root, err = _xml_only_root(hyp_text)
    info["gates"]["xml_only"] = bool(xml_ok and hyp_root is not None)
    info["gates"]["well_formed"] = hyp_root is not None
    if hyp_root is None:
        info["error"] = err
        info["gates"]["xml_only"] = False
        info["gates"]["well_formed"] = False

    if hyp_root is not None:
        reasons = []
        _compare(src_root, hyp_root, reasons)
        dedup = []
        for reason in reasons:
            if reason not in dedup:
                dedup.append(reason)
        info["structure_reasons"] = dedup
        structure_ok = not dedup and _sigs(src_root) == _sigs(hyp_root)
        info["gates"]["structure"] = structure_ok
        info["gates"]["protected_attributes"] = _preorder_attrs(src_root) == _preorder_attrs(hyp_root)
        preserved, missing = _literals_preserved(src_root, hyp_root)
        info["gates"]["protected_literals"] = preserved
        info["protected_literal_misses"] = missing[:12]
        raw = hyp_text.strip().lstrip("\ufeff")
        valid, _nerr, err_tail = dtd_validate(raw)
        info["gates"]["dtd_valid"] = bool(valid)
        if not valid:
            info["error"] = err_tail[-400:]
        typo = _typography_score(src_root, hyp_root)
        info["typography"] = typo
        info["nbsp_count"] = typo["nbsp_count"]
        info["ambiguous_spans"] = typo["ambiguous_spans"]
        info["needs_human_french_review"] = bool(typo["ambiguous_spans"])
    else:
        info["gates"]["dtd_valid"] = False
        info["gates"]["structure"] = False
        info["gates"]["protected_attributes"] = False
        info["gates"]["protected_literals"] = False

    passed = sum(1 for name in _GATE_NAMES if info["gates"][name])
    info["gate_failures"] = [name for name in _GATE_NAMES if not info["gates"][name]]
    if passed == len(_GATE_NAMES):
        info["gate_passed"] = True
        info["reward"] = info["typography"]["score"]
    else:
        info["gate_passed"] = False
        info["reward"] = round(-1.0 + 0.99 * (passed / len(_GATE_NAMES)), 4)
    return info


def gated_scalar(src_xml: str, hyp_text: str) -> float:
    """The training reward. See docs/plans/fr-punct-sft-grpo.md."""
    return float(typography_oracle(src_xml, hyp_text)["reward"])
