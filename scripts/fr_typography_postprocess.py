#!/usr/bin/env python3
"""Deterministic French typography postprocessor.

Inserts U+00A0 before linguistic : ; ? ! and inside guillemets. It does not
NFKC-normalize, wrap marks in codeph, delete prose, or split namespace names,
times, ratios, URLs, filenames, or other protected literals. It is a baseline,
not a model.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dita_reward import (  # noqa: E402
    INLINE_TAGS,
    VERBATIM_TAGS,
    _guillemet_sites,
    _merge_spans,
    _strip_doctype,
    analyze_high_marks,
    expand_nbsp,
    protected_regex_spans,
)

_TOKEN_RE = re.compile(
    r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?[^>]*\?>|<!DOCTYPE\b[^>[]*(?:\[[^\]]*\]\s*)?>|"
    r"</[^>]+>|<[^>]+>|[^<]+",
    re.DOTALL | re.IGNORECASE,
)
_ENTITY_RE = re.compile(
    r"&nbsp;|&#0*160;|&#x0*a0;|&amp;|&lt;|&gt;|&quot;|&apos;|&#([0-9]+);|&#x([0-9A-Fa-f]+);",
    re.IGNORECASE,
)


def _numeric_char(codepoint: int):
    if codepoint < 0 or codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
        return None
    if codepoint in (0xFFFE, 0xFFFF):
        return None
    return chr(codepoint)


def decode_xml_text_with_map(raw: str):
    """Decode predefined XML entities, nbsp, and one-character numeric references.

    Does not apply NFKC. Each decoded character keeps the raw span that produced it.
    """
    parts = []
    spans = []
    index = 0
    while index < len(raw):
        match = _ENTITY_RE.match(raw, index)
        if match is None:
            parts.append(raw[index])
            spans.append((index, index + 1))
            index += 1
            continue
        token = match.group(0)
        lowered = token.lower()
        if lowered == "&nbsp;" or re.fullmatch(r"&#0*160;", token, re.IGNORECASE):
            char = "\u00A0"
        elif re.fullmatch(r"&#x0*a0;", token, re.IGNORECASE):
            char = "\u00A0"
        elif token == "&amp;":
            char = "&"
        elif token == "&lt;":
            char = "<"
        elif token == "&gt;":
            char = ">"
        elif token == "&quot;":
            char = '"'
        elif token == "&apos;":
            char = "'"
        elif match.group(1):
            char = _numeric_char(int(match.group(1)))
        elif match.group(2):
            char = _numeric_char(int(match.group(2), 16))
        else:
            char = None
        if char is None:
            for offset, ch in enumerate(token):
                parts.append(ch)
                spans.append((match.start() + offset, match.start() + offset + 1))
        else:
            parts.append(char)
            spans.append((match.start(), match.end()))
        index = match.end()
    return "".join(parts), spans


def _collect_slots(root):
    slots = []

    def walk(el):
        slots.append(("text", el))
        for child in list(el):
            walk(child)
            slots.append(("tail", child))

    walk(root)
    return slots


def _parsed(kind: str, el) -> str:
    value = el.text if kind == "text" else el.tail
    return value or ""


def _groups(root):
    groups = []

    def add_inline(group, el, force_protect: bool):
        protect = force_protect or el.tag in VERBATIM_TAGS
        group.append(("text", el, protect))
        for child in list(el):
            if child.tag not in INLINE_TAGS:
                continue
            add_inline(group, child, protect)
            group.append(("tail", child, protect))

    def walk(el, group, force_protect: bool):
        if el.tag in VERBATIM_TAGS and el.tag not in INLINE_TAGS:
            block = []
            groups.append(block)
            block.append(("text", el, True))
            for child in list(el):
                if child.tag in INLINE_TAGS:
                    add_inline(block, child, True)
                else:
                    nested = []
                    groups.append(nested)
                    walk(child, nested, True)
                block.append(("tail", child, True))
            return
        group.append(("text", el, force_protect or el.tag in VERBATIM_TAGS))
        for child in list(el):
            if child.tag in INLINE_TAGS:
                add_inline(group, child, force_protect)
            else:
                child_group = []
                groups.append(child_group)
                walk(child, child_group, False)
            group.append(("tail", child, force_protect))

    root_group = []
    groups.append(root_group)
    walk(root, root_group, False)
    return [group for group in groups if group]


def postprocess_french_typography(xml_text: str) -> str:
    """Insert linguistic U+00A0 without changing tags.

    Not-well-formed input, or input whose raw text nodes do not align with the
    parsed character data, is returned unchanged.
    """
    if not isinstance(xml_text, str) or "<" not in xml_text:
        return xml_text
    tokens = _TOKEN_RE.findall(xml_text)
    if "".join(tokens) != xml_text:
        return xml_text
    root_start = None
    root_end = None
    for index, token in enumerate(tokens):
        if token.startswith("<?") or token.startswith("<!"):
            continue
        if token.startswith("<"):
            if root_start is None:
                root_start = index
            root_end = index
    if root_start is None:
        return xml_text
    text_token_indexes = [
        index for index, token in enumerate(tokens)
        if root_start < index < root_end and not token.startswith("<")
    ]
    try:
        root = ET.fromstring(expand_nbsp(_strip_doctype(xml_text)))
    except ET.ParseError:
        return xml_text
    slots = _collect_slots(root)
    if len(slots) != len(text_token_indexes):
        return xml_text

    decoded_by_key = {}
    for (kind, el), token_index in zip(slots, text_token_indexes):
        decoded, spans = decode_xml_text_with_map(tokens[token_index])
        if decoded != _parsed(kind, el):
            return xml_text
        decoded_by_key[(kind, id(el))] = (token_index, spans)

    edits = {token_index: [] for token_index in text_token_indexes}
    for group in _groups(root):
        pieces = []
        cursor = 0
        for kind, el, protect in group:
            text = _parsed(kind, el)
            token_index, spans = decoded_by_key[(kind, id(el))]
            pieces.append({
                "start": cursor,
                "text": text,
                "token_index": token_index,
                "spans": spans,
                "protect": protect,
            })
            cursor += len(text)
        flow = "".join(piece["text"] for piece in pieces)
        if not flow:
            continue
        protected = [
            (piece["start"], piece["start"] + len(piece["text"]))
            for piece in pieces
            if piece["protect"] and piece["text"]
        ]
        protected = _merge_spans(protected + protected_regex_spans(flow))

        def locate(flow_index: int):
            for piece in pieces:
                local = flow_index - piece["start"]
                if 0 <= local < len(piece["text"]):
                    return piece, local
            return None

        def insert_before(flow_index: int):
            if flow_index > 0 and flow[flow_index - 1] == " ":
                prev = locate(flow_index - 1)
                if prev is not None:
                    piece, local = prev
                    raw_lo, raw_hi = piece["spans"][local]
                    if raw_hi == raw_lo + 1:
                        edits[piece["token_index"]].append((raw_lo, 1, "\u00A0"))
                        return
            found = locate(flow_index)
            if found is None:
                return
            piece, local = found
            raw_lo, _raw_hi = piece["spans"][local]
            edits[piece["token_index"]].append((raw_lo, 0, "\u00A0"))

        def insert_after(flow_index: int):
            nxt = flow_index + 1
            if nxt < len(flow) and flow[nxt] == " ":
                found = locate(nxt)
                if found is not None:
                    piece, local = found
                    raw_lo, raw_hi = piece["spans"][local]
                    if raw_hi == raw_lo + 1:
                        edits[piece["token_index"]].append((raw_lo, 1, "\u00A0"))
                        return
            found = locate(flow_index)
            if found is None:
                return
            piece, local = found
            _raw_lo, raw_hi = piece["spans"][local]
            edits[piece["token_index"]].append((raw_hi, 0, "\u00A0"))

        for site in analyze_high_marks(flow, protected):
            if site["kind"] == "linguistic" and not site["ok"]:
                insert_before(site["index"])
        for site in _guillemet_sites(flow):
            if site["ok"]:
                continue
            if site["mark"] == "\u00ab":
                insert_after(site["index"])
            else:
                insert_before(site["index"])

    updated = list(tokens)
    for token_index, ops in edits.items():
        if not ops:
            continue
        raw = updated[token_index]
        seen = set()
        ordered = []
        for op in sorted(ops, key=lambda item: (item[0], -item[1]), reverse=True):
            if op in seen:
                continue
            seen.add(op)
            ordered.append(op)
        for pos, old_len, new in ordered:
            raw = raw[:pos] + new + raw[pos + old_len:]
        updated[token_index] = raw
    return "".join(updated)


def main():
    if len(sys.argv) > 1:
        sys.stdout.write(postprocess_french_typography(Path(sys.argv[1]).read_text(encoding="utf-8")))
    else:
        sys.stdout.write(postprocess_french_typography(sys.stdin.read()))


if __name__ == "__main__":
    main()
