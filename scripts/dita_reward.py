#!/usr/bin/env python3
"""DTD + element-skeleton reward for DITA translations.

Validator: xmllint --valid against OASIS DITA 1.3 technical-content DTDs
(dita-community/org.oasis-open.dita.dita13.doctypes), with the SVG 1.1 driver
include removed. libxml2 otherwise reports an entity-reference loop inside
svgDomain.mod. No topic in this manual uses svg-container.
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


def extract_xml(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    text = FENCE_RE.sub("", text)
    # drop a leading assistant prefix if the model echoed one
    starts = []
    for marker in ("<?xml", "<!DOCTYPE", "<topic", "<task", "<concept", "<reference", "<troubleshooting", "<glossentry", "<glossgroup"):
        i = text.find(marker)
        if i >= 0:
            starts.append(i)
    if not starts:
        return text.strip()
    return text[min(starts):].strip()


def skeleton(xml_text: str):
    root = ET.fromstring(xml_text)
    seq = []

    def walk(el):
        attrs = tuple(sorted((k, v) for k, v in el.attrib.items()))
        seq.append((el.tag, attrs))
        for child in list(el):
            walk(child)

    walk(root)
    return seq


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


def score_pair(src_xml: str, hyp_text: str) -> dict:
    hyp_xml = extract_xml(hyp_text)
    info = {
        "well_formed": False,
        "skeleton_match": False,
        "dtd_valid": False,
        "dtd_errors": None,
        "reward": -1.0,
        "hyp_xml": hyp_xml,
        "error_tail": "",
    }
    try:
        sk_s = skeleton(src_xml)
    except ET.ParseError as e:
        info["error_tail"] = f"source not well-formed: {e}"
        return info
    try:
        sk_h = skeleton(hyp_xml)
        info["well_formed"] = True
    except ET.ParseError as e:
        info["error_tail"] = f"hypothesis not well-formed: {e}"
        info["reward"] = -1.0
        return info
    info["skeleton_match"] = sk_h == sk_s
    valid, nerr, err = dtd_validate(hyp_xml)
    info["dtd_valid"] = valid
    info["dtd_errors"] = nerr
    info["error_tail"] = err[-500:]
    if info["skeleton_match"] and valid:
        info["reward"] = 1.0
        return info
    # penalty by validator error count, plus a structure penalty if the skeleton moved
    pen = 0.15 * (nerr if nerr else 1)
    if not info["skeleton_match"]:
        pen += 0.5 + 0.02 * abs(len(sk_h) - len(sk_s))
    info["reward"] = max(-1.0, -pen)
    return info
