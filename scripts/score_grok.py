#!/usr/bin/env python3
"""Score EN->RO topic translations with Grok 4.7.

No XAI_API_KEY is stored on this box (checked box-secrets card keys and the
process environment). The call goes to the same model through OpenRouter
as `x-ai/grok-4.7`. This is not a different judge model.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = """Score this English-to-Romanian translation of one short Oxygen XML Editor user-guide topic.

Rubric, integers from 1 to 5:
- faithfulness: 5 means the technical meaning, UI labels, and steps are preserved; 1 means the meaning is missing or wrong.
- fluency: 5 means natural Romanian technical prose; 1 means broken Romanian or still English.

Ignore XML tags. Judge the words a reader would see.
Return only JSON: {"faithfulness": 1-5, "fluency": 1-5, "note": "one short sentence"}
"""


def text_of(xml: str) -> str:
    if not xml:
        return ""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return re.sub(r"<[^>]+>", " ", xml)[:4000]
    parts = []
    if root.text and root.text.strip():
        parts.append(root.text.strip())
    for el in root.iter():
        if el.text and el.text.strip():
            parts.append(el.text.strip())
        if el.tail and el.tail.strip():
            parts.append(el.tail.strip())
    # dedupe consecutive
    out = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    return "\n".join(out)[:6000]


def load_key() -> str:
    if os.environ.get("XAI_API_KEY"):
        raise SystemExit("XAI_API_KEY is set; this script was left on the OpenRouter path on purpose")
    secrets = json.loads(Path("/home/box/agent-data/box-secrets.json").read_text())
    key = secrets["card"]["OPENROUTER_API_KEY"]
    if not key:
        raise SystemExit("no OpenRouter key")
    return key


def ask(key: str, src_txt: str, hyp_txt: str) -> dict:
    body = {
        "model": "x-ai/grok-4.7",
        "temperature": 0,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"SOURCE_EN:\n{src_txt}\n\nTRANSLATION_RO:\n{hyp_txt}"},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode())
    content = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.DOTALL)
    parsed = json.loads(m.group(0) if m else content)
    parsed["raw"] = content[:500]
    usage = data.get("usage") or {}
    parsed["usage"] = usage
    return parsed


def main():
    key = load_key()
    src_root = ROOT / "data" / "userguide" / "DITA"
    reports = {}
    for name in ("baseline_test.json", "after_test.json"):
        blob = json.loads((ROOT / "results" / name).read_text())
        rows_out = []
        for row in blob["rows"]:
            src = (src_root / row["path"]).read_text(encoding="utf-8")
            scored = ask(key, text_of(src), text_of(row.get("hyp_xml") or ""))
            rows_out.append({
                "path": row["path"],
                "faithfulness": scored.get("faithfulness"),
                "fluency": scored.get("fluency"),
                "note": scored.get("note"),
            })
            print(name, row["path"], scored.get("faithfulness"), scored.get("fluency"), flush=True)
        faith = [r["faithfulness"] for r in rows_out if isinstance(r["faithfulness"], (int, float))]
        flu = [r["fluency"] for r in rows_out if isinstance(r["fluency"], (int, float))]
        reports[name] = {
            "model": "x-ai/grok-4.7",
            "route": "openrouter",
            "reason": "No XAI_API_KEY in box secrets or environment. Same Grok 4.7 model via OpenRouter.",
            "rubric": RUBRIC.strip(),
            "mean_faithfulness": round(sum(faith) / len(faith), 3) if faith else None,
            "mean_fluency": round(sum(flu) / len(flu), 3) if flu else None,
            "rows": rows_out,
        }
    (ROOT / "results" / "grok_scores.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows" and kk != "rubric"} for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
