#!/usr/bin/env python3
"""Build the frozen French typography split.

Targets in the plan: 100 verified real pairs, 500 deterministic synthetic,
300 mixed literal counterexamples, 100 repair/no-op. Real pairs are included
only when a French reference exists and can be checked. This repository's
Oxygen user guide is English. No French gold is invented. The shortfall is
recorded in the manifest.

The historical 10 stay a regression list. They are not the fresh sealed test.
xml-schema-diagram-* and oxy-* each stay in one partition.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dita_reward import gated_scalar  # noqa: E402
from fr_typography_postprocess import postprocess_french_typography  # noqa: E402

SEED = 1337
OUT = ROOT / "split" / "fr_style"
NBSP = "\u00A0"

DOC = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">
<topic id="{tid}">
  <title>{title}</title>
  <body>
    {body}
  </body>
</topic>
"""


def topic(body: str, title: str, tid: str) -> str:
    return DOC.format(tid=tid, title=title, body=body)


def _find_dita() -> Path | None:
    candidates = []
    if os.environ.get("USERGUIDE_DITA"):
        candidates.append(Path(os.environ["USERGUIDE_DITA"]))
    candidates.append(ROOT / "data" / "userguide" / "DITA")
    candidates.append(Path("/tmp/ug/DITA"))
    for path in candidates:
        if path.is_dir():
            return path
    return None


def _family(name: str) -> str | None:
    if name.startswith("xml-schema-diagram-"):
        return "xml-schema-diagram"
    if name.startswith("oxy-"):
        return "oxy"
    return None


def _load_holdout_names() -> set[str]:
    names = set()
    manifest = json.loads((ROOT / "split" / "manifest_fr_punct.json").read_text())
    for key in ("test", "val", "train"):
        for rel in manifest.get(key, []):
            names.add(Path(rel).name)
    metrics = ROOT / "results" / "fr_nbsp_sft2" / "metrics.json"
    if metrics.is_file():
        for rel in json.loads(metrics.read_text()).get("held_out_paths", []):
            names.add(Path(rel).name)
    return names


def _families(dita_root: Path | None) -> dict:
    groups = {"xml-schema-diagram": [], "oxy": []}
    if dita_root is None:
        return {
            "corpus_present": False,
            "groups": groups,
            "partition": {},
            "note": "User-guide tree was not on disk. Family membership was not scanned.",
        }
    for path in sorted(dita_root.rglob("*.dita")):
        family = _family(path.name)
        if family is None:
            continue
        groups[family].append(str(path.relative_to(dita_root)))
    holdouts = _load_holdout_names()
    partition = {}
    for family, members in groups.items():
        hit = [rel for rel in members if Path(rel).name in holdouts]
        # One partition for the whole family. A held-out member keeps its
        # near-duplicates out of train and out of the fresh sealed test.
        partition[family] = {
            "partition": "regression_not_fresh_test" if hit else "unverified_real_not_used",
            "n": len(members),
            "held_out_members": hit,
            "members": members,
        }
    return {"corpus_present": True, "corpus": str(dita_root), "partition": partition}


def _finish(src_body: str, fr_body: str, title: str, tid: str) -> tuple[str, str]:
    source = topic(src_body, title, tid)
    draft = topic(fr_body, title, tid)
    target = postprocess_french_typography(draft)
    reward = gated_scalar(source, target)
    if reward != 1.0:
        raise SystemExit(f"{tid} target reward {reward}, expected 1.0")
    return source, target


def _record(ident: str, bucket: str, task: str, source: str, target: str) -> dict:
    return {
        "id": ident,
        "bucket": bucket,
        "task": task,
        "family": "synthetic",
        "source_xml": source,
        "target_xml": target,
        "origin": "deterministic-synthetic",
        "verified_real": False,
        "human_french_review_signed_off": False,
        "reviewer": None,
    }


def _synthetic_pairs(n: int, ident_prefix: str, token: str) -> list[dict]:
    nouns = [
        ("dialog", "boîte"),
        ("report", "rapport"),
        ("table", "tableau"),
        ("figure", "figure"),
        ("index", "index"),
        ("macro", "macro"),
        ("panel", "panneau"),
        ("wizard", "assistant"),
        ("field", "champ"),
        ("button", "bouton"),
    ]
    rows = []
    for i in range(n):
        en, fr = nouns[i % len(nouns)]
        kind = i % 4
        if kind == 0:
            src_body = f"<p>Open the {en}: choose Export {token}{i}.</p>"
            fr_body = f"<p>Ouvrez le {fr} : choisissez Exporter {token}{i}.</p>"
        elif kind == 1:
            src_body = f"<p>Save the {en}; then close {token}{i}.</p>"
            fr_body = f"<p>Enregistrez le {fr} ; puis fermez {token}{i}.</p>"
        elif kind == 2:
            src_body = f"<p>Open the {en}? Confirm {token}{i}.</p>"
            fr_body = f"<p>Ouvrez le {fr} ? Confirmez {token}{i}.</p>"
        else:
            src_body = f"<p>Save the {en}! Done {token}{i}.</p>"
            fr_body = f"<p>Enregistrez le {fr} ! Terminé {token}{i}.</p>"
        source, target = _finish(src_body, fr_body, "Titre", f"{ident_prefix}{i}")
        rows.append(_record(f"{ident_prefix}{i:05d}", "synthetic", "translate", source, target))
    return rows


def _mixed_pairs(n: int) -> list[dict]:
    literals = [
        ("<i>xs:override</i>", "<i>xs:override</i>"),
        ("oxy:is-editable", "oxy:is-editable"),
        ("xi:include", "xi:include"),
        ("xsl:stylesheet", "xsl:stylesheet"),
        ("12:30", "12:30"),
        ("16:9", "16:9"),
        ("topic.dita", "topic.dita"),
        ("http://example.com/a:b", "http://example.com/a:b"),
        (":before", ":before"),
        ("oxy:", "oxy:"),
    ]
    rows = []
    for i in range(n):
        shown, same = literals[i % len(literals)]
        where = i % 3
        if where == 0:
            src_body = f"<p>Read the note: keep {shown} in item {i}.</p>"
            fr_body = f"<p>Lisez la note : gardez {same} dans l'élément {i}.</p>"
            title = "Titre"
        elif where == 1:
            src_body = f"<p>Read the note: keep the token in item {i}.</p>"
            fr_body = f"<p>Lisez la note : gardez le jeton dans l'élément {i}.</p>"
            title = shown if "<" not in shown else "Titre"
            if "<" in shown:
                src_body = f"<p>Read the note: keep {shown} in item {i}.</p>"
                fr_body = f"<p>Lisez la note : gardez {same} dans l'élément {i}.</p>"
        else:
            src_body = f"<p>Keep {shown}. Read the note: item {i}.</p>"
            fr_body = f"<p>Gardez {same}. Lisez la note : élément {i}.</p>"
            title = "Titre"
        source, target = _finish(src_body, fr_body, title, f"mix{i}")
        row = _record(f"mix-{i:05d}", "mixed_literal_counterexample", "translate", source, target)
        spaced = same.replace(":", NBSP + ":", 1) if ":" in same else same
        row["trap_not_a_target"] = (
            "Inserting U+00A0 into the protected literal is not the label. "
            f"Example of the rejected edit: {spaced}"
        )
        rows.append(row)
    return rows


def _repair_and_noop(synthetic: list[dict]) -> tuple[list[dict], list[dict]]:
    repairs = []
    noops = []
    for row in synthetic:
        target = row["target_xml"]
        if NBSP not in target:
            continue
        broken = target.replace(NBSP, " ")
        if gated_scalar(broken, target) != 1.0:
            continue
        if gated_scalar(broken, broken) >= 1.0:
            continue
        repairs.append(_record(f"repair-{len(repairs):05d}", "repair", "repair", broken, target))
        if len(repairs) == 50:
            break
    for row in synthetic:
        target = row["target_xml"]
        if gated_scalar(target, target) != 1.0:
            continue
        noops.append(_record(f"noop-{len(noops):05d}", "noop", "noop", target, target))
        if len(noops) == 50:
            break
    if len(repairs) != 50 or len(noops) != 50:
        raise SystemExit(f"repair/noop shortfall repairs={len(repairs)} noops={len(noops)}")
    return repairs, noops


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic = _synthetic_pairs(500, "syn", "")
    mixed = _mixed_pairs(300)
    repairs, noops = _repair_and_noop(synthetic)
    sealed = _synthetic_pairs(40, "sealed", "SEALEDTOKEN")
    for index, row in enumerate(sealed):
        row["id"] = f"sealed-{index:05d}"
        row["bucket"] = "sealed_test"
        row["family"] = "sealed_synthetic"
    train = synthetic + mixed + repairs + noops
    train_ids = {row["id"] for row in train}
    sealed_ids = {row["id"] for row in sealed}
    if train_ids & sealed_ids:
        raise SystemExit("sealed ids overlap train")
    train_targets = {row["target_xml"] for row in train}
    if any(row["target_xml"] in train_targets for row in sealed):
        raise SystemExit("sealed completion appears in train")
    historical = json.loads((ROOT / "split" / "manifest_fr_punct.json").read_text())["test"]
    if len(historical) != 10:
        raise SystemExit(f"historical test list is {len(historical)}")
    families = _families(_find_dita())
    _write_jsonl(OUT / "train.jsonl", train)
    _write_jsonl(OUT / "sealed_test.jsonl", sealed)
    regression = {
        "role": "regression",
        "not_the_fresh_test": True,
        "do_not_train": True,
        "source": "split/manifest_fr_punct.json test",
        "paths": historical,
        "gold_translations": None,
        "note": "These 10 paths are the previous violation-count test. No French gold was invented for them.",
    }
    (OUT / "historical_regression_10.json").write_text(
        json.dumps(regression, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "seed": SEED,
        "plan_targets": {
            "verified_real_pairs": 100,
            "deterministic_synthetic": 500,
            "mixed_literal_counterexamples": 300,
            "repair_or_noop": 100,
        },
        "shipped": {
            "verified_real_pairs": 0,
            "deterministic_synthetic": len(synthetic),
            "mixed_literal_counterexamples": len(mixed),
            "repair": len(repairs),
            "noop": len(noops),
            "sealed_test": len(sealed),
        },
        "real_pair_shortfall": {
            "shipped": 0,
            "target": 100,
            "reason": (
                "The Oxygen user guide at db722d7 is English. This repo has no verified "
                "French reference translations. Model outputs under results/ were not "
                "promoted to gold. No French reviewer signed off."
            ),
        },
        "sealed_test_is_synthetic": True,
        "sealed_test_note": (
            "40 deterministic synthetic topics with the token SEALEDTOKEN. "
            "Not 100 real pairs. Disjoint from train. Not the historical 10."
        ),
        "do_not_train_on_sealed_test": True,
        "historical_10_are_regressions_not_the_fresh_test": True,
        "human_french_review_signed_off": False,
        "families": families,
        "hashes": {},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["hashes"] = {
        "train.jsonl": _sha256(OUT / "train.jsonl"),
        "sealed_test.jsonl": _sha256(OUT / "sealed_test.jsonl"),
        "historical_regression_10.json": _sha256(OUT / "historical_regression_10.json"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["shipped"], indent=2))


if __name__ == "__main__":
    main()
