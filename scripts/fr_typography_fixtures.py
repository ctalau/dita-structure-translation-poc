#!/usr/bin/env python3
"""Deterministic French typography fixtures for the context-aware oracle.

These are constructed cases, not gold translations of the Oxygen user guide.
No human French reviewer has signed off on an ambiguous span.
"""
from __future__ import annotations

NBSP = "\u00A0"
NNBSP = "\u202F"

_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">
<topic id="{tid}">
  <title>{title}</title>
  <body>
    {body}
  </body>
</topic>
"""


def topic(body: str, title: str = "Titre", tid: str = "t1") -> str:
    return _DOC.format(tid=tid, title=title, body=body)


def _add(cases, *, category, ident, source, hyp, **expect):
    row = {
        "id": ident,
        "category": category,
        "source": source,
        "hyp": hyp,
    }
    row.update(expect)
    cases.append(row)


def build_cases() -> list[dict]:
    cases = []
    stems = [
        ("note", "note"),
        ("dialog", "boîte"),
        ("file", "fichier"),
        ("list", "liste"),
        ("page", "page"),
        ("value", "valeur"),
        ("name", "nom"),
        ("step", "étape"),
    ]
    marks = [":", ";", "?", "!"]
    positives = {
        "nbsp": NBSP,
        "dec": "&#160;",
        "hex": "&#xA0;",
        "named": "&nbsp;",
    }
    negatives = {
        "space": " ",
        "glued": "",
    }
    for s_index, (en, fr) in enumerate(stems):
        for mark in marks:
            src_body = f"<p>The {en}{mark} continues {s_index}.</p>"
            src = topic(src_body, title="Title", tid=f"g{s_index}")
            for mode, sep in positives.items():
                hyp = topic(
                    f"<p>La {fr}{sep}{mark} suite {s_index}.</p>",
                    title="Titre",
                    tid=f"g{s_index}",
                )
                _add(
                    cases,
                    category="positives",
                    ident=f"pos-{mark}-{mode}-{s_index}",
                    source=src,
                    hyp=hyp,
                    expect_gate=True,
                    expect_reward=1.0,
                )
            for mode, sep in negatives.items():
                hyp = topic(
                    f"<p>La {fr}{sep}{mark} suite {s_index}.</p>",
                    title="Titre",
                    tid=f"g{s_index}",
                )
                _add(
                    cases,
                    category="negatives",
                    ident=f"neg-{mark}-{mode}-{s_index}",
                    source=src,
                    hyp=hyp,
                    expect_gate=True,
                    expect_reward=0.0,
                )

    src_inline = topic("<p>See the <b>note</b>: details.</p>")
    _add(cases, category="inline_tails", ident="inline-end-nbsp", source=src_inline,
         hyp=topic(f"<p>Voir la <b>note{NBSP}</b>: détails.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="inline_tails", ident="inline-tail-nbsp", source=src_inline,
         hyp=topic(f"<p>Voir la <b>note</b>{NBSP}: détails.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="inline_tails", ident="inline-tail-entity", source=src_inline,
         hyp=topic("<p>Voir la <b>note</b>&#160;: détails.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="inline_tails", ident="inline-tail-space", source=src_inline,
         hyp=topic("<p>Voir la <b>note</b> : détails.</p>"),
         expect_gate=True, expect_reward=0.0)
    _add(cases, category="inline_tails", ident="inline-nested-tail", source=topic("<p>See <b>the <i>note</i></b>: details.</p>"),
         hyp=topic(f"<p>Voir <b>la <i>note</i></b>{NBSP}: détails.</p>"),
         expect_gate=True, expect_reward=1.0)
    for n, wrapper in enumerate(("i", "u", "ph", "term")):
        _add(
            cases,
            category="inline_tails",
            ident=f"inline-wrap-{wrapper}",
            source=topic(f"<p>See the <{wrapper}>note</{wrapper}>: details {n}.</p>"),
            hyp=topic(f"<p>Voir la <{wrapper}>note</{wrapper}>{NBSP}: détails {n}.</p>"),
            expect_gate=True,
            expect_reward=1.0,
        )

    for n, entity in enumerate(("&#160;", "&#xA0;", "&#x00A0;", "&#0160;", "&nbsp;", "&#x0A0;")):
        _add(
            cases,
            category="entities",
            ident=f"entity-{n}",
            source=topic(f"<p>See the note: details {n}.</p>"),
            hyp=topic(f"<p>Voir la note{entity}: détails {n}.</p>"),
            expect_gate=True,
            expect_reward=1.0,
        )
    _add(cases, category="entities", ident="entity-semi", source=topic("<p>See the note; details.</p>"),
         hyp=topic("<p>Voir la note&#xA0;; détails.</p>"), expect_gate=True, expect_reward=1.0)
    _add(cases, category="entities", ident="entity-quest", source=topic("<p>See the note? details.</p>"),
         hyp=topic("<p>Voir la note&nbsp;? détails.</p>"), expect_gate=True, expect_reward=1.0)

    _add(cases, category="clusters", ident="cluster-ok", source=topic("<p>Really?!</p>"),
         hyp=topic(f"<p>Vraiment{NBSP}?!</p>"), expect_gate=True, expect_reward=1.0)
    _add(cases, category="clusters", ident="cluster-extra-nbsp", source=topic("<p>Really?!</p>"),
         hyp=topic(f"<p>Vraiment{NBSP}?{NBSP}!</p>"), expect_gate=True, expect_reward=1.0,
         reward_le_id="cluster-ok")
    _add(cases, category="clusters", ident="cluster-miss", source=topic("<p>Really?!</p>"),
         hyp=topic("<p>Vraiment?!</p>"), expect_gate=True, expect_reward=0.0)
    for n, cluster in enumerate(("?!", "!?", ":;", "?;")):
        _add(
            cases,
            category="clusters",
            ident=f"cluster-var-{n}",
            source=topic(f"<p>Really{cluster} now.</p>"),
            hyp=topic(f"<p>Vraiment{NBSP}{cluster} maintenant.</p>"),
            expect_gate=True,
            expect_reward=1.0,
        )
        _add(
            cases,
            category="clusters",
            ident=f"cluster-var-bad-{n}",
            source=topic(f"<p>Really{cluster} now.</p>"),
            hyp=topic(f"<p>Vraiment{cluster} maintenant.</p>"),
            expect_gate=True,
            expect_reward=0.0,
        )

    _add(cases, category="omitted_marks", ident="omit-colon", source=topic("<p>See the note: details.</p>"),
         hyp=topic("<p>Voir la note.</p>"), expect_gate=True, expect_reward=0.0,
         reward_le_id="neg-:-space-0", reward_lt_id="pos-:-nbsp-0")
    for n in range(8):
        _add(
            cases,
            category="omitted_marks",
            ident=f"omit-{n}",
            source=topic(f"<p>See the note: details {n}.</p>"),
            hyp=topic(f"<p>Voir la note {n}.</p>"),
            expect_gate=True,
            expect_reward=0.0,
        )

    _add(cases, category="deleted_prose", ident="delete-sentence",
         source=topic("<p>See the note: details stay.</p>"),
         hyp=topic("<p>Voir.</p>"),
         expect_gate=True, expect_reward=0.0, reward_le_id="neg-:-space-0")
    for n in range(8):
        _add(
            cases,
            category="deleted_prose",
            ident=f"delete-{n}",
            source=topic(f"<p>Read the note: then save copy {n}.</p>"),
            hyp=topic(f"<p>Lire {n}.</p>"),
            expect_gate=True,
            expect_reward=0.0,
        )

    _add(cases, category="identifier_mutation", ident="mutate-xs",
         source=topic("<p>See <i>xs:override</i> now.</p>", title="Title"),
         hyp=topic("<p>Voir <i>xs:overide</i> maintenant.</p>", title="Titre"),
         expect_gate=False, reward_lt_id="protect-xs-glued")
    for n, bad in enumerate(("xs:overide", "xs-override", "xsl:override", "oxy-override")):
        _add(
            cases,
            category="identifier_mutation",
            ident=f"mutate-{n}",
            source=topic("<p>See <i>xs:override</i> now.</p>"),
            hyp=topic(f"<p>Voir <i>{bad}</i> maintenant.</p>"),
            expect_gate=False,
        )

    _add(cases, category="wrapper_insertion", ident="wrap-codeph",
         source=topic("<p>See the note: details.</p>"),
         hyp=topic("<p>Voir la note <codeph>:</codeph> détails.</p>"),
         expect_gate=False, reward_lt_id="neg-:-space-0")
    for n in range(6):
        _add(
            cases,
            category="wrapper_insertion",
            ident=f"wrap-codeph-{n}",
            source=topic(f"<p>See the note: details {n}.</p>"),
            hyp=topic(f"<p>Voir la note <codeph>:</codeph> détails {n}.</p>"),
            expect_gate=False,
        )

    _add(cases, category="invalid_xml", ident="invalid-truncated",
         source=topic("<p>See the note: details.</p>"),
         hyp="<topic><title>Nope", expect_gate=False, reward_lt_id="neg-:-space-0")
    _add(cases, category="invalid_xml", ident="invalid-prose",
         source=topic("<p>See the note: details.</p>"),
         hyp="Here is the translation:\n" + topic("<p>Voir la note&#160;: détails.</p>"),
         expect_gate=False, reward_lt_id="pos-:-nbsp-0")
    _add(cases, category="invalid_xml", ident="invalid-think",
         source=topic("<p>See the note: details.</p>"),
         hyp="<think>check</think>\n" + topic("<p>Voir la note&#160;: détails.</p>"),
         expect_gate=False)
    _add(cases, category="invalid_xml", ident="invalid-fence",
         source=topic("<p>See the note: details.</p>"),
         hyp="```xml\n" + topic("<p>Voir la note&#160;: détails.</p>") + "\n```",
         expect_gate=False)
    for n in range(4):
        _add(
            cases,
            category="invalid_xml",
            ident=f"invalid-broken-{n}",
            source=topic(f"<p>See the note: details {n}.</p>"),
            hyp=f"<p>Voir {n}</p",
            expect_gate=False,
        )

    ordered = topic('<ul><li id="a">One</li><li id="b">Two</li></ul>')
    swapped = topic('<ul><li id="b">Deux</li><li id="a">Un</li></ul>')
    _add(cases, category="sibling_swaps", ident="swap-li", source=ordered, hyp=swapped,
         expect_gate=False, reward_lt_id="swap-li-ordered")
    _add(cases, category="sibling_swaps", ident="swap-li-ordered", source=ordered,
         hyp=topic('<ul><li id="a">Un</li><li id="b">Deux</li></ul>'),
         expect_gate=True, expect_reward=1.0)
    for n in range(4):
        src = topic(f'<ul><li id="a{n}">One {n}</li><li id="b{n}">Two {n}</li></ul>')
        hyp = topic(f'<ul><li id="b{n}">Deux {n}</li><li id="a{n}">Un {n}</li></ul>')
        _add(cases, category="sibling_swaps", ident=f"swap-{n}", source=src, hyp=hyp, expect_gate=False)

    _add(cases, category="protected_literals", ident="protect-xs-glued",
         source=topic("<p>The <i>xs:override</i> component.</p>", title="Title"),
         hyp=topic("<p>Le composant <i>xs:override</i>.</p>", title="Titre"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="protected_literals", ident="protect-xs-spaced",
         source=topic("<p>The <i>xs:override</i> component.</p>", title="Title"),
         hyp=topic(f"<p>Le composant <i>xs{NBSP}:override</i>.</p>", title="Titre"),
         expect_gate=False, reward_lt_id="protect-xs-glued")
    for n, literal in enumerate((
        "xs:override", "oxy:is-editable", "xi:include", "xsl:stylesheet", "oxy:",
    )):
        _add(
            cases,
            category="protected_literals",
            ident=f"protect-glued-{n}",
            source=topic(f"<p>Call {literal} here.</p>"),
            hyp=topic(f"<p>Appelez {literal} ici.</p>"),
            expect_gate=True,
            expect_reward=1.0,
        )
        _add(
            cases,
            category="protected_literals",
            ident=f"protect-spaced-{n}",
            source=topic(f"<p>Call {literal} here.</p>"),
            hyp=topic(f"<p>Appelez {literal.replace(':', NBSP + ':', 1)} ici.</p>"),
            expect_gate=False,
        )
    _add(cases, category="protected_literals", ident="protect-time",
         source=topic("<p>Meet at 12:30 sharp.</p>"),
         hyp=topic("<p>Rendez-vous à 12:30 précises.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="protected_literals", ident="protect-ratio",
         source=topic("<p>Use 16:9 video.</p>"),
         hyp=topic("<p>Utilisez la vidéo 16:9.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="protected_literals", ident="protect-url",
         source=topic("<p>See http://example.com/a:b now: go.</p>"),
         hyp=topic(f"<p>Voir http://example.com/a:b maintenant{NBSP}: allez.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="protected_literals", ident="protect-file",
         source=topic("<p>Edit topic.dita then save.</p>"),
         hyp=topic("<p>Éditez topic.dita puis enregistrez.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="protected_literals", ident="protect-css",
         source=topic("<p>Use :before in the rule.</p>"),
         hyp=topic("<p>Utilisez :before dans la règle.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="protected_literals", ident="protect-title-i",
         source=topic("<p>See the note: details.</p>", title="xs:override"),
         hyp=topic(f"<p>Voir la note{NBSP}: détails.</p>", title="xs:override"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="protected_literals", ident="protect-codeph",
         source=topic("<p>Keep <codeph>a:b</codeph> and the note: end.</p>"),
         hyp=topic(f"<p>Gardez <codeph>a:b</codeph> et la note{NBSP}: fin.</p>"),
         expect_gate=True, expect_reward=1.0)

    _add(cases, category="adversarial", ident="adv-codeph-hides-colon",
         source=topic("<p>See the note: details.</p>"),
         hyp=topic("<p>Voir la <codeph>note:</codeph> détails.</p>"),
         expect_gate=False, reward_lt_id="neg-:-space-0")
    _add(cases, category="adversarial", ident="adv-delete-prose",
         source=topic("<p>See the note: details stay in the file.</p>"),
         hyp=topic("<p>Voir.</p>"),
         expect_gate=True, expect_reward=0.0, reward_le_id="neg-:-space-0")
    _add(cases, category="adversarial", ident="adv-space-xs",
         source=topic("<p>The <i>xs:override</i> component: done.</p>"),
         hyp=topic(f"<p>Le composant <i>xs{NBSP}:override</i> terminé{NBSP}: oui.</p>"),
         expect_gate=False, reward_lt_id="adv-xs-glued-with-mark")
    _add(cases, category="adversarial", ident="adv-xs-glued-with-mark",
         source=topic("<p>The <i>xs:override</i> component: done.</p>"),
         hyp=topic(f"<p>Le composant <i>xs:override</i> terminé{NBSP}: oui.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="adversarial", ident="adv-extra-nbsp-same-reward",
         source=topic("<p>See the note: details.</p>"),
         hyp=topic(f"<p>Voir{NBSP}la note{NBSP}: détails.</p>"),
         expect_gate=True, expect_reward=1.0, reward_le_id="pos-:-nbsp-0")

    _add(cases, category="ambiguous", ident="amb-nnbsp",
         source=topic("<p>See the note: details.</p>"),
         hyp=topic(f"<p>Voir la note{NNBSP}: détails.</p>"),
         expect_gate=True, expect_reward=0.0, expect_ambiguous=True)
    _add(cases, category="ambiguous", ident="amb-ratio-spaced",
         source=topic("<p>Use 16 : 9 video.</p>"),
         hyp=topic("<p>Utilisez 16 : 9 vidéo.</p>"),
         expect_gate=True, expect_ambiguous=True)
    for n in range(4):
        _add(
            cases,
            category="ambiguous",
            ident=f"amb-nnbsp-{n}",
            source=topic(f"<p>See the note: details {n}.</p>"),
            hyp=topic(f"<p>Voir la note{NNBSP}: détails {n}.</p>"),
            expect_gate=True,
            expect_ambiguous=True,
        )

    _add(cases, category="nfkc", ident="nfkc-nbsp-kept",
         source=topic("<p>See the ﬁle: now.</p>"),
         hyp=topic(f"<p>Voir le ﬁchier{NBSP}: maintenant.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="nfkc", ident="nfkc-fullwidth-not-colon",
         source=topic("<p>See the note: details.</p>"),
         hyp=topic("<p>Voir la note\uFF1A détails.</p>"),
         expect_gate=True, expect_reward=0.0)

    _add(cases, category="guillemets", ident="guill-ok",
         source=topic('<p>See "W3C" now.</p>'),
         hyp=topic(f"<p>Voir «{NBSP}W3C{NBSP}» maintenant.</p>"),
         expect_gate=True, expect_reward=1.0)
    _add(cases, category="guillemets", ident="guill-bad-space",
         source=topic('<p>See "W3C" now.</p>'),
         hyp=topic("<p>Voir « W3C » maintenant.</p>"),
         expect_gate=True, expect_reward=0.0)
    _add(cases, category="guillemets", ident="guill-english",
         source=topic('<p>See "W3C" now.</p>'),
         hyp=topic('<p>Voir "W3C" maintenant.</p>'),
         expect_gate=True, expect_reward=0.0)
    _add(cases, category="guillemets", ident="guill-split-inline",
         source=topic('<p>See "<b>W3C</b>" now.</p>'),
         hyp=topic(f"<p>Voir «<b>{NBSP}W3C{NBSP}</b>» maintenant.</p>"),
         expect_gate=True, expect_reward=1.0)
    for n in range(4):
        _add(
            cases,
            category="guillemets",
            ident=f"guill-ok-{n}",
            source=topic(f'<p>See "term {n}" now.</p>'),
            hyp=topic(f"<p>Voir «{NBSP}terme {n}{NBSP}» maintenant.</p>"),
            expect_gate=True,
            expect_reward=1.0,
        )

    _add(cases, category="sibling_swaps", ident="attr-order",
         source=topic('<p id="p1" outputclass="note">See the note.</p>'),
         hyp=topic('<p outputclass="note" id="p1">Voir la note.</p>'),
         expect_gate=False)

    return cases


if __name__ == "__main__":
    print(len(build_cases()))
