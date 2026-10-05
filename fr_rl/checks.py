"""Deterministic checks for English-to-French DITA translation.

Everything here is pure Python plus lxml and sacrebleu, so the same code
scores rollouts on the pod and evaluation outputs on a laptop.

A hypothesis is compared with its English source (and, when given, a French
reference). The checks:

  xml_ok        the output parses as XML (after &nbsp; is expanded)
  skeleton_ok   same elements, same order, same attributes and values
  protected     code-like elements whose text must not change
  typo          French typography: missing (narrow) no-break space before
                : ; ! ? and inside guillemets; English quotes in prose
  calque        a short list of well-known anglicisms in software French
  titlecase     English Title Case carried into a French <title>
  untranslated  text left in English (English function-word ratio)
  chrf          chrF++ of the prose against the French reference
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from lxml import etree

# Elements whose text is code, a path, a key name, a product name...
# Their text must come back byte-identical (after whitespace normalisation).
PROTECTED = {
    "codeph", "codeblock", "filepath", "xmlelement", "xmlatt", "xmlpi",
    "xmlnsname", "textentity", "parameterentity", "numcharref",
    "userinput", "systemoutput", "cmdname", "apiname", "parmname",
    "varname", "option", "keyword", "kwd", "shortcut", "tm", "msgph",
    "msgblock", "synph", "screen",
}
# UI labels: translatable, but excluded from language checks because the
# right French label depends on the localised product UI we cannot see.
UI = {"uicontrol", "wintitle", "menucascade"}
PLACEHOLDER = "§"  # what a protected/UI element looks like in prose

NBSP_ENT = re.compile(r"&nbsp;|&#0*160;|&#x0*a0;|&#8239;|&#x202f;", re.I)
FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n|\n?```\s*$")
THINK = re.compile(r"<think>.*?</think>", re.S)
NBSPS = "  "


def clean_output(text: str) -> str:
    text = THINK.sub("", text or "")
    text = FENCE.sub("", text.strip())
    return text.strip()


def _expand(xml: str) -> str:
    def rep(m):
        s = m.group(0).lower()
        return " " if "8239" in s or "202f" in s else " "
    return NBSP_ENT.sub(rep, xml)


_PARSER = etree.XMLParser(resolve_entities=False, load_dtd=False,
                          no_network=True, remove_comments=True,
                          remove_pis=True, recover=False)


def split_prolog(doc: str) -> tuple[str, str]:
    """Split '<?xml ...?><!DOCTYPE ...>' header from the root element text."""
    m = re.search(r"<(?![?!])", doc)
    if not m:
        return doc, ""
    return doc[:m.start()], doc[m.start():]


def parse(xml: str):
    """Parse a fragment or a whole topic. Returns root element or None."""
    _, body = split_prolog(_expand(xml.strip()))
    if not body:
        return None
    try:
        return etree.fromstring(body.encode("utf-8"), _PARSER)
    except etree.XMLSyntaxError:
        return None


def _tag(e) -> str | None:
    return e.tag if isinstance(e.tag, str) else None


def skeleton(root) -> list:
    return [(e.tag, tuple(sorted(e.attrib.items())))
            for e in root.iter() if _tag(e)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace(" ", " ").replace(" ", " ")).strip()


def protected_texts(root) -> list[str]:
    out = []
    for e in root.iter():
        if _tag(e) in PROTECTED and not any(_tag(a) in PROTECTED for a in e.iterancestors()):
            out.append(_norm("".join(e.itertext())))
    return out


def prose(root, mask_ui: bool) -> str:
    """Text of the element with protected (and optionally UI) elements masked.

    indexterm text is skipped: it is metadata, not running text.
    """
    parts: list[str] = []

    def walk(e):
        t = _tag(e)
        if t in PROTECTED or (mask_ui and t in UI):
            parts.append(PLACEHOLDER)
        elif t in ("indexterm", "image", "data"):
            pass
        else:
            if e.text:
                parts.append(e.text)
            for c in e:
                walk(c)
                if c.tail:
                    parts.append(c.tail)
        if t in BLOCKS:  # keep sentences from fusing across block elements
            parts.append("\n")

    walk(root)
    return re.sub(r"[ \t\r]*\n[\s]*", "\n", "".join(parts)).strip()


BLOCKS = {
    "title", "shortdesc", "p", "li", "dt", "dd", "entry", "note", "cmd",
    "info", "stepresult", "linktext", "context", "result", "postreq",
    "prereq", "section", "desc", "glossterm", "glossdef", "cause",
    "condition", "remedy", "responsibleParty", "sli", "stentry", "lq",
    "fig", "steps", "ul", "ol", "dl", "table", "codeblock", "searchtitle",
    "navtitle", "example", "stepxmp", "choice", "tutorialinfo",
}

# ---------------------------------------------------------------- typography
# A French "high" punctuation mark needs a no-break space before it. We only
# flag marks used as punctuation: followed by whitespace, end, or a closing
# bracket/quote, so URLs, times (10:30) and key:value strings are not flagged.
_HIGH = re.compile(r"(?P<pre>.?)(?P<sp>[   ]?)(?P<mark>[:;!?])(?=\s|$|[)\]»§])")
_OPEN_G = re.compile(r"«(?![  ])")
_CLOSE_G = re.compile(r"(?<![  ])»")
_EN_QUOTES = re.compile(r"\"[^\"\n]{1,80}\"|“[^”\n]{1,80}”")


def typography(text: str) -> dict:
    bad_space = 0
    for m in _HIGH.finditer(text):
        pre, sp, mark = m.group("pre"), m.group("sp"), m.group("mark")
        if sp and sp in NBSPS:
            continue
        if mark in "!?" and pre and pre in "!?":  # '?!' cluster: one space suffices
            continue
        if pre == "" and sp == "":  # mark at start of text node
            continue
        if pre.isdigit() and mark == ":":
            continue
        bad_space += 1
    g = len(_OPEN_G.findall(text)) + len(_CLOSE_G.findall(text))
    q = len(_EN_QUOTES.findall(text))
    return {"nbsp_missing": bad_space, "guillemet_space": g, "en_quotes": q,
            "total": bad_space + g + q}


# ------------------------------------------------------------------- calques
# Each entry: (name, regex, preferred French). Matched case-insensitively on
# prose with protected and UI text masked. Kept short on purpose: every item
# is a standard rule in French software localisation style guides
# (Microsoft French Style Guide, OQLF Banque de dépannage linguistique).
CALQUES = [
    ("librairie=library", r"\blibrairies?\b", "bibliothèque"),
    ("supporter=support", r"\bsupport(?:e|es|ent|er|é|ée|és|ées|ait|aient|ant)\b", "prendre en charge"),
    ("éditer=edit", r"\bédit(?:er|e|es|ez|ons|ent|é|ée|és|ées|ant|ait|aient)\b", "modifier"),
    ("sauver=save", r"\bsauv(?:er|e|es|ez|é|ée|és|ées|ant)\b", "enregistrer"),
    ("cliquer X=click X", r"\bcliqu(?:ez|er|e|ant)\s+(?:le|la|les|l['’]|un|une|deux|droit sur sur)\b", "cliquer sur"),
    ("appuyer X=press X", r"\bappuy(?:ez|er|ant)\s+(?:le|la|les|l['’])\b", "appuyer sur"),
    ("le dialogue=dialog", r"\b(?:le|du|au|ce|un|les|des|ces)\s+dialogues?\b", "boîte de dialogue"),
    ("customiser=customize", r"\bcustomis\w*|\bcustomiz\w*", "personnaliser"),
    ("initier=initiate", r"\biniti(?:er|ez|e|é|ée|és|ées|ant)\b", "lancer, démarrer"),
    ("consistant=consistent", r"\bconsistan(?:t|te|ts|tes|ce)\b", "cohérent"),
    ("processer=process", r"\bprocess(?:er|ez|é|ée|és|ées|ant)\b", "traiter"),
    ("digital=digital", r"\bdigita(?:l|le|ux|les)\b", "numérique"),
    ("opportunité=opportunity", r"\bopportunités?\b", "occasion"),
    ("adresser=address (an issue)", r"\badress(?:er|ez|e|é|ée|és|ées)\s+(?:le|la|les|ce|ces|un|une)\s+(?:problème|question|bogue)", "résoudre"),
]
_CALQUE_RE = [(n, re.compile(rx, re.I), fix) for n, rx, fix in CALQUES]


def calques(text: str) -> dict[str, int]:
    hits = {}
    for name, rx, _ in _CALQUE_RE:
        n = len(rx.findall(text))
        if n:
            hits[name] = n
    return hits


# ---------------------------------------------------------------- title case
# Capitalised names that French Oxygen docs legitimately keep capitalised.
TITLE_OK = {"Oxygen", "XML", "Auteur", "Author", "Texte", "Grille", "Web",
            "DITA", "Eclipse", "Java", "Windows", "Mac", "Linux", "Git"}
_WORD = re.compile(r"[^\W\d_][\w'’\-]*", re.U)


def titlecase(src_title: str, hyp_title: str) -> int:
    src_words = set(_WORD.findall(src_title))
    words = _WORD.findall(hyp_title)
    bad = 0
    for i, w in enumerate(words):
        if i == 0 or len(w) < 2 or not w[0].isupper() or w.isupper():
            continue
        base = re.split(r"['’]", w)[-1] or w
        if w in src_words or base in src_words or w in TITLE_OK or base in TITLE_OK:
            continue
        if any(c.isupper() for c in w[1:]):  # camelCase product names
            continue
        bad += 1
    return bad


# -------------------------------------------------------------- untranslated
_EN_FUNC = set("""the and of to is are this that with for you your from which
when can will be by an it its or not if as on at was were have has into these
those such then there their they them using use used also only must should
would could does do how what where while""".split())
_FR_FUNC = set("""le la les de des du et est sont ce cette que qui avec pour
vous votre vos dans un une par sur pas ne se si au aux il elle ils on lors
peut peuvent être été ou plus""".split())


def english_ratio(text: str) -> float:
    toks = [t.lower() for t in _WORD.findall(text)]
    func = [t for t in toks if t in _EN_FUNC or t in _FR_FUNC]
    if len(func) < 3:
        return 0.0
    return sum(t in _EN_FUNC for t in func) / len(func)


# --------------------------------------------------------------------- chrF
_CHRF = None


def chrf(hyp: str, ref: str) -> float:
    global _CHRF
    if _CHRF is None:
        from sacrebleu.metrics import CHRF
        _CHRF = CHRF(word_order=2)
    return _CHRF.sentence_score(_norm(hyp), [_norm(ref)]).score


# ------------------------------------------------------------------ all-in-1
@dataclass
class Report:
    xml_ok: bool = False
    skeleton_ok: bool = False
    header_ok: bool = True
    protected_changed: int = 0
    protected_total: int = 0
    typo: dict = field(default_factory=dict)
    calques: dict = field(default_factory=dict)
    titlecase: int = 0
    titles: int = 0
    english_ratio: float = 0.0
    untranslated: bool = False
    len_ratio: float = 0.0
    chrf: float | None = None

    @property
    def structure_ok(self) -> bool:
        return self.xml_ok and self.skeleton_ok and self.header_ok and self.protected_changed == 0

    @property
    def n_typo(self) -> int:
        return int(self.typo.get("total", 0))

    @property
    def n_calque(self) -> int:
        return sum(self.calques.values())

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(structure_ok=self.structure_ok, n_typo=self.n_typo,
                 n_calque=self.n_calque)
        return d


def check(src: str, hyp: str, ref: str | None = None) -> Report:
    r = Report()
    hyp = clean_output(hyp)
    s_root = parse(src)
    h_root = parse(hyp)
    if s_root is None:
        raise ValueError("source does not parse")
    if h_root is None:
        return r
    r.xml_ok = True
    src_head, _ = split_prolog(src.strip())
    hyp_head, _ = split_prolog(_expand(hyp))
    if src_head.strip():
        r.header_ok = _norm(src_head) == _norm(hyp_head)
    r.skeleton_ok = skeleton(s_root) == skeleton(h_root)

    sp, hp = protected_texts(s_root), protected_texts(h_root)
    r.protected_total = len(sp)
    if len(sp) == len(hp):
        r.protected_changed = sum(a != b for a, b in zip(sp, hp))
    else:
        r.protected_changed = max(len(sp), len(hp))

    h_text = prose(h_root, mask_ui=False)
    h_lang = prose(h_root, mask_ui=True)
    s_text = prose(s_root, mask_ui=False)
    r.typo = typography(h_text)
    r.calques = calques(h_lang)

    s_titles = [e for e in s_root.iter() if _tag(e) == "title"]
    h_titles = [e for e in h_root.iter() if _tag(e) == "title"]
    r.titles = len(s_titles)
    if len(s_titles) == len(h_titles):
        r.titlecase = sum(titlecase(prose(a, True), prose(b, True))
                          for a, b in zip(s_titles, h_titles))

    r.english_ratio = round(english_ratio(h_lang), 3)
    src_en = english_ratio(prose(s_root, mask_ui=True))
    r.untranslated = src_en > 0.5 and r.english_ratio > 0.35
    r.len_ratio = round(len(_norm(h_text)) / max(1, len(_norm(s_text))), 3)
    if ref is not None:
        ref_root = parse(ref)
        ref_text = prose(ref_root, mask_ui=False) if ref_root is not None else ref
        r.chrf = round(chrf(h_text, ref_text), 2)
    return r


def reward(rep: Report) -> float:
    """Scalar RL reward in [-1, 1]. Structure is a hard gate."""
    if not (rep.xml_ok and rep.skeleton_ok and rep.header_ok):
        return -1.0
    if rep.untranslated:
        return -0.5
    r = (rep.chrf or 0.0) / 100.0
    r -= 0.25 * min(rep.protected_changed, 2)
    r -= 0.08 * min(rep.n_typo, 4)
    r -= 0.15 * min(rep.n_calque, 3)
    r -= 0.10 * min(rep.titlecase, 3)
    if rep.len_ratio < 0.75 or rep.len_ratio > 1.8:  # omission or rambling
        r -= 0.2
    return max(-1.0, min(1.0, r))
