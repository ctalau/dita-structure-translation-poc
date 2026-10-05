# Brief for reference translators (English → French, DITA XML)

You produce French **reference translations** of fragments or whole topics of the
Oxygen XML Editor user guide. They are used to train and score a small model,
so quality and exactness matter more than speed. Translate yourself, carefully.
Do not use machine-translation tools or scripts that generate French text.

## Markup: copy it exactly

- Same elements, same order, same attributes, same attribute values. Never add,
  drop, merge or reorder elements. Never translate attribute values (`id`,
  `href`, `keyref`, `outputclass`, ...).
- Whole topics: copy the `<?xml ...?>` line and the `<!DOCTYPE ...>` exactly.
- Translate text content, including `<title>`, `<xref>` text, `<uicontrol>`
  and `<menucascade>` labels, `<term>`, `<indexterm>` text, `<b>`, `<i>`.
- Copy **unchanged** the text inside: `codeph codeblock filepath xmlelement
  xmlatt userinput systemoutput cmdname apiname parmname varname option
  keyword kwd shortcut tm msgph synph`.
- Line breaks and indentation inside text may be reflowed freely.
- Escape `<`, `&` in text as in the source (`&lt;`, `&amp;`).

## French conventions (these are checked by a script)

- Put a no-break space before `:` `;` `!` `?` when used as punctuation, and
  inside guillemets. Write it as the entity `&#160;`. Examples:
  `Remarque&#160;:`, `Voulez-vous continuer&#160;?`, `«&#160;texte&#160;»`.
  Not for `:` inside URLs, times or code.
- Quotations and quoted words in prose use « » (with `&#160;` inside), not "…".
- Titles in sentence case: capitalise only the first word and proper nouns
  (product names such as Oxygen XML Editor, Web Author, Content Fusion, DITA,
  WebHelp, Git). E.g. `Editing Tables in Author Mode` → `Modification des
  tableaux en mode Auteur`.
- Terminology, standard in French software documentation:
  save → *enregistrer* (not *sauver*); library → *bibliothèque* (not
  *librairie*); support (a feature) → *prendre en charge* (not *supporter*);
  edit (verb) → *modifier* (not *éditer*; *éditeur* for "editor" is fine);
  dialog/dialog box → *boîte de dialogue* (not *le dialogue*); click X →
  *cliquer sur* X; press X → *appuyer sur* X; customize → *personnaliser*;
  initiate → *lancer/démarrer*; consistent → *cohérent*.
- Address the reader with *vous*. Imperatives for instructions
  (`Click OK` → `Cliquez sur OK`).
- Product/mode names: keep product names in English. Editing modes may be
  rendered as *mode Texte*, *mode Grille*, *mode Auteur*.

## Output format

A plain text file. For each source row, a header line `### <id>` followed by
the French XML on the next lines. Every id must appear exactly once.

```
### seg-09788d1e1e
<title>Bonnes pratiques pour les développeurs de plug-ins</title>
### seg-e32240232f
<li id="li_adt_dgk_54b">…</li>
```

## Verify before you finish

Run `python3 -m fr_rl.verify_refs <sources.jsonl> <your.txt>` from the repo
root. It must print `0 with problems`. Fix each reported item by hand.
If a problem is impossible to fix (e.g. the checker misfires on legitimate
text), leave it and explain it precisely in your final message.
