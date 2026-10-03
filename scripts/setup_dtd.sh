#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/schemas/dita13-doctypes"
if [[ ! -f "$DEST/catalog.xml" ]]; then
  git clone --depth 1 https://github.com/dita-community/org.oasis-open.dita.dita13.doctypes.git "$DEST"
fi
MOD="$DEST/doctypes/dtd/technicalContent/dtd/svgDomain.mod"
python3 - << PY
from pathlib import Path
p = Path("$MOD")
t = p.read_text()
t = t.replace(">%svg11-ditadriver;", ">")
p.write_text(t)
print("patched", p)
PY
