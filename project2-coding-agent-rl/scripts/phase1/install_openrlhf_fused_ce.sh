#!/usr/bin/env bash
# Reproducibly install the project-local fused CE extension into OpenRLHF 0.10.4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${1:-$ROOT/.venvs/phase1-openrlhf}"
PY="$VENV/bin/python"
PATCH="$ROOT/patches/openrlhf-0.10.4-fused-ce.patch"
FUSED_SRC="$ROOT/scripts/phase1/fused_ce.py"

[ -x "$PY" ] || { echo "missing Python: $PY" >&2; exit 1; }

VERSION="$($PY -c 'import importlib.metadata as m; print(m.version("openrlhf"))')"
[ "$VERSION" = "0.10.4" ] || {
  echo "unsupported OpenRLHF version: $VERSION (expected 0.10.4)" >&2
  exit 1
}

SITE="$($PY -c 'import site; print(site.getsitepackages()[0])')"
ACTOR="$SITE/openrlhf/models/actor.py"
FUSED_DST="$SITE/openrlhf/models/fused_ce.py"

if ! rg -q 'OPENRLHF_FUSED_CE_ACTIVE' "$ACTOR"; then
  patch --directory="$SITE" --strip=1 --forward < "$PATCH"
fi
install -m 0644 "$FUSED_SRC" "$FUSED_DST"

$PY -c \
  'from openrlhf.models.actor import Actor; import inspect; assert "OPENRLHF_FUSED_CE_ACTIVE" in inspect.getsource(Actor.forward)'
$PY -c 'from openrlhf.models.fused_ce import _chunk_size; assert _chunk_size() > 0'
echo "OPENRLHF_FUSED_CE_INSTALL_OK version=$VERSION actor=$ACTOR"
