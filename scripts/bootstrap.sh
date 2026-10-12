#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
git submodule update --init
uv sync --locked
# http/1.1 works around flaky http/2 clones of mathlib; then pull only the
# mathlib cache this project needs, not the whole thing
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.version GIT_CONFIG_VALUE_0=HTTP/1.1 \
MATHLIB_NO_CACHE_ON_UPDATE=1 "$HOME/.elan/bin/lake" update
"$HOME/.elan/bin/lake" exe cache get Mathlib.Tactic \
  Mathlib.Analysis.SpecialFunctions.Exp \
  Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic \
  Mathlib.Analysis.SpecialFunctions.Log.Basic \
  Mathlib.Analysis.SpecialFunctions.Sqrt
