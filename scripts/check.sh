#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
uv sync --locked
uv run python python/generate.py --check
uv run pytest -q
"$HOME/.elan/bin/lake" build
"$HOME/.elan/bin/lake" env lean ProofAudit.lean
