#!/usr/bin/env bash
# Parity with .github/workflows/ci.yml — run before pushing to main.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! command -v pixi >/dev/null 2>&1; then
  echo "error: pixi is not on PATH (install from https://pixi.sh)" >&2
  exit 1
fi
exec pixi run --frozen ci
