#!/bin/zsh
set -eu

ROOT="${0:A:h}"
PYTHON="$ROOT/.venv-local/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Factor's local PyTorch environment is missing: $ROOT/.venv-local"
  echo "Open docs/LEARNING_LAB.md for setup instructions."
  read "?Press Return to close."
  exit 1
fi

export PYTHONPATH="$ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -m factor.learn.server --workspace "$ROOT" --open "$@"
