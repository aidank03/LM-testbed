#!/bin/zsh
# Use the existing optional PyTorch environment and current source checkout.
FACTOR_LAB_ROOT="${0:A:h}"
cd "$FACTOR_LAB_ROOT" || exit 1
if [[ ! -x "$FACTOR_LAB_ROOT/.venv-local/bin/python" ]]; then
  echo "The optional PyTorch environment is missing. See docs/PYTORCH_LAB.md."
  exit 1
fi
export PYTHONPATH="$FACTOR_LAB_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec "$FACTOR_LAB_ROOT/.venv-local/bin/python" -m factor.torch_lab --workspace "$FACTOR_LAB_ROOT" "$@"
