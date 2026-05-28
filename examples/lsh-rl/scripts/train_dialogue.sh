#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
source "$DIR/configs/dialogue.env"

# ============================================================
# Allow command-line override of resume strategy
# Usage: AGL_RESUME_STRATEGY_OVERRIDE=exact bash scripts/train_dialogue.sh
# ============================================================
if [ -n "${AGL_RESUME_STRATEGY_OVERRIDE:-}" ]; then
  export AGL_RESUME_STRATEGY="$AGL_RESUME_STRATEGY_OVERRIDE"
fi
if [ -n "${AGL_NO_RESUME_OVERRIDE:-}" ]; then
  export AGL_NO_RESUME="$AGL_NO_RESUME_OVERRIDE"
fi

# Run the shared stratified training script.
exec "$DIR/run_train.sh" "$@"
