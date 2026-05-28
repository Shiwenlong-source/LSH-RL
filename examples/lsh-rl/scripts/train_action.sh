#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck disable=SC1091
source "$DIR/configs/action.env"

echo "=========================================="
echo "Training Configuration:"
echo "  Phase: Action-Only"
echo "  Adapter: $AGL_ADAPTER_PATH"
echo "  Resume Strategy: $AGL_RESUME_STRATEGY"
echo "  NO_RESUME: $AGL_NO_RESUME"
echo "  Train Data: $AGL_TRAIN_DATA_DIR"
echo "  Batch Size: $AGL_TRAIN_BATCH_SIZE"
echo "  Num Batches: $AGL_NUM_BATCHES"
echo "  Sampling: $AGL_LOW_RATIO:$AGL_MEDIUM_RATIO:$AGL_HIGH_RATIO"
echo "=========================================="

# Run the shared training script.
exec "$DIR/run_train.sh" "$@"
