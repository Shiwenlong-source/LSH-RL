#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
: "${LSH_RL_PYTHON:=python}"

# ============================================================
# GPU Selection: Set which GPUs to use
# Format: comma-separated GPU IDs (e.g., "0,1,6,7" or "0,1" for 2 GPUs)
# ============================================================
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${AGL_N_RUNNERS:=4}"
: "${AGL_N_GPUS_PER_NODE:=4}"
: "${AGL_SPLIT_ACTOR_CRITIC_GPUS:=0}"  # 0=shared mode (default), 1=split mode
: "${AGL_ACTOR_ROLLOUT_GPUS_PER_NODE:=1}"  # GPUs for rollout (used in split mode)
: "${AGL_CRITIC_GPUS_PER_NODE:=1}"  # GPUs for critic (used in split mode)
export CUDA_VISIBLE_DEVICES

# Create logs directory if it doesn't exist
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"

# Generate log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "=========================================="
echo "Starting STRATIFIED training..."
echo "Log file: $LOG_FILE"
echo "=========================================="
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "GPU Configuration:"
echo "  - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  - AGL_N_RUNNERS: $AGL_N_RUNNERS"
echo "  - AGL_N_GPUS_PER_NODE: $AGL_N_GPUS_PER_NODE"
echo "  - Split Actor/Critic GPUs: $AGL_SPLIT_ACTOR_CRITIC_GPUS"
echo "=========================================="
echo ""

# Run training and redirect all output to log file
# Use conservative defaults for 2x H100 (80G each) to avoid startup OOM under PPO/GAE.
# Any user-provided "$@" flags still override these defaults because they are appended last.
# vLLM's memory pool is incompatible with PyTorch expandable_segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
  echo "Detected incompatible PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}, unsetting for vLLM."
  unset PYTORCH_CUDA_ALLOC_CONF
fi

# Keep only selected trace types in PPO training data.
# You can override before running, e.g.:
#   AGL_ALLOWED_TRACE_TYPES="action,dialogue,self_belief" ./run_train.sh
: "${AGL_ROLEPLAY_TRAIN_PHASE:=full}"
case "$AGL_ROLEPLAY_TRAIN_PHASE" in
  full)
    : "${AGL_ALLOWED_TRACE_TYPES:=action,dialogue}"
    ;;
  dialogue_only)
    : "${AGL_ALLOWED_TRACE_TYPES:=dialogue}"
    ;;
  action_only)
    : "${AGL_ALLOWED_TRACE_TYPES:=action}"
    ;;
  *)
    echo "Unsupported AGL_ROLEPLAY_TRAIN_PHASE=${AGL_ROLEPLAY_TRAIN_PHASE}" >&2
    exit 1
    ;;
esac
echo "AGL_ROLEPLAY_TRAIN_PHASE=${AGL_ROLEPLAY_TRAIN_PHASE}"
echo "AGL_ALLOWED_TRACE_TYPES=${AGL_ALLOWED_TRACE_TYPES}"
: "${AGL_SLIM_LOGS:=0}"
echo "AGL_SLIM_LOGS=${AGL_SLIM_LOGS}"

# Training logger backends. Example:
#   AGL_TRAIN_LOGGERS="console,tensorboard" ./run_train.sh
: "${AGL_TRAIN_LOGGERS:=console,tensorboard}"
echo "AGL_TRAIN_LOGGERS=${AGL_TRAIN_LOGGERS}"

if [[ -z "${AGL_BASE_MODEL:-}" || "${AGL_BASE_MODEL}" == "xxx" ]]; then
  echo "ERROR: Set AGL_BASE_MODEL to a valid local or Hugging Face model path before training." >&2
  exit 1
fi
if [[ -z "${ROLEPLAY_ENV_BASE_URL:-}" || "${ROLEPLAY_ENV_BASE_URL}" == "xxx" ]]; then
  echo "ERROR: Set ROLEPLAY_ENV_BASE_URL to an OpenAI-compatible evaluator endpoint before training." >&2
  exit 1
fi
if [[ -z "${ROLEPLAY_ENV_MODEL:-}" || "${ROLEPLAY_ENV_MODEL}" == "xxx" ]]; then
  echo "ERROR: Set ROLEPLAY_ENV_MODEL to the evaluator model name before training." >&2
  exit 1
fi

# Stratified sampling settings
: "${AGL_STRATIFIED_TRAINING:=1}"
: "${AGL_TRAIN_DATA_DIR:=train_data}"
: "${AGL_NUM_BATCHES:=200}"
: "${AGL_LOW_RATIO:=2}"
: "${AGL_MEDIUM_RATIO:=1}"
: "${AGL_HIGH_RATIO:=1}"
: "${AGL_TRAIN_BATCH_SIZE:=4}"

echo "=========================================="
echo "Stratified Sampling Configuration"
echo "=========================================="
echo "AGL_STRATIFIED_TRAINING=${AGL_STRATIFIED_TRAINING}"
echo "AGL_TRAIN_DATA_DIR=${AGL_TRAIN_DATA_DIR}"
echo "AGL_NUM_BATCHES=${AGL_NUM_BATCHES}"
echo "AGL_TRAIN_BATCH_SIZE=${AGL_TRAIN_BATCH_SIZE}"
echo "Sampling ratios: low=${AGL_LOW_RATIO}, medium=${AGL_MEDIUM_RATIO}, high=${AGL_HIGH_RATIO}"

# Keep this estimate aligned with stratified_sampler.py:
# - floor(batch_size * normalized_ratio)
# - distribute remainder by largest fractional part among positive-ratio tiers
# - zero-ratio tier can remain zero
read -r EXPECT_LOW EXPECT_MEDIUM EXPECT_HIGH <<EOF
$("$LSH_RL_PYTHON" - "$AGL_TRAIN_BATCH_SIZE" "$AGL_LOW_RATIO" "$AGL_MEDIUM_RATIO" "$AGL_HIGH_RATIO" <<'PY'
import sys

batch_size = int(sys.argv[1])
low = float(sys.argv[2])
medium = float(sys.argv[3])
high = float(sys.argv[4])

ratios = [low, medium, high]
total = sum(ratios)
if total <= 0:
    norm = [0.0, 1.0, 0.0]
else:
    norm = [r / total for r in ratios]

raw = [batch_size * r for r in norm]
counts = [int(x) for x in raw]
remainder = batch_size - sum(counts)

if remainder > 0:
    order = sorted(
        range(3),
        key=lambda idx: ((raw[idx] - counts[idx]), norm[idx]),
        reverse=True,
    )
    for idx in order:
        if remainder == 0:
            break
        if norm[idx] > 0:
            counts[idx] += 1
            remainder -= 1

if sum(counts) == 0 and batch_size > 0:
    counts[1] = batch_size

print(counts[0], counts[1], counts[2])
PY
)
EOF

echo "Expected scenes per batch:"
echo "  - Low difficulty: ${EXPECT_LOW}"
echo "  - Medium difficulty: ${EXPECT_MEDIUM}"
echo "  - High difficulty: ${EXPECT_HIGH}"
echo "Total scenes: $((AGL_NUM_BATCHES * AGL_TRAIN_BATCH_SIZE))"
echo "=========================================="
echo ""

# Convergence-oriented defaults (overridable by user args in "$@").
: "${AGL_TOTAL_EPOCHS:=1}"
: "${AGL_ROLLOUT_GPU_MEMORY_UTIL:=0.35}"
: "${AGL_BASE_MODEL:=}"
if [[ -z "${AGL_ADAPTER_PATH+x}" ]]; then
  AGL_ADAPTER_PATH="xxx"
fi
if [[ "${AGL_ADAPTER_PATH}" == "xxx" ]]; then
  echo "ERROR: Set AGL_ADAPTER_PATH to a LoRA/PEFT adapter path, or set it to an empty string to train from the base model." >&2
  exit 1
fi
: "${AGL_ACTOR_USE_KL_LOSS:=true}"
: "${AGL_ACTOR_KL_LOSS_COEF:=0.01}"
: "${AGL_ACTOR_ENTROPY_COEFF:=0.001}"
: "${AGL_SHORT_TERM_WEIGHT:=0.50}"
: "${AGL_LONG_TERM_WEIGHT:=0.50}"
: "${AGL_CHECKPOINT_POLICY:=inference}"
: "${AGL_RESUME_STRATEGY:=lightweight}"
: "${AGL_MAX_CKPT_TO_KEEP:=1}"
: "${AGL_TEST_FREQ:=4}"
: "${AGL_SAVE_FREQ:=4}"
: "${AGL_MAX_PROMPT_LENGTH:=3000}"
: "${AGL_MAX_RESPONSE_LENGTH:=768}"
if [[ "${AGL_CHECKPOINT_POLICY}" == "inference" ]]; then
  : "${AGL_NO_RESUME:=0}"  # Allow lightweight resume from checkpoint
else
  : "${AGL_NO_RESUME:=0}"  # Allow resume
fi

echo "Training Configuration:"
echo "AGL_TOTAL_EPOCHS=${AGL_TOTAL_EPOCHS}"
echo "AGL_ROLLOUT_GPU_MEMORY_UTIL=${AGL_ROLLOUT_GPU_MEMORY_UTIL}"
echo "AGL_ADAPTER_PATH=${AGL_ADAPTER_PATH}"
echo "AGL_ACTOR_USE_KL_LOSS=${AGL_ACTOR_USE_KL_LOSS}"
echo "AGL_CHECKPOINT_POLICY=${AGL_CHECKPOINT_POLICY}"
echo "AGL_RESUME_STRATEGY=${AGL_RESUME_STRATEGY}"
echo "=========================================="
echo ""

TRAIN_ADAPTER_ARGS=()
if [[ -n "${AGL_ADAPTER_PATH}" ]]; then
  TRAIN_ADAPTER_ARGS+=(--adapter-path "$AGL_ADAPTER_PATH")
else
  echo "AGL_ADAPTER_PATH is empty: training without adapter warm start."
  # Explicitly pass an empty adapter path so argparse does not fall back to
  # train_stratified.py's DEFAULT_ADAPTER_PATH.
  TRAIN_ADAPTER_ARGS+=(--adapter-path "")
fi

# Validate train_data directory structure
if [[ ! -d "$DIR/$AGL_TRAIN_DATA_DIR" ]]; then
  echo "ERROR: Training data directory not found: $DIR/$AGL_TRAIN_DATA_DIR" >&2
  echo "Expected structure:" >&2
  echo "  $AGL_TRAIN_DATA_DIR/" >&2
  echo "    ├── low/     # Low difficulty scenes" >&2
  echo "    ├── medium/  # Medium difficulty scenes" >&2
  echo "    └── high/    # High difficulty scenes" >&2
  exit 1
fi

for tier in low medium high; do
  if [[ ! -d "$DIR/$AGL_TRAIN_DATA_DIR/$tier" ]]; then
    echo "ERROR: Missing tier directory: $DIR/$AGL_TRAIN_DATA_DIR/$tier" >&2
    exit 1
  fi
  count=$(find "$DIR/$AGL_TRAIN_DATA_DIR/$tier" -name "*.json" | wc -l)
  echo "Found $count scenes in $tier/"
done

echo ""
echo "=========================================="
echo "Starting stratified training..."
echo "=========================================="
echo ""

# Clean up resume metadata only if --no-resume is explicitly set
if [[ "$AGL_NO_RESUME" == "1" ]]; then
  RESUME_META_FILE="$DIR/checkpoints/${AGL_EXPERIMENT_NAME:-roleplay_persona}/resume_meta.json"
  if [[ -f "$RESUME_META_FILE" ]]; then
    echo "=========================================="
    echo "WARNING: AGL_NO_RESUME=1, starting from scratch!"
    echo "Removing resume metadata for fresh start: $RESUME_META_FILE"
    rm -f "$RESUME_META_FILE"
    echo "=========================================="
  fi
else
  echo "=========================================="
  echo "Resume enabled: will continue from latest checkpoint"
  echo "=========================================="
fi

# Run stratified training
AGL_ALLOWED_TRACE_TYPES="$AGL_ALLOWED_TRACE_TYPES" \
AGL_ROLEPLAY_TRAIN_PHASE="$AGL_ROLEPLAY_TRAIN_PHASE" \
AGL_SLIM_LOGS="$AGL_SLIM_LOGS" \
AGL_SHORT_TERM_WEIGHT="$AGL_SHORT_TERM_WEIGHT" \
AGL_LONG_TERM_WEIGHT="$AGL_LONG_TERM_WEIGHT" \
AGL_CHECKPOINT_POLICY="$AGL_CHECKPOINT_POLICY" \
AGL_MAX_CKPT_TO_KEEP="$AGL_MAX_CKPT_TO_KEEP" \
AGL_TEST_FREQ="$AGL_TEST_FREQ" \
AGL_SAVE_FREQ="$AGL_SAVE_FREQ" \
AGL_NO_RESUME="$AGL_NO_RESUME" \
"$DIR/with_project_cache.sh" "$LSH_RL_PYTHON" \
  "$DIR/train_stratified.py" \
  --train-data-dir "$DIR/$AGL_TRAIN_DATA_DIR" \
  --batch-size "$AGL_TRAIN_BATCH_SIZE" \
  --train-batch-size "$AGL_TRAIN_BATCH_SIZE" \
  --num-batches "$AGL_NUM_BATCHES" \
  --low-ratio "$AGL_LOW_RATIO" \
  --medium-ratio "$AGL_MEDIUM_RATIO" \
  --high-ratio "$AGL_HIGH_RATIO" \
  --model "$AGL_BASE_MODEL" \
  --base-model "$AGL_BASE_MODEL" \
  --adv-estimator gae \
  --n-runners "$AGL_N_RUNNERS" \
  --n-gpus-per-node "$AGL_N_GPUS_PER_NODE" \
  $( [[ "$AGL_SPLIT_ACTOR_CRITIC_GPUS" == "0" ]] && echo "--no-split-actor-critic-gpus" || echo "--split-actor-critic-gpus --actor-rollout-gpus-per-node $AGL_ACTOR_ROLLOUT_GPUS_PER_NODE --critic-gpus-per-node $AGL_CRITIC_GPUS_PER_NODE" ) \
  --rollout-gpu-memory-utilization "$AGL_ROLLOUT_GPU_MEMORY_UTIL" \
  --total-epochs "$AGL_TOTAL_EPOCHS" \
  --actor-ppo-micro-batch-size-per-gpu 1 \
  --rollout-log-prob-micro-batch-size-per-gpu 1 \
  --ref-log-prob-micro-batch-size-per-gpu 1 \
  --actor-kl-loss-coef "$AGL_ACTOR_KL_LOSS_COEF" \
  --actor-entropy-coeff "$AGL_ACTOR_ENTROPY_COEFF" \
  --max-prompt-length "$AGL_MAX_PROMPT_LENGTH" \
  --max-response-length "$AGL_MAX_RESPONSE_LENGTH" \
  --lora \
  --lora-rank 16 \
  "${TRAIN_ADAPTER_ARGS[@]}" \
  --evaluator-base-url "$ROLEPLAY_ENV_BASE_URL" \
  --evaluator-api-key "${ROLEPLAY_ENV_API_KEY:-xxx}" \
  --evaluator-model "$ROLEPLAY_ENV_MODEL" \
  $( [[ "$AGL_ACTOR_USE_KL_LOSS" == "true" ]] && echo "--actor-use-kl-loss" || echo "--actor-no-kl-loss" ) \
  "$@" 2>&1 | tee "$LOG_FILE"

# Export reward and training metric curves from the compact step logs.
PLOT_DIR="$LOG_DIR/plots"
IMAGES_DIR="$LOG_DIR/images/${TIMESTAMP}"
"$DIR/with_project_cache.sh" "$LSH_RL_PYTHON" \
  "$DIR/plot_training_metrics.py" \
  --log-file "$LOG_FILE" \
  --out-dir "$PLOT_DIR" \
  --images-dir "$IMAGES_DIR" \
  --smooth-window "${AGL_PLOT_SMOOTH_WINDOW:-1}" || echo "[WARN] metric plot generation failed; training log is still available at $LOG_FILE"

echo ""
echo "=========================================="
echo "Stratified training completed."
echo "Log saved to: $LOG_FILE"
echo "Metric CSV/PNG: $PLOT_DIR"
echo "Per-metric images: $IMAGES_DIR"
echo "=========================================="
