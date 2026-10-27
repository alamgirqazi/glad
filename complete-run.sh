#!/bin/bash

set -u

SEEDS=(7 21 43 142 456)      
DATASETS=("RAWFC" "LIAR-RAW")
MODELS=("nomic" "qwen")

MODES="clean contaminated trustrag ragdefender glad_dense glad_lexical glad glad_clean"

SCRIPT="defence/cache-main.py"
LOG_DIR="./run_logs"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/master_${TS}.log"

{
  echo "=== Full run started: $(date) ==="
  echo "Seeds: ${SEEDS[*]} | Modes: $MODES"
  TOTAL=$(( ${#SEEDS[@]} * ${#DATASETS[@]} * ${#MODELS[@]} ))
  COUNT=0
  for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for model in "${MODELS[@]}"; do
        COUNT=$((COUNT + 1))
        tag="${dataset}_${model}_s${seed}"
        echo ""
        echo "--- [$COUNT/$TOTAL] $tag | $(date +%H:%M:%S) ---"
        python "$SCRIPT" \
          --dataset "$dataset" --model "$model" --seed "$seed" \
          --modes $MODES \
          2>&1 | tee "$LOG_DIR/${tag}_${TS}.log" \
          || echo "FAILED: $tag"
      done
    done
  done
  echo ""
  echo "=== Finished: $(date) ==="
} 2>&1 | tee "$MASTER_LOG"