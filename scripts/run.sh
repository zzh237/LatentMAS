#!/bin/bash

source /home/ericjiang0318/miniconda3/etc/profile.d/conda.sh
conda activate py310
export AI_STUDIO_TOKEN="c3a5883e15983ab2dc8facefecc5bd9"

# Get script directory and navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=4,5
# LatentMAS run script

# python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --latent_steps 20 --prompt sequential --max_samples 100
EXPERIMENT_NAME="latent_mas_gsm8k_qwen3_14b"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_DIR="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}"
LOG_FILE="$LOG_DIR/${TIMESTAMP}_${EXPERIMENT_NAME}.log"
mkdir -p "$LOG_DIR"


nohup python3 run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --latent_steps 20 \
    --prompt sequential \
    --max_samples 100 "${@:1}" > "$LOG_FILE" 2>&1 &
PID=$!

echo "To monitor progress: tail -f $LOG_FILE"
echo "To stop training: kill $PID"
echo $PID > "$LOG_DIR/run.pid"
