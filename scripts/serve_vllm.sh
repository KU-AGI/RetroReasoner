#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible servers for RetroReasoner models.
#
# Usage:
#   bash scripts/serve_vllm.sh retroreasoner <GPUS> <START_PORT>
#   bash scripts/serve_vllm.sh roundtrip     <GPUS> <START_PORT>
#
# <GPUS> is either a count (e.g. 4 -> physical GPUs 0,1,2,3) or an explicit
# comma-separated list of physical GPU indices (e.g. 1,2,3). Each replica's
# port is START_PORT + its position in the list (0-indexed).
#
# Example (4 GPUs, starting at port 8000):
#   bash scripts/serve_vllm.sh retroreasoner 4 8000
#   # -> RetroReasoner(RL) served at localhost:8000..8003 on GPUs 0,1,2,3
#
# Example (round-trip model on GPUs 1,2,3, avoiding GPU 0 — see GPU 0 caveat
# below):
#   bash scripts/serve_vllm.sh roundtrip 1,2,3 8091
#   # -> served at localhost:8091..8093 on GPUs 1,2,3
#
# GPU 0 caveat: eval_2_save_metrics.py also runs localmapper locally on
# physical GPU 0. If a round-trip replica also lands on GPU 0, the two
# contend for that GPU and template counting can silently undercount without
# any visible error — see the README's "GPU 0 caveat" section. Prefer an
# explicit GPU list that excludes 0 when serving the round-trip model.

set -euo pipefail

MODEL_KIND=${1:?"specify 'retroreasoner' or 'roundtrip'"}
GPUS=${2:-1}
START_PORT=${3:-8000}

case "$MODEL_KIND" in
  retroreasoner) MODEL_ID="KU-AGI/RetroReasoner-RL" ;;
  roundtrip)     MODEL_ID="KU-AGI/RetroReasoner-RoundTrip-8B" ;;
  *) echo "unknown model kind: $MODEL_KIND (expected 'retroreasoner' or 'roundtrip')"; exit 1 ;;
esac

if [[ "$GPUS" == *,* ]]; then
  IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
  GPU_LIST=()
  for ((i = 0; i < GPUS; i++)); do GPU_LIST+=("$i"); done
fi

mkdir -p logs
for i in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$i]}"
  PORT=$((START_PORT + i))
  CUDA_VISIBLE_DEVICES=$GPU \
  setsid nohup vllm serve "$MODEL_ID" \
    --port "$PORT" \
    > "logs/vllm_${MODEL_KIND}_gpu${GPU}.log" 2>&1 < /dev/null &
  echo "Launched $MODEL_ID on GPU $GPU, port $PORT (log: logs/vllm_${MODEL_KIND}_gpu${GPU}.log)"
done
