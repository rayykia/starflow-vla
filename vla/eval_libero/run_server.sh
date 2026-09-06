#!/usr/bin/env bash
# Start the STARFlow-VLA policy server (run in the `nfvla` env, from anywhere).
#
#   bash vla/eval_libero/run_server.sh                              # defaults below
#   GPU=2 PORT=8000 CKPT=logs/libero_model_vla_1024_6_h8.pth bash vla/eval_libero/run_server.sh
#   bash vla/eval_libero/run_server.sh --cfg 0 --denoise 0          # extra flags -> serve_policy.py
#
# Unknown --flags are forwarded as config overrides (e.g. --action_noise_std 0.05).
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
PORT=${PORT:-8000}
CKPT=${CKPT:-logs/libero_model_vla_1024_6_h8.pth}
CONFIG=${CONFIG:-configs/starflow_vla_libero_128.yaml}
PYTHON=${PYTHON:-python}

export PYTHONPATH=          # a stray ROS install on PYTHONPATH shadows repo modules
export CUDA_VISIBLE_DEVICES=$GPU
export TOKENIZERS_PARALLELISM=false

echo "Serving $CKPT on GPU $GPU, port $PORT"
exec "$PYTHON" -u vla/eval_libero/serve_policy.py \
  --model_config_path "$CONFIG" \
  --checkpoint_path "$CKPT" \
  --port "$PORT" \
  "$@"
