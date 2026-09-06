#!/usr/bin/env bash
# STARFlow-VLA on LIBERO -- 4-GPU DDP training with wandb logging.
#
#   bash scripts/train_vla_libero.sh                    # defaults below
#   GPUS=0,4,5,6 BATCH_SIZE=64 bash scripts/train_vla_libero.sh
#   bash scripts/train_vla_libero.sh --lr 5e-5 --action_loss_weight 2.0   # extra CLI overrides
#
# Any extra argument is forwarded to vla/train_libero.py and wins over the YAML config.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0,4,5,6}
CONFIG=${CONFIG:-configs/starflow_vla_libero_128.yaml}
BATCH_SIZE=${BATCH_SIZE:-64}          # global batch; per-GPU = BATCH_SIZE / #GPUS / acc
NUM_WORKERS=${NUM_WORKERS:-8}
RUN_NAME=${RUN_NAME:-libero10-h8-$(date +%m%d-%H%M)}
LOGDIR=${LOGDIR:-./logs}
PORT=${PORT:-29511}

NGPU=$(awk -F, '{print NF}' <<< "$GPUS")

# The environment has a stray ROS install on PYTHONPATH that shadows repo modules.
export PYTHONPATH=
export CUDA_VISIBLE_DEVICES=$GPUS
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

mkdir -p "$LOGDIR"
echo "Launching on GPUs [$GPUS] ($NGPU procs), global batch $BATCH_SIZE, run '$RUN_NAME'"

exec torchrun --nproc_per_node="$NGPU" --master_port="$PORT" vla/train_libero.py \
  --model_config_path "$CONFIG" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --logdir "$LOGDIR" \
  --wandb_name "$RUN_NAME" \
  "$@"
