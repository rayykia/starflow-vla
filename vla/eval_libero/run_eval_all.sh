#!/usr/bin/env bash
# =============================================================================
# STARFlow-VLA LIBERO evaluation: many sharded clients against one batching server.
#
# Adapted from SimVLANF/evaluation/libero/run_eval_all.sh. Instead of one client
# per suite, each suite is split into NUM_PARALLEL episode-level shards that all
# talk to the same server, which fuses their requests into one sampling pass
# (AR sampling costs the same for 1 or 16 requests). Run in the `libero` env.
#
#   bash vla/eval_libero/run_eval_all.sh [port] [num_trials] [output_dir] [gpus] [num_parallel]
#   bash vla/eval_libero/run_eval_all.sh 8000 50 eval_results/starflow_vla "2 3 4" 16
#
#   SUITES="libero_10 libero_goal"   which suites (default: libero_10 -- the training subset)
#   PARALLEL_SUITES=1                launch all suites at once instead of one after another
#   NO_VIDEO=1                       skip rollout mp4s
#   PYTHON=/path/to/libero/python    client interpreter (default: python from PATH)
#   EXTRA_ARGS="--replan_steps 4"    forwarded to libero_client.py
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT=${1:-8000}
NUM_TRIALS=${2:-50}
OUTPUT_DIR=${3:-"eval_results/starflow_vla_${PORT}"}
GPUS=${4:-"0"}
NUM_PARALLEL=${5:-10}
SUITES=${SUITES:-"libero_10"}
PYTHON=${PYTHON:-python}

if ! "$PYTHON" -c "import libero.libero, openpi_client" 2>/dev/null; then
    echo "ERROR: '$PYTHON' cannot import libero / openpi_client. Run: conda activate libero"
    exit 1
fi

read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
mkdir -p "$OUTPUT_DIR/logs"

VIDEO_FLAG=""
[ "${NO_VIDEO:-0}" = "1" ] && VIDEO_FLAG="--no_video"

echo "Starting STARFlow-VLA LIBERO evaluation..."
echo "   Server Port: $PORT"
echo "   Suites: $SUITES"
echo "   Num Trials: $NUM_TRIALS"
echo "   Output Dir: $OUTPUT_DIR"
echo "   Client GPUs (rendering): $GPUS"
echo "   Shards per suite: $NUM_PARALLEL"
echo ""

PIDS=()
launch_suite() {
    local suite=$1
    for ((k=0; k<NUM_PARALLEL; k++)); do
        local gpu=${GPU_ARRAY[$((k % NUM_GPUS))]}
        local logfile="${OUTPUT_DIR}/logs/${suite}_shard${k}of${NUM_PARALLEL}.txt"
        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" -u "${SCRIPT_DIR}/libero_client.py" \
            --host 127.0.0.1 \
            --port "$PORT" \
            --task_suite "$suite" \
            --num_trials "$NUM_TRIALS" \
            --shard "${k}/${NUM_PARALLEL}" \
            --video_out "$OUTPUT_DIR" \
            $VIDEO_FLAG ${EXTRA_ARGS:-} > "$logfile" 2>&1 &
        local pid=$!
        echo "   [PID $pid] $suite shard $k/$NUM_PARALLEL (GPU $gpu) -> $logfile"
        PIDS+=($pid)
    done
}

for suite in $SUITES; do
    launch_suite "$suite"
    if [ "${PARALLEL_SUITES:-0}" != "1" ]; then
        echo "   Waiting for $suite..."
        wait "${PIDS[@]}"
        PIDS=()
        "$PYTHON" "${SCRIPT_DIR}/summarize_results.py" "$OUTPUT_DIR" | tail -n 20
    fi
done
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "   Waiting for all suites..."
    wait "${PIDS[@]}"
fi

echo ""
echo "All evaluations completed!"
"$PYTHON" "${SCRIPT_DIR}/summarize_results.py" "$OUTPUT_DIR"
