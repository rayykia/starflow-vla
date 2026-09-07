#!/bin/bash
# Watchdog for SLURM job 8139395 (starflow_libero_all_100k).
# Emits one stdout line per noteworthy event; exits when the job reaches a terminal state.
JOB=8139395
LOGDIR=/vast/projects/jgu32/lab/ruichend/starflow-vla/logs
OUT=$LOGDIR/starflow_libero_all_100k_${JOB}.out
ERR=$LOGDIR/starflow_libero_all_100k_${JOB}.err

ERR_OFF=$(stat -c %s "$ERR" 2>/dev/null || echo 0)
OUT_OFF=$(stat -c %s "$OUT" 2>/dev/null || echo 0)

cur_step() {
  tail -n 100 "$OUT" 2>/dev/null | grep -oE '[0-9,]+/100,000 steps' | tail -1 | cut -d/ -f1 | tr -d ,
}

LAST_STEP=$(cur_step); LAST_STEP=${LAST_STEP:-0}
LAST_PROGRESS_TS=$(date +%s)
NEXT_MILESTONE=$(( (LAST_STEP / 10000 + 1) * 10000 ))
STALL_REPORTED=0

while true; do
  # --- crash / failure signatures in stderr ---
  if [ -f "$ERR" ]; then
    SZ=$(stat -c %s "$ERR" 2>/dev/null || echo "$ERR_OFF")
    if [ "$SZ" -gt "$ERR_OFF" ]; then
      tail -c +$((ERR_OFF+1)) "$ERR" 2>/dev/null \
        | grep -E -i 'traceback|out of memory|runtimeerror|valueerror|keyerror|assertionerror|cuda error|nccl error|segmentation fault|core dumped|slurmstepd:|srun: error|DUE TO TIME LIMIT|CANCELLED' \
        | head -8 | sed 's/^/[job-err] /'
      ERR_OFF=$SZ
    fi
  fi

  # --- divergence / crash signatures in stdout ---
  if [ -f "$OUT" ]; then
    SZ=$(stat -c %s "$OUT" 2>/dev/null || echo "$OUT_OFF")
    if [ "$SZ" -gt "$OUT_OFF" ]; then
      tail -c +$((OUT_OFF+1)) "$OUT" 2>/dev/null \
        | grep -E -i 'loss: (-?nan|-?inf)|traceback|error|\[preview\] epoch|^Epoch [0-9]+  Time' \
        | head -8 | sed 's/^/[job-out] /'
      OUT_OFF=$SZ
    fi
  fi

  # --- progress milestones + stall detection ---
  STEP=$(cur_step); STEP=${STEP:-$LAST_STEP}
  if [ "$STEP" -gt "$LAST_STEP" ] 2>/dev/null; then
    LAST_PROGRESS_TS=$(date +%s); STALL_REPORTED=0
    if [ "$STEP" -ge "$NEXT_MILESTONE" ]; then
      RATE=$(tail -n 5 "$OUT" | grep -oE '\([0-9.]+ it/s\)' | tail -1)
      REMAIN=$(( (100000 - STEP) ))
      echo "[progress] step $STEP/100,000 $RATE - ~$((REMAIN * 100 / 100000))% remaining | $(tail -n 1 "$OUT" | cut -c1-160)"
      NEXT_MILESTONE=$(( (STEP / 10000 + 1) * 10000 ))
    fi
    LAST_STEP=$STEP
  else
    NOW=$(date +%s)
    if [ $((NOW - LAST_PROGRESS_TS)) -gt 1200 ] && [ "$STALL_REPORTED" -eq 0 ]; then
      echo "[stall] no new training steps for 20+ min (stuck at step $LAST_STEP); job still in state $(squeue -h -j $JOB -o %T 2>/dev/null)"
      STALL_REPORTED=1
    fi
  fi

  # --- terminal job state ---
  STATE=$(squeue -h -j "$JOB" -o '%T' 2>/dev/null)
  if [ -z "$STATE" ]; then
    FINAL=$(sacct -n -X -j "$JOB" -o State,ExitCode,Elapsed 2>/dev/null | head -1 | tr -s ' ')
    echo "[terminal] job $JOB left the queue at step $LAST_STEP/100,000 -- sacct:$FINAL"
    tail -n 3 "$ERR" 2>/dev/null | sed 's/^/[final-err] /'
    exit 0
  fi

  sleep 120
done
