#!/bin/bash
# All LIBERO suites, 100,000 optimizer steps, 4 x B200, 64 samples/GPU.
# Submit from anywhere:
#   mkdir -p /vast/projects/jgu32/lab/ruichend/starflow-vla/logs
#   sbatch /vast/projects/jgu32/lab/ruichend/starflow-vla/scripts/train_vla_libero_all_b200.sbatch
# Requires the training dependencies documented in vla/README.md in the nfvla env.
# SLURM settings follow nfvla/train_context_video.sbatch (50-hour wall limit).
# Checkpoints go to logs/<run name>.
#SBATCH --mail-user=ruichend@engineering.upenn.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=starflow_libero_all_100k
#SBATCH --partition=dgx-b200
#SBATCH --exclude=dgx024
#SBATCH --account=jgu32-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=50:00:00
#SBATCH --chdir=/vast/projects/jgu32/lab/ruichend/starflow-vla
#SBATCH --output=/vast/projects/jgu32/lab/ruichend/starflow-vla/logs/starflow_libero_all_100k_%J.out
#SBATCH --error=/vast/projects/jgu32/lab/ruichend/starflow-vla/logs/starflow_libero_all_100k_%J.err

set -eo pipefail
# Conda CUDA activation hooks reference unset variables; enable nounset afterward.
source /vast/home/r/ruichend/miniconda3/etc/profile.d/conda.sh
conda activate /vast/projects/jgu32/lab/ruichend/conda/envs/nfvla
set -u

export PYTHONPATH=  # Clear the ROS path that can shadow repository modules.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# Keep CUDA_VISIBLE_DEVICES as assigned by SLURM.

CONFIG=configs/starflow_vla_libero_128.yaml
DATA_ROOT=/vast/projects/jgu32/lab/ruichend/LIBERO/libero/datasets
SUBSETS=(libero_spatial libero_object libero_goal libero_90 libero_10)
NORM_STATS=vla/norm_stats/libero_all.json
RUN_NAME=${RUN_NAME:-libero-all-h8-b256-100k-${SLURM_JOB_ID}}
LOGDIR=${LOGDIR:-logs/$RUN_NAME}
mkdir -p "$LOGDIR"

# Check every requested suite before using or generating combined statistics.
for subset in "${SUBSETS[@]}"; do
    if ! compgen -G "$DATA_ROOT/$subset/*.hdf5" > /dev/null; then
        echo "No HDF5 files found in $DATA_ROOT/$subset" >&2
        exit 1
    fi
done
if [[ ! -f "$NORM_STATS" ]]; then
    python vla/compute_norm_stats.py \
        --data_root "$DATA_ROOT" \
        --subsets "${SUBSETS[@]}" \
        --output "$NORM_STATS"
fi
python - "$NORM_STATS" "${SUBSETS[@]}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    stats = json.load(f)
if set(stats.get('metadata', {}).get('subsets', [])) != set(sys.argv[2:]):
    raise SystemExit('Normalization subsets do not match; regenerate vla/norm_stats/libero_all.json')
for key, dim in [('actions', 7), ('state', 8)]:
    for stat in ['mean', 'std']:
        if len(stats.get(key, {}).get(stat, [])) != dim:
            raise SystemExit(f'Missing or invalid {key}.{stat} in normalization stats')
print(f"LIBERO normalization: {stats['metadata']['num_demos']:,} demos, "
      f"{stats['metadata']['num_steps']:,} timesteps")
PY

echo "Run: $RUN_NAME | GPUs: ${CUDA_VISIBLE_DEVICES:-SLURM default}"
echo "100,000 steps | batch/GPU: 64 | global batch: 256 | output: $LOGDIR"
# --batch_size is GLOBAL; --acc 1 gives a physical batch of 256 / 4 = 64 per GPU.
# Call torchrun directly: train_vla_libero.sh otherwise overrides SLURM's GPU mask.
exec torchrun --standalone --nnodes=1 --nproc_per_node=4 vla/train_libero.py \
    --model_config_path "$CONFIG" \
    --libero_root "$DATA_ROOT" \
    --libero_subsets "${SUBSETS[@]}" \
    --norm_stats "$NORM_STATS" \
    --batch_size 256 \
    --acc 1 \
    --max_steps 100000 \
    --epoch_length 0 \
    --fsdp 0 \
    --num_workers 8 \
    --logdir "$LOGDIR" \
    --wandb_name "$RUN_NAME"
