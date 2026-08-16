# STARFlow-VLA (LIBERO world action model)

Image + instruction -> next-H-frames video + H-step action chunk, as one
normalizing flow (see `docs/superpowers/specs/2026-08-15-starflow-vla-design.md`).

## Setup
    pip install h5py pytest imageio imageio-ffmpeg   # inside the starflow env

## Norm stats (once per subset choice)
    python vla/compute_norm_stats.py \
      --data_root /home/rayykia/Projects/LIBERO/libero/datasets \
      --subsets libero_10 --output vla/norm_stats/libero_10.json

## Train
    python vla/train_libero.py --model_config_path configs/starflow_vla_libero_128.yaml
    # multi-GPU: torchrun --nproc_per_node=4 vla/train_libero.py --model_config_path ...

## Sample / policy
    python vla/sample_libero.py --model_config_path configs/starflow_vla_libero_128.yaml \
      --checkpoint_path logs/libero_model_vla_1024_6_h8.pth --sample_index 0 --cfg 1.5

`vla.predict(...)` is the closed-loop API: returns (video, unnormalized action
chunk); execute the chunk, observe, repeat.

## Tests
    PYTHONPATH= python -m pytest vla/tests/ -v   # clear PYTHONPATH if a ROS install pollutes it
