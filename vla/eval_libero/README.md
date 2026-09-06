# LIBERO evaluation for STARFlow-VLA

Client/server benchmark harness, adapted from `SimVLANF/evaluation/libero`
(`serve_smolvlm_libero.py` / `libero_client.py` / `run_eval_all.sh`). The server
runs the STARFlow-VLA policy in the `nfvla` env; the clients run the LIBERO
simulator in the `libero` env and talk to it over WebSocket with the
`openpi_client` wire protocol.

| File | Role |
|---|---|
| `serve_policy.py` | Policy server: loads a checkpoint + YAML config, samples `[video, action chunk]`, **score-denoises** the chunk with the training `action_noise_std`, batches concurrent clients into one sampling pass |
| `libero_client.py` | LIBERO rollout loop (128×128 agentview + wrist images, 180° rotated, as in training), episode-level sharding, per-task JSON results + rollout mp4s |
| `run_server.sh` | One-liner server launcher (`GPU`, `PORT`, `CKPT`, `CONFIG` env vars) |
| `run_eval_all.sh` | Launches `NUM_PARALLEL` sharded clients per suite against one server, then summarizes |
| `summarize_results.py` | Merges the per-(task, shard) JSONs into per-task / per-suite success rates (`summary.json`, `summary.md`) |

## What differs from the SimVLANF reference

1. **Model / observation.** The client sends the agentview and wrist
   (`robot0_eye_in_hand_image`) images, rendered by LIBERO at the native dataset
   resolution (128×128) and rotated 180° exactly like `vla/dataset_libero.py`, plus the 8-D proprio state
   `[eef_pos(3), axis-angle(3), gripper_qpos(2)]`, which the server z-score-normalizes with
   the `state` mean/std of the norm-stats file (same layout as the HDF5
   `[ee_pos, ee_ori, gripper_states]` used in training) and feeds as the proprio
   condition token. Both views are VAE-encoded into the video prefix (and both are
   generated); the agentview image also goes through the SmolVLM condition encoder
   together with the prompt. Chunk = 8 actions;
   by default the whole chunk is executed before re-planning (`--replan_steps`).
2. **Score-based denoising with the training σ.** The flow is trained on
   `a + N(0, action_noise_std²)` in normalized action space, so a sample lives in
   that noisy space. The server applies one Tweedie step
   `a ← a − σ² ∇ₐ NLL(a | video, text)` with `σ = action_noise_std` **read from
   the config** (`vla/sample_libero.py::score_denoise`) instead of the reference's
   hard-coded `noise_std = 0.05`. The video is held fixed (video tokens precede
   action tokens in the AR sequence, so this is the exact conditional score).
   Sanity check on the trained checkpoint (64 dataset chunks, actions noised with
   σ=0.1): MAE to clean **0.079 → 0.013** at strength 1.0 (0.5 → 0.042, 1.5 → 0.042,
   i.e. the σ² step is right); gripper channel 0.083 → 0.004. Open-loop sampled
   chunks: MAE vs GT 0.114 → 0.084 (cfg 0), 0.118 → 0.095 (cfg 1.5).
   `--denoise 0` disables it, `--denoise_strength` scales the step,
   `--denoise_video 1` also denoises dumped videos with `noise_std`.
3. **Cross-client batching.** AR flow sampling is latency-bound (A100: 3.4 s for
   batch 1, 3.8 s for 16, 4.5 s for 32), so the server fuses the requests of all
   connected clients into one pass. It waits (≤ `--batch_wait_ms`, 500 ms) until
   every connected client has a request queued so the clients stay phase-locked;
   measured with 4 clients: batch = 4 every round, ~2.9 s per control chunk for
   all four together. Parallelism therefore comes from running many client
   processes (`run_eval_all.sh` shards each suite by episode).
4. **Bookkeeping.** Per-(task, shard) JSON results with per-episode steps/time,
   task-tagged video names, `summarize_results.py`, `--task_ids` / `--shard k/P`.

## Usage

```bash
# 1. server (nfvla env; PYTHONPATH is cleared inside the script)
conda activate nfvla
GPU=2 PORT=8000 CKPT=logs/libero_model_vla_1024_6_h8.pth bash vla/eval_libero/run_server.sh
#    extra flags go to serve_policy.py, unknown --flags override the YAML config:
GPU=2 bash vla/eval_libero/run_server.sh --cfg 0 --dump_dir eval_results/dumps --dump_every 100

# 2. clients (libero env). port, trials, output dir, rendering GPUs, shards per suite
conda activate libero
bash vla/eval_libero/run_eval_all.sh 8000 50 eval_results/starflow_vla "3 4" 16
SUITES="libero_10 libero_goal" NO_VIDEO=1 bash vla/eval_libero/run_eval_all.sh 8000 50 out "3" 16

# a single shard / quick check
python vla/eval_libero/libero_client.py --port 8000 --task_suite libero_10 --task_ids 0 --num_trials 5

# 3. summary (also usable while running)
python vla/eval_libero/summarize_results.py eval_results/starflow_vla
```

Output layout: `<out>/<suite>/task00_<desc>_ep3_success.mp4`,
`<out>/results/<suite>/task00_shard2of16.json`, `<out>/logs/<suite>_shard2of16.txt`,
`<out>/summary.{json,md}`.

Sizing: keep `NUM_PARALLEL × #suites ≤ --max_batch` (32). Each client is one
MuJoCo process (~1 CPU core, ~0.5 GB on its rendering GPU); the server needs
~25 GB. With 16 shards `libero_10` (500 episodes, ≤ 900 steps → ≤ 113 chunks)
takes roughly 500 / 16 × 4 min ≈ 2 h.

The default checkpoint was trained on `libero_10` only, so `libero_spatial /
object / goal` are out-of-distribution for it (`SUITES` defaults to `libero_10`).

## Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--model_config_path` | `configs/starflow_vla_libero_128.yaml` | training config (σ's, horizon, norm stats path) |
| `--checkpoint_path` | required | model `.pth` |
| `--cfg` | config `cfg` (1.5) | classifier-free guidance scale for sampling |
| `--denoise` / `--denoise_strength` | 1 / 1.0 | Tweedie step on the action chunk with `action_noise_std` |
| `--denoise_video` | 0 | also denoise dumped videos with `noise_std` |
| `--max_batch` / `--batch_wait_ms` | 32 / 500 | batching (see above) |
| `--dump_dir` / `--dump_every` | – / 50 | save predicted future video (mp4), obs (png), actions (npy) of every N-th call |
| `--seed`, `--host`, `--port`, `--log_every`, `--norm_stats` | | |
| any other `--flag value` | | forwarded as a config override (e.g. `--action_noise_std 0.05`) |

## Client flags

`--task_suite`, `--task_ids`, `--shard k/P` (every P-th (task, trial) episode),
`--num_trials` (50), `--seed` (7), `--replan_steps` (server horizon = 8),
`--env_resolution` (128), `--max_steps` (per-suite table), `--video_out`,
`--results_dir`, `--no_video`, `--host/--port/--connection_info`.
