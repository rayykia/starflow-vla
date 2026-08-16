# STARFlow-VLA: World Action Model on LIBERO — Design

**Date:** 2026-08-15
**Status:** Approved (pending final spec review)

## Goal

Extend the STARFlow-V architecture into a vision-language-action **world action model**:

> **Input:** one observation image + text instruction
> **Output:** a video of the next H frames + an action chunk of H actions

Everything remains a single normalizing flow trained by exact maximum likelihood. The
action chunk is treated as *one more group* in the deep block's autoregressive
sequence — generated after the video frames, through its own shallow flow.

This is **not** a port of the nfvla project: the dataloader is written from scratch
against raw LIBERO HDF5, and the action head is a flow group inside the deep AR flow,
not a feature-conditioned head.

## Decisions (locked in with user)

| Decision | Choice |
|---|---|
| Views | Single view: `agentview_rgb` only |
| Horizon | Video horizon == action horizon == H (default 8, must be `% 4 == 0`) |
| Init/scale | Small model (~300M) trained from scratch, 128×128 |
| Action tokens | One token per action step (H tokens of dim 7) in the deep block |
| Proprio | Not used in v1 |
| VAE | Wan2.2 causal video encoding: `[obs; H frames]` → `1 + H/4` latent frames |
| Dataloader | Written from scratch; reads `../LIBERO/libero/datasets/<subset>/*.hdf5` directly |
| Code layout | New `vla/` module subclassing `MetaBlock`/`Model`; upstream files untouched |

## Background: how STARFlow-V works today

- `Model` = stack of flow blocks. Bottom "shallow" blocks (2 layers each) + one deep
  top block (many layers, text-conditioned, temporally causal AR).
- Training runs shallow → deep; generation runs deep → shallow (`reverse_deep` then
  `reverse_shallow`).
- With `shallow_block_local=1`, shallow blocks process each frame independently
  (`b t h w c → (b t) 1 h w c`).
- Image-to-video: `model(obs, y, context=True)` runs a forward pass on the observation
  to fill KV caches; `model(noise, y, reverse=True, kv_caches=...)` continues the video.
- Each `MetaBlock` is an affine coupling AR flow: causal transformer predicts
  `(scale, shift)` for the next token; `x_out = (x_in - shift) / scale`,
  `logdet = -log(scale)`. `proj_in: C→channels`, `proj_out: channels→2C`.
- Wan2.2 causal video VAE: f16 spatial, 4× temporal, 48 latent channels. The first
  latent frame depends only on the first pixel frame (causal), which is what makes
  the i2v context pass valid when only the observation exists at inference.

## Architecture

### Sequence layout (deep block)

```
[ text tokens (txt_size) | video tokens: (1 + H/4) frames × (h·w) patches | action tokens: H × dim 7 ]
```

- 128×128 pixels → 8×8 latent grid → 64 tokens/frame, 48 channels (patch_size=1).
- H=8 → 3 latent frames → 192 video tokens + 8 action tokens + 64 text tokens.
- Causal attention over the whole sequence. The global SOS shift means the output at
  the last video token predicts action token `a_0`: **the chunk is generated strictly
  after (and conditioned on) the generated video**, matching the "one more frame"
  picture.

### New module: `vla/` (upstream `transformer_flow.py` untouched)

**`ActionMetaBlock(MetaBlock)`** — the deep block:
- Adds `proj_in_act: nn.Linear(action_dim, channels)` and
  `proj_out_act: nn.Linear(channels, 2·action_dim)` (zero-init weight, like `proj_out`).
- `forward(x_video, x_action, y, ...)`: permute video tokens with the block's
  permutation (asserted `PermutationIdentity` — the L2R 6-block parity already yields
  it), append action tokens after; SOS-shift the concatenated *input-space* sequence;
  embed each modality with its own `proj_in`; run the shared attention stack; split
  the output at the video/action boundary and apply the matching `proj_out`;
  affine-couple per modality → `(z_video, z_action, logdet_video, logdet_action)`.
- RoPE: video tokens use existing 3D positions `(px, py, pt)`; action token `j` gets
  `(0, 0, T_lat + j)` where `T_lat = 1 + H/4` (each action step is its own time step
  after the last frame).
- `reverse(...)`: token-by-token AR sampling as today, switching projections and
  freqs at the video→action boundary. Video noise `z_v ~ N(0,I)` spans all
  `1 + H/4` latent frames `[B, 1+H/4, 8, 8, 48]`; positions covered by the cached obs
  prefix are skipped and overwritten from `prefix_cache`, exactly as in today's i2v
  reverse. Action noise `z_a ~ N(0,I)` shaped `[B, H, 7]`. Reuses `KVCache` unchanged (cache length accounts
  for the extra H tokens). Jacobi sampling for the video part is optional future work;
  v1 uses plain sequential `reverse` (sequences are tiny: ~264 tokens).

**`ActionChunkFlow`** — the action shallow flow:
- `action_layers` (default `[2, 2]`) small `MetaBlock`s over the chunk alone, shaped
  `[B, H, 1, action_dim]` (h=H, w=1), alternating Identity/Flip permutations, RoPE
  positions over H (2D path with w=1), `use_sos` consistent with the main config.
- Runs **in parallel** to the video shallow blocks (the two modalities only meet in
  the deep block); the number of shallow flow steps per modality need not match.

**`WorldActionModel(Model)`**:
- New ctor args: `action_horizon`, `action_dim=7`, `action_layers=[2,2]`,
  `action_loss_weight=1.0`.
- `forward(x, a, y)`: video → patchify → video shallow blocks (per-frame, as today);
  actions → `ActionChunkFlow`; both → `ActionMetaBlock`. Returns
  `(z_video, z_action, outputs, logdets_video, logdets_action)`.
- `get_loss`: `NLL_video (mean over video dims) + λ · NLL_action (mean over action
  dims)`, each = `0.5·z² − logdet` with its own logdets. Separate per-modality means
  because action dims are ~0.6% of total — a joint mean would starve the policy
  gradient. `λ = action_loss_weight`.
- `forward_context(obs_latent, y)`: video branch only (no action tokens), fills deep
  KV cache — identical mechanics to today's i2v context pass.
- `reverse(z_v, z_a, y, kv_caches)`: deep reverse (video+action in one AR pass) →
  video shallow reverse → unpatchify (VAE decode happens in caller); action shallow
  reverse → normalized action chunk (un-normalization happens in the caller,
  `sample_libero.py`).
- CFG supported as today via `drop_label`; the whole sequence shares one guidance
  scale (may be 0 for pure policy use). The optional learnable self-denoiser stays
  video-only.

### Default model config (`configs/starflow_vla_libero_128.yaml`)

~300M params: `channels 1024`, `blocks 6`, `layers_per_block 2 2 2 2 2 12`,
`head_dim 64`, `patch_size 1`, `img_size 128`, `txt_size 64`,
`text google/flan-t5-xl` (frozen), `vae Wan-AI/Wan2.2-TI2V-5B-Diffusers:0.6`,
`channel_size 48`, `sos 1`, `seq_order L2R`, `use_softplus 1`, `cond_top_only 1`,
`use_final_norm 1`, `soft_clip 4`, `temporal_causal 2`, `shallow_block_local 1`,
`noise_std 0.3`, `action_horizon 8`, `action_dim 7`, `action_noise_std 0.1`,
`action_loss_weight 1.0`, `drop_label 0.1`. No `local_attn_window` (sequences are tiny).

## Data pipeline: `vla/dataset_libero.py` (from scratch)

Raw format (verified): `data/demo_k/obs/agentview_rgb [T,128,128,3] uint8`,
`data/demo_k/actions [T,7] float64` in [-1,1] (gripper dim 6 is ±1). 50 demos/file.
Task string parsed from the filename (strip `SCENE\d+_` prefix and `_demo.hdf5`,
underscores → spaces).

- **Map-style `torch.utils.data.Dataset`.** Index built once by scanning the chosen
  subset dirs (`libero_10`, `libero_90`, `libero_goal`, `libero_object`,
  `libero_spatial` — configurable list) and enumerating `(file, demo, t)` for
  `t ∈ [0, T-1]`. Demo lengths read once at startup (cheap metadata scan); HDF5
  file handles opened lazily and cached per dataloader worker.
- **Sample** at `(file, demo, t)`:
  - `frames`: `agentview_rgb[t : t+H+1]` (obs + H futures, clamped at episode end so
    the last frame repeats), 180°-rotated (`[::-1, ::-1]` — LIBERO stores images
    upside-down), `float in [-1,1]`, shape `[1+H, 3, 128, 128]`. Frame stride is
    fixed at 1 in v1 so frames and actions cover the identical time window.
  - `actions`: `actions[t : t+H]` (clamped the same way), normalized, `[H, 7]`.
  - `instruction`: task string.
- **Action normalization**: per-dim quantile (q01/q99 → [-1,1], clipped) with stats
  from a new `vla/compute_norm_stats.py` script that sweeps the chosen subsets once
  and writes `vla/norm_stats/<subsets>.json`. Un-normalization lives next to it for
  inference.
- **In the training loop** (GPU side): Wan2.2 encodes `[obs; H]` video-mode →
  `[B, 1+H/4, 48, 8, 8]`; `add_noise(noise_std)` on video latents;
  `actions += action_noise_std · N(0,I)` (dequantization noise — required, the
  gripper channel is discrete ±1).
- New dependency for the `starflow` env: `h5py`.

## Training: `vla/train_libero.py`

Adapted from `train.py`, stripped of secondary-dataset / aspect-ratio / fps / wds
machinery. Keeps: bf16 autocast + GradScaler, AdamW + cosine schedule with warmup,
grad clip/skip, wandb logging, checkpoint saving, NaN-skip. Loss terms logged
separately (`loss_video_z`, `loss_video_logdet`, `loss_action_z`,
`loss_action_logdet`). Periodic preview: pick a val sample, run the full
context→video→action generation, save the decoded rollout video + print
predicted-vs-ground-truth action chunk.

## Inference: `vla/sample_libero.py`

`predict(obs_image, instruction) → (video, action_chunk)`:
1. VAE-encode obs frame (image mode → latent frame 0) + `add_noise`.
2. `model(obs_latent, y, context=True)` fills KV caches (CFG duplication as today
   when guidance > 0).
3. `model(z_v, z_a, y, reverse=True, kv_caches=...)` → video latents + action chunk.
4. VAE-decode video; un-normalize actions.

This is the closed-loop policy API: execute the H actions, observe, repeat. A CLI
wrapper samples from dataset observations for qualitative checks.

## Testing (`vla/tests/`)

Tiny config (channels 128, blocks `1,1,2`, H=4, 32×32 images, random data — no VAE
or T5 needed for unit tests):
1. **Invertibility**: `forward` then `reverse` (teacher-forcing the same latents)
   reconstructs both video and action inputs within tolerance.
2. **Shapes/loss smoke test**: forward pass produces finite loss; logdet shapes match
   per-modality dims.
3. **Context-cache equivalence**: generation with an obs prefix via `forward_context`
   matches the distribution path of teacher-forcing the obs (first generated tokens
   consistent with cached prefix).
4. **Dataloader**: on one real HDF5 file — shapes, dtypes, [-1,1] ranges, clamping at
   episode end, instruction parsing.

## Out of scope (v1)

- Multi-view (wrist cam), proprio conditioning, LIBERO simulator eval harness,
  Jacobi sampling for the action group, long-horizon rollout beyond one chunk,
  finetuning from the 7B checkpoint.
