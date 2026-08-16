# STARFlow-VLA: A Normalizing-Flow World Action Model

**Image + instruction → future video + action chunk, as one autoregressive normalizing flow.**

STARFlow-VLA extends [STARFlow-V](https://arxiv.org/abs/2511.20462) (Apple's transformer
autoregressive flow for video generation) into a vision-language-action **world action
model**, trained on [LIBERO](https://libero-project.github.io/) robot manipulation demos:

> Given one observation image and a language instruction, the model generates a video of
> the next **H** frames *and* an **H**-step action chunk — jointly, with exact
> maximum-likelihood training, no diffusion, no separate action head.

## How it works

The action chunk is treated as *one more group* in the deep flow block's autoregressive
sequence:

```
[ text tokens | obs frame | future video latents (Wan2.2 VAE) | H action tokens ]
```

- **Training** — frames are encoded by the frozen Wan2.2 causal video VAE
  (`[obs; H frames] → 1 + H/4` latent frames); video latents run through STARFlow-V's
  per-frame shallow flow blocks, the normalized action chunk `[H, 7]` runs through its
  own small shallow flow, and one causal deep block models the whole sequence. Action
  tokens have their own 7-dim projections. The global SOS shift means the last video
  token predicts the first action — **actions are generated strictly after, and
  conditioned on, the imagined video**. Loss is a per-modality mean
  `NLL_video + λ · NLL_action`.
- **Generation** — encode the observation, run one forward pass to fill the KV caches
  (i2v-style), then a single deep-block reverse pass emits video tokens and then action
  tokens; each modality goes through its own shallow inverse, the video is VAE-decoded
  and the action chunk un-normalized for execution.
- **Verified invariants** (unit-tested): exact invertibility of the joint flow
  (~1e-6 reconstruction error), causality (action tokens provably cannot influence
  video outputs), and observation-prefix preservation through the KV-cache path.

The default config is ~283M parameters at 128×128, single agentview, H = 8 — it trains
on one 16 GB GPU.

## Repository layout

All STARFlow-VLA code lives in [`vla/`](vla/); the upstream STARFlow / STARFlow-V
implementation (`transformer_flow.py`, `train.py`, `sample.py`, `utils/`, `misc/`, …)
is kept **unmodified** and used as a library.

```
vla/
├── transformer_flow_vla.py   # ActionMetaBlock (joint deep block) + WorldActionModel
├── dataset_libero.py         # raw LIBERO HDF5 dataloader + action normalization
├── compute_norm_stats.py     # per-dimension action stats (quantile normalization)
├── train_libero.py           # training entry point
├── sample_libero.py          # predict() closed-loop policy API + CLI + previews
├── norm_stats/               # generated stats (libero_10 included)
└── tests/                    # 14 tests: invertibility, causality, context, dataset
configs/starflow_vla_libero_128.yaml   # default ~283M config
```

## Quick start

```bash
conda activate starflow
pip install -r requirements.txt h5py pytest imageio imageio-ffmpeg
```

Expects raw LIBERO HDF5 data (e.g. `.../LIBERO/libero/datasets/libero_10/*.hdf5`).
Compute action normalization stats once per subset choice:

```bash
python vla/compute_norm_stats.py \
  --data_root /path/to/LIBERO/libero/datasets \
  --subsets libero_10 --output vla/norm_stats/libero_10.json
```

**Train** (first run downloads the frozen Wan2.2 VAE and flan-t5-xl):

```bash
python vla/train_libero.py --model_config_path configs/starflow_vla_libero_128.yaml
# multi-GPU: torchrun --nproc_per_node=4 vla/train_libero.py --model_config_path ...
# any config key can be overridden on the CLI, e.g. --batch_size 32 --action_loss_weight 2.0
```

**Sample / run as a policy**:

```bash
python vla/sample_libero.py --model_config_path configs/starflow_vla_libero_128.yaml \
  --checkpoint_path logs/libero_model_vla_1024_6_h8.pth --sample_index 0 --cfg 1.5
```

```python
from vla import predict
video, action_chunk = predict(model, vae, text_encoder, tokenizer, args,
                              norm_stats, obs_image, instruction, guidance=1.5)
# video: (1+H, 3, 128, 128) in [-1, 1]; action_chunk: (H, 7), ready to execute.
```

**Tests**: `PYTHONPATH= python -m pytest vla/tests/ -v`

## Configuration

**[`vla/README.md`](vla/README.md) is the full guide** — every training hyperparameter,
model-architecture knob (widths, depths, horizon, action-flow sizes), dataset setting,
and the hard invariants the code asserts (`action_horizon % 4 == 0`, even block count,
L2R + SOS, etc.). The short version: scale the model with `channels` /
`layers_per_block` / `top_block_channels`, the horizon with `action_horizon`
(divisible by 4), and the video-vs-policy trade-off with `action_loss_weight`.

## Status / scope (v1)

Single camera view, no proprioception conditioning, one action chunk per call (no
long-horizon rollout stitching yet), no simulator evaluation harness, DDP training
(keep `fsdp: 0`). See the design doc under `docs/superpowers/specs/` for the
rationale behind each decision.

## Acknowledgments & citation

This project builds directly on Apple's open-source
[**ml-starflow**](https://github.com/apple/ml-starflow) — **STARFlow**
([NeurIPS 2025 Spotlight](https://arxiv.org/abs/2506.06276)) and **STARFlow-V**
([arXiv:2511.20462](https://arxiv.org/abs/2511.20462)) — whose code is included here
unmodified. If you use this repository, please cite their work:

```bibtex
@article{gu2025starflow,
  title={STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis},
  author={Gu, Jiatao and Chen, Tianrong and Berthelot, David and Zheng, Huangjie and Wang, Yuyang and Zhang, Ruixiang and Dinh, Laurent and Bautista, Miguel Angel and Susskind, Josh and Zhai, Shuangfei},
  journal={NeurIPS},
  year={2025}
}
```

## License

The upstream STARFlow code retains its original terms — see [LICENSE](LICENSE) and
[LICENSE_MODEL](LICENSE_MODEL) before using the code or released models.
