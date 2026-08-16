# STARFlow-VLA World Action Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend STARFlow-V into an image+text → video+action-chunk world action model trained on LIBERO, per `docs/superpowers/specs/2026-08-15-starflow-vla-design.md`.

**Architecture:** New `vla/` module subclassing the untouched upstream `transformer_flow.py`. The deep block (`ActionMetaBlock`) runs one causal AR flow over `[text | video tokens | H action tokens]` with separate 7-dim projections for actions; a small `MetaBlock` stack (`action_blocks`) is the action shallow flow; `WorldActionModel(Model)` orchestrates forward/reverse/context. Dataloader reads raw LIBERO HDF5 directly.

**Tech Stack:** PyTorch, h5py, Wan2.2 causal video VAE (frozen), flan-t5-xl (frozen), pytest.

## Global Constraints

- Python interpreter: `~/miniconda3/envs/starflow/bin/python` (referred to as `$PY` below; set `PY=~/miniconda3/envs/starflow/bin/python` in your shell). Run tests as `$PY -m pytest ... -v` from the repo root `/home/rayykia/Projects/ml-starflow`.
- NEVER modify upstream files: `transformer_flow.py`, `train.py`, `sample.py`, `dataset.py`, `utils/*`, `misc/*`. All new code lives in `vla/` and `configs/`.
- LIBERO data root: `/home/rayykia/Projects/LIBERO/libero/datasets` (subdirs `libero_10`, `libero_90`, `libero_goal`, `libero_object`, `libero_spatial`; each HDF5 has `data/demo_k/obs/agentview_rgb [T,128,128,3] uint8` and `data/demo_k/actions [T,7] float64` in [-1,1]).
- `action_horizon % 4 == 0` (Wan2.2 temporal compression). Default `action_horizon = 8`.
- Model constraints (asserted in code): `sos=1`, `seq_order='L2R'`, even `len(layers_per_block)` (so the top block gets `PermutationIdentity`), `local_attn_window=None`.
- LIBERO images are stored upside-down: always apply 180° rotation `frame[::-1, ::-1]`.
- Commit after each task with the trailer:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01AUmNA5PWYiF4eeZyLhpxDe`

---

### Task 1: LIBERO dataloader + action norm stats

**Files:**
- Create: `vla/__init__.py`
- Create: `vla/dataset_libero.py`
- Create: `vla/compute_norm_stats.py`
- Create: `vla/tests/__init__.py`, `vla/tests/conftest.py`
- Test: `vla/tests/test_dataset_libero.py`
- Generated artifact: `vla/norm_stats/libero_10.json`

**Interfaces:**
- Produces: `LiberoVLADataset(data_root: str, subsets: list[str], horizon: int = 8, norm_stats_path: str | None = None)` — map-style Dataset; `__getitem__(i) -> (frames: FloatTensor[H+1, 3, 128, 128] in [-1,1], instruction: str, actions: FloatTensor[H, 7] normalized)`.
- Produces: `load_action_norm_stats(path: str) -> dict[str, np.ndarray]` (keys `q01`, `q99`, each shape `[7]`), `normalize_actions(a: Tensor, stats) -> Tensor`, `unnormalize_actions(a: Tensor, stats) -> Tensor`, `parse_task_from_filename(path: str) -> str`.
- Produces: `create_libero_dataloader(args, dist) -> DataLoader` (uses `DistributedSampler` when `dist.distributed`, `drop_last=True`).

- [ ] **Step 1: Install dependencies**

```bash
~/miniconda3/envs/starflow/bin/pip install h5py pytest
```

- [ ] **Step 2: Create package scaffolding**

`vla/__init__.py`:
```python
"""STARFlow-VLA: world action model extensions for STARFlow-V (LIBERO)."""
```

`vla/tests/__init__.py`: empty file.

`vla/tests/conftest.py`:
```python
import sys
import pathlib

# repo root on sys.path so `import transformer_flow` / `import vla.*` work under pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
```

- [ ] **Step 3: Write the failing tests**

`vla/tests/test_dataset_libero.py`:
```python
import json
import pathlib
import numpy as np
import pytest
import torch

from vla.dataset_libero import (
    LiberoVLADataset, normalize_actions, unnormalize_actions,
    parse_task_from_filename,
)

DATA_ROOT = pathlib.Path('/home/rayykia/Projects/LIBERO/libero/datasets')
needs_data = pytest.mark.skipif(not DATA_ROOT.exists(), reason='LIBERO data not found')


def test_parse_task_from_filename():
    assert parse_task_from_filename(
        'x/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5'
    ) == 'turn on the stove and put the moka pot on it'
    # subsets without SCENE prefix keep the whole name
    assert parse_task_from_filename(
        'x/pick_up_the_black_bowl_demo.hdf5') == 'pick up the black bowl'


def test_normalize_roundtrip():
    stats = {'q01': np.array([-1.0] * 7, dtype=np.float32),
             'q99': np.array([1.0] * 7, dtype=np.float32)}
    a = torch.rand(5, 7) * 1.8 - 0.9
    a_n = normalize_actions(a, stats)
    torch.testing.assert_close(unnormalize_actions(a_n, stats), a, atol=1e-5, rtol=1e-5)
    assert a_n.abs().max() <= 1.0


@needs_data
def test_dataset_sample_shapes():
    ds = LiberoVLADataset(str(DATA_ROOT), ['libero_10'], horizon=8)
    frames, instruction, actions = ds[0]
    assert frames.shape == (9, 3, 128, 128)
    assert frames.min() >= -1.0 and frames.max() <= 1.0
    assert isinstance(instruction, str) and len(instruction) > 0
    assert actions.shape == (8, 7)
    assert torch.isfinite(actions).all()


@needs_data
def test_dataset_clamps_at_episode_end():
    ds = LiberoVLADataset(str(DATA_ROOT), ['libero_10'], horizon=8)
    # find the last index of the first demo: entries are (path, demo, t) ordered by t
    path0, demo0, _ = ds.index[0]
    last_i = max(i for i, (p, d, t) in enumerate(ds.index) if p == path0 and d == demo0)
    frames, _, actions = ds[last_i]
    # beyond the end, both frames and actions repeat the final step
    torch.testing.assert_close(frames[-1], frames[-2])
    torch.testing.assert_close(actions[-1], actions[-2])
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `$PY -m pytest vla/tests/test_dataset_libero.py -v`
Expected: FAIL / collection error with `ModuleNotFoundError: No module named 'vla.dataset_libero'`.

- [ ] **Step 5: Implement the dataset**

`vla/dataset_libero.py`:
```python
"""LIBERO HDF5 dataloader for STARFlow-VLA.

Reads raw LIBERO demos directly (no meta files):
  data/demo_k/obs/agentview_rgb : [T, 128, 128, 3] uint8 (stored upside-down)
  data/demo_k/actions           : [T, 7] float64 in [-1, 1] (gripper dim 6 is +-1)

Each sample at (file, demo, t):
  frames      : [H+1, 3, 128, 128] float in [-1, 1] (obs at t, then t+1..t+H,
                clamped at the episode end so the last frame repeats)
  instruction : task string parsed from the filename
  actions     : [H, 7] quantile-normalized actions a_t..a_{t+H-1} (same clamping)
"""
import glob
import json
import os
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


def parse_task_from_filename(filepath: str) -> str:
    base = os.path.basename(filepath)
    task = re.sub(r'_demo\.hdf5$', '', base)
    m = re.search(r'SCENE\d+_', task)
    if m:
        task = task[m.end():]
    return task.replace('_', ' ')


def load_action_norm_stats(path: str) -> dict:
    with open(path) as f:
        stats = json.load(f)['actions']
    return {k: np.asarray(stats[k], dtype=np.float32) for k in ('q01', 'q99')}


def normalize_actions(a: torch.Tensor, stats: dict) -> torch.Tensor:
    q01 = torch.as_tensor(stats['q01'], dtype=a.dtype, device=a.device)
    q99 = torch.as_tensor(stats['q99'], dtype=a.dtype, device=a.device)
    a = 2.0 * (a - q01) / (q99 - q01 + 1e-8) - 1.0
    return a.clamp(-1.0, 1.0)


def unnormalize_actions(a: torch.Tensor, stats: dict) -> torch.Tensor:
    q01 = torch.as_tensor(stats['q01'], dtype=a.dtype, device=a.device)
    q99 = torch.as_tensor(stats['q99'], dtype=a.dtype, device=a.device)
    return (a.clamp(-1.0, 1.0) + 1.0) / 2.0 * (q99 - q01 + 1e-8) + q01


class LiberoVLADataset(Dataset):

    def __init__(self, data_root: str, subsets, horizon: int = 8,
                 norm_stats_path: str | None = None):
        self.horizon = horizon
        self.norm_stats = load_action_norm_stats(norm_stats_path) if norm_stats_path else None
        self.index = []   # (h5_path, demo_key, t)
        self.tasks = {}   # h5_path -> instruction
        for subset in subsets:
            files = sorted(glob.glob(os.path.join(data_root, subset, '*.hdf5')))
            assert files, f'no HDF5 files found under {data_root}/{subset}'
            for h5_path in files:
                self.tasks[h5_path] = parse_task_from_filename(h5_path)
                with h5py.File(h5_path, 'r') as f:
                    for demo_key in sorted(f['data'].keys()):
                        T = f['data'][demo_key]['actions'].shape[0]
                        self.index += [(h5_path, demo_key, t) for t in range(T)]
        self._handles = {}  # per-worker lazy h5py.File cache (never pickled)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handles'] = {}  # h5py handles are not picklable across workers
        return state

    def _file(self, path: str) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, 'r')
        return self._handles[path]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        path, demo_key, t = self.index[i]
        g = self._file(path)['data'][demo_key]
        T = g['actions'].shape[0]
        H = self.horizon

        frames = g['obs/agentview_rgb'][t:min(t + H + 1, T)]      # (n, 128, 128, 3)
        acts = g['actions'][t:min(t + H, T)].astype(np.float32)   # (m, 7)
        if frames.shape[0] < H + 1:                               # clamp at episode end
            pad = np.repeat(frames[-1:], H + 1 - frames.shape[0], axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        if acts.shape[0] < H:
            pad = np.repeat(acts[-1:], H - acts.shape[0], axis=0)
            acts = np.concatenate([acts, pad], axis=0)

        frames = frames[:, ::-1, ::-1].copy()                     # 180-degree rotation
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        acts = torch.from_numpy(acts)
        if self.norm_stats is not None:
            acts = normalize_actions(acts, self.norm_stats)
        return frames, self.tasks[path], acts


def create_libero_dataloader(args, dist):
    dataset = LiberoVLADataset(
        args.libero_root, args.libero_subsets,
        horizon=args.action_horizon, norm_stats_path=args.norm_stats)
    local_bs = args.batch_size // dist.world_size // max(getattr(args, 'acc', 1), 1)
    sampler = DistributedSampler(dataset, shuffle=True) if dist.distributed else None
    # single-GPU: honor epoch_length as "samples per epoch" (fresh random draw
    # each epoch); in distributed mode epoch_length is ignored (full dataset)
    epoch_length = getattr(args, 'epoch_length', 0) or 0
    if sampler is None and 0 < epoch_length < len(dataset):
        sampler = torch.utils.data.RandomSampler(dataset, num_samples=epoch_length)
    loader = DataLoader(
        dataset, batch_size=local_bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0)
    return loader
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `$PY -m pytest vla/tests/test_dataset_libero.py -v`
Expected: 4 PASS.

- [ ] **Step 7: Implement the norm-stats script**

`vla/compute_norm_stats.py`:
```python
"""Compute per-dimension action normalization stats over LIBERO subsets.

Usage:
  $PY vla/compute_norm_stats.py --data_root /home/rayykia/Projects/LIBERO/libero/datasets \
      --subsets libero_10 --output vla/norm_stats/libero_10.json
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--subsets', nargs='+', default=['libero_10'])
    parser.add_argument('--output', required=True, type=str)
    args = parser.parse_args()

    all_actions, num_demos = [], 0
    for subset in args.subsets:
        for h5_path in sorted(glob.glob(os.path.join(args.data_root, subset, '*.hdf5'))):
            with h5py.File(h5_path, 'r') as f:
                for demo_key in f['data']:
                    all_actions.append(np.asarray(f['data'][demo_key]['actions'], dtype=np.float32))
                    num_demos += 1
    actions = np.concatenate(all_actions, axis=0)
    stats = {
        'actions': {
            'mean': actions.mean(0).tolist(),
            'std': actions.std(0).tolist(),
            'q01': np.quantile(actions, 0.01, axis=0).tolist(),
            'q99': np.quantile(actions, 0.99, axis=0).tolist(),
        },
        'metadata': {'subsets': args.subsets, 'num_demos': num_demos,
                     'num_steps': int(actions.shape[0])},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Wrote {args.output}: {num_demos} demos, {actions.shape[0]} steps')


if __name__ == '__main__':
    main()
```

- [ ] **Step 8: Generate the libero_10 stats file**

Run:
```bash
cd /home/rayykia/Projects/ml-starflow && $PY vla/compute_norm_stats.py \
  --data_root /home/rayykia/Projects/LIBERO/libero/datasets \
  --subsets libero_10 --output vla/norm_stats/libero_10.json
```
Expected: `Wrote vla/norm_stats/libero_10.json: 500 demos, <N> steps` (~10 files × 50 demos). Inspect the JSON: `q01`/`q99` are 7-element lists; gripper dim (index 6) is `[-1, 1]`.

- [ ] **Step 9: Commit**

```bash
git add vla/ && git commit -m "feat(vla): LIBERO HDF5 dataloader and action norm stats"
```

---

### Task 2: ActionMetaBlock — joint video+action forward (training path)

**Files:**
- Create: `vla/transformer_flow_vla.py`
- Test: `vla/tests/test_action_meta_block.py`

**Interfaces:**
- Consumes (upstream): `MetaBlock`, `PermutationIdentity`, `KVCache`, `INV_SOFTPLUS_1` from `transformer_flow`; `get_positions`, `VisionRotaryEmbeddingFast` from `misc.pe`.
- Produces: `ActionMetaBlock(MetaBlock)` with
  - `ActionMetaBlock.upgrade(block: MetaBlock, action_dim: int) -> ActionMetaBlock` (in-place class upgrade; adds `proj_in_act: Linear(action_dim, channels)`, `proj_out_act: Linear(channels, 2*action_dim)` zero-init weight).
  - `forward(x, y=None, rope=None, kv_cache=None, guidance=None, actions=None)`:
    with `actions=None` delegates to `MetaBlock.forward` (used by the context pass);
    with `actions: [B, A, Da]` returns `(z_v: [B,T,h,w,C], z_a: [B,A,Da], y, logdet_v: [B,Nv,C], logdet_a: [B,A,Da])`.
  - `get_freqs_cis(x, y, rope, num_actions=0)` — appends action positions `(0, 0, T_lat + j)`.

- [ ] **Step 1: Write the failing tests**

`vla/tests/test_action_meta_block.py`:
```python
import torch

from transformer_flow import MetaBlock, PermutationIdentity, KVCache
from misc.pe import VisionRotaryEmbeddingFast
from vla.transformer_flow_vla import ActionMetaBlock

B, T, HW, C, A, Da, TXT, TXT_DIM = 2, 2, 8, 4, 4, 7, 8, 16


def make_block():
    torch.manual_seed(0)
    blk = MetaBlock(
        in_channels=C, channels=64, img_size=HW,
        permutation=PermutationIdentity(HW * HW), pt_seq_len=HW,
        num_layers=2, head_dim=32, txt_size=TXT, txt_dim=TXT_DIM,
        use_rope=True, use_sos=True, use_softplus=True,
        use_final_norm=True, soft_clip=4)
    blk = ActionMetaBlock.upgrade(blk, action_dim=Da)
    rope = VisionRotaryEmbeddingFast(dim=16, pt_seq_len=HW, latent_len=TXT, no_buffer=True)
    return blk.eval(), rope


def make_inputs():
    torch.manual_seed(1)
    x = torch.randn(B, T, HW, HW, C)   # patchified video latents
    y = torch.randn(B, TXT, TXT_DIM)
    a = torch.randn(B, A, Da)
    return x, y, a


def test_joint_forward_shapes():
    blk, rope = make_block()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v, z_a, _, logdet_v, logdet_a = blk(x, y, rope=rope, actions=a)
    assert z_v.shape == x.shape
    assert z_a.shape == a.shape
    assert logdet_v.shape == (B, T * HW * HW, C)
    assert logdet_a.shape == (B, A, Da)
    assert torch.isfinite(z_v).all() and torch.isfinite(z_a).all()


def test_video_outputs_unaffected_by_actions():
    # causal attention: appended action tokens must not change video outputs
    blk, rope = make_block()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v_joint, _, _, ld_joint, _ = blk(x, y, rope=rope, actions=a)
        z_v_only, _, ld_only = MetaBlock.forward(blk, x, y, rope=rope)
    torch.testing.assert_close(z_v_joint, z_v_only, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ld_joint, ld_only, atol=1e-5, rtol=1e-4)


def test_delegates_when_actions_none():
    blk, rope = make_block()
    x, y, _ = make_inputs()
    with torch.no_grad():
        out = blk(x, y, rope=rope)  # MetaBlock 3-tuple path
    assert len(out) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest vla/tests/test_action_meta_block.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'vla.transformer_flow_vla'`.

- [ ] **Step 3: Implement ActionMetaBlock (forward only)**

`vla/transformer_flow_vla.py`:
```python
"""STARFlow-VLA: video + action-chunk normalizing flow on top of STARFlow-V.

The deep block runs one causal AR flow over [text | video tokens | H action
tokens]. The global SOS shift means the output at the last video token predicts
action a_0, so the chunk is generated strictly after (and conditioned on) the
video. Actions use their own input/output projections (7-dim vs 48-dim video).
"""
import tqdm
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from transformer_flow import (
    Model, MetaBlock, KVCache, PermutationIdentity, PermutationFlip,
    INV_SOFTPLUS_1,
)
from misc.pe import VisionRotaryEmbeddingFast, get_positions


class ActionMetaBlock(MetaBlock):
    """Deep block extended with an action-chunk group at the end of the AR sequence."""

    @classmethod
    def upgrade(cls, block: MetaBlock, action_dim: int) -> 'ActionMetaBlock':
        # In-place class upgrade instead of re-assembling MetaBlock's ~20 ctor
        # kwargs: keeps the flow config byte-identical to what Model built and
        # immune to upstream kwarg drift.
        assert type(block) is MetaBlock, f'cannot upgrade {type(block)}'
        assert block.use_sos, 'ActionMetaBlock requires sos=1'
        assert isinstance(block.permutation, PermutationIdentity), (
            'top block must use PermutationIdentity: use seq_order=L2R with an '
            'even number of blocks')
        block.__class__ = cls
        channels = block.proj_in.out_features
        block.action_dim = action_dim
        block.proj_in_act = torch.nn.Linear(action_dim, channels)
        block.proj_out_act = torch.nn.Linear(channels, 2 * action_dim)
        torch.nn.init.constant_(block.proj_out_act.weight, 0)
        return block

    def get_freqs_cis(self, x, y, rope, num_actions=0):
        if num_actions == 0:
            return super().get_freqs_cis(x, y, rope)
        assert not rope.is_1d and self.txt_dim > 0, 'joint path expects 3D RoPE with text'
        h, w = x.size(-3), x.size(-2)
        d = x.size(1) if x.dim() == 5 else 0
        txt_size = y.size(1) if self.txt_size > 0 and y is not None else 0
        pos = get_positions(h, w, txt_size, rope.pt_seq_len, d, mode='3d')
        t0 = float(max(d, 1))  # action step j sits at time T_lat + j, spatial (0, 0)
        act_pos = torch.stack([
            torch.zeros(num_actions), torch.zeros(num_actions),
            t0 + torch.arange(num_actions, dtype=torch.float32)], dim=-1)
        pos = torch.cat([pos, act_pos.to(pos.dtype)], dim=0)
        return rope(pos.type_as(x))

    def _couple(self, out, x_in):
        xa, xb = out.chunk(2, dim=-1)
        dtype = xa.dtype
        xa, xb, x_in = xa.float(), xb.float(), x_in.float()
        xa = F.softplus(xa + INV_SOFTPLUS_1) if self.use_softplus else xa.exp()
        z = (x_in - xb) / xa
        return z.to(dtype), -torch.log(xa)

    def forward(self, x, y=None, rope=None, kv_cache=None, guidance=None, actions=None):
        if actions is None:  # pure-video path (context pass, CFG cache filling)
            return super().forward(x, y, rope, kv_cache, guidance)
        assert kv_cache is None, 'joint forward is training-only; use context=True for caches'
        assert not guidance, 'guidance is inference-only'

        A = actions.size(1)
        freqs_cis = self.get_freqs_cis(x, y, rope, num_actions=A) if rope is not None else None

        x = self.permutation(x)                       # identity: (B, Nv, C)
        xv_in, a_in = x.clone(), actions.clone()
        Nv = x.size(1)

        # AR inputs [SOS, v_0..v_{Nv-1}, a_0..a_{A-2}], embedded per-modality.
        # Unlike video-only MetaBlock, the last video token IS fed: it predicts a_0.
        h = self.get_proj_in(torch.cat([self.get_sos_embed(x), x], dim=1))
        h = torch.cat([h, self.proj_in_act(a_in.to(h.dtype)[:, :-1])], dim=1)

        y = self.proj_txt(y) if self.txt_dim > 0 else None

        for it, block in enumerate(self.attn_blocks):
            checkpoint_attn = self.training and self.use_checkpoint > 0 and \
                ((it + 1) % self.use_checkpoint == 0)
            if self.use_checkpoint_mlp is not None:
                checkpoint_mlp = self.training and self.use_checkpoint_mlp > 0 and \
                    ((it + 1) % self.use_checkpoint_mlp == 0)
            else:
                checkpoint_mlp = checkpoint_attn
            h, y = block(h, y, None, 1.0, None, freqs_cis, None,
                         checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp)

        if self.use_final_norm:
            h, y = self.final_norm(h), self.final_norm(y) if y is not None else None

        out_v = self.get_proj_out(h[:, :Nv])          # (B, Nv, 2C), includes soft_clip
        out_a = self.proj_out_act(h[:, Nv:])          # (B, A, 2*Da)
        if self.soft_clip > 0:
            out_a = self.soft_clip * torch.tanh(out_a / self.soft_clip)

        z_v, logdet_v = self._couple(out_v, xv_in)
        z_a, logdet_a = self._couple(out_a, a_in)
        z_v = self.permutation(z_v, inverse=True)     # (B, T, h, w, C)
        return z_v, z_a, y, logdet_v, logdet_a
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest vla/tests/test_action_meta_block.py -v`
Expected: 3 PASS. `test_video_outputs_unaffected_by_actions` is the load-bearing one — it proves causal masking and position layout are right.

- [ ] **Step 5: Commit**

```bash
git add vla/transformer_flow_vla.py vla/tests/test_action_meta_block.py
git commit -m "feat(vla): ActionMetaBlock joint video+action forward"
```

---

### Task 3: ActionMetaBlock — reverse (generation path)

**Files:**
- Modify: `vla/transformer_flow_vla.py` (add methods to `ActionMetaBlock`)
- Test: `vla/tests/test_action_meta_block.py` (append)

**Interfaces:**
- Produces: `ActionMetaBlock.reverse(z: [B,T,h,w,C], z_act: [B,A,Da], y=None, guidance=0, guide_what='ab', attn_temp=1.0, annealed_guidance=False, rope=None, verbose=False, kv_cache=None, **unused) -> (x: [B,T,h,w,C], a: [B,A,Da])` — exact inverse of the joint `forward`; honors a pre-filled `kv_cache` (obs prefix) exactly like upstream i2v.
- Produces: `ActionMetaBlock.reverse_step_action(x, a, j, kv_cache, attn_temp, freqs_cis) -> (za, zb)`.

- [ ] **Step 1: Write the failing test**

Append to `vla/tests/test_action_meta_block.py`:
```python
def test_deep_block_invertibility():
    blk, rope = make_block()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v, z_a, _, _, _ = blk(x, y, rope=rope, actions=a)
        x_rec, a_rec = blk.reverse(z_v, z_a, y, rope=rope, kv_cache=KVCache())
    torch.testing.assert_close(x_rec, x, atol=2e-3, rtol=1e-3)
    torch.testing.assert_close(a_rec, a, atol=2e-3, rtol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest vla/tests/test_action_meta_block.py::test_deep_block_invertibility -v`
Expected: FAIL — `MetaBlock.reverse` (inherited) has a different signature; TypeError or shape mismatch.

- [ ] **Step 3: Implement reverse**

Add to `ActionMetaBlock` in `vla/transformer_flow_vla.py`:
```python
    def reverse_step_action(self, x, a, j, kv_cache, attn_temp=1.0, freqs_cis=None):
        # Input at action step j: last video token (j == 0) or a_{j-1}; output
        # always through the action head, predicting a_j.
        original_dtype = a.dtype
        if j == 0:
            h = self.get_proj_in(x[:, -1:].to(self.proj_in.weight.dtype))
        else:
            h = self.proj_in_act(a[:, j - 1:j].to(self.proj_in_act.weight.dtype))
        for i, block in enumerate(self.attn_blocks):
            h, _ = block(h, None, attn_temp=attn_temp, freqs_cis=freqs_cis,
                         kv_cache=partial(kv_cache, i))
        if self.use_final_norm:
            h = self.final_norm(h)
        h = self.proj_out_act(h)
        if self.soft_clip > 0:
            h = self.soft_clip * torch.tanh(h / self.soft_clip)
        xa, xb = h.chunk(2, dim=-1)
        return xa.to(original_dtype), xb.to(original_dtype)

    def reverse(self, z, z_act, y=None, guidance=0, guide_what='ab', attn_temp=1.0,
                annealed_guidance=False, rope=None, verbose=False,
                kv_cache: KVCache = None, **unused_kwargs):
        kv_cache = kv_cache if kv_cache is not None else KVCache()
        original_dtype = z.dtype
        z, z_act = z.float(), z_act.float()
        A = z_act.size(1)
        freqs_cis = self.get_freqs_cis(z, y, rope, num_actions=A) if rope is not None else None
        if guidance > 0:
            z, z_act = torch.cat([z, z], 0), torch.cat([z_act, z_act], 0)

        reuse_kv_cache = kv_cache.prefix_cache is not None and kv_cache.kv_index[0] > 0
        kv_cache = self.initialize_kv_cache(kv_cache, z, freqs_cis, reuse_kv_cache)

        z = self.permutation(z)
        if self.txt_dim > 0 and not reuse_kv_cache:
            self.reverse_step_condition(y, kv_cache, None, attn_temp, freqs_cis)
        txt_size = y.size(1) if self.txt_dim > 0 else 0

        x = z.clone()
        if reuse_kv_cache:
            x[:, :kv_cache.prefix_cache.size(1)] = kv_cache.prefix_cache

        def _scale(za):
            za = F.softplus(za + INV_SOFTPLUS_1) if self.use_softplus else za.exp()
            return za.squeeze(1)

        T = x.size(1)  # use_sos asserted at upgrade time
        for t in tqdm.trange(T, disable=not verbose, desc='VLA deep-block video', leave=False):
            if reuse_kv_cache and kv_cache.kv_index[0] > t + txt_size:
                continue  # obs prefix already cached
            za, zb = self.reverse_step(x, t, kv_cache, None, y, attn_temp, freqs_cis)
            za, zb = _scale(za.float()), zb.float().squeeze(1)
            if guidance > 0 and guide_what:
                r = (t + 1) / T if annealed_guidance else 1.0
                zb, za = self.guidance(za, zb, guidance, r, guide_what)
            x[:, t] = z[:, t] * za + zb

        a = z_act.clone()
        for j in range(A):
            za, zb = self.reverse_step_action(x, a, j, kv_cache, attn_temp, freqs_cis)
            za, zb = _scale(za.float()), zb.float().squeeze(1)
            if guidance > 0 and guide_what:
                zb, za = self.guidance(za, zb, guidance, 1.0, guide_what)
            a[:, j] = z_act[:, j] * za + zb

        if guidance > 0:
            x, a = x.chunk(2, dim=0)[0], a.chunk(2, dim=0)[0]
            kv_cache.remove_negative_cache()

        x = self.permutation(x, inverse=True)
        return x.to(original_dtype), a.to(original_dtype)
```

- [ ] **Step 4: Run all block tests to verify they pass**

Run: `$PY -m pytest vla/tests/test_action_meta_block.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add vla/transformer_flow_vla.py vla/tests/test_action_meta_block.py
git commit -m "feat(vla): ActionMetaBlock reverse (joint video+action AR sampling)"
```

---

### Task 4: WorldActionModel — full model, loss, context, reverse

**Files:**
- Modify: `vla/transformer_flow_vla.py` (add `WorldActionModel`, `setup_vla_model`)
- Modify: `vla/__init__.py` (exports)
- Test: `vla/tests/test_world_action_model.py`

**Interfaces:**
- Consumes: `ActionMetaBlock` (Tasks 2-3), upstream `Model`, `MetaBlock`, `PermutationFlip`, `VisionRotaryEmbeddingFast`.
- Produces: `WorldActionModel(Model)`:
  - ctor keyword-only extras: `action_horizon=8, action_dim=7, action_channels=256, action_head_dim=64, action_layers=(2, 2), action_loss_weight=1.0` plus all `Model` kwargs.
  - `forward(x, y=None, actions=None, reverse=False, kv_caches=None, denoiser=False, context=False, **kwargs)`:
    - training (`actions` given): returns `(z_v, z_a, outputs, logdets_v, logdets_a)`
    - `context=True` / `actions=None`: delegates to `Model` (context returns `kv_caches`)
    - `reverse=True`: `x`=video noise `[B, 1+H/4, C, h, w]`, `actions`=action noise `[B, H, 7]` → returns `(video_latents, action_chunk)`
  - `get_loss(z_v, logdets_v, z_a, logdets_a, action_loss_weight=None) -> dict` with keys `loss, loss_video_z, loss_video_logdet, loss_action_z, loss_action_logdet`.
- Produces: `setup_vla_model(args, txt_dim) -> WorldActionModel`.

- [ ] **Step 1: Write the failing tests**

`vla/tests/test_world_action_model.py`:
```python
import torch

from vla.transformer_flow_vla import WorldActionModel

B, T, C, HW, A, Da, TXT, TXT_DIM = 2, 2, 4, 8, 4, 7, 8, 32


def tiny_model():
    torch.manual_seed(0)
    return WorldActionModel(
        in_channels=C, img_size=HW, patch_size=1, channels=64,
        num_blocks=4, layers_per_block=[1, 1, 1, 2], head_dim=32,
        rope=True, pt_seq_len=HW, sos=True, txt_size=TXT, txt_dim=TXT_DIM,
        cond_top_only=True, use_softplus=True, use_final_norm=True,
        seq_order='L2R', temporal_causal=1, shallow_block_local=True, soft_clip=4,
        action_horizon=A, action_dim=Da, action_channels=32,
        action_head_dim=16, action_layers=[1, 1], action_loss_weight=1.0,
    ).eval()


def make_inputs():
    torch.manual_seed(1)
    x = torch.randn(B, T, C, HW, HW)   # latent video, channel-first as from the VAE
    y = torch.randn(B, TXT, TXT_DIM)
    a = torch.randn(B, A, Da)
    return x, y, a


def test_forward_shapes_and_loss():
    model = tiny_model()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v, z_a, outputs, ldv, lda = model(x, y, actions=a)
    assert z_v.shape == x.shape and z_a.shape == a.shape
    assert len(ldv) == 4 and len(lda) == 3  # 3 shallow + top; 2 action shallow + top
    loss = model.get_loss(z_v, ldv, z_a, lda)
    for k in ('loss', 'loss_video_z', 'loss_video_logdet',
              'loss_action_z', 'loss_action_logdet'):
        assert torch.isfinite(loss[k]), k


def test_full_model_invertibility():
    model = tiny_model()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v, z_a, _, _, _ = model(x, y, actions=a)
        x_rec, a_rec = model(z_v, y, actions=z_a, reverse=True)
    torch.testing.assert_close(x_rec, x, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(a_rec, a, atol=5e-3, rtol=1e-2)


def test_context_prefix_preserved():
    # i2v-style: obs context fills KV caches; generated frame 0 must equal the obs
    model = tiny_model()
    x, y, a = make_inputs()
    obs = x[:, :1]
    with torch.no_grad():
        kv_caches = model(obs, y, context=True)
        x_gen, a_gen = model(torch.randn_like(x), y, actions=torch.randn_like(a),
                             reverse=True, kv_caches=kv_caches)
    assert x_gen.shape == x.shape and a_gen.shape == a.shape
    torch.testing.assert_close(x_gen[:, :1], obs, atol=5e-3, rtol=1e-2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest vla/tests/test_world_action_model.py -v`
Expected: `ImportError: cannot import name 'WorldActionModel'`.

- [ ] **Step 3: Implement WorldActionModel + setup_vla_model**

Append to `vla/transformer_flow_vla.py`:
```python
class WorldActionModel(Model):
    """STARFlow-V + action chunk: image+text -> video + H actions, one joint flow."""

    def __init__(self, *, action_horizon=8, action_dim=7, action_channels=256,
                 action_head_dim=64, action_layers=(2, 2), action_loss_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        assert kwargs.get('seq_order') == 'L2R', 'STARFlow-VLA requires seq_order=L2R'
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.action_loss_weight = action_loss_weight

        # upgrade the top block in place (asserts sos + PermutationIdentity)
        self.blocks[-1] = ActionMetaBlock.upgrade(self.blocks[-1], action_dim)

        # action shallow flow: small MetaBlocks over the chunk alone, shaped
        # (B, A, 1, Da), alternating direction, 1D RoPE over the A steps
        perms = [PermutationIdentity(action_horizon), PermutationFlip(action_horizon)]
        self.action_blocks = torch.nn.ModuleList([
            MetaBlock(
                in_channels=action_dim, channels=action_channels,
                img_size=action_horizon, permutation=perms[i % 2],
                pt_seq_len=action_horizon, num_layers=num_layers,
                head_dim=action_head_dim, txt_size=0, txt_dim=0,
                use_rope=True, use_sos=self.use_sos, use_softplus=self.use_softplus,
                use_swiglu=kwargs.get('use_swiglu', False),
                use_qk_norm=kwargs.get('use_qk_norm', False),
                use_post_norm=kwargs.get('use_post_norm', False),
                use_final_norm=kwargs.get('use_final_norm', False),
                use_bias=kwargs.get('use_bias', True),
                norm_type=kwargs.get('norm_type', 'layer_norm'),
                soft_clip=kwargs.get('soft_clip', 0),
            ) for i, num_layers in enumerate(action_layers)])
        self.action_rope = VisionRotaryEmbeddingFast(
            dim=action_head_dim // 2, pt_seq_len=action_horizon,
            no_buffer=True, is_1d=True)

    def forward(self, x, y=None, actions=None, reverse=False, kv_caches=None,
                denoiser=False, context=False, **kwargs):
        if reverse:
            assert actions is not None, 'reverse needs action noise z_a as `actions`'
            return self.reverse(x, actions, y, kv_caches=kv_caches, **kwargs)
        if context or denoiser or actions is None:
            # context pass (obs only), denoiser, or plain video forward: parent
            # handles it; ActionMetaBlock.forward(actions=None) behaves as MetaBlock
            return super().forward(x, y, reverse=False, kv_caches=kv_caches,
                                   denoiser=denoiser, context=context, **kwargs)

        B, A, Da = actions.shape
        logdets_v, logdets_a, outputs = [], [], []

        # video shallow flow (per-frame with shallow_block_local, as upstream)
        x = self.patchify(x)
        outputs.append(x)
        for block in self.blocks[:-1]:
            if self.shallow_block_local and x.dim() == 5:
                x = rearrange(x, 'b t h w c -> (b t) 1 h w c')
            x, _, logdet = block(x, y, self.feat_rope, kv_cache=None)
            if self.shallow_block_local and x.dim() == 5:
                x = rearrange(x, '(b t) 1 h w c -> b t h w c',
                              b=outputs[0].size(0), t=outputs[0].size(1))
                logdet = rearrange(logdet, '(b t) l c -> b t l c',
                                   b=outputs[0].size(0), t=outputs[0].size(1))
            logdets_v.append(logdet)
            outputs.append(x)

        # action shallow flow
        a = actions.view(B, A, 1, Da)
        for block in self.action_blocks:
            a, _, logdet_a = block(a, None, self.action_rope)
            logdets_a.append(logdet_a)
        a = a.reshape(B, A, Da)

        # joint deep block
        z_v, z_a, y, logdet_v, logdet_a = self.blocks[-1](
            x, y, self.feat_rope_gen, actions=a)
        logdets_v.append(logdet_v)
        logdets_a.append(logdet_a)
        outputs.append(z_v)
        return self.unpatchify(z_v), z_a, outputs, logdets_v, logdets_a

    def get_loss(self, z_v, logdets_v, z_a, logdets_a, action_loss_weight=None):
        # separate per-modality means: action dims are <1% of total, a joint
        # mean would starve the policy gradient
        w = self.action_loss_weight if action_loss_weight is None else action_loss_weight
        loss_v_z = 0.5 * z_v.pow(2).mean(dim=tuple(range(1, z_v.dim())))
        loss_v_ld = -sum(ld.mean(dim=tuple(range(1, ld.dim()))) for ld in logdets_v)
        loss_a_z = 0.5 * z_a.pow(2).mean(dim=tuple(range(1, z_a.dim())))
        loss_a_ld = -sum(ld.mean(dim=tuple(range(1, ld.dim()))) for ld in logdets_a)
        loss = ((loss_v_z + loss_v_ld) + w * (loss_a_z + loss_a_ld)).mean()
        return {'loss': loss,
                'loss_video_z': loss_v_z.detach().mean(),
                'loss_video_logdet': loss_v_ld.detach().mean(),
                'loss_action_z': loss_a_z.detach().mean(),
                'loss_action_logdet': loss_a_ld.detach().mean()}

    def reverse(self, x, actions, y=None, guidance=0, verbose=False,
                kv_caches=None, **sampling_kwargs):
        need_caches = kv_caches is not None
        kv_caches = kv_caches if kv_caches is not None \
            else [KVCache() for _ in range(len(self.blocks))]
        B = x.size(0)

        # deep block: one AR pass generates video tokens then action tokens
        xp = self.patchify(x)
        xp, a = self.blocks[-1].reverse(
            xp, actions, y, guidance, rope=self.feat_rope_gen,
            kv_cache=kv_caches[0], verbose=verbose, **sampling_kwargs)
        x = self.unpatchify(xp)
        if not need_caches:
            kv_caches[0].delete()

        # shallow blocks are unconditional under cond_top_only: drop guidance
        if self.cond_top_only and guidance > 0:
            guidance, y = 0, y.chunk(2, dim=0)[0]

        seq = [x]
        x = self.reverse_shallow(x, y, guidance, verbose, kv_caches,
                                 False, need_caches, seq, **sampling_kwargs)

        for block in reversed(self.action_blocks):
            a = block.reverse(a.view(B, self.action_horizon, 1, self.action_dim),
                              None, 0, rope=self.action_rope, kv_cache=KVCache())
            a = a.reshape(B, self.action_horizon, self.action_dim)
        return x, a


def setup_vla_model(args, txt_dim):
    return WorldActionModel(
        in_channels=args.channel_size, img_size=args.img_size,
        patch_size=args.patch_size, channels=args.channels,
        num_blocks=len(args.layers_per_block), layers_per_block=args.layers_per_block,
        head_dim=args.head_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        rope=args.rope, pt_seq_len=args.pt_seq_len, sos=args.sos,
        txt_size=args.txt_size, txt_dim=txt_dim, cond_top_only=args.cond_top_only,
        use_softplus=args.use_softplus, use_swiglu=args.use_swiglu,
        use_bias=args.use_bias, use_qk_norm=args.use_qk_norm,
        use_post_norm=args.use_post_norm, use_final_norm=args.use_final_norm,
        norm_type=args.norm_type, soft_clip=args.soft_clip, seq_order=args.seq_order,
        temporal_causal=args.temporal_causal, shallow_block_local=args.shallow_block_local,
        top_block_channels=args.top_block_channels,
        use_checkpoint=args.gradient_checkpoint,
        use_checkpoint_mlp=args.gradient_checkpoint_mlp,
        action_horizon=args.action_horizon, action_dim=args.action_dim,
        action_channels=args.action_channels, action_head_dim=args.action_head_dim,
        action_layers=args.action_layers, action_loss_weight=args.action_loss_weight,
    )
```

Update `vla/__init__.py`:
```python
"""STARFlow-VLA: world action model extensions for STARFlow-V (LIBERO)."""
from .transformer_flow_vla import ActionMetaBlock, WorldActionModel, setup_vla_model
from .dataset_libero import (
    LiberoVLADataset, create_libero_dataloader,
    load_action_norm_stats, normalize_actions, unnormalize_actions,
)
```

- [ ] **Step 4: Run the full test suite to verify it passes**

Run: `$PY -m pytest vla/tests/ -v`
Expected: all PASS (dataset + block + model). `test_context_prefix_preserved` proves the i2v-style context→generate path end to end.

- [ ] **Step 5: Commit**

```bash
git add vla/
git commit -m "feat(vla): WorldActionModel with action shallow flow, joint loss, context reverse"
```

---

### Task 5: Config + training script

**Files:**
- Create: `configs/starflow_vla_libero_128.yaml`
- Create: `vla/train_libero.py`
- Test: `vla/tests/test_config.py`

**Interfaces:**
- Consumes: `setup_vla_model`, `create_libero_dataloader` (Tasks 1, 4); upstream `utils` (`Distributed`, `setup_encoder`, `setup_vae`, `parallelize_model`, `save_model`, `CosineLRSchedule`, `encode_text`, `drop_label`, `add_noise`, `set_random_seed`, `sync_ctx`, `Metrics`).
- Produces: `get_vla_parser() -> argparse.ArgumentParser` (extends `train.get_tarflow_parser` with `--libero_root, --libero_subsets, --norm_stats, --num_workers, --action_horizon, --action_dim, --action_channels, --action_head_dim, --action_layers, --action_loss_weight, --action_noise_std, --preview_guidance`), `load_vla_config(path) -> Namespace`, `main(args)`.

- [ ] **Step 1: Write the failing test**

`vla/tests/test_config.py`:
```python
from vla.train_libero import load_vla_config


def test_load_vla_config():
    args = load_vla_config('configs/starflow_vla_libero_128.yaml')
    assert args.action_horizon % 4 == 0
    assert len(args.layers_per_block) % 2 == 0, 'even blocks so top gets PermutationIdentity'
    assert args.sos == 1 and args.seq_order == 'L2R'
    assert args.channel_size == 48 and args.img_size == 128
    assert args.action_dim == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest vla/tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'vla.train_libero'`.

- [ ] **Step 3: Write the config**

`configs/starflow_vla_libero_128.yaml`:
```yaml
arguments:
- dataset: libero
- img_size: 128
- txt_size: 64
- channel_size: 48
- patch_size: 1
- channels: 1024
- blocks: 6
- layers_per_block: 2 2 2 2 2 12
- head_dim: 64
- pt_seq_len: 8
- rope: 1
- sos: 1
- seq_order: L2R
- nvp: 1
- use_softplus: 1
- cond_top_only: 1
- use_final_norm: 1
- soft_clip: 4
- temporal_causal: 1
- shallow_block_local: 1
- learnable_self_denoiser: 0
- noise_std: 0.3
- batch_size: 64
- epoch_length: 50000
- lr: 1e-4
- min_lr: 1e-6
- epochs: 100
- drop_label: 0.1
- grad_clip: 1
- grad_skip: 1
- loss_scaling: 1
- save_every: 10
- sample_freq: 5
- vae: Wan-AI/Wan2.2-TI2V-5B-Diffusers:0.6
- finetuned_vae: none
- text: google/flan-t5-xl
- cfg: 1.5
- fsdp: 0
- gradient_checkpoint: 0
- action_horizon: 8
- action_dim: 7
- action_channels: 256
- action_head_dim: 64
- action_layers: 2 2
- action_loss_weight: 1.0
- action_noise_std: 0.1
- libero_root: /home/rayykia/Projects/LIBERO/libero/datasets
- libero_subsets: libero_10
- norm_stats: vla/norm_stats/libero_10.json
- num_workers: 4
```

- [ ] **Step 4: Write the training script**

`vla/train_libero.py`:
```python
"""STARFlow-VLA training on LIBERO.

Usage:
  $PY vla/train_libero.py --model_config_path configs/starflow_vla_libero_128.yaml
  (multi-GPU: torchrun --nproc_per_node=N vla/train_libero.py --model_config_path ...)
"""
import argparse
import contextlib
import os
import pathlib
import sys
import time

import torch
import torch.amp
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import utils
from misc import print  # local_rank-0 print
from utils import encode_text, drop_label, add_noise
from vla.transformer_flow_vla import setup_vla_model
from vla.dataset_libero import create_libero_dataloader

WANDB_API_KEY = os.environ.get('WANDB_API_KEY', None)


def get_vla_parser():
    from train import get_tarflow_parser
    parser = get_tarflow_parser()
    parser.add_argument('--libero_root', default='/home/rayykia/Projects/LIBERO/libero/datasets', type=str)
    parser.add_argument('--libero_subsets', default=['libero_10'], type=str, nargs='+')
    parser.add_argument('--norm_stats', default='vla/norm_stats/libero_10.json', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--action_horizon', default=8, type=int)
    parser.add_argument('--action_dim', default=7, type=int)
    parser.add_argument('--action_channels', default=256, type=int)
    parser.add_argument('--action_head_dim', default=64, type=int)
    parser.add_argument('--action_layers', default=[2, 2], type=int, nargs='+')
    parser.add_argument('--action_loss_weight', default=1.0, type=float)
    parser.add_argument('--action_noise_std', default=0.1, type=float)
    parser.add_argument('--preview_guidance', default=1.5, type=float)
    return parser


def load_vla_config(config_path: str) -> argparse.Namespace:
    with open(config_path) as f:
        model_configs = yaml.safe_load(f)
    arg_str = ''
    for conf in model_configs['arguments']:
        for key in conf:
            arg_str += f'--{key} {conf[key]} '
    return get_vla_parser().parse_args(arg_str.split())


def merge_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, 'model_config_path', None):
        return args
    provided = {a[2:].replace('-', '_') for a in sys.argv[1:] if a.startswith('--')}
    merged = vars(load_vla_config(args.model_config_path))
    for k, v in vars(args).items():
        if k in provided:
            merged[k] = v
    return argparse.Namespace(**merged)


def main(args):
    args = merge_cli_args(args)
    assert args.action_horizon % 4 == 0, 'Wan2.2 needs action_horizon % 4 == 0'

    dist = utils.Distributed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = args.train_seed if args.train_seed is not None else time.time_ns() % 2**32
    utils.set_random_seed(seed + dist.rank)

    if dist.rank == 0 and WANDB_API_KEY is not None:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project='starflow-vla', name=args.wandb_name or 'libero', config=vars(args))

    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:32s}: {v}')

    # data
    data_loader = create_libero_dataloader(args, dist)
    num_batches = len(data_loader)
    print(f'{num_batches} batches/epoch, {len(data_loader.dataset):,} samples')

    # frozen encoders
    tokenizer, text_encoder = utils.setup_encoder(args, dist, device)
    text_encoder.requires_grad_(False)
    vae = utils.setup_vae(args, dist, device)
    vae.requires_grad_(False)
    args.img_size = args.img_size // vae.downsample_factor

    # model
    model = setup_vla_model(args, txt_dim=text_encoder.config.hidden_size).to(device)
    if dist.local_rank == 0:
        import torchinfo
        torchinfo.summary(model)

    if args.resume_path:
        print(f'Loading checkpoint: {args.resume_path}')
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'), strict=False)
        epoch_start = args.resume_epoch or 0
    else:
        epoch_start = 0

    model, model_ddp = utils.parallelize_model(args, model, dist, device)
    trainable = [p for p in model_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    warmup = args.warmup_steps if args.warmup_steps is not None else num_batches
    lr_schedule = utils.CosineLRSchedule(optimizer, warmup, args.epochs * num_batches,
                                         args.min_lr, args.lr)
    lr_schedule.counter += epoch_start * num_batches
    scaler = torch.amp.GradScaler() if args.loss_scaling else None

    grad_accum = max(args.acc, 1)
    model_name = f'vla_{args.channels}_{len(args.layers_per_block)}_h{args.action_horizon}'
    ckpt_file = args.logdir / f'libero_model_{model_name}.pth'
    sample_dir = args.logdir / f'libero_samples_{model_name}'
    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    print(f'{" Training ":-^80}')
    total_steps = epoch_start * num_batches
    for epoch in range(epoch_start, args.epochs):
        if hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)
        metrics = utils.Metrics()
        for it, (frames, instructions, actions) in enumerate(data_loader):
            frames = frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    x = vae.encode(frames)                       # (B, 1+H/4, 48, 8, 8)
                x, _ = add_noise(x, args.noise_std, args.noise_type)
                actions = actions + args.action_noise_std * torch.randn_like(actions)
                with torch.no_grad():
                    y = encode_text(text_encoder, tokenizer,
                                    drop_label(list(instructions), args.drop_label),
                                    args.txt_size, device)

                needs_update = (it + 1) % grad_accum == 0
                if it % grad_accum == 0:
                    optimizer.zero_grad()

                z_v, z_a, outputs, logdets_v, logdets_a = model_ddp(x, y, actions=actions)
                loss_dict = model.get_loss(z_v, logdets_v, z_a, logdets_a)
                loss = loss_dict['loss'] / grad_accum

                if dist.gather_concat(loss.detach().view(1)).isnan().any():
                    print('nan detected, skipping step')
                    continue

                with utils.sync_ctx(model_ddp, sync=needs_update) if grad_accum > 1 \
                        else contextlib.nullcontext():
                    (scaler.scale(loss) if scaler else loss).backward()

                if needs_update:
                    grad_norm, skip = None, False
                    if args.grad_clip > 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                        skip = args.grad_skip and total_steps > 100 and \
                            grad_norm.item() > args.grad_clip
                    if skip:
                        print(f'skipping update, grad_norm={grad_norm.item():.3f}')
                        optimizer.zero_grad()
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    current_lr = lr_schedule.step()
                    total_steps += 1
                    if not skip:
                        metrics.update(loss_dict)

                    if (it // grad_accum) % 10 == 9:
                        print(f'{total_steps:,} steps - ' + '  '.join(
                            f'{k}: {v:.4f}' for k, v in loss_dict.items()))
                        if dist.rank == 0 and WANDB_API_KEY is not None:
                            import wandb
                            log = {k: v.item() if torch.is_tensor(v) else v
                                   for k, v in loss_dict.items()}
                            log.update({'lr': current_lr, 'steps': total_steps})
                            if grad_norm is not None:
                                log['grad_norm'] = grad_norm.item()
                            wandb.log(log, step=total_steps)
            if args.dry_run:
                break

        if not args.dry_run:
            utils.save_model(args, dist, model, ckpt_file)
            if epoch % args.save_every == 0:
                utils.save_model(args, dist, model, str(ckpt_file) + f'_epoch{epoch + 1:04d}')
        dist.barrier()

        # rollout preview added in vla/sample_libero.py (Task 6)
        if args.sample_freq > 0 and (epoch % args.sample_freq == 0) and not args.dry_run:
            from vla.sample_libero import preview_rollout
            preview_rollout(model, vae, text_encoder, tokenizer, args, dist,
                            frames, instructions, actions, sample_dir, epoch, total_steps)

        if args.dry_run:
            break


if __name__ == '__main__':
    args = get_vla_parser().parse_args()
    main(args)
```

Note: the `preview_rollout` import is intentionally lazy and lands in Task 6; with `sample_freq: 5` the first preview call happens at epoch 0, so **until Task 6 is done, run training only with `--dry_run 1` or `--sample_freq 0`**.

- [ ] **Step 5: Run the config test to verify it passes**

Run: `$PY -m pytest vla/tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Manual GPU smoke test (single step)**

Run:
```bash
cd /home/rayykia/Projects/ml-starflow && $PY vla/train_libero.py \
  --model_config_path configs/starflow_vla_libero_128.yaml \
  --batch_size 4 --num_workers 0 --sample_freq 0 --dry_run 1
```
Expected: config dump, dataset size line, torchinfo summary (~300M params), one training step printing finite `loss`, `loss_video_z`, `loss_video_logdet`, `loss_action_z`, `loss_action_logdet`, clean exit. (Downloads Wan2.2 VAE + flan-t5-xl on first run.)

- [ ] **Step 7: Commit**

```bash
git add configs/starflow_vla_libero_128.yaml vla/train_libero.py vla/tests/test_config.py
git commit -m "feat(vla): LIBERO training script and 128px config"
```

---

### Task 6: Policy inference API, training previews, README

**Files:**
- Create: `vla/sample_libero.py`
- Create: `vla/README.md`
- Modify: `vla/__init__.py` (export `predict`)

**Interfaces:**
- Consumes: `WorldActionModel` reverse/context (Task 4), `unnormalize_actions` (Task 1), upstream `encode_text`, `add_noise`, `save_samples_unified`.
- Produces:
  - `generate_rollout(model, vae, text_encoder, tokenizer, args, obs_frames: Tensor[n,1,3,128,128] in [-1,1], instructions: list[str], guidance: float, device) -> (video_px: Tensor[n,1+H,3,128,128], actions_norm: Tensor[n,H,7])`
  - `predict(model, vae, text_encoder, tokenizer, args, norm_stats, obs_image: Tensor[3,128,128], instruction: str, guidance: float = 0.0) -> (video_px, action_chunk_unnormalized)` — the closed-loop policy API.
  - `preview_rollout(model, vae, text_encoder, tokenizer, args, dist, frames, instructions, actions, sample_dir, epoch, step)` — called from `train_libero.py`.

- [ ] **Step 1: Implement sample_libero.py**

`vla/sample_libero.py`:
```python
"""STARFlow-VLA inference: obs image + instruction -> future video + action chunk.

CLI (qualitative check against a dataset sample):
  $PY vla/sample_libero.py --model_config_path configs/starflow_vla_libero_128.yaml \
      --checkpoint_path logs/libero_model_vla_1024_6_h8.pth --sample_index 0 --cfg 1.5
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import utils
from misc import print
from utils import encode_text, add_noise, save_samples_unified
from vla.dataset_libero import LiberoVLADataset, load_action_norm_stats, unnormalize_actions


@torch.no_grad()
def generate_rollout(model, vae, text_encoder, tokenizer, args, obs_frames,
                     instructions, guidance, device):
    """obs_frames: (n, 1, 3, 128, 128) in [-1, 1]. Returns pixel video + normalized actions."""
    n = obs_frames.size(0)
    x0 = vae.encode(obs_frames.to(device))                    # (n, 1, 48, 8, 8)
    x0, _ = add_noise(x0, args.noise_std, args.noise_type)

    texts = list(instructions) + ([''] * n if guidance > 0 else [])
    y = encode_text(text_encoder, tokenizer, texts, args.txt_size, device)

    kv_caches = model(x0, y, context=True)

    lat = args.img_size  # already divided by vae.downsample_factor in the callers
    t_lat = 1 + args.action_horizon // 4
    z_v = torch.randn(n, t_lat, args.channel_size, lat, lat, device=device)
    z_a = torch.randn(n, args.action_horizon, args.action_dim, device=device)
    video_lat, actions = model(z_v, y, actions=z_a, reverse=True,
                               kv_caches=kv_caches, guidance=guidance)
    video_px = vae.decode(video_lat)                          # (n, 1+H, 3, 128, 128)
    return video_px, actions


@torch.no_grad()
def predict(model, vae, text_encoder, tokenizer, args, norm_stats,
            obs_image, instruction, guidance=0.0):
    """Closed-loop policy API. obs_image: (3, 128, 128) in [-1, 1]."""
    device = next(model.parameters()).device
    obs = obs_image[None, None].to(device)
    video_px, actions = generate_rollout(
        model, vae, text_encoder, tokenizer, args, obs, [instruction], guidance, device)
    return video_px[0], unnormalize_actions(actions[0].float().cpu(), norm_stats)


@torch.no_grad()
def preview_rollout(model, vae, text_encoder, tokenizer, args, dist,
                    frames, instructions, actions, sample_dir, epoch, step):
    """Training-time preview on the last batch: save rollout video, print action MAE."""
    model.eval()
    n = min(4, frames.size(0))
    device = frames.device
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        video_px, a_gen = generate_rollout(
            model, vae, text_encoder, tokenizer, args,
            frames[:n, :1], list(instructions[:n]), args.preview_guidance, device)
    save_samples_unified(
        samples=video_px.float(), save_dir=sample_dir, filename_prefix='rollout',
        epoch_or_iter=epoch + 1, fps=10, dist=dist,
        wandb_log=False, grid_arrangement='grid')
    mae = (a_gen.float() - actions[:n].float()).abs().mean().item()
    print(f'[preview] epoch {epoch + 1}: action MAE (normalized) = {mae:.4f}')
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True, type=str)
    parser.add_argument('--checkpoint_path', required=True, type=str)
    parser.add_argument('--sample_index', default=0, type=int)
    parser.add_argument('--cfg', default=1.5, type=float)
    parser.add_argument('--seed', default=0, type=int)
    cli = parser.parse_args()

    from vla.train_libero import load_vla_config
    args = load_vla_config(cli.model_config_path)

    dist = utils.Distributed()
    device = torch.device('cuda')
    utils.set_random_seed(cli.seed)

    tokenizer, text_encoder = utils.setup_encoder(args, dist, device)
    vae = utils.setup_vae(args, dist, device)
    args.img_size = args.img_size // vae.downsample_factor

    from vla.transformer_flow_vla import setup_vla_model
    model = setup_vla_model(args, txt_dim=text_encoder.config.hidden_size).to(device)
    model.load_state_dict(torch.load(cli.checkpoint_path, map_location='cpu'), strict=True)
    model.eval().requires_grad_(False)

    norm_stats = load_action_norm_stats(args.norm_stats)
    ds = LiberoVLADataset(args.libero_root, args.libero_subsets,
                          horizon=args.action_horizon, norm_stats_path=args.norm_stats)
    frames, instruction, gt_actions = ds[cli.sample_index]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        video, action_chunk = predict(
            model, vae, text_encoder, tokenizer, args, norm_stats,
            frames[0], instruction, guidance=cli.cfg)

    out_dir = pathlib.Path(args.logdir) / 'vla_samples'
    save_samples_unified(samples=video[None].float().cpu(), save_dir=out_dir,
                         filename_prefix='predict', epoch_or_iter=cli.sample_index, fps=10)
    print(f'instruction : {instruction}')
    print(f'predicted   :\n{action_chunk.numpy().round(3)}')
    print(f'ground truth:\n{unnormalize_actions(gt_actions, norm_stats).numpy().round(3)}')


if __name__ == '__main__':
    main()
```

Append to `vla/__init__.py`:
```python
from .sample_libero import predict, generate_rollout
```

Caveat: keep this import LAST in `__init__.py`; `sample_libero` imports from `vla.dataset_libero`, which must already be initialized.

- [ ] **Step 2: Re-run the full test suite (guard against import regressions)**

Run: `$PY -m pytest vla/tests/ -v`
Expected: all PASS (the new module must import cleanly; `vla/__init__.py` now pulls in `sample_libero`).

- [ ] **Step 3: Write vla/README.md**

```markdown
# STARFlow-VLA (LIBERO world action model)

Image + instruction -> next-H-frames video + H-step action chunk, as one
normalizing flow (see `docs/superpowers/specs/2026-08-15-starflow-vla-design.md`).

## Setup
    pip install h5py pytest   # inside the starflow env

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
    python -m pytest vla/tests/ -v
```

- [ ] **Step 4: Manual GPU verification — preview path end to end**

Run:
```bash
cd /home/rayykia/Projects/ml-starflow && $PY vla/train_libero.py \
  --model_config_path configs/starflow_vla_libero_128.yaml \
  --batch_size 4 --num_workers 0 --epochs 1 --epoch_length 40 --sample_freq 1
```
Expected: after the (short) epoch, `[preview] epoch 1: action MAE (normalized) = ...` and a `rollout_*.mp4` in `logs/libero_samples_vla_1024_6_h8/`. The video will be noise-like (untrained model) — only the plumbing is being verified.

- [ ] **Step 5: Commit**

```bash
git add vla/ && git commit -m "feat(vla): policy inference API, training previews, README"
```

---

## Self-Review Notes (already applied)

- Spec coverage: dataloader+norm stats (T1), ActionMetaBlock fwd/rev (T2-3), action shallow flow + WorldActionModel + per-modality loss + context (T4), config+training (T5), inference/policy API + previews (T6). Simulator eval, multi-view, proprio, Jacobi are explicitly out of scope per spec.
- The `test_video_outputs_unaffected_by_actions` test doubles as the design's causality guarantee; `test_context_prefix_preserved` is the spec's "context-cache equivalence" test.
- Type consistency: `forward` joint return is a 5-tuple `(z_v, z_a, outputs, logdets_v, logdets_a)` everywhere (train script, tests); `reverse` returns `(video, actions)` everywhere; norm-stats dict uses `q01`/`q99` keys in both dataset and sampler.
