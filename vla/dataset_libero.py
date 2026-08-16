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
    elif isinstance(sampler, DistributedSampler) and 0 < epoch_length < len(dataset) \
            and getattr(dist, 'local_rank', 0) == 0:
        print(f'epoch_length={epoch_length} is ignored in distributed mode '
              f'(using full dataset of {len(dataset)} samples)')
    loader = DataLoader(
        dataset, batch_size=local_bs, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0)
    return loader
