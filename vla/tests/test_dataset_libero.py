import json
import pathlib
import numpy as np
import pytest
import torch

from vla.dataset_libero import (
    LiberoVLADataset, normalize_actions, unnormalize_actions, normalize_state,
    load_norm_stats, parse_task_from_filename, PROPRIO_DIM,
)

DATA_ROOT = pathlib.Path('/data/jgu/ruichend/LIBERO/libero/datasets')
NORM_STATS = pathlib.Path(__file__).resolve().parents[1] / 'norm_stats' / 'libero_10.json'
needs_data = pytest.mark.skipif(not DATA_ROOT.exists(), reason='LIBERO data not found')


def test_norm_stats_have_state():
    stats = load_norm_stats(str(NORM_STATS))
    for name, dim in (('actions', 7), ('state', PROPRIO_DIM)):
        assert stats[name]['mean'].shape == (dim,) and stats[name]['std'].shape == (dim,)
        assert (stats[name]['std'] > 0).all()


def test_normalize_state_zscore():
    stats = {'mean': np.full(PROPRIO_DIM, 0.5, dtype=np.float32),
             'std': np.full(PROPRIO_DIM, 0.5, dtype=np.float32)}
    s = torch.tensor([[0.0, 0.5, 1.0, 1.5, -0.5, 0.25, 0.75, 2.0]])
    out = normalize_state(s, stats)
    # (s - 0.5) / (0.5 + 1e-6): unbounded, no clamping
    torch.testing.assert_close(out, torch.tensor([[-1.0, 0.0, 1.0, 2.0, -2.0, -0.5, 0.5, 3.0]]),
                               atol=1e-4, rtol=1e-4)


def test_parse_task_from_filename():
    assert parse_task_from_filename(
        'x/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5'
    ) == 'turn on the stove and put the moka pot on it'
    # subsets without SCENE prefix keep the whole name
    assert parse_task_from_filename(
        'x/pick_up_the_black_bowl_demo.hdf5') == 'pick up the black bowl'


def test_normalize_roundtrip():
    stats = {'mean': np.linspace(-0.5, 0.5, 7, dtype=np.float32),
             'std': np.linspace(0.1, 1.0, 7, dtype=np.float32)}
    a = torch.rand(5, 7) * 4.0 - 2.0
    a_n = normalize_actions(a, stats)
    expected = (a - torch.from_numpy(stats['mean'])) / (torch.from_numpy(stats['std']) + 1e-6)
    torch.testing.assert_close(a_n, expected)
    torch.testing.assert_close(unnormalize_actions(a_n, stats), a, atol=1e-5, rtol=1e-5)
    assert a_n.abs().max() > 1.0   # z-scores are not clamped to [-1, 1]


@needs_data
def test_dataset_sample_shapes():
    ds = LiberoVLADataset(str(DATA_ROOT), ['libero_10'], horizon=8)
    frames, instruction, actions, proprio = ds[0]
    assert frames.shape == (9, 2, 3, 128, 128)   # (1+H, views [agentview, wrist], C, H, W)
    assert frames.min() >= -1.0 and frames.max() <= 1.0
    assert isinstance(instruction, str) and len(instruction) > 0
    assert actions.shape == (8, 7)
    assert torch.isfinite(actions).all()
    assert proprio.shape == (PROPRIO_DIM,) and torch.isfinite(proprio).all()
    assert 2.5 < proprio[3:6].norm() < 3.6      # raw ee_ori is an axis-angle of ~pi


@needs_data
def test_dataset_normalizes_proprio():
    ds = LiberoVLADataset(str(DATA_ROOT), ['libero_10'], horizon=8, norm_stats_path=str(NORM_STATS))
    _, _, actions, proprio = ds[0]
    assert torch.isfinite(actions).all() and actions.abs().max() < 10.0   # z-scores, unclamped
    assert proprio.shape == (PROPRIO_DIM,) and proprio.abs().max() < 10.0


@needs_data
def test_dataset_clamps_at_episode_end():
    ds = LiberoVLADataset(str(DATA_ROOT), ['libero_10'], horizon=8)
    # find the last index of the first demo: entries are (path, demo, t) ordered by t
    path0, demo0, _ = ds.index[0]
    last_i = max(i for i, (p, d, t) in enumerate(ds.index) if p == path0 and d == demo0)
    frames, _, actions, _ = ds[last_i]
    # beyond the end, both frames and actions repeat the final step
    torch.testing.assert_close(frames[-1], frames[-2])
    torch.testing.assert_close(actions[-1], actions[-2])


@needs_data
def test_dataset_views_are_distinct_cameras():
    ds = LiberoVLADataset(str(DATA_ROOT), ['libero_10'], horizon=8)
    frames, _, _, _ = ds[0]
    assert not torch.allclose(frames[0, 0], frames[0, 1])   # agentview != wrist
