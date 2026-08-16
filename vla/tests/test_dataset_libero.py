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
