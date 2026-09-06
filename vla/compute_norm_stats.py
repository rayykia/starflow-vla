"""Compute per-dimension action and proprio (state) normalization stats over LIBERO subsets.

The dataset normalizes with x_norm = (x - mean) / (std + 1e-6); q01/q99 are written
for reference only.

  actions : [T, 7]  delta_xyz(3), delta_euler(3), gripper_cmd(1)
  state   : [T, 8]  ee_pos(3), ee_ori(3, axis-angle), gripper_states(2)   (proprio conditioning)

Usage:
  $PY vla/compute_norm_stats.py --data_root /data/jgu/ruichend/LIBERO/libero/datasets \
      --subsets libero_10 --output vla/norm_stats/libero_10.json
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np

PROPRIO_KEYS = ('ee_pos', 'ee_ori', 'gripper_states')  # keep in sync with vla/dataset_libero.py


def summarize(x: np.ndarray) -> dict:
    return {
        'mean': x.mean(0).tolist(),
        'std': x.std(0).tolist(),
        'q01': np.quantile(x, 0.01, axis=0).tolist(),
        'q99': np.quantile(x, 0.99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--subsets', nargs='+', default=['libero_10'])
    parser.add_argument('--output', required=True, type=str)
    args = parser.parse_args()

    all_actions, all_states, num_demos = [], [], 0
    for subset in args.subsets:
        for h5_path in sorted(glob.glob(os.path.join(args.data_root, subset, '*.hdf5'))):
            with h5py.File(h5_path, 'r') as f:
                for demo_key in f['data']:
                    demo = f['data'][demo_key]
                    all_actions.append(np.asarray(demo['actions'], dtype=np.float32))
                    all_states.append(np.concatenate(
                        [np.asarray(demo['obs'][k], dtype=np.float32) for k in PROPRIO_KEYS], axis=1))
                    num_demos += 1
    actions = np.concatenate(all_actions, axis=0)
    states = np.concatenate(all_states, axis=0)
    stats = {
        'actions': summarize(actions),
        'state': summarize(states),
        'metadata': {'subsets': args.subsets, 'num_demos': num_demos,
                     'num_steps': int(actions.shape[0]),
                     'state_layout': list(PROPRIO_KEYS)},
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Wrote {args.output}: {num_demos} demos, {actions.shape[0]} steps '
          f'(actions {actions.shape[1]}-d, state {states.shape[1]}-d)')


if __name__ == '__main__':
    main()
