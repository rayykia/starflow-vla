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
