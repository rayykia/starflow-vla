#!/usr/bin/env python3
"""Merge the per-(task, shard) JSON files written by libero_client.py.

  python vla/eval_libero/summarize_results.py <output_dir>      # <output_dir>/results/<suite>/*.json
  python vla/eval_libero/summarize_results.py <output_dir>/results/libero_10

Prints per-task and per-suite success rates and writes summary.json / summary.md
into <output_dir>. Works on partial results while an evaluation is still running.
Only needs the standard library.
"""
import glob
import json
import os
import sys
from collections import defaultdict


def collect(root):
    files = sorted(glob.glob(os.path.join(root, 'results', '*', '*.json'))) or \
        sorted(glob.glob(os.path.join(root, '*.json')))
    suites = defaultdict(lambda: defaultdict(lambda: {
        'description': '', 'successes': 0, 'trials': 0, 'steps_success': [], 'seconds': 0.0}))
    for fn in files:
        with open(fn) as f:
            r = json.load(f)
        if 'task_suite' not in r:
            continue
        t = suites[r['task_suite']][r['task_id']]
        t['description'] = r['task_description']
        t['successes'] += r['successes']
        t['trials'] += r['num_trials']
        t['seconds'] += r.get('seconds', 0.0)
        t['steps_success'] += [e['steps'] for e in r.get('episodes', []) if e['success']]
    return suites


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    suites = collect(root)
    if not suites:
        print(f'no result JSON files under {root}')
        return
    summary, lines = {}, []
    for suite in sorted(suites):
        tasks = suites[suite]
        lines.append(f'\n=== {suite} ===')
        lines.append(f'{"task":>4}  {"succ/trials":>11}  {"rate":>6}  {"avg steps (succ)":>16}  description')
        succ = trials = 0
        rates = []
        for tid in sorted(tasks):
            t = tasks[tid]
            rate = t['successes'] / max(t['trials'], 1)
            rates.append(rate)
            succ += t['successes']
            trials += t['trials']
            avg_steps = sum(t['steps_success']) / len(t['steps_success']) if t['steps_success'] else float('nan')
            lines.append(f'{tid:>4}  {t["successes"]:>4}/{t["trials"]:<6}  {rate*100:5.1f}%  '
                         f'{avg_steps:>16.1f}  {t["description"]}')
        overall = succ / max(trials, 1)
        task_mean = sum(rates) / max(len(rates), 1)
        lines.append(f'{"all":>4}  {succ:>4}/{trials:<6}  {overall*100:5.1f}%   '
                     f'(mean over {len(rates)} tasks: {task_mean*100:.1f}%)')
        summary[suite] = {
            'successes': succ, 'trials': trials, 'success_rate': overall,
            'task_mean_success_rate': task_mean, 'num_tasks': len(rates),
            'tasks': {tid: {'description': t['description'], 'successes': t['successes'],
                            'trials': t['trials'],
                            'success_rate': t['successes'] / max(t['trials'], 1)}
                      for tid, t in sorted(tasks.items())},
        }
    text = '\n'.join(lines)
    print(text)
    if os.path.isdir(root):
        with open(os.path.join(root, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(root, 'summary.md'), 'w') as f:
            f.write('```' + text + '\n```\n')
        print(f'\nwrote {os.path.join(root, "summary.json")} and summary.md')


if __name__ == '__main__':
    main()
