#!/usr/bin/env python3
"""
STARFlow-VLA LIBERO Evaluation Client

Adapted from SimVLANF/evaluation/libero/libero_client.py. Observation format
matches vla/dataset_libero.py (the training data):
1. Images: agentview + wrist (eye-in-hand), rendered at 128x128 (LIBERO's native
   dataset resolution), rotated 180 degrees (LIBERO stores/renders upside-down).
2. State: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D, sent for
   completeness (the current model does not condition on it).
3. Action: 7D delta action chunk (action_horizon=8) from the server; the first
   `--replan_steps` (default: the whole chunk) are executed before re-planning.
4. Sharding: `--task_ids 0 3 7` restricts the tasks, `--shard k/P` runs every
   P-th (task, trial) episode, so many client processes can share one batching
   server (see run_eval_all.sh) and stay load-balanced.
5. Per-(task, shard) results are written as JSON (merge: summarize_results.py).

Runs in the `libero` conda env (needs: libero, openpi_client, imageio, tqdm).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional

import imageio
import numpy as np
from tqdm import tqdm

from openpi_client import websocket_client_policy as ws_client

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 128   # training frames are the native 128x128 LIBERO renders

# Max steps per task suite (based on longest demo + buffer)
MAX_STEPS = {
    "libero_spatial": 800,   # longest demo: 193
    "libero_object": 800,    # longest demo: 254
    "libero_goal": 800,      # longest demo: 270
    "libero_10": 900,        # longest demo: 505
    "libero_90": 900,        # longest demo: 373
}

NUM_STEPS_WAIT = 10  # Wait for objects to stabilize

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to axis-angle representation.

    Uses the same convention as robosuite for consistency with training data.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# -----------------------------------------------------------------------------
# Client Policy
# -----------------------------------------------------------------------------
class WebSocketClient:
    """
    WebSocket client for the STARFlow-VLA server (vla/eval_libero/serve_policy.py).

    Requires: pip install openpi-client
    """
    def __init__(self, host: str, port: int, replan_steps: Optional[int] = None,
                 connect_timeout_s: float = 3600.0):
        # openpi_client 0.1.0 does not retry: wait here until the server is up
        t0 = time.time()
        while True:
            try:
                self.client = ws_client.WebsocketClientPolicy(host, port)
                break
            except (ConnectionRefusedError, OSError) as e:
                if time.time() - t0 > connect_timeout_s:
                    raise
                print(f"Waiting for server at ws://{host}:{port} ({e})...", flush=True)
                time.sleep(5.0)
        self.metadata = self.client.get_server_metadata()
        horizon = int(self.metadata.get("action_horizon", 0)) or None
        self.replan_steps = replan_steps or horizon
        if horizon is not None:
            assert self.replan_steps <= horizon, \
                f"replan_steps={self.replan_steps} > server action_horizon={horizon}"
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            # Build observation dict (images are already 128x128 uint8, rotated)
            element = {
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state": obs["state"],
                "prompt": goal,
            }

            # Query server
            result = self.client.infer(element)
            action_chunk = result["actions"]

            # Ensure numpy array
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)

            assert len(action_chunk) >= self.replan_steps, \
                f"Need {self.replan_steps} steps but got {len(action_chunk)}"

            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        return self.action_plan.popleft()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int):
    """Initialize a LIBERO environment."""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(
    client,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
    task_ids: Optional[List[int]] = None,
    shard: tuple = (0, 1),
    env_resolution: int = LIBERO_ENV_RESOLUTION,
    max_steps: Optional[int] = None,
    results_dir: Optional[str] = None,
) -> float:
    """
    Run LIBERO evaluation across (a subset of) the tasks in a suite.

    shard=(k, P): only episodes with (task_id * num_trials + ep) % P == k are run.
    """
    np.random.seed(seed)

    # Initialize task suite
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = max_steps or MAX_STEPS.get(task_suite_name, 400)
    task_ids = list(range(num_tasks)) if task_ids is None else [t for t in task_ids if 0 <= t < num_tasks]
    shard_k, shard_p = shard

    Path(video_out_path).mkdir(parents=True, exist_ok=True)
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Task suite: {task_suite_name}")
    print(f"   Tasks: {task_ids} ({len(task_ids)}/{num_tasks}), Trials per task: {num_trials}, "
          f"shard {shard_k}/{shard_p}")
    print(f"   Max steps: {max_steps}, env resolution: {env_resolution}, replan steps: {client.replan_steps}")

    total_episodes, total_successes = 0, 0

    for task_id in tqdm(task_ids, desc="Tasks"):
        my_eps = [ep for ep in range(num_trials) if (task_id * num_trials + ep) % shard_p == shard_k]
        if not my_eps:
            continue
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, env_resolution, seed)

        task_successes = 0
        episodes = []
        task_t0 = time.time()
        for ep in tqdm(my_eps, desc=f"{task_description[:30]}...", leave=False):
            # Reset
            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])

            replay_images = []
            t = 0
            done = False
            ep_t0 = time.time()

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    # Wait for objects to stabilize
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get images (rotated 180 degrees, as in the training data)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    if save_video:
                        replay_images.append(img)

                    # Build state vector
                    # [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ]).astype(np.float32)

                    # Pack observation
                    obs_dict = {
                        "image": img,
                        "wrist_image": wrist_img,
                        "state": state,
                    }

                    # Get action (7D delta action)
                    action = client.step(obs_dict, task_description)

                    # Execute (send delta action directly)
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    print(f"Error in rollout: {e}")
                    break

            total_episodes += 1
            episodes.append({"episode": ep, "success": bool(done), "steps": t,
                             "seconds": round(time.time() - ep_t0, 1)})

            # Save video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")[:50]
            video_path = Path(video_out_path) / f"task{task_id:02d}_{task_segment}_ep{ep}_{suffix}.mp4"
            if replay_images and save_video:
                imageio.mimwrite(str(video_path), replay_images, fps=10)

            # Print episode result
            status_icon = "[OK]" if done else "[FAIL]"
            print(f"  {status_icon} Task {task_id} Ep {ep}: {suffix.upper()} (steps={t}, "
                  f"{time.time() - ep_t0:.0f}s)", flush=True)

        env.close()
        print(f"   Task {task_id}: {task_successes}/{len(my_eps)} ({task_successes/len(my_eps)*100:.1f}%)", flush=True)
        if results_dir:
            with open(Path(results_dir) / f"task{task_id:02d}_shard{shard_k}of{shard_p}.json", "w") as f:
                json.dump({
                    "task_suite": task_suite_name, "task_id": task_id,
                    "task_description": task_description,
                    "shard": f"{shard_k}/{shard_p}",
                    "num_trials": len(my_eps), "successes": task_successes,
                    "success_rate": task_successes / len(my_eps),
                    "seconds": round(time.time() - task_t0, 1),
                    "replan_steps": client.replan_steps, "max_steps": max_steps,
                    "seed": seed, "server": getattr(client, "metadata", {}),
                    "episodes": episodes,
                }, f, indent=2)

    success_rate = total_successes / max(total_episodes, 1)
    print(f"\nTotal success rate: {total_successes}/{total_episodes} ({success_rate*100:.1f}%)")

    return success_rate


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("STARFlow-VLA LIBERO Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connection_info", type=str, default=None,
                        help="Path to server connection info JSON")
    parser.add_argument("--task_suite", type=str, default="libero_10",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="Subset of task ids to run (default: all)")
    parser.add_argument("--shard", type=str, default="0/1",
                        help="k/P: run every P-th (task, trial) episode, offset k (load-balanced sharding)")
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=None,
                        help="Actions executed per server call (default: the server's action_horizon)")
    parser.add_argument("--env_resolution", type=int, default=LIBERO_ENV_RESOLUTION)
    parser.add_argument("--max_steps", type=int, default=None, help="Override the per-suite step budget")
    parser.add_argument("--video_out", type=str, default="./eval_results")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Per-task JSON results (default: <video_out>/results/<suite>)")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording for faster evaluation")

    args = parser.parse_args()

    # Load connection info
    if args.connection_info:
        print(f"Loading connection info from: {args.connection_info}")
        while not Path(args.connection_info).exists():
            sys.stdout.write("\rWaiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
        print()
        with open(args.connection_info) as f:
            info = json.load(f)
            args.host = info["host"]
            args.port = info["port"]

    shard_k, shard_p = (int(v) for v in args.shard.split("/"))
    assert 0 <= shard_k < shard_p, f"bad --shard {args.shard}"

    print(f"Starting LIBERO evaluation client")
    print(f"   Server: ws://{args.host}:{args.port}")
    print(f"   Task suite: {args.task_suite}")
    print(f"   Task ids: {args.task_ids if args.task_ids is not None else 'all'}, shard {shard_k}/{shard_p}")
    print()

    # Initialize client (blocks until the server is up)
    client = WebSocketClient(args.host, args.port, replan_steps=args.replan_steps)
    print(f"   Server metadata: {client.metadata}")
    print(f"   Replan steps: {client.replan_steps}")
    print()

    # Run evaluation
    video_path = Path(args.video_out) / args.task_suite
    results_dir = args.results_dir or str(Path(args.video_out) / "results" / args.task_suite)
    eval_libero(
        client=client,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        video_out_path=str(video_path),
        save_video=not args.no_video,
        task_ids=args.task_ids,
        shard=(shard_k, shard_p),
        env_resolution=args.env_resolution,
        max_steps=args.max_steps,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
