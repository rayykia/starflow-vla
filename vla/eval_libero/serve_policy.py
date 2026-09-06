#!/usr/bin/env python3
"""STARFlow-VLA LIBERO policy server (WebSocket, openpi_client wire protocol).

Adapted from SimVLANF/evaluation/libero/serve_smolvlm_libero.py. What changed:
1. Loads a STARFlow-VLA checkpoint (.pth) + YAML config through the vla/ helpers.
   The checkpoint contains the SmolVLM condition encoder, so nothing else is loaded.
2. Observation = agentview + wrist images (128x128 RGB, rotated 180 deg by the
   client, exactly like the training data) + the 8-D proprio state
   [eef_pos(3), axis-angle(3), gripper_qpos(2)], which matches the HDF5
   [ee_pos, ee_ori, gripper_states] layout used in training. Both views are
   VAE-encoded into the video prefix (and generated); SmolVLM sees the agentview.
3. The sampled action chunk is score-denoised (one Tweedie step) with the
   *training* noise level `action_noise_std` from the config, instead of a
   hard-coded std (see vla/sample_libero.py::score_denoise).
4. Requests from all connected clients are batched into one sampling pass:
   AR flow sampling is latency-bound (~3.4 s for batch 1, ~3.8 s for batch 16
   on an A100), so N parallel LIBERO clients cost about the same as one.
5. Optional dumps of the model's predicted future video (debugging).

Run from the repo root in the `nfvla` env (PYTHONPATH= avoids the stray ROS install):
  PYTHONPATH= CUDA_VISIBLE_DEVICES=0 python vla/eval_libero/serve_policy.py \
      --checkpoint_path logs/libero_model_vla_1024_6_h8.pth --port 8000
Any unknown `--flag value` is forwarded as a config override (e.g. --action_noise_std 0.05).

Request  (msgpack, openpi_client.msgpack_numpy encoding):
  {"observation/image": uint8 (H, W, 3), "observation/wrist_image": uint8 (H, W, 3),
   "observation/state": float (8,), "prompt": str}
Response: {"actions": float32 (action_horizon, action_dim)}  -- un-normalized LIBERO actions.
"""
import argparse
import asyncio
import concurrent.futures
import functools
import logging
import pathlib
import re
import sys
import time
import traceback

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import utils  # noqa: E402
from vla.dataset_libero import load_norm_stats  # noqa: E402
from vla.sample_libero import predict_batch, save_video_grid  # noqa: E402
from vla.train_libero import load_vla_config  # noqa: E402
from vla.transformer_flow_vla import setup_vla_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('serve_policy')


# -----------------------------------------------------------------------------
# msgpack <-> numpy, identical to openpi_client.msgpack_numpy (which the LIBERO
# client uses), so we do not depend on the openpi package on the server side.
# -----------------------------------------------------------------------------
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')
    if isinstance(obj, np.ndarray):
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str,
                b'shape': obj.shape}
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------
class StarFlowVLAPolicy:
    """Owns the frozen encoders + model and turns a list of raw requests into actions."""

    def __init__(self, cli, config_overrides):
        self.cli = cli
        args = load_vla_config(cli.model_config_path, config_overrides)
        if cli.norm_stats:
            args.norm_stats = cli.norm_stats
        self.args = args
        self.pixel_size = args.img_size          # before the latent-size rewrite below
        cfg = args.cfg if cli.cfg is None else cli.cfg
        self.guidance = float(cfg[0] if isinstance(cfg, (list, tuple)) else cfg)  # `--cfg` is nargs='+'

        dist = utils.Distributed()
        self.device = torch.device('cuda')
        utils.set_random_seed(cli.seed)
        self.vae = utils.setup_vae(args, dist, self.device)
        args.img_size = args.img_size // self.vae.downsample_factor

        self.model = setup_vla_model(args).to(self.device)
        logger.info(f'Loading checkpoint {cli.checkpoint_path}')
        state = torch.load(cli.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.norm_stats = load_norm_stats(args.norm_stats)
        self.proprio_dim = int(args.proprio_dim)

        self.n_calls = 0
        self.n_samples = 0
        self.dump_dir = pathlib.Path(cli.dump_dir) if cli.dump_dir else None
        if self.dump_dir:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f'Model ready: action_horizon={args.action_horizon} action_dim={args.action_dim} '
            f'image={self.pixel_size}px cfg={self.guidance} '
            f'denoise={bool(cli.denoise)} (action_noise_std={args.action_noise_std}, '
            f'strength={cli.denoise_strength}) denoise_video={bool(cli.denoise_video)} '
            f'(noise_std={args.noise_std}) norm_stats={args.norm_stats}')

    @property
    def metadata(self):
        return {
            'model': 'STARFlow-VLA',
            'checkpoint': str(self.cli.checkpoint_path),
            'action_dim': int(self.args.action_dim),
            'action_horizon': int(self.args.action_horizon),
            'image_size': int(self.pixel_size),
            'proprio_dim': self.proprio_dim,
            'vlm': str(self.args.vlm),
            'cfg': float(self.guidance),
            'denoise': bool(self.cli.denoise),
            'action_noise_std': float(self.args.action_noise_std),
        }

    def _prep_image(self, img) -> np.ndarray:
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        if img.shape[:2] != (self.pixel_size, self.pixel_size):
            img = np.asarray(Image.fromarray(img).resize(
                (self.pixel_size, self.pixel_size), Image.BILINEAR))
        return np.ascontiguousarray(img[..., :3])

    def _prep_state(self, state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.proprio_dim:
            raise ValueError(f'observation/state must have {self.proprio_dim} dims, got {state.shape[0]}')
        return state

    def infer(self, requests: list) -> list:
        """requests: list of decoded obs dicts -> list of {'actions': (H, Da) float32}."""
        t0 = time.time()
        imgs = np.stack([np.stack([self._prep_image(r['observation/image']),
                                   self._prep_image(r['observation/wrist_image'])])
                         for r in requests])                                      # (n, 2, H, W, 3)
        states = np.stack([self._prep_state(r['observation/state']) for r in requests])
        prompts = [str(r.get('prompt', '')) for r in requests]
        obs = torch.from_numpy(imgs).permute(0, 1, 4, 2, 3).float() / 127.5 - 1.0  # (n, 2, 3, H, W)
        proprio = torch.from_numpy(states)                                        # (n, 8) raw

        dump = self.dump_dir is not None and self.n_calls % self.cli.dump_every == 0
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = predict_batch(
                self.model, self.vae, self.args, self.norm_stats, obs, prompts, proprio,
                guidance=self.guidance,
                denoise_actions=bool(self.cli.denoise),
                denoise_video=bool(self.cli.denoise_video),
                denoise_strength=self.cli.denoise_strength, decode_video=dump)
        actions = out['actions'].numpy().astype(np.float32)   # (n, H, Da)

        if dump:
            self._dump(out, imgs[0, 0], prompts[0])
        self.n_calls += 1
        self.n_samples += len(requests)
        dt = time.time() - t0
        if self.n_calls % self.cli.log_every == 0 or self.n_calls <= 3:
            logger.info(f'call {self.n_calls}: batch={len(requests)} {dt:.2f}s '
                        f'({dt / len(requests):.2f}s/sample) total_samples={self.n_samples} '
                        f'prompt[0]="{prompts[0]}"')
        return [{'actions': actions[i]} for i in range(len(requests))]

    def _dump(self, out, img0, prompt):
        slug = re.sub(r'[^a-z0-9]+', '_', prompt.lower())[:40]
        stem = self.dump_dir / f'call{self.n_calls:06d}_{slug}'
        try:
            save_video_grid(out['video'][:1], f'{stem}.mp4', fps=10)   # obs + H frames, views tiled
            Image.fromarray(img0).save(f'{stem}_obs.png')
            np.save(f'{stem}_actions.npy', out['actions'][0].numpy())
        except Exception as exc:  # dumps are best-effort
            logger.warning(f'dump failed: {exc}')


# -----------------------------------------------------------------------------
# Batching server
# -----------------------------------------------------------------------------
class BatchedServer:
    """Collects concurrent requests into batches; one worker thread runs the GPU.

    A batch is closed as soon as every connected client has a request queued
    (or max_batch is reached), otherwise after `wait_ms`. Waiting for all
    clients keeps them phase-locked: without it, clients that step their
    envs while a batch is running arrive one by one and every batch degrades
    to size 1. `wait_ms` bounds the cost of clients that are between episodes
    (env reset, video writing).
    """

    def __init__(self, policy: StarFlowVLAPolicy, max_batch: int, wait_ms: float):
        self.policy = policy
        self.max_batch = max_batch
        self.wait_s = wait_ms / 1000.0
        self.queue: asyncio.Queue = asyncio.Queue()
        self.n_connections = 0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def submit(self, request: dict) -> dict:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((request, fut))
        return await fut

    async def worker(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            deadline = loop.time() + self.wait_s
            while len(items) < min(self.max_batch, max(self.n_connections, 1)):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            try:
                results = await loop.run_in_executor(
                    self.executor, self.policy.infer, [req for req, _ in items])
                for (_, fut), res in zip(items, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as exc:
                logger.error(f'Inference error: {exc}')
                traceback.print_exc()
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(exc)

    async def handle_connection(self, websocket):
        peer = websocket.remote_address
        self.n_connections += 1
        logger.info(f'Connection from {peer} opened ({self.n_connections} active)')
        try:
            await websocket.send(packb(self.policy.metadata))
            async for message in websocket:
                try:
                    request = unpackb(message)
                    if not isinstance(request, dict) or 'observation/image' not in request \
                            or 'observation/wrist_image' not in request \
                            or 'observation/state' not in request:
                        raise KeyError("request needs 'observation/image' and "
                                       "'observation/wrist_image' (uint8 HxWx3), "
                                       "'observation/state' (8,) and 'prompt'")
                    result = await self.submit(request)
                    await websocket.send(packb(result))
                except Exception as exc:
                    # a str reply makes openpi's WebsocketClientPolicy raise RuntimeError
                    await websocket.send(f'Error: {exc}')
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.n_connections -= 1
            logger.info(f'Connection from {peer} closed ({self.n_connections} active)')

    async def serve(self, host: str, port: int):
        asyncio.create_task(self.worker())
        async with websockets.asyncio.server.serve(
                self.handle_connection, host, port, max_size=None, compression=None,
                ping_interval=20, ping_timeout=600):   # long inference batches must not drop pings
            logger.info(f'STARFlow-VLA policy server listening on ws://{host}:{port}')
            await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description='STARFlow-VLA LIBERO policy server')
    parser.add_argument('--model_config_path', default='configs/starflow_vla_libero_128.yaml')
    parser.add_argument('--checkpoint_path', required=True, help='model .pth state dict')
    parser.add_argument('--norm_stats', default=None, help='override the config norm_stats path')
    parser.add_argument('--cfg', type=float, default=None,
                        help='classifier-free guidance scale (default: `cfg` from the config)')
    parser.add_argument('--denoise', type=int, default=1,
                        help='score-based denoising of the action chunk with action_noise_std (1/0)')
    parser.add_argument('--denoise_strength', type=float, default=1.0,
                        help='multiplier on the sigma^2 Tweedie step (1.0 = exact)')
    parser.add_argument('--denoise_video', type=int, default=0,
                        help='also denoise dumped videos with noise_std (dumps only)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--max_batch', type=int, default=32,
                        help='max concurrent requests fused into one sampling pass')
    parser.add_argument('--batch_wait_ms', type=float, default=500.0,
                        help='max wait for the remaining connected clients before running a batch')
    parser.add_argument('--dump_dir', default=None,
                        help='save predicted video/obs/actions of every --dump_every-th call here')
    parser.add_argument('--dump_every', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=20)
    cli, overrides = parser.parse_known_args()
    if overrides:
        logger.info(f'Config overrides: {overrides}')

    policy = StarFlowVLAPolicy(cli, overrides)
    server = BatchedServer(policy, cli.max_batch, cli.batch_wait_ms)
    asyncio.run(server.serve(cli.host, cli.port))


if __name__ == '__main__':
    main()
