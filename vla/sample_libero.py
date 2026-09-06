"""STARFlow-VLA inference: obs images + instruction + proprio -> future video + action chunk.

Observations and generated videos carry both camera views (agentview, wrist) on an
explicit view axis, (n, T, V=2, 3, H, W); the flow works on their channel-concatenated
VAE latents (vla/multiview.py). SmolVLM conditions on the agentview obs only.

CLI (qualitative check against a dataset sample):
  $PY vla/sample_libero.py --model_config_path configs/starflow_vla_libero_128.yaml \
      --checkpoint_path logs/libero_model_vla_1024_6_h8.pth --sample_index 0 --cfg 1.5

The condition `y` is built by the model itself (`model.encode_condition`): SmolVLM
features of (agentview obs, instruction) projected and concatenated with a projected
proprio token. Classifier-free guidance drops the instruction only (empty string);
the null half keeps the observation and the proprio token.

Sampling happens in the model's *noisy* training space (video latents were trained
with N(0, noise_std^2) noise, normalized actions with N(0, action_noise_std^2)), so
`score_denoise` applies one Tweedie step, `x - sigma^2 * grad_x NLL(x)`, with the
training sigmas to move a sample back to clean data. This is what the LIBERO policy
server (vla/eval_libero/serve_policy.py) uses for the executed action chunk.
"""
import argparse
import pathlib
import sys

import torch
from einops import rearrange

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import utils
from misc import print, dividable
from utils import add_noise
from vla.dataset_libero import (
    LiberoVLADataset, load_norm_stats, normalize_state, unnormalize_actions,
)
from vla.multiview import encode_views, decode_views, tile_views


def save_video_grid(video_px: torch.Tensor, path, fps: int = 10):
    """Write a (B, T, 3, H, W) [-1, 1] tensor as one grid mp4; a two-view
    (B, T, V, 3, H, W) video is tiled side by side (agentview | wrist) per cell.

    Local replacement for upstream save_samples_unified's video path: the
    installed torchvision (>= 0.23) removed tv.io.write_video.
    """
    import imageio.v2 as imageio
    if video_px.dim() == 6:
        video_px = tile_views(video_px)
    video = ((video_px.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    grid = rearrange(video, '(a b) t c h w -> t (a h) (b w) c', a=dividable(video.size(0)))
    imageio.mimwrite(str(path), list(grid.numpy()), fps=fps)


@torch.no_grad()
def encode_condition(model, obs_frames, instructions, proprio, guidance):
    """-> y (n or 2n, L, C): conditional tokens, plus the null half (instruction
    dropped, obs + proprio kept) appended when guidance > 0."""
    n = obs_frames.size(0)
    images = obs_frames[:, 0, 0]                              # (n, 3, 128, 128) agentview
    if guidance > 0:                                          # one batched VLM pass
        images = torch.cat([images, images], dim=0)
        instructions = list(instructions) + [''] * n
        proprio = torch.cat([proprio, proprio], dim=0)
    return model.encode_condition(images, list(instructions), proprio)


@torch.no_grad()
def sample_latents(model, vae, args, obs_frames, instructions, proprio, guidance, device):
    """obs_frames: (n, 1, V, 3, 128, 128) in [-1, 1]; proprio: (n, Dp) normalized.

    Returns (video_lat (n, 1+H/4, V*Cv, h, w), actions (n, H, Da) normalized, y_cond
    (n, L, C)). Both samples live in the noisy training space (see module doc);
    y_cond is the conditional half of the condition tokens for `score_denoise`.
    """
    n = obs_frames.size(0)
    obs_frames, proprio = obs_frames.to(device), proprio.to(device)
    x0 = encode_views(vae, obs_frames)                        # (n, 1, V*48, 8, 8)
    x0, _ = add_noise(x0, args.noise_std, args.noise_type)

    y = encode_condition(model, obs_frames, instructions, proprio, guidance)

    kv_caches = model(x0, y, context=True)

    lat = args.img_size  # already divided by vae.downsample_factor in the callers
    t_lat = 1 + args.action_horizon // 4
    z_v = torch.randn(n, t_lat, args.channel_size, lat, lat, device=device)
    z_a = torch.randn(n, args.action_horizon, args.action_dim, device=device)
    video_lat, actions = model(z_v, y, actions=z_a, reverse=True,
                               kv_caches=kv_caches, guidance=guidance)
    return video_lat, actions, y[:n]


def score_denoise(model, video_lat, actions, y, sigma_video=0.0, sigma_action=0.0,
                  strength=1.0):
    """One-step score-based (Tweedie) denoising of a joint (video, action) sample.

    E[clean | noisy] = noisy - sigma^2 * grad_noisy NLL(noisy), where NLL is the
    model's exact joint negative log-likelihood and sigma the training noise std
    of each modality (`noise_std` for video latents, `action_noise_std` for
    normalized actions). The video is held fixed while differentiating w.r.t. the
    actions: video tokens precede action tokens in the AR sequence, so
    grad_a NLL_joint is exactly the conditional score of the chunk given the
    generated video. Pass sigma=0 to leave a modality untouched; `strength`
    scales the step (1.0 = the Tweedie estimate).
    """
    if sigma_video <= 0 and sigma_action <= 0:
        return video_lat, actions
    x = video_lat.detach().float().requires_grad_(sigma_video > 0)
    a = actions.detach().float().requires_grad_(sigma_action > 0)
    # fp32 forward: the score is cheap (one joint likelihood pass), and the
    # sigma^2-scaled step is sensitive to precision in z / logdet
    with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        z_v, z_a, _, logdets_v, logdets_a = model(x, y.float(), actions=a)
        nll = 0.5 * z_a.float().pow(2).sum() - sum(ld.float().sum() for ld in logdets_a)
        if sigma_video > 0:
            nll = nll + 0.5 * z_v.float().pow(2).sum() \
                - sum(ld.float().sum() for ld in logdets_v)
        inputs = [t for t in (x, a) if t.requires_grad]
        grads = iter(torch.autograd.grad(nll, inputs))
    if sigma_video > 0:
        video_lat = (x - strength * sigma_video ** 2 * next(grads)).detach().to(video_lat.dtype)
    if sigma_action > 0:
        actions = (a - strength * sigma_action ** 2 * next(grads)).detach().to(actions.dtype)
    return video_lat, actions


@torch.no_grad()
def generate_rollout(model, vae, args, obs_frames, instructions, proprio, guidance, device):
    """obs_frames: (n, 1, V, 3, 128, 128) in [-1, 1]; proprio (n, Dp) normalized.
    Returns pixel video + normalized actions (raw flow samples, no denoising)."""
    video_lat, actions, _ = sample_latents(
        model, vae, args, obs_frames, instructions, proprio, guidance, device)
    video_px = decode_views(vae, video_lat)                   # (n, 1+H, V, 3, 128, 128)
    return video_px, actions


@torch.no_grad()
def predict_batch(model, vae, args, norm_stats, obs_images, instructions, proprio,
                  guidance=0.0, denoise_actions=True, denoise_video=False,
                  denoise_strength=1.0, decode_video=False):
    """Batched closed-loop policy API.

    obs_images: (n, V, 3, 128, 128) in [-1, 1] (agentview, wrist); instructions: n strings;
    proprio: (n, 8) *raw* robot state [ee_pos, ee_ori(axis-angle), gripper_states]
    (normalized here with norm_stats['state']).
    Score-based denoising uses the training sigmas from `args`
    (`action_noise_std` / `noise_std`); `denoise_video` only matters when the
    decoded video is requested. Returns a dict of CPU tensors:
      actions            (n, H, Da) un-normalized (what the robot executes)
      actions_normalized (n, H, Da) in the model's z-score-normalized space
      video              (n, 1+H, V, 3, 128, 128) in [-1, 1]   (if decode_video)
    """
    device = next(model.parameters()).device
    obs = obs_images[:, None].to(device)                      # (n, 1, V, 3, 128, 128)
    proprio = normalize_state(torch.as_tensor(proprio, dtype=torch.float32), norm_stats['state'])
    video_lat, actions, y = sample_latents(
        model, vae, args, obs, list(instructions), proprio.to(device), guidance, device)
    video_lat, actions = score_denoise(
        model, video_lat, actions, y,
        sigma_video=args.noise_std if (denoise_video and decode_video) else 0.0,
        sigma_action=args.action_noise_std if denoise_actions else 0.0,
        strength=denoise_strength)
    out = {'actions_normalized': actions.float().cpu()}
    out['actions'] = unnormalize_actions(out['actions_normalized'], norm_stats['actions'])
    if decode_video:
        out['video'] = decode_views(vae, video_lat).float().cpu()
    return out


@torch.no_grad()
def predict(model, vae, args, norm_stats, obs_image, instruction, proprio,
            guidance=0.0, denoise=False):
    """Single-sample policy API. obs_image: (V, 3, 128, 128) in [-1, 1]; proprio: raw (8,).

    Returns (video (1+H, V, 3, 128, 128) in [-1, 1], un-normalized action chunk (H, Da)).
    `denoise=True` applies `score_denoise` to both modalities with the training sigmas.
    """
    out = predict_batch(model, vae, args, norm_stats, obs_image[None], [instruction],
                        torch.as_tensor(proprio, dtype=torch.float32)[None],
                        guidance=guidance, denoise_actions=denoise, denoise_video=denoise,
                        decode_video=True)
    return out['video'][0], out['actions'][0]


@torch.no_grad()
def preview_rollout(model, vae, args, dist, frames, instructions, actions, proprio,
                    sample_dir, epoch, step):
    """Training-time preview on the last batch: save rollout video (views tiled
    side by side), print action MAE.

    `proprio` is the dataset's normalized state. Returns (action MAE vs clean
    ground truth, mp4 path or None on non-zero ranks).
    """
    model.eval()
    n = min(4, frames.size(0))
    device = frames.device
    out_path = None
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        video_px, a_gen = generate_rollout(
            model, vae, args, frames[:n, :1], list(instructions[:n]), proprio[:n],
            args.preview_guidance, device)
    if dist is None or dist.local_rank == 0:
        out_path = sample_dir / f'rollout_epoch{epoch + 1:04d}.mp4'
        save_video_grid(video_px, out_path, fps=10)
        print(f'Saved preview rollout to {out_path}')
    mae = (a_gen.float() - actions[:n].float()).abs().mean().item()
    print(f'[preview] epoch {epoch + 1}: action MAE (normalized) = {mae:.4f}')
    model.train()
    return mae, out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True, type=str)
    parser.add_argument('--checkpoint_path', required=True, type=str)
    parser.add_argument('--sample_index', default=0, type=int)
    parser.add_argument('--cfg', default=1.5, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--denoise', default=1, type=int,
                        help='score-based denoising with the training noise stds (1/0)')
    cli, extra = parser.parse_known_args()

    from vla.train_libero import load_vla_config
    args = load_vla_config(cli.model_config_path, extra)  # extra: config overrides

    dist = utils.Distributed()
    device = torch.device('cuda')
    utils.set_random_seed(cli.seed)

    vae = utils.setup_vae(args, dist, device)
    args.img_size = args.img_size // vae.downsample_factor

    from vla.transformer_flow_vla import setup_vla_model
    model = setup_vla_model(args).to(device)
    model.load_state_dict(torch.load(cli.checkpoint_path, map_location='cpu'), strict=True)
    model.eval().requires_grad_(False)

    norm_stats = load_norm_stats(args.norm_stats)
    # raw (un-normalized) samples: predict() normalizes the proprio itself
    ds = LiberoVLADataset(args.libero_root, args.libero_subsets, horizon=args.action_horizon)
    frames, instruction, gt_actions_raw, proprio_raw = ds[cli.sample_index]

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        video, action_chunk = predict(
            model, vae, args, norm_stats, frames[0], instruction, proprio_raw,
            guidance=cli.cfg, denoise=bool(cli.denoise))

    out_dir = pathlib.Path(args.logdir) / 'vla_samples'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_video_grid(video[None], out_dir / f'predict_{cli.sample_index:03d}.mp4', fps=10)
    gt = gt_actions_raw
    print(f'instruction : {instruction}   (denoise={bool(cli.denoise)}, cfg={cli.cfg})')
    print(f'proprio     : {proprio_raw.numpy().round(3)}')
    print(f'predicted   :\n{action_chunk.numpy().round(3)}')
    print(f'ground truth:\n{gt.numpy().round(3)}')
    print(f'MAE (un-normalized): {(action_chunk - gt).abs().mean().item():.4f}')


if __name__ == '__main__':
    main()
