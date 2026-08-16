"""STARFlow-VLA inference: obs image + instruction -> future video + action chunk.

CLI (qualitative check against a dataset sample):
  $PY vla/sample_libero.py --model_config_path configs/starflow_vla_libero_128.yaml \
      --checkpoint_path logs/libero_model_vla_1024_6_h8.pth --sample_index 0 --cfg 1.5
"""
import argparse
import pathlib
import sys

import torch
from einops import rearrange

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import utils
from misc import print, dividable
from utils import encode_text, add_noise
from vla.dataset_libero import LiberoVLADataset, load_action_norm_stats, unnormalize_actions


def save_video_grid(video_px: torch.Tensor, path, fps: int = 10):
    """Write a (B, T, 3, H, W) [-1, 1] tensor as one grid mp4.

    Local replacement for upstream save_samples_unified's video path: the
    installed torchvision (>= 0.23) removed tv.io.write_video.
    """
    import imageio.v2 as imageio
    video = ((video_px.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    grid = rearrange(video, '(a b) t c h w -> t (a h) (b w) c', a=dividable(video.size(0)))
    imageio.mimwrite(str(path), list(grid.numpy()), fps=fps)


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
    if dist is None or dist.local_rank == 0:
        out_path = sample_dir / f'rollout_epoch{epoch + 1:04d}.mp4'
        save_video_grid(video_px, out_path, fps=10)
        print(f'Saved preview rollout to {out_path}')
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
    out_dir.mkdir(parents=True, exist_ok=True)
    save_video_grid(video[None], out_dir / f'predict_{cli.sample_index:03d}.mp4', fps=10)
    print(f'instruction : {instruction}')
    print(f'predicted   :\n{action_chunk.numpy().round(3)}')
    print(f'ground truth:\n{unnormalize_actions(gt_actions, norm_stats).numpy().round(3)}')


if __name__ == '__main__':
    main()
