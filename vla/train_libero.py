"""STARFlow-VLA training on LIBERO.

Usage:
  $PY vla/train_libero.py --model_config_path configs/starflow_vla_libero_128.yaml
  (multi-GPU: torchrun --nproc_per_node=N vla/train_libero.py --model_config_path ...)
"""
import argparse
import contextlib
import os
import pathlib
import sys
import time

import torch
import torch.amp
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import utils
from misc import print  # local_rank-0 print
from utils import drop_label, add_noise
from vla.transformer_flow_vla import setup_vla_model
from vla.dataset_libero import create_libero_dataloader
from vla.multiview import encode_views

def load_env_file(path) -> None:
    """Load KEY=VALUE lines from a dotenv-style file (existing env vars win)."""
    path = pathlib.Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def get_vla_parser():
    from train import get_tarflow_parser
    parser = get_tarflow_parser()
    parser.add_argument('--libero_root', default='/home/rayykia/Projects/LIBERO/libero/datasets', type=str)
    parser.add_argument('--libero_subsets', default=['libero_10'], type=str, nargs='+')
    parser.add_argument('--norm_stats', default='vla/norm_stats/libero_10.json', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--action_horizon', default=8, type=int)
    parser.add_argument('--action_dim', default=7, type=int)
    parser.add_argument('--action_channels', default=256, type=int)
    parser.add_argument('--action_head_dim', default=64, type=int)
    parser.add_argument('--action_layers', default=[2, 2], type=int, nargs='+')
    parser.add_argument('--action_loss_weight', default=1.0, type=float)
    parser.add_argument('--action_noise_std', default=0.1, type=float)
    parser.add_argument('--preview_guidance', default=1.5, type=float)
    # condition encoder (SmolVLM over instruction + current observation, + proprio token)
    parser.add_argument('--vlm', default='HuggingFaceTB/SmolVLM-500M-Instruct', type=str,
                        help='HF id / path of the SmolVLM (Idefics3-style) backbone')
    parser.add_argument('--vlm_image_size', default=512, type=int,
                        help='obs is resized to this for the SigLIP tower (512 -> 64 tokens)')
    parser.add_argument('--vlm_freeze', default=0, type=int,
                        help='1 = frozen backbone (no gradients); 0 = fine-tune with --vlm_lr')
    parser.add_argument('--vlm_lr', default=2e-5, type=float,
                        help='peak LR of the VLM parameter group (the flow uses --lr)')
    parser.add_argument('--vlm_freeze_steps', default=1000, type=int,
                        help='hold the VLM LR at 0 for the first N optimizer steps')
    parser.add_argument('--proprio_dim', default=8, type=int,
                        help='[ee_pos(3), ee_ori(3), gripper_states(2)]')
    parser.add_argument('--grad_skip_factor', default=10.0, type=float,
                        help='skip an update when pre-clip grad_norm > grad_clip * this')
    parser.add_argument('--wandb_env_file', default='configs/wandb.env', type=str,
                        help='dotenv file with WANDB_API_KEY/WANDB_PROJECT/WANDB_ENTITY')
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='overrides $WANDB_PROJECT (default: starflow-vla)')
    parser.add_argument('--wandb_entity', default=None, type=str, help='overrides $WANDB_ENTITY')
    parser.add_argument('--wandb_id', default=None, type=str,
                        help='fixed run id; pass the same id to resume a run')
    parser.add_argument('--no_wandb', default=0, type=int, help='disable wandb logging')
    parser.add_argument('--log_every', default=10, type=int, help='optimizer steps between logs')
    return parser


def load_vla_config(config_path: str, extra_args=()) -> argparse.Namespace:
    """Parse the YAML `arguments` list; `extra_args` (CLI-style tokens) win over the YAML."""
    with open(config_path) as f:
        model_configs = yaml.safe_load(f)
    arg_str = ''
    for conf in model_configs['arguments']:
        for key in conf:
            arg_str += f'--{key} {conf[key]} '
    return get_vla_parser().parse_args(arg_str.split() + list(extra_args))


class GroupCosineLRSchedule(utils.CosineLRSchedule):
    """Upstream cosine schedule, applied per parameter group.

    Each group may carry `lr_scale` (its peak LR relative to `lr`) and
    `freeze_steps` (LR held at 0 while the step counter is below it). Both live
    in the optimizer's param_groups, so they are checkpointed with it.
    """

    def set_lr(self, lr: float) -> float:
        if self.min_lr <= lr <= self.max_lr:
            step = int(self.counter.item())
            for pg in self.optimizer.param_groups:
                scale = 0.0 if step < pg.get('freeze_steps', 0) else pg.get('lr_scale', 1.0)
                pg['lr'] = lr * scale
        return lr


def build_param_groups(model, args):
    """[flow group, VLM group]; frozen params are excluded from every group."""
    vlm_ids = set(map(id, model.vlm.parameters())) if model.vlm is not None else set()
    vlm_params = [p for p in model.parameters() if id(p) in vlm_ids and p.requires_grad]
    flow_params = [p for p in model.parameters() if id(p) not in vlm_ids and p.requires_grad]
    groups = [{'name': 'flow', 'params': flow_params, 'lr_scale': 1.0}]
    if vlm_params:
        groups.append({'name': 'vlm', 'params': vlm_params,
                       'lr_scale': args.vlm_lr / args.lr, 'freeze_steps': args.vlm_freeze_steps})
    return groups, flow_params + vlm_params


def merge_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, 'model_config_path', None):
        return args
    provided = {a[2:].split('=', 1)[0].replace('-', '_') for a in sys.argv[1:] if a.startswith('--')}
    merged = vars(load_vla_config(args.model_config_path))
    for k, v in vars(args).items():
        if k in provided:
            merged[k] = v
    return argparse.Namespace(**merged)


def main(args):
    args = merge_cli_args(args)
    assert args.action_horizon % 4 == 0, 'Wan2.2 needs action_horizon % 4 == 0'

    dist = utils.Distributed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = args.train_seed if args.train_seed is not None else time.time_ns() % 2**32
    utils.set_random_seed(seed + dist.rank)

    load_env_file(args.wandb_env_file)
    use_wandb = bool(dist.rank == 0 and not args.no_wandb and os.environ.get('WANDB_API_KEY'))
    if use_wandb:
        import wandb
        wandb.login(key=os.environ['WANDB_API_KEY'])
        run = wandb.init(
            project=args.wandb_project or os.environ.get('WANDB_PROJECT') or 'starflow-vla',
            entity=args.wandb_entity or os.environ.get('WANDB_ENTITY') or None,
            name=args.wandb_name or f'libero-{"+".join(args.libero_subsets)}',
            id=args.wandb_id, resume='allow' if args.wandb_id else None,
            config={**vars(args), 'world_size': dist.world_size,
                    'per_gpu_batch_size': args.batch_size // dist.world_size // max(args.acc, 1)},
        )
        print(f'wandb: logging to {run.url}')
    elif dist.rank == 0:
        print('wandb: disabled (no WANDB_API_KEY found, or --no_wandb 1)')

    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:32s}: {v}')

    # data
    data_loader = create_libero_dataloader(args, dist)
    num_batches = len(data_loader)
    print(f'{num_batches} batches/epoch, {len(data_loader.dataset):,} samples')

    # frozen VAE
    vae = utils.setup_vae(args, dist, device)
    vae.requires_grad_(False)
    args.img_size = args.img_size // vae.downsample_factor

    # model (owns the SmolVLM condition encoder: frozen or fine-tuned per --vlm_freeze)
    model = setup_vla_model(args).to(device)
    if dist.local_rank == 0:
        import torchinfo
        torchinfo.summary(model, depth=2)
        n_vlm = sum(p.numel() for p in model.vlm.parameters())
        n_vlm_train = sum(p.numel() for p in model.vlm.parameters() if p.requires_grad)
        n_flow = sum(p.numel() for p in model.parameters()) - n_vlm
        vlm_note = 'frozen' if n_vlm_train == 0 else (
            f'{n_vlm_train / 1e6:.1f}M trainable, lr {args.vlm_lr:g} after {args.vlm_freeze_steps} steps')
        print(f'params: flow {n_flow / 1e6:.1f}M trainable, VLM {n_vlm / 1e6:.1f}M ({vlm_note})')

    grad_accum = max(args.acc, 1)
    model_name = f'vla_{args.channels}_{len(args.layers_per_block)}_h{args.action_horizon}'
    ckpt_file = args.logdir / f'libero_model_{model_name}.pth'
    opt_ckpt_file = args.logdir / f'libero_opt_{model_name}.pth'
    sample_dir = args.logdir / f'libero_samples_{model_name}'
    if dist.local_rank == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_path:
        print(f'Loading checkpoint: {args.resume_path}')
        model.load_state_dict(torch.load(args.resume_path, map_location='cpu'), strict=True)
        epoch_start = args.resume_epoch or 0
    else:
        epoch_start = 0

    model, model_ddp = utils.parallelize_model(args, model, dist, device)
    param_groups, trainable = build_param_groups(model, args)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    warmup = args.warmup_steps if args.warmup_steps is not None else num_batches
    lr_schedule = GroupCosineLRSchedule(optimizer, warmup, args.epochs * num_batches,
                                        args.min_lr, args.lr)

    # resume optimizer/LR-schedule state (if present) alongside the model weights
    opt_state_loaded = False
    if args.resume_path:
        if opt_ckpt_file.exists():
            print(f'Loading optimizer/LR-schedule checkpoint: {opt_ckpt_file}')
            opt_state = torch.load(opt_ckpt_file, map_location='cpu')
            optimizer.load_state_dict(opt_state['optimizer'])
            lr_schedule.load_state_dict(opt_state['lr_schedule'])
            opt_state_loaded = True
        else:
            print(f'No optimizer checkpoint found at {opt_ckpt_file}, starting fresh optimizer/LR state')
    # loaded lr_schedule state already encodes progress; only bump the counter
    # from --resume_epoch when we had no opt checkpoint to load it from
    if not opt_state_loaded:
        lr_schedule.counter += epoch_start * num_batches
    scaler = torch.amp.GradScaler() if args.loss_scaling else None

    print(f'{" Training ":-^80}')
    total_steps = epoch_start * num_batches
    for epoch in range(epoch_start, args.epochs):
        if hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)
        metrics = utils.Metrics()
        epoch_t0 = last_log_t = time.time()
        last_log_step = total_steps
        for it, (frames, instructions, actions, proprio) in enumerate(data_loader):
            frames = frames.to(device, non_blocking=True)          # (B, 1+H, V, 3, 128, 128)
            actions = actions.to(device, non_blocking=True)
            proprio = proprio.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    # each view encoded as its own clip, latents channel-concatenated
                    x = encode_views(vae, frames)                # (B, 1+H/4, V*48, 8, 8)
                assert x.size(2) == args.channel_size, (
                    f'channel_size={args.channel_size} but the fused latents have {x.size(2)} '
                    f'channels ({frames.size(2)} views x {x.size(2) // frames.size(2)})')
                x, _ = add_noise(x, args.noise_std, args.noise_type)
                actions_clean = actions
                actions = actions_clean + args.action_noise_std * torch.randn_like(actions_clean)
                # condition = SmolVLM(agentview obs, instruction) + proprio token, encoded
                # inside the (DDP) forward; text dropout for CFG keeps obs + proprio.
                # The wrist obs reaches the flow through the video prefix (frame 0).
                cond = dict(images=frames[:, 0, 0], proprio=proprio,
                            instructions=drop_label(list(instructions), args.drop_label))

                needs_update = (it + 1) % grad_accum == 0
                if it % grad_accum == 0:
                    optimizer.zero_grad()

                z_v, z_a, outputs, logdets_v, logdets_a = model_ddp(x, actions=actions, cond=cond)
                loss_dict = model.get_loss(z_v, logdets_v, z_a, logdets_a)
                loss = loss_dict['loss'] / grad_accum

                if dist.gather_concat(loss.detach().view(1)).isnan().any():
                    print('nan detected, skipping step')
                    continue

                with utils.sync_ctx(model_ddp, sync=needs_update) if grad_accum > 1 \
                        else contextlib.nullcontext():
                    (scaler.scale(loss) if scaler else loss).backward()

                if needs_update:
                    grad_norm, skip = None, False
                    if args.grad_clip > 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                        # threshold is a MULTIPLE of grad_clip: this model's healthy
                        # pre-clip norm is ~5.5, so skipping at grad_clip itself
                        # (upstream's rule) rejects every update.
                        skip_threshold = args.grad_clip * args.grad_skip_factor
                        skip = args.grad_skip and total_steps > 100 and \
                            grad_norm.item() > skip_threshold
                    if skip:
                        # unlike upstream, actually skip the step -- zeroing the grads
                        # and stepping anyway still applies AdamW decay + momentum
                        print(f'skipping update, grad_norm={grad_norm.item():.3f} '
                              f'> {skip_threshold:.3f}')
                        optimizer.zero_grad()
                        if scaler:
                            scaler.update()
                    elif scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    current_lr = lr_schedule.step()
                    total_steps += 1
                    if not skip:
                        metrics.update(loss_dict)

                    if (it // grad_accum) % args.log_every == args.log_every - 1:
                        now = time.time()
                        steps_per_sec = (total_steps - last_log_step) / max(now - last_log_t, 1e-6)
                        last_log_t, last_log_step = now, total_steps
                        print(f'epoch {epoch + 1}/{args.epochs}  {total_steps:,} steps - ' + '  '.join(
                            f'{k}: {v:.4f}' for k, v in loss_dict.items())
                            + f'  ({steps_per_sec:.2f} it/s)')
                        if use_wandb:
                            import wandb
                            log = {f'train/{k}': v.item() if torch.is_tensor(v) else v
                                   for k, v in loss_dict.items()}
                            log.update({'lr': current_lr, 'epoch': epoch + it / num_batches,
                                        'lr_vlm': next((g['lr'] for g in optimizer.param_groups
                                                        if g.get('name') == 'vlm'), 0.0),
                                        'perf/steps_per_sec': steps_per_sec,
                                        'perf/samples_per_sec': steps_per_sec * args.batch_size,
                                        'perf/gpu_mem_gb': torch.cuda.max_memory_allocated() / 2**30})
                            if grad_norm is not None:
                                log['train/grad_norm'] = grad_norm.item()
                            wandb.log(log, step=total_steps)
            if args.dry_run:
                break

        epoch_metrics = metrics.compute(dist if dist.distributed else None)  # all-gather: every rank
        if dist.local_rank == 0:  # Metrics.print is not rank-guarded
            utils.Metrics.print(epoch_metrics, epoch + 1)
        if use_wandb:
            import wandb
            wandb.log({f'epoch/{k}': v for k, v in epoch_metrics.items()}
                      | {'epoch': epoch + 1, 'perf/epoch_time_s': time.time() - epoch_t0},
                      step=total_steps)

        if not args.dry_run:
            utils.save_model(args, dist, model, ckpt_file)
            utils.save_optimizer(args, dist, optimizer, lr_schedule, opt_ckpt_file)
            if epoch % args.save_every == 0:
                utils.save_model(args, dist, model, str(ckpt_file) + f'_epoch{epoch + 1:04d}')
        dist.barrier()

        # rollout preview added in vla/sample_libero.py (Task 6)
        if args.sample_freq > 0 and (epoch % args.sample_freq == 0) and not args.dry_run:
            from vla.sample_libero import preview_rollout
            mae, video_path = preview_rollout(
                model, vae, args, dist, frames, instructions, actions_clean, proprio,
                sample_dir, epoch, total_steps)
            if use_wandb:
                import wandb
                log = {'preview/action_mae_normalized': mae, 'epoch': epoch + 1}
                if video_path is not None and os.path.exists(video_path):
                    log['preview/rollout'] = wandb.Video(str(video_path), fps=10, format='mp4')
                wandb.log(log, step=total_steps)

        if args.dry_run:
            break

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    args = get_vla_parser().parse_args()
    main(args)
