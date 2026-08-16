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
from utils import encode_text, drop_label, add_noise
from vla.transformer_flow_vla import setup_vla_model
from vla.dataset_libero import create_libero_dataloader

WANDB_API_KEY = os.environ.get('WANDB_API_KEY', None)


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
    return parser


def load_vla_config(config_path: str) -> argparse.Namespace:
    with open(config_path) as f:
        model_configs = yaml.safe_load(f)
    arg_str = ''
    for conf in model_configs['arguments']:
        for key in conf:
            arg_str += f'--{key} {conf[key]} '
    return get_vla_parser().parse_args(arg_str.split())


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

    if dist.rank == 0 and WANDB_API_KEY is not None:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project='starflow-vla', name=args.wandb_name or 'libero', config=vars(args))

    print(f'{" Config ":-^80}')
    for k, v in sorted(vars(args).items()):
        print(f'{k:32s}: {v}')

    # data
    data_loader = create_libero_dataloader(args, dist)
    num_batches = len(data_loader)
    print(f'{num_batches} batches/epoch, {len(data_loader.dataset):,} samples')

    # frozen encoders
    tokenizer, text_encoder = utils.setup_encoder(args, dist, device)
    text_encoder.requires_grad_(False)
    vae = utils.setup_vae(args, dist, device)
    vae.requires_grad_(False)
    args.img_size = args.img_size // vae.downsample_factor

    # model
    model = setup_vla_model(args, txt_dim=text_encoder.config.hidden_size).to(device)
    if dist.local_rank == 0:
        import torchinfo
        torchinfo.summary(model)

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
    trainable = [p for p in model_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, betas=(0.9, 0.95), lr=args.lr, weight_decay=1e-4)
    warmup = args.warmup_steps if args.warmup_steps is not None else num_batches
    lr_schedule = utils.CosineLRSchedule(optimizer, warmup, args.epochs * num_batches,
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
        for it, (frames, instructions, actions) in enumerate(data_loader):
            frames = frames.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    x = vae.encode(frames)                       # (B, 1+H/4, 48, 8, 8)
                x, _ = add_noise(x, args.noise_std, args.noise_type)
                actions_clean = actions
                actions = actions_clean + args.action_noise_std * torch.randn_like(actions_clean)
                with torch.no_grad():
                    y = encode_text(text_encoder, tokenizer,
                                    drop_label(list(instructions), args.drop_label),
                                    args.txt_size, device)

                needs_update = (it + 1) % grad_accum == 0
                if it % grad_accum == 0:
                    optimizer.zero_grad()

                z_v, z_a, outputs, logdets_v, logdets_a = model_ddp(x, y, actions=actions)
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
                        skip = args.grad_skip and total_steps > 100 and \
                            grad_norm.item() > args.grad_clip
                    if skip:
                        print(f'skipping update, grad_norm={grad_norm.item():.3f}')
                        optimizer.zero_grad()
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    current_lr = lr_schedule.step()
                    total_steps += 1
                    if not skip:
                        metrics.update(loss_dict)

                    if (it // grad_accum) % 10 == 9:
                        print(f'{total_steps:,} steps - ' + '  '.join(
                            f'{k}: {v:.4f}' for k, v in loss_dict.items()))
                        if dist.rank == 0 and WANDB_API_KEY is not None:
                            import wandb
                            log = {k: v.item() if torch.is_tensor(v) else v
                                   for k, v in loss_dict.items()}
                            log.update({'lr': current_lr, 'steps': total_steps})
                            if grad_norm is not None:
                                log['grad_norm'] = grad_norm.item()
                            wandb.log(log, step=total_steps)
            if args.dry_run:
                break

        if not args.dry_run:
            utils.save_model(args, dist, model, ckpt_file)
            utils.save_optimizer(args, dist, optimizer, lr_schedule, opt_ckpt_file)
            if epoch % args.save_every == 0:
                utils.save_model(args, dist, model, str(ckpt_file) + f'_epoch{epoch + 1:04d}')
        dist.barrier()

        # rollout preview added in vla/sample_libero.py (Task 6)
        if args.sample_freq > 0 and (epoch % args.sample_freq == 0) and not args.dry_run:
            from vla.sample_libero import preview_rollout
            preview_rollout(model, vae, text_encoder, tokenizer, args, dist,
                            frames, instructions, actions_clean, sample_dir, epoch, total_steps)

        if args.dry_run:
            break


if __name__ == '__main__':
    args = get_vla_parser().parse_args()
    main(args)
