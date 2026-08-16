"""STARFlow-VLA: video + action-chunk normalizing flow on top of STARFlow-V.

The deep block runs one causal AR flow over [text | video tokens | H action
tokens]. The global SOS shift means the output at the last video token predicts
action a_0, so the chunk is generated strictly after (and conditioned on) the
video. Actions use their own input/output projections (7-dim vs 48-dim video).
"""
import tqdm
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from transformer_flow import (
    Model, MetaBlock, KVCache, PermutationIdentity, PermutationFlip,
    INV_SOFTPLUS_1,
)
from misc.pe import VisionRotaryEmbeddingFast, get_positions


class ActionMetaBlock(MetaBlock):
    """Deep block extended with an action-chunk group at the end of the AR sequence."""

    @classmethod
    def upgrade(cls, block: MetaBlock, action_dim: int) -> 'ActionMetaBlock':
        # In-place class upgrade instead of re-assembling MetaBlock's ~20 ctor
        # kwargs: keeps the flow config byte-identical to what Model built and
        # immune to upstream kwarg drift.
        assert type(block) is MetaBlock, f'cannot upgrade {type(block)}'
        assert block.use_sos, 'ActionMetaBlock requires sos=1'
        assert isinstance(block.permutation, PermutationIdentity), (
            'top block must use PermutationIdentity: use seq_order=L2R with an '
            'even number of blocks')
        assert block.local_attn_window is None, \
            'ActionMetaBlock does not support local_attn_window (cache sizing excludes action tokens)'
        block.__class__ = cls
        channels = block.proj_in.out_features
        block.action_dim = action_dim
        block.proj_in_act = torch.nn.Linear(action_dim, channels)
        block.proj_out_act = torch.nn.Linear(channels, 2 * action_dim)
        torch.nn.init.constant_(block.proj_out_act.weight, 0)
        return block

    def get_freqs_cis(self, x, y, rope, num_actions=0):
        if num_actions == 0:
            return super().get_freqs_cis(x, y, rope)
        assert not rope.is_1d and self.txt_dim > 0, 'joint path expects 3D RoPE with text'
        h, w = x.size(-3), x.size(-2)
        d = x.size(1) if x.dim() == 5 else 0
        txt_size = y.size(1) if self.txt_size > 0 and y is not None else 0
        pos = get_positions(h, w, txt_size, rope.pt_seq_len, d, mode='3d')
        t0 = float(max(d, 1))  # action step j sits at time T_lat + j, spatial (0, 0)
        act_pos = torch.stack([
            torch.zeros(num_actions), torch.zeros(num_actions),
            t0 + torch.arange(num_actions, dtype=torch.float32)], dim=-1)
        pos = torch.cat([pos, act_pos.to(pos.dtype)], dim=0)
        return rope(pos.type_as(x))

    def _couple(self, out, x_in):
        xa, xb = out.chunk(2, dim=-1)
        dtype = xa.dtype
        xa, xb, x_in = xa.float(), xb.float(), x_in.float()
        xa = F.softplus(xa + INV_SOFTPLUS_1) if self.use_softplus else xa.exp()
        z = (x_in - xb) / xa
        return z.to(dtype), -torch.log(xa)

    def forward(self, x, y=None, rope=None, kv_cache=None, guidance=None, actions=None):
        if actions is None:  # pure-video path (context pass, CFG cache filling)
            return super().forward(x, y, rope, kv_cache, guidance)
        assert kv_cache is None, 'joint forward is training-only; use context=True for caches'
        assert not guidance, 'guidance is inference-only'

        A = actions.size(1)
        freqs_cis = self.get_freqs_cis(x, y, rope, num_actions=A) if rope is not None else None

        x = self.permutation(x)                       # identity: (B, Nv, C)
        xv_in, a_in = x.clone(), actions.clone()
        Nv = x.size(1)

        # AR inputs [SOS, v_0..v_{Nv-1}, a_0..a_{A-2}], embedded per-modality.
        # Unlike video-only MetaBlock, the last video token IS fed: it predicts a_0.
        h = self.get_proj_in(torch.cat([self.get_sos_embed(x), x], dim=1))
        h = torch.cat([h, self.proj_in_act(a_in.to(h.dtype)[:, :-1])], dim=1)

        y = self.proj_txt(y) if self.txt_dim > 0 else None

        for it, block in enumerate(self.attn_blocks):
            checkpoint_attn = self.training and self.use_checkpoint > 0 and \
                ((it + 1) % self.use_checkpoint == 0)
            if self.use_checkpoint_mlp is not None:
                checkpoint_mlp = self.training and self.use_checkpoint_mlp > 0 and \
                    ((it + 1) % self.use_checkpoint_mlp == 0)
            else:
                checkpoint_mlp = checkpoint_attn
            h, y = block(h, y, None, 1.0, None, freqs_cis, None,
                         checkpoint_attn=checkpoint_attn, checkpoint_mlp=checkpoint_mlp)

        if self.use_final_norm:
            h, y = self.final_norm(h), self.final_norm(y) if y is not None else None

        out_v = self.get_proj_out(h[:, :Nv])          # (B, Nv, 2C), includes soft_clip
        out_a = self.proj_out_act(h[:, Nv:])          # (B, A, 2*Da)
        if self.soft_clip > 0:
            out_a = self.soft_clip * torch.tanh(out_a / self.soft_clip)

        z_v, logdet_v = self._couple(out_v, xv_in)
        z_a, logdet_a = self._couple(out_a, a_in)
        z_v = self.permutation(z_v, inverse=True)     # (B, T, h, w, C)
        return z_v, z_a, y, logdet_v, logdet_a

    def reverse_step_action(self, x, a, j, kv_cache, attn_temp=1.0, freqs_cis=None):
        # Input at action step j: last video token (j == 0) or a_{j-1}; output
        # always through the action head, predicting a_j.
        original_dtype = a.dtype
        if j == 0:
            h = self.get_proj_in(x[:, -1:].to(self.proj_in.weight.dtype))
        else:
            h = self.proj_in_act(a[:, j - 1:j].to(self.proj_in_act.weight.dtype))
        for i, block in enumerate(self.attn_blocks):
            h, _ = block(h, None, attn_temp=attn_temp, freqs_cis=freqs_cis,
                         kv_cache=partial(kv_cache, i))
        if self.use_final_norm:
            h = self.final_norm(h)
        h = self.proj_out_act(h)
        if self.soft_clip > 0:
            h = self.soft_clip * torch.tanh(h / self.soft_clip)
        xa, xb = h.chunk(2, dim=-1)
        return xa.to(original_dtype), xb.to(original_dtype)

    def reverse(self, z, z_act, y=None, guidance=0, guide_what='ab', attn_temp=1.0,
                annealed_guidance=False, rope=None, verbose=False,
                kv_cache: KVCache = None, **unused_kwargs):
        kv_cache = kv_cache if kv_cache is not None else KVCache()
        original_dtype = z.dtype
        z, z_act = z.float(), z_act.float()
        A = z_act.size(1)
        freqs_cis = self.get_freqs_cis(z, y, rope, num_actions=A) if rope is not None else None
        if guidance > 0:
            z, z_act = torch.cat([z, z], 0), torch.cat([z_act, z_act], 0)

        reuse_kv_cache = kv_cache.prefix_cache is not None and kv_cache.kv_index[0] > 0
        kv_cache = self.initialize_kv_cache(kv_cache, z, freqs_cis, reuse_kv_cache)

        z = self.permutation(z)
        if self.txt_dim > 0 and not reuse_kv_cache:
            self.reverse_step_condition(y, kv_cache, None, attn_temp, freqs_cis)
        txt_size = y.size(1) if self.txt_dim > 0 else 0

        x = z.clone()
        if reuse_kv_cache:
            x[:, :kv_cache.prefix_cache.size(1)] = kv_cache.prefix_cache

        def _scale(za):
            za = F.softplus(za + INV_SOFTPLUS_1) if self.use_softplus else za.exp()
            return za.squeeze(1)

        T = x.size(1)  # use_sos asserted at upgrade time
        for t in tqdm.trange(T, disable=not verbose, desc='VLA deep-block video', leave=False):
            if reuse_kv_cache and kv_cache.kv_index[0] > t + txt_size:
                continue  # obs prefix already cached
            za, zb = self.reverse_step(x, t, kv_cache, None, y, attn_temp, freqs_cis)
            za, zb = _scale(za.float()), zb.float().squeeze(1)
            if guidance > 0 and guide_what:
                r = (t + 1) / T if annealed_guidance else 1.0
                zb, za = self.guidance(za, zb, guidance, r, guide_what)
            x[:, t] = z[:, t] * za + zb

        a = z_act.clone()
        for j in range(A):
            za, zb = self.reverse_step_action(x, a, j, kv_cache, attn_temp, freqs_cis)
            za, zb = _scale(za.float()), zb.float().squeeze(1)
            if guidance > 0 and guide_what:
                zb, za = self.guidance(za, zb, guidance, 1.0, guide_what)
            a[:, j] = z_act[:, j] * za + zb

        if guidance > 0:
            x, a = x.chunk(2, dim=0)[0], a.chunk(2, dim=0)[0]
            kv_cache.remove_negative_cache()

        x = self.permutation(x, inverse=True)
        return x.to(original_dtype), a.to(original_dtype)


class WorldActionModel(Model):
    """STARFlow-V + action chunk: image+text -> video + H actions, one joint flow."""

    def __init__(self, *, action_horizon=8, action_dim=7, action_channels=256,
                 action_head_dim=64, action_layers=(2, 2), action_loss_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        assert kwargs.get('seq_order') == 'L2R', 'STARFlow-VLA requires seq_order=L2R'
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.action_loss_weight = action_loss_weight

        # upgrade the top block in place (asserts sos + PermutationIdentity)
        self.blocks[-1] = ActionMetaBlock.upgrade(self.blocks[-1], action_dim)

        # action shallow flow: small MetaBlocks over the chunk alone, shaped
        # (B, A, 1, Da), alternating direction, 1D RoPE over the A steps
        perms = [PermutationIdentity(action_horizon), PermutationFlip(action_horizon)]
        self.action_blocks = torch.nn.ModuleList([
            MetaBlock(
                in_channels=action_dim, channels=action_channels,
                img_size=action_horizon, permutation=perms[i % 2],
                pt_seq_len=action_horizon, num_layers=num_layers,
                head_dim=action_head_dim, txt_size=0, txt_dim=0,
                use_rope=True, use_sos=self.use_sos, use_softplus=self.use_softplus,
                use_swiglu=kwargs.get('use_swiglu', False),
                use_qk_norm=kwargs.get('use_qk_norm', False),
                use_post_norm=kwargs.get('use_post_norm', False),
                use_final_norm=kwargs.get('use_final_norm', False),
                use_bias=kwargs.get('use_bias', True),
                norm_type=kwargs.get('norm_type', 'layer_norm'),
                soft_clip=kwargs.get('soft_clip', 0),
            ) for i, num_layers in enumerate(action_layers)])
        self.action_rope = VisionRotaryEmbeddingFast(
            dim=action_head_dim // 2, pt_seq_len=action_horizon,
            no_buffer=True, is_1d=True)

    def forward(self, x, y=None, actions=None, reverse=False, kv_caches=None,
                denoiser=False, context=False, **kwargs):
        if reverse:
            assert actions is not None, 'reverse needs action noise z_a as `actions`'
            return self.reverse(x, actions, y, kv_caches=kv_caches, **kwargs)
        if context or denoiser or actions is None:
            # context pass (obs only), denoiser, or plain video forward: parent
            # handles it; ActionMetaBlock.forward(actions=None) behaves as MetaBlock
            return super().forward(x, y, reverse=False, kv_caches=kv_caches,
                                   denoiser=denoiser, context=context, **kwargs)

        B, A, Da = actions.shape
        logdets_v, logdets_a, outputs = [], [], []

        # video shallow flow (per-frame with shallow_block_local, as upstream)
        x = self.patchify(x)
        outputs.append(x)
        for block in self.blocks[:-1]:
            if self.shallow_block_local and x.dim() == 5:
                x = rearrange(x, 'b t h w c -> (b t) 1 h w c')
            x, _, logdet = block(x, y, self.feat_rope, kv_cache=None)
            if self.shallow_block_local and x.dim() == 5:
                x = rearrange(x, '(b t) 1 h w c -> b t h w c',
                              b=outputs[0].size(0), t=outputs[0].size(1))
                logdet = rearrange(logdet, '(b t) l c -> b t l c',
                                   b=outputs[0].size(0), t=outputs[0].size(1))
            logdets_v.append(logdet)
            outputs.append(x)

        # action shallow flow
        a = actions.view(B, A, 1, Da)
        for block in self.action_blocks:
            a, _, logdet_a = block(a, None, self.action_rope)
            logdets_a.append(logdet_a)
        a = a.reshape(B, A, Da)

        # joint deep block
        z_v, z_a, y, logdet_v, logdet_a = self.blocks[-1](
            x, y, self.feat_rope_gen, actions=a)
        logdets_v.append(logdet_v)
        logdets_a.append(logdet_a)
        outputs.append(z_v)
        return self.unpatchify(z_v), z_a, outputs, logdets_v, logdets_a

    def get_loss(self, z_v, logdets_v, z_a, logdets_a, action_loss_weight=None):
        # separate per-modality means: action dims are <1% of total, a joint
        # mean would starve the policy gradient
        w = self.action_loss_weight if action_loss_weight is None else action_loss_weight
        loss_v_z = 0.5 * z_v.pow(2).mean(dim=tuple(range(1, z_v.dim())))
        loss_v_ld = -sum(ld.mean(dim=tuple(range(1, ld.dim()))) for ld in logdets_v)
        loss_a_z = 0.5 * z_a.pow(2).mean(dim=tuple(range(1, z_a.dim())))
        loss_a_ld = -sum(ld.mean(dim=tuple(range(1, ld.dim()))) for ld in logdets_a)
        loss = ((loss_v_z + loss_v_ld) + w * (loss_a_z + loss_a_ld)).mean()
        return {'loss': loss,
                'loss_video_z': loss_v_z.detach().mean(),
                'loss_video_logdet': loss_v_ld.detach().mean(),
                'loss_action_z': loss_a_z.detach().mean(),
                'loss_action_logdet': loss_a_ld.detach().mean()}

    def reverse(self, x, actions, y=None, guidance=0, verbose=False,
                kv_caches=None, **sampling_kwargs):
        need_caches = kv_caches is not None
        kv_caches = kv_caches if kv_caches is not None \
            else [KVCache() for _ in range(len(self.blocks))]
        B = x.size(0)

        # deep block: one AR pass generates video tokens then action tokens
        xp = self.patchify(x)
        xp, a = self.blocks[-1].reverse(
            xp, actions, y, guidance, rope=self.feat_rope_gen,
            kv_cache=kv_caches[0], verbose=verbose, **sampling_kwargs)
        x = self.unpatchify(xp)
        if not need_caches:
            kv_caches[0].delete()

        # shallow blocks are unconditional under cond_top_only: drop guidance
        if self.cond_top_only and guidance > 0:
            guidance, y = 0, y.chunk(2, dim=0)[0]

        seq = [x]
        x = self.reverse_shallow(x, y, guidance, verbose, kv_caches,
                                 False, need_caches, seq, **sampling_kwargs)

        for block in reversed(self.action_blocks):
            a = block.reverse(a.view(B, self.action_horizon, 1, self.action_dim),
                              None, 0, rope=self.action_rope, kv_cache=KVCache())
            a = a.reshape(B, self.action_horizon, self.action_dim)
        return x, a


def setup_vla_model(args, txt_dim):
    return WorldActionModel(
        in_channels=args.channel_size, img_size=args.img_size,
        patch_size=args.patch_size, channels=args.channels,
        num_blocks=len(args.layers_per_block), layers_per_block=args.layers_per_block,
        head_dim=args.head_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        rope=args.rope, pt_seq_len=args.pt_seq_len, sos=args.sos,
        txt_size=args.txt_size, txt_dim=txt_dim, cond_top_only=args.cond_top_only,
        use_softplus=args.use_softplus, use_swiglu=args.use_swiglu,
        use_bias=args.use_bias, use_qk_norm=args.use_qk_norm,
        use_post_norm=args.use_post_norm, use_final_norm=args.use_final_norm,
        norm_type=args.norm_type, soft_clip=args.soft_clip, seq_order=args.seq_order,
        temporal_causal=args.temporal_causal, shallow_block_local=args.shallow_block_local,
        top_block_channels=args.top_block_channels,
        use_checkpoint=args.gradient_checkpoint,
        use_checkpoint_mlp=args.gradient_checkpoint_mlp,
        action_horizon=args.action_horizon, action_dim=args.action_dim,
        action_channels=args.action_channels, action_head_dim=args.action_head_dim,
        action_layers=args.action_layers, action_loss_weight=args.action_loss_weight,
    )
