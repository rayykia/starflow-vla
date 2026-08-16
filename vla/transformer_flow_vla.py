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
