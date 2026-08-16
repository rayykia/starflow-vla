import torch

from transformer_flow import MetaBlock, PermutationIdentity, KVCache
from misc.pe import VisionRotaryEmbeddingFast
from vla.transformer_flow_vla import ActionMetaBlock

B, T, HW, C, A, Da, TXT, TXT_DIM = 2, 2, 8, 4, 4, 7, 8, 16


def make_block():
    torch.manual_seed(0)
    blk = MetaBlock(
        in_channels=C, channels=64, img_size=HW,
        permutation=PermutationIdentity(HW * HW), pt_seq_len=HW,
        num_layers=2, head_dim=32, txt_size=TXT, txt_dim=TXT_DIM,
        use_rope=True, use_sos=True, use_softplus=True,
        use_final_norm=True, soft_clip=4)
    blk = ActionMetaBlock.upgrade(blk, action_dim=Da)
    rope = VisionRotaryEmbeddingFast(dim=16, pt_seq_len=HW, latent_len=TXT, no_buffer=True)
    return blk.eval(), rope


def make_inputs():
    torch.manual_seed(1)
    x = torch.randn(B, T, HW, HW, C)   # patchified video latents
    y = torch.randn(B, TXT, TXT_DIM)
    a = torch.randn(B, A, Da)
    return x, y, a


def test_joint_forward_shapes():
    blk, rope = make_block()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v, z_a, _, logdet_v, logdet_a = blk(x, y, rope=rope, actions=a)
    assert z_v.shape == x.shape
    assert z_a.shape == a.shape
    assert logdet_v.shape == (B, T * HW * HW, C)
    assert logdet_a.shape == (B, A, Da)
    assert torch.isfinite(z_v).all() and torch.isfinite(z_a).all()


def test_video_outputs_unaffected_by_actions():
    # causal attention: appended action tokens must not change video outputs
    blk, rope = make_block()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v_joint, _, _, ld_joint, _ = blk(x, y, rope=rope, actions=a)
        z_v_only, _, ld_only = MetaBlock.forward(blk, x, y, rope=rope)
    torch.testing.assert_close(z_v_joint, z_v_only, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ld_joint, ld_only, atol=1e-5, rtol=1e-4)


def test_delegates_when_actions_none():
    blk, rope = make_block()
    x, y, _ = make_inputs()
    with torch.no_grad():
        out = blk(x, y, rope=rope)  # MetaBlock 3-tuple path
    assert len(out) == 3
