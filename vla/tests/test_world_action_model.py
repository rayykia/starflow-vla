import torch

from vla.transformer_flow_vla import WorldActionModel

# CH = flow width = condition-token width (the deep block's proj_txt is the identity);
# VLM_DIM / PROPRIO_DIM are the raw feature dims projected by build_condition
B, T, C, HW, A, Da, TXT, CH, VLM_DIM, PROPRIO_DIM = 2, 2, 4, 8, 4, 7, 8, 64, 16, 8


class FakeVLM(torch.nn.Module):
    """Deterministic stand-in for SmolVLMEncoder: (images, instructions) -> (feats, mask)."""

    def __init__(self, hidden_size=VLM_DIM, n_tokens=TXT):
        super().__init__()
        self.hidden_size, self.n_tokens = hidden_size, n_tokens
        self.proj = torch.nn.Linear(3, hidden_size)
        self.text_embed = torch.nn.Embedding(64, hidden_size)

    def forward(self, images, instructions):
        B = images.size(0)
        pooled = torch.nn.functional.adaptive_avg_pool2d(images.float(), 1).flatten(1)  # (B, 3)
        img_tok = self.proj(pooled)[:, None].expand(B, self.n_tokens // 2, -1)
        ids = torch.tensor([[len(s) % 64] * (self.n_tokens // 2) for s in instructions],
                           device=images.device)
        feats = torch.cat([img_tok, self.text_embed(ids)], dim=1)                  # (B, TXT, D)
        return feats, torch.ones(B, self.n_tokens, dtype=torch.bool, device=images.device)


def tiny_model(with_vlm=False):
    torch.manual_seed(0)
    return WorldActionModel(
        in_channels=C, img_size=HW, patch_size=1, channels=CH,
        num_blocks=4, layers_per_block=[1, 1, 1, 2], head_dim=32,
        rope=True, pt_seq_len=HW, sos=True, txt_size=TXT, txt_dim=CH,
        cond_top_only=True, use_softplus=True, use_final_norm=True,
        seq_order='L2R', temporal_causal=1, shallow_block_local=True, soft_clip=4,
        action_horizon=A, action_dim=Da, action_channels=32,
        action_head_dim=16, action_layers=[1, 1], action_loss_weight=1.0,
        vlm_dim=VLM_DIM, proprio_dim=PROPRIO_DIM, vlm=FakeVLM() if with_vlm else None,
    ).eval()


def make_inputs():
    torch.manual_seed(1)
    x = torch.randn(B, T, C, HW, HW)   # latent video, channel-first as from the VAE
    y = torch.randn(B, 1 + TXT, CH)    # condition tokens as produced by build_condition
    a = torch.randn(B, A, Da)
    return x, y, a


def make_raw_condition():
    torch.manual_seed(2)
    return torch.randn(B, TXT, VLM_DIM), torch.randn(B, PROPRIO_DIM)


def test_build_condition_layout():
    # y = [proj_p(proprio) | proj_v(vlm_features)], deep-block width, proj_txt is identity
    model = tiny_model()
    feats, proprio = make_raw_condition()
    y = model.build_condition(feats, proprio)
    assert y.shape == (B, 1 + TXT, CH)
    torch.testing.assert_close(y[:, 0], model.proprio_proj(proprio))
    torch.testing.assert_close(y[:, 1:], model.vlm_proj(feats))
    assert isinstance(model.blocks[-1].proj_txt, torch.nn.Identity)
    # per-sample: the proprio token of sample i depends only on proprio[i]
    y2 = model.build_condition(feats, proprio.flip(0))
    torch.testing.assert_close(y2[:, 0], y[:, 0].flip(0))
    torch.testing.assert_close(y2[:, 1:], y[:, 1:])


def test_cond_forward_path_matches_explicit_y():
    # forward(cond=...) encodes the condition inside forward (DDP-visible) and must
    # equal forward(y=encode_condition(...)); the obs image / instruction / proprio matter
    model = tiny_model(with_vlm=True)
    # the output heads are zero-initialised, so a fresh model ignores its condition;
    # perturb the action head to make the dependence observable
    torch.nn.init.normal_(model.blocks[-1].proj_out_act.weight, std=0.02)
    x, _, a = make_inputs()
    images = torch.rand(B, 3, 16, 16) * 2 - 1
    proprio = torch.randn(B, PROPRIO_DIM)
    instructions = ['turn on the stove', 'put the bowl in the drawer']
    with torch.no_grad():
        y = model.encode_condition(images, instructions, proprio)
        z_v, z_a, _, _, _ = model(x, y, actions=a)
        z_v2, z_a2, _, _, _ = model(x, actions=a,
                                    cond=dict(images=images, instructions=instructions, proprio=proprio))
        z_a3 = model(x, actions=a, cond=dict(images=images, instructions=['', ''], proprio=proprio))[1]
    assert y.shape == (B, 1 + TXT, CH)
    torch.testing.assert_close(z_v2, z_v)
    torch.testing.assert_close(z_a2, z_a)
    assert not torch.allclose(z_a3, z_a)       # dropping the instruction changes the actions


def test_requires_matching_txt_dim():
    import pytest
    with pytest.raises(AssertionError):
        WorldActionModel(
            in_channels=C, img_size=HW, patch_size=1, channels=CH,
            num_blocks=4, layers_per_block=[1, 1, 1, 2], head_dim=32,
            rope=True, pt_seq_len=HW, sos=True, txt_size=TXT, txt_dim=CH // 2,
            cond_top_only=True, use_softplus=True, use_final_norm=True,
            seq_order='L2R', temporal_causal=1, shallow_block_local=True, soft_clip=4,
            action_horizon=A, action_dim=Da, action_channels=32,
            action_head_dim=16, action_layers=[1, 1],
            vlm_dim=VLM_DIM, proprio_dim=PROPRIO_DIM)


def test_forward_shapes_and_loss():
    model = tiny_model()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v, z_a, outputs, ldv, lda = model(x, y, actions=a)
    assert z_v.shape == x.shape and z_a.shape == a.shape
    assert len(ldv) == 4 and len(lda) == 3  # 3 shallow + top; 2 action shallow + top
    loss = model.get_loss(z_v, ldv, z_a, lda)
    for k in ('loss', 'loss_video_z', 'loss_video_logdet',
              'loss_action_z', 'loss_action_logdet'):
        assert torch.isfinite(loss[k]), k


def test_full_model_invertibility():
    model = tiny_model()
    x, y, a = make_inputs()
    with torch.no_grad():
        z_v, z_a, _, _, _ = model(x, y, actions=a)
        x_rec, a_rec = model(z_v, y, actions=z_a, reverse=True)
    torch.testing.assert_close(x_rec, x, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(a_rec, a, atol=5e-3, rtol=1e-2)


def test_context_prefix_preserved():
    # i2v-style: obs context fills KV caches; generated frame 0 must equal the obs
    model = tiny_model()
    x, y, a = make_inputs()
    obs = x[:, :1]
    with torch.no_grad():
        kv_caches = model(obs, y, context=True)
        x_gen, a_gen = model(torch.randn_like(x), y, actions=torch.randn_like(a),
                             reverse=True, kv_caches=kv_caches)
    assert x_gen.shape == x.shape and a_gen.shape == a.shape
    torch.testing.assert_close(x_gen[:, :1], obs, atol=5e-3, rtol=1e-2)


def test_model_cfg_smoke():
    # context + classifier-free guidance: the production sampling path
    model = tiny_model()
    x, y, a = make_inputs()
    obs = x[:, :1]
    y_cfg = torch.cat([y, torch.zeros_like(y)], dim=0)  # cond + null halves
    with torch.no_grad():
        kv_caches = model(obs, y_cfg, context=True)     # use_cfg: obs batch B, y batch 2B
        x_gen, a_gen = model(torch.randn_like(x), y_cfg, actions=torch.randn_like(a),
                             reverse=True, kv_caches=kv_caches, guidance=1.0)
    assert x_gen.shape == x.shape and a_gen.shape == a.shape
    assert torch.isfinite(x_gen).all() and torch.isfinite(a_gen).all()
    torch.testing.assert_close(x_gen[:, :1], obs, atol=5e-3, rtol=1e-2)
