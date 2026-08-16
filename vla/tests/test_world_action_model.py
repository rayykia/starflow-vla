import torch

from vla.transformer_flow_vla import WorldActionModel

B, T, C, HW, A, Da, TXT, TXT_DIM = 2, 2, 4, 8, 4, 7, 8, 32


def tiny_model():
    torch.manual_seed(0)
    return WorldActionModel(
        in_channels=C, img_size=HW, patch_size=1, channels=64,
        num_blocks=4, layers_per_block=[1, 1, 1, 2], head_dim=32,
        rope=True, pt_seq_len=HW, sos=True, txt_size=TXT, txt_dim=TXT_DIM,
        cond_top_only=True, use_softplus=True, use_final_norm=True,
        seq_order='L2R', temporal_causal=1, shallow_block_local=True, soft_clip=4,
        action_horizon=A, action_dim=Da, action_channels=32,
        action_head_dim=16, action_layers=[1, 1], action_loss_weight=1.0,
    ).eval()


def make_inputs():
    torch.manual_seed(1)
    x = torch.randn(B, T, C, HW, HW)   # latent video, channel-first as from the VAE
    y = torch.randn(B, TXT, TXT_DIM)
    a = torch.randn(B, A, Da)
    return x, y, a


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
