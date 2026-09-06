import torch

from vla.multiview import encode_views, decode_views, tile_views
from vla.sample_libero import save_video_grid


class FakeVAE:
    """Per-sample-independent stand-in: 3 -> Cv channels by a fixed linear map,
    spatial /4, temporal (1 + 4k) -> (1 + k), inverted exactly by decode."""

    def __init__(self, cv=6):
        torch.manual_seed(0)
        self.cv = cv
        self.w = torch.randn(cv, 3)

    def encode(self, x):                              # (B, T, 3, H, W) -> (B, 1+(T-1)//4, cv, H/4, W/4)
        x = x[:, ::4]
        x = torch.einsum('btchw,dc->btdhw', x, self.w)
        return x[..., ::4, ::4]

    def decode(self, z):                              # inverse of encode (nearest upsampling)
        x = torch.einsum('btdhw,dc->btchw', z, torch.linalg.pinv(self.w).T)
        x = x.repeat_interleave(4, -1).repeat_interleave(4, -2)
        return x.repeat_interleave(4, 1)[:, :1 + 4 * (z.size(1) - 1)]


def test_encode_views_channel_layout():
    vae = FakeVAE()
    frames = torch.rand(2, 5, 2, 3, 16, 16) * 2 - 1              # (B, T, V, 3, H, W)
    z = encode_views(vae, frames)
    assert z.shape == (2, 2, 2 * vae.cv, 4, 4)
    # first Cv channels = agentview (view 0) encoded alone, next Cv = wrist (view 1)
    torch.testing.assert_close(z[:, :, :vae.cv], vae.encode(frames[:, :, 0]))
    torch.testing.assert_close(z[:, :, vae.cv:], vae.encode(frames[:, :, 1]))


def test_decode_views_inverts_encode_layout():
    vae = FakeVAE()
    frames = torch.rand(2, 5, 2, 3, 16, 16) * 2 - 1
    z = encode_views(vae, frames)
    x = decode_views(vae, z, num_views=2)
    assert x.shape == (2, 5, 2, 3, 16, 16)
    torch.testing.assert_close(x[:, :, 0], vae.decode(vae.encode(frames[:, :, 0])))
    torch.testing.assert_close(x[:, :, 1], vae.decode(vae.encode(frames[:, :, 1])))


def test_tile_views_side_by_side():
    video = torch.rand(1, 3, 2, 3, 8, 8)
    tiled = tile_views(video)
    assert tiled.shape == (1, 3, 3, 8, 16)
    torch.testing.assert_close(tiled[..., :8], video[:, :, 0])   # agentview on the left
    torch.testing.assert_close(tiled[..., 8:], video[:, :, 1])   # wrist on the right


def test_save_video_grid_accepts_two_views(tmp_path):
    import imageio.v2 as imageio
    video = torch.rand(1, 3, 2, 3, 16, 16) * 2 - 1                # (B, T, V, 3, H, W)
    save_video_grid(video, tmp_path / 'v.mp4', fps=2)
    frames = imageio.mimread(str(tmp_path / 'v.mp4'))
    assert len(frames) == 3 and frames[0].shape == (16, 2 * 16, 3)  # views tiled side by side
