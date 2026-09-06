"""Two-view (agentview + wrist) latents for STARFlow-VLA.

Each camera view is encoded by the frozen Wan VAE as an independent clip and the
per-view latents are channel-concatenated per timestep, so the flow generates both
views in one AR token (SimVLANF/models/wan_video_vae.py::WanVideoVAE.encode_frames).
View order follows vla/dataset_libero.py::VIEW_KEYS: the first `Cv` channels are the
agentview, the next `Cv` the wrist view.

Pixel videos carry an explicit view axis, (B, T, V, 3, H, W); fused latents are
(B, T_lat, V * Cv, h, w), i.e. exactly what `WorldActionModel` consumes with
`channel_size = V * Cv`.
"""
import torch
from einops import rearrange


def encode_views(vae, frames: torch.Tensor) -> torch.Tensor:
    """frames (B, T, V, 3, H, W) in [-1, 1] -> fused latents (B, T_lat, V * Cv, h, w)."""
    B, V = frames.size(0), frames.size(2)
    z = vae.encode(rearrange(frames, 'b t v c h w -> (b v) t c h w'))     # ((b v), T_lat, Cv, h, w)
    return rearrange(z, '(b v) t c h w -> b t (v c) h w', b=B, v=V)


def decode_views(vae, latents: torch.Tensor, num_views: int = 2) -> torch.Tensor:
    """fused latents (B, T_lat, V * Cv, h, w) -> pixels (B, T, V, 3, H, W)."""
    B = latents.size(0)
    z = rearrange(latents, 'b t (v c) h w -> (b v) t c h w', v=num_views)
    x = vae.decode(z)                                                       # ((b v), T, 3, H, W)
    return rearrange(x, '(b v) t c h w -> b t v c h w', b=B, v=num_views)


def tile_views(video: torch.Tensor) -> torch.Tensor:
    """(B, T, V, 3, H, W) -> (B, T, 3, H, V * W): views side by side, agentview left."""
    return rearrange(video, 'b t v c h w -> b t c h (v w)')
