"""SmolVLMEncoder contract tests (need the HF weights in the local cache; skipped otherwise)."""
import os
import pathlib

import pytest
import torch

MODEL = 'HuggingFaceTB/SmolVLM-500M-Instruct'


def _cached():
    hub = os.environ.get('HF_HUB_CACHE') or os.path.join(
        os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub')
    return (pathlib.Path(hub) / ('models--' + MODEL.replace('/', '--'))).exists()


needs_weights = pytest.mark.skipif(not _cached(), reason=f'{MODEL} not in the HF cache')


@pytest.fixture(scope='module')
def encoder():
    from vla.vlm_encoder import SmolVLMEncoder
    torch.manual_seed(0)
    enc = SmolVLMEncoder(MODEL, image_size=256, max_text_tokens=24, freeze=True)
    return enc.to('cuda' if torch.cuda.is_available() else 'cpu')


@needs_weights
def test_shapes_and_padding(encoder):
    device = next(encoder.parameters()).device
    images = torch.rand(2, 3, 128, 128, device=device) * 2 - 1
    feats, mask = encoder(images, ['turn on the stove', ''])
    n_img = (256 // 16) ** 2 // 16                   # 256 patches, pixel-shuffle x4 -> 16 tokens
    L = encoder.prefix_ids.size(1) + n_img + 24
    assert feats.shape == (2, L, encoder.hidden_size) and mask.shape == (2, L)
    assert mask[:, :encoder.prefix_ids.size(1) + n_img].all()
    assert mask[1].sum() < mask[0].sum()             # empty instruction -> more padding
    assert (feats[~mask] == 0).all()                 # padded positions zeroed
    assert torch.isfinite(feats).all()


@needs_weights
def test_frozen_has_no_grad_and_depends_on_inputs(encoder):
    device = next(encoder.parameters()).device
    assert not any(p.requires_grad for p in encoder.parameters())
    images = torch.rand(1, 3, 128, 128, device=device) * 2 - 1
    with torch.enable_grad():
        f1, _ = encoder(images.expand(2, -1, -1, -1), ['open the drawer', 'close the drawer'])
    assert not f1.requires_grad
    assert not torch.allclose(f1[0], f1[1])           # instruction changes the features
    f2, _ = encoder(-images, ['open the drawer'])
    assert not torch.allclose(f1[0], f2[0])           # image changes the features
    encoder.train()
    assert not encoder.model.training              # frozen backbone stays in eval mode


@needs_weights
def test_finetune_mode_backprops():
    from vla.vlm_encoder import SmolVLMEncoder
    enc = SmolVLMEncoder(MODEL, image_size=256, max_text_tokens=24, freeze=False)
    enc = enc.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(enc.parameters()).device
    images = torch.rand(1, 3, 128, 128, device=device) * 2 - 1
    feats, _ = enc(images, ['pick up the bowl'])
    feats.float().pow(2).mean().backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert all(g is not None for g in grads), 'every VLM parameter must receive a gradient (DDP)'
    assert any(g.abs().sum() > 0 for g in grads)
