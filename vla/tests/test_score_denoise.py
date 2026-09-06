import torch

from vla.sample_libero import score_denoise
from vla.tests.test_world_action_model import tiny_model, make_inputs


def _nll_action(model, x, y, a):
    z_v, z_a, _, ldv, lda = model(x, y, actions=a)
    return 0.5 * z_a.pow(2).sum() - sum(ld.sum() for ld in lda)


def test_score_denoise_shapes_and_noop():
    model = tiny_model()
    x, y, a = make_inputs()
    x2, a2 = score_denoise(model, x, a, y, sigma_video=0.0, sigma_action=0.0)
    assert x2 is x and a2 is a  # both sigmas zero: untouched
    x3, a3 = score_denoise(model, x, a, y, sigma_video=0.0, sigma_action=0.1)
    assert x3.shape == x.shape and a3.shape == a.shape
    torch.testing.assert_close(x3, x)                # video untouched when sigma_video=0
    assert torch.isfinite(a3).all() and not torch.allclose(a3, a)
    assert not a3.requires_grad and not x3.requires_grad


def test_score_denoise_matches_tweedie_step():
    # a_den = a - sigma^2 * grad_a NLL_action(a | video), strength scales the step
    model = tiny_model()
    x, y, a = make_inputs()
    sigma = 0.1
    a_req = a.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_nll_action(model, x, y, a_req), a_req)
    expected = a - sigma ** 2 * g
    _, a_den = score_denoise(model, x, a, y, sigma_action=sigma)
    torch.testing.assert_close(a_den, expected, atol=1e-5, rtol=1e-4)
    _, a_half = score_denoise(model, x, a, y, sigma_action=sigma, strength=0.5)
    torch.testing.assert_close(a_half, a - 0.5 * sigma ** 2 * g, atol=1e-5, rtol=1e-4)


def test_score_denoise_video_and_action_jointly():
    model = tiny_model()
    x, y, a = make_inputs()
    x_den, a_den = score_denoise(model, x, a, y, sigma_video=0.3, sigma_action=0.1)
    assert torch.isfinite(x_den).all() and torch.isfinite(a_den).all()
    assert not torch.allclose(x_den, x)
    # the action step is unaffected by whether the video is also denoised
    _, a_only = score_denoise(model, x, a, y, sigma_action=0.1)
    torch.testing.assert_close(a_den, a_only, atol=1e-5, rtol=1e-4)


def test_score_denoise_batch_independence():
    # per-sample scores: denoising a batch == denoising each sample alone
    model = tiny_model()
    x, y, a = make_inputs()
    _, a_batch = score_denoise(model, x, a, y, sigma_action=0.1)
    for i in range(x.size(0)):
        _, a_i = score_denoise(model, x[i:i + 1], a[i:i + 1], y[i:i + 1], sigma_action=0.1)
        torch.testing.assert_close(a_batch[i:i + 1], a_i, atol=1e-5, rtol=1e-4)
