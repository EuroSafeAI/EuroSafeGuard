"""The consistency term enters through an autograd function: its gradients equal those of adding it to the loss."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train_classifier import _AddLoss  # noqa: E402


def test_add_loss_matches_adding_to_the_loss():
    torch.manual_seed(0)
    w = torch.randn(4, 3, requires_grad=True)
    x = torch.randn(5, 4)
    h = x @ w
    aux = (1 - torch.nn.functional.cosine_similarity(h[:2], h[2:4].detach(), dim=-1)).mean()
    ref = torch.autograd.grad(torch.sigmoid(h).mean() + 0.7 * aux, w)[0]
    h = x @ w
    aux = (1 - torch.nn.functional.cosine_similarity(h[:2], h[2:4].detach(), dim=-1)).mean()
    out = torch.autograd.grad(torch.sigmoid(_AddLoss.apply(h, 0.7 * aux)).mean(), w)[0]
    assert torch.allclose(ref, out, atol=1e-6)
