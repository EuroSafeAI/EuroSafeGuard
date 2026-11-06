"""Activation probes in the style of Constitutional Classifiers++ (Cunningham et al., 2026, arXiv:2601.04603, Sec. 5).

For a text x = (x_1..x_T) with label y in {0, 1} and concatenated layer activations psi_t:

  z_t     per-token logit: linear  z_t = W^T psi_t + b              (the CC++ probe, used by EuroSafe-CC++)
                           MLP     z_t = w2 . gelu(W1 psi_t + b1) + b2 (hidden 256, used by EuroSafeGuard)
  zbar_t  = (1/M) sum_{k<M} z_{t-k}                                  sliding-window mean (SWiM), training
  L(x)    = sum_{t>=M} w_t BCE(y, sigmoid(zbar_t)),  w_t = softmax_t(zbar_t / tau)   softmax-weighted loss
  score   = max_t EMA_t(z)  with alpha = 2/(M+1)                     streaming inference (as in eurosafeguard.guard)

Rows with fewer than M scored tokens make a single prediction from the mean of their tokens.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eurosafeguard.guard import MLPProbe as _InferenceMLP

NEG = -1e9


# ----------------------------------------------------------------------------------------------- features
def _chat(tok, messages):
    kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


def encode(tok, prompt: str, response: str | None = None, max_len: int = 2048):
    """-> (ids, start, end): the chat-formatted prompt (up to the assistant header) plus the response, if any.
    Scored positions [start, end) are the prompt for prompt-only rows and the response tokens otherwise. Long
    inputs are cut from the left of the prompt first."""
    head = tok(_chat(tok, [{"role": "user", "content": str(prompt)}]), add_special_tokens=False)["input_ids"]
    resp = tok(str(response), add_special_tokens=False)["input_ids"] if response else []
    if len(head) + len(resp) > max_len:
        head = head[-max(16, max_len - len(resp)):]
        resp = resp[: max_len - len(head)]
    if not resp:
        return head, 0, len(head)
    return head + resp, len(head), len(head) + len(resp)


@torch.no_grad()
def activations(model, tok, items, layers):
    """items: list of (ids, start, end) -> psi [B, T, len(layers) * H] in the model dtype, scored mask [B, T]."""
    device = next(model.parameters()).device
    T = max(len(i[0]) for i in items)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inp = torch.full((len(items), T), pad, dtype=torch.long)
    att = torch.zeros((len(items), T), dtype=torch.long)
    sc = torch.zeros((len(items), T), dtype=torch.bool)
    for k, (ids, st, en) in enumerate(items):
        inp[k, : len(ids)] = torch.tensor(ids)
        att[k, : len(ids)] = 1
        sc[k, st:en] = True
    hs = model(input_ids=inp.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
    return torch.cat([hs[l] for l in layers], dim=-1), sc.to(device)


# ----------------------------------------------------------------------------------------------- probes
class LinearProbe(nn.Module):
    """z_t = W^T (scale * psi_t) + b; ``scale`` is a fixed per-layer 1/RMS normaliser."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, psi):
        w = self.W * self.scale
        return (psi @ w.to(psi.dtype)).float() + self.b


class MLPProbe(_InferenceMLP):
    """The released probe architecture (``eurosafeguard.guard.MLPProbe``) with the initialisation used in training."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__(dim, hidden)
        nn.init.normal_(self.W1.weight, std=1.0 / math.sqrt(dim))
        nn.init.zeros_(self.w2.weight)


def make_probe(kind: str, dim: int, hidden: int = 256) -> nn.Module:
    return MLPProbe(dim, hidden) if kind == "mlp" else LinearProbe(dim)


# ----------------------------------------------------------------------------------------------- losses and scores
def swim(z, mask, M: int):
    """Sliding-window mean over scored positions -> (zbar, valid); see the module docstring."""
    mask = mask.bool()
    zc = torch.cumsum(z * mask, 1)
    n = torch.cumsum(mask.to(z.dtype), 1)
    zbar = (zc - F.pad(zc, (M, 0))[:, :-M]) / (n - F.pad(n, (M, 0))[:, :-M]).clamp(min=1)
    valid = mask & (n >= M)
    short = mask.sum(1) < M
    if short.any():
        last = (torch.cumsum(mask.long(), 1) * mask).argmax(1)
        valid = valid.clone()
        valid[short] = False
        valid[short, last[short]] = True
    return zbar, valid


def probe_loss(z, mask, y, M: int = 16, tau: float = 1.0):
    """CC++ softmax-weighted, SWiM-smoothed per-token BCE, averaged over the batch."""
    zbar, valid = swim(z, mask, M)
    per = F.binary_cross_entropy_with_logits(zbar, y.to(zbar.dtype)[:, None].expand_as(zbar), reduction="none")
    w = torch.softmax((zbar / tau).masked_fill(~valid, NEG), dim=1)
    return (w * per.masked_fill(~valid, 0.0)).sum(1).mean()


def ema_scores(z, mask, M: int = 16):
    """Streaming score of each row: max over scored positions of the EMA of per-token logits."""
    a = 2.0 / (M + 1)
    mask = mask.bool()
    state = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
    started = torch.zeros_like(mask[:, 0])
    best = torch.full((z.shape[0],), NEG, dtype=z.dtype, device=z.device)
    for t in range(z.shape[1]):
        m = mask[:, t]
        state = torch.where(m, torch.where(started, (1 - a) * state + a * z[:, t], z[:, t]), state)
        started = started | m
        best = torch.where(m, torch.maximum(best, state), best)
    return best


@torch.no_grad()
def layer_rms_scale(model, tok, items, layers, bs: int = 8):
    """Label-free per-layer 1/RMS normaliser, estimated on the scored tokens of a few rows."""
    H = model.config.get_text_config().hidden_size
    ss = torch.zeros(len(layers), dtype=torch.float64, device=next(model.parameters()).device)
    cnt = 0
    for i in range(0, len(items), bs):
        psi, sc = activations(model, tok, items[i:i + bs], layers)
        x = psi[sc]
        ss += x.view(x.shape[0], len(layers), -1).float().pow(2).sum((0, 2)).double()
        cnt += x.shape[0] * H
    return (1.0 / (ss / cnt).sqrt()).float().repeat_interleave(H)
