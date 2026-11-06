"""EuroSafe-CC++: open recreation of the Constitutional Classifiers++ cascade (EuroSafeAI). Single-file loader.

    from eurosafe_ccpp import EuroSafeCCPP
    g = EuroSafeCCPP.from_pretrained("EuroSafeAI/EuroSafe-CCPP", seed=0)
    escalated = g.escalate(prompt, reply)             # stage 1 only (two linear probes)
    blocked = g.check(prompt, reply)                  # True -> withhold the reply (configuration of the paper's tab:pipeline)
    flagged = g.input_flag(prompt)                    # prompt probe alone at the 5% WildChat-clean budget (paper's Table 2)

Components (see README.md):
  prompt probe  linear probe on Llama-3.1-8B-Instruct activations (all 32 blocks) of the chat-formatted prompt, up to and
                including the assistant header; trained with the CC++ loss on soft labels distilled from out-of-fold
                scores of our input classifiers.
  reply probe   linear probe of the same form on the reply tokens, read with the prompt as context; trained with the
                CC++ loss on hard labels.
                For both probes, per-token logits are smoothed with an EMA (alpha = 2/(M+1), M = 16) and the score is
                their maximum over the scored tokens.
  classifier    second stage: EuroSafeGuard's input classifier gate/seed{k} (Qwen3.5-9B + LoRA + 11 sigmoid heads,
                EuroSafeAI/EuroSafeGuard), applied to "User: <prompt>\\n\\nAssistant: <reply>" (max 4096 tokens);
                score = logit of the max-head probability.
Rule (thresholds/seed{k}.json): z-score the two probes and the classifier logit on the calibration pool. Stage 1 =
max(z prompt probe, z reply probe); exchanges above stage1_thr (the WildChat-clean 94.5th percentile) are escalated.
An escalated exchange is blocked if w_probe * stage1 + w_classifier * z classifier > stage2_thr (weights 0.55 / 0.45).
Non-escalated exchanges are never blocked, so the classifier runs only on escalated traffic.

Memory: Llama-3.1-8B (~16 GB bf16) + Qwen3.5-9B (~19 GB). Pass devices=("cuda:0", "cuda:1") to split them.
Requires: torch, transformers (with Qwen3.5 support), peft, safetensors, huggingface_hub, numpy.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

NEG = -1e9


def ccpp_block(prompt_probe, reply_probe, cls_prob, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Decision from raw scores (prompt-probe and reply-probe EMA-max logits, classifier max-head probability).
    Returns (blocked, escalated). cls_prob is only read for escalated rows (pass NaN elsewhere)."""
    s1 = stage1_z(prompt_probe, reply_probe, cfg)
    esc = s1 > cfg["stage1_thr"]
    p = np.clip(np.asarray(cls_prob, dtype=float), 1e-6, 1 - 1e-6)
    zc = (np.log(p / (1 - p)) - cfg["cls_logit_mu"]) / cfg["cls_logit_sd"]
    w = cfg["weights"]
    final = np.full(len(s1), -np.inf)
    final[esc] = w["probe"] * s1[esc] + w["classifier"] * zc[esc]
    return final > cfg["stage2_thr"], esc


def stage1_z(prompt_probe, reply_probe, cfg) -> np.ndarray:
    zp = (np.asarray(prompt_probe, dtype=float) - cfg["prompt_probe_mu"]) / cfg["prompt_probe_sd"]
    zr = (np.asarray(reply_probe, dtype=float) - cfg["reply_probe_mu"]) / cfg["reply_probe_sd"]
    return np.maximum(zp, np.nan_to_num(zr, nan=-np.inf))


# ----------------------------------------------------------------------------------------------------- probes
class LinearProbe(nn.Module):
    """z_t = W . (scale * psi_t) + b, per token; psi_t = concatenated block outputs. `scale` is a fixed per-layer
    1/RMS normaliser computed before training."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, psi):
        w = self.W * self.scale
        if psi.dtype in (torch.bfloat16, torch.float16):
            return (psi @ w.to(psi.dtype)).float() + self.b
        return psi.float() @ w + self.b


def ema_max(z: torch.Tensor, mask: torch.Tensor, M: int = 16) -> torch.Tensor:
    """Streaming score: EMA of per-token logits (alpha = 2/(M+1)) over scored positions; returns the max per row."""
    a = 2.0 / (M + 1)
    state = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
    started = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
    best = torch.full((z.shape[0],), NEG, dtype=z.dtype, device=z.device)
    for t in range(z.shape[1]):
        m = mask[:, t]
        new = torch.where(started, (1 - a) * state + a * z[:, t], z[:, t])
        state = torch.where(m, new, state)
        started = started | m
        best = torch.where(m, torch.maximum(best, state), best)
    return best


def encode(tok, prompt: str, reply: str | None, scored: str, max_len: int):
    """-> (ids, start, end). scored='prompt': the chat-formatted prompt incl. the assistant header; scored='reply': the
    reply tokens after it. If the exchange is longer than max_len, the prompt is cut from the left, then the reply."""
    head = tok(tok.apply_chat_template([{"role": "user", "content": str(prompt)}], tokenize=False,
                                       add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    resp = tok(str(reply), add_special_tokens=False)["input_ids"] if reply is not None else []
    if len(head) + len(resp) > max_len:
        head = head[-max(16, max_len - len(resp)):]
        resp = resp[: max_len - len(head)]
    if scored == "prompt":
        return head, 0, len(head)
    return head + resp, len(head), len(head) + len(resp)


# ----------------------------------------------------------------------------------------------------- classifier
class LoRAClassifier:
    """Causal LM + LoRA adapter + linear head on the last real token (sigmoid per category). Same as in
    EuroSafeAI/EuroSafeGuard's eurosafeguard.py, with an optional max_length override."""

    def __init__(self, path: str, device: str, base_model: str | None = None, max_length: int | None = None):
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = json.load(open(os.path.join(path, "train_config.json")))
        base_model = base_model or cfg["model_name"]
        self.device, self.max_length = device, max_length or cfg.get("max_length", 1024)
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
        lcfg = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                          lora_dropout=cfg.get("lora_dropout", 0.05),
                          target_modules=cfg.get("lora_target_modules") or ["q_proj", "v_proj"], bias="none",
                          task_type="CAUSAL_LM")
        self.backbone = get_peft_model(base, lcfg)
        set_peft_model_state_dict(self.backbone, load_file(os.path.join(path, "adapter_model.safetensors")))
        self.backbone = self.backbone.to(device).eval()
        self.head = nn.Linear(base.config.hidden_size, cfg["num_labels"])
        self.head.load_state_dict(torch.load(os.path.join(path, "classification_head.pt"), map_location="cpu",
                                             weights_only=True))
        self.head = self.head.to(device).eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"

    @torch.no_grad()
    def probs(self, texts: list[str], batch_size: int = 4) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], return_tensors="pt", truncation=True, max_length=self.max_length,
                           padding=True).to(self.device)
            hs = self.backbone(**enc, output_hidden_states=True).hidden_states[-1]
            last = enc["attention_mask"].shape[1] - 1 - enc["attention_mask"].flip(1).long().argmax(1)
            pooled = hs[torch.arange(hs.shape[0], device=hs.device), last]
            out.append(torch.sigmoid(self.head(pooled.to(self.head.weight.dtype)).float()).cpu().numpy())
        return np.concatenate(out, 0)


# ----------------------------------------------------------------------------------------------------- guard
class EuroSafeCCPP:
    def __init__(self, root: str, gate_root: str, seed: int = 0, devices=("cuda", "cuda"), load_lm: bool = True,
                 load_classifier: bool = True):
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.c = json.load(open(os.path.join(root, "thresholds", f"seed{seed}.json")))
        self.cfg = {k: json.load(open(os.path.join(root, k, "probe_config.json"))) for k in ("prompt_probe", "reply_probe")}
        pc = self.cfg["prompt_probe"]
        self.layers, self.M = pc["layers"], pc["M"]
        self.dev_lm, self.dev_cls = devices
        self.lm = self.cls = None
        if load_lm:
            self.lm_tok = AutoTokenizer.from_pretrained(pc["base_model"])
            self.lm = AutoModelForCausalLM.from_pretrained(pc["base_model"], dtype=torch.bfloat16).to(self.dev_lm).eval()
            self.probes = {}
            for k, c in self.cfg.items():
                p = LinearProbe(c["feature_dim"])
                p.load_state_dict(load_file(os.path.join(root, k, "probe.safetensors")))
                self.probes[k] = p.to(self.dev_lm).eval()
        if load_classifier:
            sc = self.c["second_stage"]
            self.cls = LoRAClassifier(os.path.join(gate_root, "gate", f"seed{seed}"), self.dev_cls,
                                      max_length=sc["max_length"])

    @classmethod
    def from_pretrained(cls, repo_or_path: str = "EuroSafeAI/EuroSafe-CCPP", seed: int = 0,
                        gate_repo_or_path: str = "EuroSafeAI/EuroSafeGuard", cache_dir: str | None = None, **kw):
        from huggingface_hub import snapshot_download

        root = repo_or_path if os.path.isdir(repo_or_path) else snapshot_download(
            repo_or_path, cache_dir=cache_dir,
            allow_patterns=["prompt_probe/*", "reply_probe/*", f"thresholds/seed{seed}.json", "*.py"])
        gate_root = gate_repo_or_path if os.path.isdir(gate_repo_or_path) else snapshot_download(
            gate_repo_or_path, cache_dir=cache_dir, allow_patterns=[f"gate/seed{seed}/*"])
        return cls(root, gate_root, seed, **kw)

    # -- probes
    @torch.no_grad()
    def probe_scores(self, prompts: list[str], replies: list[str] | None, which: str, batch_size: int = 4) -> np.ndarray:
        """Raw EMA-max logits of the prompt probe (which='prompt_probe') or the reply probe (which='reply_probe').
        An empty reply gets NaN from the reply probe (stage 1 then uses the prompt probe alone)."""
        if self.lm is None:
            raise RuntimeError("loaded with load_lm=False")
        c, probe = self.cfg[which], self.probes[which]
        scored = "prompt" if which == "prompt_probe" else "reply"
        out = np.full(len(prompts), np.nan, dtype=np.float32)
        todo = [i for i in range(len(prompts)) if scored == "prompt" or str(replies[i])]
        items = {i: encode(self.lm_tok, prompts[i], None if scored == "prompt" else replies[i], scored, c["max_len"])
                 for i in todo}
        todo.sort(key=lambda i: len(items[i][0]))
        pad = self.lm_tok.pad_token_id if self.lm_tok.pad_token_id is not None else self.lm_tok.eos_token_id
        for b in range(0, len(todo), batch_size):
            idx = todo[b:b + batch_size]
            T = max(len(items[i][0]) for i in idx)
            inp = torch.full((len(idx), T), pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            msk = torch.zeros((len(idx), T), dtype=torch.bool)
            for k, i in enumerate(idx):
                ids, st, en = items[i]
                inp[k, :len(ids)] = torch.tensor(ids); att[k, :len(ids)] = 1; msk[k, st:en] = True
            hs = self.lm(input_ids=inp.to(self.dev_lm), attention_mask=att.to(self.dev_lm),
                         output_hidden_states=True).hidden_states
            psi = torch.cat([hs[l] for l in self.layers], dim=-1)
            out[idx] = ema_max(probe(psi), msk.to(self.dev_lm), self.M).float().cpu().numpy()
        return out

    def classifier_probs(self, prompts: list[str], replies: list[str]) -> np.ndarray:
        """Second-stage classifier: max-head probability on the exchange."""
        if self.cls is None:
            raise RuntimeError("loaded with load_classifier=False")
        t = [self.c["second_stage"]["template"].format(prompt=a, reply=b) for a, b in zip(prompts, replies)]
        return self.cls.probs(t).max(1)

    # -- decisions
    def input_flag(self, prompt):
        """Prompt probe alone, thresholded to flag 5% of WildChat-clean (the paper's Table 2 configuration)."""
        single = isinstance(prompt, str)
        P = [prompt] if single else list(prompt)
        out = [bool(x) for x in self.probe_scores(P, None, "prompt_probe") > self.c["input_only_thr_5pct"]]
        return out[0] if single else out

    def escalate(self, prompt, reply):
        """Stage 1: True if max(z prompt probe, z reply probe) > stage1_thr."""
        single = isinstance(prompt, str)
        P, R = ([prompt], [reply]) if single else (list(prompt), list(reply))
        s1 = stage1_z(self.probe_scores(P, None, "prompt_probe"), self.probe_scores(P, R, "reply_probe"), self.c)
        out = [bool(x) for x in s1 > self.c["stage1_thr"]]
        return out[0] if single else out

    def scores(self, prompts: list[str], replies: list[str]) -> np.ndarray:
        """-> raw [N, 3] (prompt probe, reply probe, classifier prob); the classifier is NaN for non-escalated rows."""
        P, R = list(prompts), list(replies)
        S = np.full((len(P), 3), np.nan)
        S[:, 0] = self.probe_scores(P, None, "prompt_probe")
        S[:, 1] = self.probe_scores(P, R, "reply_probe")
        esc = np.flatnonzero(stage1_z(S[:, 0], S[:, 1], self.c) > self.c["stage1_thr"])
        if len(esc):
            S[esc, 2] = self.classifier_probs([P[i] for i in esc], [R[i] for i in esc])
        return S

    def check(self, prompt, reply):
        """True if the exchange (prompt, reply) should be blocked."""
        single = isinstance(prompt, str)
        P, R = ([prompt], [reply]) if single else (list(prompt), list(reply))
        S = self.scores(P, R)
        out = [bool(x) for x in ccpp_block(S[:, 0], S[:, 1], S[:, 2], self.c)[0]]
        return out[0] if single else out
