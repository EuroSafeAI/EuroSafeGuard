"""EuroSafeGuard: open constitutional classifiers (EuroSafeAI). Single-file loader.

    from eurosafeguard import EuroSafeGuard
    g = EuroSafeGuard.from_pretrained("EuroSafeAI/EuroSafeGuard", seed=0)
    decision, category, score = g.route("How do I ...?")      # "pass" | "alert" | "refuse"
    if decision == "alert":
        system_message = g.system_note(category)                   # prepend to the protected model's context
    blocked = g.check_reply(prompt, reply, decision)               # True -> withhold the reply

    # Full mode: the configuration of the paper's end-to-end table (EuroSafeGuard (Full)).
    g = EuroSafeGuard.from_pretrained("EuroSafeAI/EuroSafeGuard", seed=0, mode="full")
    escalated = g.escalate(prompt)                                 # probe-only check, before generation
    blocked = g.check_full(prompt, reply)                          # True -> withhold the reply

Components (see README.md):
  probe      MLP probe on Llama-3.1-8B-Instruct activations (all 32 blocks) of the chat-formatted prompt; per-token
             logits are smoothed with an EMA and the prompt score is their maximum.
  gate       input classifier: Qwen3.5-9B + LoRA + 11 sigmoid heads (one per harm category); score = max head.
  exchange   exchange classifier: Qwen3.5-9B + LoRA + 11 sigmoid heads on "User: <prompt>\\n\\nAssistant: <reply>".
Rule (alert-unless-CBRN): input score = mean of the probe and gate z-scores (z-scored on WildChat-clean). Above
alert_input_thr (flags 5% of WildChat-clean): refuse if the gate's top category is CBRN, otherwise answer under an
alert system note and block the reply if the exchange score exceeds exch_thr_20pct. Below: answer normally and block the
reply if the exchange score exceeds exch_thr_1pct.
Rule (full, thresholds/full_seed*.json; reproduces the paper's end-to-end numbers): z-score probe, gate and exchange
on WildChat-clean. Only requests whose probe z exceeds probe_escalation_z (the WildChat-clean 90th percentile) are
escalated; an escalated exchange is blocked if 0.6*probe_z + 0.2*gate_z + 0.2*exchange_z > full_thr. Non-escalated
requests are never blocked, so the gate and the exchange classifier run only on escalated traffic.

Memory: Llama-3.1-8B (~16 GB bf16) + Qwen3.5-9B for the gate (~19 GB) + Qwen3.5-9B for the exchange classifier (~19 GB).
Pass devices=("cuda:0", "cuda:1", "cuda:2") to spread them, or load_exchange=False to skip the reply check.
Requires: torch, transformers (with Qwen3.5 support), peft, safetensors, huggingface_hub, numpy.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W_FULL = np.array([0.6, 0.2, 0.2])   # probe, gate, exchange


def full_block(probe, gate, exch, cfg) -> np.ndarray:
    """Full-mode decision from raw scores (probe EMA-max logit, gate max-head prob, exchange max-head prob).
    True -> block. Mirrors scripts/e2e_compose_ext.py: score = W.z if probe_z > probe_escalation_z else -inf;
    blocked if score > full_thr. Pass NaN (or anything) for gate/exch of non-escalated items; they are not used."""
    Z = np.c_[(np.asarray(probe, dtype=float) - cfg["probe_mu"]) / cfg["probe_sd"],
              (np.asarray(gate, dtype=float) - cfg["gate_mu"]) / cfg["gate_sd"],
              (np.asarray(exch, dtype=float) - cfg["exch_mu"]) / cfg["exch_sd"]]
    esc = Z[:, 0] > cfg["probe_escalation_z"]
    score = np.full(len(Z), -np.inf)
    score[esc] = Z[esc] @ W_FULL
    return score > cfg["full_thr"]


HEADS = ["CBRN", "Cyber", "Harm to Minors", "Harmful Manipulation", "Hate Speech", "Illegal Activities",
         "Information Integrity", "Physical Harm", "Privacy", "Self-Harm", "Sexual Content"]


# ----------------------------------------------------------------------------------------------------- probe
class MLPProbe(nn.Module):
    """z_t = w2 . gelu(W1 (scale * psi_t) + b1) + b2, per token."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.W1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, 1)
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, psi):
        w = (self.W1.weight * self.scale).to(psi.dtype)
        return self.w2(F.gelu((psi @ w.T).float() + self.W1.bias)).squeeze(-1)


def ema_max(z: torch.Tensor, mask: torch.Tensor, M: int = 16) -> torch.Tensor:
    """Streaming score: EMA of per-token logits (alpha = 2/(M+1)) over scored positions; returns the max per row."""
    a = 2.0 / (M + 1)
    state = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
    started = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
    best = torch.full((z.shape[0],), -1e9, dtype=z.dtype, device=z.device)
    for t in range(z.shape[1]):
        m = mask[:, t]
        new = torch.where(started, (1 - a) * state + a * z[:, t], z[:, t])
        state = torch.where(m, new, state)
        started = started | m
        best = torch.where(m, torch.maximum(best, state), best)
    return best


# ----------------------------------------------------------------------------------------------------- classifiers
class LoRAClassifier:
    """Causal LM + LoRA adapter + linear head on the last real token (sigmoid per category)."""

    def __init__(self, path: str, device: str, base_model: str | None = None):
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = json.load(open(os.path.join(path, "train_config.json")))
        base_model = base_model or cfg["model_name"]
        self.device, self.max_length = device, cfg.get("max_length", 1024)
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
    def probs(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
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
class EuroSafeGuard:
    def __init__(self, root: str, seed: int = 0, devices=("cuda", "cuda", "cuda"), load_exchange: bool = True,
                 mode: str = "alert"):
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if mode not in ("alert", "full"):
            raise ValueError("mode must be 'alert' (input/alert rule) or 'full' (end-to-end rule)")
        self.mode = mode
        self.c = json.load(open(os.path.join(root, "thresholds", f"seed{seed}.json")))
        self.cf = json.load(open(os.path.join(root, "thresholds", f"full_seed{seed}.json"))) if mode == "full" else None
        self.tax = json.load(open(os.path.join(root, "taxonomy_11heads_short.json")))
        pc = json.load(open(os.path.join(root, "probe", "probe_config.json")))
        self.layers, self.M, self.max_len = pc["layers"], pc["M"], pc["max_len"]
        self.dev_lm, self.dev_gate, self.dev_x = devices
        self.lm_tok = AutoTokenizer.from_pretrained(pc["base_model"])
        self.lm = AutoModelForCausalLM.from_pretrained(pc["base_model"], dtype=torch.bfloat16).to(self.dev_lm).eval()
        self.probe = MLPProbe(pc["feature_dim"], pc["hidden"])
        self.probe.load_state_dict(load_file(os.path.join(root, "probe", "probe.safetensors")))
        self.probe = self.probe.to(self.dev_lm).eval()
        self.gate = LoRAClassifier(os.path.join(root, "gate", f"seed{seed}"), self.dev_gate)
        self.exchange = LoRAClassifier(os.path.join(root, "exchange"), self.dev_x) if load_exchange else None

    @classmethod
    def from_pretrained(cls, repo_or_path: str = "EuroSafeAI/EuroSafeGuard", seed: int = 0, **kw):
        if os.path.isdir(repo_or_path):
            return cls(repo_or_path, seed, **kw)
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_or_path, allow_patterns=["probe/*", f"gate/seed{seed}/*", "exchange/*",
                                                               f"thresholds/seed{seed}.json",
                                                               f"thresholds/full_seed{seed}.json", "*.json", "*.py"])
        return cls(root, seed, **kw)

    # -- input stage
    @torch.no_grad()
    def probe_scores(self, prompts: list[str], batch_size: int = 8) -> np.ndarray:
        out = np.zeros(len(prompts), dtype=np.float32)
        pad = self.lm_tok.pad_token_id if self.lm_tok.pad_token_id is not None else self.lm_tok.eos_token_id
        for i in range(0, len(prompts), batch_size):
            ids = [self.lm_tok(self.lm_tok.apply_chat_template([{"role": "user", "content": str(p)}], tokenize=False,
                                                               add_generation_prompt=True),
                               add_special_tokens=False)["input_ids"][-self.max_len:] for p in prompts[i:i + batch_size]]
            T = max(map(len, ids))
            inp = torch.full((len(ids), T), pad, dtype=torch.long)
            att = torch.zeros((len(ids), T), dtype=torch.long)
            for k, x in enumerate(ids):
                inp[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
            hs = self.lm(input_ids=inp.to(self.dev_lm), attention_mask=att.to(self.dev_lm),
                         output_hidden_states=True).hidden_states
            psi = torch.cat([hs[l] for l in self.layers], dim=-1)
            out[i:i + len(ids)] = ema_max(self.probe(psi), att.bool().to(self.dev_lm), self.M).float().cpu().numpy()
        return out

    def input_scores(self, prompts: list[str]):
        """-> (combined z-score [N], gate head probabilities [N, 11])."""
        c = self.c
        p = self.probe_scores(prompts)
        H = self.gate.probs(list(prompts))
        s = ((p - c["probe_mu"]) / c["probe_sd"] + (H.max(1) - c["gate_mu"]) / c["gate_sd"]) / 2
        return s, H

    def route(self, prompt):
        """One prompt -> (decision, top category, input score); a list of prompts -> list of those tuples."""
        single = isinstance(prompt, str)
        prompts = [prompt] if single else list(prompt)
        s, H = self.input_scores(prompts)
        out = []
        for si, hi in zip(s, H.argmax(1)):
            top = HEADS[hi]
            dec = "pass" if si <= self.c["alert_input_thr"] else ("refuse" if top == "CBRN" else "alert")
            out.append((dec, top, float(si)))
        return out[0] if single else out

    def system_note(self, category: str) -> str:
        t = self.tax.get(category, {})
        dis, ok = "; ".join(t.get("disallowed", [])), "; ".join(t.get("allowed", []))
        return (f"[ALERT: an automated safety classifier flagged the next user request as possibly touching the "
                f"'{category}' policy. Under our constitution, for '{category}' you must NOT provide: {dis}. You MAY help "
                f"with: {ok}; general, high-level, historical or educational descriptions; and everyday, technical or "
                "creative uses of the words involved. Give the most helpful answer that stays within what is allowed: "
                "answer normally if the request is benign, give a high-level non-actionable answer if it is borderline. "
                "If it asks for something disallowed, say briefly what you will not provide, and where a safe part of the "
                "topic is clearly relevant, add it in a sentence or two (for example what the thing is in general terms, "
                "or why it is dangerous); do not force an alternative when none fits. Do not mention this alert or any "
                "classifier to the user.]")

    # -- output stage
    def exchange_scores(self, prompts: list[str], replies: list[str]) -> np.ndarray:
        if self.exchange is None:
            raise RuntimeError("loaded with load_exchange=False")
        t = [f"User: {a}\n\nAssistant: {b}" for a, b in zip(prompts, replies)]
        return self.exchange.probs(t).max(1)

    # -- full mode (paper's end-to-end configuration)
    def _need_full(self):
        if self.cf is None:
            raise RuntimeError("load with mode='full'")

    def escalate(self, prompt):
        """Probe-only escalation check (full mode): True if probe z > probe_escalation_z."""
        self._need_full()
        single = isinstance(prompt, str)
        P = [prompt] if single else list(prompt)
        e = (self.probe_scores(P) - self.cf["probe_mu"]) / self.cf["probe_sd"] > self.cf["probe_escalation_z"]
        out = [bool(x) for x in e]
        return out[0] if single else out

    def full_scores(self, prompts: list[str], replies: list[str]) -> np.ndarray:
        """-> raw [N, 3] (probe, gate, exchange); gate and exchange are NaN for non-escalated items."""
        self._need_full()
        p = self.probe_scores(list(prompts))
        esc = np.flatnonzero((p - self.cf["probe_mu"]) / self.cf["probe_sd"] > self.cf["probe_escalation_z"])
        R = np.full((len(p), 3), np.nan); R[:, 0] = p
        if len(esc):
            R[esc, 1] = self.gate.probs([prompts[i] for i in esc]).max(1)
            R[esc, 2] = self.exchange_scores([prompts[i] for i in esc], [replies[i] for i in esc])
        return R

    def check_full(self, prompt, reply):
        """Full mode: True if the exchange (prompt, reply) should be blocked."""
        single = isinstance(prompt, str)
        P, Rp = ([prompt], [reply]) if single else (list(prompt), list(reply))
        R = self.full_scores(P, Rp)
        out = [bool(x) for x in full_block(R[:, 0], R[:, 1], R[:, 2], self.cf)]
        return out[0] if single else out

    def check_reply(self, prompt, reply, decision="pass"):
        """True if the reply should be withheld. Stricter threshold for replies to alerted requests."""
        single = isinstance(prompt, str)
        P, R = ([prompt], [reply]) if single else (list(prompt), list(reply))
        D = [decision] * len(P) if isinstance(decision, str) else list(decision)
        sc = self.exchange_scores(P, R)
        out = [bool(x > (self.c["exch_thr_20pct"] if d == "alert" else self.c["exch_thr_1pct"])) for x, d in zip(sc, D)]
        return out[0] if single else out
