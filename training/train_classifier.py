"""Train an EuroSafeGuard classifier (the input classifier "gate" or the exchange classifier).

Qwen3.5-9B + LoRA + one sigmoid head per harm category, trained with masked multi-label BCE by the TACTIC trainer
(https://github.com/Leonhard17/TACTIC, pinned in pyproject.toml). This script adds three things on top of it:

  * twin consistency: each attacked prompt is put in the batch next to its plain version (same input_id, source
    "plain") and lam * (1 - cos(h_attacked, stopgrad(h_plain))) is added to the loss, h = last-token representation;
  * benign share: each draw is a benign row with probability --benign_frac, otherwise a harm category with
    probability proportional to sqrt(#rows);
  * length bucketing (token budget + sample cap per micro-batch), data parallelism over N GPUs with a manual gradient
    all-reduce (effective batch = batch_size * N, learning rates scaled by sqrt(N)), optional gradient checkpointing.

    python training/train_classifier.py --config configs/gate.json --dataset_path gate.csv --seed 0 \
        --checkpoint_dir runs/gate_s0
    torchrun --nproc_per_node=3 training/train_classifier.py --config configs/exchange.json \
        --dataset_path exchange.csv --checkpoint_dir runs/exchange

Every other flag is passed to TACTIC's classifier CLI (`python -m tactic.classifier.cli --help`). The checkpoint folder
(adapter_model.safetensors, classification_head.pt, train_config.json, tokenizer) is the format eurosafeguard loads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

RANK, WORLD = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=False)
    ap.add_argument("--config", help="JSON file of TACTIC flags (configs/gate.json, configs/exchange.json)")
    ap.add_argument("--consistency", type=float, default=None, help="twin-consistency weight lam (0 disables)")
    ap.add_argument("--benign_frac", type=float, default=None, help="share of benign draws (0: TACTIC sampling)")
    ap.add_argument("--grad_checkpointing", action="store_true", default=None)
    ap.add_argument("--token_budget", type=int, default=1800, help="max (#rows x longest row) per micro-batch")
    ap.add_argument("--max_rows", type=int, default=4, help="max rows per micro-batch before twins are added")
    ap.add_argument("-h", "--help", action="store_true")
    a, rest = ap.parse_known_args()
    if a.help:
        ap.print_help()
        print("\nAll other flags go to TACTIC's classifier CLI.")
        sys.exit(0)
    cfg = {k: v for k, v in (json.load(open(a.config)) if a.config else {}).items() if not k.startswith("_")}
    ours = cfg.pop("eurosafeguard", {})
    for k in ("consistency", "benign_frac", "grad_checkpointing"):
        if getattr(a, k) is None:
            setattr(a, k, ours.get(k, {"consistency": 0.0, "benign_frac": 0.0, "grad_checkpointing": False}[k]))
    flags = []
    given = {x.split("=")[0] for x in rest if x.startswith("--")}
    for k, v in cfg.items():                       # config values, unless overridden on the command line
        if f"--{k}" in given:
            continue
        if isinstance(v, bool):
            flags.append(f"--{k}" if v else f"--no-{k}")
        else:
            flags += [f"--{k}", str(v)]
    return a, flags + rest


def flag(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def set_flag(argv, name, value):
    if name in argv:
        argv[argv.index(name) + 1] = value


class _AddLoss(torch.autograd.Function):
    """Identity on x whose backward also sends gradient 1 to ``aux``: the same gradients as adding ``aux`` to the
    training loss, without changing the TACTIC training loop."""

    @staticmethod
    def forward(ctx, x, aux):
        ctx.aux_shape = aux.shape
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g, torch.ones(ctx.aux_shape, dtype=torch.float32, device=g.device)


def main():
    a, argv = parse_args()
    if WORLD > 1:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group("nccl")
        set_flag(argv, "--seed", str(int(flag(argv, "--seed", 0)) + RANK))       # different data on every rank
        for f in ("--lr_backbone", "--lr_head"):
            if f in argv:
                set_flag(argv, f, repr(float(flag(argv, f)) * math.sqrt(WORLD)))
    if RANK != 0:
        os.environ["WANDB_MODE"] = "disabled"

    import tactic.classifier.trainer as T
    import tactic.common.checkpoint as C
    from tactic.classifier.cli import main as tactic_main
    from tactic.classifier.model import Classifier, last_token_pool
    from tactic.data.types import Sample

    if RANK != 0:                                  # only rank 0 writes checkpoints
        for n in ("save", "save_best_meta", "save_sampler_state"):
            if hasattr(C, n):
                setattr(C, n, lambda *x, **k: None)

    df = pd.read_csv(flag(argv, "--dataset_path"), dtype=str, keep_default_na=False)
    text_col = flag(argv, "--dataset_text_column", "prompt")
    twin = {}
    if a.consistency > 0 and {"source", "input_id"} <= set(df.columns):
        plain = df[df.source == "plain"].drop_duplicates("input_id").set_index("input_id")[text_col]
        twin = {t: plain[i] for t, s, i in zip(df[text_col], df.source, df.input_id)
                if s != "plain" and i in plain.index and plain[i] != t}
    if RANK == 0:
        print(f"[eurosafeguard] world={WORLD} | {len(twin)} attacked rows with a plain twin | "
              f"consistency={a.consistency} | benign_frac={a.benign_frac}", flush=True)

    lam = a.consistency

    class TwinClassifier(Classifier):
        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                                logits_to_keep=1)
            h = last_token_pool(out.hidden_states[-1], attention_mask)
            logits = self.head(h.to(self.head.weight.dtype))
            pairs, self._pairs = getattr(self, "_pairs", None), None
            if self.training and pairs and torch.is_grad_enabled():
                i = torch.tensor([p[0] for p in pairs], device=h.device)
                j = torch.tensor([p[1] for p in pairs], device=h.device)
                cons = (1 - F.cosine_similarity(h[i].float(), h[j].detach().float(), dim=-1)).mean()
                logits = _AddLoss.apply(logits, lam * cons)
            return logits

    load_model = T.ClassifierTrainer.load_model

    def _load_model(self):
        load_model(self)
        self.model.__class__ = TwinClassifier
        if a.grad_checkpointing:
            bb = self.model.backbone
            bb.enable_input_require_grads()
            (bb.get_base_model() if hasattr(bb, "get_base_model") else bb).gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if WORLD > 1:                              # synchronous data parallel: average gradients before each step
            def average(q):
                if q.grad is not None:
                    dist.all_reduce(q.grad)
                    q.grad.div_(WORLD)
            for p in self.model.parameters():
                if p.requires_grad:
                    p.register_post_accumulate_grad_hook(average)
    T.ClassifierTrainer.load_model = _load_model

    tokenize = T.ClassifierTrainer._tokenize_batch

    def _tokenize_batch(self, samples):
        if not self.model.training or not twin:
            return tokenize(self, samples)
        ext, pairs = list(samples), []
        for k, s in enumerate(samples):
            p = twin.get(s.text)
            if p is not None:
                pairs.append((k, len(ext)))
                ext.append(Sample(text=p, label=s.label, category=s.category, complexity=0, method=None,
                                  entry_type=s.entry_type))
        self.model._pairs = pairs
        return tokenize(self, ext)
    T.ClassifierTrainer._tokenize_batch = _tokenize_batch

    collect = T.collect_batch
    if a.benign_frac > 0:
        counts = df.category.value_counts()
        harm = [c for c in counts.index if c != "Benign"]
        w = np.cumsum(np.sqrt(counts[harm].values.astype(float)))
        w /= w[-1]

        def collect(batch_size, sampler, sample_categories, sample_complexity, *, balanced=False):
            rng = getattr(sampler, "_rng", None) or random
            return [sampler(category="Benign" if rng.random() < a.benign_frac
                            else harm[int(np.searchsorted(w, rng.random()))]) for _ in range(batch_size)]

    est = lambda s: min(len(s.text) // 4, 1024)       # noqa: E731  (token estimate, capped as in the paper runs)
    buf = []

    def collect_bucketed(batch_size, sampler, sample_categories, sample_complexity, *, balanced=False):
        """Draw 24 batches at once, sort by length and serve length-homogeneous chunks."""
        if not buf:
            pool = sorted(collect(24 * batch_size, sampler, sample_categories, sample_complexity, balanced=balanced),
                          key=est)
            chunks, cur, longest = [], [], 0
            for s in pool:
                n = est(s)
                if cur and (len(cur) >= a.max_rows or (len(cur) + 1) * max(longest, n) > a.token_budget):
                    chunks.append(cur)
                    cur, longest = [], 0
                cur.append(s)
                longest = max(longest, n)
            chunks.append(cur)
            (getattr(sampler, "_rng", None) or random).shuffle(chunks)
            buf.extend(chunks)
        return buf.pop()
    T.collect_batch = collect_bucketed

    sys.argv = ["tactic.classifier.cli"] + argv
    tactic_main()
    if WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
