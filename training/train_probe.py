"""Train an activation probe on a protected model's hidden states and export it in the released format.

EuroSafeGuard's probe (paper Sec. 3.3): MLP, hidden 256, all 32 blocks of Llama-3.1-8B-Instruct, prompt-only rows,
one epoch, AdamW lr 1e-3, batch 8, max 2048 tokens. The CC++ linear probe of EuroSafe-CC++ is ``--kind linear``.

    python training/train_probe.py \
        --config configs/probe.json --train probe_train.csv --val probe_val.csv --out runs/probe
    # -> runs/probe/probe.safetensors + probe_config.json, the format of the released probe/ folder

Input CSV columns: prompt, label (0/1), and optionally response (empty for prompt-only rows; with a response only the
response tokens are scored, as in the CC++ reply probe), source and input_id (an attacked row and the row with the
same input_id and source "plain" are put in the same batch). See data/prepare.py.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe_lib import activations, ema_scores, encode, layer_rms_scale, make_probe, probe_loss


def load_items(tok, df, max_len):
    resp = df["response"] if "response" in df else [""] * len(df)
    return [encode(tok, p, r or None, max_len) for p, r in zip(df.prompt, resp)]


def plain_twins(df) -> dict[int, int]:
    """Row index of each attacked prompt-only row -> row index of its plain twin (same input_id, source 'plain')."""
    if not {"source", "input_id"} <= set(df.columns):
        return {}
    resp = df["response"] if "response" in df else pd.Series([""] * len(df))
    plain = {iid: i for i, iid, s, r in zip(df.index, df.input_id, df.source, resp) if s == "plain" and not r}
    return {i: plain[iid] for i, iid, s, r in zip(df.index, df.input_id, df.source, resp)
            if s != "plain" and not r and iid in plain}


@torch.no_grad()
def score(model, tok, probe, items, layers, M, bs):
    out = np.zeros(len(items), dtype=np.float32)
    order = np.argsort([len(i[0]) for i in items])
    for b in range(0, len(order), bs):
        bi = order[b:b + bs].tolist()
        psi, sc = activations(model, tok, [items[j] for j in bi], layers)
        out[bi] = ema_scores(probe(psi), sc, M).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file of defaults (configs/probe.json); command-line flags override it")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=("mlp", "linear"), default="mlp")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", default="all", help="'all' (every block output) or comma-separated hidden_states indices")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--M", type=int, default=16, help="smoothing window (training) / EMA span (inference)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_twins", action="store_true", help="do not add plain twins to the batches")
    a, _ = ap.parse_known_args()
    if a.config:
        ap.set_defaults(**{k: v for k, v in json.load(open(a.config)).items() if not k.startswith("_")})
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    layers = list(range(1, n_layers + 1)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    dim = len(layers) * model.config.get_text_config().hidden_size

    tr = pd.read_csv(a.train, dtype=str, keep_default_na=False).reset_index(drop=True)
    va = pd.read_csv(a.val, dtype=str, keep_default_na=False)
    y_tr, y_va = tr.label.astype(int).values, va.label.astype(int).values
    tri, vai = load_items(tok, tr, a.max_len), load_items(tok, va, a.max_len)
    twin = {} if a.no_twins else plain_twins(tr)
    print(f"{len(tr)} train rows, {len(va)} val rows, {len(twin)} attacked rows with a plain twin", flush=True)

    probe = make_probe(a.kind, dim, a.hidden).cuda()
    pick = rng.choice(len(tri), min(32, len(tri)), replace=False).tolist()
    probe.scale.copy_(layer_rms_scale(model, tok, [tri[j] for j in pick], layers, a.bs))
    opt = torch.optim.AdamW(probe.parameters(), lr=a.lr, weight_decay=1e-2)

    best_auc, best_state, step = -1.0, None, 0
    for ep in range(a.epochs):
        order = rng.permutation(len(tri)).tolist()
        for b in range(0, len(order), a.bs):
            bi = order[b:b + a.bs]
            for i in list(bi):                       # attacked rows train next to their plain version
                t = twin.get(i)
                if t is not None and t not in bi:
                    bi.append(t)
            psi, sc = activations(model, tok, [tri[j] for j in bi], layers)
            y = torch.tensor(y_tr[bi], dtype=torch.float32, device="cuda")
            loss = probe_loss(probe(psi), sc, y, a.M)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"epoch {ep + 1} step {step} loss {loss.item():.4f}", flush=True)
        auc = roc_auc_score(y_va, score(model, tok, probe.eval(), vai, layers, a.M, a.bs))
        probe.train()
        print(f"epoch {ep + 1}: val AUROC {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.detach().clone() for k, v in probe.state_dict().items()}

    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.float().contiguous().cpu() for k, v in best_state.items()}, os.path.join(a.out, "probe.safetensors"))
    cfg = {"kind": a.kind, "hidden": a.hidden, "activation": "gelu", "base_model": a.model, "layers": layers,
           "hidden_size": dim // len(layers), "feature_dim": dim,
           "scored_positions": "prompt-only rows: the chat-formatted prompt up to the assistant header; "
                               "rows with a response: the response tokens",
           "max_len": a.max_len, "inference": "EMA of per-token logits, alpha = 2/(M+1), score = max over positions",
           "M": a.M, "val_auroc": best_auc}
    json.dump(cfg, open(os.path.join(a.out, "probe_config.json"), "w"), indent=1)
    print(f"saved {a.out} (val AUROC {best_auc:.4f})")


if __name__ == "__main__":
    main()
