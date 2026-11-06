"""Command-line interface: ``eurosafeguard {check,score,calibrate}``.

  eurosafeguard check --prompt "How do I bake bread?" [--reply "..."] [--mode full]
  eurosafeguard score prompts.csv scores.csv [--reply-col reply]
  eurosafeguard calibrate --reference wildchat_scores.csv --set xstest=xstest_scores.csv:0.02 --out my_thresholds/

``score`` writes raw component scores (probe, gate, gate_top, exchange) and never prints the texts. ``calibrate`` reads
such score files and writes ``seed{k}.json`` (alert mode) and ``full_seed{k}.json`` (full mode) for ``--thresholds``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load(a, load_exchange: bool):
    from .guard import EuroSafeGuard

    devices = tuple(a.devices.split(","))
    if len(devices) == 1:
        devices = devices * 3
    g = EuroSafeGuard.from_pretrained(a.model, seed=a.seed, devices=devices, load_exchange=load_exchange,
                                      mode=a.mode)
    if getattr(a, "thresholds", None):
        g.c = json.load(open(os.path.join(a.thresholds, f"seed{a.seed}.json")))
        full = os.path.join(a.thresholds, f"full_seed{a.seed}.json")
        if a.mode == "full":
            g.cf = json.load(open(full))
    return g


def cmd_check(a):
    g = _load(a, load_exchange=a.reply is not None)
    out = {"mode": a.mode, "seed": a.seed}
    if a.mode == "alert":
        decision, category, score = g.route(a.prompt)
        out.update(decision=decision, top_category=category, input_score=round(score, 4),
                   alert_input_thr=round(g.c["alert_input_thr"], 4))
        if a.reply is not None:
            out["reply_blocked"] = decision == "refuse" or g.check_reply(a.prompt, a.reply, decision)
    else:
        if a.reply is None:
            sys.exit("--mode full needs --reply: the full rule scores the prompt together with the reply")
        raw = g.full_scores([a.prompt], [a.reply])[0]
        out.update(escalated=bool(not np.isnan(raw[1])), probe=float(raw[0]),
                   gate=None if np.isnan(raw[1]) else float(raw[1]),
                   exchange=None if np.isnan(raw[2]) else float(raw[2]),
                   blocked=bool(g.check_full(a.prompt, a.reply)))
    print(json.dumps(out, indent=1))


def cmd_score(a):
    import pandas as pd

    df = pd.read_csv(a.input, dtype=str, keep_default_na=False)
    g = _load(a, load_exchange=a.reply_col is not None)
    prompts = df[a.prompt_col].tolist()
    out = pd.DataFrame({c: df[c] for c in a.keep if c in df})
    out["probe"] = g.probe_scores(prompts, batch_size=a.batch_size)
    H = g.gate.probs(prompts, batch_size=a.batch_size)
    from .guard import HEADS
    out["gate"], out["gate_top"] = H.max(1), [HEADS[i] for i in H.argmax(1)]
    if a.reply_col is not None:
        out["exchange"] = g.exchange_scores(prompts, df[a.reply_col].tolist())
    out.to_csv(a.output, index=False)
    print(f"wrote {len(out)} rows to {a.output}")


def _read_scores(path):
    import pandas as pd

    d = pd.read_csv(path)
    missing = {"probe", "gate"} - set(d.columns)
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)} (use `eurosafeguard score` to make score files)")
    return d


def cmd_calibrate(a):
    from .calibration import CalibrationSet, fit_alert, fit_full, hash_half

    ref = _read_scores(a.reference)
    if "exchange" not in ref:
        sys.exit("the reference file needs an exchange column (score it with --reply-col)")
    sets = [CalibrationSet("reference", ref.probe.values, ref.gate.values, a.reference_cap, ref.exchange.values)]
    for spec in a.set:
        name, rest = spec.split("=", 1)
        path, cap = rest.rsplit(":", 1)
        d = _read_scores(path)
        fit = None
        if a.fit_half and "id" in d:
            fit = np.array([hash_half(i) == 0 for i in d["id"]])
        sets.append(CalibrationSet(name, d.probe.values, d.gate.values, float(cap),
                                   d.exchange.values if "exchange" in d else None, fit))
    full = fit_full(ref.probe.values, ref.gate.values, ref.exchange.values, sets,
                    escalation_quantile=a.escalation_quantile)
    alert = fit_alert(ref.probe.values, ref.gate.values, ref.exchange.values, input_budget=a.alert_budget)
    os.makedirs(a.out, exist_ok=True)
    from .guard import HEADS
    alert["heads"] = HEADS
    json.dump(alert | {"seed": a.seed}, open(os.path.join(a.out, f"seed{a.seed}.json"), "w"), indent=1)
    json.dump(full | {"seed": a.seed}, open(os.path.join(a.out, f"full_seed{a.seed}.json"), "w"), indent=1)
    print(json.dumps({"full_thr": full["full_thr"], "binding_set": full["binding_set"],
                      "probe_escalation_z": full["probe_escalation_z"],
                      "alert_input_thr": alert["alert_input_thr"]}, indent=1))
    print(f"wrote {a.out}/seed{a.seed}.json and {a.out}/full_seed{a.seed}.json; load them with --thresholds {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="eurosafeguard", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="EuroSafeAI/EuroSafeGuard-v1", help="Hugging Face repo or local folder")
        p.add_argument("--seed", type=int, default=0, help="input-classifier training run (gate/seed0..4)")
        p.add_argument("--mode", choices=("alert", "full"), default="alert")
        p.add_argument("--devices", default="cuda", help="one device, or three: probe LM, gate, exchange")
        p.add_argument("--thresholds", help="folder with your own seed{k}.json / full_seed{k}.json")

    p = sub.add_parser("check", help="decide on one prompt (and optionally its reply)")
    common(p)
    p.add_argument("--prompt", required=True)
    p.add_argument("--reply")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("score", help="write raw component scores for a CSV of prompts (and replies)")
    common(p)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--prompt-col", default="prompt")
    p.add_argument("--reply-col")
    p.add_argument("--keep", nargs="*", default=["id"], help="columns copied to the output (default: id)")
    p.add_argument("--batch-size", type=int, default=8)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("calibrate", help="fit alert- and full-mode thresholds from score files")
    p.add_argument("--reference", required=True,
                   help="scores of clean everyday traffic with replies (sets z-scores and the escalation cut)")
    p.add_argument("--reference-cap", type=float, default=0.005)
    p.add_argument("--set", action="append", default=[], metavar="NAME=SCORES.csv:CAP",
                   help="benign calibration set and its cap on the share blocked (repeatable)")
    p.add_argument("--escalation-quantile", type=float, default=0.9)
    p.add_argument("--alert-budget", type=float, default=0.05)
    p.add_argument("--fit-half", action="store_true", help="fit on the hash half 0 of each set's id column")
    p.add_argument("--seed", type=int, default=0, help="the gate run these scores came from")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_calibrate)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
