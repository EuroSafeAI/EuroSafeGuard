"""Fit EuroSafeGuard thresholds to your own traffic.

All functions take raw component scores, as returned by ``EuroSafeGuard.probe_scores``, ``gate.probs(...).max(1)`` and
``exchange_scores`` (or by ``eurosafeguard score-file``), and return a dictionary in the format of the released
``thresholds/seed{k}.json`` (alert mode) and ``thresholds/full_seed{k}.json`` (full mode), which ``EuroSafeGuard`` loads.

Paper recipe (Section 3.7): z-score every component on clean everyday traffic (WildChat-clean); escalate the requests
whose probe z-score is above the 90th percentile of that traffic; set the full-mode threshold to the lowest value that
keeps the block rate of every benign calibration set under its own cap, measured on a fit half of each set.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .guard import W_FULL

#: Per-benchmark caps on the share of benign prompts blocked, as in the paper.
PAPER_CAPS = {
    "everyday": 0.005,        # WildChat-clean, Alpaca, Dolly, OASST, GSM8K, MMLU
    "sensitive": 0.02,        # XSTest-safe, OR-Bench-80k
    "hard": 0.05,             # OR-Bench-hard
    "wrapped": 0.40,          # benign prompts wrapped in jailbreak templates
}


def zstats(x) -> tuple[float, float]:
    """Mean and (population) standard deviation used to z-score a component."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std())


def hash_half(key) -> int:
    """Deterministic 0/1 split used for fit and report halves: sha1(str(key)) mod 2."""
    return int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % 2


@dataclass
class CalibrationSet:
    """Benign traffic whose block rate must stay at or below ``cap``.

    ``exchange`` may be None for prompt-only sets (no replies available); its z-score is then 0, the benign mean.
    ``fit`` selects the rows used to fit the threshold (default: all rows).
    """
    name: str
    probe: np.ndarray
    gate: np.ndarray
    cap: float
    exchange: np.ndarray | None = None
    fit: np.ndarray | None = None


def full_scores(cfg: dict, probe, gate, exchange) -> np.ndarray:
    """Combined full-mode score: ``W . z`` for escalated requests and ``-inf`` for the rest."""
    probe = np.asarray(probe, dtype=float)
    z = np.c_[(probe - cfg["probe_mu"]) / cfg["probe_sd"],
              (np.asarray(gate, dtype=float) - cfg["gate_mu"]) / cfg["gate_sd"],
              (np.asarray(exchange, dtype=float) - cfg["exch_mu"]) / cfg["exch_sd"]]
    w = np.array([cfg["weights"][k] for k in ("probe", "gate", "exchange")]) if "weights" in cfg else W_FULL
    out = np.full(len(z), -np.inf)
    esc = z[:, 0] > cfg["probe_escalation_z"]
    out[esc] = z[esc] @ w
    return out


def fit_full(reference_probe, reference_gate, reference_exchange, sets: list[CalibrationSet],
             escalation_quantile: float = 0.9, weights=(0.6, 0.2, 0.2)) -> dict:
    """Full-mode configuration (``thresholds/full_seed{k}.json``).

    reference_*: raw scores on clean everyday traffic (prompts and the protected model's replies to them); they set
    the z-score statistics and the probe escalation cut. ``sets``: benign calibration sets with their caps; the
    threshold is the lowest one that meets every cap on the fit rows.
    """
    (pm, ps), (gm, gs), (xm, xs) = (zstats(reference_probe), zstats(reference_gate), zstats(reference_exchange))
    cfg = {"probe_mu": pm, "probe_sd": ps, "gate_mu": gm, "gate_sd": gs, "exch_mu": xm, "exch_sd": xs,
           "probe_escalation_z": float((np.quantile(np.asarray(reference_probe, float), escalation_quantile) - pm) / ps),
           "probe_escalation_quantile": escalation_quantile,
           "weights": dict(zip(("probe", "gate", "exchange"), map(float, weights)))}
    per_set = {}
    for s in sets:
        x = s.exchange if s.exchange is not None else np.full(len(s.probe), xm)   # z = 0
        sc = full_scores(cfg, s.probe, s.gate, x)
        if s.fit is not None:
            sc = sc[np.asarray(s.fit, bool)]
        if len(sc):
            with np.errstate(invalid="ignore"):   # a cap reached among non-escalated rows (-inf) sets no constraint
                q = float(np.quantile(sc, 1 - s.cap))
            per_set[s.name] = -np.inf if np.isnan(q) else q
    cfg["full_thr"] = max(per_set.values())
    cfg["binding_set"] = max(per_set, key=per_set.get)
    return cfg


def fit_alert(reference_probe, reference_gate, reference_exchange=None, input_budget: float = 0.05,
              exchange_budgets=(0.01, 0.20)) -> dict:
    """Alert-mode configuration (``thresholds/seed{k}.json``).

    The input score is the mean of the probe and gate z-scores; ``alert_input_thr`` flags ``input_budget`` of the
    reference prompts. With reference replies, ``exch_thr_1pct`` / ``exch_thr_20pct`` flag 1% / 20% of them.
    """
    (pm, ps), (gm, gs) = zstats(reference_probe), zstats(reference_gate)
    s = ((np.asarray(reference_probe, float) - pm) / ps + (np.asarray(reference_gate, float) - gm) / gs) / 2
    cfg = {"probe_mu": pm, "probe_sd": ps, "gate_mu": gm, "gate_sd": gs,
           "alert_input_thr": float(np.quantile(s, 1 - input_budget))}
    if reference_exchange is not None:
        x = np.asarray(reference_exchange, float)
        lo, hi = exchange_budgets
        cfg["exch_thr_1pct"] = float(np.quantile(x, 1 - lo))
        cfg["exch_thr_20pct"] = float(np.quantile(x, 1 - hi))
    return cfg


def block_rate(cfg: dict, probe, gate, exchange) -> float:
    """Share of rows the full-mode rule blocks."""
    return float((full_scores(cfg, probe, gate, exchange) > cfg["full_thr"]).mean())
