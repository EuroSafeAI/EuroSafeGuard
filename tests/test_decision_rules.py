"""Fast CPU tests of the decision rules, threshold fitting and released-config loading (no model weights)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eurosafeguard import HEADS, full_block
from eurosafeguard.calibration import CalibrationSet, block_rate, fit_alert, fit_full, full_scores, hash_half
from eurosafeguard.guard import EuroSafeGuard, MLPProbe, ema_max

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "tests" / "data" / "full_seed0.json").read_text())
ALERT = json.loads((ROOT / "tests" / "data" / "seed0.json").read_text())


def raw(cfg, pz, gz, xz):
    """Raw scores whose z-scores are (pz, gz, xz)."""
    return (cfg["probe_mu"] + pz * cfg["probe_sd"], cfg["gate_mu"] + gz * cfg["gate_sd"],
            cfg["exch_mu"] + xz * cfg["exch_sd"])


def test_released_config_fields():
    assert set(CFG["weights"]) == {"probe", "gate", "exchange"}
    assert np.isclose(sum(CFG["weights"].values()), 1.0)
    for k in ("probe_mu", "probe_sd", "gate_mu", "gate_sd", "exch_mu", "exch_sd", "probe_escalation_z", "full_thr"):
        assert np.isfinite(CFG[k])
    assert ALERT["heads"] == HEADS and len(HEADS) == 11


def test_not_escalated_is_never_blocked():
    # Huge gate and exchange scores do not matter if the probe does not escalate.
    p, g, x = raw(CFG, CFG["probe_escalation_z"] - 1e-3, 1e6, 1e6)
    assert not full_block([p], [g], [x], CFG)[0]


def test_weighted_sum_and_threshold():
    pz = CFG["probe_escalation_z"] + 0.5
    rest = (CFG["full_thr"] - 0.6 * pz) / 0.4          # gate z = exchange z = rest puts W.z exactly on the threshold
    p, g, x = raw(CFG, pz, rest - 1e-3, rest - 1e-3)
    assert not full_block([p], [g], [x], CFG)[0]
    p, g, x = raw(CFG, pz, rest + 1e-3, rest + 1e-3)
    assert full_block([p], [g], [x], CFG)[0]


def test_nan_scores_for_non_escalated_rows():
    p_low, _, _ = raw(CFG, 0.0, 0, 0)
    p_hi, g, x = raw(CFG, 10.0, 10.0, 10.0)
    out = full_block([p_low, p_hi], [np.nan, g], [np.nan, x], CFG)
    assert out.tolist() == [False, True]


def test_calibration_full_scores_matches_shipped_rule():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, size=(2000, 3))
    p, g, x = raw(CFG, z[:, 0], z[:, 1], z[:, 2])
    assert ((full_scores(CFG, p, g, x) > CFG["full_thr"]) == full_block(p, g, x, CFG)).all()


def test_fit_full_meets_every_cap():
    rng = np.random.default_rng(1)
    ref = [rng.normal(0, 1, 20000) for _ in range(3)]
    hard = CalibrationSet("hard", rng.normal(1.0, 1, 3000), rng.normal(1.0, 1, 3000), 0.05, rng.normal(1, 1, 3000))
    cfg = fit_full(*ref, sets=[CalibrationSet("everyday", *ref[:2], 0.005, ref[2]), hard])
    assert abs(cfg["probe_escalation_z"] - np.quantile(ref[0], 0.9) / ref[0].std()) < 0.05
    assert block_rate(cfg, *ref) <= 0.005 + 1e-9
    assert block_rate(cfg, hard.probe, hard.gate, hard.exchange) <= 0.05 + 1e-9
    assert cfg["binding_set"] in {"everyday", "hard"}


def test_fit_full_prompt_only_set_uses_zero_exchange_z():
    rng = np.random.default_rng(2)
    ref = [rng.normal(0, 1, 5000) for _ in range(3)]
    s = CalibrationSet("prompts", rng.normal(0, 1, 5000), rng.normal(0, 1, 5000), 0.01)
    cfg = fit_full(*ref, sets=[s])
    mu = np.mean(ref[2])
    assert block_rate(cfg, s.probe, s.gate, np.full(5000, mu)) <= 0.01 + 1e-9


def test_fit_alert_budget():
    rng = np.random.default_rng(3)
    p, g, x = rng.normal(-7, 3, 10000), rng.beta(1, 20, 10000), rng.beta(1, 30, 10000)
    c = fit_alert(p, g, x, input_budget=0.05)
    s = ((p - c["probe_mu"]) / c["probe_sd"] + (g - c["gate_mu"]) / c["gate_sd"]) / 2
    assert abs((s > c["alert_input_thr"]).mean() - 0.05) < 1e-3
    assert abs((x > c["exch_thr_1pct"]).mean() - 0.01) < 1e-3
    assert c["exch_thr_20pct"] < c["exch_thr_1pct"]


def test_hash_half_is_stable():
    assert [hash_half(k) for k in ("0", "1", "2", "3")] == [hash_half(k) for k in (0, 1, 2, 3)]
    assert {hash_half(i) for i in range(100)} == {0, 1}


def test_alert_route_logic():
    """route(): pass below alert_input_thr, refuse above it if the top head is CBRN, alert otherwise."""
    g = EuroSafeGuard.__new__(EuroSafeGuard)
    g.c = ALERT
    t = ALERT["alert_input_thr"]
    H = np.zeros((3, 11))
    H[0, HEADS.index("CBRN")] = H[1, HEADS.index("Cyber")] = H[2, HEADS.index("CBRN")] = 0.9
    g.input_scores = lambda prompts: (np.array([t + 1, t + 1, t - 1]), H)
    assert [d for d, _, _ in g.route(["a", "b", "c"])] == ["refuse", "alert", "pass"]
    assert g.route("a")[1] == "CBRN"


def test_ema_max_matches_reference():
    torch.manual_seed(0)
    z = torch.randn(3, 20)
    mask = torch.ones(3, 20, dtype=torch.bool)
    mask[1, 15:] = False
    a = 2 / 17
    for r in range(3):
        e, best = None, -1e9
        for t in range(20):
            if mask[r, t]:
                e = z[r, t].item() if e is None else (1 - a) * e + a * z[r, t].item()
                best = max(best, e)
        assert abs(ema_max(z, mask)[r].item() - best) < 1e-5


def test_mlp_probe_shapes():
    p = MLPProbe(32, hidden=8)
    out = p(torch.randn(2, 5, 32))
    assert out.shape == (2, 5)


@pytest.mark.parametrize("name", ["gate.json", "exchange.json", "probe.json"])
def test_training_configs_parse(name):
    cfg = json.loads((ROOT / "configs" / name).read_text())
    assert "_comment" in cfg
