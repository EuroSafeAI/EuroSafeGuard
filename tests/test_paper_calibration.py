"""The calibration module refits the released full-mode thresholds from the paper's score bundle (paper/scores)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eurosafeguard.calibration import CalibrationSet, block_rate, fit_full

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "paper" / "scores"
CAPS = {"wildchat": 0.005, "gsm8k": 0.005, "mmlu": 0.005, "alpaca": 0.005, "dolly": 0.005, "oasst": 0.005,
        "xstest_safe": 0.02, "orbench_80k": 0.02, "orbench_hard": 0.05,
        "pyrit_benign": 0.4, "bells_o_jb_benign": 0.4, "bet_benign": 0.4}


@pytest.mark.skipif(not SCORES.exists(), reason="score bundle not present")
def test_refit_released_full_seed0():
    load = lambda n: pd.read_csv(SCORES / f"{n}.csv.gz")   # noqa: E731
    wc = load("wildchat")
    wc = wc[wc.wc_clean]
    parts = {"wildchat": (wc, wc.exch_llama3)}
    of = load("overflag")
    for s in ("gsm8k", "mmlu", "xstest_safe", "orbench_80k", "orbench_hard"):
        d = of[of.source == s]
        parts[s] = (d, d.exch_llama3)
    ins = load("instructions")
    for s in ("alpaca", "dolly", "oasst"):
        parts[s] = (ins[ins.source == s], None)
    for n in ("pyrit", "bells_o_jb", "bet"):
        d = load(n)
        parts[n + "_benign"] = (d[d.benign], None)
    sets = []
    for name, (d, x) in parts.items():
        d = d[~d.in_train]
        x = None if x is None else x[~parts[name][0].in_train].values
        sets.append(CalibrationSet(name, d.probe.values, d.gate0.values, CAPS[name], x, d.half.values == 0))
    cfg = fit_full(wc.probe.values, wc.gate0.values, wc.exch_llama3.values, sets)
    released = json.loads((ROOT / "tests" / "data" / "full_seed0.json").read_text())
    for k in ("probe_mu", "probe_sd", "gate_mu", "gate_sd", "exch_mu", "exch_sd", "probe_escalation_z", "full_thr"):
        assert np.isclose(cfg[k], released[k], rtol=0, atol=1e-12), k
    assert block_rate(cfg, wc.probe.values, wc.gate0.values, wc.exch_llama3.values) < 0.005
