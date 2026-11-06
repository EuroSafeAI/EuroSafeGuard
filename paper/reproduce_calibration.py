"""Refit the released thresholds (thresholds/full_seed{k}.json and thresholds/seed{k}.json) from per-row scores.

Full mode (full_seed{k}.json), for input-classifier run k:
  probe_mu/sd, gate_mu/sd, exch_mu/sd  mean and population std on WildChat-clean (6,853 conversations; the exchange
                                       classifier reads the undefended Llama-3-8B-Instruct replies);
  probe_escalation_z                   z of the WildChat-clean 90th percentile of the probe;
  full_thr                             smallest threshold on 0.6 probe z + 0.2 input z + 0.2 exchange z (escalated rows
                                       only) that keeps every benign set under its cap on hash half 0, after dropping
                                       prompts in the input-classifier training data: WildChat-clean, MMLU, GSM8K, Alpaca,
                                       Dolly, OASST 0.5%; XSTest-safe, OR-Bench-80k 2%; OR-Bench-hard 5%; benign rows of the
                                       styled jailbreak sets (PyRIT, BELLS-O, BET) 40%.
Alert mode (seed{k}.json):
  alert_input_thr                      (probe z + input z) / 2 flags 5% of WildChat-clean prompts. Not reproduced exactly:
                                       the released values flag 4.93-5.03% of WildChat-clean with these scores; the script
                                       prints them next to the exact 5% threshold;
  exch_thr_1pct / exch_thr_20pct       exchange score flagging 1% / 20% of WildChat-clean replies.
Also prints the share of WildChat-clean traffic the deployed Full cascade blocks.
  python reproduce_calibration.py [--scores DIR]
"""
import numpy as np

from common import N_RUNS, Checker, input_score, input_stats, load, parse_args
from reproduce_end_to_end import Full

# Released values (thresholds/full_seed{k}.json, thresholds/seed{k}.json of the Hugging Face model)
PROBE = {"probe_mu": -6.815694701503735, "probe_sd": 3.444286874679132, "probe_escalation_z": 1.164447941601081,
         "exch_mu": 0.02559613844389501, "exch_sd": 0.090324355434096,
         "exch_thr_1pct": 0.5512292075157089, "exch_thr_20pct": 0.017514675483107523}
PER_RUN = [
    {"gate_mu": 0.03333534521305959, "gate_sd": 0.10919123782498313, "full_thr": 3.3815571671026943, "alert_input_thr": 1.1328190803989433},
    {"gate_mu": 0.02970501206685616, "gate_sd": 0.11025653741405626, "full_thr": 3.5707621643975536, "alert_input_thr": 1.0666154607853549},
    {"gate_mu": 0.03268957031742354, "gate_sd": 0.12878089011617874, "full_thr": 3.131896552703142, "alert_input_thr": 1.0671375889045542},
    {"gate_mu": 0.03866907581677466, "gate_sd": 0.13433606451867683, "full_thr": 3.097429574322379, "alert_input_thr": 1.2083389654317394},
    {"gate_mu": 0.020549819204687854, "gate_sd": 0.1115630808512582, "full_thr": 3.318287213890092, "alert_input_thr": 0.8620295489530881},
]
WC_BLOCKED_PAPER = "0.2"   # % of WildChat-clean blocked by the deployed cascade (paper text)


def main():
    a = parse_args(__doc__)
    names = ["wildchat", "overflag", "instructions", "pyrit", "bells_o_jb", "bet"]
    S = {n: load(a.scores, n) for n in names}
    wc = S["wildchat"]
    c = wc.wc_clean.values
    ck = Checker("Released thresholds refitted from the score bundle")
    print(f"WildChat-clean conversations: {int(c.sum())}")
    blocked = []
    for k in range(N_RUNS):
        g = Full(S, k)
        st = input_stats(wc, k)
        got = {"probe_mu": g.st["probe"][0], "probe_sd": g.st["probe"][1], "gate_mu": g.st["gate"][0],
               "gate_sd": g.st["gate"][1], "exch_mu": g.st["exch"][0], "exch_sd": g.st["exch"][1],
               "probe_escalation_z": g.esc, "full_thr": g.t,
               "exch_thr_1pct": np.quantile(wc.exch_llama3.values[c], 0.99),
               "exch_thr_20pct": np.quantile(wc.exch_llama3.values[c], 0.80)}
        print(f"\n#### Gate run {k} (binding cap: {max(g.caps, key=g.caps.get)})")
        for key, v in got.items():
            want = PER_RUN[k].get(key, PROBE.get(key))
            ck.value(key, repr(float(v)), repr(want))
        alert = input_score(wc, k, st)[c]
        rel = PER_RUN[k]["alert_input_thr"]
        print(f"- alert_input_thr (not checked): released {rel:.6f} flags {100 * (alert > rel).mean():.2f}% of "
              f"WildChat-clean; the exact 5% threshold of these scores is {np.quantile(alert, 0.95):.6f}")
        blocked.append(100 * g.block(wc, wc.exch_llama3)[c].mean())
    print("\n#### WildChat-clean blocked by the deployed Full cascade")
    print("per run: " + ", ".join(f"{b:.3f}%" for b in blocked))
    ck.value("mean over runs (%, 1 decimal)", f"{np.mean(blocked):.1f}", WC_BLOCKED_PAPER)
    ck.finish()


if __name__ == "__main__":
    main()
