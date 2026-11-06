"""Reproduce the compute overhead of the paper ("adds 15% of the protected model's FLOPs"; EuroSafe-CC++ 6%).

Cost model: a forward pass costs 2 x parameters x tokens FLOPs. The protected model (Llama-3-8B, 8B parameters) reads a
mean WildChat prompt of P = 164 tokens and writes a mean reply of R = 371 tokens (Qwen3.5 tokenizer, 1,500-row WildChat
sample; measured once, not part of this bundle). The probe reuses the protected model's activations and costs nothing.
  EuroSafeGuard (Full): on escalated requests, the 9B input classifier reads the prompt and the 9B exchange classifier
                        reads prompt + reply. Escalation share = share of WildChat-clean test-half prompts whose probe
                        score exceeds the WildChat-clean 90th percentile (computed here from the bundle).
  EuroSafe-CC++:        on escalated requests (5.5% by construction), one 9B classifier reads prompt + reply.
  python reproduce_compute.py [--scores DIR]
"""
import numpy as np

from common import Checker, load, parse_args

P, R = 164, 371            # mean prompt / reply tokens on WildChat
PROTECTED, CLS = 8e9, 9e9  # parameters of the protected model and of each classifier
CCPP_ESCALATION = 0.055


def main():
    a = parse_args(__doc__)
    wc = load(a.scores, "wildchat")
    c = wc.wc_clean.values
    test = c & ~wc.in_train.values & (wc.half.values == 1)
    esc = float((wc.probe.values[test] > np.quantile(wc.probe.values[c], 0.9)).mean())
    base = PROTECTED * (P + R)
    ours = 100 * (CLS * P + CLS * (P + R)) * esc / base
    ccpp = 100 * CLS * (P + R) * CCPP_ESCALATION / base
    ck = Checker("Compute overhead, % of the protected model's FLOPs")
    print(f"\nProbe escalation share on WildChat-clean (test half): {100 * esc:.2f}%")
    ck.value(f"EuroSafeGuard (Full) ({ours:.2f}%)", f"{ours:.0f}%", "15%")
    ck.value(f"EuroSafe-CC++ ({ccpp:.2f}%)", f"{ccpp:.0f}%", "6%")
    ck.finish()


if __name__ == "__main__":
    main()
