"""Score a prompt and an exchange with EuroSafeGuard.

Needs about 55 GB of GPU memory in bf16 (Llama-3.1-8B for the probe, two Qwen3.5-9B classifiers); pass three devices
to split them, e.g. devices=("cuda:0", "cuda:1", "cuda:1").

    python examples/quickstart.py
"""
from eurosafeguard import EuroSafeGuard

guard = EuroSafeGuard.from_pretrained("EuroSafeAI/EuroSafeGuard", seed=0, mode="full", devices=("cuda:0",) * 3)

prompt = "What is a good recipe for sourdough bread?"
reply = "Mix 500 g flour, 350 g water, 100 g active starter and 10 g salt, then let the dough rise overnight."

# Alert mode: input stage before generation, exchange check after it.
decision, category, score = guard.route(prompt)          # "pass" | "alert" | "refuse", top category, input score
print(f"input stage: {decision} (top category {category}, score {score:.2f})")
print("reply withheld (alert mode):", guard.check_reply(prompt, reply, decision))

# Full mode, the paper's end-to-end configuration: the probe screens every request and only escalated requests
# reach the two classifiers.
print("escalated:", guard.escalate(prompt))
print("blocked (full mode):", guard.check_full(prompt, reply))
