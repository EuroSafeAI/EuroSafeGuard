"""EuroSafeGuard: open constitutional classifiers.

    from eurosafeguard import EuroSafeGuard
    guard = EuroSafeGuard.from_pretrained("EuroSafeAI/EuroSafeGuard", seed=0)
    decision, category, score = guard.route("How do I bake bread?")   # "pass" | "alert" | "refuse"

`eurosafeguard/guard.py` is the same file as `eurosafeguard.py` in the Hugging Face model repository.
"""
from .guard import HEADS, EuroSafeGuard, LoRAClassifier, MLPProbe, ema_max, full_block

__all__ = ["EuroSafeGuard", "LoRAClassifier", "MLPProbe", "HEADS", "ema_max", "full_block"]
__version__ = "1.0.0"
