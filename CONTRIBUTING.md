# Contributing

Issues and pull requests are welcome: bug reports, fixes, support for other protected models, and calibration
results on new traffic.

- Run `pip install -e ".[dev]"` and `pytest` before opening a pull request. The tests run on a CPU in under a minute.
- Keep the decision rules in `eurosafeguard/guard.py` identical to `eurosafeguard.py` in the
  [model repository](https://huggingface.co/EuroSafeAI/EuroSafeGuard). Changes to them need a matching model release.
- Do not add harmful text, jailbreak templates or model generations to the repository, including in tests and issues.
  Use placeholders, or link to the gated datasets.
- Report bypasses privately, as described in [SECURITY.md](SECURITY.md).
