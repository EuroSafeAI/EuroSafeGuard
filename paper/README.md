# Reproducing the paper's numbers

This folder regenerates the headline numbers of *EuroSafeGuard: Open Constitutional Classifiers* from a bundle of
per-row scores (`scores/`, 14 MB, no text). Each script needs only `numpy` and `pandas`, runs on a CPU in a few
seconds, prints its table in markdown, marks every cell ✓ (equal to the paper) or ✗ (with the paper value), and exits
with a non-zero code if any cell differs.

```bash
pip install numpy pandas
python paper/reproduce_attacks.py        # tab:attacks
python paper/reproduce_end_to_end.py     # tab:pipeline
python paper/reproduce_calibration.py    # thresholds/full_seed{k}.json, thresholds/seed{k}.json
python paper/reproduce_compute.py        # "adds 15% of the protected model's FLOPs"
```

Every script takes `--scores DIR` to read the bundle from another place. The column dictionary is in
[`scores/README.md`](scores/README.md).

## What each script reproduces

| Script | Paper element | Cells |
|---|---|---|
| `reproduce_attacks.py` | Table *held-out jailbreaks* (tab:attacks): recall on PyRIT, BELLS-O jailbreaks, BET, GB×JF and PAIR/TAP behaviors broken through the input stage and the full pipeline, at the 5% WildChat-clean budget, for the 16 open guards, EuroSafeGuard and EuroSafe-CC++ | 144 of 144 |
| `reproduce_end_to_end.py` | End-to-end table (tab:pipeline): attack success (GuidedBench rubric and StrongREJECT ≥ 0.5) and benign failure (XSTest, OR-Bench-80k, OR-Bench-hard, WildChat, MMLU, GSM8K) for the undefended Llama-3-8B, nine weight-level defenses, EuroSafe-CC++ (two variants), EuroSafeGuard (Full) alone and in front of each defense, and gpt-oss-20b with and without the guard | 192 of 192 |
| `reproduce_calibration.py` | The released Full-mode thresholds (`probe_mu/sd`, `gate_mu/sd`, `exch_mu/sd`, `probe_escalation_z`, `full_thr`, five input-classifier runs) to the last bit; the alert-mode exchange thresholds `exch_thr_1pct` / `exch_thr_20pct`; the share of WildChat-clean blocked by the deployed cascade (0.2%) | 51 of 51 checked |
| `reproduce_compute.py` | Compute overhead: EuroSafeGuard (Full) adds 15% of the protected model's FLOPs, EuroSafe-CC++ 6% | 2 of 2 |

Expected output: each script ends with `...: all cells match` and exit code 0.

How the numbers are computed (all in the scripts' docstrings):

- **Budgets.** WildChat-clean is the set of WildChat first turns that fewer than two of seven reference guards flag
  (6,853 conversations). A "5% budget" threshold is the score that flags 5% of it. z-scores use its mean and population
  standard deviation.
- **EuroSafeGuard (Input)** = mean of the probe z-score and the input-classifier z-score. Its values in tab:attacks are
  means over the five training runs of the input classifier (gate runs 0-4).
- **EuroSafeGuard (Full)** escalates a request when its probe z-score exceeds the WildChat-clean 90th percentile and
  blocks it when 0.6 · probe z + 0.2 · input z + 0.2 · exchange z exceeds `full_thr`, the smallest threshold that keeps
  every benign calibration set under its cap on hash half 0. tab:pipeline reports means over the five runs.
- **Full pipeline of other guards** (tab:attacks): prompt score OR reply score, both thresholded at the same per-stage
  WildChat-clean quantile, chosen so that their union flags 5%.

## Not reproduced from this bundle

- **The scores themselves.** Producing them requires GPUs, the trained models, gated datasets, the protected models'
  generations and the judges (GuidedBench rubric, StrongREJECT, gpt-oss-safeguard for PAIR/TAP success). The bundle
  holds their outputs only. The refusal and broken-text decisions on each reply are precomputed (`fail_<model>`)
  because the replies are not released.
- **The training-overlap filter.** `in_train` is precomputed, since matching it needs the prompts.
- **Token counts for the compute model.** The mean WildChat prompt and reply lengths (164 and 371 tokens) are
  constants in `reproduce_compute.py`.
- **Alert-mode input threshold.** The released `alert_input_thr` values flag 4.93-5.03% of WildChat-clean with these
  scores instead of exactly 5%; `reproduce_calibration.py` prints them next to the exact 5% threshold without checking.
- **Other tables and figures** (moderation, over-refusal, CBRN, adaptive attacks, latency, figures) are not covered.
