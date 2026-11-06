# EuroSafeGuard: Open Constitutional Classifiers

[Paper](https://arxiv.org/abs/TBD) · [Models](https://huggingface.co/EuroSafeAI/EuroSafeGuard) ·
[Training prompts](https://huggingface.co/datasets/EuroSafeAI/EuroSafeGuard-prompts) · [Demo](https://guard.safe.eu)

EuroSafeGuard is an open guardrail for a language model that you do not retrain. An activation probe reads the
protected model's hidden states and screens every request. The requests it scores highest go on to a fine-tuned
input classifier and to an exchange classifier that reads the prompt together with the reply, and the reply is blocked
if a weighted sum of the three scores crosses a calibrated threshold. This repository has the full pipeline, from a
written constitution over 11 harm categories to a calibrated guardrail: inference code, the training and data
generation scripts, threshold calibration for your own traffic, and the per-row scores behind the paper's main tables.

<p align="center"><img src="assets/overview.png" width="900" alt="EuroSafeGuard: an activation probe on the protected model escalates suspicious requests to an input classifier and an exchange classifier, which decide whether the reply is shown or blocked"></p>

## Results

At the same false-positive budget on real chat traffic (5% of WildChat-clean flagged), the input stage of
EuroSafeGuard detects more held-out jailbreaks than any of 16 open guards on three of four sets, and lets the fewest
search attacks through. The table has selected rows of the paper's Table 2. Recall is in %, PAIR and TAP are the
% of behaviors broken, and "Full" also checks the reply.

| Guard | PyRIT | BELLS-O jb | BET | GB×JF | PAIR Input ↓ | PAIR Full ↓ | TAP Input ↓ | TAP Full ↓ |
|---|---|---|---|---|---|---|---|---|
| Llama Guard 3 (8B) | 78.9 | 63.1 | 80.4 | 40.8 | 98.8 | 76.2 | 95.3 | 75.0 |
| WildGuard (7B) | 98.3 | 63.4 | 83.2 | 66.4 | 66.7 | 63.1 | 40.6 | 37.5 |
| Qwen3Guard (8B) | 99.7 | 71.6 | 94.6 | 63.2 | 78.6 | 45.2 | 62.5 | 45.3 |
| HaloGuard (4B) | 98.4 | 88.2 | 92.6 | 79.2 | 48.8 | 38.1 | 35.9 | 26.6 |
| gpt-oss-safeguard (20B)† | 86.0 | 71.0 | 85.0 | 52.8 | 66.7 | 64.3 | 50.0 | 45.3 |
| **EuroSafeGuard** (9B + probe) | **100.0** | **95.9** | **97.0** | 79.3 | 20.7 | **21.7** | **19.4** | **15.0** |
| EuroSafe-CC++ (probe) | 99.9 | 93.3 | 94.9 | **79.8** | **17.9** | – | 25.0 | – |

† Native verdict. EuroSafeGuard values are means over five training runs of the input classifier. EuroSafe-CC++
is our open recreation of the Constitutional Classifiers++ cascade.

The next table has selected rows of the paper's end-to-end table: the guardrails in front of Llama-3-8B-Instruct on
1,974 held-out Jailbreak Foundry attacks. Attack success rate (ASR) is in %. Benign failure means the request was
blocked, refused or answered with broken text, also in %.

| Defense | ASR rubric ≥ 0.5 | ASR StrongREJECT ≥ 0.5 | XSTest | OR-80k | OR-hard | WildChat | MMLU | GSM8K |
|---|---|---|---|---|---|---|---|---|
| Undefended Llama-3-8B | 7.95 | 19.71 | 2.6 | 6.8 | 68.2 | 1.6 | 0.9 | 0.6 |
| Circuit breakers (retrained model) | 3.39 | 4.76 | 1.7 | 43.7 | 84.9 | 4.0 | 2.1 | 0.8 |
| EuroSafe-CC++ | 4.57 | 6.57 | 3.4 | 7.9 | 71.2 | – | – | – |
| **EuroSafeGuard (Full)** | **3.88** | **5.40** | 2.6 | 7.7 | 71.4 | 1.7 | 1.0 | 0.6 |
| EuroSafeGuard (Full) + circuit breakers | 2.61 | 2.87 | 1.7 | 43.9 | 85.3 | 4.1 | 2.1 | 0.8 |
| Undefended gpt-oss-20b | 13.17 | 28.27 | 12.0 | 14.0 | 85.1 | 4.1 | 4.8 | 11.1 |
| EuroSafeGuard (Full) + gpt-oss-20b | 5.28 | 7.92 | 12.0 | 14.6 | 86.0 | 4.3 | 5.1 | 11.1 |

The deployed cascade blocks 0.2% of WildChat-clean. Only the ~10% of requests that the probe escalates reach the two
9B classifiers. For an 8B protected model, the guardrail therefore adds about 15% of the model's FLOPs
(EuroSafe-CC++: 6%).

All numbers in this section are regenerated exactly from per-row scores by [`paper/`](paper/), on a CPU in seconds:

```bash
python paper/reproduce_attacks.py       # Table 2: all 144 cells
python paper/reproduce_end_to_end.py    # end-to-end table: all 192 cells
python paper/reproduce_calibration.py   # released thresholds; 0.2% of WildChat-clean blocked
python paper/reproduce_compute.py       # 15% / 6% of the protected model's FLOPs
```

## Install

```bash
pip install "eurosafeguard @ git+https://github.com/EuroSafeAI/EuroSafeGuard"
# optional: fast Qwen3.5 linear-attention kernels
pip install "eurosafeguard[fast] @ git+https://github.com/EuroSafeAI/EuroSafeGuard"
```

Install a PyTorch build that matches your CUDA driver first. Inference needs about 55 GB of GPU memory in bf16:
16 GB for Llama-3.1-8B-Instruct (the probe) and 19 GB for each of the two Qwen3.5-9B classifiers. You can spread
them over three devices. Llama-3.1-8B-Instruct is gated on Hugging Face, so accept its license and run
`huggingface-cli login`.

## Quickstart

```python
from eurosafeguard import EuroSafeGuard

guard = EuroSafeGuard.from_pretrained("EuroSafeAI/EuroSafeGuard", seed=0, mode="full",
                                      devices=("cuda:0", "cuda:0", "cuda:1"))   # probe LM, input clf, exchange clf
prompt, reply = "What is a good recipe for sourdough bread?", "Mix 500 g flour, 350 g water, ..."

decision, category, score = guard.route(prompt)       # input stage: "pass" | "alert" | "refuse"
blocked = guard.check_reply(prompt, reply, decision)  # alert mode: exchange check on the reply
blocked = guard.check_full(prompt, reply)             # full mode: the paper's end-to-end configuration
```

The command-line equivalent is `eurosafeguard check --prompt "..." --reply "..." --mode full`, and
[`examples/quickstart.py`](examples/quickstart.py) is a runnable script.

Both decision rules are calibrated on WildChat-clean:

- **Full mode** (`check_full`) is the configuration of the paper's end-to-end results. The probe scores every
  request. Requests whose probe z-score is above the 90th percentile of benign traffic are escalated. An escalated
  exchange is blocked if `0.6·probe_z + 0.2·gate_z + 0.2·exchange_z > full_thr`. Requests that are not escalated are
  never blocked, so the classifiers run on about 10% of traffic.
- **Alert mode** (`route` + `check_reply`) acts before generation. The input score is the mean of the probe and gate
  z-scores, and its threshold flags 5% of WildChat-clean. A flagged request is refused if its top category is CBRN.
  Any other flagged request is answered under a system note that names the policy (`guard.system_note(category)`).
  The exchange classifier then checks that reply at a stricter threshold.

`gate/seed0` to `gate/seed4` are five training runs of the input classifier on the same data. They differ, most of
all on search attacks such as PAIR, so evaluate the run you deploy. The probe reads Llama-3.1-8B-Instruct
activations. If you protect that model, you can reuse the hidden states from its own forward pass.

## Calibrate thresholds for your traffic

The released thresholds were fitted on WildChat-like traffic with Llama-3-8B-Instruct replies. To refit them, score
samples of your own benign traffic and set a cap on the share blocked for each source:

```bash
# raw component scores (probe, gate, exchange); input CSVs have prompt, reply and id columns
eurosafeguard score my_traffic.csv my_traffic_scores.csv --reply-col reply
eurosafeguard score my_sensitive_benign.csv sensitive_scores.csv --reply-col reply

# z-statistics and the escalation cut come from the reference traffic; full_thr is the lowest threshold meeting every cap
eurosafeguard calibrate --reference my_traffic_scores.csv --reference-cap 0.005 \
    --set sensitive=sensitive_scores.csv:0.02 --out my_thresholds/

eurosafeguard check --thresholds my_thresholds/ --mode full --prompt "..." --reply "..."
```

`eurosafeguard.calibration` does the same in Python (`fit_full`, `fit_alert`). Run on the paper's scores, it refits
the released `full_seed0.json` to the last bit (`tests/test_paper_calibration.py`). The paper's caps are 0.5% on
everyday traffic, 2% on harmful-sounding benign prompts, 5% on OR-Bench-hard and 40% on benign prompts wrapped in
jailbreak templates, each fitted on one hash half of its set. To measure recall on an attack set at a fixed budget,
use [`eval/recall_at_budget.py`](eval/recall_at_budget.py).

## Retrain on your own constitution

| Step | Where | What |
|---|---|---|
| 1. Policy | `data/constitution/` | 11 categories, each with disallowed and allowed subcategories (BELLS-O taxonomy) |
| 2. Data generation | `data/redact/` | [REDACT](https://github.com/CentreSecuriteIA/REDACT) recipes: prompts in two styles at four severity levels, a checker, jailbreak rewrites, replies, paraphrases |
| 3. Augmentation | `data/augment.py` | obfuscations, Best-of-N noise and jailbreak templates, applied to benign and harmful prompts alike; plain twins |
| 4. Classifiers | `training/train_classifier.py` | Qwen3.5-9B + LoRA + 11 heads with a twin consistency loss; gate and exchange configs in `configs/` |
| 5. Probe | `training/train_probe.py` | MLP probe on the protected model's activations, trained with the CC++ loss |
| 6. Calibration | `eurosafeguard calibrate` | z-statistics, escalation cut and per-benchmark caps |

See [`data/README.md`](data/README.md) and [`training/README.md`](training/README.md). To retrain the released models,
`data/prepare.py` converts the released datasets into the trainers' CSV files.

## Reproduce the paper

- **From the released scores** (CPU, seconds): see [`paper/README.md`](paper/README.md). The bundle covers Table 2,
  the end-to-end table, the released thresholds and the compute estimate. It holds the per-row scores of every guard,
  the judge scores and the refusal flags, and no text.
- **From scratch** (GPUs and gated data): score each benchmark with `eurosafeguard score`. Then compute recall at the
  WildChat-clean budget with `eval/recall_at_budget.py`, or run the reproduction scripts on your own score files in
  the format described in `paper/scores/README.md`. The protected models' replies are generated and judged
  (GuidedBench rubric, StrongREJECT) with each benchmark's official code. Competing guards are scored with their own
  templates, and that code is not in this repository.

The released code does not reproduce the paper's other tables and figures: moderation, over-refusal, CBRN transfer,
multilingual, adaptive and white-box attacks, and latency. Numbers from them below are quoted from the paper. We do
not release attack code.

## Components

| Component | Base model | Hugging Face |
|---|---|---|
| Activation probe (MLP on all 32 blocks; maximum over prompt tokens of an EMA of per-token logits) | Llama-3.1-8B-Instruct | [`probe/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard/tree/main/probe) |
| Input classifier ("gate"), five training runs | Qwen3.5-9B + LoRA + 11 heads | [`gate/seed{0..4}/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard/tree/main/gate) |
| Exchange classifier | Qwen3.5-9B + LoRA + 11 heads | [`exchange/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard/tree/main/exchange) |
| Thresholds for alert and full mode, per training run | – | [`thresholds/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard/tree/main/thresholds) |

The [model card](https://huggingface.co/EuroSafeAI/EuroSafeGuard) describes the components and both modes.
`eurosafeguard/guard.py` is the same file as `eurosafeguard.py` in the model repository. The paper's appendix
describes an optional streaming output probe. No reported result uses it, and it is not part of this release.

## EuroSafe-CC++

EuroSafe-CC++ is our open recreation of the two-stage Constitutional Classifiers++ cascade
([arXiv:2601.04603](https://arxiv.org/abs/2601.04603)) and the CC++ baseline of the paper. Its weights and thresholds are
in [EuroSafeAI/EuroSafe-CCPP](https://huggingface.co/EuroSafeAI/EuroSafe-CCPP), and `eurosafeguard/ccpp.py` is the same
file as `eurosafe_ccpp.py` there.

- **Stage 1.** Two linear probes on Llama-3.1-8B-Instruct activations (all 32 blocks), trained with the CC++ loss and
  scored with an EMA of per-token logits. The prompt probe reads the prompt. The reply probe reads the reply, with the
  prompt as context. The larger of their z-scores escalates 5.5% of WildChat-clean.
- **Stage 2.** The input classifier `gate/seed{k}` of EuroSafeGuard is applied to the exchange. An escalated exchange is
  blocked if `0.55·stage-1 score + 0.45·z(classifier logit)` exceeds a threshold that blocks as much WildChat-clean
  traffic as EuroSafeGuard (Full).

```python
from eurosafeguard import EuroSafeCCPP

ccpp = EuroSafeCCPP.from_pretrained("EuroSafeAI/EuroSafe-CCPP", seed=0, devices=("cuda:0", "cuda:0"))  # probe LM, classifier
blocked = ccpp.check(prompt, reply)   # end-to-end table configuration
flagged = ccpp.input_flag(prompt)     # prompt probe alone at the 5% WildChat-clean budget (Table 2)
```

It needs about 35 GB of GPU memory in bf16. `tests/test_ccpp.py` checks that the released thresholds reproduce the
paper's EuroSafe-CC++ row (4.57% ASR, mean over the five `gate` runs) from the score bundle. The model card lists how
the recreation differs from CC++.

## Data access

> **Content warning.** The training and evaluation data contain harmful requests and harmful model replies.

The training data is on Hugging Face:
[EuroSafeGuard-prompts](https://huggingface.co/datasets/EuroSafeAI/EuroSafeGuard-prompts) has the prompts for the input
classifiers and the probe, and
[EuroSafeGuard-replies](https://huggingface.co/datasets/EuroSafeAI/EuroSafeGuard-replies) has the prompt-reply pairs
for the exchange classifier. Both are gated behind a request form, and each request is approved by hand. A request
needs a clear research purpose, such as building or evaluating safeguards. Some rows come from datasets that do not
allow redistribution. Their text is withheld and can be rebuilt from the original sources with `data/prepare.py
--reconstruct`. This repository itself contains no harmful text.

## Limitations

The numbers in this section are quoted from the paper and are not regenerated by `paper/`.

- **White-box attacks.** An attacker with gradients of the protected model and of every model in the guardrail
  (GCG) finds suffixes that get 40% of 50 behaviors past the deployed cascade's input scores. Most of the robustness
  comes from the probe; the input classifier alone is easier to evade. In the paper's ablation, taking the larger of
  the two standardized input scores instead of their mean lowers white-box evasion from 60% to 38%.
- **Low-resource languages** are the clearest gap: on harmful prompts in low-resource languages, recall is below
  several other guards.
- **Judges are noisy.** Attack success is measured with LLM judges (GuidedBench rubric, StrongREJECT,
  gpt-oss-safeguard), which make mistakes on hard cases. Re-judging identical replies moves ASR by about 0.3 points.
- **Training runs differ.** Five input classifiers trained on the same data leave between 11% and 48% of PAIR
  behaviors broken.
- **Over-refusal.** The input stage flags benign requests on sensitive topics more often than some guards. Alert
  mode answers most of them, but check false positives on your own traffic.
- **Content moderation.** Dedicated moderation models remain ahead on general moderation benchmarks.
- **Not a complete defense.** No guardrail stops every attack. Combine it with a safety-trained model, monitoring and
  rate limits, and recalibrate it on your own traffic.

## Repository layout

```
eurosafeguard/   inference (guard.py and ccpp.py are the Hugging Face loaders), calibration, CLI
configs/         hyperparameters of the released gate, exchange classifier and probe
data/            constitution, REDACT recipes, augmentation, preparation of the released data
training/        classifier and probe training
eval/            recall at a fixed false-positive budget
paper/           score bundle and the scripts that reproduce the paper's tables
examples/        quickstart
tests/           fast CPU tests (pytest)
```

## Citation

```bibtex
@article{waibl2026eurosafeguard,
  title   = {EuroSafeGuard: Open Constitutional Classifiers},
  author  = {Waibl, Leonhard and Chen, Yen-Shan and Simko, Samuel and Michalak, Felix and Diwan, Nirav and
             Singh, Arth and Thomas, Aditya and Sch{\"o}lkopf, Bernhard and Jin, Zhijing},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2026}
}
```

## License

The code is under [Apache-2.0](LICENSE), as are the model weights. The base models keep their own licenses: the Llama
3.1 Community License and the Qwen license. The datasets mix sources with different licenses, including
non-commercial ones, and each row lists the license of its source. The score bundle in `paper/scores/` is released
under this repository's license; the evaluation sets it was computed on keep their own terms. Report bypasses
privately, as described in [SECURITY.md](SECURITY.md).
