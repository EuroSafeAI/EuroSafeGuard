# EuroSafeGuard: Open Constitutional Classifiers

[Paper](https://guard.safe.eu/paper.pdf) · [Hugging Face collection](https://huggingface.co/collections/EuroSafeAI/eurosafeguard-v1-6aedf578437a9646077d1b58) ·
[Model](https://huggingface.co/EuroSafeAI/EuroSafeGuard-v1) · [Training prompts](https://huggingface.co/datasets/EuroSafeAI/EuroSafeGuard-v1-prompts) ·
[Project page](https://guard.safe.eu)

An open guardrail for LLMs, from a written constitution to a deployed cascade: an activation probe screens all traffic
and escalates suspicious requests to an input classifier and an exchange classifier.

<p align="center"><img src="assets/overview.png" width="900" alt="EuroSafeGuard: an activation probe on the protected model escalates suspicious requests to an input classifier and an exchange classifier, which decide whether the reply is shown or blocked"></p>

<p align="center"><img src="assets/example.png" width="900" alt="A jailbreak answered by the unprotected model and blocked by EuroSafeGuard; shading shows the probe score per token"></p>

## Install

```bash
pip install "eurosafeguard @ git+https://github.com/EuroSafeAI/EuroSafeGuard"
pip install "eurosafeguard[fast] @ git+https://github.com/EuroSafeAI/EuroSafeGuard"   # optional Qwen3.5 kernels
```

Inference needs about 55 GB of GPU memory in bf16 (Llama-3.1-8B-Instruct for the probe, two Qwen3.5-9B
classifiers), on up to three devices. Llama-3.1-8B-Instruct is gated: accept its license and `huggingface-cli login`.

## Quickstart

```python
from eurosafeguard import EuroSafeGuard

guard = EuroSafeGuard.from_pretrained("EuroSafeAI/EuroSafeGuard-v1", seed=0, mode="full",
                                      devices=("cuda:0", "cuda:0", "cuda:1"))   # probe LM, input clf, exchange clf
prompt, reply = "What is a good recipe for sourdough bread?", "Mix 500 g flour, 350 g water, ..."

decision, category, score = guard.route(prompt)       # input stage: "pass" | "alert" | "refuse"
blocked = guard.check_reply(prompt, reply, decision)  # alert mode: exchange check on the reply
blocked = guard.check_full(prompt, reply)             # full mode: the paper's end-to-end configuration
```

Command line: `eurosafeguard check --prompt "..." --reply "..." --mode full`. Runnable script:
[`examples/quickstart.py`](examples/quickstart.py).

- **Full mode** (`check_full`): the probe scores every request; requests above the 90th percentile of WildChat-clean
  are escalated, and an escalated exchange is blocked if `0.6·probe_z + 0.2·gate_z + 0.2·exchange_z > full_thr`.
- **Alert mode** (`route` + `check_reply`): flagged requests are refused if CBRN, otherwise answered under a policy
  note (`guard.system_note(category)`); the exchange classifier then checks the reply.

## Results

<p align="center"><img src="assets/bars.png" width="720" alt="Mean robustness over six attack sets for EuroSafeGuard and 16 open guards, all flagging 5% of benign WildChat traffic"></p>

Input stage at the same false-positive budget (5% of WildChat-clean flagged). Recall in %; PAIR and TAP in % of
behaviors broken (lower is better). Selected rows of the paper's Table 2.

| Guard | PyRIT | BELLS-O jb | BET | GB×JF | PAIR Input | PAIR In.+Ex. | TAP Input | TAP In.+Ex. |
|---|---|---|---|---|---|---|---|---|
| Llama Guard 3 (8B) | 78.9 | 63.1 | 80.4 | 40.8 | 98.8 | 76.2 | 95.3 | 75.0 |
| Llama Guard 4 (12B) | 72.6 | 49.4 | 81.4 | 38.6 | 98.8 | 92.9 | 98.4 | 93.8 |
| ShieldGemma (9B) | 75.9 | 54.4 | 65.7 | 39.8 | 86.9 | 72.6 | 79.7 | 76.6 |
| Qwen3Guard (8B) | 99.7 | 71.6 | 94.6 | 63.2 | 78.6 | 45.2 | 62.5 | 45.3 |
| HaloGuard (4B) | 98.4 | 88.2 | 92.6 | 79.2 | 48.8 | 38.1 | 35.9 | 26.6 |
| gpt-oss-safeguard (20B)† | 86.0 | 71.0 | 85.0 | 52.8 | 66.7 | 64.3 | 50.0 | 45.3 |
| **EuroSafeGuard** (9B + probe) | **100.0** | **95.9** | **97.0** | 79.3 | 20.7 | **21.7** | **19.4** | **15.0** |
| EuroSafe-CC++ | 99.9 | 93.3 | 94.9 | **79.8** | **17.9** | 48.8 | 25.0 | 42.2 |

† Native verdict.

End to end, in front of Llama-3-8B-Instruct on 1,974 held-out Jailbreak Foundry attacks (attack success and benign
failure in %):

| Defense | ASR rubric ≥0.5 | ASR StrongREJECT ≥0.5 | XSTest | OR-80k | OR-hard | WildChat | MMLU | GSM8K |
|---|---|---|---|---|---|---|---|---|
| Undefended Llama-3-8B | 7.95 | 19.71 | 2.6 | 6.8 | 68.2 | 1.6 | 0.9 | 0.6 |
| EuroSafe-CC++ | 4.57 | 6.57 | 3.4 | 7.9 | 71.2 | 1.9 | 1.3 | 0.6 |
| **EuroSafeGuard (deployed)** | **3.88** | **5.40** | 2.6 | 7.7 | 71.4 | 1.7 | 1.0 | 0.6 |
| Undefended gpt-oss-20b | 13.17 | 28.27 | 12.0 | 14.0 | 85.1 | 4.1 | 4.8 | 11.1 |
| EuroSafeGuard (deployed) + gpt-oss-20b | 5.28 | 7.92 | 12.0 | 14.6 | 86.0 | 4.3 | 5.1 | 11.1 |

The deployed cascade blocks 0.2% of WildChat-clean and adds about 15% of an 8B protected model's FLOPs.

## Reproduce the paper

From the released per-row scores, on a CPU in seconds ([`paper/README.md`](paper/README.md)):

```bash
python paper/reproduce_attacks.py       # Table 2
python paper/reproduce_end_to_end.py    # end-to-end table
python paper/reproduce_calibration.py   # released thresholds; 0.2% of WildChat-clean blocked
python paper/reproduce_compute.py       # 15% / 6% of the protected model's FLOPs
```

To refit thresholds on your own traffic: `eurosafeguard score`, then `eurosafeguard calibrate` (see
`eurosafeguard calibrate --help`). [`eval/recall_at_budget.py`](eval/recall_at_budget.py) computes recall at a fixed
false-positive budget.

## Components

| Component | Base model | Where |
|---|---|---|
| Activation probe | Llama-3.1-8B-Instruct | [`probe/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard-v1/tree/main/probe) |
| Input classifier ("gate"), 5 training runs | Qwen3.5-9B + LoRA + 11 heads | [`gate/seed{0..4}/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard-v1/tree/main/gate) |
| Exchange classifier | Qwen3.5-9B + LoRA + 11 heads | [`exchange/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard-v1/tree/main/exchange) |
| Thresholds (alert and full mode) | – | [`thresholds/`](https://huggingface.co/EuroSafeAI/EuroSafeGuard-v1/tree/main/thresholds) |
| EuroSafe-CC++ (open CC++ recreation) | Llama-3.1-8B-Instruct probes + gate | [EuroSafeAI/EuroSafe-CCPP-v1](https://huggingface.co/EuroSafeAI/EuroSafe-CCPP-v1) |

`eurosafeguard/guard.py` and `eurosafeguard/ccpp.py` are the same files as the loaders in the two model repositories.

| Training step | Where |
|---|---|
| Constitution (11 harm categories) | [`data/constitution/`](data/constitution/) |
| Data generation ([REDACT](https://github.com/CentreSecuriteIA/REDACT)) and augmentation | [`data/`](data/README.md) |
| Classifier and probe training | [`training/`](training/README.md), [`configs/`](configs/) |
| Calibration | `eurosafeguard calibrate` |

## Data access

- [EuroSafeAI/EuroSafeGuard-v1-prompts](https://huggingface.co/datasets/EuroSafeAI/EuroSafeGuard-v1-prompts): training
  prompts for the input classifier and the probe. Gated; requests are reviewed by hand.
- Training replies (prompt-reply pairs for the exchange classifier): available on request for safety research.
- Rows from sources that do not allow redistribution are released without text; rebuild them with
  `data/prepare.py --reconstruct`.

This repository contains no harmful text.

## Citation

```bibtex
@misc{waibl2026eurosafeguard,
  title   = {EuroSafeGuard: Open Constitutional Classifiers},
  author  = {Waibl, Leonhard and Chen, Yen-Shan and Simko, Samuel and Thomas, Aditya and
             Singh, Arth and Diwan, Nirav and Michalak, Felix and Sch{\"o}lkopf, Bernhard and Jin, Zhijing},
  url     = {https://guard.safe.eu/paper.pdf},
  year    = {2026}
}
```

## License

Code and weights: [Apache-2.0](LICENSE). Base models keep their licenses (Llama 3.1 Community License, Qwen license).
Datasets list a license per row; several sources are non-commercial. Report bypasses as described in
[SECURITY.md](SECURITY.md).
