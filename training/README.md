# Training

EuroSafeGuard has three trained components. The configs in `../configs/` hold the hyperparameters of the released
checkpoints; the training files come from `data/prepare.py` (or from your own constitution, see `../data/README.md`).

| Component | Script | Config | Released as | Hardware used |
|---|---|---|---|---|
| Input classifier ("gate"), runs 0-4 | `train_classifier.py` | `configs/gate.json`, `--seed 0..4` | `gate/seed{k}/` | 1 × 96 GB GPU, 24k steps |
| Exchange classifier | `train_classifier.py` | `configs/exchange.json` | `exchange/` | 3 × 96 GB GPUs, 8k steps |
| Activation probe | `train_probe.py` | `configs/probe.json` | `probe/` | 1 × 96 GB GPU, one epoch |

Training needs a CUDA toolkit (nvcc) that matches your PyTorch. Install torch first, then
`pip install causal-conv1d --no-build-isolation`, then the training extra. The probe alone needs only
`pip install scikit-learn`.

```bash
pip install -e ".[train]"            # adds the TACTIC trainer (pinned commit), scikit-learn, wandb
python data/prepare.py --out data/train --reconstruct

# input classifier (one run per seed; the paper reports the mean over seeds 0-4)
python training/train_classifier.py --config configs/gate.json --dataset_path data/train/gate.csv \
    --seed 0 --checkpoint_dir runs/gate_s0 --wandb_run_name gate_s0

# exchange classifier (data parallel over 3 GPUs)
torchrun --nproc_per_node=3 training/train_classifier.py --config configs/exchange.json \
    --dataset_path data/train/exchange.csv --seed 0 --checkpoint_dir runs/exchange --wandb_run_name exchange

# activation probe
python training/train_probe.py --config configs/probe.json \
    --train data/train/probe_train.csv --val data/train/probe_val.csv --out runs/probe
```

Set `WANDB_MODE=offline` to train without a Weights & Biases account.

**Classifiers.** Qwen3.5-9B with a LoRA adapter (r 128, alpha 256, on the query and value projections) and a linear
head with one sigmoid output per harm category, read at the last token of the raw text. The loss is TACTIC's masked
multi-label BCE: benign rows train every head towards 0; a categorized row trains only its category's head, towards
1 (harmful), 0.8 (dual-use harmful) or 0.5 (dual-use benign). `train_classifier.py` adds, on top of TACTIC:

- *twin consistency* (gate only, weight 1.0): every attacked prompt that has a plain twin (same `input_id`,
  `source == "plain"`) is put in the batch with it, and `1 - cos(h_attacked, stopgrad(h_plain))` on the last-token
  representation is added to the loss;
- *benign share* 0.5: half of the draws are benign rows, the rest are harm categories drawn in proportion to the
  square root of their size;
- length bucketing, gradient checkpointing (gate) and data parallelism with a manual gradient all-reduce
  (exchange; learning rates are scaled by the square root of the number of GPUs).

The exchange classifier reads `User: <prompt>` + separator + `Assistant: <reply>`. Its released run used the same
consistency setting as the gate, but its training file has no plain twins, so the term was zero; `configs/exchange.json`
turns it off.

The checkpoint folder TACTIC writes (`adapter_model.safetensors`, `classification_head.pt`, `train_config.json`,
tokenizer files) is the format `eurosafeguard` loads: copy the step you want into `gate/seed{k}/` or `exchange/` of a
local model folder and pass that folder to `EuroSafeGuard.from_pretrained`. With several GPUs, only rank 0 writes
weights. After retraining, refit the thresholds (`eurosafeguard calibrate`, see the main README): the z-scores and
cut-offs belong to one set of weights.

**Probe.** An MLP (hidden 256, GELU) on the concatenated outputs of all 32 blocks of Llama-3.1-8B-Instruct (131,072
features, scaled per layer by a fixed 1/RMS), trained with the softmax-weighted, window-smoothed per-token loss of
Constitutional Classifiers++ (window M = 16) on the chat-formatted prompt, AdamW (lr 1e-3, weight decay 0.01), batch
8 plus the plain twins of its attacked rows, one epoch. At inference the per-token logits are smoothed with an EMA
(alpha = 2/17) and the prompt score is their maximum. `--kind linear` trains the linear CC++ probe instead. To protect
another model, train the probe on that model's activations (`--model`).

**Reproducibility.** The scripts are cleaned versions of the ones that trained the released checkpoints. The
consistency term is now added through a custom autograd function instead of a one-line patch to TACTIC's training
loop; the gradients are the same, but we have not retrained the released models with this exact code. Training runs
differ between seeds, most visibly on search attacks such as PAIR (paper, appendix "Training runs"), so evaluate the run you deploy.

Training needs the Qwen3.5 linear-attention kernels (`flash-linear-attention`, `causal-conv1d`) for reasonable speed and
memory; without them transformers falls back to a slow reference implementation.
