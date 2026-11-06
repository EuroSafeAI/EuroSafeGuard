# Training data

> **Content warning.** The training data contains harmful requests (violence, hate speech, sexual content, self-harm,
> weapons, cyber attacks and more), many of them wrapped in jailbreak templates and obfuscations, and harmful model
> replies. Nothing in this directory contains such text. The data itself is on Hugging Face behind a request form.

This directory holds the policy, the generation recipes and the scripts that produce the training files of the
EuroSafeGuard classifiers. There are two ways to get training data:

- **Use the released data** (recommended): `prepare.py` downloads it and writes the CSVs the trainers read.
- **Regenerate it, or build it for your own policy**: edit the taxonomy, run REDACT with the recipes in `redact/`,
  convert and augment the result.

```
constitution/taxonomy.json        the 11-category policy: disallowed and allowed subcategories per category
constitution/taxonomy_short.json  the same, shortened (6 disallowed, 4 allowed) for prompts with little room
redact/*.json                     the REDACT recipes and params we ran
redact/prompts/                   our overrides of REDACT's checker prompts
redact/run_redact.py              runs the recipes with the local generator model
redact/to_csv.py                  converts a REDACT run into gate.csv / exchange.csv
augment.py                        obfuscation, Best-of-N and template augmentation, harmful and benign alike
prepare.py                        released data -> trainer CSVs
```

## Use the released data

| Dataset | Config / file | Rows | Trains |
|---|---|---|---|
| [EuroSafeAI/EuroSafeGuard-prompts](https://huggingface.co/datasets/EuroSafeAI/EuroSafeGuard-prompts) | `gate` | 540,536 | input classifiers (gate) |
| | `probe` | 116,524 | activation probe |
| EuroSafeGuard-replies (available on request) | `exchange_train.parquet` | 262,865 | exchange classifier |

The prompts dataset is gated. Request access on its page: requests are approved by hand and need a clear
research purpose (building or evaluating safeguards). The replies dataset is available on request. Then log in and run, from the repository root:

```bash
huggingface-cli login
python data/prepare.py --out data/train                 # gate.csv, exchange.csv, probe_train.csv, probe_val.csv
python data/prepare.py --out data/train --reconstruct   # also rebuild the withheld rows (see below)
```

| File | Columns |
|---|---|
| `gate.csv` | `prompt, category, entry_type, source, input_id` |
| `exchange.csv` | `prompt_response, category, entry_type, source, input_id` |
| `probe_train.csv`, `probe_val.csv` | `prompt, response, label, source, input_id` (`response` empty: prompt-only) |

`category` is one of the 11 categories or `Benign`. `entry_type` is `harmful`, `dual_use_harmful`, `dual_use_benign`,
`benign` (a harm topic that must not be flagged) or `general_benign`; the gate's targets on the row's category head are
1, 0.8, 0.5, 0 and 0. Rows that share `input_id` are rewrites of one prompt, and the row with `source == "plain"` is
their plain twin, which the consistency loss pairs them with. `prompt_response` is exactly the string the exchange
classifier saw: `"User: " + prompt + separator + "Assistant: " + response`. The released probe data has no
validation split, so `probe_val.csv` holds out 5% of `input_id` groups (`--probe-val-frac`).

**Withheld rows.** Rows copied from gated or non-redistributable datasets (WildGuardMix, WildJailbreak, WildChat-1M,
SOSBench and the CeSIA constitution datasets) are released without their text. `--reconstruct` runs the
`reconstruct.py` shipped with each dataset, which downloads those sources with your own access (accept each one's
terms first) and fills the rows by SHA-256. Rows still without text are dropped. Without reconstruction,
`prepare.py` gives:

| | Rows released | Rows kept without `--reconstruct` |
|---|---|---|
| gate | 540,536 | 411,050 (14,379 plain twins; 93,439 rewrites paired with one) |
| exchange | 262,865 | 187,892 (138,030 benign, 49,862 harmful) |
| probe | 116,524 | 83,773 (79,481 train, 4,292 val) |

## How the data was built

**1. Policy.** The constitution follows the BELLS-O harm taxonomy: 11 categories (CBRN, Cyber, Harm to Minors,
Harmful Manipulation, Hate Speech, Illegal Activities, Information Integrity, Physical Harm, Privacy, Self-Harm,
Sexual Content), plus general benign requests. Its 4,904 entries come from the CeSIA constitution
([centrepourlasecuriteia/constitution-dataset](https://huggingface.co/datasets/centrepourlasecuriteia/constitution-dataset),
written with Claude Opus 4.6 and refined by hand). Each entry describes one behavior at one of four severity levels:
harmful, dual-use harmful, dual-use benign, benign. We dropped its three categories on sexual content involving
minors (46 entries) and generated nothing for them. `constitution/taxonomy.json` summarizes the policy: for each of the
11 categories, subcategories of the constitution it disallows (harmful and dual-use harmful entries) and allows
(benign and dual-use benign). The gate has one head per category.

**2. Prompts and replies with REDACT.** [REDACT](https://github.com/CentreSecuriteIA/REDACT) (MIT) expands each
entry into prompts, checks each prompt with a second model call, generates and checks a reply per prompt, rewrites
prompts with its jailbreak techniques and natural variations, and paraphrases inputs and outputs. One local model,
[huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) (thinking
off), fills every role: generator, checker, paraphraser, paraphrase checker. We ran the recipes in this order, all on
one data directory:

| Recipe | Stages | Params |
|---|---|---|
| `inputs_long.json` | inputs, long style (2-5 sentences) | `params.json`: 3 prompts per entry |
| `inputs_short.json` | inputs, short style (5-20 words) | `params_short.json`: 3 prompts per entry |
| `outputs.json` | one checked reply per accepted input | `params.json` |
| `jailbreaks.json` | jailbreak rewrites (escalating technique rounds, no translation) and natural variations; build | `params.json` |
| `paraphrase.json` | one checked paraphrase per input and output; build | `params.json` |

`redact/prompts/` overrides three of REDACT's prompts: the input checker (lenient, domain-level fit, the verdict
must be the first word) and the output and paraphrase checkers (verdict first). REDACT reads only the first word of a
checker reply, and without the override about 30% of good samples were rejected. The replies were generated with a
helpful-only system prompt that answers every request directly; it is not included here. REDACT's packaged output
prompt is a red-teaming persona that also turns benign requests harmful, so put your own prompt at
`redact/prompts/output/generation/template.json` before running `outputs.json`.

```bash
git clone https://github.com/CentreSecuriteIA/REDACT && pip install -e "./REDACT[vllm]"
python data/redact/run_redact.py --data-dir runs/constitution
python data/redact/to_csv.py --data-dir runs/constitution --out data/train_own
```

`run_redact.py` seeds the run with the CeSIA constitution (gated: accept its terms), builds a no-thinking copy of the
model and runs the five recipes; each resumes where it stopped. The full run took about 7.5 hours on one multi-GPU node, most of it
paraphrasing long replies. `to_csv.py` keeps the accepted rows: plain inputs (`plain`), paraphrases (`paraphrase`),
jailbreak rewrites (`redact_jb`) and variations (`redact_var`), all with the input's id, and the replies
(`exchange.csv`).

**3. Augmentation.** Every augmentation is applied to benign rows at the same rate as to harmful ones, so a wrapper
carries no label information, and an augmented row keeps its source's `input_id`, so it pairs with its plain twin.

```bash
python data/augment.py --in data/train_own/gate.csv --out data/train_own/gate_aug.csv --rate 0.1 --families obf bon
python data/augment.py --in data/train_own/gate.csv --out data/train_own/gate_tpl.csv --rate 0.1 \
    --families template --templates TrustAIRLab/in-the-wild-jailbreak-prompts:jailbreak_2023_12_25
```

- `obf`: 40 string transforms (encodings with and without a "decode and follow" instruction, look-alike characters,
  spacing, reversals, casual writing, noise, payload splitting, neutral containers).
- `bon`: Best-of-N-style character perturbations (random capitals, shuffled word interiors, inserted characters).
- `template`: the prompt inserted into an in-the-wild jailbreak template. No templates ship with this repository;
  `--templates` takes a local file or a Hugging Face dataset, e.g. the 1,405 in-the-wild jailbreak prompts of
  Shen et al. (CCS 2024), which we used.
- `--gibberish N`: content-free benign rows (keyboard mash, random characters, symbols, emoji).

The released data also has one-shot attacks from Jailbreak Foundry (`source == "jf"`, 19 attacks, with the same
model as helper). They need the Jailbreak Foundry library and a model server, so this
repository does not include a script for them. In the released data, template and Best-of-N rows were drawn at
roughly 3.5:1 and 1.6:1 harmful to benign and carry their own `input_id`s; `augment.py` uses equal rates and keeps
the ids.

**4. Public safety data and everyday benign data.** The gate data adds WildGuardMix, Aegis 2.0, FalseReject,
WildJailbreak (vanilla and adversarial), SOSBench and other public sets, plus everyday benign prompts from
UltraChat, MMLU, Alpaca, GSM8K, OASST, Dolly, ARC, OpenBookQA, PubMedQA and WildChat. Every set was checked for
exact-match overlap with the evaluation sets. The dataset cards list each source with its row count and license.

**5. Training CSV.** The released gate file (541k rows) was assembled step by step from these parts, with the
categories of WildJailbreak's harmful prompts, which have none, assigned by an earlier gate model. It is released as built; no script
here rebuilds it row for row.

## Using your own constitution

1. Write your taxonomy. REDACT reads the categories and subcategories it generates for from a taxonomy file: copy
   `src/redact/configs/taxonomy/harmful.json` from REDACT, edit it, and set `"taxonomy_dir"` in the recipes to its
   folder. Update `constitution/taxonomy.json` to match: its categories become the gate's heads.
2. Write the constitution. Run REDACT's `constitution` stage (add `"constitution"` to the stages of
   `inputs_long.json`; it needs a strong model, see REDACT's docs), or write a CSV with the CeSIA constitution's
   columns (`constitution_category, constitution_subcategory, sample_description, entry_type, source_category,
   source_group_tag, model`) and pass it with `run_redact.py --constitution FILE`.
3. Run `run_redact.py` and `to_csv.py` as above, then `augment.py`, and add benign data that matches your traffic.
