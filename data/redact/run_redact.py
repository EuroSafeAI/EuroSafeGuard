"""Run the REDACT recipes that generated the EuroSafeGuard constitution data.

One local model, huihui-ai/Huihui-Qwen3.8-27B-abliterated, fills every role (generator, checker, paraphraser,
paraphrase checker, translation). REDACT's vLLM backend applies the chat template without arguments, so we serve the
model from a folder whose template turns thinking off by default (built on first use).

Run from the repository root, with REDACT installed (`pip install -e "./REDACT[vllm]"`, MIT,
https://github.com/CentreSecuriteIA/REDACT):

    python data/redact/run_redact.py --data-dir runs/constitution

Before the first recipe, the data dir is seeded with the CeSIA constitution
(centrepourlasecuriteia/constitution-dataset, gated: accept its terms and log in), without its three categories on
sexual content involving minors, for which we generate nothing. Pass --constitution to use your own constitution CSV
in the same format. The recipes run in order; each resumes where it stopped. The merged dataset lands in
<data-dir>/Datasets/complete_dataset.csv.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = "huihui-ai/Huihui-Qwen3.8-27B-abliterated"
RECIPES = ["inputs_long", "inputs_short", "outputs", "jailbreaks", "paraphrase"]
EXCLUDE = {"Child Sexual Abuse Material", "Grooming and Sexual Solicitation of Minors",
           "Predatory Age-Inappropriate Content Targeting Minors"}
NO_THINK = "{%- if enable_thinking is undefined %}{%- set enable_thinking = false %}{%- endif %}\n"


def make_model_dir(out: Path) -> None:
    """Symlink the model snapshot into `out`, with a chat template that defaults to enable_thinking=False."""
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(MODEL))
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("chat_template.jinja", "tokenizer_config.json") and not (out / f.name).exists():
            (out / f.name).symlink_to(f.resolve())
    template = NO_THINK + (src / "chat_template.jinja").read_text()
    (out / "chat_template.jinja").write_text(template)
    cfg = json.loads((src / "tokenizer_config.json").read_text())
    cfg["chat_template"] = template
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))


def seed_constitution(csv: str | None, data_dir: Path) -> None:
    """Write the constitution where REDACT's `training` recipes read it: Data_cache/constitution/."""
    if csv is None:
        from huggingface_hub import hf_hub_download
        csv = hf_hub_download("centrepourlasecuriteia/constitution-dataset", "constitution.csv", repo_type="dataset")
    d = pd.read_csv(csv, dtype=str, keep_default_na=False)
    d = d[~d.constitution_category.isin(EXCLUDE)]
    out = data_dir / "Data_cache" / "constitution"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "merged.csv", index=False)
    for entry_type, g in d.groupby("entry_type"):
        g.to_csv(out / f"{entry_type}.csv", index=False)
    print(f"constitution: {len(d)} entries", d.entry_type.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path, default=Path("models/huihui-qwen3.8-27b-nothink"),
                    help="no-thinking copy of the model (created if missing)")
    ap.add_argument("--constitution", help="constitution CSV (default: download the CeSIA constitution)")
    ap.add_argument("--recipes", nargs="+", default=RECIPES, choices=RECIPES)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # its top-k sampler JIT-compiles and needs nvcc

    if not (a.model_dir / "config.json").exists():
        make_model_dir(a.model_dir)
    if not (a.data_dir / "Data_cache" / "constitution" / "merged.csv").exists():
        seed_constitution(a.constitution, a.data_dir)

    from redact import run_pipeline
    from redact.llms.model_config import VLLMConfig, register_model
    register_model(
        "huihui-qwen3.8-27b", backend_type="vllm", roles=["uncensored_gen", "paraphraser", "translation"],
        is_uncensored=True, supports_system_prompt=True, default_max_tokens=3000, default_temperature=0.7,
        vllm=VLLMConfig(hf_model_id=str(a.model_dir.resolve()), vram_gb=51.0,
                        vllm_kwargs={"max_model_len": 16384, "max_num_seqs": 512,
                                     "limit_mm_per_prompt": {"image": 0, "video": 0}},
                        sampling={"top_p": 0.8, "top_k": 20}))
    for name in a.recipes:
        summary = run_pipeline(str(HERE / f"{name}.json"), data_dir=str(a.data_dir))
        print(name, json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
