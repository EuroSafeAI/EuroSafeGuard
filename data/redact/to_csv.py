"""Convert a REDACT run into the trainer CSVs (gate.csv, exchange.csv), like the released data's REDACT rows.

    python data/redact/to_csv.py --data-dir runs/constitution --out data/train_own

gate.csv: every accepted input (source "plain"), its accepted paraphrases ("paraphrase"), jailbreaks ("redact_jb")
and natural variations ("redact_var"). Rewrites keep the input's sample_id as input_id, so each pairs with its plain
twin. exchange.csv: each accepted reply, harmful when its input is harmful or dual-use harmful.
Concatenate with augment.py output and your benign data as needed. Prints counts only.
"""
import argparse
from pathlib import Path

import pandas as pd

HARMFUL = {"harmful", "dual_use_harmful"}


def read(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    return d[d.accepted.str.lower() == "true"] if "accepted" in d else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path, help="REDACT run directory")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    D = a.data_dir / "Datasets"
    a.out.mkdir(parents=True, exist_ok=True)

    inp = read(D / "constitution_inputs_merged.csv").drop_duplicates("sample_id")
    meta = inp.set_index("sample_id")[["category", "entry_type"]]
    parts = [inp.assign(prompt=inp["sample"], input_id=inp.sample_id, source="plain")]
    if (D / "paraphrases_inputs.csv").exists():
        p = read(D / "paraphrases_inputs.csv")
        parts.append(p.assign(prompt=p["sample"], source="paraphrase"))
    for name, source in (("jailbreaks.csv", "redact_jb"), ("variations.csv", "redact_var")):
        if (D / name).exists():
            j = read(D / name)
            j = j[j.jailbreak.str.strip() != j.prompt.str.strip()]
            parts.append(j.assign(prompt=j.jailbreak, source=source))
    g = pd.concat(parts, ignore_index=True)
    g = g[g.input_id.isin(meta.index) & g.prompt.str.strip().ne("")]
    g["category"] = g.input_id.map(meta.category)
    g["entry_type"] = g.input_id.map(meta.entry_type)
    g.loc[g.entry_type == "general_benign", "category"] = "Benign"
    g[["prompt", "category", "entry_type", "source", "input_id"]].to_csv(a.out / "gate.csv", index=False)
    print("gate.csv", len(g), g.source.value_counts().to_dict())

    if (D / "output_responses.csv").exists():
        o = read(D / "output_responses.csv")
        o = o[o.input_id.isin(meta.index) & o.output_response.str.strip().ne("")]
        harm = o.input_id.map(meta.entry_type).isin(HARMFUL)
        x = pd.DataFrame({
            "prompt_response": "User: " + o.input_prompt + "\n\nAssistant: " + o.output_response,
            "category": o.input_id.map(meta.category).where(harm, "Benign"),
            "entry_type": harm.map({True: "harmful", False: "general_benign"}),
            "source": "redact_output", "input_id": o.input_id})
        x.to_csv(a.out / "exchange.csv", index=False)
        print("exchange.csv", len(x), x.entry_type.value_counts().to_dict())


if __name__ == "__main__":
    main()
