"""Turn the released EuroSafeGuard training data into the CSVs the trainers read.

The data is gated on Hugging Face: request access to EuroSafeAI/EuroSafeGuard-prompts and
EuroSafeAI/EuroSafeGuard-replies, then `huggingface-cli login`. Rows copied from gated or non-redistributable sources
are released without their text; --reconstruct rebuilds them with the dataset's own reconstruct.py, which needs
access to those sources too (WildGuardMix, WildJailbreak, WildChat-1M, SOSBench, the CeSIA constitution datasets).
Rows whose text is still missing are dropped, and the count is printed.

    python data/prepare.py --out data/train                    # download, write CSVs
    python data/prepare.py --out data/train --reconstruct      # also rebuild withheld rows
    python data/prepare.py --out data/train --local DIR        # use parquet files already on disk

Writes
  gate.csv         prompt, category, entry_type, source, input_id          (input classifier)
  exchange.csv     prompt_response, category, entry_type, source, input_id (exchange classifier)
  probe_train.csv  prompt, response, label, source, input_id               (activation probe; response empty)
  probe_val.csv    same columns: the `val` split, or a held-out share of input_id groups if there is none
Prints counts only, never text.
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

FILES = {  # name -> (Hugging Face dataset repo, path in the repo)
    "gate": ("EuroSafeAI/EuroSafeGuard-prompts", "data/gate_train.parquet"),
    "probe": ("EuroSafeAI/EuroSafeGuard-prompts", "data/probe_train.parquet"),
    "exchange": ("EuroSafeAI/EuroSafeGuard-replies", "data/exchange_train.parquet"),
}


def fetch(name: str, local: Path | None) -> tuple[Path, Path]:
    """-> (parquet file, the reconstruct.py shipped next to it)."""
    repo, path = FILES[name]
    if local:
        hits = sorted(local.rglob(Path(path).name))
        if not hits:
            sys.exit(f"{Path(path).name} not found under {local}")
        f = hits[0]
        return f, f.parent.parent / "reconstruct.py"
    from huggingface_hub import hf_hub_download
    return (Path(hf_hub_download(repo, path, repo_type="dataset")),
            Path(hf_hub_download(repo, "reconstruct.py", repo_type="dataset")))


def load(name: str, a) -> pd.DataFrame:
    f, script = fetch(name, a.local)
    if a.reconstruct:
        full = a.out / f"{name}_full.parquet"
        subprocess.run([sys.executable, str(script), "--in", str(f), "--out", str(full)], check=True)
        f = full
    return pd.read_parquet(f)


def keep_complete(d: pd.DataFrame, cols: list[str], name: str) -> pd.DataFrame:
    ok = d[cols].notna().all(axis=1) & d[cols].apply(lambda c: c.str.strip().ne("")).all(axis=1)
    print(f"{name}: {len(d):,} rows, {int((~ok).sum()):,} dropped (text withheld or empty), {int(ok.sum()):,} kept")
    return d[ok]


def in_val(input_id: str, frac: float) -> bool:
    return int(hashlib.sha1(input_id.encode()).hexdigest(), 16) % 10_000 < frac * 10_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--local", type=Path, help="directory holding the released parquet files (searched recursively)")
    ap.add_argument("--reconstruct", action="store_true", help="rebuild withheld rows from their original sources")
    ap.add_argument("--probe-val-frac", type=float, default=0.05,
                    help="share of input_id groups held out for probe_val.csv when the data has no val split")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # Input classifier. Rows sharing input_id are rewrites of one prompt; source == "plain" marks the plain twin.
    g = keep_complete(load("gate", a), ["text"], "gate")
    gate = g.rename(columns={"text": "prompt"})[["prompt", "category", "entry_type", "source", "input_id"]]
    gate.to_csv(a.out / "gate.csv", index=False)
    plain = gate[gate.source == "plain"].drop_duplicates("input_id").set_index("input_id").prompt
    twin = (gate.source != "plain") & gate.input_id.isin(plain.index)
    twin &= gate.prompt.ne(gate.input_id.map(plain))
    print(f"  plain rows {int((gate.source == 'plain').sum()):,}; rows with a plain twin {int(twin.sum()):,} "
          f"({gate.prompt[twin].nunique():,} distinct prompts)")
    print("  entry_type", gate.entry_type.value_counts().to_dict())

    # Exchange classifier: the exact string the classifier saw.
    x = keep_complete(load("exchange", a), ["prompt", "response"], "exchange")
    x = x.assign(prompt_response="User: " + x.prompt + x.separator + "Assistant: " + x.response)
    x[["prompt_response", "category", "entry_type", "source", "input_id"]].to_csv(a.out / "exchange.csv", index=False)
    print("  entry_type", x.entry_type.value_counts().to_dict())

    # Activation probe (prompt-only rows).
    p = keep_complete(load("probe", a), ["text"], "probe")
    p = p.assign(prompt=p.text, response="", label=p.label.astype(int))
    if p.split.isin(["val", "validation"]).any():
        val = p.split.isin(["val", "validation"])
    else:
        val = p.input_id.map(lambda i: in_val(i, a.probe_val_frac))
    cols = ["prompt", "response", "label", "source", "input_id"]
    for name, part in (("probe_train", p[~val]), ("probe_val", p[val])):
        part[cols].to_csv(a.out / f"{name}.csv", index=False)
        print(f"  {name}: {len(part):,} rows, label", part.label.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
